"""
Exact Gradient corrected Screened (EGS) potential implementation.

The EGS potential introduces density gradient corrections to the free energy functional, leading to two regimes:

.. math::
    U_{ab}(r) = \frac{Z_a Z_b \bar{e}^2 }{2r}\left [ ( 1+ \alpha ) e^{-r/\lambda_-} + ( 1 - \alpha) e^{-r/\lambda_+} \right ], \quad \nu < 1

.. math::
    U_{ab}(r) = \frac{Z_a Z_b \bar{e}^2}{r}\left [ \cos(r/\gamma_-) + \alpha' \sin(r/\gamma_-) \right ] e^{-r/\gamma_+}, \quad \nu > 1

References
----------
Stanton, L. G., & Murillo, M. S. (2015). Unified description of linear screening in dense plasmas. Phys. Rev. E, 91(3), 033104. https://doi.org/10.1103/PhysRevE.91.033104
"""

from math import cos, exp, sin
from numba import jit
from numpy import zeros, sqrt, pi, tanh, cosh
from typing import Any

from .base import PotentialBase
from ..utilities.fdints import fdm3h

@jit(nopython=True)
def egs_force(r_in: float, pot_matrix: Any) -> tuple[float, float]:
    """
    Calculate EGS potential and force.

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters (length 7).

    Returns
    -------
    u_r : float
        Potential value.
    f_r : float
        Force value.
    """
    rs = pot_matrix[6]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    nu = pot_matrix[1]
    if nu <= 1.0:
        # Monotonic regime
        A = pot_matrix[0]
        alpha = pot_matrix[2] - 1.0
        lambda_m = 1.0 / pot_matrix[4]
        lambda_p = 1.0 / pot_matrix[5]
        u_r = 0.5 * A * ((1.0 + alpha) * exp(-r / lambda_m) / r + (1.0 - alpha) * exp(-r / lambda_p) / r)
        f_r = 0.5 * A * (
            (1.0 + alpha) * exp(-r / lambda_m) * (1.0 / r + 1.0 / lambda_m) / r
            + (1.0 - alpha) * exp(-r / lambda_p) * (1.0 / r + 1.0 / lambda_p) / r
        )
    else:
        # Oscillatory regime
        A = pot_matrix[0]
        alpha_p = pot_matrix[3]
        gamma_m = 1.0 / pot_matrix[4]
        gamma_p = 1.0 / pot_matrix[5]
        cos_term = cos(r / gamma_m)
        sin_term = sin(r / gamma_m)
        exp_term = exp(-r / gamma_p)
        u_r = A * (cos_term + alpha_p * sin_term) * exp_term / r
        # Derivative
        d_cos = -sin(r / gamma_m) / gamma_m
        d_sin = cos(r / gamma_m) / gamma_m
        d_exp = -exp_term / gamma_p
        f_r = (
            A * (
                (d_cos + alpha_p * d_sin) * exp_term / r
                + (cos_term + alpha_p * sin_term) * d_exp / r
                - (cos_term + alpha_p * sin_term) * exp_term / r**2
            )
        )
    return u_r, f_r


class ExactGradientScreened(PotentialBase):
    """
    Exact Gradient corrected Screened (EGS) potential implementation.

    The EGS potential introduces density gradient corrections to the free energy functional, leading to two regimes:
    - Monotonic decay (nu <= 1)
    - Oscillatory (nu > 1)

    Attributes
    ----------
    matrix : numpy.ndarray
        Parameter matrix for all species pairs (shape: num_species, num_species, 7).
    force_function : callable
        Numba-compiled force function.
    egs_params : list or ndarray
        egs_params[i][j] = [q_iq_j/4pi eps0, nu, 1+alpha, 1-alpha, 1/lambda_m, 1/lambda_p] (nu <= 1)
        or [q_iq_j/4pi eps0, nu, 1.0, alpha_p, 1/gamma_m, 1/gamma_p] (nu > 1)
        (User can override before setup)
    a_rs : float
        Algorithm-specific short-range cutoff.
    lmbda : float
        Gradient correction factor (1/9 for Thomas-Fermi, 1 for von Weizsaecker).
    nu : float
        Dimensionless screening parameter.
    lambda_p : float
        Screening length lambda_plus (set if nu <= 1).
    lambda_m : float
        Screening length lambda_minus (set if nu <= 1).
    gamma_p : float
        Oscillatory screening length gamma_plus (set if nu > 1).
    gamma_m : float
        Oscillatory screening length gamma_minus (set if nu > 1).

    Examples
    --------
    >>> egs = ExactGradientScreened()
    >>> # Optionally override potential-specific parameters before setup:
    >>> egs.egs_params = [[[1, 0.5, 2, 2, 0.1, 1], [1, 0.5, 2, 2, 0.1, 1]], [[1, 0.5, 2, 2, 0.1, 1], [1, 0.5, 2, 2, 0.1, 1]]]  # shape (num_species, num_species, 6)
    >>> egs.a_rs = 0.1
    >>> species_list = []  # or None
    >>> egs.setup(params, species_list)
    """
    def __init__(self) -> None:
        super().__init__()
        self.type = "egs"
        self.screening_length_type = "egs"
        self.params = None
        self.lmbda = 1.0/9.0 # Default value for Thomas-Fermi screening
        self.nu = None
        self.lambda_p = None
        self.lambda_m = None
        self.gamma_p = None
        self.gamma_m = None

    def initialize_potential_parameters(self, species_list: list[Any]) -> None:
        """
        Initialize EGS-specific parameters for all species pairs.

        If `egs_params` is set by the user, use those values. Otherwise, compute them from the electron background
        as in the legacy implementation.

        Parameters
        ----------
        species_list : list
            List of species objects. The last species is assumed to be the electron background.
        """
        if self.params is not None:
            return
        num_species = len(species_list)
        # Electron background is assumed to be the last species
        eb = species_list[-1]
        # lambda factor: 1/9 = Thomas-Fermi, 1 = von Weizsaecker
        self.lmbda = getattr(self, 'lmbda', 1.0/9.0)
        # Calculate nu
        self.nu = 3.0 / pi**1.5 * eb.landau_length / eb.deBroglie_wavelength
        dIdeta = -3.0 / 2.0 * fdm3h(eb.dimensionless_chemical_potential)
        self.nu *= self.lmbda * dIdeta
        # Degeneracy parameter
        theta = eb.degeneracy_parameter
        if 0.1 <= theta <= 12:
            Ntheta = 1.0 + 2.8343 * theta**2 - 0.2151 * theta**3 + 5.2759 * theta**4
            Dtheta = 1.0 + 3.9431 * theta**2 + 7.9138 * theta**4
            h = Ntheta / Dtheta * tanh(1.0 / theta)
            gradh = -(Ntheta / Dtheta) / cosh(1 / theta) ** 2 / (theta**2) - tanh(1.0 / theta) * (
                Ntheta * (7.8862 * theta + 31.6552 * theta**3) / Dtheta**2
                + (5.6686 * theta - 0.6453 * theta**2 + 21.1036 * theta**3) / Dtheta
            )
            b = 1.0 - 2.0 / (8.0 * (eb.Fermi_wavenumber * eb.ThomasFermi_wavelength) ** 2) * (h - 2.0 * theta * gradh)
        else:
            b = 1.0
        # Compute all parameters
        if self.nu <= 1:
            self.lambda_p = eb.ThomasFermi_wavelength * sqrt(self.nu / (2.0 * b + 2.0 * sqrt(b**2 - self.nu)))
            self.lambda_m = eb.ThomasFermi_wavelength * sqrt(self.nu / (2.0 * b - 2.0 * sqrt(b**2 - self.nu)))
            alpha = b / sqrt(b - self.nu)
        else:
            self.gamma_m = eb.ThomasFermi_wavelength * sqrt(self.nu / (sqrt(self.nu) - b))
            self.gamma_p = eb.ThomasFermi_wavelength * sqrt(self.nu / (sqrt(self.nu) + b))
            alpha_p = b / sqrt(self.nu - b)
        
        fourpie0 = self.fourpie0 if self.fourpie0 is not None else 1.0
        self.params = zeros((num_species, num_species, 6))
        for i, sp1 in enumerate(species_list):
            for j, sp2 in enumerate(species_list):
                self.params[i, j, 0] = sp1.charge * sp2.charge / fourpie0
                if self.nu <= 1:
                    self.params[i, j, 1] = self.nu
                    self.params[i, j, 2] = 1.0 + alpha
                    self.params[i, j, 3] = 1.0 - alpha
                    self.params[i, j, 4] = 1.0 / self.lambda_m
                    self.params[i, j, 5] = 1.0 / self.lambda_p
                else:
                    self.params[i, j, 1] = self.nu
                    self.params[i, j, 2] = 1.0
                    self.params[i, j, 3] = alpha_p
                    self.params[i, j, 4] = 1.0 / self.gamma_m
                    self.params[i, j, 5] = 1.0 / self.gamma_p

    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create the parameter matrix for EGS interactions from `egs_params` and algorithm-specific parameters.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        num_species = len(species_list)
        self.matrix = zeros((num_species, num_species, 8))
        for i in range(num_species):
            for j in range(num_species):
                self.matrix[i, j, :7] = self.params[i][j]
                self.matrix[i, j, 7] = self.a_rs

    def set_force_function(self) -> None:
        """
        Set the force calculation function for EGS.

        Returns
        -------
        None
        """
        self.force_function = egs_force

    def pretty_print_info(self):
        """
        Print potential-specific parameters in a user-friendly way.

        Returns
        -------
        None
        """
        if self.params is None:
            raise ValueError("egs_params must be set before pretty printing info.")

        params = self.params[0][0]
        msg = f"Potential type: {self.type}\n"
        msg += f"screening type : {self.screening_length_type}\n"
        msg += f"nu = {self.nu}\n"
        if self.nu is not None and self.nu <= 1:
            msg += f"lambda_m = {getattr(self, 'lambda_m', 'N/A')}\n"
            msg += f"lambda_p = {getattr(self, 'lambda_p', 'N/A')}\n"
            msg += f"egs_params[0][0] = [q_iq_j/4pi eps0, nu, 1+alpha, 1-alpha, 1/lambda_m, 1/lambda_p, 0.0] = {params}\n"
        else:
            msg += f"gamma_m = {getattr(self, 'gamma_m', 'N/A')}\n"
            msg += f"gamma_p = {getattr(self, 'gamma_p', 'N/A')}\n"
            msg += f"egs_params[0][0] = [q_iq_j/4pi eps0, nu, 1.0, alpha_p, 1/gamma_m, 1/gamma_p, 0.0] = {params}\n"
        return msg

    def potential_derivatives(self,r_in: float, pot_matrix: Any) -> tuple[float, float, float]:
        """
        Calculate the first and second derivatives of the EGS potential.

        Parameters
        ----------
        r_in : float
            Distance between two particles.
        pot_matrix : numpy.ndarray
            Potential parameters.

        Returns
        -------
        u_r : float
            Potential value.
        dv_dr : float
            First derivative of the potential.
        d2v_dr2 : float
            Second derivative of the potential (not implemented, returns 0.0).
        """

        rs = pot_matrix[6]
        r = r_in * (r_in >= rs) + rs * (r_in < rs)
        nu = pot_matrix[1]
        if nu <= 1.0:
            # Monotonic regime
            A = pot_matrix[0]
            alpha = pot_matrix[2] - 1.0
            lambda_m = 1.0 / pot_matrix[4]
            lambda_p = 1.0 / pot_matrix[5]
            u_r = 0.5 * A * ((1.0 + alpha) * exp(-r / lambda_m) / r + (1.0 - alpha) * exp(-r / lambda_p) / r)
            f_r = 0.5 * A * (
                (1.0 + alpha) * exp(-r / lambda_m) * (1.0 / r + 1.0 / lambda_m) / r
                + (1.0 - alpha) * exp(-r / lambda_p) * (1.0 / r + 1.0 / lambda_p) / r
            )
        else:
            # Oscillatory regime
            A = pot_matrix[0]
            alpha_p = pot_matrix[3]
            gamma_m = 1.0 / pot_matrix[4]
            gamma_p = 1.0 / pot_matrix[5]
            cos_term = cos(r / gamma_m)
            sin_term = sin(r / gamma_m)
            exp_term = exp(-r / gamma_p)
            u_r = A * (cos_term + alpha_p * sin_term) * exp_term / r
            # Derivative
            d_cos = -sin(r / gamma_m) / gamma_m
            d_sin = cos(r / gamma_m) / gamma_m
            d_exp = -exp_term / gamma_p
            f_r = (
                A * (
                    (d_cos + alpha_p * d_sin) * exp_term / r
                    + (cos_term + alpha_p * sin_term) * d_exp / r
                    - (cos_term + alpha_p * sin_term) * exp_term / r**2
                )
            )
        d2v_dr2 = 0.0 # TODO: Implement second derivative
        return u_r, -f_r, d2v_dr2


