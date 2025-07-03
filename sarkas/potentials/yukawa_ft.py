"""
Yukawa-Friedel Tail (YFT) potential implementation.

The Yukawa-Friedel Tail potential combines a screened Coulomb (Yukawa) term with a Friedel oscillatory tail:

.. math::
    U_{\rm YFT}(r) = A_Y \frac{e^{- \kappa_Y r} }{r} + B_F \frac{e^{- \kappa_F r} }{r^3}\cos(Q_F r + \phi_F)

References
----------
Dharma-Wardana, M.W.C., Stanek, L.J., Murillo, M.S., Yukawa-Friedel-tail pair potentials for warm dense matter applications, Phys. Rev. E 106, 065208 (2022). https://doi.org/10.1103/PhysRevE.106.065208

"""


from numba import jit
from numpy import array, exp, zeros, cos, sin
from typing import Any, Callable, Optional
from warnings import warn

from .base import PotentialBase

@jit(nopython=True)
def yukawa_ft_force(r_in, pot_matrix):
    """
    Calculate Yukawa-Friedel Tail potential and force.

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters (length 7):
            [A_Y, kappa_Y, B_F, kappa_F, Q_F, phi_F, a_rs]

    Returns
    -------
    u_r : float
        Potential value.
    f_r : float
        Force value.
    """
    rs = pot_matrix[6]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    u_y = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
    f_y = u_y * (1.0 / r + pot_matrix[1])
    u_ft_exp = pot_matrix[2] * exp(-pot_matrix[3] * r) / r**3
    u_ft = u_ft_exp * cos(pot_matrix[4] * r + pot_matrix[5])
    f_ft = u_ft * (3.0 / r + pot_matrix[3])
    f_ft += pot_matrix[4] * u_ft_exp * sin(pot_matrix[4] * r + pot_matrix[5])
    u_r = u_y + u_ft
    f_r = f_y + f_ft
    return u_r, f_r


class YukawaFriedelTail(PotentialBase):
    """
    Yukawa-Friedel Tail (YFT) potential implementation.

    The YFT potential combines a screened Coulomb (Yukawa) term with a Friedel oscillatory tail.
    Only the PP (particle-particle) method is supported.

    Attributes
    ----------
    matrix : numpy.ndarray
        Parameter matrix for all species pairs (shape: num_species, num_species, 7).
    force_function : callable
        Numba-compiled force function.
    yft_params : list of float
        List of YFT parameters: [A_Y, kappa_Y, B_F, kappa_F, Q_F, phi_F].

    Usage Example
    -------------
    >>> yft = YukawaFriedelTail(testing_mode=True)
    >>> print(yft.pretty_print_info())
    """
    def __init__(self, testing_mode: bool = False) -> None:
        super().__init__()
        self.type = "yukawa_friedel_tail"
        self.screening_length_type = "yft"
        self.params: Optional[list[float]] = None  # [A_Y, kappa_Y, B_F, kappa_F, Q_F, phi_F]
        self.force_function = yukawa_ft_force

        if testing_mode:
            self._set_physical_constants(units = 'cgs')  # Set physical constants in CGS units
            # Al rho = 2.7 g/cm^3, T = 1 eV from Dharma-Wardana et al. 2022
            self.params = array([5.643048e-17, 3.449355e+08, 3.529030e-36, 0.000000e+00, 3.742168e+08, -6.724320e+00])
            self.matrix = zeros((1, 1, 7))
            self.matrix[0, 0, :6] = self.params
            self.matrix[0, 0, 6] = 0.1  #

    def initialize_potential_parameters(self, species_list: list[Any], **kwargs: Any) -> None:
        """
        Initialize YFT-specific parameters for all species pairs.
        If self.params is set by the user, use those values.
        Otherwise, set to zeros (user must set real values).
        """
        if self.params is not None:
            return
        else:
            self.params = zeros((len(species_list), len(species_list), 6))
            if 'yft_params' in kwargs:
                self.params = kwargs['yft_params']
            else:
                raise ValueError("yft_params must be provided in kwargs. It should be an array of shape (num_species, num_species, 6)")

    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create the parameter matrix for YFT interactions.
        
        Parameters
        ----------
        species_list : list
            List of species objects containing charge, mass, name attributes.
        """
        num_species = len(species_list)
        self.matrix = zeros((num_species, num_species, 7))
        for i in range(num_species):
            for j in range(num_species):
                self.matrix[i, j, :6] = self.params[i][j]
                self.matrix[i, j, 6] = self.a_rs

    def set_force_function(self) -> None:
        """
        Set the force calculation function for YFT.
        """
        self.force_function = yukawa_ft_force

    def pretty_print_info(self):
        """
        Print potential-specific parameters in a user-friendly way.
        """
        
        msg = (
            f"Potential type: {self.type}\n"
            f"A_Y = {self.matrix[0,0,0]:.6e}, kappa_Y = {self.matrix[0,0,1]:.6e}\n"
            f"B_F = {self.matrix[0,0,2]:.6e}, kappa_F = {self.matrix[0,0,3]:.6e}\n"
            f"Q_F = {self.matrix[0,0,4]:.6e}, phi_F = {self.matrix[0,0,5]:.6e}\n"
            f"a_rs = {self.matrix[0,0,6]:.6e}\n"
        )
        return msg
    
    def set_algorithm_parameters(self, algorithm_type: str, **kwargs: Any) -> None:
        """
        Only the PP method is supported for YFT. Raise error otherwise.

        Parameters
        ----------
        algorithm_type : str
            Type of algorithm ('pp', ...)
        """
        if algorithm_type != "pp":
            raise NotImplementedError("Yukawa-Friedel Tail potential is only implemented for the PP method.")
        else:
            self.algorithm_type = algorithm_type

        if 'a_rs' in kwargs:
            self.a_rs = kwargs['a_rs']
        else:
            raise ValueError("a_rs must be provided in kwargs")

    def potential_derivatives(self, r: float, pot_matrix: Any) -> tuple[float, float, float]:
        """
        Calculate the first and second derivatives of the Yukawa-Friedel Tail potential.

        Parameters
        ----------
        r : float
            Distance between two particles.
        pot_matrix : numpy.ndarray
            Potential parameters.

        Returns
        -------
        U : float
            Potential value.
        dv_dr : float
            First derivative of the potential.
        d2v_dr2 : float
            Second derivative of the potential (not implemented, returns 0.0).
        """
        u_y = pot_matrix[0] * exp(-pot_matrix[1] * r) / r
        f_y = -u_y * (1.0 / r + pot_matrix[1])
        u_ft_exp = pot_matrix[2] * exp(-pot_matrix[3] * r) / r**3
        u_ft = u_ft_exp * cos(pot_matrix[4] * r + pot_matrix[5])
        f_ft = -u_ft * (3.0 / r + pot_matrix[3])
        f_ft += -pot_matrix[4] * u_ft_exp * sin(pot_matrix[4] * r + pot_matrix[5])
        u_r = u_y + u_ft
        dv_dr = f_y + f_ft
        d2v_dr2 = 0.0
        return u_r, dv_dr, d2v_dr2
