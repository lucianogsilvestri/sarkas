"""
Module for handling Yukawa potential.

The Yukawa potential is a screened Coulomb potential commonly used in plasma physics
for modeling interactions between charged particles in a screening medium.

Potential
*********

The Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distance :math:`r` is defined as

.. math::
    U_{ab}(r) = \\frac{q_a q_b}{4 \\pi \\epsilon_0} \frac{e^{- \\kappa r} }{r}

where :math:`\\kappa = 1/\\lambda` is the screening parameter and :math:`\\lambda` is the screening length.

Potential Matrix Structure
**************************

The elements of the :attr:`matrix` are:

.. code-block:: python

    matrix[i, j, 0] = q_i * q_j / (4 * pi * eps0)  # Charge product
    matrix[i, j, 1] = kappa = 1 / screening_length  # Screening parameter
    matrix[i, j, 2] = alpha                         # Ewald parameter (PPPM only)
    matrix[i, j, 3] = a_rs                          # Short-range cutoff

"""

from math import erfc
from tokenize import OP
from numba import jit
from numpy import exp, inf, pi, sqrt, zeros, array
from scipy.integrate import quad
from scipy.special import gamma
from warnings import warn
from typing import Any, Callable, Optional

from .base import PotentialBase


@jit(nopython=True)
def yukawa_force_pppm(r_in: float, pot_matrix: Any) -> tuple[float, float]:
    """
    Calculate Yukawa potential and force with PPPM decomposition.

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters for a specific species pair.
        pot_matrix[0] = q_i * q_j / (4 * pi * eps0)
        pot_matrix[1] = kappa (screening parameter)
        pot_matrix[2] = alpha (Ewald parameter)
        pot_matrix[3] = a_rs (short-range cutoff)

    Returns
    -------
    u_r : float
        Short-range potential value.
    f_r : float
        Short-range force magnitude.
    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]
    rs = pot_matrix[3]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    exp_pos = exp(kappa_r)
    exp_neg = exp(-kappa_r)
    erfc_pos = erfc(alpha_r + 0.5 * kappa_alpha)
    erfc_neg = erfc(alpha_r - 0.5 * kappa_alpha)
    u_r = (pot_matrix[0] * 0.5 / r) * (exp_pos * erfc_pos + exp_neg * erfc_neg)
    f1 = (0.5 / r) * exp_pos * erfc_pos * (1.0 / r - kappa)
    f2 = (0.5 / r) * exp_neg * erfc_neg * (1.0 / r + kappa)
    sqrt_pi_inv = 1.0 / sqrt(pi)
    f3 = (alpha * sqrt_pi_inv / r) * (
        exp(-((alpha_r + 0.5 * kappa_alpha) ** 2)) * exp_pos
        + exp(-((alpha_r - 0.5 * kappa_alpha) ** 2)) * exp_neg
    )
    f_r = pot_matrix[0] * (f1 + f2 + f3)
    return u_r, f_r


@jit(nopython=True)
def yukawa_force(r_in: float, pot_matrix: Any) -> tuple[float, float]:
    """
    Calculate Yukawa potential and force (direct, no Ewald decomposition).

    Parameters
    ----------
    r_in : float
        Distance between two particles.
    pot_matrix : numpy.ndarray
        Potential parameters for a specific species pair.
        pot_matrix[0] = q_i * q_j / (4 * pi * eps0)
        pot_matrix[1] = kappa (screening parameter)
        pot_matrix[2] = unused (for compatibility)
        pot_matrix[3] = a_rs (short-range cutoff)

    Returns
    -------
    u_r : float
        Potential value.
    f_r : float
        Force magnitude.
    """
    rs = pot_matrix[3]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    kappa_r = pot_matrix[1] * r
    u_r = pot_matrix[0] * exp(-kappa_r) / r
    f_r = u_r * (1.0 / r + pot_matrix[1])
    return u_r, f_r


def calc_force_error_quad(potential):
    r"""
    Calculate the force error by integrating over the neglected volume.
    
    The force error is calculated from:
    
    .. math::
        \Delta F = \left[ 4\pi \int_{r_c}^{\infty} dr \, r^2 \left(\frac{dU(r)}{dr}\right)^2 \right]^{1/2}
    
    where :math:`U(r)` is the Yukawa potential, :math:`r_c` is the cutoff radius.
    
    Parameters
    ----------
    potential : Yukawa
        Yukawa potential instance with all parameters set
        
    Returns
    -------
    f_err : float
        Force error estimate
        
    Examples
    --------
    >>> # Assuming a properly set up Yukawa potential instance
    >>> f_err = calc_force_error_quad(yukawa_potential)
    >>> print(f"Force error: {f_err:.2e}")
    """
    if potential.matrix is None or potential.rc is None:
        raise ValueError("Potential matrix and cutoff radius must be set")
    
    # Create normalized parameter matrix for integration
    params = potential.matrix.copy()
    
    # Rescale parameters to avoid numerical issues in quad
    params[:, :, 0] /= potential.matrix[0, 0, 0]  # Normalize charge factor
    params[:, :, 1] *= potential.a_ws              # Scale kappa
    params[:, :, 3] /= potential.a_ws              # Scale short-range cutoff
    
    # Scaled cutoff radius
    r_c = potential.rc / potential.a_ws
    
    # Solid angle factor for different dimensions
    solid_angle = 2 * pi**(potential.dimensions / 2) / gamma(potential.dimensions / 2)
    
    def integrand(r):
        """Integrand for force error calculation."""
        _, f_r = potential.force(r, params[0, 0])
        return solid_angle * r**(potential.dimensions - 1) * f_r**2
    
    # Perform integration
    result, _ = quad(integrand, a=r_c, b=inf)
    
    # Calculate normalization factor
    if potential.dimensions == 3:
        normalization = sqrt(result * 3 / (4 * pi))
    elif potential.dimensions == 2:
        normalization = sqrt(result / (2 * pi))
    else:  # 1D
        normalization = sqrt(result / 2)
    
    # Apply charge and density factors
    charge_factor = potential.QFactor / (potential.matrix[0, 0, 0] * potential.total_num_ptcls)
    f_err = normalization * charge_factor
    
    return f_err


class Yukawa(PotentialBase):
    """
    Yukawa (screened Coulomb) potential implementation.

    The Yukawa potential is widely used in plasma physics to model interactions
    between charged particles in a screening medium such as a plasma or electrolyte.

    Attributes
    ----------
    screening_length : float
        Characteristic screening length (Debye length in plasmas).
    kappa : float
        Screening parameter (1/screening_length).
    coupling_constant : float
        Effective coupling strength.
    screening_length_type : str
        Type of screening length calculation.
    yukawa_params : list or ndarray
        yukawa_params[i][j][0] = q_i * q_j / (4π ε₀)
        yukawa_params[i][j][1] = kappa (screening parameter)
        (User can override before setup)
    pppm_alpha_ewald : float
        Algorithm-specific parameter for PPPM.
    a_rs : float
        Algorithm-specific short-range cutoff.

    Extensibility
    -------------
    To implement a custom Yukawa-like potential, subclass Yukawa and override
    the relevant methods (e.g., initialize_potential_parameters, create_parameter_matrix,
    set_force_function).

    Usage Example
    -------------
    >>> y = Yukawa()
    >>> # Optionally override potential-specific parameters before setup:
    >>> y.yukawa_params = [[[1.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [1.0, 2.0]]]  # shape (num_species, num_species, 2)
    >>> y.pppm_alpha_ewald = 0.25
    >>> y.a_rs = 0.1
    >>> y.setup(params, species_list)
    """
    _aliases = ['yukawa', 'screened_coulomb', 'debye_huckel']

    def __init__(self) -> None:
        super().__init__()
        self.name = 'yukawa'
        self.screening_length: Optional[float] = None
        self.screening_length_type: str = 'thomas_fermi'
        self.kappa: Optional[float] = None
        self.coupling_constant: Optional[float] = None
        self.params = None
        self.force = yukawa_force_pppm
    
    def setup(self, params, species, **kwargs):
        """
        Setup the Yukawa potential with simulation parameters and species.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing physical constants, screening length, etc.
        species : list
            List of species objects containing charge, mass, number, etc.
        **kwargs : dict
            Additional Yukawa-specific parameters:
            - screening_length : float, screening length
            - screening_length_type : str, type of screening
            - kappa : float, screening parameter (alternative to screening_length)
        """
        
        # Update common parameters
        self.update_common_params(params, species)
        
        # Setup screening parameters
        self.setup_screening(params, **kwargs)
        
        # Validate screening length
        if self.screening_length is None and 'kappa' in kwargs:
            self.kappa = kwargs['kappa']
            self.screening_length = 1.0 / self.kappa
        elif self.screening_length is None:
            raise ValueError("Either screening_length or kappa must be specified")
        
        # Allocate parameter matrix (4 parameters per species pair)
        self.allocate_matrix(species, 4)
        
        # Setup charge interaction matrix
        self.setup_charge_matrix(species, param_index=0)
        
        # Set screening parameter for all species pairs
        self.matrix[:, :, 1] = 1.0 / self.screening_length
        
        # Store kappa value for convenience
        if self.kappa is None:
            self.kappa = self.matrix[0, 0, 1] * self.a_ws
        
        # Set short-range cutoff
        self.set_short_range_cutoff()
        
        # Setup force calculation functions based on method
        self._setup_force_functions()
        
        # Set potential derivatives function
        self.potential_derivatives = potential_derivatives
        
        # Set force error calculation function
        self.calc_force_error_quad = lambda: calc_force_error_quad(self)
        
        # Calculate effective coupling constant if not provided
        if self.coupling_constant is None and hasattr(params, 'coupling_constant'):
            self.coupling_constant = params.coupling_constant
    
    def set_force_functions(self):
        """Setup appropriate force calculation functions based on method."""
        if self.algorithm_type == "pp":
            self.force = yukawa_force
        elif self.algorithm_type == "pppm":
            if self.pppm_alpha_ewald is None:
                raise ValueError("PPPM method requires pppm_alpha_ewald parameter")
            self.force = yukawa_force_pppm
        else:
            raise ValueError(f"Unsupported method '{self.algorithm_type}' for Yukawa potential")
    
    def pretty_print_info(self):
        """Print Yukawa-specific parameters in a user-friendly way."""

        #TODO: Add more information about the potential
        msg = f"Yukawa potential:\n"
        msg += f"Screening length: {self.screening_length}\n"
        msg += f"Screening length type: {self.screening_length_type}\n"
        msg += f"Kappa: {self.kappa}\n"
        msg += f"Coupling constant: {self.coupling_constant}\n"
        msg += f"PPPM alpha Ewald: {self.pppm_alpha_ewald}\n"
        msg += f"Short-range cutoff: {self.a_rs}\n"     
        return msg
    
    def initialize_potential_parameters(self, species_list: list[Any]) -> None:
        """
        Initialize Yukawa-specific parameters for all species pairs.
        If self.yukawa_params is set by the user, use those values.
        Otherwise, compute defaults.
        """
        if self.params is not None:
            return

        fourpie0 = self.fourpie0 if self.fourpie0 is not None else 1.0
        kappa = self.kappa if self.kappa is not None else (1.0 / self.screening_length if self.screening_length is not None else raise ValueError("Screening length must be set"))

        self.params = array([
            [
                [sp1.charge * sp2.charge / fourpie0, kappa]
                for sp2 in species_list
            ]
            for sp1 in species_list
        ])

    def set_algorithm_parameters(self, algorithm_type: Optional[str], **kwargs):
        """Set algorithm-specific parameters.
        
        Parameters
        ----------
        algorithm_type : Optional[str]
            Algorithm type
        **kwargs : dict
            Additional parameters
        """
        if algorithm_type is not None:
            self.algorithm_type = algorithm_type

        if algorithm_type is not None and algorithm_type == "pppm":
            if 'pppm_alpha_ewald' in kwargs:
                self.pppm_alpha_ewald = kwargs['pppm_alpha_ewald']
            else:
                raise ValueError("pppm_alpha_ewald must be provided in kwargs")

        if 'a_rs' in kwargs:
            self.a_rs = kwargs['a_rs']

    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create parameter matrix for Yukawa interactions from self.yukawa_params and algorithm-specific parameters.
        """
        num_species = len(species_list)
        self.matrix = zeros((num_species, num_species, 4))
        for i in range(num_species):
            for j in range(num_species):
                self.matrix[i, j, 0] = self.params[i][j][0]
                self.matrix[i, j, 1] = self.params[i][j][1]
                self.matrix[i, j, 2] = self.pppm_alpha_ewald if hasattr(self, 'pppm_alpha_ewald') and self.pppm_alpha_ewald is not None else 0.0
                self.matrix[i, j, 3] = self.a_rs

    def potential_derivatives(self, r: float, pot_matrix: Any) -> tuple[float, float, float]:
        """
        Calculate the first and second derivatives of the Yukawa potential.

        Parameters
        ----------
        r : float
            Distance between particles.
        pot_matrix : numpy.ndarray
            Potential parameters.

        Returns
        -------
        U : float
            Potential value.
        dU_dr : float
            First derivative of the potential.
        d2U_dr2 : float
            Second derivative of the potential.
        """
        kappa = pot_matrix[1]
        kappa_r = kappa * r
        exp_term = exp(-kappa_r)
        u_base = exp_term / r
        du_dr = -(1.0 + kappa_r) * u_base / r
        d2u_dr2 = -(1.0 / r + kappa) * du_dr + u_base / r**2
        # Apply charge factor if needed in context
        return u_base, du_dr, d2u_dr2
