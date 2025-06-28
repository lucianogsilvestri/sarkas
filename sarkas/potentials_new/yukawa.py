r"""
Module for handling Yukawa potential.

The Yukawa potential is a screened Coulomb potential commonly used in plasma physics
to model interactions between charged particles in a screening medium.

Potential
*********

The Yukawa potential between two charges :math:`q_i` and :math:`q_j` at distance :math:`r` is defined as

.. math::
    U_{ab}(r) = \frac{q_a q_b}{4 \pi \epsilon_0} \frac{e^{- \kappa r} }{r}

where :math:`\kappa = 1/\lambda` is the screening parameter and :math:`\lambda` is the screening length.

Potential Matrix Structure
**************************

The elements of the :attr:`matrix` are:

.. code-block:: python

    matrix[i, j, 0] = q_i * q_j / (4 * pi * eps0)  # Charge product
    matrix[i, j, 1] = kappa = 1 / screening_length  # Screening parameter
    matrix[i, j, 2] = alpha                         # Ewald parameter (PPPM only)
    matrix[i, j, 3] = a_rs                          # Short-range cutoff

References
----------
Yukawa, H. (1935). On the interaction of elementary particles. 
Proceedings of the Physico-Mathematical Society of Japan, 17, 48-57.
"""

from math import erfc
from numba import jit
from numba.core.types import float64
from numpy import exp, inf, pi, sqrt, zeros
from scipy.integrate import quad
from scipy.special import gamma
from warnings import warn

from .base import LongRangePotentialBase


@jit(nopython=True)
def yukawa_force_pppm(r_in, pot_matrix):
    """
    Numba'd function to calculate Yukawa potential and force with PPPM decomposition.
    
    This function computes the short-range part of the Yukawa potential when using
    the PPPM algorithm, which splits the interaction into short and long-range parts.
    
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
        Short-range potential value
    f_r : float
        Short-range force magnitude

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([1.0, 0.5, 0.25, 0.0001])
    >>> yukawa_force_pppm(r, pot_matrix)
    (0.16287410244138842, 0.18025091684402375)
    """
    kappa = pot_matrix[1]
    alpha = pot_matrix[2]  # Ewald parameter alpha
    
    # Short-range cutoff to deal with divergence
    rs = pot_matrix[3]
    # Branchless programming to handle r < rs
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    # Pre-compute common terms
    kappa_alpha = kappa / alpha
    alpha_r = alpha * r
    kappa_r = kappa * r
    
    # Short-range Yukawa potential with Ewald decomposition
    exp_pos = exp(kappa_r)
    exp_neg = exp(-kappa_r)
    erfc_pos = erfc(alpha_r + 0.5 * kappa_alpha)
    erfc_neg = erfc(alpha_r - 0.5 * kappa_alpha)
    
    u_r = (pot_matrix[0] * 0.5 / r) * (exp_pos * erfc_pos + exp_neg * erfc_neg)
    
    # Force calculation: F = -dU/dr
    # Derivative of the exponential terms and 1/r
    f1 = (0.5 / r) * exp_pos * erfc_pos * (1.0 / r - kappa)
    f2 = (0.5 / r) * exp_neg * erfc_neg * (1.0 / r + kappa)
    
    # Derivative of erfc terms: d/dr erfc(ax + b) = -2a/sqrt(pi) * exp(-(ax + b)^2)
    sqrt_pi_inv = 1.0 / sqrt(pi)
    f3 = (alpha * sqrt_pi_inv / r) * (
        exp(-((alpha_r + 0.5 * kappa_alpha) ** 2)) * exp_pos
        + exp(-((alpha_r - 0.5 * kappa_alpha) ** 2)) * exp_neg
    )
    
    f_r = pot_matrix[0] * (f1 + f2 + f3)
    
    return u_r, f_r


@jit(nopython=True)
def yukawa_force(r_in, pot_matrix):
    """
    Numba'd function to calculate Yukawa potential and force.
    
    This is the standard Yukawa potential without Ewald decomposition,
    used for direct particle-particle calculations.
    
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
        Potential value
    f_r : float
        Force magnitude

    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([1.0, 1.0, 0.0, 0.0001])
    >>> yukawa_force(r, pot_matrix)
    (0.06766764161830635, 0.10150146242745953)
    """
    # Short-range cutoff to deal with divergence
    rs = pot_matrix[3]
    # Branchless programming to handle r < rs
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    # Yukawa potential: U = A * exp(-kappa * r) / r
    kappa_r = pot_matrix[1] * r
    u_r = pot_matrix[0] * exp(-kappa_r) / r
    
    # Force: F = U * (1/r + kappa)
    f_r = u_r * (1.0 / r + pot_matrix[1])
    
    return u_r, f_r


def potential_derivatives(r, pot_matrix):
    """
    Calculate the first and second derivatives of the Yukawa potential.
    
    Parameters
    ----------
    r : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    U : float
        Potential value
    dU_dr : float
        First derivative of the potential
    d2U_dr2 : float
        Second derivative of the potential
    """
    kappa = pot_matrix[1]
    kappa_r = kappa * r
    
    # Base functions
    exp_term = exp(-kappa_r)
    u_base = exp_term / r
    
    # First derivative
    du_dr = -(1.0 + kappa_r) * u_base / r
    
    # Second derivative
    d2u_dr2 = -(1.0 / r + kappa) * du_dr + u_base / r**2
    
    # Apply charge factor
    U = pot_matrix[0] * u_base
    dU_dr = pot_matrix[0] * du_dr
    d2U_dr2 = pot_matrix[0] * d2u_dr2
    
    return U, dU_dr, d2U_dr2


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


class Yukawa(LongRangePotentialBase):
    """
    Yukawa (screened Coulomb) potential implementation.
    
    The Yukawa potential is widely used in plasma physics to model interactions
    between charged particles in a screening medium such as a plasma or electrolyte.
    
    Attributes
    ----------
    screening_length : float
        Characteristic screening length (Debye length in plasmas)
    kappa : float
        Screening parameter (1/screening_length)
    coupling_constant : float
        Effective coupling strength
    screening_length_type : str
        Type of screening length calculation
    """
    
    _aliases = ['yukawa', 'screened_coulomb', 'debye_huckel']
    
    def __init__(self):
        super().__init__()
        self.name = 'yukawa'
        
        # Yukawa-specific parameters
        self.screening_length = None
        self.screening_length_type = 'thomas_fermi'
        self.kappa = None
        self.coupling_constant = None
    
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
        # Validate inputs
        self.validate_inputs(params, species)
        self.validate_charges(species)
        
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
    
    def _setup_force_functions(self):
        """Setup appropriate force calculation functions based on method."""
        if self.method == "pp":
            self.force = yukawa_force
        elif self.method == "pppm":
            if self.pppm_alpha_ewald is None:
                raise ValueError("PPPM method requires pppm_alpha_ewald parameter")
            self.force = yukawa_force_pppm
            # Add Ewald parameter to matrix
            self.matrix[:, :, 2] = self.pppm_alpha_ewald
        else:
            raise ValueError(f"Unsupported method '{self.method}' for Yukawa potential")
    
    def update_screening_length(self, new_screening_length):
        """
        Update the screening length and recalculate parameters.
        
        Parameters
        ----------
        new_screening_length : float
            New screening length value
        """
        self.screening_length = new_screening_length
        self.kappa = 1.0 / new_screening_length
        
        if self.matrix is not None:
            self.matrix[:, :, 1] = 1.0 / self.screening_length
    
    def get_effective_coupling(self):
        """
        Calculate the effective coupling constant for the Yukawa potential.
        
        Returns
        -------
        float
            Effective coupling constant
        """
        if self.coupling_constant is not None:
            return self.coupling_constant
        
        # Basic estimate based on system parameters
        if self.matrix is not None and self.a_ws is not None:
            charge_factor = abs(self.matrix[0, 0, 0])
            return charge_factor / (self.fourpie0 * self.a_ws)
        
        return None
    
    def pretty_print_info(self):
        """Print Yukawa-specific parameters in a user-friendly way."""
        msg = super().pretty_print()
        
        if self.screening_length is not None:
            msg += f"Screening type: {self.screening_length_type}\n"
            msg += f"Screening length: {self.screening_length:.6e}"
            if 'length' in self.units_dict:
                msg += f" {self.units_dict['length']}"
            msg += "\n"
            
            if self.a_ws is not None:
                msg += f"κ = {self.a_ws / self.screening_length:.4f}\n"
        
        if self.coupling_constant is not None:
            msg += f"Γ_eff = {self.coupling_constant:.2f}\n"
        
        return msg
    
    def get_info(self):
        """
        Return Yukawa potential information.
        
        Returns
        -------
        dict
            Dictionary containing potential properties
        """
        info = super().get_info()
        info.update({
            'screening_length': self.screening_length,
            'screening_length_type': self.screening_length_type,
            'kappa': self.kappa,
            'coupling_constant': self.coupling_constant,
            'effective_coupling': self.get_effective_coupling()
        })
        return info
    
    def validate_parameters(self):
        """
        Validate Yukawa potential parameters.
        
        Raises
        ------
        ValueError
            If parameters are invalid or inconsistent
        """
        super().validate_force_function()
        
        if self.screening_length is None or self.screening_length <= 0:
            raise ValueError("Screening length must be positive")
        
        if self.kappa is None or self.kappa <= 0:
            raise ValueError("Screening parameter kappa must be positive")
        
        if abs(1.0 / self.screening_length - self.kappa) > 1e-10:
            raise ValueError("Inconsistent screening length and kappa values")
    
    def __str__(self):
        """String representation of the Yukawa potential."""
        return f"Yukawa(screening_length={self.screening_length:.3e}, method='{self.method}')"
    
    def __repr__(self):
        """Detailed string representation of the Yukawa potential."""
        return (f"Yukawa(screening_length={self.screening_length}, "
                f"kappa={self.kappa}, method='{self.method}', "
                f"dimensions={self.dimensions})")


# Deprecated functions for backward compatibility
def force_deriv(r, pot_matrix):
    """
    Calculate the second derivative of the potential.
    
    .. deprecated:: 1.0
        Use `potential_derivatives` instead.
    
    Parameters
    ----------
    r : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    d2v_dr2 : float
        Second derivative of the potential
        
    Raises
    ------
    DeprecationWarning
        This function is deprecated
    """
    warn(
        "Deprecated feature. It will be removed in a future release. "
        "Use potential_derivatives instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    
    _, _, d2v_dr2 = potential_derivatives(r, pot_matrix)
    return d2v_dr2


def update_params(potential, species):
    """
    Legacy function for updating Yukawa potential parameters.
    
    .. deprecated:: 1.0
        Use the `Yukawa.setup()` method instead.
    
    Parameters
    ----------
    potential : object
        Legacy potential object
    species : list
        List of species objects
        
    Raises
    ------
    DeprecationWarning
        This function is deprecated
    """
    warn(
        "Deprecated feature. It will be removed in a future release. "
        "Use the Yukawa class setup() method instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    
    # Legacy parameter setup for backward compatibility
    potential.matrix = zeros((len(species) - 1, len(species) - 1, 4))
    potential.matrix[:, :, 1] = 1.0 / potential.screening_length
    
    if not hasattr(potential, "kappa") or potential.kappa is None:
        potential.kappa = potential.matrix[0, 0, 1] * potential.a_ws
    
    for i, sp1 in enumerate(species[:-1]):  # species[-1] is the electronic background
        q1 = sp1.charge
        for j, sp2 in enumerate(species[:-1]):
            q2 = sp2.charge
            potential.matrix[i, j, 0] = q1 * q2 / potential.fourpie0
    
    potential.matrix[:, :, -1] = potential.a_rs
    potential.potential_derivatives = potential_derivatives
    
    if potential.method == "pp":
        potential.force = yukawa_force
        potential.calc_force_error_quad = calc_force_error_quad
    elif potential.method == "pppm":
        potential.force = yukawa_force_pppm
        potential.matrix[:, :, 2] = potential.pppm_alpha_ewald


def pretty_print_info(potential):
    """
    Legacy function for printing potential information.
    
    .. deprecated:: 1.0
        Use the `Yukawa.pretty_print_info()` method instead.
    
    Parameters
    ----------
    potential : object
        Legacy potential object
        
    Raises
    ------
    DeprecationWarning
        This function is deprecated
    """
    warn(
        "Deprecated feature. It will be removed in a future release. "
        "Use the Yukawa class pretty_print_info() method instead.",
        category=DeprecationWarning,
        stacklevel=2
    )
    
    msg = (
        f"screening type : {potential.screening_length_type}\n"
        f"screening length = {potential.screening_length:.6e} {potential.units_dict['length']}\n"
        f"kappa = {potential.a_ws / potential.screening_length:.4f}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)