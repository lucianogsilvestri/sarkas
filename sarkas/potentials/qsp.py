"""
Quantum Statistical Potential implementation.

This module implements Quantum Statistical Potentials (QSP) which include
quantum effects like Pauli exclusion principle and diffraction terms.

Potential
*********

Quantum Statistical Potentials are defined by three terms:

.. math::
    U(r) = U_{\\rm coulomb}(r) + U_{\\rm diff}(r) + U_{\\rm pauli}(r)

where:

1. Coulomb interaction: :math:`U_{\\rm coulomb}(r) = \\frac{q_i q_j}{4\\pi \\epsilon_0 r}`

2. Diffraction term (three variants):
   - Deutsch: :math:`U_{\\rm deutsch}(r) = \\frac{q_i q_j}{4\\pi \\epsilon_0} \\frac{e^{-2\\pi r/\\Lambda_{ij}}}{r}`
   - Kelbg: Complex form with erfc terms
   - Hansen: Simplified exponential form

3. Pauli exclusion: :math:`U_{\\rm pauli}(r) = k_B T \\ln(2) e^{-4\\pi r^2/\\Lambda_{ij}^2}`

where :math:`\\Lambda_{ij}` is the thermal de Broglie wavelength between species i and j.

Potential Attributes
********************

The parameter matrix has shape (num_species, num_species, 8):

.. code-block::

    matrix[i,j,0] = q_i * q_j / (4π ε₀)     # Coulomb prefactor
    matrix[i,j,1] = 2π/λ_deB or √(2π)/λ_deB # Diffraction parameter  
    matrix[i,j,2] = Pauli prefactor         # k_B T ln(2) or custom
    matrix[i,j,3] = Pauli exponent          # √(4π)/λ_deB or custom
    matrix[i,j,4] = Pauli amplitude         # A_θ factor
    matrix[i,j,5] = Diffraction flag        # 1.0 for e-e/e-i, 0.0 for i-i
    matrix[i,j,6] = α_ewald                 # Ewald parameter
    matrix[i,j,7] = a_rs                    # Short-range cutoff
"""

from math import erfc, log
from numba import jit
from numba.core.types import float64, UniTuple
from numpy import array, array2string, exp, inf, pi, sqrt, zeros, isclose, ndarray
from scipy.integrate import quad
from scipy.special import gamma
from warnings import warn
from typing import Any
from scipy.constants import physical_constants

from .base import PotentialBase
from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import TWOPI


class QuantumStatisticalPotential(PotentialBase):
    """
    Quantum Statistical Potential (QSP) implementation.

    This potential includes quantum effects relevant for dense plasmas:
    - Pauli exclusion principle (fermionic statistics)
    - Diffraction effects (wave nature of particles)
    - Multiple formulations (Deutsch, Kelbg, Hansen)

    Attributes
    ----------
    qsp_params : ndarray
        qsp_params[i][j] = [Coulomb prefactor, diffraction param, Pauli prefactor, Pauli exponent, Pauli amplitude, Diffraction flag, alpha_ewald, a_rs]
        (User can override before setup)
    qsp_type : str
        Type of QSP formulation ('deutsch', 'kelbg', 'hansen')
    qsp_pauli : bool
        Whether to include Pauli exclusion term
    ee_diffractive_length : float, optional
        Custom electron-electron diffractive length
    ei_diffractive_length : float or list, optional
        Custom electron-ion diffractive length(s)
    ai : float
        Ion Wigner-Seitz radius
    pppm_alpha_ewald : float
        Algorithm-specific parameter for PPPM.
    a_rs : float
        Algorithm-specific short-range cutoff.

    Examples
    --------
    >>> qsp = QuantumStatisticalPotential()
    >>> # Optionally override potential-specific parameters before setup:
    >>> qsp.qsp_params = np.ones((2,2,8))  # or a user-defined parameter array
    >>> qsp.a_rs = 0.1
    >>> species_list = []  # or None
    >>> qsp.setup(params, species_list)
    """
    
    def __init__(self):
        super().__init__()
        self.type = "qsp"
        
        # Physical constants. Set in setup
        self.hbar = None
        self.deBroglie_const = None

        # QSP-specific parameters
        self.qsp_type = "deutsch"  # Default formulation
        self.qsp_pauli = True      # Include Pauli term by default
        
        # Custom diffractive lengths (optional)
        self.ee_diffractive_length = None
        self.ei_diffractive_length = None
        
        # Ion properties
        self.ai = None  # Ion Wigner-Seitz radius
        
        # Algorithm requirements
        self.pppm_alpha_ewald = 0.0 
        self.algorithm_type = "pppm"  # QSP requires PPPM
        
        # User-overridable full parameter matrix
        self.params = None
        
        # Validate QSP type
        self.valid_types = ['deutsch', 'kelbg', 'hansen']
        self.force = None
        
    def initialize_potential_parameters(self, species_list: list[Any]) -> None:
        """
        Set the potential-specific parameters (matrix[:, :, :6]) for QSP.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        
        two_pi = 2.0 * pi

        num_species = len(species_list)
        if self.params is not None:
            return 

        self.params = zeros((num_species, num_species, 5))

        for i, sp1 in enumerate(species_list):
            m1 = sp1.mass
            q1 = sp1.charge

            for j, sp2 in enumerate(species_list[i:], start=i):
                m2 = sp2.mass
                q2 = sp2.charge
                reduced_mass = (m1 * m2) / (m1 + m2)                

                # q *e / 4pi eps0
                self.params[i, j, 0] = q1 * q2 / self.fourpie0
                self.params[j, i, 0] = self.params[i, j, 0]

                # 2pi / lambda_ij or sqrt(2pi) / lambda_ij
                if sp1.name == "e" and sp2.name == "e":
                    lambda_ij = self.lambda_ee
                elif sp1.name == "e" and sp2.name != "e":
                    lambda_ij = self.lambda_ei[j - 1]
                else:
                    lambda_ij = self.lambda_ii[i - 1]
                self.params[i, j, 1] = sqrt(two_pi) / lambda_ij if self.qsp_type == "kelbg" else two_pi / lambda_ij
                self.params[j, i, 1] = self.params[i, j, 1]
                # Pauli term
                if sp1.name == "e" and sp2.name == "e":
                    self.params[i, j, 2] = self.ee_pauli_params[0]
                    self.params[i, j, 3] = self.ee_pauli_params[1]
                
                # Diffraction term
                if sp1.name == "e" and sp2.name == "e":
                    self.params[i, j, 4] = 1.0

    def _set_physical_constants(self, units: str, **kwargs: Any) -> None:
        """
        Set physical constants from simulation parameters.
        """
        super()._set_physical_constants(units)
        self.hbar = physical_constants["reduced Planck constant"][0]
        self.a0 = physical_constants["Bohr radius"][0]

        if self.units in ['cgs', 'atomic', 'hartree']:
            J2erg = 1.0e7  # erg/J
            
            self.hbar *= J2erg
            self.a0 *= 1e2
        elif self.units == "eV":
            eV2J = physical_constants["electron volt-joule relationship"][0]
            self.hbar *= eV2J
            self.a0 *= 1e2

        self.deBroglie_const = TWOPI * self.hbar**2 / self.kB

    def _setup_species_parameters(self, species_list):
        """Setup QSP-specific species parameters."""
        
        # Make sure electron are first species
        if species_list[0].name == "e":
            electron_species = species_list[0]
        else:
            raise ValueError("Electrons must be first species")
        
        self._set_ee_parameters(electron_species)
        self._set_ei_parameters(electron_species, species_list)
        self._set_ii_parameters(species_list)

        # Validate custom diffractive lengths if provided
        self._validate_custom_diffractive_lengths()
    
    def _validate_custom_diffractive_lengths(self):
        """Validate custom diffractive length parameters."""
        if hasattr(self, 'ei_diffractive_length') and self.ei_diffractive_length is not None:
            num_ion_species = self.num_species - 1  # Total minus electrons
            
            if isinstance(self.ei_diffractive_length, (list, tuple, ndarray)):
                if len(self.ei_diffractive_length) != num_ion_species:
                    raise ValueError(
                        f"ei_diffractive_length list must have {num_ion_species} "
                        f"elements (one per ion species)"
                    )
            else:
                # Convert single value to list for consistent handling
                self.ei_diffractive_length = [self.ei_diffractive_length] * num_ion_species
    
    def create_parameter_matrix(self, species_list: list[Any]) -> None:
        """
        Create the parameter matrix for QSP interactions.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        num_species = len(species_list)
        self.matrix = zeros((num_species, num_species, 7))
        
        if self.params is None:
            self.initialize_potential_parameters(species_list)
        
        self.matrix[:, :, :5] = self.params
        self.matrix[:, :, 6] = self.pppm_alpha_ewald if self.pppm_alpha_ewald is not None else 0.0
        self.matrix[:, :, 7] = self.a_rs

        self.matrix = self.params
    
    def _set_interaction_parameters(self, i, j, electron_species, 
                                   total_ion_temperature, theta,
                                   deBroglie_const, log_2):
        """Set parameters for specific species pair interaction."""
        # Get species properties
        q1 = species_list[i].charge
        q2 = species_list[j].charge
        m1 = species_list[i].mass
        m2 = species_list[j].mass
        
        # Reduced mass
        reduced_mass = (m1 * m2) / (m1 + m2)
        
        # Coulomb prefactor
        self.matrix[i, j, 0] = q1 * q2 / fourpie0
        
        # Determine interaction type and set parameters
        is_electron_i = (i == 0)  # Assume electrons are first species
        is_electron_j = (j == 0)
        
        if is_electron_i and is_electron_j:
            # Electron-electron interaction
            self._set_ee_parameters(i, j, electron_species, reduced_mass, 
                                  deBroglie_const, theta, log_2)
        elif is_electron_i or is_electron_j:
            # Electron-ion interaction
            ion_index = j - 1 if is_electron_i else i - 1
            self._set_ei_parameters(i, j, electron_species, reduced_mass,
                                  deBroglie_const, ion_index)
        else:
            # Ion-ion interaction (no quantum effects)
            self._set_ii_parameters(i, j, reduced_mass, deBroglie_const, 
                                  total_ion_temperature)
    
    def _set_ee_parameters(self, electron_species):
        """Set electron-electron interaction parameters."""
        # Calculate or use custom de Broglie wavelength
        reduced_mass = 0.5 * electron_species.mass
        self.lambda_ee = sqrt(self.deBroglie_const / (reduced_mass * electron_species.temperature))
        theta = getattr(electron_species, 'degeneracy_parameter', 1.0) # kB T / E_F
                
        # Pauli parameters (type-dependent)
        self.ee_pauli_params = zeros(2)

        if self.qsp_type == "hansen":
            self.ee_pauli_params[0] = log_2 * self.kB * electron_species.temperature
            self.ee_pauli_params[1] = 4.0 * pi / (log_2 * self.lambda_ee**2)
        else:
            # Deutsch/Kelbg use Jones-Murillo corrections
            a1, a2, a3 = 0.2975, 6.090, 1.541
            b1, b2, b3, b4 = 0.0842, 0.1027, 1.096, 1.359
            A_theta = 1.0 + a1 / (1.0 + a2 * theta**a3)
            B_theta = 1.0 + b1 * exp(-b2 * theta**b3) / theta**b4
            
            self.ee_pauli_params[0] = - self.kB * electron_species.temperature
            self.ee_pauli_params[1] = sqrt(TWOPI * B_theta) / self.lambda_ee
            
    def _set_ei_parameters(self, electron_species, species_list):
        """Set electron-ion interaction parameters."""
        num_ion_species = len(species_list) - 1
        self.lambda_ei = zeros(num_ion_species)

        self.ei_temperatures = zeros(num_ion_species, dtype = species_list[1].temperature.dtype)
        self.lambda_ei = zeros(num_ion_species)

        for j, species in enumerate(species_list[1:], start=1):  # Skip electrons

            reduced_mass = (species.mass * electron_species.mass) / (species.mass + electron_species.mass)
            
            # Note: that for T_e = T_i, the ei_temperatures is equal to T_e
            # for T_e/m_e >> T_i/m_i, the ei_temperature is equal to T_e
            if self.qsp_type == "deutsch": # Allows for two-temperature plasma
                self.ei_temperatures[j - 1] = reduced_mass * (species.temperature/species.mass + electron_species.temperature/electron_species.mass)
            else:
                # Use electron temperature for all ion species
                self.ei_temperatures[j - 1] = electron_species.temperature

            # Note that m_e << m_i, as such the ei_temperatures is equal to T_e and reduced_mass is approximately m_i
            self.lambda_ei[j - 1] = sqrt(self.deBroglie_const / (reduced_mass * self.ei_temperatures[j - 1]))
           
    def _set_ii_parameters(self, species_list):
        """Set ion-ion interaction parameters (classical)."""
        
        self.total_ion_number_density = 0.0
        # Calculate ion properties
        self.total_ion_temperature = 0.0

        # Ion Wigner-Seitz radius
        four_pi = 4.0 * pi
        self.ai = (3.0 / (four_pi * self.total_ion_number_density)) ** (1.0 / 3.0)

        num_ion_species = len(species_list) - 1

        # For completeness, we could set the parameters for ion-ion interactions
        # but it is not needed as the potential is not used for ion-ion interactions
        self.lambda_ii = zeros(num_ion_species)
        for i, sp1 in enumerate(species_list[1:], start=1):  # Skip electrons
            self.total_ion_temperature += sp1.concentration * sp1.temperature
            self.total_ion_number_density += sp1.number_density
            self.lambda_ii[i - 1] = sqrt(self.deBroglie_const / (sp1.mass * sp1.temperature))

    def set_force_function(self):
        """Set the appropriate force function based on QSP type."""
        if self.qsp_type == "deutsch":
            self.force_function = deutsch_force
        elif self.qsp_type == "hansen":
            self.force_function = hansen_force
        elif self.qsp_type == "kelbg":
            self.force_function = kelbg_force
        else:
            raise ValueError(f"Unknown QSP type: {self.qsp_type}")
    
    def set_algorithm_parameters(self, **kwargs):
        """Set algorithm-specific parameters for QSP."""
        
        if 'algorithm_type' in kwargs:
            self.algorithm_type = kwargs['algorithm_type']
        else:
            raise ValueError("algorithm_type must be provided in kwargs")
        
        if 'alpha_ewald' in kwargs:
            self.pppm_alpha_ewald = kwargs['alpha_ewald']
        else:
            raise ValueError("alpha_ewald must be provided in kwargs")

        if 'a_rs' in kwargs:
            self.a_rs = kwargs['a_rs']
    
    def setup(self, params, species_list, **kwargs):
        """Setup QSP potential with additional species list storage."""

        self._set_physical_constants(params.units, **kwargs)
        super()._copy_parameters(params)
        self._setup_species_parameters(species_list)
        self.initialize_potential_parameters(species_list)
        self.set_algorithm_parameters(**kwargs)
        self.create_parameter_matrix(species_list)
        self.set_force_function()
        self.validate_setup()

    def estimate_force_error(self, rc, algorithm_type="pppm", **kwargs):
        """
        Estimate force error for QSP potential using quadrature integration.
        
        Parameters
        ----------
        rc : float
            Cutoff radius
        algorithm_type : str
            Algorithm type (must be 'pppm')
        **kwargs : dict
            Additional parameters
            
        Returns
        -------
        float
            Estimated force error
        """
        
        return self._calculate_force_error_quadrature(rc)
    
    def _calculate_force_error_quadrature(self, rc):
        """Calculate force error using numerical quadrature."""
        pot_matrix = self.matrix.copy()
        
        # Rescale parameters for dimensionless calculation
        pot_matrix[:, :, 0] /= self.matrix[0, 0, 0]  # Normalize by e-e coupling
        pot_matrix[:, :, 1] *= self.a_ws              # Scale diffraction lengths
        pot_matrix[:, :, 2] /= self.matrix[0, 0, 0]   # Scale Pauli prefactor
        pot_matrix[:, :, 3] *= self.a_ws              # Scale Pauli exponent
        pot_matrix[:, :, 6] *= self.a_ws              # Scale Ewald parameter
        pot_matrix[:, :, 7] /= self.a_ws              # Scale cutoff
        
        r_c = rc / self.a_ws
        
        # Solid angle for integration
        dimensions = getattr(self, 'dimensions', 3)
        solid_angle = 2.0 * pi**(dimensions / 2) / gamma(dimensions / 2)
        
        # Force error integrand
        def integrand(r):
            force_magnitude = self.force_function(r, pot_matrix[0, 0])[1]
            return solid_angle * r**(dimensions - 1) * force_magnitude**2
        
        # Numerical integration
        f_err_a, _ = quad(integrand, a=r_c, b=inf)
        
        # Scale back to physical units
        QFactor = self.QFactor / (self.matrix[0, 0, 0] * self.total_num_ptcls)
        f_err = sqrt(f_err_a * 3.0 / (4.0 * pi)) * QFactor
        
        return f_err
    
    def pretty_print_info(self):
        """
        Print QSP potential information in a user-friendly way.

        Returns
        -------
        None
        """
        # Better formatting for arrays
        ee_params_str = array2string(self.ee_pauli_params, precision=6, separator=', ', floatmode='scientific')
        lambda_ei_str = array2string(self.lambda_ei, precision=6, separator=', ', floatmode='scientific')
        lambda_ii_str = array2string(self.lambda_ii, precision=6, separator=', ', floatmode='scientific')
        ei_diffraction_lengths = array2string(self.matrix[0, :, 1], precision=6, separator=', ', floatmode='scientific')
        msg = f"QSP type: {self.qsp_type}\n"
        msg += f"Pauli term: {self.qsp_pauli}\n"
        msg += f"Electron Pauli parameters: {ee_params_str}\n"
        msg += f"Electron de Broglie wavelength: {self.lambda_ee:.6e}\n"
        msg += f"Electron screening length: {self.matrix[0, 0, 1]:.6e}\n"
        msg += f"Electron-ion de Broglie wavelength: {lambda_ei_str}\n"
        msg += f"Electron-ion diffraction lengths: {ei_diffraction_lengths}\n"
        msg += f"Ion-ion de Broglie wavelength: {lambda_ii_str}\n"
        
        return msg

    def potential_derivatives(self, r_in: float, pot_matrix: Any):
        """
        Calculate the first and second derivatives of the QSP potential.
        """
        raise NotImplementedError("QSP potential derivatives not implemented")
    
# Numba-compiled force functions
@jit(nopython=True)
def pauli_force(r, pot_matrix):
    """
    Calculate Pauli exclusion term of QSP potential.
    
    Parameters
    ----------
    r : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    u_r : float
        Pauli potential energy
    f_r : float
        Pauli force magnitude
    """

    D = pot_matrix[2]  # Pauli prefactor
    F = pot_matrix[3]  # Pauli exponent
    A = pot_matrix[4]  # Pauli amplitude
    
    r2 = r * r
    F2 = F * F
    
    # Pauli potential: D * ln(1 - 0.5 * A * exp(-F^2 * r^2))
    exp_term = exp(-F2 * r2)
    u_r = D * log(1.0 - 0.5 * A * exp_term)
    
    # Pauli force: derivative of potential
    f_r = -D * r * F2 * A * exp_term / (1.0 - 0.5 * A * exp_term)
    
    return u_r, f_r


@jit(nopython=True)
def deutsch_force(r_in, pot_matrix):
    """
    Calculate Deutsch QSP force between particles.
    
    Parameters
    ----------
    r_in : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters [A, C, D, F, A_pauli, E, alpha, rs]
        
    Returns
    -------
    u_r : float
        Total potential energy
    f_r : float
        Total force magnitude
    """
    A = pot_matrix[0]      # Coulomb prefactor
    C = pot_matrix[1]      # Diffraction parameter
    E = pot_matrix[5]      # Diffraction flag
    alpha = pot_matrix[6]  # Ewald parameter
    rs = pot_matrix[7]     # Short-range cutoff
    
    # Apply short-range cutoff
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    r2 = r * r
    a2 = alpha * alpha
    
    # Ewald short-range terms
    u_ewald = A * erfc(alpha * r) / r
    f_ewald = u_ewald / r + A * (2.0 * alpha / sqrt(pi)) * exp(-a2 * r2) / r
    
    # Diffraction term
    u_diff = -A * exp(-C * r) / r
    f_diff = u_diff * (1.0 / r + C)
    
    # Pauli term
    u_pauli, f_pauli = pauli_force(r, pot_matrix)
    
    # Total
    u_r = u_ewald + E * u_diff + u_pauli
    f_r = f_ewald + E * f_diff + f_pauli
    
    return u_r, f_r


@jit(nopython=True)
def hansen_force(r_in, pot_matrix):
    """
    Calculate Hansen QSP force between particles.
    
    Parameters
    ----------
    r_in : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    u_r : float
        Total potential energy
    f_r : float
        Total force magnitude
    """
    A = pot_matrix[0]      # Coulomb prefactor
    C = pot_matrix[1]      # Diffraction parameter
    D = pot_matrix[2]      # Pauli prefactor
    F = pot_matrix[3]      # Pauli exponent
    alpha = pot_matrix[6]  # Ewald parameter
    rs = pot_matrix[7]     # Short-range cutoff
    
    # Apply short-range cutoff
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    r2 = r * r
    a2 = alpha * alpha
    
    # Ewald short-range terms
    u_ewald = A * erfc(alpha * r) / r
    f_ewald = u_ewald / r + A * (2.0 * alpha / sqrt(pi)) * exp(-a2 * r2) / r
    
    # Diffraction term
    u_diff = -A * exp(-C * r) / r
    f_diff = u_diff / r + A * C * exp(-C * r) / r
    
    # Pauli term (Hansen form)
    u_pauli = D * exp(-F * r2)
    f_pauli = 2.0 * r * D * F * exp(-F * r2)
    
    # Total
    u_r = u_ewald + u_diff + u_pauli
    f_r = f_ewald + f_diff + f_pauli
    
    return u_r, f_r


@jit(nopython=True)
def kelbg_force(r_in, pot_matrix):
    """
    Calculate Kelbg QSP force between particles.
    
    Parameters
    ----------
    r_in : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    u_r : float
        Total potential energy
    f_r : float
        Total force magnitude
    """
    A = pot_matrix[0]      # Coulomb prefactor
    C = pot_matrix[1]      # sqrt(2π)/λ_deB
    E = pot_matrix[5]      # Diffraction flag
    alpha = pot_matrix[6]  # Ewald parameter
    rs = pot_matrix[7]     # Short-range cutoff
    
    # Apply short-range cutoff
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    r2 = r * r
    C2 = C * C
    a2 = alpha * alpha
    
    # Ewald short-range terms
    u_ewald = A * erfc(alpha * r) / r
    f_ewald = u_ewald / r + A * (2.0 * alpha / sqrt(pi) / r) * exp(-a2 * r2)
    
    # Kelbg diffraction terms
    erfc_arg = C * r
    u_diff_1 = A * C * sqrt(pi) * erfc(erfc_arg)
    u_diff_2 = -A * exp(-C2 * r2) / r
    
    f_diff_1 = A * 2.0 * C2 * exp(-C2 * r2)
    f_diff_2 = u_diff_2 * (1.0 / r + 2.0 * C2 * r)
    
    # Pauli term
    u_pauli, f_pauli = pauli_force(r, pot_matrix)
    
    # Total
    u_r = u_ewald + E * (u_diff_1 + u_diff_2) + u_pauli
    f_r = f_ewald + E * (f_diff_1 + f_diff_2) + f_pauli
    
    return u_r, f_r