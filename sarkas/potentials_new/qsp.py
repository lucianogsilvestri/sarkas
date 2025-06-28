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
from numpy import array, exp, inf, pi, sqrt, zeros, isclose, ndarray
from scipy.integrate import quad
from scipy.special import gamma
from warnings import warn

from .base import PotentialBase
from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import TWOPI


class QuantumStatisticalPotential(PotentialBase):
    """
    Quantum Statistical Potential implementation.
    
    This potential includes quantum effects relevant for dense plasmas:
    - Pauli exclusion principle (fermionic statistics)
    - Diffraction effects (wave nature of particles)
    - Multiple formulations (Deutsch, Kelbg, Hansen)
    
    Attributes
    ----------
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
    """
    
    def __init__(self):
        super().__init__()
        self.type = "qsp"
        self.screening_length_type = "qsp"
        
        # QSP-specific parameters
        self.qsp_type = "deutsch"  # Default formulation
        self.qsp_pauli = True      # Include Pauli term by default
        
        # Custom diffractive lengths (optional)
        self.ee_diffractive_length = None
        self.ei_diffractive_length = None
        
        # Ion properties
        self.ai = None  # Ion Wigner-Seitz radius
        
        # Algorithm requirements
        self.pppm_alpha_ewald = None
        self.algorithm_type = "pppm"  # QSP requires PPPM
        
    def initialize_potential_parameters(self):
        """Initialize QSP-specific parameters."""
        # QSP uses Thomas-Fermi screening
        self.screening_length_type = "qsp"
        
        # Validate QSP type
        valid_types = ['deutsch', 'kelbg', 'hansen']
        if self.qsp_type.lower() not in valid_types:
            raise ValueError(f"qsp_type must be one of {valid_types}")
        self.qsp_type = self.qsp_type.lower()
        
        # Matrix will have 8 parameters per species pair
        pass
    
    def _setup_species_parameters(self, species_list):
        """Setup QSP-specific species parameters."""
        # Calculate ion properties
        total_ion_temperature = 0.0
        total_ion_number_density = 0.0
        
        # Assume electrons are first species, ions follow
        for species in species_list[1:]:  # Skip electrons
            total_ion_temperature += species.concentration * species.temperature
            total_ion_number_density += species.number_density
        
        # Ion Wigner-Seitz radius
        four_pi = 2.0 * TWOPI
        self.ai = (3.0 / (four_pi * total_ion_number_density)) ** (1.0 / 3.0)
        
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
    
    def create_parameter_matrix(self):
        """Create parameter matrix for QSP interactions."""
        if self.algorithm_type != "pppm":
            raise ValueError("QSP interaction requires PPPM algorithm")
        
        # Check charge neutrality
        if not isclose(self.total_net_charge, 0.0):
            warn("Total net charge is not zero.", category=AlgorithmWarning)
        
        self.matrix = zeros((self.num_species, self.num_species, 8))
        
        # Physical constants
        four_pi = 2.0 * TWOPI
        log_2 = log(2.0)
        deBroglie_const = TWOPI * self.hbar**2 / self.kB
        
        # Get electron properties (assume first species)
        electron_species = None
        total_ion_temperature = 0.0
        
        # Find electron species and calculate ion temperature
        for i, species in enumerate(self._species_list):
            if species.name == "e" or i == 0:  # Electron species
                electron_species = species
            else:  # Ion species
                total_ion_temperature += species.concentration * species.temperature
        
        if electron_species is None:
            raise ValueError("Could not identify electron species")
        
        theta = getattr(electron_species, 'degeneracy_parameter', 1.0)
        
        # Fill parameter matrix
        for i in range(self.num_species):
            for j in range(self.num_species):
                self._set_interaction_parameters(i, j, electron_species, 
                                               total_ion_temperature, theta,
                                               deBroglie_const, log_2)
        
        # Apply global settings
        if not self.qsp_pauli:
            self.matrix[:, :, 2] = 0.0  # Disable Pauli prefactor
            self.matrix[:, :, 3] = 0.0  # Disable Pauli exponent
            self.matrix[:, :, 4] = 0.0  # Disable Pauli amplitude
        
        # Set algorithm parameters
        self.matrix[:, :, 6] = self.ppmp_alpha_ewald or 0.0
        self.matrix[:, :, 7] = self.a_rs
    
    def _set_interaction_parameters(self, i, j, electron_species, 
                                   total_ion_temperature, theta,
                                   deBroglie_const, log_2):
        """Set parameters for specific species pair interaction."""
        # Get species properties
        q1 = self.species_charges[i]
        q2 = self.species_charges[j]
        m1 = self.species_masses[i]
        m2 = self.species_masses[j]
        
        # Reduced mass
        reduced_mass = (m1 * m2) / (m1 + m2)
        
        # Coulomb prefactor
        self.matrix[i, j, 0] = q1 * q2 / self.fourpie0
        
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
    
    def _set_ee_parameters(self, i, j, electron_species, reduced_mass,
                          deBroglie_const, theta, log_2):
        """Set electron-electron interaction parameters."""
        # Calculate or use custom de Broglie wavelength
        if hasattr(self, 'ee_diffractive_length') and self.ee_diffractive_length is not None:
            lambda_deB = self.ee_diffractive_length
        else:
            lambda_deB = sqrt(deBroglie_const / (reduced_mass * electron_species.temperature))
        
        # Diffraction parameter
        if self.qsp_type == "kelbg":
            self.matrix[i, j, 1] = sqrt(TWOPI) / lambda_deB
        else:
            self.matrix[i, j, 1] = TWOPI / lambda_deB
        
        # Pauli parameters (type-dependent)
        if self.qsp_type == "hansen":
            self.matrix[i, j, 2] = log_2 * self.kB * electron_species.temperature
            self.matrix[i, j, 3] = sqrt(2.0 * TWOPI / (log_2 * lambda_deB**2))
            self.matrix[i, j, 4] = 1.0
        else:
            # Deutsch/Kelbg use Jones-Murillo corrections
            a1, a2, a3 = 0.2975, 6.090, 1.541
            b1, b2, b3, b4 = 0.0842, 0.1027, 1.096, 1.359
            A_theta = 1.0 + a1 / (1.0 + a2 * theta**a3)
            B_theta = 1.0 + b1 * exp(-b2 * theta**b3) / theta**b4
            
            self.matrix[i, j, 2] = -self.kB * electron_species.temperature
            self.matrix[i, j, 3] = sqrt(TWOPI * B_theta) / lambda_deB
            self.matrix[i, j, 4] = A_theta
        
        # Enable diffraction term
        self.matrix[i, j, 5] = 1.0
    
    def _set_ei_parameters(self, i, j, electron_species, reduced_mass,
                          deBroglie_const, ion_index):
        """Set electron-ion interaction parameters."""
        # Calculate or use custom de Broglie wavelength
        if (hasattr(self, 'ei_diffractive_length') and 
            self.ei_diffractive_length is not None):
            lambda_deB = self.ei_diffractive_length[ion_index]
        else:
            lambda_deB = sqrt(deBroglie_const / (reduced_mass * electron_species.temperature))
        
        # Diffraction parameter
        if self.qsp_type == "kelbg":
            self.matrix[i, j, 1] = sqrt(TWOPI) / lambda_deB
        else:
            self.matrix[i, j, 1] = TWOPI / lambda_deB
        
        # No Pauli term for e-i interactions
        self.matrix[i, j, 2] = 0.0
        self.matrix[i, j, 3] = 0.0
        self.matrix[i, j, 4] = 0.0
        
        # Enable diffraction term
        self.matrix[i, j, 5] = 1.0
    
    def _set_ii_parameters(self, i, j, reduced_mass, deBroglie_const, 
                          total_ion_temperature):
        """Set ion-ion interaction parameters (classical)."""
        # Classical ion-ion interactions (no quantum effects)
        lambda_deB = sqrt(deBroglie_const / (reduced_mass * total_ion_temperature))
        
        # Diffraction parameter (not used for i-i)
        if self.qsp_type == "kelbg":
            self.matrix[i, j, 1] = sqrt(TWOPI) / lambda_deB
        else:
            self.matrix[i, j, 1] = TWOPI / lambda_deB
        
        # No quantum terms for i-i interactions
        self.matrix[i, j, 2] = 0.0  # No Pauli term
        self.matrix[i, j, 3] = 0.0
        self.matrix[i, j, 4] = 0.0
        self.matrix[i, j, 5] = 0.0  # No diffraction term
    
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
    
    def set_algorithm_parameters(self, algorithm_type, **kwargs):
        """Set algorithm-specific parameters for QSP."""
        if algorithm_type != "pppm":
            raise ValueError("QSP potential requires PPPM algorithm")
        
        self.algorithm_type = algorithm_type
        self.pppm_alpha_ewald = kwargs.get('alpha_ewald', 0.0)
        
        # Recreate matrix with new parameters
        if hasattr(self, '_species_list'):  # Only if already set up
            self.create_parameter_matrix()
            self.set_force_function()
    
    def setup(self, params, species_list):
        """Setup QSP potential with additional species list storage."""
        # Store species list for parameter matrix creation
        self._species_list = species_list
        
        # Call parent setup
        super().setup(params, species_list)
    
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
        if algorithm_type != "pppm":
            warn("QSP force error calculation requires PPPM algorithm", 
                 category=AlgorithmWarning)
            return inf
        
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
        """Print QSP potential information."""
        # Extract wavelengths and screening lengths
        ii_scr_len = 1.0 / self.matrix[1, 1, 1] if self.matrix[1, 1, 1] != 0 else inf
        ei_scr_len = 1.0 / self.matrix[0, 1, 1] if self.matrix[0, 1, 1] != 0 else inf  
        ee_scr_len = 1.0 / self.matrix[0, 0, 1] if self.matrix[0, 0, 1] != 0 else inf
        
        # de Broglie wavelengths
        if self.qsp_type == "kelbg":
            e_deBroglie_lambda = sqrt(2.0) * pi / self.matrix[0, 0, 1]
            i_deBroglie_lambda = sqrt(2.0) * pi / self.matrix[1, 1, 1] if self.matrix[1, 1, 1] != 0 else inf
        else:
            e_deBroglie_lambda = TWOPI / self.matrix[0, 0, 1]
            i_deBroglie_lambda = TWOPI / self.matrix[1, 1, 1] if self.matrix[1, 1, 1] != 0 else inf
        
        msg = f"QSP type: {self.qsp_type}\\n"
        msg += f"Pauli term: {self.qsp_pauli}\\n"
        msg += f"e de Broglie wavelength = {e_deBroglie_lambda / self.a_ws:.4e} a_ws = {e_deBroglie_lambda:.6e} {self.units_dict['length']}\\n"
        
        if i_deBroglie_lambda != inf:
            msg += f"ion de Broglie wavelength = {i_deBroglie_lambda / self.a_ws:.4e} a_ws = {i_deBroglie_lambda:.6e} {self.units_dict['length']}\\n"
        
        msg += f"Screening lengths refer to the diffraction term exponential argument.\\n"
        msg += f"e-e screening length = {ee_scr_len / self.a_ws:.4e} a_ws = {ee_scr_len:.6e} {self.units_dict['length']}\\n"
        
        if ei_scr_len != inf:
            msg += f"e-i screening length = {ei_scr_len / self.a_ws:.4e} a_ws = {ei_scr_len:.6e} {self.units_dict['length']}\\n"
        
        if ii_scr_len != inf:
            msg += f"i-i screening length = {ii_scr_len / self.a_ws:.4e} a_ws = {ii_scr_len:.6e} {self.units_dict['length']}\\n"
        
        msg += f"e-i coupling constant = {self.coupling_constant:.4e}\\n"
        
        if hasattr(self, 'ai') and self.ai is not None:
            msg += f"Ion Wigner-Seitz radius = {self.ai / self.a_ws:.4e} a_ws = {self.ai:.6e} {self.units_dict['length']}"
        
        print(msg)


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