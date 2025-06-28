"""
Coulomb potential implementation.

This module implements the Coulomb potential for charged particle interactions.
It supports both pure Coulomb (PP method) and Ewald-split Coulomb (PPPM method).

Potential
*********

The Coulomb potential between particles i and j is:

.. math::
    U_{ij}(r) = \\frac{q_i q_j}{4\\pi \\epsilon_0 r}

For PPPM, the short-range part uses the complementary error function:

.. math::
    U_{ij}^{\\text{short}}(r) = \\frac{q_i q_j}{4\\pi \\epsilon_0 r} \\text{erfc}(\\alpha r)

Potential Attributes
********************

The elements of the parameter matrix are:

.. code-block::

    matrix[i,j,0] = q_i * q_j / (4π ε₀)  # Force prefactor
    matrix[i,j,1] = alpha_ewald            # Ewald parameter (PPPM only)  
    matrix[i,j,2] = a_rs                   # Short-range cutoff
"""

from math import erfc
from numba import jit
from numpy import exp, inf, pi, sqrt, zeros
from warnings import warn

from .base import PotentialBase
from ..utilities.exceptions import AlgorithmWarning
from ..utilities.maths import force_error_analytic_pp


class CoulombPotential(PotentialBase):
    """
    Coulomb potential implementation.
    
    The Coulomb potential between particles i and j is:
    U_ij(r) = q_i * q_j / (4π ε₀ r)
    
    This implementation supports both pure Coulomb (PP method with care)
    and Ewald-split Coulomb (PPPM method).
    
    Attributes
    ----------
    matrix : numpy.ndarray
        Shape (num_species, num_species, 3)
        matrix[i,j,0] = q_i * q_j / (4π ε₀)  # Force prefactor
        matrix[i,j,1] = alpha_ewald            # Ewald parameter (PPPM only)
        matrix[i,j,2] = a_rs                   # Short-range cutoff
    pppm_alpha_ewald : float
        Ewald parameter for PPPM decomposition
    algorithm_type : str
        Current algorithm type ('pp' or 'pppm')
    """
    
    def __init__(self):
        super().__init__()
        self.type = "coulomb"
        self.screening_length_type = "coulomb"  # Unscreened
        
        # PPPM-specific parameters (set by algorithm)
        self.pppm_alpha_ewald = None
        self.algorithm_type = "pppm"  # Will be set by interaction solver
    
    def initialize_potential_parameters(self):
        """Initialize Coulomb-specific parameters."""
        # Coulomb is unscreened
        self.screening_length = inf
        
        # Matrix will have 3 parameters per species pair:
        # [force_prefactor, ewald_parameter, short_range_cutoff]
    
    def create_parameter_matrix(self, species_list, **kwargs):
        """
        Create parameter matrix for Coulomb interactions.
        
        Parameters
        ----------
        species_list : list
            List of species objects containing charge, mass, name attributes
        **kwargs : dict
            Additional parameters:
            - alpha_ewald : float, optional (overrides self.pppm_alpha_ewald)
            - short_cutoff : float, optional (overrides self.a_rs)
        
        Notes
        -----
        Matrix structure: [force_prefactor, ewald_parameter, short_cutoff]
        Handles electron background species automatically.
        """
        
        # Handle electron background species
        active_species = self._get_active_species(species_list)
        num_active = len(active_species)
        
        # Initialize matrix
        self.matrix = zeros((num_active, num_active, 3))
        
        # Get parameters (allow kwargs to override)
        alpha_ewald = kwargs.get('alpha_ewald', self.pppm_alpha_ewald)
        short_cutoff = kwargs.get('short_cutoff', self.a_rs)
        
        # Fill matrix elements
        for i, sp1 in enumerate(active_species):
            for j, sp2 in enumerate(active_species):
                # Force prefactor: q_i * q_j / (4π ε₀)
                self.matrix[i, j, 0] = sp1.charge * sp2.charge / self.fourpie0
                
                # Ewald parameter (for PPPM decomposition)
                self.matrix[i, j, 1] = alpha_ewald if alpha_ewald is not None else 0.0
                
                # Short-range cutoff
                self.matrix[i, j, 2] = short_cutoff
    
    def _get_active_species(self, species_list):
        """
        Get list of active species (excluding electron background).
        
        Parameters
        ----------
        species_list : list
            Full list of species
            
        Returns
        -------
        list
            Active species (excluding electron background)
        """
        # Filter out electron background species
        active_species = []
        for species in species_list:
            # Check various possible names for electron background
            if hasattr(species, 'name'):
                name = species.name.lower()
                if name not in ['electron_background', 'e_background', 'background']:
                    active_species.append(species)
            else:
                # If no name attribute, assume it's active
                active_species.append(species)
        
        return active_species
    
    def set_force_function(self):
        """Set the appropriate force function based on algorithm."""
        if self.algorithm_type == "pppm":
            self.force_function = coulomb_force_pppm
        else:
            self.force_function = coulomb_force
    
    def set_algorithm_parameters(self, algorithm_type, **kwargs):
        """
        Set algorithm-specific parameters.
        
        This method is called by interaction solvers to configure
        the potential for their specific needs.
        
        Parameters
        ----------
        algorithm_type : str
            Type of algorithm ('pp', 'pppm', 'fmm')
        **kwargs : dict
            Algorithm-specific parameters:
            - alpha_ewald : float (for PPPM)
        """
        self.algorithm_type = algorithm_type
        
        if algorithm_type == "pppm":
            self.pppm_alpha_ewald = kwargs.get('alpha_ewald', 0.0)
            # Recreate matrix with Ewald parameters
            self.create_parameter_matrix()
            self.set_force_function()
        
        elif algorithm_type == "pp":
            # Warn about potential issues with pure Coulomb PP
            warn("Use the PP method with care for pure Coulomb interactions.", 
                 category=AlgorithmWarning)
    
    def estimate_force_error(self, rc, algorithm_type="pp", **kwargs):
        """
        Estimate force error for Coulomb potential.
        
        Parameters
        ----------
        rc : float
            Cutoff radius
        algorithm_type : str
            Algorithm type
        **kwargs : dict
            Additional parameters:
            - alpha_ewald : float (for PPPM force error estimation)
            
        Returns
        -------
        float
            Estimated force error
        """
        if algorithm_type == "pp":
            # For pure PP, force error is essentially infinite due to long range
            warn("PP method with pure Coulomb has large force errors due to long-range nature",
                 category=AlgorithmWarning)
            return inf
            
        elif algorithm_type == "pppm":
            alpha_ewald = kwargs.get('alpha_ewald', self.pppm_alpha_ewald)
            if alpha_ewald is not None:
                # Use analytical PP force error formula for PPPM
                rescaling_constant = sqrt(self.total_num_ptcls) * self.a_ws**2
                return force_error_analytic_pp(
                    self.type, rc, self.screening_length, alpha_ewald, rescaling_constant
                )
        
        return 0.0
    
    def pretty_print_info(self):
        """Print Coulomb potential information."""
        msg = f"Effective coupling constant: Gamma_eff = {self.coupling_constant:.2f}\\n"
        
        if self.algorithm_type != "fmm":
            msg += f"Short-range cutoff radius: a_rs = {self.a_rs:.6e} {self.units_dict['length']}\\n"
        
        if self.algorithm_type == "pppm" and self.pppm_alpha_ewald is not None:
            msg += f"Ewald parameter: alpha = {self.pppm_alpha_ewald:.6e} {self.units_dict['inverse length']}\\n"
        
        print(msg)


# Numba-compiled force functions
@jit(nopython=True)
def coulomb_force(r_in, pot_matrix):
    """
    Calculate bare Coulomb potential and force.
    
    Parameters
    ----------
    r_in : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters [force_prefactor, ewald_param, short_cutoff]
        
    Returns
    -------
    u_r : float
        Potential energy
    f_r : float
        Force magnitude
        
    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([1.0, 0.0, 0.0])
    >>> coulomb_force(r, pot_matrix)
    (0.5, 0.25)
    """
    # Apply short-range cutoff
    rs = pot_matrix[2]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    # Coulomb potential: U = q1*q2/(4πε₀*r)
    u_r = pot_matrix[0] / r
    # Force: F = dU/dr = q1*q2/(4πε₀*r²)
    f_r = u_r / r
    
    return u_r, f_r


@jit(nopython=True)
def coulomb_force_pppm(r_in, pot_matrix):
    """
    Calculate Ewald-split Coulomb potential and force for PPPM.
    
    Parameters
    ----------
    r_in : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters [force_prefactor, alpha_ewald, short_cutoff]
        
    Returns
    -------
    u_r : float
        Short-range potential energy
    f_r : float
        Short-range force magnitude
        
    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([1.0, 0.5, 0.0])
    >>> coulomb_force_pppm(r, pot_matrix)
    (0.07864960352514257, 0.14310167611771996)
    """
    # Apply short-range cutoff
    rs = pot_matrix[2]
    r = r_in * (r_in >= rs) + rs * (r_in < rs)
    
    alpha = pot_matrix[1]  # Ewald parameter
    alpha_r = alpha * r
    
    # Short-range Ewald potential: U = q1*q2*erfc(α*r)/(4πε₀*r)
    u_r = pot_matrix[0] * erfc(alpha_r) / r
    
    # Short-range Ewald force
    r2 = r * r
    f1 = erfc(alpha_r) / r2
    f2 = (2.0 * alpha / sqrt(pi) / r) * exp(-(alpha_r**2))
    f_r = pot_matrix[0] * (f1 + f2)
    
    return u_r, f_r


def potential_derivatives(r, pot_matrix):
    """
    Calculate potential derivatives for Coulomb potential.
    
    Parameters
    ----------
    r : float
        Distance between particles
    pot_matrix : numpy.ndarray
        Potential parameters
        
    Returns
    -------
    u_r : float
        Potential value
    dv_dr : float
        First derivative
    d2v_dr2 : float
        Second derivative
        
    Examples
    --------
    >>> import numpy as np
    >>> r = 2.0
    >>> pot_matrix = np.array([1.0, 0.0, 0.0])
    >>> potential_derivatives(r, pot_matrix)
    (0.5, -0.25, 0.25)
    """
    u_r = pot_matrix[0] / r
    dv_dr = -u_r / r
    d2v_dr2 = 2.0 * dv_dr / r
    
    return u_r, dv_dr, d2v_dr2