"""
Base class for all potentials.

This module defines the abstract base class that all potential implementations
must inherit from. It provides common functionality for parameter management,
setup coordination, and interface contracts.
"""

from abc import ABC, abstractmethod
from numpy import inf
from warnings import warn

from ..utilities.exceptions import AlgorithmWarning


class PotentialBase(ABC):
    """
    Abstract base class for all potentials.
    
    This class defines the interface that all potentials must implement.
    It handles common functionality like parameter management, setup coordination,
    and provides hooks for potential-specific implementations.
    
    Attributes
    ----------
    type : str
        Type of potential (e.g., 'coulomb', 'yukawa', 'lj')
    matrix : numpy.ndarray
        Matrix of potential parameters for all species pairs
    force_function : callable
        Numba-compiled force calculation function
    screening_length : float
        Screening length for the potential (inf for unscreened)
    screening_length_type : str
        Type of screening calculation used
    num_species : int
        Number of particle species
    species_charges : numpy.ndarray
        Array of species charges
    """
    
    def __init__(self):
        # Potential identification
        self.type = None
        
        # Core potential data
        self.matrix = None
        self.force_function = None
        
        # Screening parameters
        self.screening_length = None
        self.screening_length_type = "unscreened"
        
        # System parameters (copied from simulation params)
        self.num_species = None
        self.species_charges = None
        self.species_masses = None
        self.species_num = None
        
        # Physical constants (copied from simulation params)
        self.fourpie0 = None
        self.kB = None
        self.a_ws = None
        self.T_desired = None
        self.eV2J = None
        self.eV2K = None
        
        # System properties
        self.total_num_ptcls = None
        self.total_net_charge = None
        self.QFactor = None
        self.coupling_constant = None
        
        # Computational parameters
        self.a_rs = 0.0  # Short-range cutoff
        self.force_error = None
        
        # Units
        self.units = None
        self.units_dict = None
    
    def setup(self, params, species_list):
        """
        Setup the potential with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters object containing physical constants,
            box properties, and system parameters
        species_list : list
            List of species objects containing per-species properties
        """
        # Copy essential parameters from simulation
        self._copy_parameters(params)
        
        # Setup species-specific parameters
        self._setup_species_parameters(species_list)
        
        # Calculate screening if applicable
        self._setup_screening(species_list)
        
        # Initialize potential-specific parameters
        self.initialize_potential_parameters()
        
        # Create the parameter matrix
        self.create_parameter_matrix()
        
        # Set the force function
        self.set_force_function()
        
        # Validate the setup
        self.validate_setup()
    
    def _copy_parameters(self, params):
        """Copy necessary parameters from simulation params."""
        # Physical constants
        self.fourpie0 = params.fourpie0
        self.kB = params.kB
        self.a_ws = params.a_ws
        self.T_desired = params.T_desired
        self.eV2J = getattr(params, 'eV2J', 1.0)
        self.eV2K = getattr(params, 'eV2K', 1.0)
        
        # System properties
        self.total_num_ptcls = params.total_num_ptcls
        self.total_net_charge = params.total_net_charge
        self.QFactor = params.QFactor
        self.coupling_constant = getattr(params, 'coupling_constant', 0.0)
        
        # Species information
        self.num_species = params.num_species
        self.species_charges = params.species_charges.copy()
        self.species_masses = params.species_masses.copy()
        self.species_num = params.species_num.copy()
        
        # Units
        self.units = params.units
        self.units_dict = params.units_dict
        
        # Short-range cutoff
        self.a_rs = getattr(params, 'a_rs', 0.0)
    
    def _setup_species_parameters(self, species_list):
        """
        Setup parameters specific to species.
        
        This method can be overridden by subclasses that need
        species-specific parameters (e.g., Lennard-Jones sigmas).
        
        Parameters
        ----------
        species_list : list
            List of species objects
        """
        # Default implementation - can be overridden
        pass
    
    def _setup_screening(self, species_list):
        """Setup screening parameters if applicable."""
        if hasattr(self, 'screening_length_type') and self.screening_length_type != "unscreened":
            self.calculate_screening_length(species_list)
    
    @abstractmethod
    def initialize_potential_parameters(self):
        """
        Initialize potential-specific parameters.
        
        This method should set any potential-specific attributes
        and prepare for matrix creation.
        """
        pass
    
    @abstractmethod
    def create_parameter_matrix(self):
        """
        Create the parameter matrix for species interactions.
        
        This method should populate self.matrix with the appropriate
        potential parameters for all species pairs.
        """
        pass
    
    @abstractmethod
    def set_force_function(self):
        """
        Set the force calculation function.
        
        This method should set self.force_function to the appropriate
        Numba-compiled function for force calculations.
        """
        pass
    
    def calculate_screening_length(self, species_list):
        """
        Calculate screening length based on screening_length_type.
        
        Parameters
        ----------
        species_list : list
            List of species objects
        """
        if self.screening_length_type in ["thomas-fermi", "tf"]:
            # Use electron Thomas-Fermi wavelength
            if hasattr(species_list[-1], 'ThomasFermi_wavelength'):
                self.screening_length = species_list[-1].ThomasFermi_wavelength
            else:
                warn("Thomas-Fermi wavelength not available, using unscreened", 
                     category=AlgorithmWarning)
                self.screening_length = inf
                
        elif self.screening_length_type in ["debye", "debye-huckel", "dh"]:
            # Use Debye length
            if hasattr(species_list[-1], 'debye_length'):
                self.screening_length = species_list[-1].debye_length
            else:
                warn("Debye length not available, using unscreened", 
                     category=AlgorithmWarning)
                self.screening_length = inf
                
        elif self.screening_length_type == "custom":
            if self.screening_length is None:
                raise ValueError("Custom screening length not specified")
                
        elif self.screening_length_type in ["coulomb", "unscreened"]:
            self.screening_length = inf
            
        else:
            warn(f"Unknown screening type {self.screening_length_type}, using unscreened",
                 category=AlgorithmWarning)
            self.screening_length = inf
    
    def validate_setup(self):
        """Validate that the potential is properly set up."""
        if self.matrix is None:
            raise ValueError("Parameter matrix not initialized")
        if self.force_function is None:
            raise ValueError("Force function not set")
        if self.screening_length is None:
            raise ValueError("Screening length not set")
    
    def get_force_function(self):
        """
        Return the force calculation function.
        
        Returns
        -------
        callable
            Numba-compiled force function
        """
        return self.force_function
    
    def get_parameter_matrix(self):
        """
        Return the parameter matrix.
        
        Returns
        -------
        numpy.ndarray
            Parameter matrix for all species pairs
        """
        return self.matrix
    
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
            Algorithm-specific parameters
        """
        # Default implementation - can be overridden by subclasses
        pass
    
    @abstractmethod
    def pretty_print_info(self):
        """Print potential-specific information."""
        pass
    
    def estimate_force_error(self, rc, algorithm_type="pp", **kwargs):
        """
        Estimate force error for given cutoff radius.
        
        Parameters
        ----------
        rc : float
            Cutoff radius
        algorithm_type : str
            Algorithm type ('pp', 'pppm', etc.)
        **kwargs : dict
            Additional algorithm-specific parameters
            
        Returns
        -------
        float
            Estimated force error
        """
        # Default implementation - should be overridden by specific potentials
        return 0.0