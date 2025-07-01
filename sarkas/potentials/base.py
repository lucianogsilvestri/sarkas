"""
Base class for all potentials.

This module defines the abstract base class that all potential implementations
must inherit from. It provides common functionality for parameter management,
setup coordination, and interface contracts.
"""

from abc import ABC, abstractmethod
from numpy import inf, pi
from warnings import warn
from typing import Any, Dict, List, Callable, Optional

from scipy.constants import physical_constants

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
        Type of potential (e.g., 'coulomb', 'yukawa', 'lj').
    matrix : numpy.ndarray
        Matrix of potential parameters for all species pairs.
    force_function : callable
        Numba-compiled force calculation function.
    screening_length : float
        Screening length for the potential (inf for unscreened).
    screening_length_type : str
        Type of screening calculation used.
    num_species : int
        Number of particle species.
    species_charges : numpy.ndarray
        Array of species charges.

    Extensibility
    -------------
    To implement a new potential, subclass PotentialBase and implement the following methods:

    - initialize_potential_parameters(self):
        Set any potential-specific attributes and prepare for matrix creation.
    - create_parameter_matrix(self):
        Populate self.matrix with the appropriate potential parameters for all species pairs.
    - set_force_function(self):
        Set self.force_function to the appropriate Numba-compiled function for force calculations.
    - pretty_print_info(self):
        Print information about the potential and its parameters.

    Optionally, override _setup_species_parameters and calculate_screening_length for custom logic.

    Example
    -------
    Subclassing the base class::

        class MyCustomPotential(PotentialBase):
            def initialize_potential_parameters(self):
                # Set up custom parameters
                pass
            def create_parameter_matrix(self):
                # Build the parameter matrix
                pass
            def set_force_function(self):
                # Assign the force function
                pass
            def pretty_print_info(self):
                print(f"Custom potential: {self.type}")
    """

    def __init__(self) -> None:
        # Potential identification
        self.type: Optional[str] = None

        # Core potential data
        self.params: Any = None
        self.matrix: Any = None
        self.force_function: Optional[Callable] = None


        # Physical constants (copied from simulation params)
        self.eps0 = physical_constants["vacuum electric permittivity"][0]
        self.fourpie0 = 4.0 * pi * self.eps0
        self.kB = physical_constants["Boltzmann constant"][0]
        self.eV2J = physical_constants["electron volt-joule relationship"][0]
        self.eV2K = physical_constants["electron volt-kelvin relationship"][0]

        # System properties
        self.total_num_ptcls: Optional[int] = None
        self.total_net_charge: Optional[float] = None
        self.QFactor: Optional[float] = None
        self.coupling_constant: Optional[float] = None

        # Computational parameters
        self.algorithm_type: str = "pp"
        self.pppm_alpha_ewald: Optional[float] = None
        self.a_rs =  0.0  # Short-range cutoff. This is set to avoid division by zero when two particles get too close.
        self.force_error = inf

        # Units
        self.units: str = "si"
        self.units_dict: Dict[str, str] = {
            "length": "m",
            "time": "s",
            "mass": "kg",
            "temperature": "K",
            "energy": "J",
        }

    def setup(self, params: Any, species_list: List[Any], **kwargs) -> None:
        """
        Setup the potential with simulation parameters.

        Parameters
        ----------
        params : object
            Simulation parameters object containing physical constants,
            box properties, and system parameters.
        species_list : list
            List of species objects containing per-species properties.
        """
        self._set_physical_constants(params.units, **kwargs)
        self._copy_parameters(params)
        self._setup_species_parameters(species_list)
        self.initialize_potential_parameters(species_list)
        self.set_algorithm_parameters(**kwargs)
        self.validate_params()
        self.validate_algorithm_parameters()
        self.create_parameter_matrix()
        self.set_force_function()
        self.validate_setup()

    def _set_physical_constants(self, units: str) -> None:
        """
        Set physical constants from simulation parameters.
        """
        if units is not None:
            self.units = units
        
        if self.units in ['cgs', 'atomic', 'hartree']:
            self.fourpie0 = 1.0
            self.eps0 = 1.0
            J2erg = 1.0e7  # erg/J
            self.kB = self.kB * J2erg
        elif self.units in ['si', 'mks']:
            self.eps0 = physical_constants["vacuum electric permittivity"][0]
            self.fourpie0 = 4.0 * pi * self.eps0
        elif self.units == "dimensionless":
            self.fourpie0 = 1.0
            self.eps0 = 1.0
            self.kB = 1.0
            

    def _copy_parameters(self, params: Any) -> None:
        """
        Copy necessary parameters from simulation params.

        Parameters
        ----------
        params : object
            Simulation parameters object.
        """
        self.a_ws = params.a_ws
        self.T_desired = params.T_desired
        self.total_num_ptcls = params.total_num_ptcls
        self.total_net_charge = params.total_net_charge
        self.QFactor = params.QFactor
        self.coupling_constant = getattr(params, 'coupling_constant', 0.0)
        self.units = params.units
        self.units_dict = params.units_dict

    def _setup_species_parameters(self, species_list: List[Any]) -> None:
        """
        Setup parameters specific to species.

        This method can be overridden by subclasses that need
        species-specific parameters (e.g., Lennard-Jones sigmas).

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        pass

    def _setup_screening(self, species_list: List[Any]) -> None:
        """
        Setup screening parameters if applicable.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        if hasattr(self, 'screening_length_type') and self.screening_length_type != "unscreened":
            self.calculate_screening_length(species_list)

    @abstractmethod
    def initialize_potential_parameters(self, species_list: List[Any]) -> None:
        """
        Initialize potential-specific parameters.

        This method should set any potential-specific attributes
        and prepare for matrix creation.

        Parameters
        ----------
        species_list : list
            List of species objects.
        """
        pass

    @abstractmethod
    def create_parameter_matrix(self) -> None:
        """
        Create the parameter matrix for species interactions.

        This method should populate self.matrix with the appropriate
        potential parameters for all species pairs.
        """
        pass

    @abstractmethod
    def set_force_function(self) -> None:
        """
        Set the force calculation function.

        This method should set self.force_function to the appropriate
        Numba-compiled function for force calculations.
        """
        pass

    def calculate_screening_length(self, species_list: List[Any]) -> None:
        """
        Calculate screening length based on screening_length_type.

        Parameters
        ----------
        species_list : list
            List of species objects.
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
    
    def validate_params(self):
        """Validate that the potential parameters are set correctly."""
        if self.params is None:
            raise ValueError("self.params is still None after initialize_potential_parameters. Check that it is set correctly.")
        
    def validate_algorithm_parameters(self):
        """Validate that the algorithm parameters are set correctly."""
    
        if self.algorithm_type == "pppm" :
            if self.pppm_alpha_ewald is None:
                raise AttributeError("PPPM alpha Ewald must be set")
    
    def validate_setup(self):
        """Validate that the potential is properly set up."""
        if self.matrix is None:
            raise ValueError("Parameter matrix not initialized")
        if self.force_function is None:
            raise ValueError("Force function not set")
        
        u_r, f_r = self.force_function(2.0, self.matrix[0,0])
        if u_r is None or f_r is None:
            raise ValueError("Force function returned None")
        if not isinstance(u_r, float) or not isinstance(f_r, float):
            raise ValueError("Force function returned non-float values")
        if not isinstance(u_r, float) or not isinstance(f_r, float):
            raise ValueError("Force function returned non-float values")
        if not isinstance(u_r, float) or not isinstance(f_r, float):
            raise ValueError("Force function returned non-float values")
