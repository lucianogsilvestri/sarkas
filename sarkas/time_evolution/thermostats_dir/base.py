"""
Base classes for all thermostats.
"""

from abc import ABC, abstractmethod
from numpy import sqrt, random


class ThermostatBase(ABC):
    """
    Abstract base class for all thermostats.
    
    All thermostats must implement the update() method which handles:
    1. Temperature calculation (if needed)
    2. Thermostat-specific velocity modifications
    3. Conservation properties maintenance
    
    Attributes
    ----------
    name : str
        Thermostat name (auto-generated from class name)
    target_temperature : float
        Target temperature for the thermostat
    coupling_parameter : float
        Thermostat-specific coupling strength parameter
    relaxation_time : float
        Characteristic relaxation time
    supports_anisotropic : bool
        Whether thermostat supports anisotropic temperature control
    supports_local : bool
        Whether thermostat supports local temperature control
    conserves_momentum : bool
        Whether thermostat conserves total momentum
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.target_temperatures = None
        self.coupling_parameter = None
        self.relaxation_time = None
        
        # Capability flags
        self.supports_anisotropic = False
        self.supports_local = False
        self.conserves_momentum = True
        self.is_deterministic = True
        
        # System parameters
        self.dimensions = None
        self.total_num_ptcls = None
        self.species_masses = None
        self.species_num = None
        self.degrees_of_freedom = None
        
        # Thermostat state variables
        self.current_temperatures = None
        self.kinetic_energy = None
        self.instantaneous_temperatures = None
        
        # Timestep information
        self.dt = None
        
    @abstractmethod
    def update(self, ptcls):
        """
        Apply thermostat to particle velocities.
        
        This method must modify particle velocities to control temperature
        according to the specific thermostat algorithm.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to thermostat (velocities modified in-place)
        """
        pass
    
    def setup(self, params, **kwargs):
        """
        Setup thermostat with simulation parameters.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters containing temperature, masses, etc.
        **kwargs : dict
            Additional thermostat-specific parameters
        """
        # Basic parameters
        self.target_temperatures = params.species_temperatures.copy()

        # Overwrite with kwargs if provided
        if 'target_temperatures' in kwargs:
            self.target_temperatures = kwargs['target_temperatures']

        if self.target_temperatures is None:
            raise ValueError("Target temperature must be specified")
            
        self.dt = getattr(params, 'dt', None) or kwargs.get('dt')

        if self.dt is None:
            raise ValueError("Timestep (dt) must be specified")
            
        # System parameters
        self.dimensions = params.dimensions
        self.total_num_ptcls = params.total_num_ptcls
        self.species_masses = params.species_masses
        self.species_num = params.species_num
        
        # Calculate degrees of freedom
        self.degrees_of_freedom = self.total_num_ptcls * self.dimensions
        
        # Thermostat-specific parameters
        self.coupling_parameter = (getattr(params, 'thermostat_coupling', None) or
                                  kwargs.get('coupling_parameter') or
                                  kwargs.get('coupling_strength', 1.0))
        
        self.relaxation_time = (getattr(params, 'thermostat_relaxation_time', None) or
                               kwargs.get('relaxation_time'))
        
        # Pre-compute constants for performance
        self._sqrt_dt = sqrt(self.dt)
        
        # Initialize thermostat state
        self._initialize_state(params, **kwargs)
        
        # Validate parameters
        self.validate_parameters()
    
    def _initialize_state(self, params, **kwargs):
        """Initialize thermostat-specific state variables."""
        # Base implementation - override in subclasses
        self.current_temperatures = self.target_temperatures
        self.kinetic_energy = 0.0
        self.instantaneous_temperatures = 0.0
    
    def validate_parameters(self):
        """
        Validate thermostat parameters.
        Override in subclasses for specific validation requirements.
        """
        if self.target_temperatures <= 0:
            raise ValueError("Target temperature must be positive")
        
        if self.dt <= 0:
            raise ValueError("Timestep must be positive")
    
    def get_info(self):
        """
        Return thermostat information.
        
        Returns
        -------
        dict
            Dictionary containing thermostat properties
        """
        return {
            'name': self.name,
            'target_temperatures': self.target_temperature,
            'instantaneous_temperatures': self.instantaneous_temperatures,
            'supports_anisotropic': self.supports_anisotropic,
            'supports_local': self.supports_local,
            'conserves_momentum': self.conserves_momentum,
            'is_deterministic': self.is_deterministic,
        }
    
    def pretty_print(self):
        """Print thermostat information in a user-friendly format."""
        info = self.get_info()
        msg = f"\nTHERMOSTAT: {info['name'].upper()}\n"
        msg += f"Target temperature: {info['target_temperatures']:.6e}\n"
        
        if 'instantaneous_temperatures' in info and info['instantaneous_temperatures'] is not None:
            msg += f"Instantaneous temperature: {info['instantaneous_temperatures']:.6e}\n"

        capabilities = []
        if info['supports_anisotropic']:
            capabilities.append("anisotropic control")
        if info['supports_local']:
            capabilities.append("local control")
        if info['conserves_momentum']:
            capabilities.append("momentum conservation")
        if info['is_deterministic']:
            capabilities.append("deterministic")
        else:
            capabilities.append("stochastic")
            
        if capabilities:
            msg += f"Capabilities: {', '.join(capabilities)}"

        return msg


class StochasticThermostatBase(ThermostatBase):
    """
    Base class for stochastic thermostats.
    
    Provides common functionality for thermostats that use random forces
    or velocity modifications (Langevin, Anderson, etc.).
    """
    
    def __init__(self):
        super().__init__()
        self.is_deterministic = False
        self.random_seed = None
        self.friction_coefficient = None
        
        # Random number generation
        self.rng = None
        
    def setup(self, params, **kwargs):
        super().setup(params, **kwargs)
        
        # Setup random number generation
        self.random_seed = kwargs.get('random_seed', None)
        self.rng = random.RandomState(self.random_seed)
        
        # Friction coefficient (common for stochastic thermostats)
        self.friction_coefficient = kwargs.get('friction_coefficient', 1.0)
    
    def generate_random_velocities(self, shape, temperature, mass):
        """
        Generate random velocities from Maxwell-Boltzmann distribution.
        
        Parameters
        ----------
        shape : tuple
            Shape of velocity array
        temperature : float
            Temperature for velocity distribution
        mass : float
            Particle mass
            
        Returns
        -------
        numpy.ndarray
            Random velocities
        """
        # Maxwell-Boltzmann: v ~ N(0, k_B*T/m)
        # Assuming k_B = 1 (reduced units)
        sigma = sqrt(temperature / mass)
        return self.rng.normal(0.0, sigma, shape)


class DeterministicThermostatBase(ThermostatBase):
    """
    Base class for deterministic thermostats.
    
    Provides common functionality for thermostats that use deterministic
    equations of motion (Nosé-Hoover, Gaussian, etc.).
    """
    
    def __init__(self):
        super().__init__()
        self.is_deterministic = True
        self.conserves_energy = False  # Extended system conserves different quantity
        
        # Extended system variables
        self.extended_variables = None
        self.num_extended_variables = 0
        
    def setup(self, params, **kwargs):
        super().setup(params, **kwargs)
        
        # Initialize extended system variables
        self._initialize_extended_system(**kwargs)
    
    def _initialize_extended_system(self, **kwargs):
        """Initialize extended system variables (override in subclasses)."""
        pass
    
    @abstractmethod
    def update_extended_variables(self, ptcls):
        """
        Update extended system variables.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
        """
        pass


class GlobalThermostatBase(ThermostatBase):
    """
    Base class for global temperature control methods.
    
    These methods control the total temperature of the system without
    maintaining detailed balance or canonical sampling properties.
    """
    
    def __init__(self):
        super().__init__()
        self.supports_anisotropic = True
        self.simple_rescaling = True