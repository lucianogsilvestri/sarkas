"""
Base classes for stochastic integrators.

This module provides base classes for all integrators that include
stochastic forces, random number generation, and temperature control.
"""

from abc import abstractmethod
from numpy import sqrt, zeros, random, format_float_scientific
from .base import IntegratorBase


class StochasticIntegratorBase(IntegratorBase):
    """
    Base class for stochastic integrators.
    
    Provides common functionality for integrators that use stochastic forces,
    including random number generation, noise amplitude calculations, and
    fluctuation-dissipation theorem enforcement.
    
    Attributes
    ----------
    target_temperatures : numpy.ndarray
        Target temperature for each species
    kB : float
        Boltzmann constant
    random_seed : int
        Seed for random number generator
    rng : numpy.random.RandomState
        Random number generator instance
    friction_coefficients : numpy.ndarray
        Friction coefficients for each species
    noise_amplitudes : numpy.ndarray
        Noise amplitudes for each species
    supports_canonical_sampling : bool
        Whether integrator provides proper canonical ensemble sampling
    """
    
    def __init__(self):
        super().__init__()
        
        # Stochastic integrator properties
        self.is_stochastic = True
        self.supports_canonical_sampling = False
        self.conserves_energy = False  # Stochastic integrators don't conserve energy
        
        # Temperature control
        self.target_temperatures = None
        self.kB = 1.0  # Boltzmann constant (reduced units)
        self.thermalization = True  # Always provides temperature control
        
        # Random number generation
        self.random_seed = None
        self.rng = None
        
        # Stochastic parameters (common across many methods)
        self.friction_coefficients = None
        self.noise_amplitudes = None
        
        # Performance optimization
        self._sqrt_dt = None
        self._dt_15 = None  # dt^1.5 for some algorithms
        
    def setup(self, params, **kwargs):
        """Setup stochastic integrator with common parameters."""
        super().setup(params, **kwargs)
        
        # Setup random number generation
        self.random_seed = kwargs.get('random_seed', None)
        self.rng = random.RandomState(self.random_seed)
        
        # Get target temperatures
        self.target_temperatures = self._setup_temperatures(params, **kwargs)
        
        # Boltzmann constant
        self.kB = (getattr( params, 'kB', 1.0) or kwargs.get('kB', 1.0) )
        
        # Pre-compute time-related constants
        self._sqrt_dt = sqrt(self.dt)
        self._dt_15 = self.dt ** 1.5
        
        # Setup stochastic parameters (override in subclasses)
        self._setup_stochastic_parameters(params, **kwargs)
        
    def _setup_temperatures(self, params, **kwargs):
        """Setup target temperatures for each species."""
        temperatures = (getattr(params, 'thermostat_temperatures', None) or
                       getattr(params, 'species_temperature', None) or
                       kwargs.get('species_temperatures') or
                       kwargs.get('target_temperatures'))
        
        if temperatures is None:
            raise ValueError("Target temperature(s) must be specified for stochastic integrator")
        
        # Handle scalar or array temperatures
        if hasattr(temperatures, '__len__'):
            if len(temperatures) != len(self.species_num):
                raise ValueError(f"Number of temperatures ({len(temperatures)}) "
                               f"must match number of species ({len(self.species_num)})")
            return temperatures
        else:
            # Scalar temperature - use same for all species
            temp_array = zeros(len(self.species_num))
            temp_array.fill(temperatures)
            return temp_array
    
    def _setup_stochastic_parameters(self, params, **kwargs):
        """Setup stochastic-specific parameters (override in subclasses)."""
        # Default implementation - subclasses should override
        pass
    
    def generate_random_forces(self, shape=None):
        """
        Generate random forces/numbers for stochastic integration.
        
        Parameters
        ----------
        shape : tuple, optional
            Shape of random array. If None, uses (total_num_ptcls, dimensions)
            
        Returns
        -------
        numpy.ndarray
            Random numbers from standard normal distribution
        """
        if shape is None:
            shape = (self.total_num_ptcls, self.dimensions)
        
        return self.rng.normal(0.0, 1.0, shape)
    
    def calculate_noise_amplitude(self, temperature, mass, friction=None):
        """
        Calculate noise amplitude from fluctuation-dissipation theorem.
        
        For Langevin-type dynamics: σ = sqrt(2γkT/m)
        
        Parameters
        ----------
        temperature : float
            Temperature
        mass : float
            Particle mass
        friction : float, optional
            Friction coefficient (if applicable)
            
        Returns
        -------
        float
            Noise amplitude
        """
        if friction is not None:
            # Standard Langevin: σ = sqrt(2γkT/m)
            return sqrt(2.0 * friction * self.kB * temperature / mass)
        else:
            # Generic thermal noise: σ = sqrt(kT/m)
            return sqrt(self.kB * temperature / mass)
    
    def validate_stochastic_parameters(self):
        """Validate stochastic integrator parameters."""
        # Check temperatures
        if (self.target_temperatures <= 0).any():
            raise ValueError("All target temperatures must be positive")
        
        # Check friction coefficients if present
        if self.friction_coefficients is not None:
            if (self.friction_coefficients <= 0).any():
                raise ValueError("All friction coefficients must be positive")
    
    def get_ensemble_properties(self):
        """
        Get information about ensemble sampling properties.
        
        Returns
        -------
        dict
            Dictionary with ensemble information
        """
        return {
            'ensemble': 'NVT' if self.supports_canonical_sampling else 'Non-equilibrium',
            'canonical_sampling': self.supports_canonical_sampling,
            'temperature_control': True,
            'energy_conservation': False,
            'momentum_conservation': True,  # Usually true unless stated otherwise
        }
    
    def get_info(self):
        """Return stochastic integrator information."""
        info = super().get_info()
        info.update({
            'is_stochastic': True,
            'supports_canonical_sampling': self.supports_canonical_sampling,
            'target_temperatures': [format_float_scientific(s, precision = 6) for s in self.target_temperatures] if self.target_temperatures is not None else None,
            'random_seed': self.random_seed,
            'kB': self.kB,
            'friction_coefficients': [format_float_scientific(f, precision=6) for f in self.friction_coefficients] if self.friction_coefficients is not None else None,
            'noise_amplitudes': [format_float_scientific(s, precision=6) for s in self.noise_amplitudes] if self.noise_amplitudes is not None else None,
        })
        info.update(self.get_ensemble_properties())
        return info

class UnderdampedLangevinBase(StochasticIntegratorBase):
    """
    Base class for underdamped Langevin integrators.
    
    These integrators solve the full Langevin equation including
    inertial effects:
        m dv/dt = F - γmv + sqrt(2γmkT)η(t)
        dr/dt = v
    """
    
    def __init__(self):
        super().__init__()
        self.order = 2  # Second-order in time  
        self.supports_canonical_sampling = True
        self.has_velocities = True
    
    def get_info(self):
        """Return underdamped Langevin integrator information."""
        info = super().get_info()
        info.update({
            'order': self.order,
            'has_velocities': self.has_velocities,
        })
        return info


class StochasticThermostatIntegratorBase(StochasticIntegratorBase):
    """
    Base class for integrators that combine deterministic integration
    with stochastic thermostatting (like Andersen, stochastic velocity rescaling).
    
    These methods use standard deterministic integration but apply
    stochastic temperature control at regular intervals.
    """
    
    def __init__(self):
        super().__init__()
        self.thermostat_frequency = 1  # How often to apply stochastic thermostat
        self.deterministic_integration = True  # Uses deterministic base integration
    
    def get_info(self):
        """Return stochastic thermostat integrator information."""
        info = super().get_info()
        info.update({
            'thermostat_frequency': self.thermostat_frequency,
            'deterministic_integration': self.deterministic_integration,
        })
        return info
    
    @abstractmethod
    def apply_stochastic_thermostat(self, ptcls):
        """Apply stochastic thermostatting to particles."""
        pass