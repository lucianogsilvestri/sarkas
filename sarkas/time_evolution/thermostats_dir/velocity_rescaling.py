"""
Velocity rescaling thermostat implementation.

The velocity rescaling thermostat is the simplest possible temperature control
method that directly scales all velocities to achieve the target temperature
at every timestep. While not physically realistic, it provides perfect
temperature control for equilibration and testing purposes.
"""

from numpy import sqrt, zeros, asarray
from numba import jit
from .base import GlobalThermostatBase


class VelocityRescaling(GlobalThermostatBase):
    """
    Direct velocity rescaling thermostat for immediate temperature control.
    
    This thermostat rescales particle velocities at every timestep to exactly
    match the target temperature. It provides perfect temperature control but
    does not produce realistic dynamics or proper statistical sampling.
    
    The scaling factor is simply:
        λ = sqrt(T_target / T_current)
    
    This method is useful for:
    - System equilibration
    - Testing and validation
    - Cases where perfect temperature control is more important than dynamics
    
    Attributes
    ----------
    thermalization_timestep : int
        Timestep at which to begin thermostatting
    target_temperatures : numpy.ndarray
        Target temperature for each species
    per_species : bool
        Whether to apply thermostat per species or globally
    apply_every_step : bool
        Whether to apply rescaling every timestep
    rescaling_frequency : int
        How often to apply rescaling (in timesteps)
    """
    
    _aliases = ['velocity_rescaling', 'direct_rescaling', 'perfect_scaling']
    
    def __init__(self):
        super().__init__()
        self.name = 'velocity_rescaling'
        self.supports_anisotropic = True
        self.supports_local = False
        self.conserves_momentum = True
        self.is_deterministic = True
        self.simple_rescaling = True
        self.target_temperatures = None
 
        # Internal state
        self.current_timestep = 0
        
    def setup(self, params, **kwargs):
        """
        Setup the velocity rescaling thermostat.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        **kwargs : dict
            Additional thermostat parameters:
            - thermalization_timestep : int, when to start thermostatting
            - per_species : bool, whether to thermostat per species
            - target_temperatures : array-like, target temp per species
            - apply_every_step : bool, whether to rescale every timestep
            - rescaling_frequency : int, rescaling frequency in timesteps
        """
        super().setup(params, **kwargs)
        
        # Initialize timestep counter
        self.current_timestep = 0
        
    def update(self, ptcls):
        """
        Apply velocity rescaling to particle velocities.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data (velocities modified in-place)
        """
       
        _, current_temperatures = ptcls.calculate_species_kinetic_temperature()
        self.instantaneous_temperatures = current_temperatures
        self._rescale_per_species(
            ptcls.vel,
            self.target_temperatures,
            current_temperatures,
            ptcls.species_num
            )

        self.current_timestep += 1
    
    @staticmethod
    @jit(nopython=True)
    def _rescale_per_species(vel, T_desired, T_current, species_np):
        """
        Numba'd function to rescale velocities per species.
        
        Parameters
        ----------
        vel : numpy.ndarray
            Particles' velocities to rescale. Shape: (dimensions, total_particles)
        T_desired : numpy.ndarray
            Target temperature of each species
        T_current : numpy.ndarray
            Current temperature of each species
        species_np : numpy.ndarray
            Number of each species
        """
        fact = sqrt(T_desired / T_current)
        species_start = 0
        species_end = 0

        for i, num in enumerate(species_np):
            species_end += num
            vel[:, species_start:species_end] *= fact[i]
            species_start += num
    
    def get_info(self):
        """
        Return velocity rescaling thermostat information.
        
        Returns
        -------
        dict
            Dictionary containing thermostat properties
        """
        info = super().get_info()
        info.update({
            'target_temperatures': self.target_temperatures.tolist() if self.target_temperatures is not None else None,
            'current_timestep': self.current_timestep,
        })
        return info