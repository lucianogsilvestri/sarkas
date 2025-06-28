"""
Berendsen thermostat implementation.

The Berendsen thermostat is a simple and effective velocity scaling thermostat
that weakly couples the system to a heat bath. It provides good temperature
control but does not produce a canonical ensemble.

References
----------
Berendsen, H. J. C., Postma, J. P. M., van Gunsteren, W. F., DiNola, A., & Haak, J. R. (1984).
Molecular dynamics with coupling to an external bath. The Journal of Chemical Physics, 81(8), 3684-3690.
"""

from numpy import sqrt
from numba import jit
from .base import GlobalThermostatBase


class Berendsen(GlobalThermostatBase):
    """
    Berendsen thermostat for temperature control.
    
    The Berendsen thermostat rescales particle velocities to drive the system
    temperature towards a target value. The coupling strength is controlled by
    a relaxation time parameter (tau).
    
    The scaling factor is given by:
        λ = sqrt(1 + (T_target/T_current - 1) * dt/tau)
    
    Where:
        - T_target is the desired temperature
        - T_current is the instantaneous temperature
        - dt is the timestep
        - tau is the relaxation time
    
    Attributes
    ----------
    tau : float
        Relaxation time parameter (coupling strength)
    thermalization_timestep : int
        Timestep at which to begin thermostatting
    target_temperatures : numpy.ndarray
        Target temperature for each species
    per_species : bool
        Whether to apply thermostat per species or globally
    """
    
    _aliases = ['berendsen_thermostat', 'velocity_scaling']
    
    def __init__(self):
        super().__init__()
        self.name = 'berendsen'
        self.supports_anisotropic = False
        self.supports_local = False
        self.conserves_momentum = True
        self.is_deterministic = True
        
        # Berendsen-specific parameters
        self.tau = None
        
        # Internal state
        self.current_timestep = 0
        
    def setup(self, params, **kwargs):
        """
        Setup the Berendsen thermostat.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        **kwargs : dict
            Additional thermostat parameters:
            - tau : float, relaxation time
            - thermalization_timestep : int, when to start thermostatting
            - per_species : bool, whether to thermostat per species
            - target_temperatures : array-like, target temp per species
        """
        super().setup(params, **kwargs)
        
        # Berendsen-specific parameters
        self.tau = kwargs.get('tau', self.relaxation_time)
        if self.tau is None:
            # Default tau to 10 timesteps
            self.tau = 10.0 * self.dt
                   
        # Convert tau to coupling factor (dt/tau)
        self._coupling_factor = self.dt / self.tau
        
        # Initialize timestep counter
        self.current_timestep = 0
        
    def update(self, ptcls):
        """
        Apply Berendsen thermostat to particle velocities.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data (velocities modified in-place)
        """
        _, current_temperatures = ptcls.calculate_species_kinetic_temperature()
        self.instantaneous_temperatures = current_temperatures
        # Apply Berendsen scaling using numba-accelerated function
        self._berendsen_update(
            ptcls.vel,
            self.target_temperatures,
            current_temperatures,
            ptcls.species_num,
            self._coupling_factor
        )
        # Increment timestep counter
        self.current_timestep += 1
    
    @staticmethod
    @jit(nopython=True)
    def _berendsen_update(vel, T_desired, T, species_np, tau):
        """
        Numba'd function to update particle velocity based on Berendsen thermostat.

        Parameters
        ----------
        vel : numpy.ndarray
            Particles' velocities to rescale. Shape: (dimensions, total_particles)
        T_desired : numpy.ndarray
            Target temperature of each species.
        T : numpy.ndarray
            Instantaneous temperature of each species.
        species_np : numpy.ndarray
            Number of each species.
        tau : float
            Coupling factor (dt/relaxation_time).

        """
        fact = sqrt(1.0 + (T_desired / T - 1.0) * tau)
        species_start = 0
        species_end = 0

        for i, num in enumerate(species_np):
            species_end += num
            vel[:, species_start:species_end] *= fact[i]
            species_start += num
    
    def validate_parameters(self):
        """
        Validate Berendsen thermostat parameters.
        """
        super().validate_parameters()
        
        if self.tau <= 0:
            raise ValueError("Berendsen relaxation time (tau) must be positive")
        
        if self.tau < 2 * self.dt:
            print(f"Warning: Very small relaxation time tau={self.tau:.2e}, "
                  f"recommended tau >= {2*self.dt:.2e} (2*dt)")
        
        if self.thermalization_timestep < 0:
            raise ValueError("Thermalization timestep must be non-negative")
        
        if self.target_temperatures is not None:
            if (self.target_temperatures <= 0).any():
                raise ValueError("All target temperatures must be positive")
    
    def get_info(self):
        """
        Return Berendsen thermostat information.
        
        Returns
        -------
        dict
            Dictionary containing thermostat properties
        """
        info = super().get_info()
        info.update({
            'tau': self.tau,
            'coupling_factor': self._coupling_factor,
            'target_temperatures': self.target_temperatures.tolist() if self.target_temperatures is not None else None,
            'current_timestep': self.current_timestep,
        })
        return info
    
    def pretty_print(self):
        """Print Berendsen thermostat information."""
        msg = super().pretty_print()
        
        msg += f"Relaxation time (tau): {self.tau:.6f}\n"
        msg += f"Coupling factor (dt/tau): {self._coupling_factor:.6f}\n"
        if self.target_temperatures is not None and len(self.target_temperatures) > 1:
            msg += f"Target temperatures: {T:.6e [for T in self.target_temperatures]}\n"

        return msg