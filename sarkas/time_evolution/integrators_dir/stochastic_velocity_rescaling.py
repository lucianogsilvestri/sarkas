"""
Stochastic Velocity Rescaling (Bussi-Donadio-Parrinello) thermostat integrator.

The stochastic velocity rescaling thermostat provides proper canonical ensemble 
sampling while maintaining good dynamical properties. It's an improvement over
the Berendsen thermostat that adds the correct amount of stochastic noise to
ensure canonical sampling.

The algorithm rescales kinetic energy according to a stochastic differential equation
that produces the correct canonical distribution of kinetic energies.

References
----------
Bussi, G., Donadio, D., & Parrinello, M. (2007). Canonical sampling through velocity rescaling.
The Journal of Chemical Physics, 126(1), 014101.
"""

from numpy import sqrt, zeros, random as np_random, sum as np_sum
from numba import jit
from .stochastic_base import StochasticThermostatIntegratorBase


class StochasticVelocityRescaling(StochasticThermostatIntegratorBase):
    """
    Stochastic Velocity Rescaling (Bussi-Donadio-Parrinello) thermostat integrator.
    
    This integrator combines Velocity Verlet integration with stochastic velocity
    rescaling that produces proper canonical ensemble sampling. The method rescales
    the kinetic energy according to a stochastic process that maintains the correct
    canonical distribution.
    
    The scaling factor is determined by:
        dK/dt = (K₀ - K)/τ + 2√(KK₀/Nf)/τ dW
    
    Where K is kinetic energy, K₀ is target kinetic energy, τ is coupling time,
    Nf is degrees of freedom, and dW is white noise.
    
    Attributes
    ----------
    coupling_time : float or numpy.ndarray
        Coupling time τ for each species
    apply_every_step : bool
        Whether to apply thermostat every timestep
    thermostat_frequency : int
        How often to apply thermostat (in timesteps)
    """
    
    _aliases = ['bussi', 'bussi_thermostat', 'csvr', 'stochastic_rescaling']
    
    def __init__(self):
        super().__init__()
        self.name = 'stochastic_velocity_rescaling'
        self.order = 2
        self.supports_canonical_sampling = True
        self.supports_constraints = True
        
        # SVR-specific parameters
        self.coupling_time = None
        self.apply_every_step = True
        
    def _setup_stochastic_parameters(self, params, **kwargs):
        """Setup stochastic velocity rescaling parameters."""
        # Get coupling time
        self.coupling_time = (getattr(params, 'svr_coupling_time', None) or
                             getattr(params, 'coupling_time', None) or
                             kwargs.get('coupling_time') or
                             kwargs.get('tau'))
        
        if self.coupling_time is None:
            # Default: coupling time = 10*dt for moderate coupling
            self.coupling_time = 10.0 * self.dt
        
        # Handle scalar or array coupling time
        if hasattr(self.coupling_time, '__len__'):
            if len(self.coupling_time) != len(self.species_num):
                raise ValueError(f"Number of coupling times ({len(self.coupling_time)}) "
                               f"must match number of species ({len(self.species_num)})")
        else:
            # Scalar coupling time - use same for all species
            tau_value = self.coupling_time
            self.coupling_time = zeros(len(self.species_num))
            self.coupling_time.fill(tau_value)
        
        # Setup thermostat frequency
        self.apply_every_step = kwargs.get('apply_every_step', True)
        if not self.apply_every_step:
            self.thermostat_frequency = kwargs.get('thermostat_frequency', 1)
        
        # Validate parameters
        self.validate_svr_parameters()
    
    def update(self, ptcls):
        """
        Update particles using stochastic velocity rescaling integration.
        
        Performs Velocity Verlet integration followed by stochastic
        velocity rescaling for temperature control.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        # Standard Velocity Verlet integration
        self._velocity_verlet_step(ptcls)
        
        # Apply stochastic thermostat
        if self._should_apply_thermostat():
            self.apply_stochastic_thermostat(ptcls)
    
    def _velocity_verlet_step(self, ptcls):
        """Perform standard Velocity Verlet integration step."""
        # First half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        
        # Full step position update
        ptcls.pos += ptcls.vel * self.dt
        
        # Apply boundary conditions
        if self._enforce_bc is not None:
            self._enforce_bc(ptcls)
        
        # Update accelerations at new positions
        if self._update_accel is not None:
            self._update_accel(ptcls)
        
        # Second half step velocity update
        ptcls.vel += 0.5 * ptcls.acc * self.dt
    
    def _should_apply_thermostat(self):
        """Determine if thermostat should be applied at current timestep."""
        if self.apply_every_step:
            return True
        else:
            # Apply according to frequency (would need timestep counter)
            return True  # Simplified for now
    
    def apply_stochastic_thermostat(self, ptcls):
        """
        Apply stochastic velocity rescaling thermostat.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
        """
        # Calculate current kinetic energies per species
        current_kinetic_energies, current_temperatures = ptcls.calculate_species_kinetic_temperature()
        
        # Generate random numbers for stochastic process
        random_numbers = self.rng.normal(0.0, 1.0, len(self.species_num))
        
        # Apply stochastic velocity rescaling
        self._stochastic_rescaling(
            ptcls.vel,
            current_kinetic_energies,
            self.target_temperatures,
            self.coupling_time,
            self.species_num,
            self.species_masses,
            self.dimensions,
            self.dt,
            self.kB,
            random_numbers
        )
    

    
    @staticmethod
    @jit(nopython=True)
    def _stochastic_rescaling(vel, current_ke, target_temps, coupling_times, 
                            species_num, masses, dimensions, dt, kB, randoms):
        """
        Numba-accelerated stochastic velocity rescaling.
        
        Parameters
        ----------
        vel : numpy.ndarray
            Particle velocities (modified in-place)
        current_ke : numpy.ndarray
            Current kinetic energy for each species
        target_temps : numpy.ndarray
            Target temperatures for each species
        coupling_times : numpy.ndarray
            Coupling times for each species
        species_num : numpy.ndarray
            Number of particles in each species
        masses : numpy.ndarray
            Masses of each species
        dimensions : int
            Number of spatial dimensions
        dt : float
            Timestep
        kB : float
            Boltzmann constant
        randoms : numpy.ndarray
            Random numbers for stochastic process
        """
        species_start = 0
        
        for i, num_particles in enumerate(species_num):

            species_end = species_start + num_particles
            
            # Calculate degrees of freedom for this species
            nf = num_particles * dimensions
            
            # Target kinetic energy: K₀ = (1/2) * Nf * kB * T
            target_ke = 0.5 * nf * kB * target_temps[i]
            
            if current_ke[i] <= 0 or target_ke <= 0:
                species_start = species_end
                continue
            
            # Coupling factor
            tau = coupling_times[i]
            c = dt / tau
            
            # Stochastic velocity rescaling factor
            # Based on: dK/dt = (K₀ - K)/τ + 2√(KK₀/Nf)/τ dW
            
            # Deterministic part: exp(-dt/τ)
            deterministic_factor = 1.0 - c
            
            # Stochastic part: 2√(KK₀/Nf)/τ dt dW
            
            # For Nf > 1, use the full stochastic expression
            stochastic_factor = c * sqrt(2.0 * current_ke[i] * target_ke / nf) * randoms[i]
            
            # New kinetic energy
            new_ke = deterministic_factor * current_ke[i] + target_ke * c + stochastic_factor
            
            # Ensure positive kinetic energy
            if new_ke <= 0:
                new_ke = target_ke * 1e-6  # Small positive value
        
            # Calculate scaling factor
            scaling_factor = sqrt(new_ke / current_ke[i]) * (current_ke[i] > 0) + (current_ke[i] <= 0) * 1.0
            
            # Apply scaling to velocities
            vel[species_start:species_end] *= scaling_factor
            
            species_start = species_end
    
    def validate_svr_parameters(self):
        """Validate stochastic velocity rescaling parameters."""
        # Base class validates temperatures
        super().validate_stochastic_parameters()
        
        # Check coupling times
        if (self.coupling_time <= 0).any():
            raise ValueError("All coupling times must be positive")
        
        # Check for very small coupling times
        min_tau = self.coupling_time.min()
        if min_tau < 2 * self.dt:
            print(f"Warning: Very small coupling time τ={min_tau:.2e}, "
                  f"recommended τ >= {2*self.dt:.2e} (2*dt)")
        
        # Check for very large coupling times
        max_tau = self.coupling_time.max()
        if max_tau > 1000 * self.dt:
            print(f"Warning: Very large coupling time τ={max_tau:.2e}, "
                  f"thermostat may be ineffective")
    
    def validate_timestep(self, params):
        """Validate timestep for SVR integrator."""
        super().validate_timestep(params)
        
        # Additional SVR-specific checks
        if hasattr(self, 'coupling_time') and self.coupling_time is not None:
            min_tau = self.coupling_time.min()
            # For stability, dt should be much smaller than tau
            stable_dt = 0.1 * min_tau
            if self.dt > stable_dt:
                print(f"Warning: Large timestep for SVR integrator dt={self.dt:.2e}, "
                      f"recommended dt < {stable_dt:.2e} for τ={min_tau:.2e}")
    
    def get_info(self):
        """Return stochastic velocity rescaling integrator information."""
        info = super().get_info()
        info.update({
            'coupling_time': self.coupling_time.tolist() if self.coupling_time is not None else None,
            'apply_every_step': self.apply_every_step,
            'thermostat_frequency': getattr(self, 'thermostat_frequency', 1),
        })
        return info
    
    def pretty_print(self):
        """Print stochastic velocity rescaling integrator information."""
        super().pretty_print()
        if self.coupling_time is not None:
            if len(self.coupling_time) == 1:
                print(f"Coupling time (τ): {self.coupling_time[0]:.6f}")
            else:
                print(f"Coupling times (τ): {self.coupling_time}")
        
        print(f"Apply every step: {self.apply_every_step}")
        if not self.apply_every_step:
            print(f"Thermostat frequency: {getattr(self, 'thermostat_frequency', 1)}")


# Aliases for different names commonly used
class Bussi(StochasticVelocityRescaling):
    """Alias for Stochastic Velocity Rescaling (Bussi method)."""
    
    _aliases = ['bussi_donadio_parrinello']
    
    def __init__(self):
        super().__init__()
        self.name = 'bussi'


class CSVR(StochasticVelocityRescaling):
    """Alias for Canonical Sampling through Velocity Rescaling."""
    
    _aliases = ['canonical_velocity_rescaling']
    
    def __init__(self):
        super().__init__()
        self.name = 'csvr'