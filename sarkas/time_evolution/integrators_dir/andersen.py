"""
Andersen thermostat integrator implementation.

The Andersen thermostat combines deterministic integration (usually Velocity Verlet)
with stochastic velocity reassignment at random intervals. This provides temperature
control and proper canonical ensemble sampling.

The algorithm performs normal Velocity Verlet integration but at each timestep,
each particle has a probability ν*dt of having its velocity reassigned from a
Maxwell-Boltzmann distribution at the target temperature.

References
----------
Andersen, H. C. (1980). Molecular dynamics simulations at constant pressure and/or temperature.
The Journal of Chemical Physics, 72(4), 2384-2393.
"""

from numpy import sqrt, zeros, exp, log
from numba import jit
from .stochastic_base import StochasticThermostatIntegratorBase


class Andersen(StochasticThermostatIntegratorBase):
    """
    Andersen thermostat integrator.
    
    This integrator combines Velocity Verlet integration with stochastic velocity
    reassignment. At each timestep, particles have a probability ν*dt of having
    their velocities drawn from a Maxwell-Boltzmann distribution.
    
    The collision frequency ν controls the strength of coupling to the heat bath:
    - Small ν: weak coupling, slow equilibration
    - Large ν: strong coupling, fast equilibration but more perturbation to dynamics
    
    Attributes
    ----------
    collision_frequency : float or numpy.ndarray
        Collision frequency ν for each species (inverse time units)
    collision_probability : numpy.ndarray
        Probability of collision per timestep for each species
    """
    
    _aliases = ['andersen_thermostat', 'andersen_integrator']
    
    def __init__(self):
        super().__init__()
        self.name = 'andersen'
        self.order = 2
        self.supports_canonical_sampling = True
        self.supports_constraints = True
        
        # Andersen-specific parameters
        self.collision_frequency = None
        self.collision_probability = None
        
    def _setup_stochastic_parameters(self, params, **kwargs):
        """Setup Andersen-specific parameters."""
        # Get collision frequency
        self.collision_frequency = (getattr(params, 'andersen_collision_frequency', None) or
                                   getattr(params, 'collision_frequency', None) or
                                   kwargs.get('collision_frequency') or
                                   kwargs.get('nu'))
        
        if self.collision_frequency is None:
            # Default: collision frequency = 1/(10*dt) for moderate coupling
            self.collision_frequency = 0.1 / self.dt
        
        # Handle scalar or array collision frequency
        if hasattr(self.collision_frequency, '__len__'):
            if len(self.collision_frequency) != len(self.species_num):
                raise ValueError(f"Number of collision frequencies ({len(self.collision_frequency)}) "
                               f"must match number of species ({len(self.species_num)})")
        else:
            # Scalar frequency - use same for all species
            freq_value = self.collision_frequency
            self.collision_frequency = zeros(len(self.species_num))
            self.collision_frequency.fill(freq_value)
        
        # Calculate collision probabilities per timestep
        self.collision_probability = 1.0 - exp(-self.collision_frequency * self.dt)
        
        # Validate parameters
        self.validate_andersen_parameters()
    
    def update(self, ptcls):
        """
        Update particles using Andersen thermostat integration.
        
        Performs Velocity Verlet integration followed by stochastic
        velocity reassignment based on collision probabilities.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        # Standard Velocity Verlet integration
        self._velocity_verlet_step(ptcls)
        
        # Apply stochastic thermostat
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
    
    def apply_stochastic_thermostat(self, ptcls):
        """
        Apply Andersen stochastic thermostat to particle velocities.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
        """
        # Generate random numbers for collision decisions and new velocities
        collision_randoms = self.rng.random(self.total_num_ptcls)
        velocity_randoms = self.generate_random_forces()
        
        # Apply thermostat using numba-accelerated function
        self._andersen_thermostat(
            ptcls.vel,
            velocity_randoms,
            collision_randoms,
            self.collision_probability,
            self.target_temperatures,
            self.species_masses,
            self.species_num,
            self.dimensions,
            self.kB
        )
    
    @staticmethod
    @jit(nopython=True)
    def _andersen_thermostat(vel, vel_random, collision_random, collision_prob, 
                           temperatures, masses, species_num, dimensions, kB):
        """
        Numba-accelerated Andersen thermostat application.
        
        Parameters
        ----------
        vel : numpy.ndarray
            Particle velocities (modified in-place)
        vel_random : numpy.ndarray
            Random numbers for new velocities
        collision_random : numpy.ndarray
            Random numbers for collision decisions
        collision_prob : numpy.ndarray
            Collision probability for each species
        temperatures : numpy.ndarray
            Target temperature for each species
        masses : numpy.ndarray
            Mass of each species
        species_num : numpy.ndarray
            Number of particles in each species
        dimensions : int
            Number of spatial dimensions
        kB : float
            Boltzmann constant
        """
        particle_idx = 0
        
        for species_idx, num_particles in enumerate(species_num):
            if num_particles == 0:
                continue
                
            # Get species parameters
            prob = collision_prob[species_idx]
            temp = temperatures[species_idx]
            mass = masses[species_idx]
            
            # Maxwell-Boltzmann velocity scale: sqrt(kT/m)
            vel_scale = sqrt(kB * temp / mass)
            
            # Check each particle in this species
            for i in range(num_particles):
                # Collision occurs with probability prob
                if collision_random[particle_idx] < prob:
                    # Reassign velocity from Maxwell-Boltzmann distribution
                    for d in range(dimensions):
                        vel[particle_idx, d] = vel_scale * vel_random[particle_idx, d]
                
                particle_idx += 1
    
    def validate_andersen_parameters(self):
        """Validate Andersen thermostat parameters."""
        # Base class validates temperatures
        super().validate_stochastic_parameters()
        
        # Check collision frequencies
        if (self.collision_frequency <= 0).any():
            raise ValueError("All collision frequencies must be positive")
        
        # Check for reasonable collision probabilities
        if (self.collision_probability >= 1.0).any():
            print("Warning: Some collision probabilities >= 1.0, "
                  "consider reducing collision frequency or timestep")
        
        # Warn about very high collision frequencies
        max_freq = self.collision_frequency.max()
        if max_freq * self.dt > 0.5:
            print(f"Warning: High collision frequency ν*dt = {max_freq * self.dt:.3f}, "
                  f"dynamics may be overly perturbed")
    
    def validate_timestep(self, params):
        """Validate timestep for Andersen integrator."""
        super().validate_timestep(params)
        
        # Additional Andersen-specific checks
        if hasattr(self, 'collision_frequency') and self.collision_frequency is not None:
            max_freq = self.collision_frequency.max()
            # For reasonable dynamics, ν*dt should be << 1
            reasonable_dt = 0.1 / max_freq
            if self.dt > reasonable_dt:
                print(f"Warning: Large timestep for Andersen integrator dt={self.dt:.2e}, "
                      f"recommended dt < {reasonable_dt:.2e} for ν={max_freq:.2e}")
    
    def get_info(self):
        """Return Andersen integrator information."""
        info = super().get_info()
        info.update({
            'collision_frequency': self.collision_frequency.tolist() if self.collision_frequency is not None else None,
            'collision_probability': self.collision_probability.tolist() if self.collision_probability is not None else None,
        })
        return info
    
    def pretty_print(self):
        """Print Andersen integrator information."""
        super().pretty_print()
        if self.collision_frequency is not None:
            if len(self.collision_frequency) == 1:
                print(f"Collision frequency (ν): {self.collision_frequency[0]:.6f}")
                print(f"Collision probability: {self.collision_probability[0]:.6f}")
            else:
                print(f"Collision frequencies (ν): {self.collision_frequency}")
                print(f"Collision probabilities: {self.collision_probability}")


# Alias for backward compatibility
class AndersenThermostat(Andersen):
    """Alias for Andersen integrator."""
    
    _aliases = ['andersen_velocity_reassignment']
    
    def __init__(self):
        super().__init__()
        self.name = 'andersen_thermostat'