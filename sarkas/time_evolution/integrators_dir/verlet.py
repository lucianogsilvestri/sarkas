"""
Verlet integrators for molecular dynamics simulations.

This module implements the classic Verlet algorithms:
- PositionVerlet: Original Verlet algorithm using positions
- VelocityVerlet: Velocity Verlet (leapfrog) algorithm

Both algorithms are second-order symplectic integrators that conserve energy
in Hamiltonian systems.
"""

from numpy import zeros
from .base import IntegratorBase


class PositionVerlet(IntegratorBase):
    """
    Position Verlet integrator (original Verlet algorithm).
    
    This is the original Verlet algorithm that uses positions at two previous
    time steps. It's equivalent to velocity Verlet but doesn't explicitly
    track velocities during the integration step.
    
    Algorithm:
        r(t+dt) = 2*r(t) - r(t-dt) + a(t)*dt^2
        v(t) = [r(t+dt) - r(t-dt)] / (2*dt)
    
    References
    ----------
    Verlet, L. (1967). Computer "experiments" on classical fluids. I. 
    Thermodynamical properties of Lennard-Jones molecules. 
    Physical Review, 159(1), 98-103.
    """
    
    _aliases = ['position_verlet', 'original_verlet']
    
    def __init__(self):
        super().__init__()
        self.name = 'position_verlet'
        self.order = 2
        self.supports_magnetic_field = False
        self.supports_adaptive_timestep = False
        self.supports_constraints = True
        
        # Storage for previous positions
        self.pos_old = None
        self.first_step = True
        
    def setup(self, params, **kwargs):
        """
        Setup the Position Verlet integrator.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        **kwargs : dict
            Additional integrator-specific parameters
        """
        super().setup(params, **kwargs)
        
        # Initialize storage for previous positions
        self.pos_old = zeros( (params.num_particles, 3), )
        self.first_step = True
        
    def update(self, ptcls):
        """
        Update particles using the Position Verlet algorithm.
        
        For the first timestep, uses velocity Verlet to bootstrap the algorithm.
        Subsequent steps use the classic position Verlet formula.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        if self.first_step:
            # Bootstrap with velocity Verlet for first step
            self._first_step_update(ptcls)
            self.first_step = False
        else:
            # Standard Position Verlet update
            self._standard_update(ptcls)
            
        # Apply boundary conditions
        if self._enforce_bc is not None:
            self._enforce_bc(ptcls)
            
        # Update accelerations for next step
        if self._update_accel is not None:
            self._update_accel(ptcls)
            
        # Apply thermostat if configured
        if self._apply_thermostat is not None:
            self._apply_thermostat(ptcls)
    
    def _first_step_update(self, ptcls):
        """
        First timestep using velocity Verlet to bootstrap.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
        """
        # Store current position as "old" position
        self.pos_old[:] = ptcls.pos
        
        # Update velocity (half step)
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        
        # Update position (full step)
        ptcls.pos += ptcls.vel * self.dt
        
        # Update velocity (second half step) - will be done after force calculation
        
    def _standard_update(self, ptcls):
        """
        Standard Position Verlet update for subsequent timesteps.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
        """
        # Store current position before updating
        pos_current = ptcls.pos.copy()
        
        # Position Verlet: r(t+dt) = 2*r(t) - r(t-dt) + a(t)*dt^2
        ptcls.pos = 2.0 * ptcls.pos - self.pos_old + ptcls.acc * (self.dt * self.dt)
        
        # Update velocity: v(t) = [r(t+dt) - r(t-dt)] / (2*dt)
        ptcls.vel = (ptcls.pos - self.pos_old) / (2.0 * self.dt)
        
        # Update old position for next step
        self.pos_old[:] = pos_current
        
    def validate_timestep(self, params):
        """
        Validate timestep for Position Verlet stability.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        """
        super().validate_timestep(params)
        
        # Additional Position Verlet specific checks
        if hasattr(params, 'max_force') and params.max_force > 0:
            # Estimate maximum safe timestep based on forces
            if hasattr(params, 'min_mass') and params.min_mass > 0:
                max_accel = params.max_force / params.min_mass
                safe_dt = 0.1 * (2.0 / max_accel) ** 0.5
                if self.dt > safe_dt:
                    print(f"Warning: Large timestep for Position Verlet dt={self.dt:.2e}, "
                          f"recommended dt < {safe_dt:.2e}")


class VelocityVerlet(IntegratorBase):
    """
    Velocity Verlet integrator (also known as leapfrog method).
    
    This is the most commonly used Verlet variant that explicitly tracks
    both positions and velocities. It's numerically identical to Position
    Verlet but more convenient for most applications.
    
    Algorithm:
        v(t+dt/2) = v(t) + a(t)*dt/2
        r(t+dt) = r(t) + v(t+dt/2)*dt
        # [compute forces/accelerations at new positions]
        v(t+dt) = v(t+dt/2) + a(t+dt)*dt/2
    
    References
    ----------
    Swope, W. C., Andersen, H. C., Berens, P. H., & Wilson, K. R. (1982). 
    A computer simulation method for the calculation of equilibrium constants 
    for the formation of physical clusters of molecules: Application to small 
    water clusters. The Journal of Chemical Physics, 76(1), 637-649.
    """
    
    _aliases = ['velocity_verlet', 'verlet', 'leapfrog']
    
    def __init__(self):
        super().__init__()
        self.name = 'velocity_verlet'
        self.order = 2
        self.supports_magnetic_field = False
        self.supports_adaptive_timestep = True
        self.supports_constraints = True
        
    def update(self, ptcls):
        """
        Update particles using the Velocity Verlet algorithm.
        
        This implements the standard velocity Verlet algorithm with the
        kick-drift-kick pattern.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        # First half step velocity update (kick)
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        
        # Full step position update (drift)
        ptcls.pos += ptcls.vel * self.dt
        
        # Apply boundary conditions
        self._enforce_bc(ptcls)
            
        # Update accelerations at new positions
        self._update_accel(ptcls)
            
        # Second half step velocity update (kick)
        ptcls.vel += 0.5 * ptcls.acc * self.dt
        
        # Apply thermostat if configured
        if self._apply_thermostat is not None:
            self._apply_thermostat(ptcls)
            
    def validate_timestep(self, params):
        """
        Validate timestep for Velocity Verlet stability.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        """
        super().validate_timestep(params)
        
        # Additional Velocity Verlet specific checks
        if hasattr(params, 'characteristic_frequency'):
            # For harmonic systems, dt should be much smaller than period
            char_freq = params.characteristic_frequency
            if char_freq > 0:
                max_dt = 0.1 * (2.0 * 3.14159 / char_freq)  # 10% of period
                if self.dt > max_dt:
                    print(f"Warning: Large timestep for oscillatory system dt={self.dt:.2e}, "
                          f"recommended dt < {max_dt:.2e}")


# Additional utility class for backward compatibility
class Verlet(VelocityVerlet):
    """
    Alias for VelocityVerlet for backward compatibility.
    
    This class simply inherits from VelocityVerlet and is provided
    for users who expect a generic "Verlet" class.
    """
    
    _aliases = ['verlet_default', 'standard_verlet']
    
    def __init__(self):
        super().__init__()
        self.name = 'verlet'