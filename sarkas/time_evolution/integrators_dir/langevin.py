"""
Langevin integrator for molecular dynamics simulations.

The Langevin integrator combines time integration with stochastic thermostatting
by adding friction and random forces to the equations of motion. This provides
both temperature control and proper canonical (NVT) ensemble sampling.

The Langevin equation of motion is:
    m * dv/dt = F - γ * m * v + R(t)

Where:
    - F is the conservative force
    - γ is the friction coefficient  
    - R(t) is a random force with <R(t)> = 0 and <R(t)R(t')> = 2γmkT δ(t-t')

References
----------
Leimkuhler, B., & Matthews, C. (2015). Molecular dynamics: with deterministic and 
stochastic numerical methods. Springer.
"""

from numpy import sqrt, zeros, copy, repeat
from .stochastic_base import UnderdampedLangevinBase


class Langevin(UnderdampedLangevinBase):
    """
    Langevin integrator with built-in thermostatting.
    
    This integrator implements the Langevin equation of motion which includes
    both deterministic forces and stochastic friction/random forces. It provides
    temperature control and proper canonical ensemble sampling without requiring
    a separate thermostat.
    
    The algorithm used is a second-order accurate integrator that properly
    handles the coupling between positions, velocities, and random forces.
    
    Attributes
    ----------
    langevin_gamma : float or numpy.ndarray
        Friction coefficient(s) for each species
    thermostat_temperatures : float or numpy.ndarray
        Target temperature(s) for each species
    kB : float
        Boltzmann constant (typically 1.0 in reduced units)
    sigma : numpy.ndarray
        Noise amplitude for each species
    c1 : float
        Integration coefficient 1
    c2 : float
        Integration coefficient 2
    """
    
    _aliases = ['langevin_integrator', 'stochastic_integrator']
    
    def __init__(self):
        super().__init__()
        self.name = 'langevin'
        self.order = 2
        self.supports_magnetic_field = False
        self.supports_adaptive_timestep = False
        self.supports_constraints = True
        
        # Langevin-specific parameters
        self.langevin_gamma = None
        
        # Computed coefficients
        self.sigma = None
        self.c1 = None
        self.c2 = None
        
        # Internal state
        self.thermalization = True  # Langevin always provides thermalization
        
    def setup(self, params, **kwargs):
        """
        Setup the Langevin integrator.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        **kwargs : dict
            Additional integrator parameters:
            - langevin_gamma : float or array, friction coefficient(s)
            - thermostat_temperatures : float or array, target temperature(s)
            - kB : float, Boltzmann constant
        """
        super().setup(params, **kwargs)
        
    def _setup_stochastic_parameters(self, params, **kwargs):
        """Setup Langevin-specific stochastic parameters."""
        # Get Langevin parameters
        self.langevin_gamma = (getattr(params, 'langevin_gamma', None) or
                               kwargs.get('langevin_gamma') or
                               kwargs.get('langevin_gamma') or
                               kwargs.get('friction_coefficient'))
        
        if self.langevin_gamma is None:
            raise ValueError("Langevin friction coefficient (langevin_gamma) must be specified")
        
        # Handle scalar or array gamma
        if hasattr(self.langevin_gamma, '__len__'):
            if len(self.langevin_gamma) != len(self.species_num):
                raise ValueError(f"Number of gamma values ({len(self.langevin_gamma)}) "
                               f"must match number of species ({len(self.species_num)})")
        else:
            # Scalar gamma - use same for all species
            gamma_value = self.langevin_gamma
            self.langevin_gamma = zeros(len(self.species_num))
            self.langevin_gamma.fill(gamma_value)
        
        # Store friction coefficients for base class
        self.friction_coefficients = self.langevin_gamma
        
        # Compute Langevin coefficients
        self._compute_coefficients()
        
        # Validate parameters
        self.validate_langevin_parameters()
    
    def _compute_coefficients(self):
        """Compute Langevin integration coefficients."""
        # Noise amplitude: σ = sqrt(2γkT/m)
        self.sigma = sqrt(2.0 * self.langevin_gamma * self.kB * 
                         self.target_temperatures / self.species_masses)
        
        # Integration coefficients
        self.c1 = 1.0 - 0.5 * self.langevin_gamma * self.dt
        self.c2 = 1.0 / (1.0 + 0.5 * self.langevin_gamma * self.dt)

        # Create sigma array for all particles (vectorized)
        self.sigma_all = repeat(self.sigma, self.species_num)

        self.noise_amplitudes = self.sigma

    def update(self, ptcls):
        """
        Update particles using the Langevin integrator.
        
        This implements a second-order accurate Langevin integrator that
        properly handles the stochastic forces and provides temperature control.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        # Generate random numbers for all particles
        beta = self.generate_random_forces()
        
        # Position update: r(t+dt) = r(t) + c1*dt*v(t) + 0.5*dt^2*a(t) + 0.5*σ*dt^1.5*β
        ptcls.pos[:, :self.dimensions] += (
            self.c1 * self.dt * ptcls.vel[:, :self.dimensions]
            + 0.5 * self.dt**2 * ptcls.acc[:, :self.dimensions]
            + 0.5 * self.sigma_all[:, None] * self._dt_15 * beta[:, :self.dimensions]
        )
        
        # Apply boundary conditions
        self._enforce_bc(ptcls)
        
        # Store old accelerations
        acc_old = ptcls.acc.copy()
        
        # Update accelerations at new positions
        self._update_accel(ptcls)
        
        # Velocity update with stochastic term
        # v(t+dt) = c1*v(t) + 0.5*dt*(a(t) + a(t+dt)) + σ*sqrt(dt)*β
        ptcls.vel[:, :self.dimensions] = (
            self.c1 * self.c2 * ptcls.vel[:, :self.dimensions]
            + 0.5 * self.c2 * self.dt * 
            (ptcls.acc[:, :self.dimensions] + acc_old[:, :self.dimensions])
            + self.c2 * self.sigma_all[:, None] * sqrt(self.dt) * beta[:, :self.dimensions]
        )
    
    def validate_langevin_parameters(self):
        """
        Validate Langevin integrator parameters.
        """
        # Check gamma values
        if (self.langevin_gamma <= 0).any():
            raise ValueError("All Langevin friction coefficients must be positive")
        
        # Check temperatures
        # Base class validates temperatures
        super().validate_stochastic_parameters()
        
        # Check for stability - gamma should not be too large
        max_gamma = self.langevin_gamma.max()
        if max_gamma * self.dt > 2.0:
            print(f"Warning: Large damping parameter γ*dt = {max_gamma * self.dt:.3f}, "
                  f"recommended γ*dt < 2.0 for stability")
        
        # Check for overdamped regime
        min_gamma = self.langevin_gamma.min()
        if min_gamma * self.dt > 0.5:
            print(f"Warning: Strong damping γ*dt = {min_gamma * self.dt:.3f}, "
                  f"system may be in overdamped regime")
    
    def validate_timestep(self, params):
        """
        Validate timestep for Langevin integrator stability.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        """
        super().validate_timestep(params)
        
        # Additional Langevin-specific timestep checks
        if hasattr(self, 'langevin_gamma') and self.langevin_gamma is not None:
            max_gamma = self.langevin_gamma.max()
            # For stability, dt should satisfy γ*dt << 1
            stable_dt = 0.1 / max_gamma  # Conservative estimate
            if self.dt > stable_dt:
                print(f"Warning: Large timestep for Langevin integrator dt={self.dt:.2e}, "
                      f"recommended dt < {stable_dt:.2e} for γ={max_gamma:.2e}")
    
    def get_info(self):
        """
        Return Langevin integrator information.
        
        Returns
        -------
        dict
            Dictionary containing integrator properties
        """
        info = super().get_info()
        info.update({
            'langevin_gamma': self.langevin_gamma,
            'c1': self.c1,
            'c2': self.c2,
        })
        return info
    
    def pretty_print(self):
        """Print Langevin integrator information."""
        
        msg = super().pretty_print()
        msg += f"langevin_gamma: {[f'{g:.6e}' for g in self.langevin_gamma]}\n"
        msg += f"target_temperatures: {[f'{t:.6e}' for t in self.target_temperatures]}\n"
        msg += f"sigma: {[f'{s:.6e}' for s in self.sigma]}\n"
        msg += f"c1: {self.c1:.6f}, c2: {self.c2:.6f}\n"
        
        # N = -log(0.01) / (2.0 * self.langevin_gamma * self.dt)
        # Np = -log(0.01) / (2.0 * self.langevin_gamma * 2.0 * pi / wp_tot)
        # lang_msg = (
        #     f"langevin_gamma = {self.langevin_gamma:.4e} {self.units_dict['Hertz']}\n"
        #     f"langevin_gamma * dt = {self.langevin_gamma * self.dt:.2e}\n"
        #     f"Timestep to decay to 0.01: exp( - 2 gamma N dt) = 0.01 ==> N = {N:.2e}\n"
        #     f"langevin_gamma * (2 pi / w_p) = {self.langevin_gamma * (2.0 * pi/ wp_tot):.2e}\n"
        #     f"Plasma cycles to decay to 0.01: exp( - 2 gamma N_p dt) = 0.01 ==> N_p = {Np:.2e}\n"
        # )
        # msg += lang_msg
        
        return msg
    
# Alias for backward compatibility
class LangevinIntegrator(Langevin):
    """Alias for Langevin integrator."""
    
    _aliases = ['langevin_thermostat']
    
    def __init__(self):
        super().__init__()
        self.name = 'langevin_integrator'