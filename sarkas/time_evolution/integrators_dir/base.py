"""
Base classes for all time integrators.
"""

from abc import ABC, abstractmethod
from numpy import zeros, array, sqrt, pi


class IntegratorBase(ABC):
    """
    Abstract base class for all time integrators.
    
    All integrators must implement the update() method which handles:
    1. Force/acceleration computation (via self.update_accelerations)
    2. Time integration step
    3. Boundary condition enforcement
    4. Thermostat application (if configured)
    
    Attributes
    ----------
    name : str
        Integrator name (auto-generated from class name)
    dt : float
        Integration timestep
    order : int
        Integration order (1, 2, 4, etc.)
    supports_magnetic_field : bool
        Whether integrator can handle magnetic fields
    supports_adaptive_timestep : bool
        Whether integrator supports adaptive timestep control
    """
    
    def __init__(self):
        self.name = self.__class__.__name__.lower()
        self.dt = None
        self.order = 2  # Default integration order
        
        # Capability flags
        self.supports_magnetic_field = False
        self.supports_adaptive_timestep = False
        self.supports_constraints = False
        
        # System parameters
        self.box_lengths = None
        self.species_masses = None
        self.species_num = None
        self.total_num_ptcls = None
        self.dimensions = None
        
        # Force calculation method (linked from interaction algorithm)
        self.update_accelerations = None
        
        # Boundary conditions
        self.boundary_conditions = None
        self.enforce_bc = None
        
        # Thermostat (internal)
        self.thermostat = None
        self.thermalization = False
        self.thermalization_timestep = 0
        
        # Performance optimization: pre-bound methods
        self._update_accel = None
        self._enforce_bc = None
        self._apply_thermostat = None
        
    @abstractmethod
    def update(self, ptcls):
        """
        Advance particles by one timestep.
        
        This method must handle:
        1. Force calculation (via self.update_accelerations)
        2. Time integration
        3. Boundary conditions (via self.enforce_bc)
        4. Thermostat (via self.thermostat if configured)
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to update in-place
        """
        pass
    
    def setup(self, params, **kwargs):
        """
        Setup integrator with simulation parameters.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters containing dt, box_lengths, etc.
        **kwargs : dict
            Additional integrator-specific parameters
        """
        # Basic parameters
        self.dt = getattr(params, 'dt', None) or kwargs.get('dt')
        if self.dt is None:
            raise ValueError("Timestep (dt) must be specified")
            
        self.box_lengths = params.box_lengths
        self.species_masses = params.species_masses
        self.species_num = params.species_num
        self.total_num_ptcls = params.total_num_ptcls
        self.dimensions = params.dimensions
        
        # Setup boundary conditions
        self._setup_boundary_conditions(params)
        
        # Setup thermostat (internal to integrator)
        self._setup_thermostat(params, **kwargs)
        
        # Validate timestep for stability
        self.validate_timestep(params)
        
        # Performance optimization: pre-bind methods
        self._prebind_methods()
    
    def link_interaction_algorithm(self, algorithm):
        """
        Link the interaction algorithm for force calculations.
        
        Parameters
        ----------
        algorithm : InteractionSolverBase
            The interaction algorithm (PPPM, VerletList, etc.)
        """
        self.update_accelerations = algorithm.update
        self._update_accel = algorithm.update  # Pre-bound for performance
    
    def _setup_boundary_conditions(self, params):
        """Setup boundary condition enforcement method."""
        from ..boundary_conditions import get_boundary_condition
        
        # Get boundary condition type
        bc_type = getattr(params, 'boundary_conditions', 'periodic')
        self.boundary_conditions = bc_type.lower()
        
        # Get box lengths for BC initialization
        if not hasattr(self, 'box_lengths') or self.box_lengths is None:
            raise ValueError("Box lengths must be set before setting up boundary conditions")
        
        # Create boundary condition instance with required parameters
        bc_params = {'box_lengths': self.box_lengths}
        
        # Add timestep if the BC requires it (e.g., reflecting BC)
        if bc_type.lower() in ['reflecting', 'rbc', 'reflecting_bc', 'mirror', 'elastic']:
            if not hasattr(self, 'dt') or self.dt is None:
                raise ValueError("Timestep (dt) must be set before setting up reflecting boundary conditions")
            bc_params['dt'] = self.dt
        
        # Add any additional BC-specific parameters from params
        bc_params.update(getattr(params, 'boundary_condition_params', {}))
        
        # Create the boundary condition instance
        self.enforce_bc = get_boundary_condition(self.boundary_conditions, **bc_params)
        
        # Pre-bind the enforce method for performance
        self._enforce_bc = self.enforce_bc.enforce

        
    def _setup_thermostat(self, params, **kwargs):
        """Setup internal thermostat if requested."""
        # Check if thermostat is requested
        use_thermostat = (getattr(params, 'thermalization', False) or 
                         kwargs.get('use_thermostat', False))
        
        if use_thermostat:
            from ..thermostats import get_thermostat
            
            thermostat_type = (getattr(params, 'thermostat_type', 'berendsen') or
                              kwargs.get('thermostat_type', 'berendsen'))
            
            thermostat_class = get_thermostat(thermostat_type)
            self.thermostat = thermostat_class()
            self.thermostat.setup(params, **kwargs)
            
            self.thermalization = True
            self.thermalization_timestep = getattr(params, 'thermalization_timestep', 0)
            self._apply_thermostat = self.thermostat.update  # Pre-bound
        else:
            self.thermostat = None
            self.thermalization = False
            self._apply_thermostat = None
    
    def _prebind_methods(self):
        """Pre-bind frequently used methods for performance."""
        # These will be set when algorithms are linked
        pass
    
    def validate_timestep(self, params):
        """
        Validate timestep for stability.
        Override in subclasses for specific stability requirements.
        
        Parameters
        ----------
        params : Parameters
            Simulation parameters
        """
        # Base validation - check for reasonable timestep
        if self.dt <= 0:
            raise ValueError("Timestep must be positive")
        
        # Warn for very large timesteps
        if hasattr(params, 'species_plasma_frequencies'):
            max_freq = params.species_plasma_frequencies.max()
            if max_freq > 0:
                max_dt = 0.1 * (2 * pi / max_freq)  # 10% of plasma period
                if self.dt > max_dt:
                    print(f"Warning: Large timestep dt={self.dt:.2e}, "
                          f"recommended dt < {max_dt:.2e}")
    
    def should_apply_thermostat(self, timestep=None):
        """
        Determine if thermostat should be applied at this timestep.
        
        Parameters
        ----------
        timestep : int, optional
            Current timestep number
            
        Returns
        -------
        bool
            True if thermostat should be applied
        """
        if not self.thermalization or not self.thermostat:
            return False
        
        if timestep is not None:
            return timestep >= self.thermalization_timestep
        
        return True  # Default: apply if thermostat is configured
    
    def get_info(self):
        """
        Return integrator information.
        
        Returns
        -------
        dict
            Dictionary containing integrator properties
        """
        return {
            'name': self.name,
            'order': self.order,
            'timestep': self.dt,
            'supports_magnetic_field': self.supports_magnetic_field,
            'supports_adaptive_timestep': self.supports_adaptive_timestep,
            'supports_constraints': self.supports_constraints,
            'boundary_conditions': self.boundary_conditions,
            'thermostat': self.thermostat.name if self.thermostat else None,
            'thermalization': self.thermalization,
        }
    
    def pretty_print(self):
        """Print integrator information in a user-friendly format."""
        info = self.get_info()
        
        msg = f"\nINTEGRATOR: {info['name'].upper()}\n"
        msg += f"Order: {info['order']}\n"
        msg += f"Timestep: {info['timestep']:.6e}\n"
        msg += f"Boundary conditions: {info['boundary_conditions']}\n"
        if info['thermostat']:
            msg += f"Thermostat: {info['thermostat']}\n"
        else:
            msg += "Thermostat: None\n"
        
        return msg


class MagneticBase(IntegratorBase):
    """
    Base class for integrators that handle magnetic fields.
    
    Provides shared magnetic field setup, cyclotron frequency calculations,
    and helper functions for magnetic rotations.
    """
    
    def __init__(self):
        super().__init__()
        self.supports_magnetic_field = True
        
        # Magnetic field parameters
        self.magnetic_field = None
        self.magnetic_field_uvector = None
        self.species_cyclotron_frequencies = None
        self.omega_c = None
        
        # Pre-computed trigonometric functions
        self.sdt = None
        self.cdt = None
        self.ccodt = None
        self.ssodt = None
        
        # Working arrays for velocity components
        self.v_B = None
        self.v_F = None
        
    def setup(self, params, **kwargs):
        super().setup(params, **kwargs)
        
        if not getattr(params, 'magnetized', False):
            raise ValueError(f"{self.__class__.__name__} requires magnetized=True")
            
        self.magnetic_field = params.magnetic_field.copy()
        self.species_cyclotron_frequencies = params.species_cyclotron_frequencies.copy()
        
        self._setup_magnetic_field()
        self._validate_magnetic_timestep()
    
    def _setup_magnetic_field(self):
        """Setup magnetic field unit vector and cyclotron frequencies."""
        from scipy.linalg import norm
        
        # Create unit vector
        self.magnetic_field_uvector = self.magnetic_field / norm(self.magnetic_field)
        
        # Setup per-particle cyclotron frequencies
        self.omega_c = zeros((self.total_num_ptcls, 3))
        sp_start = 0
        for ic, sp_np in enumerate(self.species_num):
            sp_end = sp_start + sp_np
            self.omega_c[sp_start:sp_end, :] = self.species_cyclotron_frequencies[ic]
            sp_start = sp_end
        
        # Allocate working arrays
        self.v_B = zeros((self.total_num_ptcls, 3))
        self.v_F = zeros((self.total_num_ptcls, 3))
    
    def magnetic_helpers(self, coefficient):
        """
        Pre-compute trigonometric functions for magnetic integration.
        
        Parameters
        ----------
        coefficient : float
            Timestep coefficient (0.5 for Verlet, 1.0 for Boris/Leapfrog)
        """
        from numpy import sin, cos
        
        theta = self.omega_c * self.dt * coefficient
        self.sdt = sin(theta)
        self.cdt = cos(theta)
        self.ccodt = 1.0 - self.cdt
        self.ssodt = 1.0 - self.sdt / theta  # Handle theta=0 case properly
    
    def _validate_magnetic_timestep(self):
        """Validate timestep for magnetic field stability."""
        max_cyclotron_freq = abs(self.species_cyclotron_frequencies).max()
        if max_cyclotron_freq > 0:
            max_dt = 0.1 * (2 * pi / max_cyclotron_freq)  # 10% of cyclotron period
            if self.dt > max_dt:
                print(f"Warning: Large timestep for magnetic integrator dt={self.dt:.2e}, "
                      f"recommended dt < {max_dt:.2e}")
    
    def is_z_direction_field(self):
        """Check if magnetic field is purely in z-direction."""
        return abs(self.magnetic_field_uvector @ array([0.0, 0.0, 1.0])) > 0.999
    

class RungeKuttaBase(IntegratorBase):
    """
    Base class for Runge-Kutta integrators.
    
    Provides common RK functionality including Butcher tableau handling
    and multi-stage force evaluations.
    """
    
    def __init__(self):
        super().__init__()
        self.stages = None
        self.butcher_a = None  # Butcher tableau A matrix
        self.butcher_b = None  # Butcher tableau b vector
        self.butcher_c = None  # Butcher tableau c vector
        
        # Working arrays for RK stages
        self.k_pos = None  # Position derivatives
        self.k_vel = None  # Velocity derivatives
        self.temp_pos = None  # Temporary position storage
        self.temp_vel = None  # Temporary velocity storage
        
    def setup(self, params, **kwargs):
        super().setup(params, **kwargs)
        
        # Initialize working arrays
        self.k_pos = zeros((self.stages, self.total_num_ptcls, self.dimensions))
        self.k_vel = zeros((self.stages, self.total_num_ptcls, self.dimensions))
        self.temp_pos = zeros((self.total_num_ptcls, self.dimensions))
        self.temp_vel = zeros((self.total_num_ptcls, self.dimensions))
    
    def setup_butcher_tableau(self, a, b, c):
        """
        Setup Butcher tableau for RK method.
        
        Parameters
        ----------
        a : numpy.ndarray
            A matrix of Butcher tableau
        b : numpy.ndarray
            b vector of Butcher tableau
        c : numpy.ndarray
            c vector of Butcher tableau
        """
        self.butcher_a = array(a)
        self.butcher_b = array(b)
        self.butcher_c = array(c)
        self.stages = len(b)


class MultiTimestepBase(IntegratorBase):
    """
    Base class for multi-timestep integrators.
    
    Handles force splitting and multiple timestep management for
    algorithms like RESPA.
    """
    
    def __init__(self):
        super().__init__()
        self.outer_timestep = None
        self.inner_timestep = None
        self.timestep_ratio = None
        self.force_groups = {}  # Different force components
        self.supports_force_splitting = True
        
    def setup(self, params, **kwargs):
        super().setup(params, **kwargs)
        self.outer_timestep = self.dt
        self.timestep_ratio = kwargs.get('timestep_ratio', 4)
        self.inner_timestep = self.outer_timestep / self.timestep_ratio
        
    @abstractmethod
    def split_forces(self, ptcls):
        """
        Split forces into fast and slow components.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data
            
        Returns
        -------
        dict
            Dictionary with 'fast' and 'slow' force components
        """
        pass