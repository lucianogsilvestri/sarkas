"""
Base class for boundary conditions.
"""

from abc import ABC, abstractmethod
import numpy as np


class BoundaryConditionBase(ABC):
    """
    Abstract base class for boundary condition implementations.
    
    This class defines the interface that all boundary condition methods should follow.
    It provides a framework for implementing different types of boundary conditions
    with consistent interfaces and validation.
    
    Attributes
    ----------
    name : str
        Name of the boundary condition
    box_lengths : numpy.ndarray
        Box dimensions
    dimensions : int
        Number of spatial dimensions
    requires_velocities : bool
        Whether this BC needs access to particle velocities
    requires_accelerations : bool
        Whether this BC needs access to particle accelerations
    requires_charges : bool
        Whether this BC needs access to particle charges
    requires_timestep : bool
        Whether this BC needs timestep information
    modifies_positions : bool
        Whether this BC modifies particle positions
    modifies_velocities : bool
        Whether this BC modifies particle velocities
    modifies_charges : bool
        Whether this BC modifies particle charges
    """
    
    def __init__(self, box_lengths, dimensions = 3, **kwargs):
        self.name = self.__class__.__name__.lower()
        
        # Remove 'bc' suffix if present
        if self.name.endswith('bc'):
            self.name = self.name[:-2]
        
        # System parameters
        self.box_lengths = np.asarray(box_lengths, dtype=float)
        
        self.dimensions = dimensions
        
        # Capability flags (set by subclasses)
        self.requires_velocities = False
        self.requires_accelerations = False
        self.requires_charges = False
        self.requires_timestep = False
        
        # Modification flags (set by subclasses)
        self.modifies_positions = True
        self.modifies_velocities = False
        self.modifies_accelerations = False
        self.modifies_charges = False
        
        # Subclass-specific setup
        self.setup(**kwargs)
        
        # Validate parameters
        self.validate_parameters()
    
    def setup(self, **kwargs):
        """
        Setup boundary condition with specific parameters.
        Override in subclasses for BC-specific initialization.
        
        Parameters
        ----------
        **kwargs : dict
            Additional setup parameters
        """
        pass
    
    @abstractmethod
    def enforce(self, ptcls):
        """
        Enforce boundary conditions on particles.
        
        This method must be implemented by all subclasses to define
        the specific boundary condition behavior.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to apply boundary conditions to
        """
        pass
    
    def __call__(self, ptcls):
        """
        Function-like interface for boundary condition enforcement.
        
        This allows the boundary condition to be called like a function:
        bc(ptcls) instead of bc.enforce(ptcls)
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to apply boundary conditions to
        """
        return self.enforce(ptcls)
    
    def validate_parameters(self):
        """
        Validate boundary condition parameters.
        Override in subclasses for specific validation requirements.
        """
        if self.box_lengths is None:
            raise ValueError("Box lengths must be specified")
        
        if (self.box_lengths <= 0).any():
            raise ValueError("All box lengths must be positive")
        
        if self.dimensions <= 0:
            raise ValueError("Dimensions must be positive")
        
        if self.dimensions > 3:
            raise ValueError("Maximum 3 dimensions supported")
    
    def get_info(self):
        """
        Get information about this boundary condition.
        
        Returns
        -------
        dict
            Dictionary containing BC properties
        """
        return {
            'name': self.name,
            'box_lengths': self.box_lengths.tolist(),
            'dimensions': self.dimensions,
            'requires_velocities': self.requires_velocities,
            'requires_accelerations': self.requires_accelerations,
            'requires_charges': self.requires_charges,
            'requires_timestep': self.requires_timestep,
            'modifies_positions': self.modifies_positions,
            'modifies_velocities': self.modifies_velocities,
            'modifies_accelerations': self.modifies_accelerations,
            'modifies_charges': self.modifies_charges,
        }
    
    def pretty_print(self):
        """Print boundary condition information in a user-friendly format."""
        info = self.get_info()
        msg = f"Boundary Condition: {info['name'].upper()}\n"
        
        
        requirements = []
        if info['requires_velocities']:
            requirements.append("velocities")
        if info['requires_accelerations']:
            requirements.append("accelerations")
        if info['requires_charges']:
            requirements.append("charges")
        if info['requires_timestep']:
            requirements.append("timestep")
        
        if requirements:
            msg += f"Requires: {', '.join(requirements)}\n"
        
        modifications = []
        if info['modifies_positions']:
            modifications.append("positions")
        if info['modifies_velocities']:
            modifications.append("velocities")
        if info['modifies_accelerations']:
            modifications.append("accelerations")
        if info['modifies_charges']:
            modifications.append("charges")
        
        if modifications:
            msg += f"Modifies: {', '.join(modifications)}\n"
        
    def check_particle_requirements(self, ptcls):
        """
        Check if particle object has all required attributes.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data object
            
        Raises
        ------
        AttributeError
            If required attributes are missing
        """
        required_attrs = ['pos']
        
        if self.requires_velocities:
            required_attrs.append('vel')
        if self.requires_accelerations:
            required_attrs.append('acc')
        if self.requires_charges:
            required_attrs.append('charges')
        
        for attr in required_attrs:
            if not hasattr(ptcls, attr):
                raise AttributeError(f"Particle object missing required attribute: {attr}")
        
        # Check dimensions match
        if ptcls.pos.shape[1] != self.dimensions:
            raise ValueError(f"Particle position dimensions ({ptcls.pos.shape[1]}) "
                           f"don't match BC dimensions ({self.dimensions})")