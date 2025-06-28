"""
Boundary conditions registry and factory system.

This module provides automatic discovery and registration of all boundary condition classes.
Users can add new boundary conditions by simply creating a new file that defines a class
inheriting from BoundaryConditionBase.

Usage:
    from algorithms.boundary_conditions import get_boundary_condition
    
    bc = get_boundary_condition('periodic', box_lengths=[10, 10, 10])
    bc.enforce(ptcls)
"""

from .base import BoundaryConditionBase
from .implementations import *

# Boundary condition registry
BOUNDARY_CONDITION_REGISTRY = {}
_ALIAS_REGISTRY = {}


def register_boundary_condition(name, bc_class, aliases=None):
    """
    Register a boundary condition class.
    
    Parameters
    ----------
    name : str
        Primary name for the boundary condition
    bc_class : class
        Class inheriting from BoundaryConditionBase
    aliases : list, optional
        Alternative names for the boundary condition
    """
    if not issubclass(bc_class, BoundaryConditionBase):
        raise ValueError(f"{bc_class.__name__} must inherit from BoundaryConditionBase")
    
    name = name.lower()
    BOUNDARY_CONDITION_REGISTRY[name] = bc_class
    
    # Register aliases
    if aliases:
        for alias in aliases:
            alias = alias.lower()
            _ALIAS_REGISTRY[alias] = name


def get_boundary_condition(name, **params):
    """
    Get boundary condition instance by name.
    
    Parameters
    ----------
    name : str
        Boundary condition name or alias
    **params : dict
        Parameters to pass to the boundary condition constructor
        
    Returns
    -------
    BoundaryConditionBase
        Boundary condition instance
        
    Raises
    ------
    ValueError
        If boundary condition name is not found
    """
    name = name.lower()
    
    # Check aliases first
    if name in _ALIAS_REGISTRY:
        name = _ALIAS_REGISTRY[name]
    
    if name not in BOUNDARY_CONDITION_REGISTRY:
        available = list(BOUNDARY_CONDITION_REGISTRY.keys())
        available_aliases = list(_ALIAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown boundary condition '{name}'. "
            f"Available boundary conditions: {available}. "
            f"Available aliases: {available_aliases}"
        )
    
    bc_class = BOUNDARY_CONDITION_REGISTRY[name]
    return bc_class(**params)


def get_boundary_condition_class(name):
    """
    Get boundary condition class (not instance) by name.
    
    Parameters
    ----------
    name : str
        Boundary condition name or alias
        
    Returns
    -------
    class
        Boundary condition class
    """
    name = name.lower()
    
    # Check aliases first
    if name in _ALIAS_REGISTRY:
        name = _ALIAS_REGISTRY[name]
    
    if name not in BOUNDARY_CONDITION_REGISTRY:
        available = list(BOUNDARY_CONDITION_REGISTRY.keys())
        raise ValueError(f"Unknown boundary condition '{name}'. Available: {available}")
    
    return BOUNDARY_CONDITION_REGISTRY[name]


def list_boundary_conditions():
    """
    List all available boundary conditions.
    
    Returns
    -------
    list
        List of available boundary condition names
    """
    return list(BOUNDARY_CONDITION_REGISTRY.keys())


def list_boundary_condition_aliases():
    """
    List all available boundary condition aliases.
    
    Returns
    -------
    dict
        Dictionary mapping aliases to primary names
    """
    return _ALIAS_REGISTRY.copy()


def get_boundary_condition_info(name=None):
    """
    Get information about boundary condition(s).
    
    Parameters
    ----------
    name : str, optional
        Specific boundary condition name. If None, returns info for all.
        
    Returns
    -------
    dict or list
        Boundary condition information
    """
    if name is not None:
        bc_class = get_boundary_condition_class(name)
        # Create temporary instance to get info
        temp_bc = bc_class(box_lengths=[1, 1, 1])  # Dummy box lengths
        return temp_bc.get_info()
    else:
        # Return info for all boundary conditions
        all_info = {}
        for name, bc_class in BOUNDARY_CONDITION_REGISTRY.items():
            temp_bc = bc_class(box_lengths=[1, 1, 1])  # Dummy box lengths
            all_info[name] = temp_bc.get_info()
        return all_info


def _auto_register():
    """
    Automatically register all BoundaryConditionBase subclasses.
    
    This function is called automatically when the module is imported.
    It discovers all classes that inherit from BoundaryConditionBase and registers them.
    """
    def get_all_subclasses(cls):
        """Recursively get all subclasses."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses
    
    # Get all subclasses of BoundaryConditionBase
    all_bcs = get_all_subclasses(BoundaryConditionBase)
    
    for bc_class in all_bcs:
        # Skip abstract base classes
        if bc_class.__name__.endswith('Base'):
            continue
            
        # Get class name for registration
        class_name = bc_class.__name__.lower()
        
        # Remove 'bc' suffix if present
        if class_name.endswith('bc'):
            class_name = class_name[:-2]
        
        # Register the class
        register_boundary_condition(class_name, bc_class)
        
        # Register aliases if defined
        if hasattr(bc_class, '_aliases'):
            register_boundary_condition(class_name, bc_class, bc_class._aliases)


# Auto-register all boundary conditions when module is imported
_auto_register()

# Public API
__all__ = [
    'BoundaryConditionBase',
    'register_boundary_condition',
    'get_boundary_condition',
    'get_boundary_condition_class',
    'list_boundary_conditions',
    'list_boundary_condition_aliases',
    'get_boundary_condition_info'
]