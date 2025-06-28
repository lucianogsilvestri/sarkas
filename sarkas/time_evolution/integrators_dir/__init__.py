"""
Time integrator registry and factory system.

This module provides automatic discovery and registration of all integrator classes.
Users can add new integrators by simply creating a new file that defines a class
inheriting from IntegratorBase.

Usage:
    from algorithms.integrators import get_integrator, list_integrators
    
    integrator_class = get_integrator('verlet')
    integrator = integrator_class()
    integrator.setup(params)
"""

from .base import IntegratorBase, RungeKuttaBase, MultiTimestepBase
from .stochastic_base import (
    StochasticIntegratorBase,
    UnderdampedLangevinBase,
    StochasticThermostatIntegratorBase
)

# Import all integrator modules directly (flat structure)
# Deterministic integrators
from .verlet import *
from .leapfrog import *

# Runge-Kutta integrators
# from .rk2 import *
# from .rk4 import *
# from .rk45 import *

# Stochastic integrators
from .langevin import *
from .andersen import *
from .stochastic_velocity_rescaling import *

# Magnetic field integrators
# from .boris import *
# from .magnetic_verlet import *

# Multi-timestep integrators (when implemented)
# from .respa import *

# Advanced integrators (when implemented)
# from .adaptive_timestep import *
# from .constrained import *

# Integrator registry
INTEGRATOR_REGISTRY = {}
_ALIAS_REGISTRY = {}


def register_integrator(name, integrator_class, aliases=None):
    """
    Register an integrator class.
    
    Parameters
    ----------
    name : str
        Primary name for the integrator
    integrator_class : class
        Class inheriting from IntegratorBase
    aliases : list, optional
        Alternative names for the integrator
    """
    if not issubclass(integrator_class, IntegratorBase):
        raise ValueError(f"{integrator_class.__name__} must inherit from IntegratorBase")
    
    name = name.lower()
    INTEGRATOR_REGISTRY[name] = integrator_class
    
    # Register aliases
    if aliases:
        for alias in aliases:
            alias = alias.lower()
            _ALIAS_REGISTRY[alias] = name


def get_integrator(name):
    """
    Get integrator class by name.
    
    Parameters
    ----------
    name : str
        Integrator name or alias
        
    Returns
    -------
    class
        Integrator class
        
    Raises
    ------
    ValueError
        If integrator name is not found
    """
    name = name.lower()
    
    # Check aliases first
    if name in _ALIAS_REGISTRY:
        name = _ALIAS_REGISTRY[name]
    
    if name not in INTEGRATOR_REGISTRY:
        available = list(INTEGRATOR_REGISTRY.keys())
        available_aliases = list(_ALIAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown integrator '{name}'. "
            f"Available integrators: {available}. "
            f"Available aliases: {available_aliases}"
        )
    
    return INTEGRATOR_REGISTRY[name]


def list_integrators():
    """
    List all available integrators.
    
    Returns
    -------
    list
        List of available integrator names
    """
    return list(INTEGRATOR_REGISTRY.keys())


def list_integrator_aliases():
    """
    List all available integrator aliases.
    
    Returns
    -------
    dict
        Dictionary mapping aliases to primary names
    """
    return _ALIAS_REGISTRY.copy()


def get_integrator_info(name=None):
    """
    Get information about integrator(s).
    
    Parameters
    ----------
    name : str, optional
        Specific integrator name. If None, returns info for all.
        
    Returns
    -------
    dict or list
        Integrator information
    """
    if name is not None:
        integrator_class = get_integrator(name)
        # Create temporary instance to get info
        temp_integrator = integrator_class()
        return temp_integrator.get_info()
    else:
        # Return info for all integrators
        all_info = {}
        for name, integrator_class in INTEGRATOR_REGISTRY.items():
            temp_integrator = integrator_class()
            all_info[name] = temp_integrator.get_info()
        return all_info


def get_integrators_by_type(integrator_type):
    """
    Get all integrators of a specific type.
    
    Parameters
    ----------
    integrator_type : str
        Type of integrator ('deterministic', 'stochastic', 'runge_kutta', 'magnetic', 'multi_timestep')
        
    Returns
    -------
    dict
        Dictionary of integrator names and classes of the specified type
    """
    type_mapping = {
        'deterministic': IntegratorBase,
        'stochastic': StochasticIntegratorBase,
        'runge_kutta': RungeKuttaBase,
        'multi_timestep': MultiTimestepBase
    }
    
    if integrator_type not in type_mapping:
        raise ValueError(f"Unknown integrator type '{integrator_type}'. "
                        f"Available types: {list(type_mapping.keys())}")
    
    base_class = type_mapping[integrator_type]
    filtered_integrators = {}
    
    for name, integrator_class in INTEGRATOR_REGISTRY.items():
        if integrator_type == 'deterministic':
            # Deterministic = IntegratorBase but not stochastic
            if (issubclass(integrator_class, IntegratorBase) and 
                not issubclass(integrator_class, StochasticIntegratorBase)):
                filtered_integrators[name] = integrator_class
        else:
            # For other types, direct subclass check
            if issubclass(integrator_class, base_class):
                filtered_integrators[name] = integrator_class
    
    return filtered_integrators


def _auto_register():
    """
    Automatically register all IntegratorBase subclasses.
    
    This function is called automatically when the module is imported.
    It discovers all classes that inherit from IntegratorBase and registers them.
    """
    def get_all_subclasses(cls):
        """Recursively get all subclasses."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses
    
    # Get all subclasses of IntegratorBase
    all_integrators = get_all_subclasses(IntegratorBase)
    
    for integrator_class in all_integrators:
        # Skip abstract base classes
        if integrator_class.__name__.endswith('Base'):
            continue
            
        # Get class name for registration
        class_name = integrator_class.__name__.lower()
        
        # Register the class
        register_integrator(class_name, integrator_class)
        
        # Register aliases if defined
        if hasattr(integrator_class, '_aliases'):
            register_integrator(class_name, integrator_class, integrator_class._aliases)


# Auto-register all integrators when module is imported
_auto_register()

# Public API
__all__ = [
    # Base classes
    'IntegratorBase', 
    'RungeKuttaBase', 
    'MultiTimestepBase',
    'StochasticIntegratorBase',
    'UnderdampedLangevinBase',
    'StochasticThermostatIntegratorBase',
    
    # Factory functions
    'register_integrator', 
    'get_integrator', 
    'list_integrators',
    'list_integrator_aliases',
    'get_integrator_info',
    'get_integrators_by_type'
]