"""
Thermostat registry and factory system.

This module provides automatic discovery and registration of all thermostat classes.
Users can add new thermostats by simply creating a new file that defines a class
inheriting from ThermostatBase.

Usage:
    from algorithms.thermostats import get_thermostat, list_thermostats
    
    thermostat_class = get_thermostat('berendsen')
    thermostat = thermostat_class()
    thermostat.setup(params)
"""

from .base import ThermostatBase, StochasticThermostatBase, DeterministicThermostatBase, GlobalThermostatBase

# Import all thermostat modules to trigger registration
from .stochastic import *
from .deterministic import *
from .global_methods import *
# from .advanced import *        # Uncomment when implemented

# Thermostat registry
THERMOSTAT_REGISTRY = {}
_ALIAS_REGISTRY = {}


def register_thermostat(name, thermostat_class, aliases=None):
    """
    Register a thermostat class.
    
    Parameters
    ----------
    name : str
        Primary name for the thermostat
    thermostat_class : class
        Class inheriting from ThermostatBase
    aliases : list, optional
        Alternative names for the thermostat
    """
    if not issubclass(thermostat_class, ThermostatBase):
        raise ValueError(f"{thermostat_class.__name__} must inherit from ThermostatBase")
    
    name = name.lower()
    THERMOSTAT_REGISTRY[name] = thermostat_class
    
    # Register aliases
    if aliases:
        for alias in aliases:
            alias = alias.lower()
            _ALIAS_REGISTRY[alias] = name


def get_thermostat(name):
    """
    Get thermostat class by name.
    
    Parameters
    ----------
    name : str
        Thermostat name or alias
        
    Returns
    -------
    class
        Thermostat class
        
    Raises
    ------
    ValueError
        If thermostat name is not found
    """
    name = name.lower()
    
    # Check aliases first
    if name in _ALIAS_REGISTRY:
        name = _ALIAS_REGISTRY[name]
    
    if name not in THERMOSTAT_REGISTRY:
        available = list(THERMOSTAT_REGISTRY.keys())
        available_aliases = list(_ALIAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown thermostat '{name}'. "
            f"Available thermostats: {available}. "
            f"Available aliases: {available_aliases}"
        )
    
    return THERMOSTAT_REGISTRY[name]


def list_thermostats():
    """
    List all available thermostats.
    
    Returns
    -------
    list
        List of available thermostat names
    """
    return list(THERMOSTAT_REGISTRY.keys())


def list_thermostat_aliases():
    """
    List all available thermostat aliases.
    
    Returns
    -------
    dict
        Dictionary mapping aliases to primary names
    """
    return _ALIAS_REGISTRY.copy()


def get_thermostat_info(name=None):
    """
    Get information about thermostat(s).
    
    Parameters
    ----------
    name : str, optional
        Specific thermostat name. If None, returns info for all.
        
    Returns
    -------
    dict or list
        Thermostat information
    """
    if name is not None:
        thermostat_class = get_thermostat(name)
        # Create temporary instance to get info
        temp_thermostat = thermostat_class()
        return temp_thermostat.get_info()
    else:
        # Return info for all thermostats
        all_info = {}
        for name, thermostat_class in THERMOSTAT_REGISTRY.items():
            temp_thermostat = thermostat_class()
            all_info[name] = temp_thermostat.get_info()
        return all_info


def get_thermostats_by_type(thermostat_type):
    """
    Get all thermostats of a specific type.
    
    Parameters
    ----------
    thermostat_type : str
        Type of thermostat ('stochastic', 'deterministic', 'global')
        
    Returns
    -------
    dict
        Dictionary of thermostat names and classes of the specified type
    """
    type_mapping = {
        'stochastic': StochasticThermostatBase,
        'deterministic': DeterministicThermostatBase,
        'global': GlobalThermostatBase
    }
    
    if thermostat_type not in type_mapping:
        raise ValueError(f"Unknown thermostat type '{thermostat_type}'. "
                        f"Available types: {list(type_mapping.keys())}")
    
    base_class = type_mapping[thermostat_type]
    filtered_thermostats = {}
    
    for name, thermostat_class in THERMOSTAT_REGISTRY.items():
        if issubclass(thermostat_class, base_class):
            filtered_thermostats[name] = thermostat_class
    
    return filtered_thermostats


def _auto_register():
    """
    Automatically register all ThermostatBase subclasses.
    
    This function is called automatically when the module is imported.
    It discovers all classes that inherit from ThermostatBase and registers them.
    """
    def get_all_subclasses(cls):
        """Recursively get all subclasses."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses
    
    # Get all subclasses of ThermostatBase
    all_thermostats = get_all_subclasses(ThermostatBase)
    
    for thermostat_class in all_thermostats:
        # Skip abstract base classes
        if thermostat_class.__name__.endswith('Base'):
            continue
            
        # Get class name for registration
        class_name = thermostat_class.__name__.lower()
        
        # Register the class
        register_thermostat(class_name, thermostat_class)
        
        # Register aliases if defined
        if hasattr(thermostat_class, '_aliases'):
            register_thermostat(class_name, thermostat_class, thermostat_class._aliases)


# Auto-register all thermostats when module is imported
_auto_register()

# Public API
__all__ = [
    'ThermostatBase', 
    'StochasticThermostatBase', 
    'DeterministicThermostatBase',
    'GlobalThermostatBase',
    'register_thermostat', 
    'get_thermostat', 
    'list_thermostats',
    'list_thermostat_aliases',
    'get_thermostat_info',
    'get_thermostats_by_type'
]