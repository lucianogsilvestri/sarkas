"""
Potential energy functions for molecular dynamics simulations.

This module provides a comprehensive collection of potential energy functions
commonly used in molecular dynamics and plasma physics simulations, including
both short-range and long-range interactions.

The module supports multiple computational methods:
- Particle-Particle (PP): Direct pairwise calculations
- Particle-Particle Particle-Mesh (PPPM): Efficient long-range calculations
- Ewald summation: Alternative long-range method

Usage:
    from potentials import get_potential, list_potentials
    
    potential_class = get_potential('yukawa')
    potential = potential_class()
    potential.setup(params, species)

Available Potentials:
    - Yukawa: Screened Coulomb potential
    - Coulomb: Pure Coulomb potential (when implemented)
    - Lennard-Jones: Short-range potential (when implemented)
    - Morse: Anharmonic potential (when implemented)
"""

from .base import PotentialBase, PairPotentialBase, LongRangePotentialBase
from .yukawa import Yukawa, yukawa_force, yukawa_force_pppm, potential_derivatives

# Potential registry for factory pattern
POTENTIAL_REGISTRY = {}
_ALIAS_REGISTRY = {}


def register_potential(name, potential_class, aliases=None):
    """
    Register a potential class.
    
    Parameters
    ----------
    name : str
        Primary name for the potential
    potential_class : class
        Class inheriting from PotentialBase
    aliases : list, optional
        Alternative names for the potential
    """
    if not issubclass(potential_class, PotentialBase):
        raise ValueError(f"{potential_class.__name__} must inherit from PotentialBase")
    
    name = name.lower()
    POTENTIAL_REGISTRY[name] = potential_class
    
    # Register aliases
    if aliases:
        for alias in aliases:
            alias = alias.lower()
            _ALIAS_REGISTRY[alias] = name


def get_potential(name):
    """
    Get potential class by name.
    
    Parameters
    ----------
    name : str
        Potential name or alias
        
    Returns
    -------
    class
        Potential class
        
    Raises
    ------
    ValueError
        If potential name is not found
    """
    name = name.lower()
    
    # Check aliases first
    if name in _ALIAS_REGISTRY:
        name = _ALIAS_REGISTRY[name]
    
    if name not in POTENTIAL_REGISTRY:
        available = list(POTENTIAL_REGISTRY.keys())
        available_aliases = list(_ALIAS_REGISTRY.keys())
        raise ValueError(
            f"Unknown potential '{name}'. "
            f"Available potentials: {available}. "
            f"Available aliases: {available_aliases}"
        )
    
    return POTENTIAL_REGISTRY[name]


def list_potentials():
    """
    List all available potentials.
    
    Returns
    -------
    list
        List of available potential names
    """
    return list(POTENTIAL_REGISTRY.keys())


def list_potential_aliases():
    """
    List all available potential aliases.
    
    Returns
    -------
    dict
        Dictionary mapping aliases to primary names
    """
    return _ALIAS_REGISTRY.copy()


def get_potential_info(name=None):
    """
    Get information about potential(s).
    
    Parameters
    ----------
    name : str, optional
        Specific potential name. If None, returns info for all.
        
    Returns
    -------
    dict or list
        Potential information
    """
    if name is not None:
        potential_class = get_potential(name)
        # Create temporary instance to get info
        temp_potential = potential_class()
        return temp_potential.get_info()
    else:
        # Return info for all potentials
        all_info = {}
        for name, potential_class in POTENTIAL_REGISTRY.items():
            temp_potential = potential_class()
            all_info[name] = temp_potential.get_info()
        return all_info


def get_potentials_by_type(potential_type):
    """
    Get all potentials of a specific type.
    
    Parameters
    ----------
    potential_type : str
        Type of potential ('pair', 'long_range', 'short_range', 'screened')
        
    Returns
    -------
    dict
        Dictionary of potential names and classes of the specified type
    """
    type_mapping = {
        'pair': PairPotentialBase,
        'long_range': LongRangePotentialBase,
        'short_range': PairPotentialBase,  # Most short-range are pair potentials
        'screened': LongRangePotentialBase  # Screened potentials support long-range methods
    }
    
    if potential_type not in type_mapping:
        raise ValueError(f"Unknown potential type '{potential_type}'. "
                        f"Available types: {list(type_mapping.keys())}")
    
    base_class = type_mapping[potential_type]
    filtered_potentials = {}
    
    for name, potential_class in POTENTIAL_REGISTRY.items():
        if potential_type == 'short_range':
            # Short-range = PairPotentialBase but not LongRangePotentialBase
            if (issubclass(potential_class, PairPotentialBase) and 
                not issubclass(potential_class, LongRangePotentialBase)):
                filtered_potentials[name] = potential_class
        else:
            # For other types, direct subclass check
            if issubclass(potential_class, base_class):
                filtered_potentials[name] = potential_class
    
    return filtered_potentials


def get_potentials_by_method(method):
    """
    Get all potentials that support a specific computational method.
    
    Parameters
    ----------
    method : str
        Computational method ('pp', 'pppm', 'ewald')
        
    Returns
    -------
    dict
        Dictionary of potential names and classes that support the method
    """
    method = method.lower()
    supported_potentials = {}
    
    for name, potential_class in POTENTIAL_REGISTRY.items():
        temp_potential = potential_class()
        
        if method == 'pp':
            # All potentials support particle-particle
            supported_potentials[name] = potential_class
        elif method == 'pppm':
            if getattr(temp_potential, 'supports_pppm', False):
                supported_potentials[name] = potential_class
        elif method == 'ewald':
            if getattr(temp_potential, 'supports_ewald', False):
                supported_potentials[name] = potential_class
        else:
            raise ValueError(f"Unknown method '{method}'. "
                           f"Available methods: ['pp', 'pppm', 'ewald']")
    
    return supported_potentials


def _auto_register():
    """
    Automatically register all PotentialBase subclasses.
    
    This function is called automatically when the module is imported.
    It discovers all classes that inherit from PotentialBase and registers them.
    """
    def get_all_subclasses(cls):
        """Recursively get all subclasses."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses
    
    # Get all subclasses of PotentialBase
    all_potentials = get_all_subclasses(PotentialBase)
    
    for potential_class in all_potentials:
        # Skip abstract base classes
        if potential_class.__name__.endswith('Base'):
            continue
            
        # Get class name for registration
        class_name = potential_class.__name__.lower()
        
        # Register the class
        register_potential(class_name, potential_class)
        
        # Register aliases if defined
        if hasattr(potential_class, '_aliases'):
            register_potential(class_name, potential_class, potential_class._aliases)


# Auto-register all potentials when module is imported
_auto_register()


# Utility functions for backward compatibility and convenience
def create_potential(potential_type, **kwargs):
    """
    Create and setup a potential instance.
    
    Parameters
    ----------
    potential_type : str
        Type of potential to create
    **kwargs : dict
        Parameters to pass to the potential setup method
        
    Returns
    -------
    PotentialBase
        Configured potential instance
        
    Examples
    --------
    >>> potential = create_potential('yukawa', screening_length=1.0)
    >>> # Still need to call setup with params and species
    """
    potential_class = get_potential(potential_type)
    potential = potential_class()
    
    # Store kwargs for later setup if needed
    potential._creation_kwargs = kwargs
    
    return potential


def validate_potential_method_compatibility(potential_name, method):
    """
    Validate that a potential supports a given computational method.
    
    Parameters
    ----------
    potential_name : str
        Name of the potential
    method : str
        Computational method ('pp', 'pppm', 'ewald')
        
    Returns
    -------
    bool
        True if potential supports the method
        
    Raises
    ------
    ValueError
        If potential or method is unknown
    """
    potential_class = get_potential(potential_name)
    temp_potential = potential_class()
    
    method = method.lower()
    if method == 'pp':
        return True  # All potentials support particle-particle
    elif method == 'pppm':
        return getattr(temp_potential, 'supports_pppm', False)
    elif method == 'ewald':
        return getattr(temp_potential, 'supports_ewald', False)
    else:
        raise ValueError(f"Unknown method '{method}'")


# Public API
__all__ = [
    # Base classes
    'PotentialBase', 
    'PairPotentialBase', 
    'LongRangePotentialBase',
    
    # Potential implementations
    'Yukawa',
    
    # Force functions (for advanced users)
    'yukawa_force',
    'yukawa_force_pppm',
    'potential_derivatives',
    
    # Factory functions
    'register_potential', 
    'get_potential', 
    'list_potentials',
    'list_potential_aliases',
    'get_potential_info',
    'get_potentials_by_type',
    'get_potentials_by_method',
    'create_potential',
    'validate_potential_method_compatibility'
]