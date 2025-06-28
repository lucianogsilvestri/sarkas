"""
Physics module for Sarkas plasma simulation.

This module contains pure, optimized functions for physics calculations
extracted from the Particles class to improve modularity and performance.
"""

from .thermodynamics import (
    kinetic_energy,
    temperature_from_kinetic_energy,
    pressure_kinetic_contribution,
    pressure_virial_contribution,
    pressure_tensor_total,
    pressure_scalar,
    species_kinetic_energy,
    species_temperature,
    species_enthalpy,
)

from .mechanical import (
    momentum,
    center_of_mass_velocity,
    remove_center_of_mass_motion,
    angular_momentum,
    total_angular_momentum,
    force_torque,
    stress_tensor,
    species_momentum,
    species_center_of_mass_velocity,
    calculate_momentum,
    calculate_center_of_mass_velocity,
)

from .transport import (
    electric_current,
    species_electric_current,
    diffusion_flux,
    diffusion_flux_from_particles,
    velocity_moments,
)

from .aggregation import (
    species_sum,
    species_mean,
    species_weighted_sum,
    species_variance,
    get_species_counts,
    validate_species_aggregation,
    # Backward compatibility aliases
    species_vector_sum,
    species_tensor_sum,
    # Legacy functions
    scalar_species_loop,
    vector_species_loop,
    tensor_species_loop,
)

__all__ = [
    # Thermodynamics
    'kinetic_energy',
    'temperature_from_kinetic_energy',
    'pressure_kinetic_contribution',
    'pressure_virial_contribution',
    'pressure_tensor_total',
    'pressure_scalar',
    'species_kinetic_energy',
    'species_temperature',
    'species_enthalpy',
    
    # Mechanical
    'momentum',
    'center_of_mass_velocity',
    'remove_center_of_mass_motion',
    'angular_momentum',
    'total_angular_momentum',
    'force_torque',
    'stress_tensor',
    'species_momentum',
    'species_center_of_mass_velocity',
    'calculate_momentum',
    'calculate_center_of_mass_velocity',
    
    # Transport
    'electric_current',
    'species_electric_current',
    'diffusion_flux',
    'diffusion_flux_from_particles',
    'velocity_moments',
    
    # Aggregation
    'species_sum',
    'species_mean',
    'species_weighted_sum',
    'species_variance',
    'get_species_counts',
    'validate_species_aggregation',
    'species_vector_sum',  # Backward compatibility
    'species_tensor_sum',  # Backward compatibility
    'scalar_species_loop',  # Legacy
    'vector_species_loop',  # Legacy
    'tensor_species_loop',  # Legacy
] 