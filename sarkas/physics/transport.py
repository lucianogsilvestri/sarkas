"""
Transport properties calculator for Sarkas physics calculations.

This module provides pure, optimized functions for transport properties
calculations including electric current, heat flux, diffusion flux,
and velocity moments.
"""

from numpy import ndarray, ones_like, zeros, sum, sqrt, array
from numba import jit  
from typing import Union, Tuple

from .aggregation import (
    species_sum, fast_species_sum, species_mean, 
    get_species_counts, validate_species_aggregation
)


@jit(nopython=True)
def electric_current(velocities: ndarray, charges: ndarray) -> ndarray:
    """Calculate electric current per particle.
    
    The electric current density per particle is calculated as:
    
    .. math::
        \\vec{j}_i = q_i \\vec{v}_i
    
    where :math:`q_i` is the charge and :math:`\\vec{v}_i` is the velocity vector
    of particle :math:`i`.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)`` where N is the number of particles
    charges : numpy.ndarray  
        Particle charges with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle electric current with shape ``(N, 3)``
        
    Examples
    --------
    Calculate electric current for a system of particles:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.transport import electric_current
        
        # Generate random velocities and charges
        velocities = np.random.randn(1000, 3)
        charges = np.random.randn(1000)
        
        # Calculate electric current
        j = electric_current(velocities, charges)
        
        # Verify results
        assert j.shape == (1000, 3)
        assert np.allclose(j, velocities * charges[:, None])
    """
    for i in range(len(velocities)):
        for j in range(3):
            j[i, j] = charges[i] * velocities[i, j]
    return j


def species_electric_current(velocities: ndarray, charges: ndarray, 
                           species_id: ndarray, num_species: int) -> ndarray:
    """Calculate total electric current for each species.
    
    The total electric current for species :math:`s` is calculated as:
    
    .. math::
        \\vec{J}_s = \\sum_{i \\in s} q_i \\vec{v}_i
    
    where the sum is over all particles belonging to species :math:`s`.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    charges : numpy.ndarray
        Particle charges with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Total electric current per species with shape ``(num_species, 3)``
        
    Examples
    --------
    Calculate species electric currents:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.transport import species_electric_current
        
        velocities = np.random.randn(1000, 3)
        charges = np.random.randn(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        J_species = species_electric_current(velocities, charges, species_id, num_species=2)
        assert J_species.shape == (2, 3)
    """
    # Calculate per-particle electric current
    per_particle_current = electric_current(velocities, charges)
    
    # Sum over species (vector)
    return fast_species_sum(per_particle_current, species_id, num_species)


def diffusion_flux(velocities: ndarray, concentrations: ndarray,
                   weights: ndarray) -> ndarray:
    """
    Calculate generalized diffusion flux for multi-species systems.

    The diffusion flux represents species transport relative to a 
    reference velocity computed as a weighted average over all species.
    This function supports barycentric (mass-based), molar-based, or 
    volume-based diffusion fluxes depending on the weights provided.

    The flux is computed as:
    
    .. math::
        J_i = \\rho_i (v_i - v_{ref})

    where the reference velocity :math:`v_{ref}` is defined as:

    .. math::
        v_{ref} = \\frac{\\sum_i \\rho_i v_i}{\\sum_i \\rho_i}

    where:
        - :math:`\\rho_i` is the species-specific weighting factor times concentration
        - :math:`v_i` is the average velocity of species i
        - :math:`v_{ref}` is the reference velocity, computed as the weighted average over all species velocities

    Parameters
    ----------
    velocities : ndarray
        Average velocities for each species. Shape: ``(num_species, 3)``.

    concentrations : ndarray
        Number density (or another additive concentration-like measure) 
        of each species. Shape: ``(num_species,)``.

    weights : ndarray
        Weighting property of each species used to compute the reference 
        velocity. Shape: ``(num_species,)``. Examples:
            - species masses for barycentric (mass-based) flux
            - unity or molar masses for molar diffusion flux
            - partial molar volumes for volume-based flux

    Returns
    -------
    ndarray
        Diffusion flux for each species. Shape: ``(num_species, 3)``.

    Examples
    --------
    >>> import numpy as np
    >>> vel = np.array([[100.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    >>> conc = np.array([1e20, 1e20])
    >>> masses = np.array([1.67e-27, 9.11e-31])  # Proton, electron
    >>> J_diff = diffusion_flux(vel, conc, masses)
    >>> print(f"Diffusion flux: {J_diff}")

    Notes
    -----
    - Total diffusion flux sums to zero when computed with consistent weights.
    - The choice of weights determines the reference frame.
    - This function generalizes to molar or volume-based fluxes by adjusting `weights`.
    """

    return _diffusion_flux_kernel(velocities, concentrations, weights)


@jit(nopython=True)
def _diffusion_flux_kernel(velocities: ndarray, concentrations: ndarray,
                          weights: ndarray) -> ndarray:
    """
    Numba-compiled kernel for diffusion flux calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Species average velocities. Shape: (N_species, 3).
    concentrations : ndarray
        Species concentrations. Shape: (N_species,).
    weights : ndarray
        Species weights Shape: (N_species,).
        
    Returns
    -------
    ndarray
        Diffusion flux for each species. Shape: (N_species, 3).
    """
    n_species = velocities.shape[0]
    diffusion_flux = zeros((n_species, 3))
    
    # Calculate total weights density
    total_weights_density = 0.0
    for i in range(n_species):
        total_weights_density += concentrations[i] * weights[i]
    
    # Calculate center of weights velocity
    cm_velocity = zeros(3)
    if total_weights_density > 0:
        for i in range(n_species):
            weights_fraction = (concentrations[i] * weights[i]) / total_weights_density
            for j in range(3):
                cm_velocity[j] += weights_fraction * velocities[i, j]
        
    # Calculate diffusion flux: ρᵢ * (vᵢ - v_cm)
    for i in range(n_species):
        weights_density_i = concentrations[i] * weights[i]
        for j in range(3):
            diffusion_flux[i, j] = weights_density_i * (velocities[i, j] - cm_velocity[j])
    
    return diffusion_flux

@jit(nopython=True)
def diffusion_flux_from_particles(velocities: ndarray, species_id: ndarray, 
                                 num_species: int, weights: ndarray = None,
                                 ) -> ndarray:
    """Calculate diffusion flux from particle-level data.
    
    This is a convenience function that aggregates particle velocities to species
    level and then calls the main diffusion_flux function.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
    weights : numpy.ndarray
        Species weights for reference velocity calculation with shape ``(num_species,)``.
        
    Returns
    -------
    numpy.ndarray
        Diffusion flux per species with shape ``(num_species, 3)``
    """
    
    # Calculate species average velocities (vector)
    species_velocities = fast_species_sum(velocities, species_id, num_species)
    
    # Get species counts for normalization
    species_counts = species_sum(species_id, ones_like(species_id), num_species)

    concentrations = species_counts / sum(species_counts)

    # Call the main diffusion flux function
    return _diffusion_flux_kernel(species_velocities, concentrations, weights)


@jit(nopython=True)
def velocity_moments(velocities: ndarray, species_id: ndarray, 
                    num_species: int, max_moments: int = 4) -> ndarray:
    """Calculate velocity moments for each species.
    
    The :math:`n`-th velocity moment for species :math:`s` is calculated as:
    
    .. math::
        \\langle v^n \\rangle_s = \\frac{1}{N_s} \\sum_{i \\in s} |\\vec{v}_i|^n
    
    where :math:`N_s` is the number of particles in species :math:`s`.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
    max_moments : int
        Maximum moment order to calculate (default: 4)
        
    Returns
    -------
    numpy.ndarray
        Velocity moments per species with shape ``(num_species, max_moments)``
        
    Examples
    --------
    Calculate velocity moments:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.transport import velocity_moments
        
        velocities = np.random.randn(1000, 3)
        species_id = np.random.randint(0, 2, 1000)
        
        moments = velocity_moments(velocities, species_id, num_species=2, max_moments=4)
        assert moments.shape == (2, 4)
    """
    # Calculate velocity magnitudes
    speed = sqrt(sum(velocities**2, axis=1))
    
    # Initialize result array
    result = zeros((num_species, max_moments))
    
    # Calculate moments for each species
    for sp in range(num_species):
        # Get particles belonging to this species
        mask = species_id == sp
        if any(mask):
            species_speeds = speed[mask]
            species_count = sum(mask)
            
            # Calculate moments
            for moment in range(max_moments):
                result[sp, moment] = sum(species_speeds**(moment + 1)) / species_count
    
    return result