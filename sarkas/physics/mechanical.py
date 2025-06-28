"""
Mechanical properties module for Sarkas plasma simulation.

This module contains pure, optimized functions for mechanical properties calculations
extracted from the Particles class. All functions are designed to be:
- Pure functions with no side effects
- Numba-compiled for performance
- Compatible with species aggregation
- Well-documented with MyST format
"""

import numpy as np
from numba import jit
from numpy import ndarray
from .aggregation import species_sum, fast_species_sum


@jit(nopython=True)
def momentum(velocities: ndarray, masses: ndarray) -> ndarray:
    """Calculate momentum per particle.
    
    The momentum of each particle is calculated using the classical formula:
    
    .. math::
        \\vec{p}_i = m_i \\vec{v}_i
    
    where :math:`m_i` is the mass and :math:`\\vec{v}_i` is the velocity vector
    of particle :math:`i`.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)`` where N is the number of particles
    masses : numpy.ndarray  
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle momentum with shape ``(N, 3)``
        
    Examples
    --------
    Calculate momentum for a system of particles:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import momentum
        
        # Generate random velocities and masses
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        # Calculate momentum
        p = momentum(velocities, masses)
        
        # Verify results
        assert p.shape == (1000, 3)
        assert np.allclose(p, velocities * masses[:, None])
    """
    return masses[:, np.newaxis] * velocities


@jit(nopython=True)
def center_of_mass_velocity(velocities: ndarray, masses: ndarray) -> ndarray:
    """Calculate center of mass velocity.
    
    The center of mass velocity is calculated as:
    
    .. math::
        \\vec{v}_{\\text{CM}} = \\frac{\\sum_i m_i \\vec{v}_i}{\\sum_i m_i}
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Center of mass velocity with shape ``(3,)``
        
    Examples
    --------
    Calculate center of mass velocity:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import center_of_mass_velocity
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        v_cm = center_of_mass_velocity(velocities, masses)
        assert v_cm.shape == (3,)
    """
    total_mass = np.sum(masses)
    weighted_velocities = masses[:, np.newaxis] * velocities
    return np.sum(weighted_velocities, axis=0) / total_mass


@jit(nopython=True)
def remove_center_of_mass_motion(velocities: ndarray, masses: ndarray) -> ndarray:
    """Remove center of mass motion from velocities.
    
    This function subtracts the center of mass velocity from all particle
    velocities, resulting in a system with zero total momentum:
    
    .. math::
        \\vec{v}'_i = \\vec{v}_i - \\vec{v}_{\\text{CM}}
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Velocities with center of mass motion removed, shape ``(N, 3)``
        
    Examples
    --------
    Remove center of mass motion:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import remove_center_of_mass_motion
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        v_prime = remove_center_of_mass_motion(velocities, masses)
        
        # Verify total momentum is zero
        total_momentum = (v_prime * masses[:, None]).sum(axis=0)
        assert np.allclose(total_momentum, 0.0, atol=1e-12)
    """
    v_cm = center_of_mass_velocity(velocities, masses)
    return velocities - v_cm


@jit(nopython=True)
def angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray) -> ndarray:
    """Calculate angular momentum per particle.
    
    The angular momentum of each particle is calculated as:
    
    .. math::
        \\vec{L}_i = m_i \\vec{r}_i \\times \\vec{v}_i
    
    where :math:`\\vec{r}_i` is the position vector and :math:`\\vec{v}_i` is
    the velocity vector of particle :math:`i`.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle angular momentum with shape ``(N, 3)``
        
    Examples
    --------
    Calculate angular momentum:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import angular_momentum
        
        positions = np.random.randn(1000, 3)
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        L = angular_momentum(positions, velocities, masses)
        assert L.shape == (1000, 3)
    """
    # Calculate cross product for each particle
    L = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        L[i] = masses[i] * np.cross(positions[i], velocities[i])
    return L


@jit(nopython=True)
def total_angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray) -> ndarray:
    """Calculate total angular momentum of the system.
    
    The total angular momentum is the sum of all individual angular momenta:
    
    .. math::
        \\vec{L}_{\\text{total}} = \\sum_i \\vec{L}_i
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Total angular momentum with shape ``(3,)``
        
    Examples
    --------
    Calculate total angular momentum:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import total_angular_momentum
        
        positions = np.random.randn(1000, 3)
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        L_total = total_angular_momentum(positions, velocities, masses)
        assert L_total.shape == (3,)
    """
    L_per_particle = angular_momentum(positions, velocities, masses)
    return np.sum(L_per_particle, axis=0)


@jit(nopython=True)
def force_torque(positions: ndarray, forces: ndarray) -> ndarray:
    """Calculate torque per particle.
    
    The torque on each particle is calculated as:
    
    .. math::
        \\vec{\\tau}_i = \\vec{r}_i \\times \\vec{F}_i
    
    where :math:`\\vec{r}_i` is the position vector and :math:`\\vec{F}_i` is
    the force vector on particle :math:`i`.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    forces : numpy.ndarray
        Particle forces with shape ``(N, 3)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle torque with shape ``(N, 3)``
        
    Examples
    --------
    Calculate torque:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import force_torque
        
        positions = np.random.randn(1000, 3)
        forces = np.random.randn(1000, 3)
        
        tau = force_torque(positions, forces)
        assert tau.shape == (1000, 3)
    """
    # Calculate cross product for each particle
    tau = np.zeros_like(positions)
    for i in range(positions.shape[0]):
        tau[i] = np.cross(positions[i], forces[i])
    return tau


@jit(nopython=True)
def stress_tensor(positions: ndarray, forces: ndarray, volume: float) -> ndarray:
    """Calculate stress tensor from forces.
    
    The stress tensor is calculated as:
    
    .. math::
        \\sigma_{\\alpha\\beta} = -\\frac{1}{V} \\sum_i r_{i,\\alpha} F_{i,\\beta}
    
    where :math:`V` is the system volume, :math:`r_{i,\\alpha}` is the
    :math:`\\alpha`-component of the position of particle :math:`i`, and
    :math:`F_{i,\\beta}` is the :math:`\\beta`-component of the force on particle :math:`i`.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    forces : numpy.ndarray
        Particle forces with shape ``(N, 3)``
    volume : float
        System volume
        
    Returns
    -------
    numpy.ndarray
        Stress tensor with shape ``(3, 3)``
        
    Examples
    --------
    Calculate stress tensor:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import stress_tensor
        
        positions = np.random.randn(1000, 3)
        forces = np.random.randn(1000, 3)
        volume = 1000.0
        
        sigma = stress_tensor(positions, forces, volume)
        assert sigma.shape == (3, 3)
    """
    sigma = np.zeros((3, 3))
    for i in range(positions.shape[0]):
        for alpha in range(3):
            for beta in range(3):
                sigma[alpha, beta] -= positions[i, alpha] * forces[i, beta]
    return sigma / volume


# Species-level mechanical calculations
def species_momentum(velocities: ndarray, masses: ndarray, species_id: ndarray, 
                    num_species: int) -> ndarray:
    """Calculate total momentum for each species.
    
    The total momentum for species :math:`s` is calculated as:
    
    .. math::
        \\vec{P}_s = \\sum_{i \\in s} m_i \\vec{v}_i
    
    where the sum is over all particles belonging to species :math:`s`.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Total momentum per species with shape ``(num_species, 3)``
        
    Examples
    --------
    Calculate species momenta:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import species_momentum
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        P_species = species_momentum(velocities, masses, species_id, num_species=2)
        assert P_species.shape == (2, 3)
    """
    # Calculate momentum per particle first
    p_per_particle = momentum(velocities, masses)
    # Aggregate by species using fast vector aggregation
    return fast_species_sum(p_per_particle, species_id, num_species)


def species_angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray,
                           species_id: ndarray, num_species: int) -> ndarray:
    """Calculate total angular momentum for each species.
    
    The total angular momentum for species :math:`s` is calculated as:
    
    .. math::
        \\vec{L}_s = \\sum_{i \\in s} m_i \\vec{r}_i \\times \\vec{v}_i
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Total angular momentum per species with shape ``(num_species, 3)``
        
    Examples
    --------
    Calculate species angular momenta:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import species_angular_momentum
        
        positions = np.random.randn(1000, 3)
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        L_species = species_angular_momentum(positions, velocities, masses,
                                           species_id, num_species=2)
        assert L_species.shape == (2, 3)
    """
    # Calculate angular momentum per particle first
    L_per_particle = angular_momentum(positions, velocities, masses)
    # Aggregate by species using fast vector aggregation
    return fast_species_sum(L_per_particle, species_id, num_species)


def species_center_of_mass_velocity(velocities: ndarray, masses: ndarray,
                                  species_id: ndarray, num_species: int) -> ndarray:
    """Calculate center of mass velocity for each species.
    
    The center of mass velocity for species :math:`s` is calculated as:
    
    .. math::
        \\vec{v}_{\\text{CM},s} = \\frac{\\sum_{i \\in s} m_i \\vec{v}_i}{\\sum_{i \\in s} m_i}
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Center of mass velocity per species with shape ``(num_species, 3)``
        
    Examples
    --------
    Calculate species center of mass velocities:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import species_center_of_mass_velocity
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        v_cm_species = species_center_of_mass_velocity(velocities, masses,
                                                     species_id, num_species=2)
        assert v_cm_species.shape == (2, 3)
    """
    v_cm_species = np.zeros((num_species, 3))
    
    for s in range(num_species):
        mask = species_id == s
        if np.any(mask):
            species_velocities = velocities[mask]
            species_masses = masses[mask]
            v_cm_species[s] = center_of_mass_velocity(species_velocities, species_masses)
    
    return v_cm_species


# Conservation and validation functions
def validate_momentum_conservation(velocities: ndarray, masses: ndarray, 
                                 tolerance: float = 1e-12) -> dict:
    """Validate momentum conservation.
    
    This function checks if the total momentum is conserved within the
    specified tolerance.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    tolerance : float
        Tolerance for conservation check
        
    Returns
    -------
    dict
        Dictionary containing total momentum and conservation status
        
    Examples
    --------
    Validate momentum conservation:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import validate_momentum_conservation
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        result = validate_momentum_conservation(velocities, masses)
        print(f"Total momentum: {result['total_momentum']}")
        print(f"Conserved: {result['conserved']}")
    """
    total_momentum = np.sum(momentum(velocities, masses), axis=0)
    conserved = np.allclose(total_momentum, 0.0, atol=tolerance)
    
    return {
        'total_momentum': total_momentum,
        'conserved': conserved,
        'magnitude': np.linalg.norm(total_momentum)
    }


def validate_angular_momentum_conservation(positions: ndarray, velocities: ndarray,
                                         masses: ndarray, tolerance: float = 1e-12) -> dict:
    """Validate angular momentum conservation.
    
    This function checks if the total angular momentum is conserved within the
    specified tolerance.
    
    Parameters
    ----------
    positions : numpy.ndarray
        Particle positions with shape ``(N, 3)``
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    tolerance : float
        Tolerance for conservation check
        
    Returns
    -------
    dict
        Dictionary containing total angular momentum and conservation status
        
    Examples
    --------
    Validate angular momentum conservation:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.mechanical import validate_angular_momentum_conservation
        
        positions = np.random.randn(1000, 3)
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        result = validate_angular_momentum_conservation(positions, velocities, masses)
        print(f"Total angular momentum: {result['total_angular_momentum']}")
        print(f"Conserved: {result['conserved']}")
    """
    total_angular_momentum_vec = total_angular_momentum(positions, velocities, masses)
    conserved = np.allclose(total_angular_momentum_vec, 0.0, atol=tolerance)
    
    return {
        'total_angular_momentum': total_angular_momentum_vec,
        'conserved': conserved,
        'magnitude': np.linalg.norm(total_angular_momentum_vec)
    }


# Backward compatibility wrappers
def calculate_momentum(velocities: ndarray, masses: ndarray) -> ndarray:
    """Backward compatibility wrapper for momentum calculation.
    
    This function maintains the same interface as the original Particles
    class method for backward compatibility.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle momentum with shape ``(N, 3)``
    """
    return momentum(velocities, masses)


def calculate_center_of_mass_velocity(velocities: ndarray, masses: ndarray) -> ndarray:
    """Backward compatibility wrapper for center of mass velocity calculation.
    
    This function maintains the same interface as the original Particles
    class method for backward compatibility.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Center of mass velocity with shape ``(3,)``
    """
    return center_of_mass_velocity(velocities, masses) 