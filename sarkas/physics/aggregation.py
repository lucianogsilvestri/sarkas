"""
Species aggregation utilities for Sarkas physics calculations.

This module provides optimized, pure functions for aggregating per-particle
quantities by species. These utilities are designed to be highly efficient
and reusable across all physics modules.
"""

import numpy as np  # type: ignore
from numba import jit, njit  # type: ignore


@jit(nopython=True)
def species_sum(per_particle_array: np.ndarray, species_id: np.ndarray, 
                num_species: int) -> np.ndarray:
    """Calculate sum for each species.
    
    The sum for species :math:`s` is calculated as:
    
    .. math::
        S_s = \\sum_{i \\in s} a_i
    
    where :math:`a_i` is the per-particle quantity for particle :math:`i`.
    
    Parameters
    ----------
    per_particle_array : numpy.ndarray
        Per-particle quantities with shape ``(N,)`` or ``(N, ...)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Sum per species with shape ``(num_species,)`` or ``(num_species, ...)``
        
    Examples
    --------
    Calculate species sums:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import species_sum
        
        # Scalar quantities
        masses = np.random.rand(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        total_mass = species_sum(masses, species_id, num_species=2)
        assert total_mass.shape == (2,)
        
        # Vector quantities
        velocities = np.random.randn(1000, 3)
        total_velocity = species_sum(velocities, species_id, num_species=2)
        assert total_velocity.shape == (2, 3)
        
        # Tensor quantities
        stress_tensors = np.random.randn(1000, 3, 3)
        total_stress = species_sum(stress_tensors, species_id, num_species=2)
        assert total_stress.shape == (2, 3, 3)
    """

    if per_particle_array.ndim == 1:
        # Scalar quantities
        result = np.zeros(num_species, dtype=per_particle_array.dtype)
        for i in range(len(per_particle_array)):
            sp_id = int(species_id[i])
            result[sp_id] += per_particle_array[i]
    else:
        # Vector/tensor quantities
        shape = (num_species,) + per_particle_array.shape[1:]
        result = np.zeros(shape, dtype=per_particle_array.dtype)
        for i in range(len(per_particle_array)):
            sp_id = int(species_id[i])
            result[sp_id] += per_particle_array[i]
    
    return result


@jit(nopython=True)
def species_mean(per_particle_array: np.ndarray, species_id: np.ndarray, 
                num_species: int) -> np.ndarray:
    """Calculate mean for each species.
    
    The mean for species :math:`s` is calculated as:
    
    .. math::
        \\langle a \\rangle_s = \\frac{1}{N_s} \\sum_{i \\in s} a_i
    
    where :math:`N_s` is the number of particles in species :math:`s`.
    
    Parameters
    ----------
    per_particle_array : numpy.ndarray
        Per-particle quantities with shape ``(N,)`` or ``(N, ...)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Mean per species with shape ``(num_species,)`` or ``(num_species, ...)``
        
    Examples
    --------
    Calculate species means:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import species_mean
        
        velocities = np.random.randn(1000, 3)
        species_id = np.random.randint(0, 2, 1000)
        
        mean_velocity = species_mean(velocities, species_id, num_species=2)
        assert mean_velocity.shape == (2, 3)
    """
    if num_species == 0 or per_particle_array.size == 0 or species_id.size == 0:
        if per_particle_array.ndim == 1:
            return np.zeros(num_species, dtype=per_particle_array.dtype)
        else:
            shape = (num_species,) + per_particle_array.shape[1:]
            return np.zeros(shape, dtype=per_particle_array.dtype)
    # Get species counts
    species_counts = get_species_counts(species_id, num_species)
    
    # Calculate sums
    sums = species_sum(per_particle_array, species_id, num_species)
    
    # Calculate means, avoiding division by zero
    if per_particle_array.ndim == 1:
        result = np.zeros(num_species, dtype=per_particle_array.dtype)
        for i in range(num_species):
            if species_counts[i] > 0:
                result[i] = sums[i] / species_counts[i]
    else:
        shape = (num_species,) + per_particle_array.shape[1:]
        result = np.zeros(shape, dtype=per_particle_array.dtype)
        for i in range(num_species):
            if species_counts[i] > 0:
                result[i] = sums[i] / species_counts[i]
    
    return result


@jit(nopython=True)
def species_weighted_sum(per_particle_array: np.ndarray, weights: np.ndarray, 
                        species_id: np.ndarray, num_species: int) -> np.ndarray:
    """Calculate weighted sum for each species.
    
    The weighted sum for species :math:`s` is calculated as:
    
    .. math::
        S_s = \\sum_{i \\in s} w_i a_i
    
    where :math:`w_i` is the weight for particle :math:`i`.
    
    Parameters
    ----------
    per_particle_array : numpy.ndarray
        Per-particle quantities with shape ``(N,)`` or ``(N, ...)``
    weights : numpy.ndarray
        Weights with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Weighted sum per species with shape ``(num_species,)`` or ``(num_species, ...)``
        
    Examples
    --------
    Calculate species weighted sums:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import species_weighted_sum
        
        energies = np.random.rand(1000)
        masses = np.random.rand(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        weighted_energy = species_weighted_sum(energies, masses, species_id, num_species=2)
        assert weighted_energy.shape == (2,)
    """
    if per_particle_array.ndim == 1:
        # Scalar quantities
        result = np.zeros(num_species, dtype=per_particle_array.dtype)
        for i in range(len(per_particle_array)):
            sp_id = species_id[i]
            result[sp_id] += weights[i] * per_particle_array[i]
    else:
        # Vector/tensor quantities
        shape = (num_species,) + per_particle_array.shape[1:]
        result = np.zeros(shape, dtype=per_particle_array.dtype)
        for i in range(len(per_particle_array)):
            sp_id = species_id[i]
            result[sp_id] += weights[i] * per_particle_array[i]
    
    return result


@jit(nopython=True)
def species_variance(per_particle_array: np.ndarray, species_id: np.ndarray, 
                    num_species: int) -> np.ndarray:
    """Calculate variance for each species.
    
    The variance for species :math:`s` is calculated as:
    
    .. math::
        \\sigma_s^2 = \\frac{1}{N_s-1} \\sum_{i \\in s} (a_i - \\langle a \\rangle_s)^2
    
    Parameters
    ----------
    per_particle_array : numpy.ndarray
        Per-particle quantities with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Variance per species with shape ``(num_species,)``
        
    Examples
    --------
    Calculate species variances:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import species_variance
        
        velocities = np.random.randn(1000, 3)
        speed = np.linalg.norm(velocities, axis=1)
        species_id = np.random.randint(0, 2, 1000)
        
        speed_variance = species_variance(speed, species_id, num_species=2)
        assert speed_variance.shape == (2,)
    """
    # Calculate means first
    means = species_mean(per_particle_array, species_id, num_species)
    
    # Calculate squared deviations
    squared_deviations = (per_particle_array - means[species_id])**2
    
    # Calculate variance using species_sum
    sums = species_sum(squared_deviations, species_id, num_species)
    species_counts = get_species_counts(species_id, num_species)
    
    result = np.zeros(num_species, dtype=per_particle_array.dtype)
    for i in range(num_species):
        if species_counts[i] > 1:
            result[i] = sums[i] / (species_counts[i] - 1)
    
    return result


@jit(nopython=True)
def get_species_counts(species_id: np.ndarray, num_species: int) -> np.ndarray:
    """Get the number of particles per species.
    
    Parameters
    ----------
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Number of particles per species with shape ``(num_species,)``
        
    Examples
    --------
    Get species counts:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import get_species_counts
        
        species_id = np.random.randint(0, 2, 1000)
        
        counts = get_species_counts(species_id, num_species=2)
        assert counts.shape == (2,)
        assert counts.sum() == 1000
    """
    return np.bincount(species_id, minlength=num_species)


def validate_species_aggregation(per_particle_array: np.ndarray, species_id: np.ndarray, 
                               num_species: int) -> bool:
    """Validate inputs for species aggregation.
    
    This function performs various checks to ensure the inputs are valid
    for species aggregation:
    
    - Array shapes are compatible
    - Species IDs are within valid range
    - No NaN or infinite values
    
    Parameters
    ----------
    per_particle_array : numpy.ndarray
        Per-particle quantities
    species_id : numpy.ndarray
        Species identifier for each particle
    num_species : int
        Number of species
        
    Returns
    -------
    bool
        True if inputs are valid for species aggregation
        
    Examples
    --------
    Validate species aggregation inputs:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.aggregation import validate_species_aggregation
        
        velocities = np.random.randn(1000, 3)
        species_id = np.random.randint(0, 2, 1000)
        
        is_valid = validate_species_aggregation(velocities, species_id, num_species=2)
        assert is_valid
    """
    # Check array shapes
    if per_particle_array.shape[0] != species_id.shape[0]:
        return False
    
    # Check species ID range
    if np.any(species_id < 0) or np.any(species_id >= num_species):
        return False
    
    # Check for NaN or infinite values
    if np.any(np.isnan(per_particle_array)) or np.any(np.isinf(per_particle_array)):
        return False
    
    return True


# Backward compatibility aliases
def species_vector_sum(per_particle_vectors: np.ndarray, species_id: np.ndarray, 
                      num_species: int) -> np.ndarray:
    """Backward compatibility alias for species_sum.
    
    This function is deprecated. Use species_sum instead.
    """
    return species_sum(per_particle_vectors, species_id, num_species)


def species_tensor_sum(per_particle_tensors: np.ndarray, species_id: np.ndarray, 
                      num_species: int) -> np.ndarray:
    """Backward compatibility alias for species_sum.
    
    This function is deprecated. Use species_sum instead.
    """
    return species_sum(per_particle_tensors, species_id, num_species)


# Legacy functions for backward compatibility
def scalar_species_loop(observable: np.ndarray, species_num: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility.
    
    This function is deprecated. Use species_sum instead.
    """
    num_species = len(species_num)
    result = np.zeros(num_species, dtype=observable.dtype)
    
    start_idx = 0
    for sp in range(num_species):
        end_idx = start_idx + species_num[sp]
        result[sp] = observable[start_idx:end_idx].sum()
        start_idx = end_idx
    
    return result


def vector_species_loop(observable: np.ndarray, species_num: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility.
    
    This function is deprecated. Use species_sum instead.
    """
    num_species = len(species_num)
    result = np.zeros((num_species, observable.shape[1]), dtype=observable.dtype)
    
    start_idx = 0
    for sp in range(num_species):
        end_idx = start_idx + species_num[sp]
        result[sp] = observable[start_idx:end_idx].sum(axis=0)
        start_idx = end_idx
    
    return result


def tensor_species_loop(observable: np.ndarray, species_num: np.ndarray) -> np.ndarray:
    """Legacy function for backward compatibility.
    
    This function is deprecated. Use species_sum instead.
    """
    num_species = len(species_num)
    result = np.zeros((num_species,) + observable.shape[1:], dtype=observable.dtype)
    
    start_idx = 0
    for sp in range(num_species):
        end_idx = start_idx + species_num[sp]
        result[sp] = observable[start_idx:end_idx].sum(axis=0)
        start_idx = end_idx
    
    return result


@njit
def bincount_vector(species_id, vectors, num_species):
    result = np.zeros((num_species, vectors.shape[1]), dtype=vectors.dtype)
    for j in range(vectors.shape[1]):
        result[:, j] = np.bincount(species_id, weights=vectors[:, j], minlength=num_species)
    return result


@njit
def bincount_tensor(species_id, tensors, num_species):
    result = np.zeros((num_species, tensors.shape[1], tensors.shape[2]), dtype=tensors.dtype)
    for i in range(tensors.shape[1]):
        for j in range(tensors.shape[2]):
            result[:, i, j] = np.bincount(species_id, weights=tensors[:, i, j], minlength=num_species)
    return result


def fast_species_sum(per_particle_array, species_id, num_species):
    """
    Fast species aggregation for vectors/tensors using Numba and bincount.
    Use species_sum for scalars, fast_species_sum for vectors/tensors.
    """
    if per_particle_array.ndim == 1:
        raise ValueError("Use species_sum for scalar aggregation.")
    elif per_particle_array.ndim == 2:
        return bincount_vector(species_id, per_particle_array, num_species)
    elif per_particle_array.ndim == 3:
        return bincount_tensor(species_id, per_particle_array, num_species)
    else:
        raise ValueError("Unsupported array shape for fast_species_sum.") 