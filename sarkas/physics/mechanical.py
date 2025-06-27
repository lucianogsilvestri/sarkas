"""
Mechanical property calculations for molecular dynamics simulations.

This module provides optimized functions for calculating mechanical properties
including pressure, stress, momentum, and virial tensors. All functions are
compiled with Numba for high performance.

Physical Background
------------------
Mechanical properties describe the response of matter to applied forces and
constraints. Key quantities include:

- Pressure: Force per unit area, related to particle collisions with boundaries
- Stress tensor: Generalization of pressure including shear components
- Momentum: Linear and angular momentum conservation
- Virial: Work done against interparticle forces

The virial theorem connects kinetic and potential energies:
<T> = -(1/2) * <r · ∇U>

Where T is kinetic energy, r is position, and U is potential energy.

Notes
-----
All functions use SI units unless otherwise specified:
- Positions: m
- Velocities: m/s
- Masses: kg
- Forces: N
- Pressures: Pa (N/m²)
- Volumes: m³
"""

import numpy as np
from numba import jit, prange
from numpy import ndarray
from typing import Tuple, Optional

# Physical constants
_BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


@jit(nopython=True)
def _momentum_kernel(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Numba-compiled kernel for momentum calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,).
        
    Returns
    -------
    ndarray
        Momentum of each particle. Shape: (N, 3).
    """
    n_particles = velocities.shape[0]
    n_dimensions = velocities.shape[1]
    momenta = np.zeros((n_particles, n_dimensions))
    
    for i in range(n_particles):
        for j in range(n_dimensions):
            momenta[i, j] = masses[i] * velocities[i, j]
    
    return momenta

def momentum(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Calculate the momentum of particles.
    
    Uses Newton's definition of momentum:
    p = m * v
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
        
    Returns
    -------
    ndarray
        Momentum of each particle. Shape: (N, 3).
        Units: kg⋅m/s
        
    Examples
    --------
    >>> import numpy as np
    >>> # Single particle
    >>> vel = np.array([[10.0, 5.0, 0.0]])
    >>> mass = np.array([2.0])
    >>> p = momentum(vel, mass)
    >>> print(f"Momentum: {p[0]}")  # [20.0, 10.0, 0.0]
    
    >>> # Multiple particles
    >>> vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    >>> mass = np.array([1.0, 2.0])
    >>> p = momentum(vel, mass)
    >>> print(f"Momenta: {p}")
    
    Notes
    -----
    - Handles both uniform and non-uniform masses
    - Optimized with Numba compilation
    - Vector quantity preserving all components
    """
    if velocities.size == 0:
        return np.array([]).reshape(0, 3)
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(velocities.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    return _momentum_kernel(velocities, masses_broadcast)


@jit(nopython=True)
def _center_of_mass_velocity_kernel(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Numba-compiled kernel for center of mass velocity calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,).
        
    Returns
    -------
    ndarray
        Center of mass velocity. Shape: (3,).
    """
    total_mass = 0.0
    cm_velocity = np.zeros(3)
    
    for i in range(velocities.shape[0]):
        total_mass += masses[i]
        for j in range(3):
            cm_velocity[j] += masses[i] * velocities[i, j]
    
    if total_mass > 0.0:
        for j in range(3):
            cm_velocity[j] /= total_mass
    
    return cm_velocity


def center_of_mass_velocity(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Calculate the center of mass velocity.
    
    Uses the definition:
    v_cm = Σ(m_i * v_i) / Σ(m_i)
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
        
    Returns
    -------
    ndarray
        Center of mass velocity. Shape: (3,).
        Units: m/s
        
    Examples
    --------
    >>> import numpy as np
    >>> # Two particles with different masses
    >>> vel = np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    >>> mass = np.array([1.0, 2.0])
    >>> v_cm = center_of_mass_velocity(vel, mass)
    >>> print(f"CM velocity: {v_cm}")  # [0.0, 0.0, 0.0]
    
    >>> # Three particles
    >>> vel = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    >>> mass = np.array([1.0, 1.0, 1.0])
    >>> v_cm = center_of_mass_velocity(vel, mass)
    >>> print(f"CM velocity: {v_cm}")  # [2.0, 0.0, 0.0]
    
    Notes
    -----
    - Handles both uniform and non-uniform masses
    - Returns zero if total mass is zero
    - Optimized with Numba compilation
    """
    if velocities.size == 0:
        return np.zeros(3)
    
    if masses.size == 1:
        # For uniform mass, center of mass velocity is just the mean
        return np.mean(velocities, axis=0)
    
    return _center_of_mass_velocity_kernel(velocities, masses)


@jit(nopython=True)
def _remove_cm_motion_kernel(velocities: ndarray, masses: ndarray, 
                            cm_velocity: ndarray) -> ndarray:
    """
    Numba-compiled kernel for removing center of mass motion.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,).
    cm_velocity : ndarray
        Center of mass velocity. Shape: (3,).
        
    Returns
    -------
    ndarray
        Corrected velocities. Shape: (N, 3).
    """
    n_particles = velocities.shape[0]
    corrected_velocities = np.zeros_like(velocities)
    
    for i in range(n_particles):
        for j in range(3):
            corrected_velocities[i, j] = velocities[i, j] - cm_velocity[j]
    
    return corrected_velocities


def remove_center_of_mass_motion(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Remove center of mass motion from particle velocities.
    
    Subtracts the center of mass velocity from all particles:
    v'_i = v_i - v_cm
    
    This enforces conservation of total momentum (total momentum = 0).
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        
    Returns
    -------
    ndarray
        Corrected velocities with center of mass motion removed. Shape: (N, 3).
        Units: m/s
        
    Examples
    --------
    >>> import numpy as np
    >>> # System with net momentum
    >>> vel = np.array([[10.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    >>> mass = np.array([1.0, 1.0])
    >>> vel_corrected = remove_center_of_mass_motion(vel, mass)
    >>> print(f"Original: {vel}")
    >>> print(f"Corrected: {vel_corrected}")
    >>> # Total momentum should be zero after correction
    >>> p_total = momentum(vel_corrected, mass).sum(axis=0)
    >>> print(f"Total momentum: {p_total}")  # Should be [0, 0, 0]
    
    Notes
    -----
    - Essential for maintaining conservation laws
    - Handles both uniform and non-uniform masses
    - Optimized with Numba compilation
    - Creates new array (does not modify input)
    """
    if velocities.size == 0:
        return np.array([]).reshape(0, 3)
    
    if masses.size == 1:
        masses_broadcast = np.full(velocities.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    # Calculate center of mass velocity
    cm_vel = _center_of_mass_velocity_kernel(velocities, masses_broadcast)
    
    # Remove center of mass motion
    return _remove_cm_motion_kernel(velocities, masses_broadcast, cm_vel)


def momentum_vector(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Calculate total momentum vector of the system.
    
    The total momentum is given by:
    P = Σᵢ mᵢ * vᵢ
    
    where mᵢ is the mass and vᵢ is the velocity of particle i.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
        
    Returns
    -------
    ndarray
        Total momentum vector. Shape: (3,).
        Units: kg⋅m/s
        
    Examples
    --------
    >>> import numpy as np
    >>> # Two particles moving in opposite directions
    >>> vel = np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    >>> mass = np.array([2.0, 4.0])
    >>> P = momentum_vector(vel, mass)
    >>> print(f"Total momentum: {P}")  # [0.0, 0.0, 0.0]
    
    Notes
    -----
    - For isolated systems, total momentum should be conserved
    - Useful for checking momentum conservation in simulations
    - Can be used to calculate center of mass velocity
    """
    if velocities.size == 0:
        return np.zeros(3)
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(velocities.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    # Calculate individual particle momenta and sum
    particle_momenta = _momentum_kernel(velocities, masses_broadcast)
    return particle_momenta.sum(axis=0)


@jit(nopython=True)
def _angular_momentum_kernel(positions: ndarray, velocities: ndarray, 
                            masses: ndarray, origin: ndarray) -> ndarray:
    """
    Numba-compiled kernel for angular momentum calculation.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions. Shape: (N, 3).
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,).
    origin : ndarray
        Origin point for angular momentum calculation. Shape: (3,).
        
    Returns
    -------
    ndarray
        Angular momentum for each particle. Shape: (N, 3).
    """
    n_particles = positions.shape[0]
    angular_momenta = np.zeros((n_particles, 3))
    
    for i in range(n_particles):
        # Position vector relative to origin
        r = np.zeros(3)
        for j in range(3):
            r[j] = positions[i, j] - origin[j]
        
        # Linear momentum
        p = np.zeros(3)
        for j in range(3):
            p[j] = masses[i] * velocities[i, j]
        
        # Angular momentum L = r × p
        angular_momenta[i, 0] = r[1] * p[2] - r[2] * p[1]  # x-component
        angular_momenta[i, 1] = r[2] * p[0] - r[0] * p[2]  # y-component
        angular_momenta[i, 2] = r[0] * p[1] - r[1] * p[0]  # z-component
    
    return angular_momenta


def angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray,
                    origin: Optional[ndarray] = None) -> ndarray:
    """
    Calculate total angular momentum of the system.
    
    The angular momentum is given by:
    L = Σᵢ rᵢ × (mᵢ * vᵢ)
    
    where rᵢ is the position vector relative to the origin,
    mᵢ is the mass, and vᵢ is the velocity of particle i.
    
    Parameters
    ----------
    positions : ndarray
        Particle positions. Shape: (N, 3).
        Units: m
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
    origin : ndarray, optional
        Origin point for angular momentum calculation. Shape: (3,).
        If None, uses the origin (0, 0, 0).
        Units: m
        
    Returns
    -------
    ndarray
        Total angular momentum vector. Shape: (3,).
        Units: kg⋅m²/s
        
    Examples
    --------
    >>> import numpy as np
    >>> # Particle in circular motion
    >>> pos = np.array([[1.0, 0.0, 0.0]])
    >>> vel = np.array([[0.0, 10.0, 0.0]])  # Tangential velocity
    >>> mass = np.array([2.0])
    >>> L = angular_momentum(pos, vel, mass)
    >>> print(f"Angular momentum: {L}")  # [0, 0, 20] kg⋅m²/s
    
    Notes
    -----
    - Angular momentum is conserved in the absence of external torques
    - The choice of origin affects the magnitude and direction
    - Useful for analyzing rotational motion and stability
    """
    if positions.size == 0 or velocities.size == 0:
        return np.zeros(3)
    
    if origin is None:
        origin = np.zeros(3)
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(positions.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    # Calculate individual particle angular momenta and sum
    particle_angular_momenta = _angular_momentum_kernel(
        positions, velocities, masses_broadcast, origin
    )
    return particle_angular_momenta.sum(axis=0)


@jit(nopython=True)
def _pressure_tensor_kinetic_kernel(velocities: ndarray, masses: ndarray, 
                                   volume: float) -> ndarray:
    """
    Numba-compiled kernel for kinetic pressure tensor calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,).
    volume : float
        System volume.
        
    Returns
    -------
    ndarray
        Kinetic pressure tensor. Shape: (3, 3).
    """
    n_particles = velocities.shape[0]
    pressure_tensor = np.zeros((3, 3))
    
    # Calculate kinetic pressure tensor: P_αβ = (1/V) * Σᵢ mᵢ * vᵢα * vᵢβ
    for i in range(n_particles):
        for α in range(3):
            for β in range(3):
                pressure_tensor[α, β] += masses[i] * velocities[i, α] * velocities[i, β]
    
    # Normalize by volume
    for α in range(3):
        for β in range(3):
            pressure_tensor[α, β] /= volume
    
    return pressure_tensor


def pressure_tensor_kinetic(velocities: ndarray, masses: ndarray, 
                          volume: float) -> ndarray:
    """
    Calculate kinetic contribution to pressure tensor.
    
    The kinetic pressure tensor is given by:
    P_kin_αβ = (1/V) * Σᵢ mᵢ * vᵢα * vᵢβ
    
    This represents the pressure due to particle motion (translational kinetic energy).
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
    volume : float
        System volume.
        Units: m³
        
    Returns
    -------
    ndarray
        Kinetic pressure tensor. Shape: (3, 3).
        Units: Pa (N/m²)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Isotropic velocity distribution
    >>> vel = np.random.normal(0, 100, (1000, 3))  # Random velocities
    >>> mass = np.ones(1000) * 1e-26  # Uniform mass
    >>> P_kin = pressure_tensor_kinetic(vel, mass, volume=1e-15)
    >>> print(f"Diagonal pressure: {np.diag(P_kin)}")
    
    Notes
    -----
    - Diagonal elements represent normal pressure
    - Off-diagonal elements represent shear stress
    - For isotropic systems, off-diagonal elements average to zero
    - Related to temperature via ideal gas law: P = nkT
    """
    if velocities.size == 0:
        return np.zeros((3, 3))
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(velocities.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    return _pressure_tensor_kinetic_kernel(velocities, masses_broadcast, volume)


def pressure_tensor_virial(virial_tensor: ndarray, volume: float) -> ndarray:
    """
    Calculate virial contribution to pressure tensor.
    
    The virial pressure tensor is given by:
    P_vir = (1/V) * W
    
    where W is the virial tensor representing work done against interparticle forces.
    
    Parameters
    ----------
    virial_tensor : ndarray
        Virial tensor from force calculations. Shape: (3, 3) or (N_species, N_species, 3, 3).
        Units: J (energy units)
    volume : float
        System volume.
        Units: m³
        
    Returns
    -------
    ndarray
        Virial pressure tensor. Shape: (3, 3).
        Units: Pa (N/m²)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Isotropic virial (like ideal gas)
    >>> virial = np.diag([1e-15, 1e-15, 1e-15])  # J
    >>> P_vir = pressure_tensor_virial(virial, volume=1e-18)
    >>> print(f"Virial pressure: {np.diag(P_vir)}")  # Pa
    
    Notes
    -----
    - Represents pressure due to interparticle interactions
    - Combined with kinetic pressure gives total pressure
    - Virial theorem relates this to potential energy
    - Sign convention: repulsive forces give positive pressure
    """
    if virial_tensor.size == 0:
        return np.zeros((3, 3))
    
    # Handle different virial tensor shapes
    if virial_tensor.ndim == 4:
        # Sum over species pairs for multi-species systems
        total_virial = virial_tensor.sum(axis=(0, 1))
    elif virial_tensor.ndim == 3:
        # Sum over species
        total_virial = virial_tensor.sum(axis=0)
    else:
        # Single tensor
        total_virial = virial_tensor
    
    return total_virial / volume


def pressure_scalar(pressure_tensor: ndarray, dimensions: int = 3) -> float:
    """
    Calculate scalar pressure from pressure tensor.
    
    The scalar pressure is the trace of the pressure tensor:
    P = (1/d) * Tr(P_tensor) = (1/d) * Σᵢ P_ii
    
    where d is the number of dimensions.
    
    Parameters
    ----------
    pressure_tensor : ndarray
        Pressure tensor. Shape: (3, 3).
        Units: Pa
    dimensions : int, optional
        Number of dimensions for averaging. Default: 3.
        
    Returns
    -------
    float
        Scalar pressure.
        Units: Pa
        
    Examples
    --------
    >>> import numpy as np
    >>> # Isotropic pressure tensor
    >>> P_tensor = np.diag([1e5, 1e5, 1e5])  # 1 bar each direction
    >>> P_scalar = pressure_scalar(P_tensor)
    >>> print(f"Scalar pressure: {P_scalar:.0f} Pa")  # 1e5 Pa
    
    Notes
    -----
    - For isotropic systems, all diagonal elements are equal
    - Off-diagonal elements don't contribute to scalar pressure
    - This is the pressure that would be measured by a pressure gauge
    """
    if pressure_tensor.size == 0:
        return 0.0
    
    # Calculate trace (sum of diagonal elements)
    trace = np.trace(pressure_tensor)
    
    return trace / dimensions


@jit(nopython=True)
def _stress_tensor_kernel(forces: ndarray, positions: ndarray, 
                         volume: float) -> ndarray:
    """
    Numba-compiled kernel for stress tensor calculation.
    
    Parameters
    ----------
    forces : ndarray
        Forces on particles. Shape: (N, 3).
    positions : ndarray
        Particle positions. Shape: (N, 3).
    volume : float
        System volume.
        
    Returns
    -------
    ndarray
        Stress tensor. Shape: (3, 3).
    """
    n_particles = forces.shape[0]
    stress_tensor = np.zeros((3, 3))
    
    # Calculate stress tensor: σ_αβ = -(1/V) * Σᵢ Fᵢα * rᵢβ
    for i in range(n_particles):
        for α in range(3):
            for β in range(3):
                stress_tensor[α, β] -= forces[i, α] * positions[i, β]
    
    # Normalize by volume
    for α in range(3):
        for β in range(3):
            stress_tensor[α, β] /= volume
    
    return stress_tensor


def stress_tensor(forces: ndarray, positions: ndarray, volume: float) -> ndarray:
    """
    Calculate stress tensor from forces and positions.
    
    The stress tensor is given by:
    σ_αβ = -(1/V) * Σᵢ Fᵢα * rᵢβ
    
    This represents the internal stress state of the material.
    
    Parameters
    ----------
    forces : ndarray
        Forces on particles. Shape: (N, 3).
        Units: N
    positions : ndarray
        Particle positions. Shape: (N, 3).
        Units: m
    volume : float
        System volume.
        Units: m³
        
    Returns
    -------
    ndarray
        Stress tensor. Shape: (3, 3).
        Units: Pa (N/m²)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Uniform compression
    >>> forces = np.array([[1e-12, 0, 0], [-1e-12, 0, 0]])  # N
    >>> positions = np.array([[1e-9, 0, 0], [-1e-9, 0, 0]])  # m
    >>> sigma = stress_tensor(forces, positions, volume=1e-24)
    >>> print(f"Stress tensor:\\n{sigma}")
    
    Notes
    -----
    - Positive values indicate tensile stress
    - Negative values indicate compressive stress
    - Diagonal elements are normal stresses
    - Off-diagonal elements are shear stresses
    - Related to pressure tensor but includes spatial correlations
    """
    if forces.size == 0 or positions.size == 0:
        return np.zeros((3, 3))
    
    return _stress_tensor_kernel(forces, positions, volume)


@jit(nopython=True)
def _species_pressure_kernel(velocities: ndarray, masses: ndarray,
                            virial_tensor: ndarray, species_start: ndarray,
                            species_num: ndarray, volume: float) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Numba-compiled kernel for species-specific pressure calculations.
    
    Parameters
    ----------
    velocities : ndarray
        All particle velocities. Shape: (N, 3).
    masses : ndarray
        All particle masses. Shape: (N,).
    virial_tensor : ndarray
        Virial tensor. Shape: (N_species, N_species, 3, 3).
    species_start : ndarray
        Starting indices for each species. Shape: (N_species,).
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
    volume : float
        System volume.
        
    Returns
    -------
    Tuple[ndarray, ndarray, ndarray]
        - Scalar pressure for each species. Shape: (N_species,)
        - Kinetic pressure tensor for each species. Shape: (N_species, 3, 3)
        - Potential pressure tensor for each species. Shape: (N_species, 3, 3)
    """
    n_species = species_num.shape[0]
    pressure_scalar = np.zeros(n_species)
    pressure_kin = np.zeros((n_species, 3, 3))
    pressure_pot = np.zeros((n_species, 3, 3))
    
    # Calculate kinetic pressure for each species
    for sp in range(n_species):
        start_idx = species_start[sp]
        end_idx = start_idx + species_num[sp]
        
        # Kinetic contribution
        for i in range(start_idx, end_idx):
            for α in range(3):
                for β in range(3):
                    pressure_kin[sp, α, β] += masses[i] * velocities[i, α] * velocities[i, β]
        
        # Normalize by volume
        for α in range(3):
            for β in range(3):
                pressure_kin[sp, α, β] /= volume
    
    # Potential (virial) contribution
    if virial_tensor.size > 0:
        for sp in range(n_species):
            for α in range(3):
                for β in range(3):
                    # Sum over all species interactions
                    for sp2 in range(n_species):
                        pressure_pot[sp, α, β] += virial_tensor[sp, sp2, α, β]
                    pressure_pot[sp, α, β] /= volume
    
    # Calculate scalar pressure
    for sp in range(n_species):
        trace_kin = pressure_kin[sp, 0, 0] + pressure_kin[sp, 1, 1] + pressure_kin[sp, 2, 2]
        trace_pot = pressure_pot[sp, 0, 0] + pressure_pot[sp, 1, 1] + pressure_pot[sp, 2, 2]
        pressure_scalar[sp] = (trace_kin + trace_pot) / 3.0
    
    return pressure_scalar, pressure_kin, pressure_pot


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def calculate_species_pressure_tensor(velocities: ndarray, virial_tensor: ndarray,
                                    species_num: ndarray, masses: ndarray,
                                    volume: float) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Calculate pressure tensor for each species (backward compatibility).
    
    Parameters
    ----------
    velocities : ndarray
        All particle velocities. Shape: (N, 3).
    virial_tensor : ndarray
        Virial tensor. Shape: (N_species, N_species, 3, 3).
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
    masses : ndarray
        All particle masses. Shape: (N,).
    volume : float
        System volume.
        
    Returns
    -------
    Tuple[ndarray, ndarray, ndarray]
        - Scalar pressure for each species. Shape: (N_species,)
        - Kinetic pressure tensor for each species. Shape: (N_species, 3, 3)
        - Potential pressure tensor for each species. Shape: (N_species, 3, 3)
        
    Notes
    -----
    This function maintains backward compatibility with existing Sarkas code.
    """
    if velocities.size == 0:
        n_species = len(species_num)
        return (np.zeros(n_species), 
                np.zeros((n_species, 3, 3)), 
                np.zeros((n_species, 3, 3)))
    
    # Calculate starting indices for each species
    species_start = np.zeros(len(species_num), dtype=np.int64)
    for i in range(1, len(species_num)):
        species_start[i] = species_start[i-1] + species_num[i-1]
    
    return _species_pressure_kernel(
        velocities, masses, virial_tensor, species_start, species_num, volume
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ideal_gas_pressure(number_density: float, temperature: float, 
                      kB: float = _BOLTZMANN_CONSTANT) -> float:
    """
    Calculate ideal gas pressure using equation of state.
    
    P = n * kB * T
    
    Parameters
    ----------
    number_density : float
        Number density of particles.
        Units: particles/m³
    temperature : float
        Temperature.
        Units: K
    kB : float, optional
        Boltzmann constant. Default: 1.380649e-23 J/K.
        
    Returns
    -------
    float
        Ideal gas pressure.
        Units: Pa
    """
    return number_density * kB * temperature


def pressure_from_kinetic_energy(kinetic_energy: float, volume: float,
                                dimensions: int = 3) -> float:
    """
    Calculate pressure from total kinetic energy using virial theorem.
    
    For ideal gas: P = (2/3) * (E_kinetic / V)
    
    Parameters
    ----------
    kinetic_energy : float
        Total kinetic energy.
        Units: J
    volume : float
        System volume.
        Units: m³
    dimensions : int, optional
        Number of dimensions. Default: 3.
        
    Returns
    -------
    float
        Pressure.
        Units: Pa
    """
    return (2.0 / dimensions) * (kinetic_energy / volume)


def bulk_modulus_from_pressure_derivative(pressure_func, volume: float, 
                                        dv_fraction: float = 1e-6) -> float:
    """
    Calculate bulk modulus from pressure-volume relationship.
    
    K = -V * (∂P/∂V)
    
    Parameters
    ----------
    pressure_func : callable
        Function that returns pressure given volume.
    volume : float
        Reference volume.
        Units: m³
    dv_fraction : float, optional
        Fractional volume change for derivative. Default: 1e-6.
        
    Returns
    -------
    float
        Bulk modulus.
        Units: Pa
    """
    dv = volume * dv_fraction
    v1 = volume - dv/2
    v2 = volume + dv/2
    
    p1 = pressure_func(v1)
    p2 = pressure_func(v2)
    
    dP_dV = (p2 - p1) / dv
    
    return -volume * dP_dV


# =============================================================================
# VALIDATION AND TESTING FUNCTIONS
# =============================================================================

def _validate_mechanical_functions():
    """
    Validate all mechanical functions with basic test cases.
    
    Raises
    ------
    AssertionError
        If any validation test fails.
    """
    # Test data
    vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    mass = np.array([2.0, 1.0])
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    
    # Test momentum
    p = momentum(vel, mass)
    expected_p = np.array([[20.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    assert np.allclose(p, expected_p), f"Momentum test failed: {p} != {expected_p}"
    
    # Test center of mass velocity
    cm_vel = center_of_mass_velocity(vel, mass)
    expected_cm_vel = np.array([20.0/3.0, 5.0/3.0, 0.0])
    assert np.allclose(cm_vel, expected_cm_vel), f"CM vel test failed: {cm_vel} != {expected_cm_vel}"
    
    # Test remove center of mass motion
    vel_corrected = remove_center_of_mass_motion(vel, mass)
    p_total = momentum(vel_corrected, mass).sum(axis=0)
    assert np.allclose(p_total, [0.0, 0.0, 0.0], atol=1e-10), f"CM removal test failed: {p_total}"
    
    # Test angular momentum
    L = angular_momentum(pos, vel, mass)
    # L = r × p; for particle 1: [1,0,0] × [20,0,0] = [0,0,0]
    # for particle 2: [0,1,0] × [0,5,0] = [0,0,0]
    # Actually: particle 2 at (0,1,0) with velocity (0,5,0) gives L_z = 0*5 - 1*0 = 0
    # Let's recalculate: L = [0,0,0] for both particles in this setup
    expected_L = np.array([0.0, 0.0, 0.0])
    assert np.allclose(L, expected_L), f"Angular momentum test failed: {L} != {expected_L}"
    
    # Test kinetic pressure tensor
    volume = 1e-15
    P_kin = pressure_tensor_kinetic(vel, mass, volume)
    expected_P_kin = np.array([[200.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 0.0]]) / volume
    assert np.allclose(P_kin, expected_P_kin), f"Kinetic pressure test failed"
    
    # Test scalar pressure
    P_scalar = pressure_scalar(P_kin)
    expected_P_scalar = (200.0 + 25.0 + 0.0) / (3.0 * volume)
    assert np.allclose(P_scalar, expected_P_scalar), f"Scalar pressure test failed"
    
    print("All mechanical function validation tests passed!")


if __name__ == "__main__":
    # Run validation tests
    _validate_mechanical_functions()
    
    # Basic performance test
    import time
    
    n_particles = 10000
    vel = np.random.randn(n_particles, 3) * 100.0
    mass = np.random.uniform(0.1, 10.0, n_particles)
    volume = 1e-12
    
    start_time = time.time()
    for _ in range(100):
        P_kin = pressure_tensor_kinetic(vel, mass, volume)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Kinetic pressure tensor calculation: {avg_time*1e6:.2f} μs average")
    print(f"Performance: {avg_time/n_particles*1e9:.3f} ns per particle")
