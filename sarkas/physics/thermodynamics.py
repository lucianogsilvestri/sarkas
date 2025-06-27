"""
Thermodynamics calculations for molecular dynamics simulations.

This module provides pure functions for calculating thermodynamic properties
from particle velocities and masses. All functions are optimized with Numba
compilation for high performance.

Notes
-----
All functions use SI units unless otherwise specified:
- Velocities: m/s
- Masses: kg
- Energies: J
- Temperatures: K
- Boltzmann constant: J/K
"""

import numpy as np
from numba import jit
from numpy import ndarray
from typing import Tuple, Optional

# Constants
_BOLTZMANN_CONSTANT = 1.380649e-23  # J/K


@jit(nopython=True)
def _compute_kinetic_energy_kernel(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Numba-compiled kernel for kinetic energy calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
    masses : ndarray
        Particle masses. Shape: (N,).
        
    Returns
    -------
    ndarray
        Kinetic energy of each particle. Shape: (N,).
    """
    n_particles = velocities.shape[0]
    kinetic_energies = np.zeros(n_particles)
    
    for i in range(n_particles):
        v_squared = 0.0
        for j in range(velocities.shape[1]):
            v_squared += velocities[i, j] * velocities[i, j]
        kinetic_energies[i] = 0.5 * masses[i] * v_squared
    
    return kinetic_energies

def _compute_kinetic_energy_vectorized(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Vectorized kinetic energy calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
    masses : ndarray
        Particle masses. Shape: (N,).
        
    Returns
    -------
    ndarray
        Kinetic energy of each particle. Shape: (N,).
    """
    # Compute velocity squared for each particle
    v_squared = np.sum(velocities * velocities, axis=1)
    return 0.5 * masses * v_squared

def kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Calculate the kinetic energy of particles.
    
    Uses the classical kinetic energy formula:
    KE = (1/2) * m * v²
    
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
        Kinetic energy of each particle. Shape: (N,).
        Units: J
        
    Examples
    --------
    >>> import numpy as np
    >>> # Single particle moving at 10 m/s in x-direction
    >>> vel = np.array([[10.0, 0.0, 0.0]])
    >>> mass = np.array([1.0])  # 1 kg
    >>> ke = kinetic_energy(vel, mass)
    >>> print(f"Kinetic energy: {ke[0]:.1f} J")  # Should be 50.0 J
    
    >>> # Multiple particles
    >>> vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    >>> mass = np.array([1.0, 2.0])
    >>> ke = kinetic_energy(vel, mass)
    >>> print(f"KE: {ke[0]:.1f} J, {ke[1]:.1f} J")  # 50.0 J, 25.0 J
    
    Notes
    -----
    - Handles both uniform and non-uniform masses
    - Optimized with Numba for high performance
    - Zero allocations in compiled kernels
    """
    # Input validation
    if velocities.size == 0:
        return np.array([])
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(velocities.shape[0], masses[0])
    else:
        masses_broadcast = masses
    
    # Use vectorized version for better performance
    return _compute_kinetic_energy_kernel(velocities, masses_broadcast)


@jit(nopython=True)
def _temperature_from_ke_kernel(kinetic_energies: ndarray, 
                               dimensions: int, kB: float) -> ndarray:
    """
    Numba-compiled kernel for temperature calculation from kinetic energy.
    
    Parameters
    ----------
    kinetic_energies : ndarray
        Kinetic energies. Shape: (N,).
    dimensions : int
        Number of spatial dimensions (typically 3).
    kB : float
        Boltzmann constant.
        
    Returns
    -------
    ndarray
        Temperature of each particle. Shape: (N,).
    """
    n_particles = kinetic_energies.shape[0]
    temperatures = np.zeros(n_particles)
    
    for i in range(n_particles):
        # T = 2*KE / (dimensions * kB)
        temperatures[i] = 2.0 * kinetic_energies[i] / (dimensions * kB)
    
    return temperatures


def temperature_from_kinetic_energy(ke: ndarray, 
                                  dimensions: int, kB: float) -> ndarray:
    """
    Calculate temperature from kinetic energy using equipartition theorem.
    
    Uses the equipartition theorem:
    <KE> = (1/2) * dimensions * kB * T
    Therefore: T = 2*KE / (dimensions * kB)
    
    Parameters
    ----------
    ke : ndarray
        Kinetic energy of each particle. Shape: (N,).
    dimensions : int
        Number of spatial dimensions (typically 3).
    kB : float
        Boltzmann constant.
        
    Returns
    -------
    ndarray
        Temperature of each particle. Shape: (N,).
        Units: K
        
    Examples
    --------
    >>> import numpy as np
    >>> # Particle with kinetic energy at room temperature
    >>> ke = np.array([6.21e-21])  # J (kB * 300 K * 3/2)
    >>> T = temperature_from_kinetic_energy(ke, 3, 1.380649e-23)
    >>> print(f"Temperature: {T[0]:.1f} K")  # Should be ~300 K
    
    Notes
    -----
    - Based on equipartition theorem
    - Masses parameter kept for API consistency
    - Optimized with Numba compilation
    """
    if ke.size == 0:
        return np.array([])
    
    return _temperature_from_ke_kernel(ke, dimensions, kB)


def temperature_from_velocities(velocities: ndarray, masses: ndarray, 
                               dimensions: int, kB: float) -> ndarray:
    """
    Calculate temperature directly from velocities.
    
    Combines kinetic energy calculation with temperature calculation:
    KE = (1/2) * m * v²
    T = 2*KE / (dimensions * kB) = m * v² / (dimensions * kB)
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        Units: m/s
    masses : ndarray
        Particle masses. Shape: (N,) or (1,) for uniform mass.
        Units: kg
    dimensions : int
        Number of spatial dimensions (typically 3).
    kB : float
        Boltzmann constant.
        Units: J/K
        
    Returns
    -------
    ndarray
        Temperature of each particle. Shape: (N,).
        Units: K
        
    Examples
    --------
    >>> import numpy as np
    >>> # Particle at thermal velocity for 300 K
    >>> vel = np.array([[517.0, 0.0, 0.0]])  # m/s (approximate for 300 K)
    >>> mass = np.array([4.65e-26])  # kg (argon atom)
    >>> T = temperature_from_velocities(vel, mass, 3, 1.380649e-23)
    >>> print(f"Temperature: {T[0]:.1f} K")
    
    Notes
    -----
    - More efficient than calculating kinetic energy separately
    - Handles both uniform and non-uniform masses
    - Optimized with Numba compilation
    """
    if velocities.size == 0:
        return np.array([])
    
    # First calculate kinetic energy
    ke = kinetic_energy(velocities, masses)
    
    # Then convert to temperature
    return temperature_from_kinetic_energy(ke, dimensions, kB)


# Backward compatibility wrapper functions
def calculate_kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray:
    """
    Backward compatibility wrapper for kinetic_energy().
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,) or (1,).
        
    Returns
    -------
    ndarray
        Kinetic energy of each particle. Shape: (N,).
        
    Notes
    -----
    This function maintains backward compatibility with the existing
    Sarkas Particles class interface.
    """
    return kinetic_energy(velocities, masses)


def calculate_temperature(velocities: ndarray, masses: ndarray, 
                         dimensions: int = 3, kB: float = _BOLTZMANN_CONSTANT,
                         **kwargs) -> ndarray:
    """
    Backward compatibility wrapper for temperature_from_velocities().
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    masses : ndarray
        Particle masses. Shape: (N,) or (1,).
    dimensions : int, optional
        Number of spatial dimensions. Default: 3.
    kB : float, optional
        Boltzmann constant. Default: 1.380649e-23 J/K.
    **kwargs
        Additional keyword arguments (ignored for compatibility).
        
    Returns
    -------
    ndarray
        Temperature of each particle. Shape: (N,).
        
    Notes
    -----
    This function maintains backward compatibility with the existing
    Sarkas Particles class interface.
    """
    return temperature_from_velocities(velocities, masses, dimensions, kB)

# =============================================================================
# ENTHALPY CALCULATIONS
# =============================================================================

@jit(nopython=True)
def _enthalpy_kernel(kinetic_energy: ndarray, potential_energy: ndarray,
                    pressure: float, volume_per_particle: float) -> ndarray:
    """
    Numba-compiled kernel for enthalpy calculation.
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energy of each particle. Shape: (N,).
    potential_energy : ndarray
        Potential energy of each particle. Shape: (N,).
    pressure : float
        System pressure.
    volume_per_particle : float
        Volume per particle.
        
    Returns
    -------
    ndarray
        Enthalpy of each particle. Shape: (N,).
    """
    n_particles = kinetic_energy.shape[0]
    enthalpies = np.zeros(n_particles)
    
    for i in range(n_particles):
        # H = U + PV where U = KE + PE
        total_energy = kinetic_energy[i] + potential_energy[i]
        enthalpies[i] = total_energy + pressure * volume_per_particle
    
    return enthalpies


def enthalpy(kinetic_energy: ndarray, potential_energy: ndarray,
            pressure: float, volume: float) -> ndarray:
    """
    Calculate enthalpy of particles.
    
    Enthalpy is defined as:
    H = U + PV
    
    where U is internal energy (kinetic + potential), P is pressure, and V is volume.
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energy of each particle. Shape: (N,).
        Units: J
    potential_energy : ndarray
        Potential energy of each particle. Shape: (N,).
        Units: J
    pressure : float
        System pressure.
        Units: Pa
    volume : float
        Total system volume.
        Units: m³
        
    Returns
    -------
    ndarray
        Enthalpy of each particle. Shape: (N,).
        Units: J
        
    Examples
    --------
    >>> import numpy as np
    >>> ke = np.array([1e-20, 2e-20])  # J
    >>> pe = np.array([0.5e-20, 1.5e-20])  # J
    >>> pressure = 1e5  # Pa
    >>> volume = 1e-15  # m³
    >>> H = enthalpy(ke, pe, pressure, volume)
    >>> print(f"Enthalpy: {H}")
    
    Notes
    -----
    - Enthalpy is conserved in constant pressure processes
    - The PV term is often small for condensed phases
    - Important thermodynamic potential for many processes
    - For ideal gas: H = (5/2)NkT (monatomic)
    """
    if kinetic_energy.size == 0:
        return np.array([])
    
    n_particles = len(kinetic_energy)
    volume_per_particle = volume / n_particles if n_particles > 0 else 0.0
    
    return _enthalpy_kernel(kinetic_energy, potential_energy, pressure, volume_per_particle)


def specific_enthalpy(kinetic_energy: ndarray, potential_energy: ndarray,
                     pressure: float, volume: float, masses: ndarray) -> ndarray:
    """
    Calculate specific enthalpy (enthalpy per unit mass).
    
    Specific enthalpy is:
    h = H/m = (U + PV)/m
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energy of each particle. Shape: (N,).
        Units: J
    potential_energy : ndarray
        Potential energy of each particle. Shape: (N,).
        Units: J
    pressure : float
        System pressure.
        Units: Pa
    volume : float
        Total system volume.
        Units: m³
    masses : ndarray
        Particle masses. Shape: (N,).
        Units: kg
        
    Returns
    -------
    ndarray
        Specific enthalpy of each particle. Shape: (N,).
        Units: J/kg
        
    Notes
    -----
    - Useful for comparing particles of different masses
    - Important in fluid mechanics and thermodynamics
    - Independent of system size when properly normalized
    """
    H = enthalpy(kinetic_energy, potential_energy, pressure, volume)
    
    if masses.size == 1:
        # Broadcast uniform mass
        masses_broadcast = np.full(len(H), masses[0])
    else:
        masses_broadcast = masses
    
    return H / masses_broadcast


@jit(nopython=True)
def _enthalpy_species_kernel(kinetic_energy: ndarray, potential_energy: ndarray,
                            pressure: ndarray, volume: float,
                            species_start: ndarray, species_num: ndarray) -> ndarray:
    """
    Numba-compiled kernel for species-specific enthalpy calculation.
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energy of all particles. Shape: (N,).
    potential_energy : ndarray
        Potential energy of all particles. Shape: (N,).
    pressure : ndarray
        Pressure for each species. Shape: (N_species,).
    volume : float
        Total system volume.
    species_start : ndarray
        Starting indices for each species. Shape: (N_species,).
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
        
    Returns
    -------
    ndarray
        Total enthalpy for each species. Shape: (N_species,).
    """
    n_species = species_num.shape[0]
    species_enthalpy = np.zeros(n_species)
    
    for sp in range(n_species):
        start_idx = species_start[sp]
        end_idx = start_idx + species_num[sp]
        
        # Sum energy for this species
        total_energy = 0.0
        for i in range(start_idx, end_idx):
            total_energy += kinetic_energy[i] + potential_energy[i]
        
        # Add PV term
        species_volume = volume * species_num[sp] / species_num.sum()
        species_enthalpy[sp] = total_energy + pressure[sp] * species_volume
    
    return species_enthalpy


def species_enthalpy(kinetic_energy: ndarray, potential_energy: ndarray,
                    pressure: ndarray, volume: float, species_num: ndarray) -> ndarray:
    """
    Calculate enthalpy for each species.
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energy of all particles. Shape: (N,).
        Units: J
    potential_energy : ndarray
        Potential energy of all particles. Shape: (N,).
        Units: J
    pressure : ndarray
        Pressure for each species. Shape: (N_species,).
        Units: Pa
    volume : float
        Total system volume.
        Units: m³
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
        
    Returns
    -------
    ndarray
        Total enthalpy for each species. Shape: (N_species,).
        Units: J
        
    Notes
    -----
    Assumes particles are ordered by species in the input arrays.
    Each species gets a volume fraction proportional to its particle count.
    """
    if kinetic_energy.size == 0:
        return np.zeros(len(species_num))
    
    # Calculate starting indices for each species
    species_start = np.zeros(len(species_num), dtype=np.int64)
    for i in range(1, len(species_num)):
        species_start[i] = species_start[i-1] + species_num[i-1]
    
    return _enthalpy_species_kernel(
        kinetic_energy, potential_energy, pressure, volume, species_start, species_num
    )


# =============================================================================
# ADDITIONAL THERMODYNAMIC UTILITIES
# =============================================================================

def heat_capacity_constant_volume(kinetic_energy: ndarray, temperature: float,
                                 dimensions: int = 3, kB: float = _BOLTZMANN_CONSTANT) -> float:
    """
    Calculate heat capacity at constant volume using equipartition theorem.
    
    For ideal gas: Cv = (d/2) * N * kB
    where d is the number of degrees of freedom.
    
    Parameters
    ----------
    kinetic_energy : ndarray
        Kinetic energies of particles. Shape: (N,).
        Units: J
    temperature : float
        System temperature.
        Units: K
    dimensions : int, optional
        Number of spatial dimensions. Default: 3.
    kB : float, optional
        Boltzmann constant. Default: 1.380649e-23 J/K.
        
    Returns
    -------
    float
        Heat capacity at constant volume.
        Units: J/K
        
    Notes
    -----
    - Based on equipartition theorem for ideal gas
    - Real systems may deviate due to interactions
    - Quantum effects important at low temperatures
    """
    n_particles = len(kinetic_energy)
    return (dimensions / 2.0) * n_particles * kB


def heat_capacity_constant_pressure(cv: float, n_particles: int,
                                   kB: float = _BOLTZMANN_CONSTANT) -> float:
    """
    Calculate heat capacity at constant pressure for ideal gas.
    
    For ideal gas: Cp = Cv + N * kB
    
    Parameters
    ----------
    cv : float
        Heat capacity at constant volume.
        Units: J/K
    n_particles : int
        Number of particles.
    kB : float, optional
        Boltzmann constant. Default: 1.380649e-23 J/K.
        
    Returns
    -------
    float
        Heat capacity at constant pressure.
        Units: J/K
    """
    return cv + n_particles * kB


def adiabatic_index(cp: float, cv: float) -> float:
    """
    Calculate adiabatic index (heat capacity ratio).
    
    γ = Cp / Cv
    
    Parameters
    ----------
    cp : float
        Heat capacity at constant pressure.
        Units: J/K
    cv : float
        Heat capacity at constant volume.
        Units: J/K
        
    Returns
    -------
    float
        Adiabatic index (dimensionless).
        
    Notes
    -----
    - For monatomic ideal gas: γ = 5/3
    - For diatomic ideal gas: γ = 7/5
    - Important for adiabatic processes and sound speed
    """
    if cv == 0:
        return float('inf')
    return cp / cv



# Utility functions for performance benchmarking
def _benchmark_kinetic_energy(n_particles: int = 10000, n_iterations: int = 100) -> float:
    """
    Benchmark kinetic energy calculation performance.
    
    Parameters
    ----------
    n_particles : int, optional
        Number of particles for benchmark. Default: 10000.
    n_iterations : int, optional
        Number of iterations for timing. Default: 100.
        
    Returns
    -------
    float
        Average time per iteration in seconds.
    """
    import time
    
    # Generate random test data
    np.random.seed(42)
    velocities = np.random.randn(n_particles, 3) * 100.0
    masses = np.random.uniform(0.1, 10.0, n_particles)
    
    # Warm up Numba compilation
    _ = kinetic_energy(velocities[:10], masses[:10])
    
    # Time the calculation
    start_time = time.time()
    for _ in range(n_iterations):
        _ = kinetic_energy(velocities, masses)
    end_time = time.time()
    
    return (end_time - start_time) / n_iterations


def _validate_functions():
    """
    Validate all functions with basic test cases.
    
    Raises
    ------
    AssertionError
        If any validation test fails.
    """
    # Test data
    vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    mass = np.array([1.0, 2.0])
    
    # Test kinetic energy
    ke = kinetic_energy(vel, mass)
    expected_ke = np.array([50.0, 25.0])
    assert np.allclose(ke, expected_ke), f"KE test failed: {ke} != {expected_ke}"
    
    # Test temperature
    T = temperature_from_kinetic_energy(ke, 3, 1.0)
    expected_T = np.array([100.0/3.0, 50.0/3.0])
    assert np.allclose(T, expected_T), f"T test failed: {T} != {expected_T}"
    
    # Test enthalpy calculation
    pressure = 1e5  # Pa
    volume = 1e-15  # m³
    pe = np.array([0.5e-20, 1.0e-20])  # Potential energies
    H = enthalpy(ke, pe, pressure, volume)
    expected_H = ke + pe + pressure * (volume / len(ke))
    assert np.allclose(H, expected_H), f"Enthalpy test failed: {H} != {expected_H}"
    
    print("All thermodynamics validation tests passed!")


if __name__ == "__main__":
    # Run validation tests
    _validate_functions()
    
    # Run performance benchmark
    avg_time = _benchmark_kinetic_energy()
    print(f"Average kinetic energy calculation time: {avg_time*1e6:.2f} μs")
