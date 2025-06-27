"""
Transport property calculations for plasma molecular dynamics simulations.

This module provides optimized functions for calculating transport properties
including electric current, heat flux, and diffusion in plasma systems.
All functions are compiled with Numba for high performance.

Physical Background
------------------
Transport properties describe the flow of conserved quantities (charge, energy, mass)
in response to gradients in the system. Key transport coefficients include:

- Electrical conductivity: relates current density to electric field
- Thermal conductivity: relates heat flux to temperature gradient  
- Diffusion coefficient: relates particle flux to concentration gradient

Notes
-----
All functions use SI units unless otherwise specified:
- Velocities: m/s
- Charges: C (Coulombs)
- Energies: J (Joules)
- Positions: m
- Current densities: A/m² (Amperes per square meter)
- Heat flux: W/m² (Watts per square meter)
"""

import numpy as np
from numba import jit
from numpy import ndarray
from typing import Optional

# Physical constants
_ELEMENTARY_CHARGE = 1.602176634e-19  # C
_BOLTZMANN_CONSTANT = 1.380649e-23   # J/K


@jit(nopython=True)
def _electric_current_kernel(velocities: ndarray, charges: ndarray) -> ndarray:
    """
    Numba-compiled kernel for electric current calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    charges : ndarray
        Particle charges. Shape: (N,).
        
    Returns
    -------
    ndarray
        Electric current vector for each particle. Shape: (N, 3).
    """
    n_particles = velocities.shape[0]
    currents = np.zeros((n_particles, 3))
    
    for i in range(n_particles):
        for j in range(3):
            currents[i, j] = charges[i] * velocities[i, j]
    
    return currents


def electric_current_density(velocities: ndarray, charges: ndarray, 
                           volume: float = 1.0) -> ndarray:
    """
    Calculate electric current density from particle velocities and charges.
    
    The electric current density is given by:
    J = (1/V) * Σᵢ qᵢ * vᵢ
    
    where qᵢ is the charge and vᵢ is the velocity of particle i,
    and V is the system volume.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3) where N is number of particles.
        Units: m/s
    charges : ndarray
        Particle charges. Shape: (N,).
        Units: C
    volume : float, optional
        System volume for density calculation. Default: 1.0.
        Units: m³
        
    Returns
    -------
    ndarray
        Electric current density vector. Shape: (3,).
        Units: A/m²
        
    Examples
    --------
    >>> import numpy as np
    >>> # Two particles with opposite charges
    >>> vel = np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    >>> charges = np.array([1.6e-19, -1.6e-19])  # +e, -e
    >>> J = electric_current_density(vel, charges, volume=1e-15)
    >>> print(f"Current density: {J} A/m²")
    
    Notes
    -----
    - Positive charges moving in +x direction contribute positive current
    - Negative charges moving in -x direction also contribute positive current
    - Current density is extensive (scales with particle density)
    """
    if velocities.size == 0:
        return np.zeros(3)
    
    # Calculate individual particle currents
    particle_currents = _electric_current_kernel(velocities, charges)
    
    # Sum to get total current and divide by volume for density
    total_current = particle_currents.sum(axis=0)
    
    return total_current / volume


def electric_current_vector(velocities: ndarray, charges: ndarray) -> ndarray:
    """
    Calculate total electric current vector (not density).
    
    The total electric current is:
    I = Σᵢ qᵢ * vᵢ
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    charges : ndarray
        Particle charges. Shape: (N,).
        Units: C
        
    Returns
    -------
    ndarray
        Total electric current vector. Shape: (3,).
        Units: A (Amperes)
        
    Examples
    --------
    >>> import numpy as np
    >>> # Single electron moving at thermal velocity
    >>> vel = np.array([[1e5, 0.0, 0.0]])  # 100 km/s
    >>> charges = np.array([-1.6e-19])  # electron charge
    >>> I = electric_current_vector(vel, charges)
    >>> print(f"Current: {I} A")
    
    Notes
    -----
    - This gives the total current, not current density
    - Useful for calculating total system current
    """
    if velocities.size == 0:
        return np.zeros(3)
    
    # Calculate individual particle currents
    particle_currents = _electric_current_kernel(velocities, charges)
    
    # Sum to get total current vector
    return particle_currents.sum(axis=0)


@jit(nopython=True)
def _heat_flux_kernel(velocities: ndarray, kinetic_energy: ndarray) -> ndarray:
    """
    Numba-compiled kernel for convective heat flux calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    kinetic_energy : ndarray
        Particle kinetic energies. Shape: (N,).
        
    Returns
    -------
    ndarray
        Heat flux vector for each particle. Shape: (N, 3).
    """
    n_particles = velocities.shape[0]
    heat_flux = np.zeros((n_particles, 3))
    
    for i in range(n_particles):
        for j in range(3):
            heat_flux[i, j] = kinetic_energy[i] * velocities[i, j]
    
    return heat_flux


def heat_flux_vector(velocities: ndarray, kinetic_energy: ndarray, 
                    volume: float = 1.0) -> ndarray:
    """
    Calculate convective heat flux vector.
    
    The convective heat flux is given by:
    q = (1/V) * Σᵢ Eₖᵢ * vᵢ
    
    where Eₖᵢ is the kinetic energy and vᵢ is the velocity of particle i.
    This represents energy transport due to particle motion.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    kinetic_energy : ndarray
        Particle kinetic energies. Shape: (N,).
        Units: J
    volume : float, optional
        System volume. Default: 1.0.
        Units: m³
        
    Returns
    -------
    ndarray
        Heat flux vector. Shape: (3,).
        Units: W/m²
        
    Examples
    --------
    >>> import numpy as np
    >>> # Hot particle moving to the right
    >>> vel = np.array([[1000.0, 0.0, 0.0]])
    >>> ke = np.array([1.0e-18])  # High kinetic energy
    >>> q = heat_flux_vector(vel, ke, volume=1e-15)
    >>> print(f"Heat flux: {q} W/m²")
    
    Notes
    -----
    - Represents convective (advective) heat transport
    - Does not include conductive heat flux
    - Positive flux indicates heat flow in positive coordinate direction
    """
    if velocities.size == 0:
        return np.zeros(3)
    
    # Calculate individual particle heat flux contributions
    particle_heat_flux = _heat_flux_kernel(velocities, kinetic_energy)
    
    # Sum and normalize by volume
    total_heat_flux = particle_heat_flux.sum(axis=0)
    
    return total_heat_flux / volume


@jit(nopython=True)
def _heat_flux_tensor_kernel(velocities: ndarray, kinetic_energy: ndarray,
                            stress_tensor: ndarray) -> ndarray:
    """
    Numba-compiled kernel for full heat flux tensor calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    kinetic_energy : ndarray  
        Particle kinetic energies. Shape: (N,).
    stress_tensor : ndarray
        Stress tensor for each particle. Shape: (N, 3, 3).
        
    Returns
    -------
    ndarray
        Heat flux tensor. Shape: (3, 3).
    """
    n_particles = velocities.shape[0]
    heat_flux_tensor = np.zeros((3, 3))
    
    # Convective contribution: Σᵢ Eₖᵢ * vᵢ ⊗ vᵢ
    for i in range(n_particles):
        for α in range(3):
            for β in range(3):
                heat_flux_tensor[α, β] += kinetic_energy[i] * velocities[i, α] * velocities[i, β]
    
    # Add stress tensor contribution if provided
    if stress_tensor.size > 0:
        for i in range(n_particles):
            for α in range(3):
                for β in range(3):
                    for γ in range(3):
                        heat_flux_tensor[α, β] += velocities[i, γ] * stress_tensor[i, α, β]
    
    return heat_flux_tensor


def heat_flux_tensor(velocities: ndarray, kinetic_energy: ndarray,
                    stress_tensor: Optional[ndarray] = None,
                    volume: float = 1.0) -> ndarray:
    """
    Calculate the full heat flux tensor including convective and conductive terms.
    
    The heat flux tensor includes:
    1. Convective term: (1/V) * Σᵢ Eₖᵢ * vᵢ ⊗ vᵢ  
    2. Stress work term: (1/V) * Σᵢ vᵢ · σᵢ (if stress tensor provided)
    
    This gives the full microscopic heat flux tensor for energy transport analysis.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    kinetic_energy : ndarray
        Particle kinetic energies. Shape: (N,).
        Units: J
    stress_tensor : ndarray, optional
        Stress tensor for each particle. Shape: (N, 3, 3).
        Units: Pa (N/m²)
    volume : float, optional
        System volume. Default: 1.0.
        Units: m³
        
    Returns
    -------
    ndarray
        Heat flux tensor. Shape: (3, 3).
        Units: W/m²
        
    Examples
    --------
    >>> import numpy as np
    >>> vel = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    >>> ke = np.array([1e-20, 1e-20])
    >>> q_tensor = heat_flux_tensor(vel, ke, volume=1e-18)
    >>> print(f"Heat flux tensor:\\n{q_tensor}")
    
    Notes
    -----
    - Diagonal elements represent normal heat flux
    - Off-diagonal elements represent shear heat flux
    - Units are energy flux per unit area
    """
    if velocities.size == 0:
        return np.zeros((3, 3))
    
    # Prepare stress tensor (empty if not provided)
    if stress_tensor is None:
        stress_tensor = np.array([]).reshape(0, 3, 3)
    
    # Calculate heat flux tensor
    heat_flux_tens = _heat_flux_tensor_kernel(velocities, kinetic_energy, stress_tensor)
    
    # Normalize by volume
    return heat_flux_tens / volume


@jit(nopython=True)
def _diffusion_flux_kernel(species_velocities: ndarray, concentrations: ndarray,
                          weights: ndarray) -> ndarray:
    """
    Numba-compiled kernel for diffusion flux calculation.
    
    Parameters
    ----------
    species_velocities : ndarray
        Average velocities of each species. Shape: (N_species, 3).
    concentrations : ndarray
        Concentrations of each species. Shape: (N_species,).
    weights : ndarray
        Specific weights of each species used to compute the reference velocity. Shape: (N_species,).
        - For barycentric (mass-based) flux: species masses
        - For molar diffusion flux: typically uniform or molar masses
        - For volume-based flux: partial molar volumes or specific volumes 
        
    Returns
    -------
    ndarray
        Diffusion flux for each species. Shape: (N_species, 3).
    """
    n_species = species_velocities.shape[0]
    diffusion_flux = np.zeros((n_species, 3))
    
    # NOTE: Should we use com calculation from mechanical here? Nope since remove_com_motion is not numba'd.
    # Calculate total mass density
    total_weights_density = 0.0
    for i in range(n_species):
        total_weights_density += concentrations[i] * weights[i]
    
    # Calculate center of mass velocity
    cm_velocity = np.zeros(3)
    if total_weights_density > 0:
        for i in range(n_species):
            weights_fraction = (concentrations[i] * weights[i]) / total_weights_density
            for j in range(3):
                cm_velocity[j] += weights_fraction * species_velocities[i, j]
    
    # Calculate diffusion flux: ρᵢ * (vᵢ - v_cm)
    for i in range(n_species):
        weights_density_i = concentrations[i] * weights[i]
        for j in range(3):
            diffusion_flux[i, j] = weights_density_i * (species_velocities[i, j] - cm_velocity[j])
    
    return diffusion_flux


def diffusion_flux(species_velocities: ndarray, concentrations: ndarray,
                  weights: ndarray) -> ndarray:
    """
    Calculate generalized diffusion flux for multi-species systems.

    The diffusion flux represents species transport relative to a 
    reference velocity computed as a weighted average over all species.
    This function supports barycentric (mass-based), molar-based, or 
    volume-based diffusion fluxes depending on the weights provided.

    The flux is computed as:
        Jᵢ = ρᵢ * (vᵢ - v_ref)
    
    where:
        - ρᵢ is the species-specific weighting factor times concentration
        - vᵢ is the average velocity of species i
        - v_ref is the reference velocity, computed as the weighted average 
          over all species velocities

    
    Parameters
    ----------
    species_velocities : ndarray
        Average velocities for each species. Shape: (N_species, 3).
    concentrations : ndarray
        Number density of each species. Shape: (N_species,).
    weights : ndarray
        Weighting property of each species used to compute the reference 
        velocity. Shape: (N_species,). Examples:
            - `weights = species_masses` -> species masses for barycentric (mass-based) flux
            - `weights = species_num_densities` -> unity or molar masses for molar diffusion flux
            - `weights = species_volumes` -> partial molar volumes for volume-based flux
    Returns
    -------
    ndarray
        Diffusion flux for each species. Shape: (N_species, 3).
        
    Examples
    --------
    >>> import numpy as np
    >>> # Two species with different drift velocities
    >>> vel = np.array([[100.0, 0.0, 0.0], [-50.0, 0.0, 0.0]])
    >>> conc = np.array([1e20, 1e20])  # Equal concentrations
    >>> masses = np.array([1.67e-27, 9.11e-31])  # Proton, electron
    >>> J_diff = diffusion_flux(vel, conc, masses)
    >>> print(f"Diffusion flux: {J_diff}")
    
    """
    if species_velocities.size == 0:
        return np.array([]).reshape(0, 3)
    
    return _diffusion_flux_kernel(species_velocities, concentrations, weights)


@jit(nopython=True)
def _species_current_kernel(velocities: ndarray, charges: ndarray,
                           species_start: ndarray, species_num: ndarray) -> ndarray:
    """
    Numba-compiled kernel for species-specific current calculation.
    
    Parameters
    ----------
    velocities : ndarray
        All particle velocities. Shape: (N, 3).
    charges : ndarray
        All particle charges. Shape: (N,).
    species_start : ndarray
        Starting index for each species. Shape: (N_species,).
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
        
    Returns
    -------
    ndarray
        Current vector for each species. Shape: (N_species, 3).
    """
    n_species = species_num.shape[0]
    species_current = np.zeros((n_species, 3))
    
    for sp in range(n_species):
        start_idx = species_start[sp]
        end_idx = start_idx + species_num[sp]
        
        for i in range(start_idx, end_idx):
            for j in range(3):
                species_current[sp, j] += charges[i] * velocities[i, j]
    
    return species_current


# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# =============================================================================

def calculate_electric_current(velocities: ndarray, charges: ndarray) -> ndarray:
    """
    Backward compatibility wrapper for electric current calculation.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    charges : ndarray
        Particle charges. Shape: (N,).
        
    Returns
    -------
    ndarray
        Electric current for each particle. Shape: (N, 3).
        
    Notes
    -----
    This function maintains backward compatibility with existing Sarkas code.
    """
    return _electric_current_kernel(velocities, charges)


def calculate_species_electric_current(velocities: ndarray, charges: ndarray,
                                     species_num: ndarray) -> ndarray:
    """
    Calculate electric current for each species.
    
    Parameters
    ----------
    velocities : ndarray
        All particle velocities. Shape: (N, 3).
    charges : ndarray
        All particle charges. Shape: (N,).
    species_num : ndarray
        Number of particles per species. Shape: (N_species,).
        
    Returns
    -------
    ndarray
        Current vector for each species. Shape: (N_species, 3).
        
    Notes
    -----
    Assumes particles are ordered by species in the input arrays.
    """
    if velocities.size == 0:
        return np.array([]).reshape(0, 3)
    
    # Calculate starting indices for each species
    species_start = np.zeros(len(species_num), dtype=np.int64)
    for i in range(1, len(species_num)):
        species_start[i] = species_start[i-1] + species_num[i-1]
    
    return _species_current_kernel(velocities, charges, species_start, species_num)


# =============================================================================
# UTILITY FUNCTIONS FOR TRANSPORT ANALYSIS
# =============================================================================

@jit(nopython=True)
def _thermal_velocity_kernel(velocities: ndarray, temperature: float,
                            mass: float) -> ndarray:
    """
    Calculate thermal velocities (deviations from mean).
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
    temperature : float
        System temperature.
    mass : float
        Particle mass.
        
    Returns
    -------
    ndarray
        Thermal velocities. Shape: (N, 3).
    """
    n_particles = velocities.shape[0]
    thermal_vel = np.zeros((n_particles, 3))
    
    # Calculate mean velocity
    mean_vel = np.zeros(3)
    for i in range(n_particles):
        for j in range(3):
            mean_vel[j] += velocities[i, j]
    mean_vel /= n_particles
    
    # Calculate thermal velocities (deviations from mean)
    for i in range(n_particles):
        for j in range(3):
            thermal_vel[i, j] = velocities[i, j] - mean_vel[j]
    
    return thermal_vel


def electrical_conductivity_tensor(velocities: ndarray, charges: ndarray,
                                 electric_field: ndarray, volume: float,
                                 dt: float) -> ndarray:
    """
    Calculate electrical conductivity tensor from response to electric field.
    
    The conductivity tensor σ relates current density to electric field:
    J = σ · E
    
    This can be calculated from the velocity response to an applied field.
    
    Parameters
    ----------
    velocities : ndarray
        Particle velocities. Shape: (N, 3).
        Units: m/s
    charges : ndarray
        Particle charges. Shape: (N,).
        Units: C
    electric_field : ndarray
        Applied electric field. Shape: (3,).
        Units: V/m
    volume : float
        System volume.
        Units: m³
    dt : float
        Time step for velocity response.
        Units: s
        
    Returns
    -------
    ndarray
        Electrical conductivity tensor. Shape: (3, 3).
        Units: S/m (Siemens per meter)
        
    Notes
    -----
    This is a simplified calculation assuming linear response.
    More sophisticated methods would use Green-Kubo relations.
    """
    if velocities.size == 0 or np.linalg.norm(electric_field) == 0:
        return np.zeros((3, 3))
    
    # Calculate current density
    current_density = electric_current_density(velocities, charges, volume)
    
    # Simple linear response approximation
    # σ = J / E (this is a rough approximation)
    conductivity = np.zeros((3, 3))
    
    for i in range(3):
        if abs(electric_field[i]) > 1e-15:
            conductivity[i, i] = current_density[i] / electric_field[i]
    
    return conductivity


def thermal_conductivity_from_heat_flux(heat_flux: ndarray, 
                                      temperature_gradient: ndarray) -> ndarray:
    """
    Calculate thermal conductivity from heat flux and temperature gradient.
    
    Fourier's law: q = -κ · ∇T
    Therefore: κ = -q / ∇T
    
    Parameters
    ----------
    heat_flux : ndarray
        Heat flux vector. Shape: (3,).
        Units: W/m²
    temperature_gradient : ndarray
        Temperature gradient vector. Shape: (3,).
        Units: K/m
        
    Returns
    -------
    ndarray
        Thermal conductivity tensor. Shape: (3, 3).
        Units: W/(m·K)
        
    Notes
    -----
    This gives a simplified diagonal thermal conductivity.
    Full tensor requires more sophisticated analysis.
    """
    if temperature_gradient.size == 0:
        return np.zeros((3, 3))
    
    conductivity = np.zeros((3, 3))
    
    for i in range(3):
        if abs(temperature_gradient[i]) > 1e-15:
            conductivity[i, i] = -heat_flux[i] / temperature_gradient[i]
    
    return conductivity


# =============================================================================
# VALIDATION AND TESTING FUNCTIONS
# =============================================================================

def _validate_transport_functions():
    """
    Validate all transport functions with basic test cases.
    
    Raises
    ------
    AssertionError
        If any validation test fails.
    """
    # Test data
    vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    charges = np.array([1.6e-19, -1.6e-19])  # +e, -e
    
    # Test electric current
    I_total = electric_current_vector(vel, charges)
    expected_I = np.array([1.6e-18, -8e-19, 0.0])
    assert np.allclose(I_total, expected_I), f"Current test failed: {I_total} != {expected_I}"
    
    # Test current density
    J = electric_current_density(vel, charges, volume=1e-15)
    expected_J = expected_I / 1e-15
    assert np.allclose(J, expected_J), f"Current density test failed: {J} != {expected_J}"
    
    # Test heat flux
    ke = np.array([1e-20, 0.5e-20])
    q = heat_flux_vector(vel, ke, volume=1e-15)
    expected_q = np.array([1e-18, 2.5e-21, 0.0]) / 1e-15
    assert np.allclose(q, expected_q), f"Heat flux test failed: {q} != {expected_q}"
    
    # Test diffusion flux
    vel_species = np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0]])
    conc = np.array([1e20, 1e20])
    masses = np.array([1.67e-27, 9.11e-31])
    J_diff = diffusion_flux(vel_species, conc, masses)
    
    # Check that total diffusion flux is approximately zero
    J_total = J_diff.sum(axis=0)
    assert np.allclose(J_total, [0.0, 0.0, 0.0], atol=1e-30), f"Diffusion conservation failed: {J_total}"
    
    print("All transport function validation tests passed!")


if __name__ == "__main__":
    # Run validation tests
    _validate_transport_functions()
    
    # Basic performance test
    import time
    
    n_particles = 10000
    vel = np.random.randn(n_particles, 3) * 100.0
    charges = np.random.choice([-1.6e-19, 1.6e-19], n_particles)
    
    start_time = time.time()
    for _ in range(100):
        J = electric_current_density(vel, charges, volume=1e-12)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Electric current density calculation: {avg_time*1e6:.2f} μs average")
    print(f"Performance: {avg_time/n_particles*1e9:.3f} ns per particle")
