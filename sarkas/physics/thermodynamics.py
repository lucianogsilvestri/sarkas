"""
Thermodynamics module for Sarkas plasma simulation.

This module contains pure, optimized functions for thermodynamics calculations
extracted from the Particles class. All functions are designed to be:
- Pure functions with no side effects
- Numba-compiled for performance
- Compatible with species aggregation
- Well-documented with MyST format
"""

from numba import jit
from numpy import ndarray, zeros, sum, isnan, isinf
from numpy.linalg import norm
from .aggregation import species_sum, fast_species_sum


@jit(nopython=True)
def kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray:
    """Calculate kinetic energy per particle.
    
    The kinetic energy of each particle is calculated using the classical formula:
    
    .. math::
        KE_i = \\frac{1}{2} m_i |\\vec{v}_i|^2
    
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
        Per-particle kinetic energy with shape ``(N,)``
        
    Examples
    --------
    Calculate kinetic energy for a system of particles:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import kinetic_energy
        
        # Generate random velocities and masses
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        # Calculate kinetic energy
        ke = kinetic_energy(velocities, masses)
        
        # Verify results
        assert ke.shape == (1000,)
        assert np.all(ke >= 0)  # Kinetic energy always positive
    """
    return 0.5 * masses * (velocities * velocities).sum(axis=-1)


@jit(nopython=True)
def temperature_from_kinetic_energy(kinetic_energy: ndarray, 
                                  dimensions: int, kB: float) -> float:
    """Calculate temperature from kinetic energy using equipartition theorem.
    
    The temperature is calculated from the average kinetic energy using the
    equipartition theorem:
    
    .. math::
        T = \\frac{2 \\langle KE \\rangle}{d k_B}
    
    where :math:`\\langle KE \\rangle` is the average kinetic energy,
    :math:`d` is the number of dimensions, and :math:`k_B` is the Boltzmann constant.
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy with shape ``(N,)``
    dimensions : int
        Number of degrees of freedom (typically 3 for 3D systems)
    kB : float
        Boltzmann constant in appropriate units
        
    Returns
    -------
    float
        System temperature
        
    Examples
    --------
    Calculate temperature from kinetic energy:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import temperature_from_kinetic_energy
        
        # Kinetic energy in Joules, kB in J/K
        ke = np.array([1.5e-21, 2.0e-21, 1.8e-21])  # Per-particle KE
        kB = 1.380649e-23  # Boltzmann constant
        
        T = temperature_from_kinetic_energy(ke, dimensions=3, kB=kB)
        print(f"Temperature: {T:.2f} K")
    """
    return 2.0 * kinetic_energy.mean() / (dimensions * kB)


@jit(nopython=True)
def pressure_kinetic_contribution(velocities: ndarray, masses: ndarray, 
                                volume: float) -> ndarray:
    """Calculate kinetic contribution to pressure tensor.
    
    The kinetic pressure tensor is calculated as:
    
    .. math::
        P^{\\text{kin}}_{\\alpha\\beta} = \\frac{1}{V} \\sum_i m_i v_{i,\\alpha} v_{i,\\beta}
    
    where :math:`V` is the system volume, :math:`m_i` is the mass of particle :math:`i`,
    and :math:`v_{i,\\alpha}` is the :math:`\\alpha`-component of the velocity.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    volume : float
        System volume
        
    Returns
    -------
    numpy.ndarray
        Kinetic pressure tensor with shape ``(3, 3)``
        
    Examples
    --------
    Calculate kinetic pressure tensor:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import pressure_kinetic_contribution
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        volume = 1000.0  # cubic units
        
        P_kin = pressure_kinetic_contribution(velocities, masses, volume)
        assert P_kin.shape == (3, 3)
        assert np.allclose(P_kin, P_kin.T)  # Should be symmetric
    """
    N = velocities.shape[0]
    P_kin = zeros((3, 3))
    
    # Calculate outer products efficiently
    for i in range(3):
        for j in range(3):
            P_kin[i, j] = sum(masses * velocities[:, i] * velocities[:, j])
    
    return P_kin / volume


@jit(nopython=True)
def pressure_virial_contribution(virial_tensor: ndarray, volume: float) -> ndarray:
    """Calculate virial contribution to pressure tensor.
    
    The virial pressure tensor is calculated as:
    
    .. math::
        P^{\\text{vir}} = \\frac{1}{V} \\mathcal{V}
    
    where :math:`\\mathcal{V}` is the virial tensor and :math:`V` is the system volume.
    
    Parameters
    ----------
    virial_tensor : numpy.ndarray
        Virial tensor with shape ``(3, 3)``
    volume : float
        System volume
        
    Returns
    -------
    numpy.ndarray
        Virial pressure tensor with shape ``(3, 3)``
        
    Examples
    --------
    Calculate virial pressure contribution:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import pressure_virial_contribution
        
        # Example virial tensor (symmetric)
        virial = np.array([[1.0, 0.1, 0.2],
                          [0.1, 1.5, 0.3],
                          [0.2, 0.3, 2.0]])
        volume = 1000.0
        
        P_vir = pressure_virial_contribution(virial, volume)
        assert P_vir.shape == (3, 3)
    """
    return virial_tensor / volume


def pressure_tensor_total(velocities: ndarray, masses: ndarray, 
                         virial_tensor: ndarray, volume: float) -> ndarray:
    """Calculate total pressure tensor from kinetic and virial contributions.
    
    The total pressure tensor is the sum of kinetic and virial contributions:
    
    .. math::
        P = P^{\\text{kin}} + P^{\\text{vir}}
    
    This is the complete thermodynamic pressure tensor from kinetic theory.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    virial_tensor : numpy.ndarray
        Virial tensor with shape ``(3, 3)``
    volume : float
        System volume
        
    Returns
    -------
    numpy.ndarray
        Total pressure tensor with shape ``(3, 3)``
        
    Examples
    --------
    Calculate total pressure tensor:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import pressure_tensor_total
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        virial = np.eye(3)  # Identity matrix
        volume = 1000.0
        
        P_total = pressure_tensor_total(velocities, masses, virial, volume)
        assert P_total.shape == (3, 3)
    """
    P_kin = pressure_kinetic_contribution(velocities, masses, volume)
    P_vir = pressure_virial_contribution(virial_tensor, volume)
    return P_kin + P_vir


@jit(nopython=True)
def pressure_scalar(pressure_tensor: ndarray, dimensions: int) -> float:
    """Calculate scalar pressure from pressure tensor.
    
    The scalar pressure is calculated as the trace of the pressure tensor:
    
    .. math::
        P = \\frac{1}{d} \\text{Tr}(P_{\\alpha\\beta})
    
    where :math:`d` is the number of dimensions.
    
    Parameters
    ----------
    pressure_tensor : numpy.ndarray
        Pressure tensor with shape ``(3, 3)``
    dimensions : int
        Number of dimensions (typically 3)
        
    Returns
    -------
    float
        Scalar pressure
        
    Examples
    --------
    Calculate scalar pressure:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import pressure_scalar
        
        # Isotropic pressure tensor
        P_tensor = np.eye(3) * 1.0  # P = 1.0 in all directions
        
        P_scalar = pressure_scalar(P_tensor, dimensions=3)
        assert np.isclose(P_scalar, 1.0)
    """
    for i in range(3):
        P_scalar += pressure_tensor[i, i]
    return P_scalar / dimensions


# Species-level thermodynamic calculations
def species_kinetic_energy(kinetic_energy: ndarray, species_id: ndarray, 
                          num_species: int) -> ndarray:
    """Calculate total kinetic energy for each species.
    
    The total kinetic energy for species :math:`s` is calculated as:
    
    .. math::
        KE_s = \\sum_{i \\in s} KE_i
    
    where the sum is over all particles belonging to species :math:`s`.
    
    This function uses optimized species aggregation for efficiency.
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Total kinetic energy per species with shape ``(num_species,)``
        
    Examples
    --------
    Calculate species kinetic energies:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import species_kinetic_energy
        
        # 1000 particles, 2 species
        ke = np.random.rand(1000)
        species_id = np.random.randint(0, 2, 1000)
        
        species_ke = species_kinetic_energy(ke, species_id, num_species=2)
        assert species_ke.shape == (2,)
        assert np.isclose(species_ke.sum(), ke.sum())  # Conservation
    """
    # Use numpy.bincount for efficient aggregation
    return species_sum(kinetic_energy, species_id, num_species)


def species_temperature(kinetic_energy: ndarray, species_id: ndarray,
                       species_num: ndarray, dimensions: int, kB: float) -> ndarray:
    """Calculate temperature for each species.
    
    The temperature for species :math:`s` is calculated as:
    
    .. math::
        T_s = \\frac{2 KE_s}{N_s d k_B}
    
    where :math:`KE_s` is the total kinetic energy of species :math:`s`,
    :math:`N_s` is the number of particles in species :math:`s`,
    :math:`d` is the number of dimensions, and :math:`k_B` is the Boltzmann constant.
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy with shape ``(N,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    species_num : numpy.ndarray
        Number of particles per species with shape ``(num_species,)``
    dimensions : int
        Number of degrees of freedom
    kB : float
        Boltzmann constant
        
    Returns
    -------
    numpy.ndarray
        Temperature per species with shape ``(num_species,)``
        
    Examples
    --------
    Calculate species temperatures:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import species_temperature
        
        ke = np.random.rand(1000)
        species_id = np.random.randint(0, 2, 1000)
        species_num = np.array([600, 400])  # 600 particles in species 0, 400 in species 1
        kB = 1.380649e-23
        
        T_species = species_temperature(ke, species_id, species_num, dimensions=3, kB=kB)
        assert T_species.shape == (2,)
    """
    species_ke = species_kinetic_energy(kinetic_energy, species_id, len(species_num))
    const = 2.0 / (kB * species_num * dimensions)
    return const * species_ke


def species_pressure_tensor(velocities: ndarray, masses: ndarray, 
                          virial_species_tensor: ndarray, species_id: ndarray,
                          num_species: int, volume: float) -> ndarray:
    """Calculate pressure tensor for each species.
    
    The pressure tensor for each species is calculated using both kinetic and
    virial contributions, similar to the total pressure tensor but separated
    by species.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    virial_species_tensor : numpy.ndarray
        Virial tensor per species with shape ``(num_species, 3, 3)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
    volume : float
        System volume
        
    Returns
    -------
    numpy.ndarray
        Pressure tensor per species with shape ``(num_species, 3, 3)``
        
    Examples
    --------
    Calculate species pressure tensors:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import species_pressure_tensor
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        virial_species = np.random.randn(2, 3, 3)  # 2 species
        species_id = np.random.randint(0, 2, 1000)
        volume = 1000.0
        
        P_species = species_pressure_tensor(velocities, masses, virial_species,
                                          species_id, num_species=2, volume=volume)
        assert P_species.shape == (2, 3, 3)
    """
    # Calculate kinetic contribution per species (vector/tensor)
    P_kin_species = fast_species_sum(
        pressure_kinetic_contribution(velocities, masses, volume),
        species_id, num_species
    )
    # Add virial contribution (assume virial_species_tensor is already per species)
    P_vir_species = virial_species_tensor / volume
    return P_kin_species + P_vir_species


# Thermodynamic potentials and state functions
@jit(nopython=True)
def internal_energy(kinetic_energy: ndarray, potential_energy: ndarray) -> ndarray:
    """Calculate internal energy per particle.
    
    The internal energy is the sum of kinetic and potential energies:
    
    .. math::
        U_i = KE_i + PE_i
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy with shape ``(N,)``
    potential_energy : numpy.ndarray
        Per-particle potential energy with shape ``(N,)``
        
    Returns
    -------
    numpy.ndarray
        Per-particle internal energy with shape ``(N,)``
        
    Examples
    --------
    Calculate internal energy:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import internal_energy
        
        ke = np.random.rand(1000)
        pe = np.random.rand(1000)
        
        u = internal_energy(ke, pe)
        assert u.shape == (1000,)
        assert np.allclose(u, ke + pe)
    """
    return kinetic_energy + potential_energy


def enthalpy_per_particle(internal_energy: ndarray, pressure: float, 
                         volume: float, num_particles: int) -> ndarray:
    """Calculate enthalpy per particle.
    
    The enthalpy per particle is calculated as:
    
    .. math::
        H_i = U_i + \\frac{P V}{N}
    
    where :math:`U_i` is the internal energy, :math:`P` is the pressure,
    :math:`V` is the volume, and :math:`N` is the number of particles.
    
    Parameters
    ----------
    internal_energy : numpy.ndarray
        Per-particle internal energy with shape ``(N,)``
    pressure : float
        System pressure
    volume : float
        System volume
    num_particles : int
        Number of particles
        
    Returns
    -------
    numpy.ndarray
        Per-particle enthalpy with shape ``(N,)``
        
    Examples
    --------
    Calculate enthalpy per particle:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import enthalpy_per_particle
        
        u = np.random.rand(1000)
        pressure = 1.0
        volume = 1000.0
        num_particles = 1000
        
        h = enthalpy_per_particle(u, pressure, volume, num_particles)
        assert h.shape == (1000,)
    """
    pv_per_particle = (pressure * volume) / num_particles
    return internal_energy + pv_per_particle


def species_enthalpy(kinetic_energy: ndarray, potential_energy: ndarray,
                    species_pressure: ndarray, species_volumes: ndarray,
                    species_id: ndarray, num_species: int) -> ndarray:
    """Calculate enthalpy for each species.
    
    The enthalpy for species :math:`s` is calculated as:
    
    .. math::
        H_s = U_s + P_s V_s
    
    where :math:`U_s = KE_s + PE_s` is the internal energy of species :math:`s`.
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy with shape ``(N,)``
    potential_energy : numpy.ndarray
        Per-particle potential energy with shape ``(N,)``
    species_pressure : numpy.ndarray
        Pressure per species with shape ``(num_species,)``
    species_volumes : numpy.ndarray
        Volume per species with shape ``(num_species,)``
    species_id : numpy.ndarray
        Species identifier for each particle with shape ``(N,)``
    num_species : int
        Number of species
        
    Returns
    -------
    numpy.ndarray
        Enthalpy per species with shape ``(num_species,)``
        
    Examples
    --------
    Calculate species enthalpies:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import species_enthalpy
        
        ke = np.random.rand(1000)
        pe = np.random.rand(1000)
        species_pressure = np.array([1.0, 1.5])
        species_volumes = np.array([500.0, 500.0])
        species_id = np.random.randint(0, 2, 1000)
        
        h_species = species_enthalpy(ke, pe, species_pressure, species_volumes,
                                   species_id, num_species=2)
        assert h_species.shape == (2,)
    """
    # Calculate internal energy per species
    species_ke = species_kinetic_energy(kinetic_energy, species_id, num_species)
    species_pe = species_kinetic_energy(potential_energy, species_id, num_species)
    species_u = species_ke + species_pe
    
    # Add PV contribution
    return species_u + species_pressure * species_volumes


# Equation of state calculations
def ideal_gas_pressure(number_density: float, temperature: float, kB: float) -> float:
    """Calculate pressure from ideal gas equation of state.
    
    The ideal gas equation of state is:
    
    .. math::
        P = n k_B T
    
    where :math:`n` is the number density, :math:`k_B` is the Boltzmann constant,
    and :math:`T` is the temperature.
    
    Parameters
    ----------
    number_density : float
        Number density
    temperature : float
        Temperature
    kB : float
        Boltzmann constant
        
    Returns
    -------
    float
        Pressure from ideal gas equation of state
        
    Examples
    --------
    Calculate ideal gas pressure:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import ideal_gas_pressure
        
        n = 1e20  # particles/m³
        T = 300.0  # K
        kB = 1.380649e-23  # J/K
        
        P = ideal_gas_pressure(n, T, kB)
        print(f"Pressure: {P:.2e} Pa")
    """
    return number_density * kB * temperature


def species_equation_of_state(species_number_density: ndarray, 
                             species_temperature: ndarray, kB: float) -> ndarray:
    """Calculate pressure for each species using equation of state.
    
    The pressure for each species is calculated using the ideal gas equation
    of state, assuming ideal gas behavior for each species independently.
    
    Parameters
    ----------
    species_number_density : numpy.ndarray
        Number density per species with shape ``(num_species,)``
    species_temperature : numpy.ndarray
        Temperature per species with shape ``(num_species,)``
    kB : float
        Boltzmann constant
        
    Returns
    -------
    numpy.ndarray
        Pressure per species with shape ``(num_species,)``
        
    Examples
    --------
    Calculate species pressures:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import species_equation_of_state
        
        n_species = np.array([1e20, 5e19])  # particles/m³
        T_species = np.array([300.0, 400.0])  # K
        kB = 1.380649e-23
        
        P_species = species_equation_of_state(n_species, T_species, kB)
        assert P_species.shape == (2,)
    """
    return species_number_density * kB * species_temperature


# Validation and utility functions
def validate_thermodynamic_inputs(velocities: ndarray, masses: ndarray) -> bool:
    """Validate inputs for thermodynamic calculations.
    
    This function performs various checks to ensure the inputs are valid
    for thermodynamic calculations:
    
    - Array shapes are compatible
    - Masses are positive
    - No NaN or infinite values
    - Reasonable velocity magnitudes
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities
    masses : numpy.ndarray
        Particle masses
        
    Returns
    -------
    bool
        True if inputs are valid for thermodynamic calculations
        
    Examples
    --------
    Validate thermodynamic inputs:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import validate_thermodynamic_inputs
        
        velocities = np.random.randn(1000, 3)
        masses = np.ones(1000)
        
        is_valid = validate_thermodynamic_inputs(velocities, masses)
        assert is_valid
    """
    # Check array shapes
    if velocities.shape[0] != masses.shape[0]:
        return False
    
    if velocities.shape[1] != 3:
        return False
    
    # Check for valid values
    if any(masses <= 0):
        return False
    
    if any(isnan(velocities)) or any(isnan(masses)):
        return False
    
    if any(isinf(velocities)) or any(isinf(masses)):
        return False
    
    # Check for reasonable velocity magnitudes (less than speed of light)
    velocity_magnitudes = norm(velocities, axis=1)
    if any(velocity_magnitudes > 3e8):  # Speed of light
        return False
    
    return True


def equipartition_check(kinetic_energy: ndarray, temperature: float, 
                       dimensions: int, kB: float) -> dict:
    """Check equipartition theorem: :math:`\\langle KE \\rangle = \\frac{d}{2} k_B T`
    
    This function validates the equipartition theorem by comparing the
    average kinetic energy with the theoretical expectation.
    
    Parameters
    ----------
    kinetic_energy : numpy.ndarray
        Per-particle kinetic energy
    temperature : float
        System temperature
    dimensions : int
        Number of degrees of freedom
    kB : float
        Boltzmann constant
        
    Returns
    -------
    dict
        Dictionary containing theoretical expectation, actual value, and deviation
        
    Examples
    --------
    Check equipartition theorem:
    
    .. code-block:: python
    
        import numpy as np
        from sarkas.physics.thermodynamics import equipartition_check
        
        ke = np.random.rand(1000)
        T = 300.0
        kB = 1.380649e-23
        
        result = equipartition_check(ke, T, dimensions=3, kB=kB)
        print(f"Deviation: {result['deviation']:.2e}")
    """
    theoretical_ke = 0.5 * dimensions * kB * temperature
    actual_ke = kinetic_energy.mean()
    deviation = abs(actual_ke - theoretical_ke) / theoretical_ke
    
    return {
        'theoretical_ke': theoretical_ke,
        'actual_ke': actual_ke,
        'deviation': deviation
    }


# Backward compatibility wrappers
def calculate_kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray:
    """Backward compatibility wrapper for kinetic energy calculation.
    
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
        Per-particle kinetic energy with shape ``(N,)``
    """
    return kinetic_energy(velocities, masses)


def calculate_temperature(velocities: ndarray, masses: ndarray, **kwargs) -> float:
    """Backward compatibility wrapper for temperature calculation.
    
    This function maintains the same interface as the original Particles
    class method for backward compatibility.
    
    Parameters
    ----------
    velocities : numpy.ndarray
        Particle velocities with shape ``(N, 3)``
    masses : numpy.ndarray
        Particle masses with shape ``(N,)``
    **kwargs
        Additional arguments (dimensions, kB, etc.)
        
    Returns
    -------
    float
        System temperature
    """
    ke = kinetic_energy(velocities, masses)
    return temperature_from_kinetic_energy(ke, **kwargs) 