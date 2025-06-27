## **Overview**

This phase extracts physics calculations from the monolithic `Particles` class into focused, reusable, and optimized calculator modules. The goal is to improve performance through Numba compilation while maintaining 100% backward compatibility.

**Duration**: 3 weeks  
**Risk Level**: Low (pure function extraction)  
**Expected Performance Gain**: 15-30% in physics calculations

### **Step 1: Create Physics Module Structure**

```bash
# Create new directory structure
mkdir -p sarkas/physics/tests
mkdir -p sarkas/physics/species_arrays

# Create module files
touch sarkas/physics/__init__.py
touch sarkas/physics/thermodynamics.py
touch sarkas/physics/species_arrays/__init__.py
touch sarkas/physics/species_arrays/soa_tensors.py
touch sarkas/physics/tests/__init__.py
touch sarkas/physics/tests/test_thermodynamics.py
touch sarkas/physics/tests/test_species_soa.py
```

### **Step 2: Extract Thermodynamics Functions**

#### **2.1 Implement SoA Species Arrays System**

**Prompt for Claude:**

> Create `sarkas/physics/species_arrays/soa_tensors.py` that implements Structure of Arrays for species-level high-dimensional arrays. This is specifically for species arrays (not particle arrays) where memory efficiency and vectorized operations provide significant benefits.
> 
> **Requirements:**
> 1. **Adaptive storage** - Use SoA for large species count, AoS for small species count
> 2. **Symmetric tensor optimization** - Exploit tensor symmetry to reduce memory usage
> 3. **Backward compatibility** - Provide tensor interface when needed
> 4. **Vectorized operations** - Optimize for component-wise calculations
> 5. **Memory efficiency** - Target 50-80% memory reduction for large species systems
> 
> **Classes to implement:**
> ```python
> class SpeciesTensorArrays:
>     """SoA implementation for species tensor quantities."""
>     
>     def __init__(self, num_species: int, soa_threshold: int = 50):
>         """Initialize with adaptive storage based on species count."""
>         
>     # Pressure tensor (3x3 symmetric)
>     def init_pressure_tensor(self):
>         """Initialize pressure tensor storage."""
>         
>     def get_pressure_tensor(self) -> ndarray:
>         """Get pressure tensor with backward compatibility."""
>         
>     def set_pressure_component(self, component: str, values: ndarray):
>         """Set pressure tensor component (xx, yy, zz, xy, xz, yz)."""
>         
>     def get_pressure_trace(self) -> ndarray:
>         """Fast vectorized pressure trace calculation."""
>         
>     # Virial tensor (3x3 symmetric)  
>     def init_virial_tensor(self):
>         """Initialize virial tensor storage."""
>         
>     def get_virial_tensor(self) -> ndarray:
>         """Get virial tensor with backward compatibility."""
>         
>     # Heat flux tensor (num_species x num_species x 3)
>     def init_heat_flux_tensor(self):
>         """Initialize heat flux tensor storage."""
>         
>     def get_heat_flux_tensor(self) -> ndarray:
>         """Get heat flux tensor with backward compatibility."""
>         
>     # Velocity moments (num_species x max_moments x 3)
>     def init_velocity_moments(self, max_moments: int):
>         """Initialize velocity moments storage."""
>         
>     def get_velocity_moments(self) -> ndarray:
>         """Get velocity moments with backward compatibility."""
> ```
> 
> **Memory optimization targets:**
> - **Pressure/Virial tensors**: Store only 6 components instead of 9 (use symmetry)
> - **Heat flux**: Use SoA layout for vectorized species operations
> - **Velocity moments**: Separate storage by moment order and dimension
> - **Adaptive threshold**: Use SoA only when `num_species > soa_threshold`
> 
> **Performance optimization:**
> ```python
> @jit(nopython=True)
> def elastic_energy_density(stress_tensor: ndarray, strain_tensor: ndarray) -> float
>     """Calculate elastic energy density: u = (1/2) * σ : ε"""
> 
> # Species interaction pressure (for mixtures)
> @jit(nopython=True)
> def cross_species_pressure(velocities_A: ndarray, velocities_B: ndarray,
>                           masses_A: ndarray, masses_B: ndarray,
>                           interaction_forces: ndarray, volume: float) -> ndarray
>     """Calculate pressure contribution from A-B species interactions."""
> 
> def species_pressure_matrix(velocities: ndarray, masses: ndarray, forces: ndarray,
>                           species_id: ndarray, num_species: int, volume: float,
>                           tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """
>     Calculate species-species pressure interaction matrix:
>     P_ij = pressure contribution from species i-j interactions
>     """
> 
> # Backward compatibility wrappers with SoA optimization
> def calculate_species_pressure_tensor(velocities: ndarray, virial_tensor: ndarray, 
>                                     species_id: ndarray, masses: ndarray, 
>                                     volume: float, use_soa: bool = True) -> ndarray
>     """Backward compatible species pressure tensor calculation with SoA option."""
> 
> def calculate_species_momentum(velocities: ndarray, masses: ndarray,
>                              species_id: ndarray, num_species: int,
>                              use_soa: bool = True) -> ndarray
>     """Backward compatible species momentum calculation with SoA optimization."""
> ```
> 
> **Enhanced mechanical property requirements:**
> - Complete pressure tensor decomposition by species and interaction type
> - Support for anisotropic pressure calculations (important for magnetized plasmas)
> - Cross-species interaction pressure for mixture analysis
> - Stress tensor calculations for material property determination
> - Integration with SoA species arrays for memory efficiency
> 
> **SoA mechanical properties optimization:**
> - Pressure tensor components stored separately for vectorized operations
> - Species-species pressure interaction matrices using efficient storage
> - Momentum calculations optimized for large species count
> - Stress tensor components using symmetric storage

### **Step 3: Update Particles Class Methods with Enhanced SoA Integration**

**Prompt for Claude:**

> Update the remaining physics calculation methods in `sarkas/particles.py` to use the new transport and mechanical calculators with comprehensive SoA integration:
> 
> **Enhanced methods to update:**
> ```python
> def calculate_species_electric_current(self):
>     """Enhanced electric current calculation with SoA optimization."""
>     if self._use_soa_species and self.num_species > 50:
>         from sarkas.physics.transport import calculate_species_electric_current_soa
>         self.species_electric_current = calculate_species_electric_current_soa(
>             self.vel, self.charges, self.id, self.num_species, self._species_arrays
>         )
>     else:
>         # Traditional calculation for small species count
>         self._calculate_species_electric_current_traditional()
> 
> def calculate_species_heat_flux(self):
>     """Enhanced heat flux calculation with comprehensive transport physics."""
>     # Calculate per-particle energies first
>     if not hasattr(self, 'kinetic_energy'):
>         self.calculate_kinetic_energy()
>     
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.transport import calculate_species_heat_flux_soa
>         calculate_species_heat_flux_soa(
>             self.vel, self.kinetic_energy, self.potential_energy,
>             self.id, self._species_arrays
>         )
>     else:
>         self._calculate_species_heat_flux_traditional()
> 
> def calculate_species_diffusion_flux(self):
>     """NEW: Comprehensive diffusion flux calculation for mixtures."""
>     if self.num_species < 2:
>         # No diffusion in single-component system
>         self.species_diffusion_flux = zeros((1, 3))
>         return
>     
>     # Calculate concentration gradients (requires spatial binning)
>     concentration_gradients = self._calculate_concentration_gradients()
>     
>     # Get transport coefficients (from parameters or estimation)
>     transport_coeffs = self._get_transport_coefficients()
>     
>     if self._use_soa_species and self.num_species > 10:
>         from sarkas.physics.transport import calculate_species_diffusion_flux_soa
>         self.species_diffusion_flux = calculate_species_diffusion_flux_soa(
>             self.vel, self.id, concentration_gradients, transport_coeffs, self._species_arrays
>         )
>     else:
>         from sarkas.physics.transport import multicomponent_diffusion_flux
>         self.species_diffusion_flux = multicomponent_diffusion_flux(
>             self.vel, self.id, concentration_gradients, transport_coeffs['diffusion_matrix']
>         )
> 
> def calculate_interdiffusion_coefficients(self):
>     """NEW: Calculate interdiffusion coefficients from particle motion."""
>     if self.num_species < 2:
>         return
>     
>     # Mean square displacement method
>     if hasattr(self, '_previous_positions'):
>         time_interval = self.current_time - self._previous_time
>         msd_matrix = self._calculate_species_msd_matrix(time_interval)
>         self.interdiffusion_matrix = msd_matrix / (6.0 * time_interval)  # 3D diffusion
>     
>     # Green-Kubo method (if velocity correlation data available)
>     if hasattr(self, 'velocity_correlation_history'):
>         self.interdiffusion_matrix_gk = self._calculate_diffusion_green_kubo()
> 
> def calculate_species_pressure_tensor(self):
>     """Enhanced pressure tensor with species decomposition and SoA."""
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.mechanical import pressure_tensor_species_decomposition
>         pressure_tensor_species_decomposition(
>             self.vel, self.masses, self.virial_species_tensor, self.id,
>             self.box_volume, self._species_arrays
>         )
>         
>         # Calculate additional pressure quantities
>         self._calculate_pressure_anisotropy()
>         self._calculate_cross_species_pressure()
>     else:
>         self._calculate_species_pressure_tensor_traditional()
> 
> def calculate_species_momentum(self):
>     """Enhanced momentum calculation with conservation checking."""
>     if self._use_soa_species and self.num_species > 50:
>         from sarkas.physics.mechanical import calculate_species_momentum
>         self.species_momentum = calculate_species_momentum(
>             self.vel, self.masses, self.id, self.num_species, use_soa=True
>         )
>     else:
>         self.species_momentum = self._aggregate_by_species_vector(
>             self.momentum if hasattr(self, 'momentum') else self.vel * self.masses[:, None]
>         )
>     
>     # Validate momentum conservation
>     total_momentum = self.species_momentum.sum(axis=0)
>     if np.linalg.norm(total_momentum) > 1e-12:
>         warn(f"Momentum conservation violated: |p_total| = {np.linalg.norm(total_momentum)}")
> 
> # NEW: Comprehensive transport property calculations
> def calculate_all_transport_properties(self):
>     """Calculate all transport properties in optimized order."""
>     # Basic quantities first
>     self.calculate_kinetic_energy()
>     self.calculate_species_momentum()
>     
>     # Transport fluxes
>     self.calculate_species_electric_current()
>     self.calculate_species_heat_flux()
>     
>     # Diffusion properties (for mixtures)
>     if self.num_species > 1:
>         self.calculate_species_diffusion_flux()
>         self.calculate_interdiffusion_coefficients()
>     
>     # Pressure properties
>     self.calculate_species_pressure_tensor()
> 
> # NEW: Enhanced species array management
> def _calculate_concentration_gradients(self):
>     """Calculate concentration gradients using spatial binning."""
>     # Implement spatial binning to calculate ∇c_i for each species
>     # This is needed for diffusion flux calculations
>     grid_size = int(self.total_num_ptcls**(1/3)) + 1
>     gradients = zeros((self.num_species, 3))
>     
>     # Implementation of concentration gradient calculation
>     # Using finite differences on spatial grid
>     return gradients
> 
> def _get_transport_coefficients(self):
>     """Get or estimate transport coefficients for diffusion calculations."""
>     # Return transport coefficients from parameters or estimation
>     if hasattr(self, 'transport_coefficients'):
>         return self.transport_coefficients
>     else:
>         # Estimate using kinetic theory or empirical relations
>         return self._estimate_transport_coefficients()
> 
> def _estimate_transport_coefficients(self):
>     """Estimate transport coefficients using kinetic theory."""
>     # Simple estimation for binary diffusion coefficients
>     # D_ij ≈ (3/16) * sqrt(π * kT * (m_i + m_j)/(2 * m_i * m_j)) / (n * σ_ij^2)
>     
>     diffusion_matrix = zeros((self.num_species, self.num_species))
>     thermal_diffusion_ratios = zeros(self.num_species)
>     
>     # Kinetic theory estimates
>     for i in range(self.num_species):
>         for j in range(i+1, self.num_species):
>             # Binary diffusion coefficient estimation
>             reduced_mass = (self.species_masses[i] * self.species_masses[j] / 
>                           (self.species_masses[i] + self.species_masses[j]))
>             temperature = self.species_temperature.mean() if hasattr(self, 'species_temperature') else 1.0
>             
>             # Simple hard sphere estimate (can be improved with actual cross-sections)
>             sigma_ij = 1e-10  # Placeholder collision cross-section
>             diffusion_matrix[i, j] = diffusion_matrix[j, i] = (
>                 (3.0/16.0) * sqrt(pi * self.kB * temperature / reduced_mass) / 
>                 (self.total_num_density * sigma_ij**2)
>             )
>     
>     return {
>         'diffusion_matrix': diffusion_matrix,
>         'thermal_diffusion_ratios': thermal_diffusion_ratios,
>         'pressure_diffusion_coeffs': zeros(self.num_species)
>     }
> 
> def _calculate_pressure_anisotropy(self):
>     """Calculate pressure anisotropy for each species."""
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         pressure_components = self._species_arrays.get_pressure_components()
>         # Calculate β = 1 - P_perp/P_parallel for each species
>         self.species_pressure_anisotropy = (
>             1.0 - (pressure_components['yy'] + pressure_components['zz']) / 
>             (2.0 * pressure_components['xx'])
>         )
> 
> def _calculate_cross_species_pressure(self):
>     """Calculate pressure contributions from cross-species interactions."""
>     if self.num_species > 1:
>         from sarkas.physics.mechanical import species_pressure_matrix
>         self.species_pressure_matrix = species_pressure_matrix(
>             self.vel, self.masses, self.acc * self.masses[:, None],  # forces
>             self.id, self.num_species, self.box_volume, self._species_arrays
>         )
> 
> # Enhanced memory and performance monitoring
> def get_soa_memory_report(self):
>     """Generate detailed memory usage report for SoA species arrays."""
>     if not hasattr(self, '_species_arrays'):
>         return "SoA species arrays not initialized"
>     
>     report = {
>         'num_species': self.num_species,
>         'soa_enabled': self._use_soa_species,
>         'memory_usage': {},
>         'estimated_traditional_memory': {},
>         'memory_savings': {}
>     }
>     
>     # Calculate memory usage for each tensor type
>     if hasattr(self._species_arrays, 'pressure_xx'):
>         soa_pressure_memory = (6 * self.num_species * 8)  # 6 components, 8 bytes each
>         traditional_pressure_memory = (self.num_species * 3 * 3 * 8)  # Full tensor
>         
>         report['memory_usage']['pressure_tensor'] = f"{soa_pressure_memory} bytes"
>         report['estimated_traditional_memory']['pressure_tensor'] = f"{traditional_pressure_memory} bytes"
>         report['memory_savings']['pressure_tensor'] = f"{(1 - soa_pressure_memory/traditional_pressure_memory)*100:.1f}%"
>     
>     return report
> 
> def benchmark_soa_performance(self, iterations=10):
>     """Benchmark SoA vs traditional performance for current system."""
>     import time
>     
>     if self.num_species < 10:
>         return "Benchmark requires >= 10 species for meaningful comparison"
>     
>     results = {}
>     
>     # Benchmark pressure tensor calculation
>     # SoA version
>     times_soa = []
>     for _ in range(iterations):
>         start = time.perf_counter()
>         self.calculate_species_pressure_tensor()  # Uses SoA
>         times_soa.append(time.perf_counter() - start)
>     
>     # Traditional version (temporarily disable SoA)
>     self._use_soa_species = False
>     times_traditional = []
>     for _ in range(iterations):
>         start = time.perf_counter()
>         self._calculate_species_pressure_tensor_traditional()
>         times_traditional.append(time.perf_counter() - start)
>     
>     self._use_soa_species = True  # Re-enable
>     
>     results['pressure_tensor'] = {
>         'soa_time': f"{np.mean(times_soa)*1000:.3f} ± {np.std(times_soa)*1000:.3f} ms",
>         'traditional_time': f"{np.mean(times_traditional)*1000:.3f} ± {np.std(times_traditional)*1000:.3f} ms",
>         'speedup': f"{np.mean(times_traditional)/np.mean(times_soa):.2f}x"
>     }
>     
>     return results
> ```
> 
> **Integration requirements:**
> - All existing tests must still pass without modification
> - Results must be numerically identical for SoA vs traditional (within machine precision)
> - Automatic SoA usage based on species count thresholds
> - Comprehensive diffusion physics for multi-component systems
> - Enhanced transport property calculations with cross-effects
> - Performance monitoring and benchmarking capabilities
> - Memory usage reporting and optimization validation

---

## **Week 3: Species Aggregation Utilities with SoA Optimization**

### **Step 1: Create Enhanced Aggregation Module**

**Prompt for Claude:**

> Create `sarkas/physics/aggregation.py` that provides reusable utilities for aggregating per-particle quantities by species, with special optimization for SoA species arrays and high-dimensional tensor operations.
> 
> **Functions to implement:**
> ```python
> # Basic aggregation functions (optimized with numba)
> @jit(nopython=True)
> def species_sum(per_particle_quantity: ndarray, species_id: ndarray, num_species: int) -> ndarray
>     """Sum per-particle quantities by species using optimized bincount."""
> 
> @jit(nopython=True)  
> def species_mean(per_particle_quantity: ndarray, species_id: ndarray, 
>                 species_num: ndarray) -> ndarray
>     """Calculate mean per-particle quantities by species."""
> 
> @jit(nopython=True)
> def species_weighted_sum(per_particle_quantity: ndarray, weights: ndarray,
>                         species_id: ndarray, num_species: int) -> ndarray
>     """Weighted sum by species (e.g., mass-weighted averages)."""
> 
> # Vector aggregation (for 3D quantities)
> @jit(nopython=True)
> def species_vector_sum(per_particle_vectors: ndarray, species_id: ndarray, 
>                       num_species: int) -> ndarray
>     """Sum vector quantities by species (e.g., momentum, current)."""
> 
> @jit(nopython=True)
> def species_vector_mean(per_particle_vectors: ndarray, species_id: ndarray,
>                        species_num: ndarray) -> ndarray
>     """Calculate mean vector quantities by species."""
> 
> # Tensor aggregation (for pressure, stress tensors)
> @jit(nopython=True)
> def species_tensor_sum(per_particle_tensors: ndarray, species_id: ndarray,
>                       num_species: int) -> ndarray
>     """Sum tensor quantities by species (per-particle tensors -> species tensors)."""
> 
> def species_tensor_aggregation_soa(per_particle_tensors: ndarray, species_id: ndarray,
>                                   tensor_arrays: SpeciesTensorArrays):
>     """
>     Aggregate tensors directly into SoA storage for memory efficiency.
>     Avoids creating intermediate full tensors.
>     """
> 
> # Advanced aggregation functions
> @jit(nopython=True)
> def species_histogram(values: ndarray, species_id: ndarray, bins: ndarray,
>                      num_species: int) -> ndarray
>     """Create histograms of values for each species."""
> 
> @jit(nopython=True)
> def species_variance(per_particle_quantity: ndarray, species_id: ndarray,
>                     species_num: ndarray) -> ndarray
>     """Calculate variance within each species."""
> 
> @jit(nopython=True)
> def species_correlation_matrix(quantity_A: ndarray, quantity_B: ndarray,
>                               species_id: ndarray, num_species: int) -> ndarray
>     """Calculate correlation matrix between quantities for each species."""
> 
> # Cross-species interaction aggregation
> @jit(nopython=True)
> def cross_species_interaction_sum(per_pair_quantities: ndarray, 
>                                  species_i: ndarray, species_j: ndarray,
>                                  num_species: int) -> ndarray
>     """
>     Aggregate pair interactions into species-species matrix.
>     Used for virial pressure contributions, interaction energies, etc.
>     """
> 
> @jit(nopython=True)
> def species_pair_force_matrix(forces: ndarray, species_id: ndarray,
>                              interaction_pairs: ndarray, num_species: int) -> ndarray
>     """Create species-species force interaction matrix."""
> 
> # SoA-optimized aggregation functions
> def aggregate_to_soa_pressure_tensor(per_particle_stress: ndarray, species_id: ndarray,
>                                     tensor_arrays: SpeciesTensorArrays):
>     """
>     Aggregate per-particle stress directly into SoA pressure tensor components.
>     Much more memory efficient than creating intermediate full tensors.
>     """
> 
> def aggregate_to_soa_heat_flux(per_particle_heat_flux: ndarray, species_id: ndarray,
>                               tensor_arrays: SpeciesTensorArrays):
>     """Aggregate heat flux directly into SoA storage."""
> 
> def aggregate_cross_species_virial_soa(virial_contributions: ndarray, 
>                                       species_pairs: ndarray,
>                                       tensor_arrays: SpeciesTensorArrays):
>     """Aggregate cross-species virial contributions directly into SoA storage."""
> 
> # Advanced species analysis functions
> @jit(nopython=True)
> def species_radial_distribution(positions: ndarray, species_id: ndarray,
>                                box_lengths: ndarray, bins: ndarray) -> ndarray
>     """Calculate radial distribution function between and within species."""
> 
> @jit(nopython=True)
> def species_spatial_distribution(positions: ndarray, species_id: ndarray,
>                                 grid_shape: tuple, box_lengths: ndarray) -> ndarray
>     """Calculate spatial distribution of each species on a grid."""
> 
> @jit(nopython=True)
> def species_velocity_distribution(velocities: ndarray, species_id: ndarray,
>                                  velocity_bins: ndarray, num_species: int) -> ndarray
>     """Calculate velocity distribution for each species."""
> 
> # Flux aggregation for transport properties
> @jit(nopython=True)
> def aggregate_diffusion_flux_matrix(particle_flux: ndarray, species_id: ndarray,
>                                    num_species: int) -> ndarray
>     """
>     Aggregate particle fluxes into species diffusion flux matrix.
>     Returns J_ij matrix where J_ij is flux of species i due to species j.
>     """
> 
> @jit(nopython=True)
> def aggregate_transport_coefficients(velocities: ndarray, forces: ndarray,
>                                     species_id: ndarray, temperature: ndarray,
>                                     num_species: int) -> dict
>     """
>     Aggregate microscopic quantities to calculate transport coefficients:
>     - Self-diffusion coefficients
>     - Mutual diffusion coefficients  
>     - Thermal diffusion ratios
>     - Electrical conductivity
>     """
> 
> # Performance comparison utilities
> def benchmark_aggregation_methods(per_particle_data: ndarray, species_id: ndarray,
>                                  num_species: int, method: str = 'all') -> dict
>     """
>     Benchmark different aggregation methods:
>     - Traditional loops
>     - NumPy bincount
>     - Numba compiled functions
>     - SoA-optimized aggregation
>     """
> 
> # Validation and debugging utilities
> def validate_species_aggregation(per_particle_quantity: ndarray, 
>                                 aggregated_quantity: ndarray,
>                                 species_id: ndarray, species_num: ndarray) -> bool
>     """Validate that aggregation conserves total quantities."""
> 
> def debug_species_distribution(species_id: ndarray, num_species: int) -> dict
>     """Debug species distribution - check for load balancing issues."""
> 
> # Backward compatibility wrappers
> def species_summation(per_particle_data: ndarray, species_id: ndarray, 
>                      num_species: int) -> ndarray
>     """Backward compatible species summation."""
>     return species_sum(per_particle_data, species_id, num_species)
> ```
> 
> **SoA Integration Requirements:**
> - Direct aggregation into SoA tensor storage to avoid memory overhead
> - Component-wise tensor aggregation for symmetric tensors
> - Efficient handling of sparse species-species interaction matrices
> - Memory-conscious aggregation for large species count systems
> 
> **Performance Requirements:**
> - All aggregation functions compiled with Numba for maximum speed
> - Vectorized operations using NumPy bincount where possible
> - Optimized loop ordering for cache efficiency
> - Minimal temporary array creation
> - Support for weighted aggregation (mass-weighted, charge-weighted, etc.)
> 
> **Transport Physics Requirements:**
> - Aggregation functions specific to transport coefficient calculation
> - Support for Green-Kubo relation calculations
> - Cross-species interaction aggregation for mixture properties
> - Flux matrix aggregation for multicomponent diffusion)
> def calculate_pressure_trace_soa(pressure_xx, pressure_yy, pressure_zz):
>     """Fast vectorized trace calculation."""
>     return (pressure_xx + pressure_yy + pressure_zz) / 3.0
> 
> @jit(nopython=True)
> def calculate_pressure_deviatoric_soa(pressure_diagonal, pressure_off_diag):
>     """Fast deviatoric stress calculation."""
>     # Implementation here
> 
> @jit(nopython=True)
> def tensor_component_operations(component_arrays):
>     """Vectorized operations on tensor components."""
>     # Max, mean, std per component across species
> ```
> 
> **Backward compatibility requirements:**
> - Existing code accessing `species_pressure_tensor[i, j, k]` must work
> - Lazy reconstruction of full tensors only when explicitly accessed
> - Performance warning for inefficient usage patterns
> - Seamless fallback to AoS for small species counts

#### **2.2 Implement Pure Thermodynamics Calculator**

**Prompt for Claude:**

> Create `sarkas/physics/thermodynamics.py` that extracts thermodynamics calculations from the Sarkas `Particles` class, now incorporating the SoA species arrays system. Follow these requirements:
> 
> **Requirements:**
> 1. **Pure functions only** - No class state, only input/output
> 2. **Numba compilation** - Use `@jit(nopython=True)` for all calculation kernels
> 3. **NumPy optimization** - Prefer vectorized operations over loops where possible
> 4. **SoA integration** - Work efficiently with species SoA arrays
> 5. **Comprehensive docstrings** - Include physics equations and units
> 6. **Backward compatibility** - Provide wrapper functions matching current interface
> 
> **Functions to implement:**
> ```python
> # Core per-particle calculations
> @jit(nopython=True)
> def kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray
>     """Calculate kinetic energy: KE = 0.5 * m * v²"""
> 
> @jit(nopython=True)
> def temperature_from_kinetic_energy(ke: ndarray, masses: ndarray, dimensions: int, kB: float) -> ndarray
>     """Calculate temperature: T = 2*KE / (dimensions * kB)"""
> 
> @jit(nopython=True)
> def momentum(velocities: ndarray, masses: ndarray) -> ndarray
>     """Calculate momentum: p = m * v"""
> 
> @jit(nopython=True)
> def center_of_mass_velocity(velocities: ndarray, masses: ndarray) -> ndarray
>     """Calculate center of mass velocity."""
> 
> @jit(nopython=True)
> def remove_center_of_mass_motion(velocities: ndarray, masses: ndarray) -> ndarray
>     """Remove center of mass motion from velocities."""
> 
> # Species-level calculations with SoA optimization
> def calculate_species_kinetic_energy_soa(kinetic_energy: ndarray, species_id: ndarray, 
>                                         tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """Calculate species kinetic energy using SoA species arrays."""
> 
> def calculate_species_temperature_soa(kinetic_energy: ndarray, species_id: ndarray,
>                                      species_masses: ndarray, dimensions: int, kB: float,
>                                      tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """Calculate species temperature using SoA optimization."""
> 
> # Pressure tensor calculations
> @jit(nopython=True)
> def pressure_tensor_kinetic_contribution(velocities: ndarray, masses: ndarray, 
>                                         volume: float, dimensions: int) -> ndarray
>     """Calculate kinetic contribution to pressure tensor."""
> 
> def pressure_tensor_virial_contribution_soa(virial_tensor_components: dict, 
>                                            volume: float) -> dict
>     """Calculate virial contribution using SoA virial components."""
> 
> def calculate_species_pressure_tensor_soa(velocities: ndarray, masses: ndarray,
>                                          virial_tensor: ndarray, species_id: ndarray,
>                                          volume: float, tensor_arrays: SpeciesTensorArrays):
>     """Calculate species pressure tensor with SoA optimization."""
> 
> # Advanced thermodynamic quantities
> def calculate_enthalpy_soa(kinetic_energy: ndarray, potential_energy: ndarray,
>                           pressure_components: dict, volume: float,
>                           species_id: ndarray, tensor_arrays: SpeciesTensorArrays):
>     """Calculate species enthalpy: H = U + PV"""
> 
> # Backward compatibility wrappers
> def calculate_kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray
>     """Backward compatible wrapper."""
>     return kinetic_energy(velocities, masses)
> 
> def calculate_species_temperature(kinetic_energy: ndarray, species_id: ndarray,
>                                  species_masses: ndarray, **kwargs) -> ndarray
>     """Backward compatible species temperature calculation."""
> ```
> 
> **SoA Integration Guidelines:**
> - Functions should accept both traditional arrays and SoA tensor components
> - Optimize for component-wise operations when using SoA
> - Provide efficient species aggregation using optimized algorithms
> - Include performance comparison functions (SoA vs traditional)
> 
> **Performance targets:**
> - 20-40% faster than current implementation through Numba compilation
> - 30-60% faster species aggregation for large species count through SoA
> - 50-80% memory reduction for species tensors when using SoA
> - Zero memory allocations in compiled kernels (except for output arrays)

#### **2.3 Create Comprehensive Tests with SoA Validation**

**Prompt for Claude:**

> Create comprehensive tests in `sarkas/physics/tests/test_thermodynamics.py` and `sarkas/physics/tests/test_species_soa.py` for the thermodynamics calculator and SoA species arrays. Requirements:
> 
> **Test Coverage for thermodynamics.py:**
> 1. **Physics validation** - Verify correct physics (energy conservation, equipartition theorem)
> 2. **SoA integration** - Test calculations work correctly with SoA species arrays
> 3. **Performance tests** - Compare SoA vs traditional approaches for different species counts
> 4. **Edge cases** - Empty arrays, single particles, zero velocities, large species count
> 5. **Numerical precision** - Verify results match analytical solutions
> 
> **Test Coverage for SoA species arrays:**
> 1. **Memory efficiency** - Verify memory usage reduction for large species count
> 2. **Backward compatibility** - Ensure tensor interface works exactly like traditional arrays
> 3. **Adaptive storage** - Test automatic SoA/AoS selection based on species count
> 4. **Tensor operations** - Verify symmetric tensor storage and reconstruction
> 5. **Component access** - Test efficient component-wise operations
> 
> **Specific test cases:**
> ```python
> # Physics validation
> def test_kinetic_energy_conservation()         # Energy conservation laws
> def test_temperature_equipartition_theorem()   # Verify 1/2 kT per degree of freedom
> def test_pressure_tensor_symmetry()            # Verify pressure tensor symmetry
> def test_species_thermodynamics_consistency()  # Sum of species = total
> 
> # SoA specific tests
> def test_soa_memory_usage_large_species()      # Memory efficiency for 1000+ species
> def test_soa_pressure_tensor_reconstruction()  # Backward compatibility
> def test_soa_component_operations()            # Fast component-wise calculations
> def test_adaptive_storage_threshold()          # Automatic SoA/AoS selection
> 
> # Performance benchmarks
> def test_soa_vs_aos_performance()              # Speed comparison across species counts
> def test_species_aggregation_performance()     # Aggregation speed with SoA
> def test_tensor_operation_performance()        # Tensor ops with SoA vs traditional
> 
> # Integration tests
> def test_full_thermodynamics_calculation_soa() # End-to-end with SoA
> def test_backward_compatibility_interface()    # Existing interface works
> ```
> 
> **Performance benchmarking:**
> ```python
> def benchmark_species_tensor_performance():
>     """Benchmark SoA vs traditional for different species counts."""
>     species_counts = [10, 50, 100, 500, 1000, 5000]
>     
>     for num_species in species_counts:
>         # Test memory usage
>         memory_traditional = measure_memory_traditional_tensors(num_species)
>         memory_soa = measure_memory_soa_tensors(num_species)
>         
>         # Test calculation speed
>         time_traditional = benchmark_pressure_calculations_traditional(num_species)
>         time_soa = benchmark_pressure_calculations_soa(num_species)
>         
>         print(f"Species {num_species}:")
>         print(f"  Memory: Traditional={memory_traditional}MB, SoA={memory_soa}MB")
>         print(f"  Speed: Traditional={time_traditional}ms, SoA={time_soa}ms")
> ```
> 
> **Validation requirements:**
> - All physics calculations must be numerically identical (within machine precision)
> - SoA and traditional approaches must give identical results
> - Performance improvements must be measurable and documented
> - Memory usage reduction must be verified for large species counts

#### **2.4 Update Particles Class with SoA Integration**

**Prompt for Claude:**

> Update the `sarkas/particles.py` file to use the new thermodynamics calculator and SoA species arrays while maintaining 100% backward compatibility. Requirements:
> 
> **Migration strategy:**
> 1. **Import new systems** at the top of the file
> 2. **Keep all existing method signatures** exactly the same
> 3. **Integrate SoA species arrays** for high-dimensional species quantities
> 4. **Maintain all attributes** that user code expects
> 5. **Add performance toggles** for gradual migration and testing
> 
> **Key integration points:**
> ```python
> class Particles:
>     def __init__(self):
>         # ... existing initialization
>         
>         # Add SoA species arrays for high-dimensional quantities
>         self._species_arrays = None  # Initialized in setup()
>         self._use_soa_species = True  # Enable SoA for species arrays
>         
>         # Import thermodynamics calculator
>         from sarkas.physics.thermodynamics import ThermodynamicsCalculator
>         self._thermo_calc = ThermodynamicsCalculator()
> 
>     def setup(self, parameters, species):
>         """Enhanced setup with SoA species arrays."""
>         # ... existing setup code
>         
>         # Initialize SoA species arrays for high-dimensional quantities
>         if self._use_soa_species:
>             from sarkas.physics.species_arrays.soa_tensors import SpeciesTensorArrays
>             self._species_arrays = SpeciesTensorArrays(self.num_species)
>             
>             # Initialize high-dimensional arrays
>             self._species_arrays.init_pressure_tensor()
>             self._species_arrays.init_virial_tensor() 
>             self._species_arrays.init_heat_flux_tensor()
>             if "Velocity Moments" in self.observables_list:
>                 self._species_arrays.init_velocity_moments(self.max_velocity_distribution_moment)
> 
>     # Updated methods to use new calculator + SoA
>     def calculate_kinetic_energy(self):
>         """Enhanced method using new thermodynamics calculator."""
>         from sarkas.physics.thermodynamics import kinetic_energy
>         self.kinetic_energy = kinetic_energy(self.vel, self.masses)
> 
>     def calculate_species_kinetic_temperature(self):
>         """Enhanced method using SoA species arrays."""
>         # Calculate per-particle kinetic energy
>         self.calculate_kinetic_energy()
>         
>         if self._use_soa_species and self.num_species > 50:
>             # Use SoA optimization for large species count
>             from sarkas.physics.thermodynamics import calculate_species_temperature_soa
>             self.species_temperature = calculate_species_temperature_soa(
>                 self.kinetic_energy, self.id, self.species_masses,
>                 self.dimensions, self.kB, self._species_arrays
>             )
>         else:
>             # Use traditional approach for small species count
>             self._calculate_species_temperature_traditional()
> 
>     def calculate_species_pressure_tensor(self):
>         """Enhanced method with SoA pressure tensor storage."""
>         if self._use_soa_species and hasattr(self, '_species_arrays'):
>             # Calculate using SoA optimization
>             from sarkas.physics.thermodynamics import calculate_species_pressure_tensor_soa
>             calculate_species_pressure_tensor_soa(
>                 self.vel, self.masses, self.virial_species_tensor, self.id,
>                 self.box_volume, self._species_arrays
>             )
>         else:
>             # Traditional calculation
>             self._calculate_species_pressure_tensor_traditional()
> 
>     # Backward compatibility properties
>     @property
>     def species_pressure_tensor(self):
>         """Backward compatible access to pressure tensor."""
>         if self._use_soa_species and hasattr(self, '_species_arrays'):
>             return self._species_arrays.get_pressure_tensor()
>         else:
>             return self._species_pressure_tensor
> 
>     @property
>     def species_virial_tensor(self):
>         """Backward compatible access to virial tensor."""
>         if self._use_soa_species and hasattr(self, '_species_arrays'):
>             return self._species_arrays.get_virial_tensor()
>         else:
>             return self._species_virial_tensor
> 
>     @property  
>     def species_heat_flux_tensor(self):
>         """Backward compatible access to heat flux tensor."""
>         if self._use_soa_species and hasattr(self, '_species_arrays'):
>             return self._species_arrays.get_heat_flux_tensor()
>         else:
>             return self._species_heat_flux_tensor
> 
>     # Performance comparison methods
>     def benchmark_soa_vs_traditional(self):
>         """Compare SoA vs traditional performance for current system."""
>         # Implementation for performance comparison
> 
>     def get_memory_usage_report(self):
>         """Report memory usage for species arrays."""
>         # Implementation for memory usage analysis
> ```
> 
> **Implementation requirements:**
> - All existing tests must still pass without modification
> - Results must be numerically identical (within machine precision)
> - SoA usage should be automatic based on species count
> - Performance toggle: `particles.set_soa_species_arrays(enabled=True/False)`
> - Memory usage should be monitored and reported

---

## **Week 2: Transport Properties Calculator**

### **Step 1: Create Transport Calculator with Diffusion Support**

**Prompt for Claude:**

> Create `sarkas/physics/transport.py` that extracts transport property calculations from the Particles class, with special focus on inter-diffusion flux calculations for mixtures. This module should handle electric current, heat flux, and comprehensive diffusion calculations.
> 
> **Functions to implement:**
> ```python
> # Electric current calculations
> @jit(nopython=True)
> def electric_current_density(velocities: ndarray, charges: ndarray) -> ndarray
>     """Calculate electric current density: J = ρ * v = q * n * v"""
> 
> @jit(nopython=True)
> def electric_current_vector(velocities: ndarray, charges: ndarray) -> ndarray
>     """Calculate electric current vector per particle."""
> 
> # Heat flux calculations  
> @jit(nopython=True)
> def heat_flux_vector(velocities: ndarray, kinetic_energy: ndarray, 
>                     potential_energy: ndarray) -> ndarray
>     """Calculate heat flux vector: q = Σ(E_i * v_i) where E_i = KE_i + PE_i"""
> 
> @jit(nopython=True)
> def heat_flux_tensor(velocities: ndarray, kinetic_energy: ndarray, 
>                     stress_tensor: ndarray, volume: float) -> ndarray
>     """Calculate heat flux tensor including convective and conductive parts."""
> 
> # Comprehensive diffusion calculations for mixtures
> @jit(nopython=True)
> def binary_diffusion_flux(velocities_A: ndarray, velocities_B: ndarray,
>                          number_density_A: float, number_density_B: float,
>                          concentration_gradient: ndarray) -> ndarray
>     """Calculate binary diffusion flux: J_A = -D_AB * ∇c_A"""
> 
> @jit(nopython=True)
> def multicomponent_diffusion_flux(velocities: ndarray, species_id: ndarray,
>                                  concentration_gradients: ndarray,
>                                  diffusion_matrix: ndarray) -> ndarray
>     """Calculate multicomponent diffusion flux: J_i = -Σ_j D_ij * ∇c_j"""
> 
> @jit(nopython=True)
> def thermal_diffusion_flux(velocities: ndarray, temperature_gradient: ndarray,
>                           species_id: ndarray, thermal_diffusion_ratios: ndarray) -> ndarray
>     """Calculate thermal diffusion (Soret effect): J_T = -D_T * ∇T"""
> 
> @jit(nopython=True)
> def pressure_diffusion_flux(velocities: ndarray, pressure_gradient: ndarray,
>                            species_id: ndarray, masses: ndarray) -> ndarray
>     """Calculate pressure diffusion (baro-diffusion): J_p = -D_p * ∇p"""
> 
> # Combined diffusion flux (Stefan-Maxwell equations)
> def stefan_maxwell_diffusion_flux(velocities: ndarray, species_id: ndarray,
>                                  concentration_gradients: ndarray,
>                                  temperature_gradient: ndarray,
>                                  pressure_gradient: ndarray,
>                                  transport_coefficients: dict) -> ndarray
>     """
>     Calculate total diffusion flux using Stefan-Maxwell equations:
>     ∇(μ_i/T) = Σ_j (x_i*x_j/D_ij) * (v_j - v_i) + (other driving forces)
>     
>     Parameters
>     ----------
>     transport_coefficients : dict
>         Contains 'diffusion_matrix', 'thermal_diffusion_ratios', 'pressure_diffusion_coeffs'
>     """
> 
> # Species-level transport calculations with SoA optimization
> def calculate_species_electric_current_soa(velocities: ndarray, charges: ndarray,
>                                           species_id: ndarray, num_species: int,
>                                           tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """Calculate species electric current using SoA optimization."""
> 
> def calculate_species_heat_flux_soa(velocities: ndarray, kinetic_energy: ndarray,
>                                   potential_energy: ndarray, species_id: ndarray,
>                                   tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """Calculate species heat flux using SoA storage."""
> 
> def calculate_species_diffusion_flux_soa(velocities: ndarray, species_id: ndarray,
>                                         concentration_gradients: ndarray,
>                                         transport_coefficients: dict,
>                                         tensor_arrays: SpeciesTensorArrays) -> ndarray
>     """Calculate species diffusion fluxes with SoA optimization."""
> 
> # Flux coupling calculations (Onsager relations)
> def calculate_coupled_fluxes(driving_forces: dict, onsager_coefficients: ndarray,
>                             species_id: ndarray) -> dict
>     """
>     Calculate coupled transport fluxes using Onsager reciprocal relations:
>     J_i = Σ_j L_ij * X_j
>     
>     Parameters
>     ----------
>     driving_forces : dict
>         Contains 'concentration_gradients', 'temperature_gradient', 'electric_field'
>     onsager_coefficients : ndarray
>         Onsager transport coefficient matrix L_ij
>     """
> 
> # Advanced transport phenomena
> @jit(nopython=True)
> def calculate_viscous_flux(velocity_gradients: ndarray, viscosity_tensor: ndarray) -> ndarray
>     """Calculate viscous momentum flux: τ_ij = -η_ijkl * (∂v_k/∂x_l)"""
> 
> @jit(nopython=True)  
> def calculate_electromagnetic_flux(electric_field: ndarray, magnetic_field: ndarray,
>                                   velocities: ndarray, charges: ndarray) -> tuple
>     """Calculate electromagnetic momentum flux (Lorentz force contribution)."""
> 
> # Backward compatibility wrappers
> def calculate_electric_current(velocities: ndarray, charges: ndarray) -> ndarray
>     """Backward compatible electric current calculation."""
>     return electric_current_vector(velocities, charges)
> 
> def calculate_species_electric_current(velocities: ndarray, charges: ndarray, 
>                                       species_id: ndarray, num_species: int) -> ndarray
>     """Backward compatible species electric current calculation."""
> ```
> 
> **Diffusion flux calculation requirements:**
> - Support for binary and multicomponent systems
> - Include all driving forces: concentration, temperature, pressure gradients
> - Implement Stefan-Maxwell framework for rigorous multicomponent diffusion
> - Support coupled transport phenomena (cross-effects)
> - Optimize for plasma systems with multiple ion species
> 
> **SoA integration for transport quantities:**
> - Electric current vectors stored efficiently for large species count
> - Heat flux tensors using symmetric storage
> - Diffusion flux matrices optimized for species-pair interactions
> - Memory-efficient storage for transport coefficient matrices
> 
> **Performance requirements:**
> - All functions use `@jit(nopython=True)` for maximum speed
> - Vectorized operations for species-level calculations
> - Efficient matrix operations for multicomponent diffusion
> - Minimal memory allocations in transport flux calculations

### **Step 2: Create Enhanced Mechanical Properties Calculator**

**Prompt for Claude:**

> Create `sarkas/physics/mechanical.py` for mechanical property calculations (pressure, stress, momentum) with enhanced support for species-level quantities and SoA optimization.
> 
> **Functions to implement:**
> ```python
> # Momentum calculations
> @jit(nopython=True)
> def momentum_vector(velocities: ndarray, masses: ndarray) -> ndarray
>     """Calculate momentum vector: p = m * v"""
> 
> @jit(nopython=True)
> def angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray,
>                     center_of_mass: ndarray) -> ndarray
>     """Calculate angular momentum: L = r × p"""
> 
> @jit(nopython=True)
> def species_momentum_balance(velocities: ndarray, masses: ndarray, forces: ndarray,
>                            species_id: ndarray, dt: float) -> ndarray
>     """Calculate momentum balance per species for validation."""
> 
> # Enhanced pressure calculations with species decomposition
> @jit(nopython=True)
> def pressure_tensor_kinetic(velocities: ndarray, masses: ndarray, volume: float) -> ndarray
>     """Calculate kinetic contribution to pressure tensor: P_kin = (1/V) * Σ m_i * v_i ⊗ v_i"""
> 
> @jit(nopython=True)
> def pressure_tensor_virial(virial_tensor: ndarray, volume: float) -> ndarray
>     """Calculate virial contribution to pressure tensor: P_vir = (1/V) * Σ r_i ⊗ F_i"""
> 
> @jit(nopython=True)
> def pressure_tensor_total(velocities: ndarray, masses: ndarray, virial_tensor: ndarray,
>                          volume: float) -> ndarray
>     """Calculate total pressure tensor: P = P_kin + P_vir"""
> 
> def pressure_tensor_species_decomposition(velocities: ndarray, masses: ndarray,
>                                         virial_tensor: ndarray, species_id: ndarray,
>                                         volume: float, tensor_arrays: SpeciesTensorArrays):
>     """Calculate species-decomposed pressure tensor with SoA optimization."""
> 
> # Advanced pressure quantities
> @jit(nopython=True)
> def pressure_scalar(pressure_tensor: ndarray, dimensions: int) -> float
>     """Calculate scalar pressure: P = (1/d) * Tr(P_tensor)"""
> 
> @jit(nopython=True)
> def pressure_deviatoric(pressure_tensor: ndarray) -> ndarray
>     """Calculate deviatoric pressure tensor: P_dev = P - (1/3)*Tr(P)*I"""
> 
> @jit(nopython=True)
> def pressure_anisotropy(pressure_tensor: ndarray) -> float
>     """Calculate pressure anisotropy: β = 1 - P_perp/P_parallel"""
> 
> # Species-specific pressure calculations
> def calculate_species_pressure_contributions(velocities: ndarray, masses: ndarray,
>                                            species_id: ndarray, volume: float,
>                                            tensor_arrays: SpeciesTensorArrays) -> dict
>     """
>     Calculate detailed pressure contributions by species:
>     - Kinetic pressure per species
>     - Cross-species interaction pressure
>     - Species pressure tensor decomposition
>     """
> 
> # Stress tensor calculations
> @jit(nopython=True)
> def stress_tensor_elastic(strain_tensor: ndarray, elastic_moduli: ndarray) -> ndarray
>     """Calculate elastic stress tensor: σ = C : ε"""
> 
> @jit(nopython=True)
> def stress_tensor_viscous(velocity_gradients: ndarray, viscosity_coefficients: ndarray) -> ndarray
>     """Calculate viscous stress tensor: σ_visc = η * (∇v + ∇v^T)"""
> 
> @jit(nopython=True)
> def stress_tensor_total(elastic_stress: ndarray, viscous_stress: ndarray,
>                        pressure_tensor: ndarray) -> ndarray
>     """Calculate total stress tensor: σ_total = σ_elastic + σ_viscous - P*I"""
> 
> # Species stress tensor with SoA
> def calculate_species_stress_tensor_soa(forces: ndarray, positions: ndarray,
>                                       species_id: ndarray, volume: float,
>                                       tensor_arrays: SpeciesTensorArrays):
>     """Calculate species stress tensor using SoA storage."""
> 
> # Advanced mechanical quantities
> @jit(nopython=True)
> def bulk_modulus(pressure_tensor: ndarray, volume_strain: float) -> float
>     """Calculate bulk modulus: K = -V * (∂P/∂V)"""
> 
> @jit(nopython=True)
> def shear_modulus(stress_tensor: ndarray, shear_strain: ndarray) -> float
>     """Calculate shear modulus: G = σ_shear / γ_shear"""
> 
> @jit(nopython=True# Phase 1: Physics Calculators Extraction - Implementation Guide

### **Step 2: Refactor Existing Aggregation Code with SoA Integration**

**Prompt for Claude:**

> Refactor all species aggregation logic in `sarkas/particles.py` to use the new enhanced aggregation utilities with comprehensive SoA integration. Replace manual loops with optimized functions and integrate with the SoA species tensor system:
> 
> **Pattern replacement for basic aggregation:**
> ```python
> # BEFORE: Manual species loop (slow, not cache-friendly)
> def _aggregate_by_species_old(self, per_particle_data):
>     sp_start = 0
>     species_totals = zeros(self.num_species)
>     for sp in range(self.num_species):
>         sp_end = sp_start + self.species_num[sp]
>         species_totals[sp] = per_particle_data[sp_start:sp_end].sum()
>         sp_start = sp_end
>     return species_totals
> 
> # AFTER: Optimized aggregation (fast, vectorized)
> def _aggregate_by_species(self, per_particle_data):
>     from sarkas.physics.aggregation import species_sum
>     return species_sum(per_particle_data, self.id, self.num_species)
> ```
> 
> **Enhanced aggregation for tensors with SoA:**
> ```python
> # BEFORE: Tensor aggregation creating full intermediate tensors
> def _aggregate_pressure_tensor_old(self, per_particle_stress):
>     # Creates large intermediate tensor
>     species_stress_tensor = zeros((self.num_species, 3, 3))
>     for sp in range(self.num_species):
>         mask = self.id == sp
>         species_stress_tensor[sp] = per_particle_stress[mask].sum(axis=0)
>     return species_stress_tensor
> 
> # AFTER: Direct SoA aggregation (memory efficient)
> def _aggregate_pressure_tensor_soa(self, per_particle_stress):
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.aggregation import aggregate_to_soa_pressure_tensor
>         aggregate_to_soa_pressure_tensor(per_particle_stress, self.id, self._species_arrays)
>     else:
>         # Fallback to traditional method for small species count
>         return self._aggregate_pressure_tensor_old(per_particle_stress)
> ```
> 
> **Methods to update with enhanced aggregation:**
> ```python
> def calculate_species_kinetic_energy(self):
>     """Enhanced with optimized aggregation."""
>     if not hasattr(self, 'kinetic_energy'):
>         self.calculate_kinetic_energy()
>     
>     from sarkas.physics.aggregation import species_sum
>     self.species_kinetic_energy = species_sum(self.kinetic_energy, self.id, self.num_species)
> 
> def calculate_species_potential_energy(self):
>     """Enhanced with optimized aggregation."""
>     from sarkas.physics.aggregation import species_sum
>     self.species_potential_energy = species_sum(self.potential_energy, self.id, self.num_species)
> 
> def calculate_species_momentum(self):
>     """Enhanced with vector aggregation."""
>     if not hasattr(self, 'momentum'):
>         from sarkas.physics.thermodynamics import momentum
>         self.momentum = momentum(self.vel, self.masses)
>     
>     from sarkas.physics.aggregation import species_vector_sum
>     self.species_momentum = species_vector_sum(self.momentum, self.id, self.num_species)
> 
> def calculate_species_electric_current(self):
>     """Enhanced with SoA electric current aggregation."""
>     if not hasattr(self, 'electric_current'):
>         from sarkas.physics.transport import electric_current_vector
>         self.electric_current = electric_current_vector(self.vel, self.charges)
>     
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.aggregation import aggregate_to_soa_heat_flux
>         # Store electric current in heat flux tensor structure (both are 3-vectors)
>         aggregate_to_soa_heat_flux(self.electric_current, self.id, self._species_arrays)
>         self.species_electric_current = self._species_arrays.get_electric_current_vector()
>     else:
>         from sarkas.physics.aggregation import species_vector_sum
>         self.species_electric_current = species_vector_sum(self.electric_current, self.id, self.num_species)
> 
> def calculate_species_pressure_tensor(self):
>     """Enhanced with SoA pressure tensor aggregation."""
>     # Calculate per-particle stress tensor (if not already available)
>     if not hasattr(self, 'per_particle_stress'):
>         self._calculate_per_particle_stress()
>     
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.aggregation import aggregate_to_soa_pressure_tensor
>         aggregate_to_soa_pressure_tensor(self.per_particle_stress, self.id, self._species_arrays)
>     else:
>         from sarkas.physics.aggregation import species_tensor_sum
>         traditional_tensor = species_tensor_sum(self.per_particle_stress, self.id, self.num_species)
>         self.species_pressure_tensor = traditional_tensor
> 
> def calculate_species_heat_flux(self):
>     """Enhanced with comprehensive heat flux calculation."""
>     # Calculate per-particle heat flux
>     if not hasattr(self, 'kinetic_energy'):
>         self.calculate_kinetic_energy()
>     
>     from sarkas.physics.transport import heat_flux_vector
>     per_particle_heat_flux = heat_flux_vector(self.vel, self.kinetic_energy, self.potential_energy)
>     
>     if self._use_soa_species and hasattr(self, '_species_arrays'):
>         from sarkas.physics.aggregation import aggregate_to_soa_heat_flux
>         aggregate_to_soa_heat_flux(per_particle_heat_flux, self.id, self._species_arrays)
>     else:
>         from sarkas.physics.aggregation import species_vector_sum
>         self.species_heat_flux = species_vector_sum(per_particle_heat_flux, self.id, self.num_species)
> 
> def calculate_species_diffusion_flux(self):
>     """Enhanced diffusion flux with cross-species interactions."""
>     if self.num_species < 2:
>         self.species_diffusion_flux = zeros((1, ## **Week 1: Thermodynamics Calculator**
