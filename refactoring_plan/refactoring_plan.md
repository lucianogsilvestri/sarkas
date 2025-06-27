# Sarkas Refactoring Plan: Speed & Modularity Focus

## **Objectives**
1. **Execution Speed**: Eliminate Python overhead in hot paths, optimize memory access patterns
2. **Modularity**: Clean separation of concerns, testable components, reusable calculators
3. **Backward Compatibility**: Maintain exact same user interface and YAML format
4. **Risk Management**: Incremental changes with validation at each step

---

## **Phase 1: Physics Calculators Extraction (Weeks 1-3)**

### **Week 1: Thermodynamics Calculator**

**Files to Create:**
```
sarkas/
├── physics/
│   ├── __init__.py
│   ├── thermodynamics.py          # NEW - Extract from particles.py
│   └── tests/
│       └── test_thermodynamics.py # NEW - Pure function tests
```

**Files to Modify:**
- `sarkas/particles.py` - Update methods to delegate to calculator
- `sarkas/core.py` - Update any direct thermodynamics calculations

**Specific Methods to Extract:**
```python
# From particles.py -> physics/thermodynamics.py
- calculate_kinetic_energy()
- calculate_species_kinetic_temperature() 
- calculate_momentum()
- calculate_center_of_mass_velocity()
- remove_center_of_mass_motion()
- calculate_species_momentum()
```

**Success Criteria:**
- [ ] All existing tests pass
- [ ] New physics tests achieve 100% coverage
- [ ] Performance improvement: 5-15% faster thermodynamics calculations
- [ ] Memory usage: Same or better

---

### **Week 2: Observable Calculator**

**Files to Create:**
```
sarkas/physics/observables.py       # NEW - Extract observable calculations
sarkas/physics/tests/test_observables.py
```

**Files to Modify:**
- `sarkas/particles.py` - Update observable methods

**Methods to Extract:**
```python
# From particles.py -> physics/observables.py
- calculate_electric_current()
- calculate_species_electric_current()
- calculate_species_heat_flux()
- calculate_species_pressure_tensor()
- calculate_species_diffusion_flux()
- calculate_species_velocity_moments()
```

**Success Criteria:**
- [ ] Observable calculations 10-20% faster (compiled)
- [ ] Each observable function testable in isolation
- [ ] Zero regression in simulation results

---

### **Week 3: Species Aggregation Utilities**

**Files to Create:**
```
sarkas/physics/aggregation.py       # NEW - Reusable species utilities
sarkas/physics/tests/test_aggregation.py
```

**Methods to Extract:**
```python
# Reusable utilities extracted from various places
- species_sum(per_particle_array, species_id, num_species)
- species_mean(per_particle_array, species_id, num_species) 
- species_vector_sum(per_particle_vectors, species_id, num_species)
- species_tensor_sum(per_particle_tensors, species_id, num_species)
```

**Success Criteria:**
- [ ] All species calculations use same optimized utilities
- [ ] 20-30% reduction in species aggregation time
- [ ] Code duplication eliminated across modules

---

## **Phase 2: Algorithm Extraction (Weeks 4-7)**

### **Week 4: Force Calculator Interface**

**Files to Create:**
```
sarkas/algorithms/
├── __init__.py
├── base.py                         # NEW - InteractionSolverBase (already designed)
└── tests/
    └── test_base.py
```

**Files to Modify:**
- `sarkas/potentials/core.py` - Implement new interface
- `sarkas/integrators/base.py` - Update to use algorithm interface

**Success Criteria:**
- [ ] Clean algorithm interface defined
- [ ] All existing algorithms implement interface
- [ ] No performance regression

---

### **Week 5: PPPM Algorithm Cleanup**

**Files to Create:**
```
sarkas/algorithms/pppm.py           # NEW - Clean PPPM implementation
sarkas/algorithms/tests/test_pppm.py
```

**Files to Modify:**
- `sarkas/potentials/coulomb.py` - Use new PPPM algorithm
- Remove PPPM code scattered across multiple files

**Methods to Consolidate:**
```python
# Scattered PPPM code -> algorithms/pppm.py
- All mesh-related calculations
- Green's function computation
- Force decomposition logic
- Error estimation
```

**Success Criteria:**
- [ ] PPPM code consolidated in single module
- [ ] 10-15% PPPM performance improvement
- [ ] Easier to optimize and maintain

---

### **Week 6: Cell List Algorithm**

**Files to Create:**
```
sarkas/algorithms/cell_list.py      # NEW - Clean implementation
sarkas/algorithms/tests/test_cell_list.py
```

**Files to Modify:**
- `sarkas/potentials/` - Update to use new cell list

**Success Criteria:**
- [ ] Cell list algorithm isolated and optimized
- [ ] Memory access patterns improved
- [ ] 5-10% performance gain in short-range calculations

---

### **Week 7: Algorithm Factory System**

**Files to Create:**
```
sarkas/algorithms/factory.py        # NEW - Algorithm selection
sarkas/algorithms/registry.py       # NEW - Auto-discovery
```

**Success Criteria:**
- [ ] Easy algorithm selection and swapping
- [ ] Plugin system for new algorithms
- [ ] Runtime algorithm optimization

---

## **Phase 3: Performance Optimization (Weeks 8-10)**

### **Week 8: Vectorization & Compilation Optimization**

**Files to Modify:**
- `sarkas/physics/thermodynamics.py` - Replace loops with NumPy vectorization
- `sarkas/physics/observables.py` - Optimize with pure NumPy operations
- `sarkas/algorithms/pppm.py` - Vectorize mesh operations
- `sarkas/algorithms/cell_list.py` - Optimize neighbor searching

**Specific Optimizations:**
```python
# BEFORE: Loop-based calculation
for i in range(N):
    ke[i] = 0.5 * masses[i] * (vel[i]**2).sum()

# AFTER: Pure NumPy vectorization  
ke = 0.5 * masses * (vel**2).sum(axis=1)
```

**Success Criteria:**
- [ ] 15-30% speedup in physics calculations
- [ ] All hot path loops replaced with NumPy operations
- [ ] Maximum SIMD utilization achieved

---

### **Week 9: Memory Pre-allocation & Pool Management**

**Files to Create:**
```
sarkas/optimization/
├── __init__.py
├── memory_pools.py                 # NEW - Pre-allocated work arrays
└── array_cache.py                  # NEW - Reusable array management
```

**Files to Modify:**
- `sarkas/particles.py` - Use pre-allocated arrays for calculations
- `sarkas/algorithms/` - Eliminate allocations in hot paths
- `sarkas/physics/` - Reuse work arrays

**Success Criteria:**
- [ ] Zero allocations in simulation main loop
- [ ] 10-15% performance improvement
- [ ] 50% reduction in garbage collection overhead

---

### **Week 10: Algorithm-Specific Performance Tuning**

**Files to Modify:**
- `sarkas/algorithms/pppm.py` - Optimize Green's function calculations
- `sarkas/algorithms/cell_list.py` - Improve neighbor list efficiency
- `sarkas/potentials/coulomb.py` - Optimize force kernel

**Specific Optimizations:**
- PPPM mesh operations batching
- Cell list memory access pattern optimization
- Force calculation kernel optimization
- Species loop unrolling where beneficial

**Success Criteria:**
- [ ] 20-40% improvement in force calculations
- [ ] Optimal memory access patterns
- [ ] Reduced function call overhead

---

## **Phase 4: Configuration System + Hybrid Unit System (Weeks 11-13)**

### **Week 11: Configuration Parsing + Unit System Foundation**

**Files to Create:**
```
sarkas/config/
├── __init__.py
├── loader.py                       # NEW - YAML parsing
├── validator.py                    # NEW - Input validation
├── schema.py                       # NEW - Configuration schema
├── units/                          # NEW - Hybrid unit system
│   ├── __init__.py
│   ├── normalizer.py              # NEW - Dimensional → Dimensionless conversion
│   ├── converter.py               # NEW - Dimensionless → Output units conversion
│   ├── reference_scales.py        # NEW - Auto-determine reference scales
│   └── constants.py               # NEW - Physical constants (for I/O only)
└── tests/
    ├── test_loader.py
    ├── test_validator.py
    └── test_units.py              # NEW - Unit system tests
```

**Files to Modify:**
- `sarkas/processes.py` - Use new config system with unit normalization

**Key Implementation:**
```python
# sarkas/config/units/normalizer.py
class DimensionalNormalizer:
    """Converts dimensional input to internal dimensionless units"""
    
    def __init__(self, species_list, reference_choice="automatic"):
        self.species = species_list
        self.reference_scales = self._determine_reference_scales(reference_choice)
        
    def _determine_reference_scales(self, choice):
        """Automatically choose reference scales from the system"""
        if choice == "automatic":
            # Use the most abundant species as reference
            dominant_species = max(self.species, key=lambda s: s.num * s.number_density)
            return {
                'length': self._calculate_wigner_seitz_radius(dominant_species),
                'energy': dominant_species.temperature * kB,
                'mass': dominant_species.mass,
                'charge': abs(dominant_species.charge),
                'time': 1.0 / self._calculate_plasma_frequency(dominant_species)
            }
    
    def normalize_all_quantities(self):
        """Convert all dimensional quantities to dimensionless"""
        for species in self.species:
            species._internal_temperature = species.temperature * kB / self.reference_scales['energy']
            species._internal_density = species.number_density * self.reference_scales['length']**3
            species._internal_mass = species.mass / self.reference_scales['mass']
            species._internal_charge = species.charge / self.reference_scales['charge']
```

**Success Criteria:**
- [ ] Clean separation of config parsing from simulation logic
- [ ] Unit normalization integrated into config pipeline
- [ ] Better error messages for invalid inputs
- [ ] Faster simulation startup (no unit conversions in main loop)
- [ ] All dimensional input automatically converted to dimensionless internally

---

### **Week 12: Parameter System + Internal Normalization**

**Files to Create:**
```
sarkas/config/parameters/
├── __init__.py
├── physical_constants.py          # NEW - Physical constants (for I/O only)
├── simulation_config.py           # NEW - Simulation settings
├── system_properties.py           # NEW - Derived properties
├── internal_parameters.py         # NEW - Normalized internal parameters
└── tests/
```

**Files to Modify:**
- `sarkas/core.py` - Replace Parameters class with normalized version

**Key Implementation:**
```python
# sarkas/config/parameters/internal_parameters.py
class InternalParameters:
    """All internal parameters are dimensionless - no physical constants stored!"""
    
    def __init__(self, dimensional_params, normalizer):
        # No tiny physical constants!
        self.kB = 1.0  # Normalized away
        self.eps0 = 1.0  # Normalized away  
        self.fourpie0 = 1.0  # In normalized units
        self.hbar = 1.0  # Normalized away
        
        # All quantities are O(1)
        self.dt_normalized = dimensional_params.dt / normalizer.reference_scales['time']
        self.box_lengths_normalized = dimensional_params.box_lengths / normalizer.reference_scales['length']
        
        # Species properties (all normalized)
        self.species_temperatures = [s._internal_temperature for s in normalizer.species]
        self.species_densities = [s._internal_density for s in normalizer.species]
        self.species_masses = [s._internal_mass for s in normalizer.species]
        self.species_charges = [s._internal_charge for s in normalizer.species]
        
        # Store reference scales for output conversion
        self.reference_scales = normalizer.reference_scales
```

**Success Criteria:**
- [ ] All internal calculations use O(1) dimensionless numbers
- [ ] No physical constants stored in internal parameters
- [ ] 10-20% performance improvement from better numerics
- [ ] Clean separation of dimensional I/O vs dimensionless calculations
- [ ] Immutable objects where appropriate
- [ ] Easier testing and validation

---

### **Week 13: Output Conversion + Component Factory**

**Files to Create:**
```
sarkas/config/
├── factory.py                      # NEW - Component creation
├── registry.py                     # NEW - Component registry
└── output/                         # NEW - Output conversion system
    ├── __init__.py
    ├── dimensional_output.py       # NEW - Convert results to dimensional
    ├── unit_formatting.py          # NEW - Pretty printing with units
    └── tests/
```

**Key Implementation:**
```python
# sarkas/config/output/dimensional_output.py
class OutputConverter:
    """Converts internal dimensionless results back to user's requested units"""
    
    def __init__(self, normalizer, output_units="same_as_input"):
        self.normalizer = normalizer
        self.output_units = output_units
        
    def convert_positions(self, pos_normalized):
        """Convert normalized positions back to dimensional"""
        return pos_normalized * self.normalizer.reference_scales['length']
        
    def convert_energies(self, energy_normalized):
        """Convert normalized energies back to dimensional"""
        energy_dimensional = energy_normalized * self.normalizer.reference_scales['energy']
        
        if self.output_units == "eV":
            return energy_dimensional / (kB * self.normalizer.reference_scales['temperature'])
        elif self.output_units == "SI":
            return energy_dimensional
            
    def convert_all_results(self, results_normalized):
        """Convert all simulation results back to dimensional form"""
        # Seamlessly convert positions, velocities, energies, forces, etc.
        # User never sees the internal normalization complexity
```

**Success Criteria:**
- [ ] 100% backward compatibility maintained
- [ ] Results identical to dimensional calculations (within numerical precision)
- [ ] Clean component creation with normalized parameters
- [ ] User interface completely unchanged
- [ ] Easy to swap implementations
- [ ] Plugin system foundation

---

**Phase 4 Overall Benefits:**
- **Numerical Precision**: All internal calculations use well-conditioned O(1) numbers
- **Performance**: 5-15% immediate speedup from eliminating unit conversions
- **Stability**: No numerical precision issues from tiny physical constants
- **Amplification**: Enhances benefits from Phases 1-3 vectorization and optimization
- **User Experience**: Completely transparent - users see no changes to input/output format
- **Modularity**: Clean separation of dimensional I/O from dimensionless calculations

---

## **Phase 5: Process Class Decomposition (Weeks 14-16)**

### **Week 14: Simulation Engine Extraction**

**Files to Create:**
```
sarkas/engine/
├── __init__.py
├── simulation_engine.py           # NEW - Pure simulation logic
├── orchestrator.py                # NEW - High-level coordination
└── tests/
```

**Files to Modify:**
- `sarkas/processes.py` - Extract core simulation logic

**Success Criteria:**
- [ ] Core simulation loop isolated and optimized
- [ ] Testable simulation logic
- [ ] Reduced Python overhead in main loop

---

### **Week 15: I/O System Extraction**

**Files to Create:**
```
sarkas/io/
├── __init__.py
├── checkpoint_manager.py          # NEW - Save/load logic
├── output_manager.py              # NEW - Data output
├── hdf5_handler.py                # NEW - HDF5 operations
└── tests/
```

**Files to Modify:**
- `sarkas/utilities/io.py` - Refactor into focused modules

**Success Criteria:**
- [ ] Clean I/O responsibilities
- [ ] Faster checkpoint operations
- [ ] Better error handling

---

### **Week 16: Legacy Interface Maintenance**

**Files to Create:**
```
sarkas/legacy/
├── __init__.py
├── simulation_facade.py           # NEW - Backward compatible interface
├── parameters_facade.py           # NEW - Legacy Parameters interface
└── particles_facade.py            # NEW - Legacy Particles interface
```

**Files to Modify:**
- `sarkas/processes.py` - Implement as facade over new system

**Success Criteria:**
- [ ] 100% backward compatibility maintained
- [ ] All existing user code works unchanged
- [ ] Performance improvements delivered transparently

---

## **Phase 6: Performance Validation & Optimization (Weeks 17-18)**

### **Week 17: Comprehensive Benchmarking**

**Files to Create:**
```
benchmarks/
├── performance_suite.py           # NEW - Comprehensive benchmarks
├── memory_profiling.py            # NEW - Memory usage analysis
├── regression_tests.py            # NEW - Performance regression detection
└── reports/                       # Benchmark results
```

**Success Criteria:**
- [ ] Comprehensive performance baseline established
- [ ] All optimizations validated
- [ ] Performance regression detection system

---

### **Week 18: Final Optimization & Documentation**

**Activities:**
- Profile and optimize remaining bottlenecks
- Update all documentation
- Create migration guide
- Performance tuning based on benchmark results

**Success Criteria:**
- [ ] Overall performance improvement: 30-50%
- [ ] Memory usage reduction: 15-25%
- [ ] Complete documentation update
- [ ] Migration guide for advanced users

---

## **Success Metrics**

### **Performance Targets (Revised):**
- **Overall simulation speed**: 30-50% improvement (achievable without SoA)
- **Memory usage**: 10-15% reduction (through pre-allocation, not layout changes)
- **Startup time**: 20-30% faster
- **Force calculations**: 30-40% faster (through better algorithms, not memory layout)
- **Physics calculations**: 40-60% faster (through vectorization and compilation)

### **Code Quality Targets:**
- **Test coverage**: >90% for all new modules
- **Cyclomatic complexity**: <10 for all new functions
- **Module coupling**: Minimized dependencies between components
- **Documentation**: 100% API documentation coverage

### **Compatibility Targets:**
- **User interface**: 100% backward compatible
- **YAML format**: No breaking changes
- **Python API**: All existing code works unchanged
- **Output format**: Identical results (within numerical precision)

---

## **Risk Mitigation**

### **Technical Risks:**
1. **Performance Regression**: Continuous benchmarking at each step
2. **Breaking Changes**: Comprehensive test suite + facade pattern
3. **Memory Issues**: Memory profiling + gradual rollout
4. **Numba Compatibility**: Test compilation at each step

### **Project Risks:**
1. **Scope Creep**: Strict phase boundaries + success criteria
2. **Timeline Delays**: Buffer time built into each phase
3. **Resource Conflicts**: Modular approach allows parallel work
4. **Quality Issues**: Test-driven development + code reviews

---

## **Validation Strategy**

### **Each Week:**
- [ ] All existing tests pass
- [ ] New functionality tests achieve >90% coverage
- [ ] Performance benchmarks show improvement or no regression
- [ ] Memory profiling shows no leaks or excessive usage

### **Each Phase:**
- [ ] Integration testing with full simulation runs
- [ ] Backward compatibility validation
- [ ] Performance targets met
- [ ] Documentation updated

### **Final Validation:**
- [ ] Full simulation suite comparison (old vs new)
- [ ] Performance improvement targets achieved
- [ ] User acceptance testing with real workflows
- [ ] Production deployment readiness

This plan delivers concrete performance and modularity improvements while maintaining the user-friendly interface that makes Sarkas valuable to researchers.