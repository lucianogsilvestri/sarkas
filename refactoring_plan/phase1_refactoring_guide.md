# Phase 1: Physics Calculators Extraction - Implementation Guide

## **Overview**

This phase extracts physics calculations from the monolithic `Particles` class into focused, reusable, and optimized calculator modules. The goal is to improve performance through Numba compilation while maintaining 100% backward compatibility.

**Duration**: 3 weeks  
**Risk Level**: Low (pure function extraction)  
**Expected Performance Gain**: 15-30% in physics calculations

---

## **Pre-Refactoring: Performance Baseline**

### **Step 0: Profile Current Performance**

Before making any changes, establish a comprehensive performance baseline to validate improvements.

#### **0.1 Create Profiling Script**

Create `benchmarks/baseline_profiler.py`:


#### **0.2 Run Baseline Profiling**

```bash
# Install profiling dependencies
pip install memory-profiler matplotlib

# Run baseline profiling (save output)
cd sarkas
python benchmarks/baseline_profiler.py > baseline_profile_output.txt 2>&1

# This will create:
# - baseline_performance.json (timing data)
# - baseline_performance_scaling.png (scaling plots)
# - baseline_profile_*.prof (detailed profiles)
```

#### **0.3 Analyze Current Bottlenecks**

```bash
# Visualize detailed profiles
python -c "
import pstats
stats = pstats.Stats('baseline_profile_calculate_species_kinetic_temperature.prof')
stats.sort_stats('cumulative')
stats.print_stats(10)
"

# Look for:
# 1. Python function call overhead
# 2. NumPy array creation
# 3. Loop inefficiencies
# 4. Memory allocation patterns
```

---

## **Week 1: Thermodynamics Calculator**

### **Step 1: Create Physics Module Structure**

```bash
# Create new directory structure
mkdir -p sarkas/physics/tests

# Create module files
touch sarkas/physics/__init__.py
touch sarkas/physics/thermodynamics.py
touch sarkas/physics/tests/__init__.py
touch sarkas/physics/tests/test_thermodynamics.py
```

### **Step 2: Extract Thermodynamics Functions**

#### **2.1 Implement Pure Calculator (sarkas/physics/thermodynamics.py)**

**Prompt for Claude:**

> I need you to create a new file `sarkas/physics/thermodynamics.py` that extracts thermodynamics calculations from the Sarkas `Particles` class. Follow these requirements:
> 
> **Requirements:**
> 1. **Pure functions only** - No class state, only input/output
> 2. **Numba compilation** - Use `@jit(nopython=True)` for all calculation kernels
> 3. **NumPy optimization** - Prefer vectorized operations over loops where possible
> 4. **Comprehensive docstrings** - Include physics equations and units
> 5. **Backward compatibility** - Provide wrapper functions matching current interface
> 
> **Functions to implement:**
> ```python
> # Core calculations
> kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray
> temperature_from_kinetic_energy(ke: ndarray, masses: ndarray, dimensions: int, kB: float) -> ndarray
> temperature_from_velocities(velocities: ndarray, masses: ndarray, dimensions: int, kB: float) -> ndarray
> momentum(velocities: ndarray, masses: ndarray) -> ndarray
> center_of_mass_velocity(velocities: ndarray, masses: ndarray) -> ndarray
> remove_center_of_mass_motion(velocities: ndarray, masses: ndarray) -> ndarray
> 
> # Backward compatibility wrappers
> calculate_kinetic_energy(velocities: ndarray, masses: ndarray) -> ndarray
> calculate_temperature(velocities: ndarray, masses: ndarray, **kwargs) -> ndarray
> ```
> 
> **Optimization guidelines:**
> - Use `@jit(nopython=True)` for inner calculation loops
> - Prefer `ndarray.sum(axis=1)` over manual loops where possible
> - Pre-allocate output arrays with `np.zeros()`
> - Include both loop-based (for Numba) and vectorized versions where beneficial
> 
> **Testing requirements:**
> - Each function must be testable in isolation
> - Include examples in docstrings that can be used as tests
> - Handle edge cases (empty arrays, single particles, zero velocities)
> 
> **Performance targets:**
> - 20-40% faster than current implementation through Numba compilation
> - Zero memory allocations in compiled kernels (except for output arrays)
> - Consistent performance across different system sizes

#### **2.2 Create Comprehensive Tests**

**Prompt for Claude:**

> Create comprehensive tests in `sarkas/physics/tests/test_thermodynamics.py` for the thermodynamics calculator. Requirements:
> 
> **Test Coverage:**
> 1. **Physics validation** - Verify correct physics (energy conservation, equipartition theorem)
> 2. **Edge cases** - Empty arrays, single particles, zero velocities, NaN handling
> 3. **Performance tests** - Compare speed with current implementation
> 4. **Numerical precision** - Verify results match analytical solutions
> 5. **Type safety** - Test with different NumPy dtypes
> 
> **Specific test cases:**
> ```python
> def test_kinetic_energy_simple_cases()         # Known analytical results
> def test_kinetic_energy_conservation()         # Physics validation
> def test_temperature_equipartition_theorem()   # Verify kT relationship
> def test_center_of_mass_conservation()         # Momentum conservation
> def test_edge_cases()                          # Empty arrays, single particles
> def test_performance_vs_baseline()             # Speed comparison
> def test_backward_compatibility()              # Wrapper functions work
> ```
> 
> **Performance benchmarking:**
> - Include benchmarks comparing new functions vs. current Particles methods
> - Test with system sizes: 1K, 10K, 100K particles
> - Verify compilation time vs. runtime trade-offs
> - Memory usage validation

#### **2.3 Update Particles Class**

**Prompt for Claude:**

> Update the `sarkas/particles.py` file to use the new thermodynamics calculator while maintaining 100% backward compatibility. Requirements:
> 
> **Migration strategy:**
> 1. **Import new calculator** at the top of the file
> 2. **Keep all existing method signatures** exactly the same
> 3. **Delegate calculations** to new pure functions
> 4. **Maintain all attributes** that user code expects (e.g., `self.kinetic_energy`)
> 5. **Add performance toggle** for gradual migration
> 
> **Methods to update:**
> ```python
> def calculate_kinetic_energy(self):
>     """Existing interface - now delegates to pure calculator"""
>     
> def calculate_species_kinetic_temperature(self):
>     """Existing interface - now uses optimized calculations"""
>     
> def calculate_total_kinetic_energy(self):
>     """Existing interface - now more efficient"""
> ```
> 
> **Implementation pattern:**
> ```python
> # BEFORE (example)
> def calculate_kinetic_energy(self):
>     self.kinetic_energy = 0.5 * self.masses * (self.vel * self.vel).sum(axis=-1)
> 
> # AFTER (example)
> def calculate_kinetic_energy(self):
>     from sarkas.physics.thermodynamics import kinetic_energy
>     self.kinetic_energy = kinetic_energy(self.vel, self.masses)
> ```
> 
> **Validation requirements:**
> - All existing tests must still pass
> - Results must be numerically identical (within machine precision)
> - No changes to public interface
> - Optional performance flag: `calculate_kinetic_energy(use_fast=True)`

### **Step 3: Integration and Validation**

#### **3.1 Run Updated Tests**

```bash
# Run all existing tests to ensure no regressions
python -m pytest sarkas/tests/ -v

# Run new physics tests
python -m pytest sarkas/physics/tests/ -v

# Run performance comparison
python benchmarks/compare_thermodynamics_performance.py
```

#### **3.2 Performance Validation**

Create `benchmarks/compare_thermodynamics_performance.py`:

```python
"""
Compare performance of old vs new thermodynamics calculations.
Validates that refactoring delivers expected performance improvements.
"""

import time
import numpy as np
import json
from sarkas.particles import Particles
from sarkas.physics.thermodynamics import kinetic_energy, temperature_from_velocities

def compare_performance():
    """Compare old vs new implementations."""
    
    # Load baseline results
    with open('baseline_performance.json', 'r') as f:
        baseline = json.load(f)
    
    particle_counts = [1000, 5000, 10000, 50000]
    
    print("=== PERFORMANCE COMPARISON ===\n")
    
    for N in particle_counts:
        print(f"Testing {N} particles:")
        
        # Setup test system
        particles = Particles()
        particles.setup_test_system(N)  # You'll need to implement this helper
        
        # Test kinetic energy calculation
        # Old method (delegated to new, but with Particles overhead)
        times_old = []
        for _ in range(10):
            start = time.perf_counter()
            particles.calculate_kinetic_energy()
            times_old.append(time.perf_counter() - start)
        
        # New method (direct pure function)
        times_new = []
        for _ in range(10):
            start = time.perf_counter()
            ke = kinetic_energy(particles.vel, particles.masses)
            times_new.append(time.perf_counter() - start)
        
        old_time = np.mean(times_old)
        new_time = np.mean(times_new)
        speedup = old_time / new_time
        
        # Compare with baseline
        baseline_time = baseline[str(N)]['calculate_kinetic_energy']['mean_time']
        baseline_speedup = baseline_time / new_time
        
        print(f"  Kinetic Energy:")
        print(f"    Baseline: {baseline_time*1000:.3f} ms")
        print(f"    Particles method: {old_time*1000:.3f} ms")  
        print(f"    Pure function: {new_time*1000:.3f} ms")
        print(f"    Speedup vs Particles: {speedup:.2f}x")
        print(f"    Speedup vs baseline: {baseline_speedup:.2f}x")
        print()

if __name__ == "__main__":
    compare_performance()
```

### **Step 4: Success Criteria Validation**

- [ ] **All existing tests pass** - No regressions introduced
- [ ] **New physics tests achieve >95% coverage** - Comprehensive validation
- [ ] **Performance improvement: 15-30%** - Measured speedup in calculations
- [ ] **Memory usage: Same or better** - No memory leaks or excessive allocation
- [ ] **Backward compatibility: 100%** - All user code works unchanged

---

## **Week 2: Transport Properties Calculator**

### **Step 1: Create Transport Calculator**

**Prompt for Claude:**

> Create `sarkas/physics/transport.py` that extracts transport property calculations from the Particles class. This module should handle electric current, heat flux, and diffusion calculations.
> 
> **Functions to implement:**
> ```python
> # Electric current calculations
> electric_current_density(velocities: ndarray, charges: ndarray) -> ndarray
> electric_current_vector(velocities: ndarray, charges: ndarray) -> ndarray
> 
> # Heat flux calculations  
> heat_flux_vector(velocities: ndarray, kinetic_energy: ndarray, positions: ndarray) -> ndarray
> heat_flux_tensor(velocities: ndarray, kinetic_energy: ndarray, stress_tensor: ndarray) -> ndarray
> 
> # Diffusion calculations
> diffusion_flux(velocities: ndarray, concentrations: ndarray, gradients: ndarray) -> ndarray
> 
> # Backward compatibility
> calculate_electric_current(velocities: ndarray, charges: ndarray) -> ndarray
> calculate_species_electric_current(velocities: ndarray, charges: ndarray, species_id: ndarray) -> ndarray
> ```
> 
> **Requirements:**
> - All functions use `@jit(nopython=True)` for performance
> - Include comprehensive physics documentation
> - Handle vector and tensor quantities properly
> - Optimize for typical plasma simulation patterns

### **Step 2: Create Mechanical Properties Calculator**

**Prompt for Claude:**

> Create `sarkas/physics/mechanical.py` for mechanical property calculations (pressure, stress, momentum).
> 
> **Functions to implement:**
> ```python
> # Momentum calculations
> momentum_vector(velocities: ndarray, masses: ndarray) -> ndarray
> angular_momentum(positions: ndarray, velocities: ndarray, masses: ndarray) -> ndarray
> 
> # Pressure calculations
> pressure_tensor_kinetic(velocities: ndarray, masses: ndarray, volume: float) -> ndarray
> pressure_tensor_virial(virial_tensor: ndarray, volume: float) -> ndarray
> pressure_scalar(pressure_tensor: ndarray, dimensions: int) -> float
> 
> # Stress calculations
> stress_tensor(forces: ndarray, positions: ndarray, volume: float) -> ndarray
> 
> # Backward compatibility
> calculate_species_pressure_tensor(velocities: ndarray, virial_tensor: ndarray, 
>                                  species_id: ndarray, masses: ndarray, volume: float) -> ndarray
> ```

### **Step 3: Update Particles Class Methods**

**Prompt for Claude:**

> Update the remaining physics calculation methods in `sarkas/particles.py` to use the new transport and mechanical calculators:
> 
> **Methods to update:**
> ```python
> def calculate_species_electric_current(self)
> def calculate_species_heat_flux(self)  
> def calculate_species_pressure_tensor(self)
> def calculate_species_momentum(self)
> def calculate_species_diffusion_flux(self)
> ```
> 
> **Pattern to follow:**
> 1. Import appropriate calculator function
> 2. Call pure function with particle data
> 3. Store result in expected attribute
> 4. Maintain exact same interface and behavior

---

## **Week 3: Species Aggregation Utilities**

### **Step 1: Create Aggregation Module**

**Prompt for Claude:**

> Create `sarkas/physics/aggregation.py` that provides reusable utilities for aggregating per-particle quantities by species.
> 
> **Functions to implement:**
> ```python
> @jit(nopython=True)
> def species_sum(per_particle_quantity: ndarray, species_id: ndarray, num_species: int) -> ndarray
> 
> @jit(nopython=True)  
> def species_mean(per_particle_quantity: ndarray, species_id: ndarray, num_species: int) -> ndarray
> 
> @jit(nopython=True)
> def species_vector_sum(per_particle_vectors: ndarray, species_id: ndarray, num_species: int) -> ndarray
> 
> @jit(nopython=True)
> def species_tensor_sum(per_particle_tensors: ndarray, species_id: ndarray, num_species: int) -> ndarray
> 
> @jit(nopython=True)
> def species_histogram(values: ndarray, species_id: ndarray, bins: ndarray) -> ndarray
> ```
> 
> **Optimization requirements:**
> - Use NumPy's `bincount` where possible for maximum speed
> - Handle different array shapes (scalars, vectors, tensors)
> - Optimize for common species patterns (2-3 species typical)
> - Include bounds checking and error handling

### **Step 2: Refactor Existing Aggregation Code**

**Prompt for Claude:**

> Refactor all species aggregation logic in `sarkas/particles.py` to use the new aggregation utilities. Replace manual loops with optimized functions:
> 
> **Pattern replacement:**
> ```python
> # BEFORE: Manual species loop (slow)
> sp_start = 0
> species_totals = zeros(self.num_species)
> for sp in range(self.num_species):
>     sp_end = sp_start + self.species_num[sp]
>     species_totals[sp] = per_particle_data[sp_start:sp_end].sum()
>     sp_start = sp_end
> 
> # AFTER: Optimized aggregation (fast)
> from sarkas.physics.aggregation import species_sum
> species_totals = species_sum(per_particle_data, self.species_id, self.num_species)
> ```
> 
> **Methods to update:**
> - All `calculate_species_*` methods
> - Any manual species loops
> - RDF and observable calculations that use species

### **Step 3: Final Integration and Performance Validation**

#### **3.1 Run Complete Test Suite**

```bash
# Run all tests
python -m pytest sarkas/ -v --tb=short

# Run performance regression tests
python benchmarks/phase1_final_validation.py

# Run memory leak tests
python benchmarks/memory_validation.py
```

#### **3.2 Compare Against Baseline**

Create `benchmarks/phase1_final_validation.py`:

```python
"""
Final validation that Phase 1 refactoring delivers expected benefits.
"""

def validate_phase1_success():
    """Comprehensive validation of Phase 1 results."""
    
    print("=== PHASE 1 FINAL VALIDATION ===\n")
    
    # 1. Performance comparison
    run_performance_comparison()
    
    # 2. Memory usage comparison  
    run_memory_comparison()
    
    # 3. Accuracy validation
    run_accuracy_validation()
    
    # 4. Backward compatibility validation
    run_compatibility_validation()
    
    print("=== PHASE 1 COMPLETE ===")

def run_performance_comparison():
    """Compare performance against baseline."""
    # Load baseline and current results
    # Report speedups for each calculation type
    # Verify we hit 15-30% improvement target
    
def run_memory_comparison():
    """Validate memory usage improvements."""
    # Profile memory usage
    # Ensure no memory leaks
    # Validate garbage collection improvements
    
def run_accuracy_validation():
    """Ensure numerical accuracy maintained."""
    # Run identical calculations with old and new code
    # Verify results match within machine precision
    # Test edge cases and boundary conditions
    
def run_compatibility_validation():
    """Test backward compatibility."""
    # Run existing user scripts
    # Verify all public interfaces unchanged
    # Test with real YAML input files
```

---

## **Success Criteria for Phase 1**

### **Performance Targets:**
- [ ] **15-30% speedup** in thermodynamics calculations (kinetic energy, temperature)
- [ ] **20-40% speedup** in transport calculations (electric current, heat flux)  
- [ ] **25-50% speedup** in species aggregation operations
- [ ] **Overall simulation speedup: 5-15%** (limited by non-physics operations)

### **Code Quality Targets:**
- [ ] **>95% test coverage** for all new physics modules
- [ ] **100% backward compatibility** - all existing code works unchanged
- [ ] **Zero memory leaks** or increased memory usage
- [ ] **Clean module boundaries** - physics calculations isolated from data management

### **Validation Targets:**
- [ ] **Numerical accuracy maintained** - results identical within machine precision
- [ ] **All existing tests pass** - no regressions introduced
- [ ] **Performance benchmarks met** - measured improvements documented
- [ ] **Code maintainability improved** - physics functions testable in isolation

---

## **Troubleshooting Common Issues**

### **Issue 1: Numba Compilation Errors**

```python
# Problem: Complex NumPy operations not supported in nopython mode
# Solution: Use explicit loops or supported NumPy subset

# Bad:
result = np.linalg.norm(vectors, axis=1)  # Not supported in nopython

# Good:
@jit(nopython=True)
def vector_norms(vectors):
    norms = np.zeros(vectors.shape[0])
    for i in range(vectors.shape[0]):
        norm_sq = 0.0
        for j in range(vectors.shape[1]):
            norm_sq += vectors[i, j]**2
        norms[i] = np.sqrt(norm_sq)
    return norms
```

### **Issue 2: Performance Regression**

```python
# Problem: New implementation slower than expected
# Diagnosis: Profile compilation time vs runtime

import time
from numba import jit

@jit(nopython=True)
def slow_function(data):
    # First call includes compilation time
    pass

# Warm up compilation
slow_function(small_test_data)

# Then benchmark runtime only
start = time.perf_counter()
result = slow_function(real_data)
runtime = time.perf_counter() - start
```

### **Issue 3: Numerical Precision Issues**

```python
# Problem: Results don't exactly match original
# Solution: Understand floating point precision limits

# Use appropriate tolerances in tests
np.testing.assert_allclose(old_result, new_result, rtol=1e-14, atol=1e-16)

# Check for order-of-operations differences
# Ensure same mathematical operations in same order
```

---

---

## **Phase 1 Completion Checklist**

### **Week 1 Completion:**
- [ ] `sarkas/physics/thermodynamics.py` created with all required functions
- [ ] `sarkas/physics/tests/test_thermodynamics.py` with >95% coverage
- [ ] `sarkas/particles.py` updated to use thermodynamics calculator
- [ ] All existing tests pass
- [ ] Performance improvement: 15-30% in kinetic energy calculations
- [ ] Baseline vs new performance documented

### **Week 2 Completion:**
- [ ] `sarkas/physics/transport.py` created and tested
- [ ] `sarkas/physics/mechanical.py` created and tested
- [ ] All transport and mechanical methods in Particles updated
- [ ] Performance improvement: 20-40% in transport calculations
- [ ] Memory usage stable or improved

### **Week 3 Completion:**
- [ ] `sarkas/physics/aggregation.py` created with optimized species utilities
- [ ] All species loops replaced with optimized aggregation functions
- [ ] Performance improvement: 25-50% in species calculations
- [ ] Code duplication eliminated across modules
- [ ] Full backward compatibility maintained

### **Final Phase 1 Validation:**
- [ ] Overall simulation speedup: 5-15% measured
- [ ] All baseline performance targets met or exceeded
- [ ] Zero regressions in existing functionality
- [ ] Complete test coverage for new modules
- [ ] Documentation updated for new physics modules

---

## **Post-Phase 1: Preparing for Phase 2**

### **Create Phase 2 Planning Document**

After Phase 1 completion, create `docs/phase2_preparation.md`:

```markdown
# Phase 2 Preparation: Algorithm Extraction

## Lessons Learned from Phase 1
- Performance gains achieved: [Document actual results]
- Challenges encountered: [Document issues and solutions]
- Best practices identified: [What worked well]
- Areas for improvement: [What could be done better]

## Phase 2 Targets Based on Phase 1 Results
- Expected algorithm performance gains: [Based on Phase 1 experience]
- Risk assessment: [Updated based on Phase 1 learnings]
- Timeline adjustments: [If needed based on Phase 1 schedule]
```

### **Update Performance Baseline**

```python
# benchmarks/phase1_to_phase2_baseline.py
"""
Create new baseline after Phase 1 for Phase 2 comparison.
This becomes the new starting point for algorithm optimization.
"""

def create_phase2_baseline():
    """Create baseline for Phase 2 using Phase 1 optimized code."""
    
    # Profile force calculations (target for Phase 2)
    profile_pppm_performance()
    profile_cell_list_performance()
    profile_integration_performance()
    
    # Save results for Phase 2 comparison
    save_phase2_baseline()

def profile_pppm_performance():
    """Profile PPPM algorithm performance with Phase 1 optimizations."""
    # Test PPPM with new physics calculators
    # Measure charge assignment, FFT operations, force interpolation
    # Document bottlenecks for Phase 2 targeting

def profile_cell_list_performance():
    """Profile cell list algorithm performance."""
    # Test cell list construction and neighbor finding
    # Measure with new aggregation utilities
    # Identify optimization opportunities

def profile_integration_performance():
    """Profile time integration performance."""
    # Test Verlet integration with new physics calculators
    # Measure position/velocity updates
    # Look for vectorization opportunities
```

---

## **Communication and Documentation**

### **Create Progress Report Template**

```markdown
# Weekly Progress Report Template

## Week [X] Summary
**Objective:** [What was planned for this week]
**Status:** [Completed/In Progress/Delayed]

## Achievements
- [ ] [Specific deliverable 1]
- [ ] [Specific deliverable 2]
- [ ] [Performance target met/not met]

## Performance Results
| Metric | Baseline | Current | Improvement |
|--------|----------|---------|-------------|
| Kinetic Energy Calc | X.XX ms | X.XX ms | X.X% |
| Species Aggregation | X.XX ms | X.XX ms | X.X% |
| Memory Usage | X.XX MB | X.XX MB | X.X% |

## Issues Encountered
- **Issue:** [Brief description]
  - **Solution:** [How it was resolved]
  - **Impact:** [Timeline/performance impact]

## Next Week Plan
- [ ] [Specific task 1]
- [ ] [Specific task 2]
- [ ] [Risk mitigation items]

## Code Quality Metrics
- Test coverage: XX%
- Performance regression tests: Pass/Fail
- Memory leak tests: Pass/Fail
- Backward compatibility: Pass/Fail
```

### **Create User Communication Plan**

```markdown
# User Communication Strategy

## Phase 1 Announcements

### Week 1 Announcement (Internal Testing)
"We're beginning Phase 1 of performance optimization focusing on physics calculations. 
No user-facing changes expected. Internal testing shows 15-30% improvements in 
thermodynamics calculations."

### Week 2 Announcement (Expanded Testing)
"Phase 1 continues with transport property optimizations. All existing interfaces 
maintained. Early results show 20-40% improvements in electric current and heat 
flux calculations."

### Week 3 Announcement (Completion)
"Phase 1 complete! Physics calculations now 15-30% faster with no changes to user 
interface. All existing YAML files and Python scripts work unchanged."

## Documentation Updates
- Update performance section in main docs
- Add new physics module documentation
- Create migration guide for advanced users who want to use new calculators directly
```

---

## **Integration with Larger Sarkas Ecosystem**

### **Consider Impact on Related Tools**

```python
# tools/compatibility_check.py
"""
Verify that Phase 1 changes don't break related Sarkas tools and workflows.
"""

def check_postprocessing_compatibility():
    """Verify post-processing tools work with optimized calculations."""
    # Test observables calculations
    # Test transport coefficient calculations
    # Verify plotting and analysis tools

def check_input_output_compatibility():
    """Verify I/O systems work with new structure."""
    # Test checkpoint loading/saving
    # Test HDF5 output formats
    # Verify restart functionality

def check_jupyter_notebook_examples():
    """Test all example notebooks still work."""
    # Run all tutorial notebooks
    # Verify example scripts
    # Test documentation examples
```

### **Plan for Future GPU Acceleration**

```python
# future_planning/gpu_readiness.py
"""
Assess how Phase 1 changes prepare for future GPU acceleration.
"""

def assess_gpu_readiness():
    """Evaluate GPU acceleration potential after Phase 1."""
    
    # Pure functions are GPU-friendly
    gpu_ready_functions = [
        'kinetic_energy',
        'electric_current_density', 
        'species_sum',
        # ... list all pure functions
    ]
    
    # Identify remaining GPU blockers
    gpu_blockers = [
        'Complex class interactions',
        'Python object overhead',
        'Non-contiguous memory access'
    ]
    
    # Plan GPU acceleration roadmap
    create_gpu_acceleration_plan()
```

---

## **Quality Assurance and Risk Management**

### **Automated Quality Checks**

```python
# qa/automated_checks.py
"""
Automated quality assurance checks to run after each change.
"""

def run_comprehensive_qa():
    """Run all quality assurance checks."""
    
    results = {
        'performance': check_performance_targets(),
        'accuracy': check_numerical_accuracy(),
        'compatibility': check_backward_compatibility(),
        'memory': check_memory_usage(),
        'style': check_code_style(),
        'coverage': check_test_coverage()
    }
    
    # Generate QA report
    generate_qa_report(results)
    
    # Alert if any checks fail
    if any(not result['passed'] for result in results.values()):
        send_alert("QA checks failed - manual review required")

def check_performance_targets():
    """Verify performance targets are met."""
    # Run performance benchmarks
    # Compare against baseline targets
    # Return pass/fail with details

def check_numerical_accuracy():
    """Verify numerical accuracy maintained."""
    # Run reference calculations
    # Compare with high precision
    # Check for precision degradation

def check_backward_compatibility():
    """Verify all existing interfaces work."""
    # Test with real user examples
    # Verify YAML parsing unchanged
    # Test observable calculations
```

### **Risk Mitigation Strategies**

```markdown
# Risk Mitigation Plan

## Technical Risks

### Risk: Performance Regression
**Probability:** Medium
**Impact:** High
**Mitigation:**
- Continuous performance benchmarking
- Automated regression detection
- Rollback plan for each week's changes
- Performance gates in CI/CD

### Risk: Numerical Accuracy Loss
**Probability:** Low
**Impact:** Critical
**Mitigation:**
- Comprehensive accuracy testing
- Reference calculation validation
- Floating point precision monitoring
- Physics equation verification

### Risk: Breaking User Code
**Probability:** Low
**Impact:** Critical
**Mitigation:**
- 100% backward compatibility requirement
- Facade pattern for interface preservation
- Extensive user scenario testing
- Gradual rollout strategy

## Project Risks

### Risk: Timeline Delays
**Probability:** Medium
**Impact:** Medium
**Mitigation:**
- Buffer time in each phase
- Parallel work streams where possible
- Clear deliverable milestones
- Regular progress checkpoints

### Risk: Scope Creep
**Probability:** High
**Impact:** Medium
**Mitigation:**
- Strict phase boundaries
- Clear success criteria
- Regular scope reviews
- Defer non-critical improvements
```

This guide provides a complete roadmap for Phase 1 implementation with clear steps, validation criteria, troubleshooting guidance, and preparation for subsequent phases. Each step includes specific prompts for Claude to ensure consistent, high-quality implementation while maintaining the user-friendly interface that makes Sarkas valuable to researchers.
