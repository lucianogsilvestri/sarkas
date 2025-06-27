"""
Comprehensive tests for sarkas.physics.thermodynamics module.

This test suite validates physics calculations, edge cases, performance,
numerical precision, and type safety for all thermodynamics functions.
"""

import numpy as np
import pytest
import warnings
import time
import gc
from typing import Tuple, List
from unittest.mock import MagicMock

# Import the module under test
try:
    from sarkas.physics.thermodynamics import (
        kinetic_energy, 
        temperature_from_kinetic_energy,
        temperature_from_velocities,
        calculate_kinetic_energy,
        calculate_temperature,
        _benchmark_kinetic_energy,
        _validate_functions
    )
    THERMODYNAMICS_AVAILABLE = True
except ImportError:
    THERMODYNAMICS_AVAILABLE = False
    pytestmark = pytest.mark.skip("thermodynamics module not available")

# Physical constants
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
TOLERANCE = 1e-10
RELATIVE_TOLERANCE = 1e-12


class TestKineticEnergySimpleCases:
    """Test kinetic energy calculations with known analytical results."""
    
    def test_single_particle_at_rest(self):
        """Test particle at rest has zero kinetic energy."""
        vel = np.array([[0.0, 0.0, 0.0]])
        mass = np.array([1.0])
        
        ke = kinetic_energy(vel, mass)
        
        assert np.allclose(ke, [0.0], atol=TOLERANCE)
        assert ke.shape == (1,)
    
    def test_single_particle_moving(self):
        """Test single particle with known velocity."""
        # Particle moving at 10 m/s in x-direction, mass = 2 kg
        # KE = 0.5 * 2 * 10^2 = 100 J
        vel = np.array([[10.0, 0.0, 0.0]])
        mass = np.array([2.0])
        
        ke = kinetic_energy(vel, mass)
        expected = np.array([100.0])
        
        assert np.allclose(ke, expected, rtol=RELATIVE_TOLERANCE)
    
    def test_multiple_particles_known_result(self):
        """Test multiple particles with calculated expected results."""
        vel = np.array([
            [10.0, 0.0, 0.0],   # KE = 0.5 * 1 * 100 = 50 J
            [0.0, 5.0, 0.0],    # KE = 0.5 * 2 * 25 = 25 J
            [3.0, 4.0, 0.0]     # KE = 0.5 * 0.5 * 25 = 6.25 J
        ])
        mass = np.array([1.0, 2.0, 0.5])
        
        ke = kinetic_energy(vel, mass)
        expected = np.array([50.0, 25.0, 6.25])
        
        assert np.allclose(ke, expected, rtol=RELATIVE_TOLERANCE)
    
    def test_uniform_mass_broadcasting(self):
        """Test uniform mass broadcasting works correctly."""
        vel = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mass = np.array([2.0])  # Single mass should be broadcasted
        
        ke = kinetic_energy(vel, mass)
        expected = np.array([1.0, 4.0])  # 0.5 * 2 * v^2
        
        assert np.allclose(ke, expected, rtol=RELATIVE_TOLERANCE)
    
    def test_3d_motion(self):
        """Test particles moving in 3D space."""
        vel = np.array([[3.0, 4.0, 0.0]])  # |v| = 5 m/s
        mass = np.array([2.0])
        
        ke = kinetic_energy(vel, mass)
        expected = np.array([25.0])  # 0.5 * 2 * 25
        
        assert np.allclose(ke, expected, rtol=RELATIVE_TOLERANCE)


class TestKineticEnergyConservation:
    """Test energy conservation principles."""
    
    
    def test_scaling_invariance(self):
        """Test that scaling velocities scales kinetic energy by v^2."""
        vel = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 0.0]])
        mass = np.array([1.0, 2.0])
        
        ke_original = kinetic_energy(vel, mass)
        
        # Scale velocities by factor of 2
        vel_scaled = vel * 2.0
        ke_scaled = kinetic_energy(vel_scaled, mass)
        
        # Kinetic energy should scale by 4 (2^2)
        assert np.allclose(ke_scaled, ke_original * 4.0, rtol=RELATIVE_TOLERANCE)
    
    def test_additivity_property(self):
        """Test that kinetic energy is additive for independent components."""
        # Test with velocities in different directions
        vel_x = np.array([[3.0, 0.0, 0.0]])
        vel_y = np.array([[0.0, 4.0, 0.0]])
        vel_combined = np.array([[3.0, 4.0, 0.0]])
        mass = np.array([2.0])
        
        ke_x = kinetic_energy(vel_x, mass)
        ke_y = kinetic_energy(vel_y, mass)
        ke_combined = kinetic_energy(vel_combined, mass)
        
        # KE_combined should equal KE_x + KE_y
        assert np.allclose(ke_combined, ke_x + ke_y, rtol=RELATIVE_TOLERANCE)


class TestTemperatureEquipartitionTheorem:
    """Test temperature calculations using equipartition theorem."""
    
    def test_equipartition_theorem_3d(self):
        """Test equipartition theorem: <KE> = (3/2) * kB * T."""
        # Create a particle with specific kinetic energy
        ke = np.array([3.0 * BOLTZMANN_CONSTANT * 300.0])  # 3 * kB * T for 3D
        mass = np.array([1.0])
        
        T = temperature_from_kinetic_energy(ke, 3, BOLTZMANN_CONSTANT)
        
        # Should recover T = 300 K * 2 (since we used 3*kB*T instead of 1.5*kB*T)
        expected_T = 600.0
        assert np.allclose(T, [expected_T], rtol=RELATIVE_TOLERANCE)
    
    def test_equipartition_theorem_2d(self):
        """Test equipartition theorem in 2D: <KE> = kB * T."""
        ke = np.array([BOLTZMANN_CONSTANT * 300.0])  # kB * T for 2D
        mass = np.array([1.0])
        
        T = temperature_from_kinetic_energy(ke, 2, BOLTZMANN_CONSTANT)
        
        # Should recover T = 300 K * 2 (since formula is T = 2*KE/(dimensions*kB))
        expected_T = 300.0
        assert np.allclose(T, [expected_T], rtol=RELATIVE_TOLERANCE)
    
    def test_temperature_from_velocities_consistency(self):
        """Test that temperature_from_velocities gives same result as two-step process."""
        vel = np.array([[100.0, 200.0, 300.0], [50.0, 100.0, 150.0]])
        mass = np.array([1.67e-27, 1.67e-27])  # Proton mass
        
        # Method 1: Direct calculation
        T_direct = temperature_from_velocities(vel, mass, 3, BOLTZMANN_CONSTANT)
        
        # Method 2: Two-step process
        ke = kinetic_energy(vel, mass)
        T_two_step = temperature_from_kinetic_energy(ke, 3, BOLTZMANN_CONSTANT)
        
        assert np.allclose(T_direct, T_two_step, rtol=RELATIVE_TOLERANCE)
    
    def test_maxwell_boltzmann_temperature(self):
        """Test temperature calculation for Maxwell-Boltzmann distributed velocities."""
        np.random.seed(42)
        n_particles = 10000
        T_target = 300.0  # K
        mass = 1.67e-27  # kg (proton)
        
        # Generate Maxwell-Boltzmann distributed velocities
        sigma = np.sqrt(BOLTZMANN_CONSTANT * T_target / mass)
        vel = np.random.normal(0, sigma, (n_particles, 3))
        mass_array = np.full(n_particles, mass)
        
        # Calculate temperature
        T_calculated = temperature_from_velocities(vel, mass_array, 3, BOLTZMANN_CONSTANT)
        T_mean = T_calculated.mean()
        
        # Should be close to target temperature (within statistical fluctuations)
        assert abs(T_mean - T_target) < 10.0  # Allow 10K deviation due to statistics


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arrays(self):
        """Test behavior with empty input arrays."""
        vel_empty = np.array([]).reshape(0, 3)
        mass_empty = np.array([])
        
        # All functions should handle empty arrays gracefully
        ke = kinetic_energy(vel_empty, mass_empty)
        assert ke.size == 0
        assert ke.shape == (0,)
        
        T = temperature_from_velocities(vel_empty, mass_empty, 3, BOLTZMANN_CONSTANT)
        assert T.size == 0
        

    def test_single_particle(self):
        """Test single particle calculations."""
        vel = np.array([[5.0, 0.0, 0.0]])
        mass = np.array([2.0])
        
        ke = kinetic_energy(vel, mass)
        assert ke.shape == (1,)
        assert np.allclose(ke, [25.0])
    
    
    def test_zero_velocities(self):
        """Test with all zero velocities."""
        vel = np.zeros((5, 3))
        mass = np.ones(5)
        
        ke = kinetic_energy(vel, mass)
        assert np.allclose(ke, np.zeros(5))
        
        T = temperature_from_velocities(vel, mass, 3, BOLTZMANN_CONSTANT)
        assert np.allclose(T, np.zeros(5))
        
    
    def test_zero_mass_handling(self):
        """Test behavior with zero masses."""
        vel = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        mass = np.array([0.0, 1.0])
        
        ke = kinetic_energy(vel, mass)
        expected = np.array([0.0, 2.0])  # Zero mass gives zero KE
        assert np.allclose(ke, expected)
    
    def test_nan_and_inf_handling(self):
        """Test handling of NaN and infinity values."""
        vel = np.array([[np.nan, 0.0, 0.0], [np.inf, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mass = np.array([1.0, 1.0, 1.0])
        
        ke = kinetic_energy(vel, mass)
        
        # First element should be NaN, second should be inf, third should be 0.5
        assert np.isnan(ke[0])
        assert np.isinf(ke[1])
        assert np.allclose(ke[2], [0.5])
    
    def test_negative_mass_handling(self):
        """Test behavior with negative masses (unphysical but should not crash)."""
        vel = np.array([[1.0, 0.0, 0.0]])
        mass = np.array([-1.0])
        
        # Should not crash, but result will be negative
        ke = kinetic_energy(vel, mass)
        assert ke[0] == -0.5


class TestPerformanceVsBaseline:
    """Performance comparison tests."""
    
    def create_baseline_kinetic_energy(self, velocities, masses):
        """Create baseline implementation for comparison."""
        return 0.5 * masses * (velocities * velocities).sum(axis=-1)
    
    def create_baseline_temperature(self, velocities, masses, dimensions, kB):
        """Create baseline temperature implementation."""
        ke = self.create_baseline_kinetic_energy(velocities, masses)
        return 2.0 * ke / (dimensions * kB)
    
    @pytest.mark.parametrize("n_particles", [1000, 10000, 100000])
    def test_kinetic_energy_performance(self, n_particles):
        """Test kinetic energy calculation performance vs baseline."""
        np.random.seed(42)
        vel = np.random.randn(n_particles, 3) * 100.0
        mass = np.random.uniform(0.1, 10.0, n_particles)
        
        # Warm up both implementations
        _ = kinetic_energy(vel[:10], mass[:10])
        _ = self.create_baseline_kinetic_energy(vel[:10], mass[:10])
        
        # Time new implementation
        n_iterations = 10
        result_new = kinetic_energy(vel, mass) # Warm-up call
        start_time = time.time()
        for _ in range(n_iterations):
            result_new = kinetic_energy(vel, mass)
        time_new = (time.time() - start_time) / n_iterations
        
        # Time baseline implementation
        start_time = time.time()
        for _ in range(n_iterations):
            result_baseline = self.create_baseline_kinetic_energy(vel, mass)
        time_baseline = (time.time() - start_time) / n_iterations
        
        # Verify results are the same
        assert np.allclose(result_new, result_baseline, rtol=1e-10)
        
        # New implementation should be faster (or at least not much slower)
        speedup_ratio = time_baseline / time_new
        print(f"N={n_particles}: New={time_new:.6f}s, Baseline={time_baseline:.6f}s, "
              f"Speedup={speedup_ratio:.2f}x")
        
        # Allow some tolerance for timing variations
        assert speedup_ratio >= 0.5, f"Performance regression: {speedup_ratio:.2f}x"
    
    def test_memory_usage_validation(self):
        """Test that memory usage is reasonable."""
        n_particles = 100000
        vel = np.random.randn(n_particles, 3) * 100.0
        mass = np.random.uniform(0.1, 10.0, n_particles)
        
        # Force garbage collection
        gc.collect()
        
        # Calculate kinetic energy multiple times
        for _ in range(10):
            ke = kinetic_energy(vel, mass)
            del ke
        
        # Should not accumulate excessive memory
        gc.collect()
        # This is more of a visual check during development
    
    def test_compilation_vs_runtime_tradeoff(self):
        """Test Numba compilation time vs runtime benefits."""
        # Small arrays (compilation overhead might dominate)
        vel_small = np.random.randn(10, 3)
        mass_small = np.ones(10)
        
        # Large arrays (runtime benefits should dominate)
        vel_large = np.random.randn(100000, 3)
        mass_large = np.ones(100000)

        _ = kinetic_energy(vel_small, mass_small) # Warm-up call        
        
        # Time small arrays
        start_time = time.time()
        ke_small = kinetic_energy(vel_small, mass_small)
        time_small = time.time() - start_time
        
        # Time large arrays
        start_time = time.time()
        ke_large = kinetic_energy(vel_large, mass_large)
        time_large = time.time() - start_time
        
        # For large arrays, per-particle time should be much smaller
        time_per_particle_small = time_small / 10
        time_per_particle_large = time_large / 100000
        
        # This is mainly for informational purposes
        print(f"Small array: {time_per_particle_small:.8e}s per particle")
        print(f"Large array: {time_per_particle_large:.8e}s per particle")


class TestNumericalPrecision:
    """Test numerical precision and accuracy."""
    
    def test_float32_vs_float64_precision(self):
        """Test precision differences between float32 and float64."""
        vel_f64 = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        vel_f32 = vel_f64.astype(np.float32)
        mass_f64 = np.array([2.0], dtype=np.float64)
        mass_f32 = mass_f64.astype(np.float32)
        
        ke_f64 = kinetic_energy(vel_f64, mass_f64)
        ke_f32 = kinetic_energy(vel_f32, mass_f32)
        
        # Results should be close but float32 might have less precision
        assert np.allclose(ke_f64, ke_f32, rtol=1e-6)
    
    def test_large_number_precision(self):
        """Test precision with very large numbers."""
        vel = np.array([[1e10, 0.0, 0.0]])
        mass = np.array([1e-20])
        
        ke = kinetic_energy(vel, mass)
        expected = 0.5 * 1e-20 * 1e20  # Should be 5.0
        
        assert np.allclose(ke, [expected], rtol=1e-10)
    
    def test_small_number_precision(self):
        """Test precision with very small numbers."""
        vel = np.array([[1e-10, 0.0, 0.0]])
        mass = np.array([1e10])
        
        ke = kinetic_energy(vel, mass)
        expected = 0.5 * 1e10 * 1e-20  # Should be 5e-11
        
        assert np.allclose(ke, [expected], rtol=1e-10)
    
    def test_analytical_solution_accuracy(self):
        """Test accuracy against known analytical solutions."""
        # Perfect sphere of uniform density rotating
        # For a particle at distance r with angular velocity ω: v = ω * r
        omega = 2.0  # rad/s
        r = 3.0      # m
        mass = 5.0   # kg
        
        vel = np.array([[0.0, omega * r, 0.0]])  # Tangential velocity
        mass_array = np.array([mass])
        
        ke = kinetic_energy(vel, mass_array)
        expected = 0.5 * mass * (omega * r) ** 2
        
        assert np.allclose(ke, [expected], rtol=1e-14)


class TestBackwardCompatibility:
    """Test backward compatibility wrapper functions."""
    
    def test_calculate_kinetic_energy_wrapper(self):
        """Test calculate_kinetic_energy wrapper function."""
        vel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mass = np.array([1.0, 2.0])
        
        # Both should give same result
        ke_new = kinetic_energy(vel, mass)
        ke_wrapper = calculate_kinetic_energy(vel, mass)
        
        assert np.allclose(ke_new, ke_wrapper, rtol=1e-14)
    
    def test_calculate_temperature_wrapper(self):
        """Test calculate_temperature wrapper function."""
        vel = np.array([[100.0, 0.0, 0.0]])
        mass = np.array([1.67e-27])
        
        # Test with default parameters
        T_wrapper = calculate_temperature(vel, mass)
        T_direct = temperature_from_velocities(vel, mass, 3, BOLTZMANN_CONSTANT)
        
        assert np.allclose(T_wrapper, T_direct, rtol=1e-14)
    
    def test_calculate_temperature_with_kwargs(self):
        """Test calculate_temperature wrapper with additional kwargs."""
        vel = np.array([[100.0, 0.0, 0.0]])
        mass = np.array([1.67e-27])
        
        # Test with custom parameters and extra kwargs (should be ignored)
        T_wrapper = calculate_temperature(
            vel, mass, 
            dimensions=2, 
            kB=BOLTZMANN_CONSTANT,
            extra_param="ignored"  # Should be ignored
        )
        T_direct = temperature_from_velocities(vel, mass, 2, BOLTZMANN_CONSTANT)
        
        assert np.allclose(T_wrapper, T_direct, rtol=1e-14)


class TestModuleIntegration:
    """Integration tests for the entire module."""
    
    def test_module_validation_function(self):
        """Test the built-in validation function."""
        # Should not raise any exceptions
        _validate_functions()
    
    def test_benchmark_function(self):
        """Test the built-in benchmark function."""
        # Should return a reasonable time
        avg_time = _benchmark_kinetic_energy(n_particles=1000, n_iterations=10)
        
        assert isinstance(avg_time, float)
        assert avg_time > 0.0
        assert avg_time < 1.0  # Should be much less than 1 second
    
    def test_all_functions_together(self):
        """Test all functions working together in a typical workflow."""
        np.random.seed(42)
        n_particles = 1000
        
        # Generate test data
        vel = np.random.randn(n_particles, 3) * 100.0
        mass = np.random.uniform(0.5, 2.0, n_particles)
        
        # Calculate all quantities
        ke = kinetic_energy(vel, mass)
        T = temperature_from_velocities(vel, mass, 3, BOLTZMANN_CONSTANT)

        # Verify shapes and basic properties
        assert ke.shape == (n_particles,)
        assert T.shape == (n_particles,)

        # Verify no NaN or inf values
        assert not np.isnan(ke).any()
        assert not np.isnan(T).any()
  
        # Verify physics constraints
        assert (ke >= 0).all()  # Kinetic energy non-negative
        assert (T >= 0).all()   # Temperature non-negative
        

class TestPhysicsValidation:
    """Additional physics validation tests."""
    
    def test_virial_theorem_consistency(self):
        """Test consistency with virial theorem for harmonic oscillator."""
        # For a 3D harmonic oscillator: <T> = <V> (virial theorem)
        # We can't test potential energy here, but we can test kinetic energy scaling
        np.random.seed(42)
        
        # Generate velocities for harmonic oscillator (Gaussian distribution)
        n_particles = 10000
        T_target = 300.0  # K
        mass = 1.67e-27   # kg
        
        sigma = np.sqrt(BOLTZMANN_CONSTANT * T_target / mass)
        vel = np.random.normal(0, sigma, (n_particles, 3))
        mass_array = np.full(n_particles, mass)
        
        # Calculate average kinetic energy per particle
        ke = kinetic_energy(vel, mass_array)
        avg_ke_per_particle = ke.mean()
        
        # For 3D: <KE> = (3/2) * kB * T
        expected_avg_ke = 1.5 * BOLTZMANN_CONSTANT * T_target
        
        # Should be within statistical error (generous tolerance for Monte Carlo)
        relative_error = abs(avg_ke_per_particle - expected_avg_ke) / expected_avg_ke
        assert relative_error < 0.05, f"Relative error: {relative_error:.3f}"
    
    def test_galilean_invariance(self):
        """Test that physics is invariant under Galilean transformations."""
        vel = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mass = np.array([1.0, 2.0])
        
        # Calculate kinetic energy in original frame
        ke_original = kinetic_energy(vel, mass)
        
        # Apply Galilean transformation (add constant velocity)
        boost_velocity = np.array([10.0, 20.0, 30.0])
        vel_boosted = vel + boost_velocity
        
        # Kinetic energy should change (not invariant), but in a predictable way
        ke_boosted = kinetic_energy(vel_boosted, mass)
        
        # The difference should be: ΔKE = m*v_boost*v_original + 0.5*m*v_boost^2
        v_boost_sq = np.sum(boost_velocity**2)
        expected_change = np.array([
            mass[0] * np.dot(boost_velocity, vel[0]) + 0.5 * mass[0] * v_boost_sq,
            mass[1] * np.dot(boost_velocity, vel[1]) + 0.5 * mass[1] * v_boost_sq
        ])
        
        actual_change = ke_boosted - ke_original
        assert np.allclose(actual_change, expected_change, rtol=1e-12)
    
    def test_rotational_invariance(self):
        """Test that kinetic energy is invariant under rotations."""
        # Original velocity
        vel = np.array([[1.0, 0.0, 0.0]])
        mass = np.array([1.0])
        
        ke_original = kinetic_energy(vel, mass)
        
        # Rotate by 90 degrees around z-axis
        rotation_matrix = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        
        vel_rotated = vel @ rotation_matrix.T
        ke_rotated = kinetic_energy(vel_rotated, mass)
        
        # Kinetic energy should be invariant under rotation
        assert np.allclose(ke_original, ke_rotated, rtol=1e-14)


class TestStressAndEdgeCases:
    """Stress tests and additional edge cases."""
    
    def test_very_large_systems(self):
        """Test with very large particle systems."""
        try:
            n_particles = 1000000  # 1 million particles
            vel = np.random.randn(n_particles, 3).astype(np.float32)  # Use float32 to save memory
            mass = np.ones(n_particles, dtype=np.float32)
            
            start_time = time.time()
            ke = kinetic_energy(vel, mass)
            elapsed_time = time.time() - start_time
            
            assert ke.shape == (n_particles,)
            assert not np.isnan(ke).any()
            print(f"Large system test: {n_particles} particles in {elapsed_time:.3f}s")
            
        except MemoryError:
            pytest.skip("Not enough memory for large system test")
    
    def test_extreme_velocity_values(self):
        """Test with extreme velocity values."""
        # Test with very high velocities (relativistic regime - not physically correct but numerically interesting)
        vel_high = np.array([[3e8, 0.0, 0.0]])  # Speed of light
        mass = np.array([1e-30])  # Very small mass
        
        ke_high = kinetic_energy(vel_high, mass)
        assert np.isfinite(ke_high[0])
        
        # Test with very low velocities
        vel_low = np.array([[1e-100, 0.0, 0.0]])
        ke_low = kinetic_energy(vel_low, mass)
        assert ke_low[0] >= 0.0
        assert np.isfinite(ke_low[0])
    
    def test_pathological_mass_distributions(self):
        """Test with pathological mass distributions."""
        vel = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        
        # Very different mass scales
        mass = np.array([1e-30, 1.0, 1e30])
        
        ke = kinetic_energy(vel, mass)
        expected = 0.5 * mass * 1.0  # v^2 = 1
        
        assert np.allclose(ke, expected, rtol=1e-10)
        assert np.isfinite(ke).all()
    
    def test_thread_safety_simulation(self):
        """Simulate thread safety by running multiple calculations simultaneously."""
        import concurrent.futures
        
        def calculate_ke_batch(seed):
            np.random.seed(seed)
            vel = np.random.randn(1000, 3)
            mass = np.ones(1000)
            return kinetic_energy(vel, mass)
        
        # Run multiple calculations "simultaneously"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(calculate_ke_batch, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # All should complete without error
        for result in results:
            assert result.shape == (1000,)
            assert not np.isnan(result).any()


class TestDocumentationExamples:
    """Test all examples from function docstrings."""
    
    def test_kinetic_energy_docstring_examples(self):
        """Test examples from kinetic_energy docstring."""
        # Example 1: Single particle
        vel = np.array([[10.0, 0.0, 0.0]])
        mass = np.array([1.0])
        ke = kinetic_energy(vel, mass)
        assert np.allclose(ke, [50.0], rtol=1e-10)
        
        # Example 2: Multiple particles
        vel = np.array([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        mass = np.array([1.0, 2.0])
        ke = kinetic_energy(vel, mass)
        expected = np.array([50.0, 25.0])
        assert np.allclose(ke, expected, rtol=1e-10)
    
    def test_temperature_from_ke_docstring_example(self):
        """Test example from temperature_from_kinetic_energy docstring."""
        # Approximate kinetic energy at room temperature
        ke = np.array([6.21e-21])  # J
        T = temperature_from_kinetic_energy(ke, 3, BOLTZMANN_CONSTANT)
        
        # Should be approximately 300 K
        assert abs(T[0] - 300.0) < 1.0  # Within 1 K


class TestParametrizedCases:
    """Parametrized tests for systematic coverage."""
    
    @pytest.mark.parametrize("n_particles", [1, 10, 100, 1000])
    @pytest.mark.parametrize("dimensions", [1, 2, 3])
    def test_temperature_scaling(self, n_particles, dimensions):
        """Test temperature calculation for different system sizes and dimensions."""
        np.random.seed(42)
        if dimensions == 1:
            vel = np.random.randn(n_particles, 3)
            vel[:, 1:] = 0  # Only x-component
        elif dimensions == 2:
            vel = np.random.randn(n_particles, 3)
            vel[:, 2] = 0   # Only x and y components
        else:
            vel = np.random.randn(n_particles, 3)
        
        mass = np.ones(n_particles)
        
        T = temperature_from_velocities(vel, mass, dimensions, BOLTZMANN_CONSTANT)
        
        assert T.shape == (n_particles,)
        assert (T >= 0).all()
        assert np.isfinite(T).all()
    
    @pytest.mark.parametrize("mass_ratio", [1e-6, 1e-3, 1.0, 1e3, 1e6])
    def test_mass_ratio_effects(self, mass_ratio):
        """Test effects of different mass ratios."""
        vel = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        mass = np.array([1.0, mass_ratio])
        
        ke = kinetic_energy(vel, mass)
        expected = 0.5 * mass * 1.0
        
        assert np.allclose(ke, expected, rtol=1e-10)

# Performance benchmarking suite
class TestPerformanceBenchmarks:
    """Comprehensive performance benchmarking."""
    
    def setup_method(self):
        """Set up benchmark data."""
        self.system_sizes = [1000, 10000]  # Reduced for faster testing
        self.n_iterations = 3  # Reduced for faster testing
        
        # Pre-generate test data
        self.test_data = {}
        for size in self.system_sizes:
            np.random.seed(42)  # Consistent data
            self.test_data[size] = {
                'vel': np.random.randn(size, 3) * 100.0,
                'mass': np.random.uniform(0.1, 10.0, size)
            }
    
    def benchmark_function(self, func, *args, n_iterations=None):
        """Generic function benchmarking."""
        if n_iterations is None:
            n_iterations = self.n_iterations
        
        # Warm up
        _ = func(*args)
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.time()
            result = func(*args)
            times.append(time.time() - start)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'result_shape': result.shape if hasattr(result, 'shape') else None
        }
    
    @pytest.mark.slow
    def test_kinetic_energy_scaling(self):
        """Benchmark kinetic energy calculation scaling."""
        results = {}
        
        for size in self.system_sizes:
            vel = self.test_data[size]['vel']
            mass = self.test_data[size]['mass']
            
            benchmark = self.benchmark_function(kinetic_energy, vel, mass)
            results[size] = benchmark
            
            print(f"Kinetic Energy - N={size}: "
                  f"{benchmark['mean_time']:.6f}±{benchmark['std_time']:.6f}s "
                  f"({benchmark['mean_time']/size*1e6:.3f} μs/particle)")
        
        # Check that function works (basic validation)
        for size in self.system_sizes:
            assert results[size]['result_shape'] == (size,)
            assert results[size]['mean_time'] > 0
    
    @pytest.mark.slow
    def test_temperature_calculation_benchmark(self):
        """Benchmark temperature calculation."""
        size = 10000
        vel = self.test_data[size]['vel']
        mass = self.test_data[size]['mass']
        
        # Benchmark direct method
        benchmark_direct = self.benchmark_function(
            temperature_from_velocities, vel, mass, 3, BOLTZMANN_CONSTANT
        )
        
        # Benchmark two-step method
        def two_step_temperature(vel, mass):
            ke = kinetic_energy(vel, mass)
            return temperature_from_kinetic_energy(ke, 3, BOLTZMANN_CONSTANT)
        
        benchmark_two_step = self.benchmark_function(two_step_temperature, vel, mass)
        
        print(f"Temperature N={size}:")
        print(f"  Direct: {benchmark_direct['mean_time']:.6f}s")
        print(f"  Two-step: {benchmark_two_step['mean_time']:.6f}s")
        
        # Basic validation - both should work and give reasonable times
        assert benchmark_direct['mean_time'] > 0
        assert benchmark_two_step['mean_time'] > 0
        assert benchmark_direct['result_shape'] == (size,)
    
    @pytest.mark.slow  
    def test_memory_efficiency(self):
        """Test memory efficiency of calculations (simplified version)."""
        # Simplified test that doesn't require psutil
        initial_arrays = []
        
        # Perform calculations and store references
        for i in range(10):  # Reduced iterations
            vel = np.random.randn(1000, 3)  # Smaller arrays
            mass = np.ones(1000)
            ke = kinetic_energy(vel, mass)
            initial_arrays.append((vel, mass, ke))
        
        # Clear references
        del initial_arrays
        
        # Basic test - should not crash or cause obvious memory issues
        # More sophisticated memory testing would require additional dependencies
        
        # Perform one more calculation to ensure everything still works
        vel = np.random.randn(1000, 3)
        mass = np.ones(1000)
        ke = kinetic_energy(vel, mass)
        
        assert ke.shape == (1000,)
        assert not np.isnan(ke).any()


# Integration with pytest markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as a slow performance test"
    )


if __name__ == "__main__":
    # Run basic validation when script is executed directly
    print("Running thermodynamics module tests...")
    
    if not THERMODYNAMICS_AVAILABLE:
        print("❌ Thermodynamics module not available")
        exit(1)
    
    # Quick validation
    try:
        _validate_functions()
        print("✓ Built-in validation passed")
    except Exception as e:
        print(f"❌ Built-in validation failed: {e}")
        exit(1)
    
    # Run a subset of tests manually
    try:
        test_ke = TestKineticEnergySimpleCases()
        test_ke.test_single_particle_at_rest()
        test_ke.test_single_particle_moving()
        print("✓ Kinetic energy tests passed")
        
        test_temp = TestTemperatureEquipartitionTheorem()
        test_temp.test_temperature_from_velocities_consistency()
        print("✓ Temperature tests passed")
        
        test_cm = TestCenterOfMassConservation()
        test_cm.test_total_momentum_conservation()
        print("✓ Center of mass tests passed")
        
        test_edge = TestEdgeCases()
        test_edge.test_empty_arrays()
        test_edge.test_single_particle()
        print("✓ Edge case tests passed")
        
        # Performance check
        avg_time = _benchmark_kinetic_energy(n_particles=1000, n_iterations=10)
        print(f"✓ Performance benchmark: {avg_time*1e6:.2f} μs average")
        
        print("\nAll manual tests passed! Run 'pytest sarkas/physics/tests/test_thermodynamics.py -v' for full test suite.")
        
    except Exception as e:
        print(f"❌ Manual tests failed: {e}")
        exit(1)