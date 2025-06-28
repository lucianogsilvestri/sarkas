#!/usr/bin/env python3
"""
Performance comparison between species_sum and np.bincount with weights.

This script benchmarks the performance of:
1. species_sum (Numba-compiled custom function)
2. np.bincount with weights (NumPy's optimized C implementation)
3. fast_species_sum (Numba-compiled bincount for each dimension)

For different system sizes and data types.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from numba import njit
from sarkas.physics.aggregation import species_sum


@njit
def bincount_scalar(species_id, values, num_species):
    """Fast scalar aggregation using np.bincount."""
    return np.bincount(species_id, weights=values, minlength=num_species)


@njit
def bincount_vector(species_id, vectors, num_species):
    """Fast vector aggregation using np.bincount for each component."""
    result = np.zeros((num_species, vectors.shape[1]), dtype=vectors.dtype)
    for j in range(vectors.shape[1]):
        result[:, j] = np.bincount(species_id, weights=vectors[:, j], minlength=num_species)
    return result


@njit
def bincount_tensor(species_id, tensors, num_species):
    """Fast tensor aggregation using np.bincount for each component."""
    result = np.zeros((num_species, tensors.shape[1], tensors.shape[2]), dtype=tensors.dtype)
    for i in range(tensors.shape[1]):
        for j in range(tensors.shape[2]):
            result[:, i, j] = np.bincount(species_id, weights=tensors[:, i, j], minlength=num_species)
    return result


def fast_species_sum(per_particle_array, species_id, num_species):
    """Fast species aggregation that dispatches based on array shape."""
    if per_particle_array.ndim == 1:
        return bincount_scalar(species_id, per_particle_array, num_species)
    elif per_particle_array.ndim == 2:
        return bincount_vector(species_id, per_particle_array, num_species)
    elif per_particle_array.ndim == 3:
        return bincount_tensor(species_id, per_particle_array, num_species)
    else:
        raise ValueError("Unsupported array shape for fast_species_sum")


def generate_test_data(N, num_species=3):
    """Generate test data for benchmarking."""
    # Random species IDs
    species_id = np.random.randint(0, num_species, N)
    
    # Random values to sum
    values = np.random.randn(N)
    
    # Random vectors
    vectors = np.random.randn(N, 3)
    
    # Random tensors
    tensors = np.random.randn(N, 3, 3)
    
    return species_id, values, vectors, tensors


def benchmark_scalar_sum(species_id, values, num_species, num_runs=100):
    """Benchmark scalar sum operations."""
    results = {}
    
    # Test species_sum
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_species_sum = species_sum(values, species_id, num_species)
    species_sum_time = (time.perf_counter() - start_time) / num_runs
    
    # Test np.bincount with weights
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_bincount = np.bincount(species_id, weights=values, minlength=num_species)
    bincount_time = (time.perf_counter() - start_time) / num_runs
    
    # Test fast_species_sum
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_fast = fast_species_sum(values, species_id, num_species)
    fast_time = (time.perf_counter() - start_time) / num_runs
    
    # Verify results are identical
    np.testing.assert_allclose(result_species_sum, result_bincount, rtol=1e-15)
    np.testing.assert_allclose(result_species_sum, result_fast, rtol=1e-15)
    
    results['species_sum'] = species_sum_time
    results['bincount'] = bincount_time
    results['fast_species_sum'] = fast_time
    results['speedup_vs_species_sum'] = species_sum_time / fast_time
    results['speedup_vs_bincount'] = bincount_time / fast_time
    
    return results


def benchmark_vector_sum(species_id, vectors, num_species, num_runs=100):
    """Benchmark vector sum operations."""
    results = {}
    
    # Test fast_species_sum
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_fast = fast_species_sum(vectors, species_id, num_species)
    fast_time = (time.perf_counter() - start_time) / num_runs
    
    # Test bincount_vector directly
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_bincount = bincount_vector(species_id, vectors, num_species)
    bincount_time = (time.perf_counter() - start_time) / num_runs
    
    # Verify results are identical
    np.testing.assert_allclose(result_fast, result_bincount, rtol=1e-15)
    
    results['fast_species_sum'] = fast_time
    results['bincount_vector'] = bincount_time
    results['speedup_vs_bincount'] = bincount_time / fast_time
    
    return results


def benchmark_tensor_sum(species_id, tensors, num_species, num_runs=100):
    """Benchmark tensor sum operations."""
    results = {}
    
    # Test fast_species_sum
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_fast = fast_species_sum(tensors, species_id, num_species)
    fast_time = (time.perf_counter() - start_time) / num_runs
    
    # Test bincount_tensor directly
    start_time = time.perf_counter()
    for _ in range(num_runs):
        result_bincount = bincount_tensor(species_id, tensors, num_species)
    bincount_time = (time.perf_counter() - start_time) / num_runs
    
    # Verify results are identical
    np.testing.assert_allclose(result_fast, result_bincount, rtol=1e-15)
    
    results['fast_species_sum'] = fast_time
    results['bincount_tensor'] = bincount_time
    results['speedup_vs_bincount'] = bincount_time / fast_time
    
    return results


def run_benchmarks():
    """Run comprehensive benchmarks."""
    system_sizes = [1000, 5000, 10000, 100000]
    num_species = 3
    num_runs = 100
    
    print("Performance Comparison: species_sum vs np.bincount vs fast_species_sum")
    print("=" * 70)
    print(f"Number of species: {num_species}")
    print(f"Number of runs per test: {num_runs}")
    print()
    
    # Store results for plotting
    scalar_results = {'N': [], 'species_sum': [], 'bincount': [], 'fast_species_sum': [], 'speedup': []}
    vector_results = {'N': [], 'species_sum': [], 'fast_species_sum': [], 'bincount_vector': [], 'speedup': []}
    tensor_results = {'N': [], 'species_sum': [], 'fast_species_sum': [], 'bincount_tensor': [], 'speedup': []}
    
    for N in system_sizes:
        print(f"Testing N = {N:,}")
        print("-" * 40)
        
        # Generate test data
        species_id, values, vectors, tensors = generate_test_data(N, num_species)
        
        # Benchmark scalar operations
        scalar_bench = benchmark_scalar_sum(species_id, values, num_species, num_runs)
        print(f"  Scalar sum:")
        print(f"    species_sum: {scalar_bench['species_sum']*1000:.3f} ms")
        print(f"    np.bincount: {scalar_bench['bincount']*1000:.3f} ms")
        print(f"    fast_species_sum: {scalar_bench['fast_species_sum']*1000:.3f} ms")
        print(f"    speedup vs species_sum: {scalar_bench['speedup_vs_species_sum']:.2f}x")
        print(f"    speedup vs bincount: {scalar_bench['speedup_vs_bincount']:.2f}x")
        
        scalar_results['N'].append(N)
        scalar_results['species_sum'].append(scalar_bench['species_sum']*1000)
        scalar_results['bincount'].append(scalar_bench['bincount']*1000)
        scalar_results['fast_species_sum'].append(scalar_bench['fast_species_sum']*1000)
        scalar_results['speedup'].append(scalar_bench['speedup_vs_species_sum'])
        
        # Benchmark vector operations
        vector_bench = benchmark_vector_sum(species_id, vectors, num_species, num_runs)
        print(f"  Vector sum:")
        print(f"    fast_species_sum: {vector_bench['fast_species_sum']*1000:.3f} ms")
        print(f"    bincount_vector: {vector_bench['bincount_vector']*1000:.3f} ms")
        print(f"    speedup vs bincount: {vector_bench['speedup_vs_bincount']:.2f}x")
        
        vector_results['N'].append(N)
        vector_results['fast_species_sum'].append(vector_bench['fast_species_sum']*1000)
        vector_results['bincount_vector'].append(vector_bench['bincount_vector']*1000)
        vector_results['speedup'].append(vector_bench['speedup_vs_bincount'])
        
        # Benchmark tensor operations
        tensor_bench = benchmark_tensor_sum(species_id, tensors, num_species, num_runs)
        print(f"  Tensor sum:")
        print(f"    fast_species_sum: {tensor_bench['fast_species_sum']*1000:.3f} ms")
        print(f"    bincount_tensor: {tensor_bench['bincount_tensor']*1000:.3f} ms")
        print(f"    speedup vs bincount: {tensor_bench['speedup_vs_bincount']:.2f}x")
        print()
        
        tensor_results['N'].append(N)
        tensor_results['fast_species_sum'].append(tensor_bench['fast_species_sum']*1000)
        tensor_results['bincount_tensor'].append(tensor_bench['bincount_tensor']*1000)
        tensor_results['speedup'].append(tensor_bench['speedup_vs_bincount'])
    
    return scalar_results, vector_results, tensor_results


def plot_results(scalar_results, vector_results, tensor_results):
    """Create performance comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: species_sum vs np.bincount vs fast_species_sum', fontsize=16)
    
    # Scalar performance
    ax1 = axes[0, 0]
    ax1.loglog(scalar_results['N'], scalar_results['species_sum'], 'o-', label='species_sum', linewidth=2)
    ax1.loglog(scalar_results['N'], scalar_results['bincount'], 's-', label='np.bincount', linewidth=2)
    ax1.loglog(scalar_results['N'], scalar_results['fast_species_sum'], '^-', label='fast_species_sum', linewidth=2)
    ax1.set_xlabel('Number of particles (N)')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Scalar Sum Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Vector performance
    ax2 = axes[0, 1]
    ax2.loglog(vector_results['N'], vector_results['fast_species_sum'], '^-', label='fast_species_sum', linewidth=2)
    ax2.loglog(vector_results['N'], vector_results['bincount_vector'], 's-', label='bincount_vector', linewidth=2)
    ax2.set_xlabel('Number of particles (N)')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Vector Sum Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Tensor performance
    ax3 = axes[1, 0]
    ax3.loglog(tensor_results['N'], tensor_results['species_sum'], 'o-', label='species_sum', linewidth=2)
    ax3.loglog(tensor_results['N'], tensor_results['fast_species_sum'], '^-', label='fast_species_sum', linewidth=2)
    ax3.loglog(tensor_results['N'], tensor_results['bincount_tensor'], 's-', label='bincount_tensor', linewidth=2)
    ax3.set_xlabel('Number of particles (N)')
    ax3.set_ylabel('Time (ms)')
    ax3.set_title('Tensor Sum Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Speedup comparison
    ax4 = axes[1, 1]
    ax4.semilogx(scalar_results['N'], scalar_results['speedup'], 'o-', label='scalar (fast/species_sum)', linewidth=2)
    ax4.semilogx(vector_results['N'], vector_results['speedup'], 's-', label='vector (fast/species_sum)', linewidth=2)
    ax4.semilogx(tensor_results['N'], tensor_results['speedup'], '^-', label='tensor (fast/species_sum)', linewidth=2)
    ax4.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='baseline')
    ax4.set_xlabel('Number of particles (N)')
    ax4.set_ylabel('Speedup')
    ax4.set_title('Performance Speedup (fast_species_sum vs species_sum)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('aggregation_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary(scalar_results, vector_results, tensor_results):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("\nScalar Operations (fast_species_sum vs others):")
    print("  fast_species_sum speedup over species_sum:")
    for i, N in enumerate(scalar_results['N']):
        speedup = scalar_results['speedup'][i]
        print(f"    N={N:,}: {speedup:.2f}x faster")
    
    print("\nVector Operations (fast_species_sum vs others):")
    print("  fast_species_sum speedup over species_sum:")
    for i, N in enumerate(vector_results['N']):
        speedup = vector_results['speedup'][i]
        print(f"    N={N:,}: {speedup:.2f}x faster")
    
    print("\nTensor Operations (fast_species_sum vs others):")
    print("  fast_species_sum speedup over species_sum:")
    for i, N in enumerate(tensor_results['N']):
        speedup = tensor_results['speedup'][i]
        print(f"    N={N:,}: {speedup:.2f}x faster")
    


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run benchmarks
    scalar_results, vector_results, tensor_results = run_benchmarks()
    
    # Print summary
    print_summary(scalar_results, vector_results, tensor_results)
    
    # Create plots
    try:
        plot_results(scalar_results, vector_results, tensor_results)
        print("\nPerformance plots saved as 'aggregation_performance_comparison.png'")
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    print("\nBenchmark completed!") 