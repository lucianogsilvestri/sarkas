import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Import required packages
try:
    import pyfftw
    from pyfftw.builders import fftn, ifftn
    from numpy.fft import fftshift, ifftshift
    PYFFTW_AVAILABLE = True
    print("pyfftw imported successfully")
except ImportError:
    PYFFTW_AVAILABLE = False
    print("pyfftw not available - install with: pip install pyfftw")
    exit(1)

class FFTWObjects:
    """Optimized FFT objects for reuse across timesteps"""
    
    def __init__(self, mesh_sizes, threads=None):
        self.shape = (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0])
        self.threads = threads
        
        # Create aligned arrays for optimal SIMD performance
        self.fft_input = pyfftw.empty_aligned(self.shape, dtype=complex)
        self.fft_output = pyfftw.empty_aligned(self.shape, dtype=complex)
        
        # Prepare kwargs for FFTW creation
        fftw_kwargs = {
            'flags': ['FFTW_MEASURE'],
            'axes': (0, 1, 2)  # FFT over all axes like fftn
        }
        if threads is not None:
            fftw_kwargs['threads'] = threads
        
        # Create forward FFT object (for charge density)
        self.forward = pyfftw.FFTW(
            self.fft_input, self.fft_output,
            direction='FFTW_FORWARD',
            **fftw_kwargs
        )
        
        # Create backward FFT object (for electric fields and potential) 
        self.backward = pyfftw.FFTW(
            self.fft_input, self.fft_output,
            direction='FFTW_BACKWARD',
            **fftw_kwargs
        )
    
    def forward_transform(self, input_array, output_array):
        """Perform forward FFT with data copy.
                        
        Parameters
        ----------
        input_array : np.ndarray
            Input array containing data in real space. Shape must match the FFTW object.
        output_array : np.ndarray
            Output array to store the result in k-space. Must match the FFTW object shape.

        Returns
        -------
        np.ndarray
            The output array containing the transformed data in k-space.
        """
        self.fft_input[:] = input_array
        result = self.forward()  # result is a VIEW of self.fft_output
        output_array[:] = result  # Copy data INTO user's existing array
    
    def backward_transform(self, input_array, output_array):
        """Perform backward FFT with data copy.
        
        Parameters
        ----------
        input_array : np.ndarray
            Input array containing data in k-space. Shape must match the FFTW object.
        output_array : np.ndarray
            Output array to store the result in real space. Must match the FFTW object shape.

        Returns
        -------
        np.ndarray
            The output array containing the transformed data in real space.        
        """
        self.fft_input[:] = input_array
        result = self.backward()  # result is a VIEW of self.fft_output
        output_array[:] = result  # Copy data INTO user's existing array

class CorrectFFTWObjects:
    """FFT objects using builders internally - guarantees identical results to original code"""
    
    def __init__(self, mesh_sizes, threads=None):
        self.shape = (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0])
        # Create dummy array of correct shape and type
        dummy_array = pyfftw.empty_aligned(self.shape, dtype=complex)
        
        # Prepare kwargs for builders
        builder_kwargs = {}
        if threads is not None:
            builder_kwargs['threads'] = threads
        
        # Use builders to create the objects (this should match exactly)
        self.forward_fft = fftn(dummy_array, **builder_kwargs)
        self.backward_fft = ifftn(dummy_array, **builder_kwargs)
    
    def forward_transform(self, input_array):
        """Perform forward FFT - reuses the builder object"""
        # Copy data into the builder's input array
        self.forward_fft.input_array[:] = input_array
        return self.forward_fft().copy()
    
    def backward_transform(self, input_array):
        """Perform backward FFT - reuses the builder object"""
        # Copy data into the builder's input array  
        self.backward_fft.input_array[:] = input_array
        return self.backward_fft().copy()

def simulate_original_fft_approach(rho_r, phi_k, E_kx, E_ky, E_kz, virial_data, mesh_volume):
    """
    Simulate your original FFT approach - creates new FFT objects every call
    This represents what happens in your current update() function
    """
    
    # Step 1: Forward FFT on charge density (like your original code)
    fftw_n = fftn(rho_r)
    rho_k_fft = fftw_n()
    rho_k = fftshift(rho_k_fft)
    
    # Step 2: Multiple inverse FFTs (like your original code)
    # Electric field components
    E_kx_unsh = ifftshift(E_kx)
    E_ky_unsh = ifftshift(E_ky)
    E_kz_unsh = ifftshift(E_kz)
    
    ifftw_n = ifftn(E_kx_unsh)
    E_x = ifftw_n()
    ifftw_n = ifftn(E_ky_unsh)
    E_y = ifftw_n()
    ifftw_n = ifftn(E_kz_unsh)
    E_z = ifftw_n()
    
    # Potential
    phi_k_shift = ifftshift(phi_k)
    ifftw_n = ifftn(phi_k_shift)
    phi_r_cmplx = ifftw_n()
    
    # Virial components (6 more IFFTs)
    results = []
    for virial_k in virial_data:
        ifftw_n = ifftn(ifftshift(virial_k))
        virial_r = ifftw_n().real / mesh_volume
        results.append(virial_r)
    
    # Normalize
    E_x_r = E_x.real / mesh_volume
    E_y_r = E_y.real / mesh_volume
    E_z_r = E_z.real / mesh_volume
    phi_r = phi_r_cmplx.real / mesh_volume
    
    return rho_k, E_x_r, E_y_r, E_z_r, phi_r, results

def simulate_optimized_fft_approach(rho_r, phi_k, E_kx, E_ky, E_kz, virial_data, mesh_volume, fft_objects):
    """
    Simulate the optimized FFT approach - reuses FFT objects
    """
    
    mesh_sizes = fft_objects.shape

    # Allocate memory for the output arrays of the FFT
    rho_k = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_x = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_y = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_z = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    phi_r = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xx = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_yy = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_zz = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xy = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xz = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_yz = pyfftw.empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)

    # Step 1: Forward FFT using reusable object
    fft_objects.forward_transform(rho_r, rho_k)
    rho_k = fftshift(rho_k)
    
    # Step 2: Multiple inverse FFTs using the same reusable object
    # Electric field components
    fft_objects.backward_transform(ifftshift(E_kx), E_x)
    fft_objects.backward_transform(ifftshift(E_ky), E_y)
    fft_objects.backward_transform(ifftshift(E_kz), E_z)
    
    # Potential
    fft_objects.backward_transform(ifftshift(phi_k), phi_r)
    
    # Virial components (6 more IFFTs using same object)
    results = []
    for virial_k, virial_r in zip(virial_data, [virial_r_xx, virial_r_yy, virial_r_zz, virial_r_xy, virial_r_xz, virial_r_yz]):
        fft_objects.backward_transform(ifftshift(virial_k), virial_r)
        virial_r = virial_r.real / mesh_volume
        results.append(virial_r)
    
    # Normalize
    E_x_r = E_x.real / mesh_volume
    E_y_r = E_y.real / mesh_volume
    E_z_r = E_z.real / mesh_volume
    phi_r = phi_r.real / mesh_volume
    
    return rho_k, E_x_r, E_y_r, E_z_r, phi_r, results

def generate_test_data(mesh_sizes):
    """Generate realistic test data for the benchmark"""
    shape = (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0])
    
    # Simulate charge density (typically real values from charge assignment)
    rho_r = np.random.random(shape) + 1j * np.random.random(shape) * 0.1
    
    # Simulate potential in k-space (complex)
    phi_k = (np.random.random(shape) + 1j * np.random.random(shape)) * 0.5
    
    # Simulate electric field components in k-space
    E_kx = (np.random.random(shape) + 1j * np.random.random(shape)) * 0.3
    E_ky = (np.random.random(shape) + 1j * np.random.random(shape)) * 0.3
    E_kz = (np.random.random(shape) + 1j * np.random.random(shape)) * 0.3
    
    # Simulate 6 virial tensor components
    virial_data = []
    for i in range(6):
        virial_k = (np.random.random(shape) + 1j * np.random.random(shape)) * 0.2
        virial_data.append(virial_k)
    
    return rho_r, phi_k, E_kx, E_ky, E_kz, virial_data

def run_benchmark(threads=None):
    """Run the complete benchmark comparing original vs optimized approaches"""
    
    # Test different mesh sizes
    test_sizes = [
        np.array([16, 16, 16]),
        np.array([32, 32, 32]), 
        np.array([64, 64, 64]),
        np.array([96, 96, 96]),
        np.array([128, 128, 128]),
        np.array([256, 256, 256])
        # np.array([512, 512, 512]),
    ]
    
    results = []
    n_timesteps = 5  # Simulate multiple timesteps
    
    print("FFT Optimization Benchmark")
    print("="*50)
    print(f"Simulating {n_timesteps} timesteps for each mesh size")
    print(f"Each timestep performs: 1 forward FFT + 10 inverse FFTs")
    print()
    
    for mesh_sizes in test_sizes:
        print(f"Testing mesh size: {mesh_sizes} ({np.prod(mesh_sizes):,} points)")
        
        # Generate test data
        rho_r, phi_k, E_kx, E_ky, E_kz, virial_data = generate_test_data(mesh_sizes)
        mesh_volume = 1.0  # Dummy value
        
        # Create optimized FFT objects ONCE (like in your simulation setup)
        print("  Creating optimized FFT objects...")
        setup_start = time.perf_counter()
        fft_objects = FFTWObjects(mesh_sizes, threads=threads)  # Use the correct class
        setup_time = time.perf_counter() - setup_start
        
        # Benchmark original approach (creates FFT objects every timestep)
        print("  Benchmarking original approach...")
        original_times = []
        for step in range(n_timesteps):
            start_time = time.perf_counter()
            _ = simulate_original_fft_approach(rho_r, phi_k, E_kx, 
                                             E_ky, E_kz, 
                                             virial_data, mesh_volume)
            end_time = time.perf_counter()
            original_times.append(end_time - start_time)
        
        original_avg = np.mean(original_times)
        original_std = np.std(original_times)
        
        # Benchmark optimized approach (reuses FFT objects)
        print("  Benchmarking optimized approach...")
        optimized_times = []
        for step in range(n_timesteps):
            start_time = time.perf_counter()
            _ = simulate_optimized_fft_approach(rho_r, phi_k, E_kx,
                                              E_ky, E_kz,
                                              virial_data, mesh_volume,
                                              fft_objects)
            end_time = time.perf_counter()
            optimized_times.append(end_time - start_time)
        
        optimized_avg = np.mean(optimized_times)
        optimized_std = np.std(optimized_times)
        
        # Calculate speedup
        speedup = original_avg / optimized_avg
        
        # Store results
        result = {
            'mesh_size': tuple(mesh_sizes),
            'total_points': np.prod(mesh_sizes),
            'setup_time': setup_time,
            'original_time': original_avg,
            'original_std': original_std,
            'optimized_time': optimized_avg,
            'optimized_std': optimized_std,
            'speedup': speedup,
            'time_saved_per_step': original_avg - optimized_avg
        }
        results.append(result)
        
        print(f"    Original:  {original_avg:.4f}s ± {original_std:.4f}s")
        print(f"    Optimized: {optimized_avg:.4f}s ± {optimized_std:.4f}s")
        print(f"    Speedup:   {speedup:.1f}x faster")
        print(f"    Time saved per timestep: {(original_avg - optimized_avg)*1000:.1f}ms")
        print()
    
    return results

def create_plots(results):
    """Create visualization of the benchmark results"""
    df = pd.DataFrame(results)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('FFT Optimization Benchmark Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Execution time comparison
    x = range(len(df))
    mesh_labels = [f"{int(row['mesh_size'][0])}³" for _, row in df.iterrows()]
    
    ax1.bar([i-0.2 for i in x], df['original_time'], 0.4, label='Original', 
            color='#ff7f7f', alpha=0.8)
    ax1.bar([i+0.2 for i in x], df['optimized_time'], 0.4, label='Optimized', 
            color='#7f7fff', alpha=0.8)
    ax1.set_xlabel('Mesh Size (points)')
    ax1.set_ylabel('Time per timestep (seconds)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(mesh_labels)
    ax1.legend()
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup factor
    ax2.plot(x, df['speedup'], 'o-', color='#2ca02c', linewidth=3, markersize=8)
    ax2.set_xlabel('Mesh Size (points)')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Performance Improvement (Higher = Better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(mesh_labels)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No improvement')
    
    # Plot 3: Time saved per timestep
    time_saved_ms = df['time_saved_per_step'] * 1000
    ax3.bar(x, time_saved_ms, color='#ff7f0e', alpha=0.8)
    ax3.set_xlabel('Mesh Size (points)')
    ax3.set_ylabel('Time Saved per Timestep (ms)')
    ax3.set_title('Absolute Time Savings')
    ax3.set_xticks(x)
    ax3.set_xticklabels(mesh_labels)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Log-log scaling
    ax4.loglog(df['total_points'], df['original_time'], 'o-', label='Original', 
               color='#ff7f7f', linewidth=2, markersize=8)
    ax4.loglog(df['total_points'], df['optimized_time'], 'o-', label='Optimized', 
               color='#7f7fff', linewidth=2, markersize=8)
    ax4.set_xlabel('Total Grid Points')
    ax4.set_ylabel('Time per timestep (seconds)')
    ax4.set_title('Scaling with Problem Size (Log-Log)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """Print detailed summary of results"""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    print(f"\nAverage speedup across all mesh sizes: {df['speedup'].mean():.1f}x")
    print(f"Maximum speedup achieved: {df['speedup'].max():.1f}x")
    print(f"Total time saved per timestep (largest mesh): {df['time_saved_per_step'].iloc[-1]*1000:.1f}ms")
    
    # Calculate time savings for a full simulation
    print(f"\nFor a 10,000 timestep simulation with 128³ mesh:")
    largest_mesh_savings = df['time_saved_per_step'].iloc[-1] * 10000
    print(f"  Time saved: {largest_mesh_savings/60:.1f} minutes ({largest_mesh_savings/3600:.1f} hours)")
    
    print(f"\nSetup overhead for optimized approach:")
    print(f"  Average FFT object creation time: {df['setup_time'].mean()*1000:.1f}ms")
    print(f"  Break-even point: ~{int(df['setup_time'].mean() / df['time_saved_per_step'].mean())} timesteps")
    
    print("\nDetailed Results:")
    print("-" * 80)
    for _, row in df.iterrows():
        mesh_str = "x".join(str(int(x)) for x in row['mesh_size'])
        print(f"Mesh {mesh_str}:")
        print(f"  Original:     {row['original_time']*1000:.1f}ms ± {row['original_std']*1000:.1f}ms")
        print(f"  Optimized:    {row['optimized_time']*1000:.1f}ms ± {row['optimized_std']*1000:.1f}ms")
        print(f"  Speedup:      {row['speedup']:.1f}x")
        print(f"  Time saved:   {row['time_saved_per_step']*1000:.1f}ms per timestep")
        print()

def debug_fft_differences():
    """Debug function to identify why FFT results don't match"""
    print("FFT Consistency Debug")
    print("=" * 40)
    
    # Create test data
    mesh_sizes = np.array([32, 32, 32])
    shape = (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0])
    rho_m = np.random.random(shape) + 1j * np.random.random(shape)
    
    print(f"Test array shape: {shape}")
    print(f"Test array dtype: {rho_m.dtype}")
    print()
    
    # Method 1: Your original approach
    print("Method 1: fftn builders")
    fftw_n = fftn(rho_m)
    rho_k_fft = fftw_n()
    print(f"  Result dtype: {rho_k_fft.dtype}")
    print(f"  Result shape: {rho_k_fft.shape}")
    print(f"  Max value: {np.max(np.abs(rho_k_fft)):.6f}")
    print(f"  Mean value: {np.mean(np.abs(rho_k_fft)):.6f}")
    print()
    
    # Method 2: Direct FFTW object with correct axes
    print("Method 2: Direct FFTW object with axes=(0,1,2)")
    aligned_input2 = pyfftw.empty_aligned(shape, dtype=complex)
    aligned_output2 = pyfftw.empty_aligned(shape, dtype=complex)
    # aligned_input2[:] = rho_m
    
    # Test with axes=(0,1,2) and no normalization
    fft_correct_axes = pyfftw.FFTW(aligned_input2, aligned_output2, 
                                  direction='FFTW_FORWARD',
                                  axes=(0, 1, 2)
                                  )  # All axes like fftn
    aligned_input2[:] = rho_m  # Copy data into aligned input
    rho_k_fft_2 = fft_correct_axes().copy()
    # rho_k_fft_2 = fft_correct_axes(rho_m).copy()
    
    print(f"  Result dtype: {rho_k_fft_2.dtype}")
    print(f"  Result shape: {rho_k_fft_2.shape}")
    print(f"  Max value: {np.max(np.abs(rho_k_fft_2)):.6f}")
    print(f"  Mean value: {np.mean(np.abs(rho_k_fft_2)):.6f}")
    print()
    
    # Method 3: More direct approach - match exactly
    print("Method 3: Default FFTW object with aligned arrays")
    
    # Create FFTW object with identical parameters to builders
    aligned_input = pyfftw.empty_aligned(shape, dtype=complex)
    aligned_output = pyfftw.empty_aligned(shape, dtype=complex)
    
    # Try to match fftn exactly
    fft_exact = pyfftw.FFTW(aligned_input, aligned_output, 
                           direction='FFTW_FORWARD')           # fftn default
    aligned_input[:] = rho_m
    rho_k_fft_3 = fft_exact().copy()
    
    print(f"  Result dtype: {rho_k_fft_3.dtype}")
    print(f"  Result shape: {rho_k_fft_3.shape}")
    print(f"  Max value: {np.max(np.abs(rho_k_fft_3)):.6f}")
    print(f"  Mean value: {np.mean(np.abs(rho_k_fft_3)):.6f}")
    print()
    
    # Compare all methods
    print("Consistency checks:")
    print(f"  Method 1 vs Method 2: {np.allclose(rho_k_fft, rho_k_fft_2, rtol=1e-12)}")
    print(f"  Method 1 vs Method 3: {np.allclose(rho_k_fft, rho_k_fft_3, rtol=1e-12)}")
    print(f"  Method 2 vs Method 3: {np.allclose(rho_k_fft_2, rho_k_fft_3, rtol=1e-12)}")
    
    if not np.allclose(rho_k_fft, rho_k_fft_2, rtol=1e-12):
        diff = rho_k_fft - rho_k_fft_2
        print(f"  Max difference (1 vs 2): {np.max(np.abs(diff)):.2e}")
        print(f"  Relative error: {np.max(np.abs(diff))/np.max(np.abs(rho_k_fft)):.2e}")
    
    if not np.allclose(rho_k_fft, rho_k_fft_3, rtol=1e-12):
        diff = rho_k_fft - rho_k_fft_3
        print(f"  Max difference (1 vs 3): {np.max(np.abs(diff)):.2e}")
        print(f"  Relative error: {np.max(np.abs(diff))/np.max(np.abs(rho_k_fft)):.2e}")
    
    # Method 4: Let's try using the same planning as builders internally
    print("\nMethod 4: Using builders approach internally")
    
    correct_fft = CorrectFFTWObjects(mesh_sizes)
    rho_k_fft_4 = correct_fft.forward_transform(rho_m)
    
    print(f"  Method 1 vs Method 4: {np.allclose(rho_k_fft, rho_k_fft_4, rtol=1e-14)}")
    
    return rho_k_fft, rho_k_fft_2, rho_k_fft_3, rho_k_fft_4

def main():
    """Main function to run the complete benchmark"""
    if not PYFFTW_AVAILABLE:
        print("This benchmark requires pyfftw. Install with: pip install pyfftw")
        return
    

    # Use argparse to allow user to specify number of threads for FFTW
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark FFT optimization in Sarkas")
    parser.add_argument('--threads', type=int, default=None, help='Number of threads to use for FFTW (default: all available)')
    args = parser.parse_args()



    # First, debug the FFT consistency issue
    print("Debugging FFT consistency before running benchmark...\n")
    debug_fft_differences()
    
    print("\n" + "="*60)
    print("Starting full benchmark...")
    print("="*60)
    
    # Run benchmark
    results = run_benchmark(threads=args.threads)
    
    # Create visualizations
    create_plots(results)
    
    # Print summary
    print_summary(results)

if __name__ == "__main__":
    main()