"""
Baseline performance profiler for Sarkas physics calculations.
Run this BEFORE refactoring to establish performance targets.
"""

import time
import numpy as np
import cProfile
import pstats
import os
from memory_profiler import profile
import matplotlib.pyplot as plt

# Import current Sarkas
from sarkas.processes import PreProcess
# from sarkas.tools.observables import RadialDistributionFunction, Thermodynamics, VelocityAutoCorrelationFunction

class BaselineProfiler:
    """Profile current Sarkas performance before refactoring."""
    
    def __init__(self, particle_counts=[1_000, 5_000, 10_000, 50_000, 100_000]):
        self.particle_counts = particle_counts
        self.results = {}
        
    def setup_test_system(self, N):
        """Create test particle system."""
        
        # Create the file path to the YAML input file
        input_file_name = os.path.join('input_files', 'BIM_cgs.yaml')
        args = {'Particles': [
                {"Species" :{
                    'name': 'H',
                    'number_density': 8.1e+27,
                    'atomic_weight': 1.0,           
                    'Z': 1.0,
                    'temperature_eV': 14.68,
                    'num': N//2,
                    'replace': True
                    }
                },
                {"Species" :{
                    'name' :'He',
                    'number_density': 8.1e+27,
                    'atomic_weight': 4,
                    'Z': 2.0,
                    'temperature_eV': 14.68,
                    'num': N//2,
                    'replace': True
                    }
                },
            ],
            "IO" : {
                     'verbose': False,
                    'job_dir': f'N{N}'
                        }
            }
        pre = PreProcess(input_file_name)
        pre.setup(read_yaml=True, other_inputs=args)
        
        return pre.particles
    
    def profile_method(self, particles, method_name, iterations=11):
        """Profile a specific Particles method."""
        method = getattr(particles, method_name)
        
        # Warm up
        method()
        
        # Time multiple iterations
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            method(use_fast = True)
            times.append(time.perf_counter() - start)
            
        return {
            'mean_time': np.mean(times[1:]),  # Skip first warm-up iteration
            'std_time': np.std(times[1:]),
            'min_time': np.min(times[1:]),
            'max_time': np.max(times[1:])
        }
    
    def profile_all_physics_methods(self):
        """Profile all physics calculation methods."""
        
        methods_to_profile = [
            'calculate_kinetic_energy',
            'calculate_species_kinetic_temperature', 
            'calculate_species_momentum',
            'calculate_species_electric_current',
            'calculate_species_pressure_tensor',
            # Add other methods you want to profile
        ]
        
        print("=== BASELINE PERFORMANCE PROFILING ===\n")
        
        for N in self.particle_counts:
            print(f"Profiling {N} particles:")
            particles = self.setup_test_system(N)
            
            self.results[N] = {}
            
            for method in methods_to_profile:
                if hasattr(particles, method):
                    try:
                        result = self.profile_method(particles, method)
                        self.results[N][method] = result
                        print(f"  {method}: {result['mean_time']:.4e} ± {result['std_time']:.4e} mu s")
                    except Exception as e:
                        print(f"  {method}: ERROR - {e}")
                        self.results[N][method] = {'error': str(e)}
                else:
                    print(f"  {method}: NOT FOUND")
            print()
    
    @profile  # Memory profiler decorator
    def memory_profile_large_system(self, N=100_000):
        """Profile memory usage for large system."""
        print(f"Memory profiling with {N} particles...")
        particles = self.setup_test_system(N)
        
        # Profile memory for each calculation
        particles.calculate_kinetic_energy(use_fast=True)
        particles.calculate_species_kinetic_temperature(use_fast=True)
        particles.calculate_species_momentum(use_fast=True)
        
        return particles
    
    def detailed_profile_hottest_method(self, method_name='calculate_species_kinetic_temperature', N=100_000):
        """Detailed cProfile analysis of hottest method."""
        particles = self.setup_test_system(N)
        method = getattr(particles, method_name)
        
        print(f"\n=== DETAILED PROFILE: {method_name} ===")
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run method multiple times
        for _ in range(100):
            method()
            
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        # Save detailed profile
        stats.dump_stats(f'baseline_profile_{method_name}.prof')
        print(f"Detailed profile saved to: baseline_profile_{method_name}.prof")
    
    def save_baseline_results(self):
        """Save baseline results for comparison after refactoring."""
        import json
        
        # Convert numpy types to JSON-serializable
        json_results = {}
        for N, methods in self.results.items():
            json_results[str(N)] = {}
            for method, result in methods.items():
                if 'error' not in result:
                    json_results[str(N)][method] = {
                        'mean_time': float(result['mean_time']),
                        'std_time': float(result['std_time']),
                        'min_time': float(result['min_time']),
                        'max_time': float(result['max_time'])
                    }
                else:
                    json_results[str(N)][method] = result
        
        with open('baseline_performance.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print("Baseline results saved to: baseline_performance.json")
    
    def plot_baseline_scaling(self):
        """Plot how performance scales with particle count."""
        methods = ['calculate_kinetic_energy', 'calculate_species_kinetic_temperature']
        
        plt.figure(figsize=(12, 6))
        
        for i, method in enumerate(methods):
            plt.subplot(1, 2, i+1)
            
            N_values = []
            times = []
            errors = []
            
            for N in self.particle_counts:
                if N in self.results and method in self.results[N]:
                    result = self.results[N][method]
                    if 'error' not in result:
                        N_values.append(N)
                        times.append(result['mean_time'] )  
                        errors.append(result['std_time'] )
            
            if N_values:
                plt.errorbar(N_values, times, yerr=errors, capsize=5)
                plt.xlabel('Number of Particles')
                plt.ylabel('Time (mu s)')
                plt.title(f'Baseline: {method}')
                plt.grid(True, alpha=0.3)
                plt.loglog()  # Log-log scale to see scaling
        
        plt.tight_layout()
        plt.savefig('baseline_performance_scaling.png', dpi=150)
        print("Performance scaling plot saved to: baseline_performance_scaling.png")

def run_complete_baseline():
    """Run complete baseline profiling suite."""
    profiler = BaselineProfiler()
    
    print("Starting comprehensive baseline profiling...")
    print("This will take several minutes to complete.\n")
    
    # 1. Profile all methods across different system sizes
    profiler.profile_all_physics_methods()
    
    # 2. Memory profiling
    profiler.memory_profile_large_system()
    
    # 3. Detailed profiling of hottest methods
    profiler.detailed_profile_hottest_method('calculate_species_kinetic_temperature')
    profiler.detailed_profile_hottest_method('calculate_kinetic_energy')
    
    # 4. Save results and plots
    profiler.save_baseline_results()
    profiler.plot_baseline_scaling()
    
    print("\n=== BASELINE PROFILING COMPLETE ===")
    print("Files created:")
    print("  - baseline_performance.json")
    print("  - baseline_performance_scaling.png") 
    print("  - baseline_profile_*.prof")
    print("  - memory profiling output")

if __name__ == "__main__":
    run_complete_baseline()