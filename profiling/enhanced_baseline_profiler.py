"""
Enhanced baseline performance profiler for Sarkas physics calculations.
Run this BEFORE refactoring to establish comprehensive performance targets.
"""

import time
import numpy as np
import cProfile
import pstats
import os
import psutil
import gc
from pathlib import Path
from datetime import datetime
import json
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import current Sarkas
from sarkas.processes import PreProcess
from sarkas.utilities.timing import SarkasTimer

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class ProfileResult:
    """Structured profiling result."""
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    memory_peak: float
    memory_delta: float
    allocations: int
    iterations: int
    
    def to_dict(self):
        return asdict(self)

@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    cpu_usage_start: float
    cpu_usage_end: float
    memory_available_start: float
    memory_available_end: float
    gc_collections_start: Dict[int, int]
    gc_collections_end: Dict[int, int]

class EnhancedBaselineProfiler:
    """Enhanced profiler with comprehensive metrics and visualization."""
    
    def __init__(self, test_particle_count: int = 10_000, output_dir: str = "profiling_results"):
        self.test_particle_count = test_particle_count
        self.output_dir = Path(output_dir)
        self.results = {}
        self.system_metrics = {}
        self.timer = SarkasTimer()
        
        # Create output directory structure
        self.create_output_directories()
        
        # Physics methods to profile
        self.physics_methods = [
            'calculate_kinetic_energy',
            'calculate_species_kinetic_temperature', 
            # 'calculate_species_momentum',
            # 'calculate_center_of_mass_velocity',
            # 'remove_center_of_mass_motion',
            # 'calculate_species_electric_current',
            # 'calculate_species_pressure_tensor',
            # 'calculate_species_heat_flux',
            # 'calculate_species_diffusion_flux',
            # 'calculate_species_velocity_moments'
        ]
        
        print(f"Enhanced Baseline Profiler initialized")
        print(f"Test system: {self.test_particle_count:,} particles")
        print(f"Output directory: {self.output_dir}")
        
    def create_output_directories(self):
        """Create organized output directory structure."""
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'data': self.output_dir / 'data',
            'plots': self.output_dir / 'plots', 
            'profiles': self.output_dir / 'detailed_profiles',
            'reports': self.output_dir / 'reports'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
            
        print(f"Created output directories in: {self.output_dir}")
    
    def setup_test_system(self, N: int):
        """Create standardized test particle system."""
        input_file_name = os.path.join('input_files', 'BIM_cgs.yaml')
        
        args = {
            'Particles': [
                {"Species": {
                    'name': 'H',
                    'number_density': 8.1e+27,
                    'atomic_weight': 1.0,           
                    'Z': 1.0,
                    'temperature_eV': 14.68,
                    'num': N//2,
                    'replace': True
                }},
                {"Species": {
                    'name': 'He',
                    'number_density': 8.1e+27,
                    'atomic_weight': 4,
                    'Z': 2.0,  
                    'temperature_eV': 14.68,
                    'num': N//2,
                    'replace': True
                }}
            ],
            "IO": {
                'verbose': False,
                'job_dir': f'profiling_N{N}'
            }
        }
        
        pre = PreProcess(input_file_name)
        pre.setup(read_yaml=True, other_inputs=args)
        
        return pre.particles
    
    @contextmanager
    def memory_tracker(self):
        """Context manager for tracking memory usage."""
        tracemalloc.start()
        process = psutil.Process()
        
        # Initial measurements
        gc.collect()  # Clean slate
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield initial_memory
        finally:
            # Final measurements
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            self.last_memory_delta = final_memory - initial_memory
            self.last_memory_peak = peak / 1024 / 1024  # MB
    
    def get_system_metrics(self) -> SystemMetrics:
        """Capture system-level performance metrics."""
        return SystemMetrics(
            cpu_usage_start=psutil.cpu_percent(),
            cpu_usage_end=0,  # Will be updated
            memory_available_start=psutil.virtual_memory().available / 1024 / 1024,
            memory_available_end=0,  # Will be updated
            gc_collections_start={i: gc.get_count()[i] for i in range(3)},
            gc_collections_end={i: 0 for i in range(3)}  # Will be updated
        )
    
    def profile_method_enhanced(self, particles, method_name: str, iterations: int = 21) -> ProfileResult:
        """Enhanced method profiling with memory tracking."""
        if not hasattr(particles, method_name):
            raise AttributeError(f"Method {method_name} not found")
            
        method = getattr(particles, method_name)
        
        # System metrics before
        sys_metrics = self.get_system_metrics()
        
        times = []
        memory_deltas = []
        
        # Warm up runs
        for _ in range(3):
            try:
                method()
            except:
                # Try with use_fast parameter if method supports it
                try:
                    method(use_fast=True)
                except:
                    method()
        
        # Timed runs with memory tracking
        for i in range(iterations):
            gc.collect()  # Clean start
            
            with self.memory_tracker():
                start_time = time.perf_counter_ns()
                
                try:
                    method()
                except:
                    try:
                        method(use_fast=True)
                    except:
                        method()
                        
                end_time = time.perf_counter_ns()
            
            times.append((end_time - start_time) / 1e6)  # Convert to milliseconds
            memory_deltas.append(self.last_memory_delta)
        
        # System metrics after
        sys_metrics.cpu_usage_end = psutil.cpu_percent()
        sys_metrics.memory_available_end = psutil.virtual_memory().available / 1024 / 1024
        sys_metrics.gc_collections_end = {i: gc.get_count()[i] for i in range(3)}
        
        self.system_metrics[method_name] = sys_metrics
        
        # Calculate statistics (skip first few iterations for warm-up)
        clean_times = times[3:]
        clean_memory = memory_deltas[3:]
        
        return ProfileResult(
            mean_time=np.mean(clean_times),
            std_time=np.std(clean_times),
            min_time=np.min(clean_times),
            max_time=np.max(clean_times),
            memory_peak=self.last_memory_peak,
            memory_delta=np.mean(clean_memory),
            allocations=sum(sys_metrics.gc_collections_end[i] - sys_metrics.gc_collections_start[i] 
                          for i in range(3)),
            iterations=len(clean_times)
        )
    
    def profile_all_physics_methods(self):
        """Profile all physics calculation methods with enhanced metrics."""
        print(f"\n=== ENHANCED BASELINE PROFILING ({self.test_particle_count:,} particles) ===\n")
        
        # Setup test system
        particles = self.setup_test_system(self.test_particle_count)
        
        self.results = {}
        successful_methods = []
        failed_methods = []
        
        for method_name in self.physics_methods:
            print(f"Profiling {method_name}...", end=' ')
            
            try:
                result = self.profile_method_enhanced(particles, method_name)
                self.results[method_name] = result
                successful_methods.append(method_name)
                
                print(f"✓ {result.mean_time:.3f} ± {result.std_time:.3f} ms")
                
            except Exception as e:
                self.results[method_name] = {'error': str(e)}
                failed_methods.append((method_name, str(e)))
                print(f"✗ ERROR: {e}")
        
        print(f"\nProfiling Summary:")
        print(f"  Successful: {len(successful_methods)}")
        print(f"  Failed: {len(failed_methods)}")
        
        if failed_methods:
            print(f"\nFailed methods:")
            for method, error in failed_methods:
                print(f"  - {method}: {error}")
    
    def detailed_cprofile_analysis(self, method_name: str, runs: int = 50):
        """Detailed cProfile analysis of specific method."""
        particles = self.setup_test_system(self.test_particle_count)
        
        if not hasattr(particles, method_name):
            print(f"Method {method_name} not found, skipping detailed profile")
            return
            
        method = getattr(particles, method_name)
        
        print(f"\nDetailed profiling: {method_name} ({runs} runs)")
        
        # Profile with cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        for _ in range(runs):
            try:
                method()
            except:
                try:
                    method(use_fast=True)
                except:
                    method()
        
        profiler.disable()
        
        # Save and analyze
        profile_file = self.dirs['profiles'] / f'detailed_{method_name}.prof'
        profiler.dump_stats(str(profile_file))
        
        # Generate text report
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        
        report_file = self.dirs['reports'] / f'detailed_{method_name}_report.txt'
        with open(report_file, 'w') as f:
            stats.print_stats(file=f)
        
        print(f"  Detailed profile saved: {profile_file}")
        print(f"  Report saved: {report_file}")
    
    def create_performance_visualizations(self):
        """Create comprehensive performance visualization suite."""
        if not self.results:
            print("No results to visualize")
            return
        
        # Filter successful results
        successful_results = {k: v for k, v in self.results.items() 
                            if isinstance(v, ProfileResult)}
        
        if not successful_results:
            print("No successful results to visualize")
            return
        
        # 1. Performance Overview Dashboard
        self.create_performance_dashboard(successful_results)
        
        # 2. Memory Usage Analysis  
        self.create_memory_analysis(successful_results)
        
        # 3. Detailed Method Comparison
        self.create_method_comparison(successful_results)
        
        # 4. System Resource Usage
        self.create_system_metrics_plot()
        
        print(f"All visualizations saved to: {self.dirs['plots']}")
    
    def create_performance_dashboard(self, results: Dict[str, ProfileResult]):
        """Create main performance dashboard."""
        methods = list(results.keys())
        times = [results[m].mean_time for m in methods]
        errors = [results[m].std_time for m in methods]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sarkas Baseline Performance Dashboard ({self.test_particle_count:,} particles)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Execution times with error bars
        bars = ax1.barh(methods, times, xerr=errors, capsize=5, alpha=0.7)
        ax1.set_xlabel('Execution Time (ms)')
        ax1.set_title('Method Execution Times')
        ax1.grid(True, alpha=0.3)
        
        # Color bars by performance
        times_array = np.array(times)
        colors = plt.cm.RdYlBu_r(times_array / times_array.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 2. Memory usage
        memory_peaks = [results[m].memory_peak for m in methods]
        ax2.bar(range(len(methods)), memory_peaks, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Memory Usage per Method')
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels([m.replace('calculate_', '').replace('species_', '') 
                            for m in methods], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance variability (coefficient of variation)
        cv = [results[m].std_time / results[m].mean_time * 100 for m in methods]
        ax3.scatter(times, cv, alpha=0.7, s=100)
        ax3.set_xlabel('Mean Time (ms)')
        ax3.set_ylabel('Coefficient of Variation (%)')
        ax3.set_title('Performance Consistency')
        ax3.grid(True, alpha=0.3)
        
        # Add method labels to scatter plot
        for i, method in enumerate(methods):
            ax3.annotate(method.replace('calculate_', '').replace('species_', ''), 
                        (times[i], cv[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
        
        # 4. Performance vs Memory trade-off
        ax4.scatter(times, memory_peaks, alpha=0.7, s=100, c=cv, cmap='viridis')
        ax4.set_xlabel('Execution Time (ms)')
        ax4.set_ylabel('Peak Memory (MB)')
        ax4.set_title('Time vs Memory Trade-off')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for consistency metric
        cbar = plt.colorbar(ax4.collections[0], ax=ax4)
        cbar.set_label('Variability (%)')
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_memory_analysis(self, results: Dict[str, ProfileResult]):
        """Create detailed memory usage analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        methods = list(results.keys())
        memory_peaks = [results[m].memory_peak for m in methods]
        memory_deltas = [results[m].memory_delta for m in methods]
        
        # 1. Memory peaks
        ax1.barh(methods, memory_peaks, alpha=0.7, color='skyblue')
        ax1.set_xlabel('Peak Memory Usage (MB)')
        ax1.set_title('Peak Memory per Method')
        ax1.grid(True, alpha=0.3)
        
        # 2. Memory allocation patterns
        ax2.barh(methods, memory_deltas, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Memory Delta (MB)')
        ax2.set_title('Memory Allocation per Call')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'memory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_method_comparison(self, results: Dict[str, ProfileResult]):
        """Create detailed method comparison plots."""
        # Create DataFrame for easier plotting
        data = []
        for method, result in results.items():
            data.append({
                'Method': method.replace('calculate_', '').replace('species_', ''),
                'Mean Time (ms)': result.mean_time,
                'Std Time (ms)': result.std_time,
                'Min Time (ms)': result.min_time,
                'Max Time (ms)': result.max_time,
                'Memory Peak (MB)': result.memory_peak,
                'Memory Delta (MB)': result.memory_delta
            })
        
        df = pd.DataFrame(data)
        
        # Box plot of timing distribution
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create box plot data
        box_data = []
        labels = []
        for method, result in results.items():
            # Simulate distribution from mean and std
            simulated_times = np.random.normal(result.mean_time, result.std_time, 1000)
            simulated_times = np.clip(simulated_times, result.min_time, result.max_time)
            box_data.append(simulated_times)
            labels.append(method.replace('calculate_', '').replace('species_', ''))
        
        bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Execution Time (ms)')
        ax.set_title('Method Performance Distribution')
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_system_metrics_plot(self):
        """Create system resource usage visualization."""
        if not self.system_metrics:
            return
        
        methods = list(self.system_metrics.keys())
        cpu_usage = [self.system_metrics[m].cpu_usage_end - self.system_metrics[m].cpu_usage_start 
                    for m in methods]
        memory_change = [self.system_metrics[m].memory_available_start - self.system_metrics[m].memory_available_end 
                        for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CPU usage change
        ax1.barh(methods, cpu_usage, alpha=0.7, color='orange')
        ax1.set_xlabel('CPU Usage Change (%)')
        ax1.set_title('CPU Impact per Method')
        ax1.grid(True, alpha=0.3)
        
        # Memory usage change
        ax2.barh(methods, memory_change, alpha=0.7, color='purple')
        ax2.set_xlabel('Memory Usage Change (MB)')
        ax2.set_title('System Memory Impact')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'system_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comprehensive_results(self):
        """Save all results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. JSON format for easy loading
        json_results = {
            'metadata': {
                'timestamp': timestamp,
                'particle_count': self.test_particle_count,
                'python_version': os.sys.version,
                'numpy_version': np.__version__
            },
            'results': {}
        }
        
        for method, result in self.results.items():
            if isinstance(result, ProfileResult):
                json_results['results'][method] = result.to_dict()
            else:
                json_results['results'][method] = result
        
        json_file = self.dirs['data'] / f'baseline_results_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # 2. CSV format for spreadsheet analysis
        if any(isinstance(r, ProfileResult) for r in self.results.values()):
            csv_data = []
            for method, result in self.results.items():
                if isinstance(result, ProfileResult):
                    row = {'method': method}
                    row.update(result.to_dict())
                    csv_data.append(row)
            
            df = pd.DataFrame(csv_data)
            csv_file = self.dirs['data'] / f'baseline_results_{timestamp}.csv'
            df.to_csv(csv_file, index=False)
        
        # 3. Human-readable summary report
        self.create_summary_report(timestamp)
        
        print(f"\nResults saved:")
        print(f"  JSON: {json_file}")
        if 'csv_file' in locals():
            print(f"  CSV: {csv_file}")
        print(f"  Summary: {self.dirs['reports'] / f'baseline_summary_{timestamp}.txt'}")
    
    def create_summary_report(self, timestamp: str):
        """Create human-readable summary report."""
        report_file = self.dirs['reports'] / f'baseline_summary_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SARKAS BASELINE PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test System: {self.test_particle_count:,} particles\n")
            f.write(f"Python Version: {os.sys.version}\n")
            f.write(f"NumPy Version: {np.__version__}\n\n")
            
            # Performance summary
            successful_results = {k: v for k, v in self.results.items() 
                                if isinstance(v, ProfileResult)}
            
            if successful_results:
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-" * 50 + "\n")
                
                # Sort by execution time
                sorted_methods = sorted(successful_results.items(), 
                                      key=lambda x: x[1].mean_time, reverse=True)
                
                f.write(f"{'Method':<40} {'Time (ms)':<15} {'Memory (MB)':<15}\n")
                f.write("-" * 70 + "\n")
                
                for method, result in sorted_methods:
                    f.write(f"{method:<40} {result.mean_time:<15.3f} {result.memory_peak:<15.2f}\n")
                
                # Hotspots identification
                f.write(f"\nHOTSPOTS (Top 3 slowest methods):\n")
                f.write("-" * 30 + "\n")
                for i, (method, result) in enumerate(sorted_methods[:3]):
                    f.write(f"{i+1}. {method}: {result.mean_time:.3f} ms "
                           f"(±{result.std_time:.3f} ms)\n")
                
                # Memory analysis
                f.write(f"\nMEMORY USAGE (Top 3 memory consumers):\n")
                f.write("-" * 30 + "\n")
                memory_sorted = sorted(successful_results.items(), 
                                     key=lambda x: x[1].memory_peak, reverse=True)
                for i, (method, result) in enumerate(memory_sorted[:3]):
                    f.write(f"{i+1}. {method}: {result.memory_peak:.2f} MB peak\n")
            
            # Failed methods
            failed_methods = {k: v for k, v in self.results.items() 
                            if not isinstance(v, ProfileResult)}
            
            if failed_methods:
                f.write(f"\nFAILED METHODS\n")
                f.write("-" * 20 + "\n")
                for method, error in failed_methods.items():
                    f.write(f"{method}: {error}\n")
    
    def run_complete_baseline(self):
        """Run the complete enhanced baseline profiling suite."""
        print("Starting Enhanced Baseline Profiling Suite")
        print("=" * 60)
        
        start_time = time.time()
        
        # 1. Profile all physics methods
        self.profile_all_physics_methods()
        
        # 2. Detailed profiling of key methods
        key_methods = ['calculate_kinetic_energy', 'calculate_species_kinetic_temperature']
        for method in key_methods:
            if method in self.results and isinstance(self.results[method], ProfileResult):
                self.detailed_cprofile_analysis(method)
        
        # 3. Create visualizations
        self.create_performance_visualizations()
        
        # 4. Save all results
        self.save_comprehensive_results()
        
        total_time = time.time() - start_time
        
        print(f"\n" + "="*60)
        print("ENHANCED BASELINE PROFILING COMPLETE")
        print("="*60)
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Output directory: {self.output_dir}")
        print(f"Methods profiled: {len([r for r in self.results.values() if isinstance(r, ProfileResult)])}")
        print(f"Visualizations created: {len(list(self.dirs['plots'].glob('*.png')))}")

def main():
    """Main execution function."""
    profiler = EnhancedBaselineProfiler(
        test_particle_count=10_000,
        output_dir="sarkas_profiling_baseline"
    )
    
    profiler.run_complete_baseline()

if __name__ == "__main__":
    main()