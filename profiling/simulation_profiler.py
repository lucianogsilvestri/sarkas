"""
Complete Sarkas simulation profiler for full workflow performance analysis.
Profiles the entire simulation lifecycle: setup, equilibration, production, and I/O.
"""

import time
import numpy as np
import os
import psutil
import gc
import cProfile
import pstats
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import json
import tracemalloc
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import Sarkas
from sarkas.processes import Simulation
from sarkas.utilities.timing import SarkasTimer

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class SimulationPhaseMetrics:
    """Metrics for a simulation phase."""
    phase_name: str
    duration_seconds: float
    peak_memory_mb: float
    memory_delta_mb: float
    cpu_usage_percent: float
    steps_completed: int
    steps_per_second: float
    io_operations: int
    gc_collections: Dict[int, int]
    
    def to_dict(self):
        return asdict(self)

@dataclass
class SimulationProfileResult:
    """Complete simulation profiling result."""
    total_duration: float
    peak_memory_usage: float
    total_memory_allocated: float
    phases: Dict[str, SimulationPhaseMetrics]
    system_info: Dict[str, Any]
    
    def to_dict(self):
        result = asdict(self)
        # Convert phase metrics to dict
        result['phases'] = {k: v.to_dict() for k, v in self.phases.items()}
        return result

class SimulationProfiler:
    """Comprehensive profiler for complete Sarkas simulations."""
    
    def __init__(self, particle_count: int = 10_000, 
                 equilibration_steps: int = 1000,
                 production_steps: int = 2000,
                 output_dir: str = "simulation_profiling"):
        
        self.particle_count = particle_count
        self.equilibration_steps = equilibration_steps
        self.production_steps = production_steps
        self.output_dir = Path(output_dir)
        
        # Profiling data storage
        self.phase_metrics = {}
        self.timeline_data = []
        self.memory_timeline = []
        self.timer = SarkasTimer()
        
        # System monitoring
        self.process = psutil.Process()
        self.initial_memory = 0
        
        # Create output directories
        self.create_output_directories()
        
        print(f"Simulation Profiler initialized")
        print(f"Test system: {self.particle_count:,} particles")
        print(f"Equilibration steps: {self.equilibration_steps:,}")
        print(f"Production steps: {self.production_steps:,}")
        print(f"Output directory: {self.output_dir}")
    
    def create_output_directories(self):
        """Create organized output directory structure."""
        self.output_dir.mkdir(exist_ok=True)
        
        self.dirs = {
            'data': self.output_dir / 'data',
            'plots': self.output_dir / 'plots',
            'profiles': self.output_dir / 'detailed_profiles',
            'reports': self.output_dir / 'reports',
            'simulation_output': self.output_dir / 'simulation_files'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
    
    @contextmanager
    def phase_profiler(self, phase_name: str, expected_steps: int = 0):
        """Context manager for profiling simulation phases."""
        print(f"\n📊 Starting profiling: {phase_name}")
        
        # Initial measurements
        gc.collect()
        tracemalloc.start()
        
        start_time = time.perf_counter()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        gc_start = {i: gc.get_count()[i] for i in range(3)}
        
        try:
            yield
        except Exception as e:
            print(f"❌ Error in {phase_name}: {e}")
            raise
        finally:
            # Final measurements
            end_time = time.perf_counter()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            gc_end = {i: gc.get_count()[i] for i in range(3)}
            
            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            peak_memory_mb = peak_mem / 1024 / 1024  # MB
            cpu_usage = max(end_cpu - start_cpu, 0)
            
            gc_collections = {i: gc_end[i] - gc_start[i] for i in range(3)}
            
            # Calculate performance metrics
            steps_per_second = expected_steps / duration if duration > 0 and expected_steps > 0 else 0
            
            # Store phase metrics
            self.phase_metrics[phase_name] = SimulationPhaseMetrics(
                phase_name=phase_name,
                duration_seconds=duration,
                peak_memory_mb=peak_memory_mb,
                memory_delta_mb=memory_delta,
                cpu_usage_percent=cpu_usage,
                steps_completed=expected_steps,
                steps_per_second=steps_per_second,
                io_operations=0,  # Could be enhanced to track I/O
                gc_collections=gc_collections
            )
            
            # Add to timeline
            self.timeline_data.append({
                'phase': phase_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'memory_start': start_memory,
                'memory_end': end_memory,
                'memory_peak': peak_memory_mb
            })
            
            print(f"✅ {phase_name} completed:")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Peak memory: {peak_memory_mb:.1f} MB")
            if expected_steps > 0:
                print(f"   Performance: {steps_per_second:.1f} steps/second")
    
    def setup_simulation_config(self):
        """Create simulation configuration with profiling parameters."""
        input_file_name = os.path.join('input_files', 'BIM_cgs.yaml')
        
        # Ensure simulation output goes to our directory
        sim_output_dir = str(self.dirs['simulation_output'])
        
        args = {
            'Particles': [
                {"Species": {
                    'name': 'H',
                    'number_density': 8.1e+27,
                    'atomic_weight': 1.0,           
                    'Z': 1.0,
                    'temperature_eV': 14.68,
                    'num': self.particle_count // 2,
                    'replace': True
                }},
                {"Species": {
                    'name': 'He',
                    'number_density': 8.1e+27,
                    'atomic_weight': 4,
                    'Z': 2.0,
                    'temperature_eV': 14.68,
                    'num': self.particle_count // 2,
                    'replace': True
                }}
            ],
            'Parameters': {
                'equilibration_steps': self.equilibration_steps,
                'production_steps': self.production_steps,
                'eq_dump_step': max(1, self.equilibration_steps // 10),  # 10 dumps during eq
                'prod_dump_step': max(1, self.production_steps // 20),   # 20 dumps during prod
            },
            "IO": {
                'verbose': True,
                'job_dir': f'profiling_N{self.particle_count}',
                'job_id': f'prof_{datetime.now().strftime("%H%M%S")}',
            }
        }
        
        return input_file_name, args
    
    def profile_complete_simulation(self, detailed_profiling: bool = True):
        """Profile a complete simulation with all phases."""
        print("="*80)
        print("COMPLETE SARKAS SIMULATION PROFILING")
        print("="*80)
        
        # Overall timing
        total_start_time = time.perf_counter()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Get system information
        system_info = self.get_system_info()
        
        try:
            # Phase 1: Setup and Initialization
            with self.phase_profiler("Initialization", 0):
                input_file, args = self.setup_simulation_config()
                sim = Simulation(input_file)
                sim.setup(read_yaml=True, other_inputs=args)
            
            # Phase 2: Equilibration (if any steps)
            if self.equilibration_steps > 0:
                if detailed_profiling:
                    # Detailed profiling of equilibration
                    with self.phase_profiler("Equilibration_Detailed", self.equilibration_steps):
                        self.detailed_phase_profiling(sim, "equilibration")
                else:
                    with self.phase_profiler("Equilibration", self.equilibration_steps):
                        # Run equilibration phase
                        self.run_simulation_phase(sim, "equilibration")
            
            # Phase 3: Production
            if detailed_profiling:
                with self.phase_profiler("Production_Detailed", self.production_steps):
                    self.detailed_phase_profiling(sim, "production")
            else:
                with self.phase_profiler("Production", self.production_steps):
                    self.run_simulation_phase(sim, "production")
            
            # Phase 4: Finalization and I/O
            with self.phase_profiler("Finalization", 0):
                # Any final I/O operations, cleanup, etc.
                sim.timer.stop()
                final_memory = self.process.memory_info().rss / 1024 / 1024
                
        except Exception as e:
            print(f"❌ Simulation failed: {e}")
            raise
        
        # Calculate total metrics
        total_duration = time.perf_counter() - total_start_time
        peak_memory = max([phase.peak_memory_mb for phase in self.phase_metrics.values()])
        total_memory_allocated = final_memory - self.initial_memory
        
        # Create complete result
        self.profile_result = SimulationProfileResult(
            total_duration=total_duration,
            peak_memory_usage=peak_memory,
            total_memory_allocated=total_memory_allocated,
            phases=self.phase_metrics,
            system_info=system_info
        )
        
        print(f"\n🎯 SIMULATION PROFILING COMPLETE")
        print(f"   Total time: {total_duration:.2f} seconds")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Memory allocated: {total_memory_allocated:.1f} MB")
        
        return self.profile_result
    
    def run_simulation_phase(self, sim, phase: str):
        """Run a simulation phase with basic monitoring."""
        if phase == "equilibration":
            # Check if equilibration is needed
            if hasattr(sim, 'equilibrate') and self.equilibration_steps > 0:
                sim.equilibrate()
        elif phase == "production":
            # Run production phase
            sim.produce()
    
    def detailed_phase_profiling(self, sim, phase: str):
        """Run detailed profiling during simulation phases."""
        # Create detailed profiler
        profiler = cProfile.Profile()
        
        # Start detailed profiling
        profiler.enable()
        
        try:
            self.run_simulation_phase(sim, phase)
        finally:
            profiler.disable()
            
            # Save detailed profile
            profile_file = self.dirs['profiles'] / f'detailed_{phase}_profile.prof'
            profiler.dump_stats(str(profile_file))
            
            # Generate report
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            report_file = self.dirs['reports'] / f'detailed_{phase}_report.txt'
            with open(report_file, 'w') as f:
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    stats.print_stats()  # Now prints to file
                finally:
                    sys.stdout = old_stdout  # Always restore stdout
            
            print(f"   Detailed profile saved: {profile_file}")
    
    def get_system_info(self):
        """Collect system information."""
        return {
            'python_version': os.sys.version,
            'numpy_version': np.__version__,
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total / 1024 / 1024 / 1024,  # GB
            'memory_available': psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
            'platform': os.sys.platform,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_simulation_visualizations(self):
        """Create comprehensive simulation performance visualizations."""
        if not self.phase_metrics:
            print("No profiling data to visualize")
            return
        
        # 1. Phase timing overview
        self.create_phase_timing_plot()
        
        # 2. Memory usage timeline
        self.create_memory_timeline_plot()
        
        # 3. Performance metrics dashboard
        self.create_performance_dashboard()
        
        # 4. Detailed phase analysis
        self.create_phase_analysis_plots()
        
        print(f"All visualizations saved to: {self.dirs['plots']}")
    
    def create_phase_timing_plot(self):
        """Create phase timing visualization."""
        phases = list(self.phase_metrics.keys())
        durations = [self.phase_metrics[p].duration_seconds for p in phases]
        colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart of phase durations
        bars = ax1.bar(phases, durations, color=colors, alpha=0.8)
        ax1.set_ylabel('Duration (seconds)')
        ax1.set_title('Simulation Phase Durations')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add duration labels on bars
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{duration:.1f}s', ha='center', va='bottom')
        
        # Pie chart of time distribution
        ax2.pie(durations, labels=phases, colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Time Distribution by Phase')
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'phase_timing.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_memory_timeline_plot(self):
        """Create memory usage timeline."""
        if not self.timeline_data:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Memory timeline
        times = []
        memories = []
        phase_boundaries = []
        
        for i, phase_data in enumerate(self.timeline_data):
            times.extend([phase_data['start_time'], phase_data['end_time']])
            memories.extend([phase_data['memory_start'], phase_data['memory_end']])
            
            if i > 0:  # Add phase boundary
                phase_boundaries.append(phase_data['start_time'])
        
        # Normalize times to start from 0
        start_time = min(times)
        times = [(t - start_time) for t in times]
        phase_boundaries = [(t - start_time) for t in phase_boundaries]
        
        ax1.plot(times, memories, 'b-', linewidth=2, marker='o', markersize=4)
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.set_title('Memory Usage Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Add phase boundaries
        for boundary in phase_boundaries:
            ax1.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        # Add phase labels
        for i, phase_data in enumerate(self.timeline_data):
            phase_start = phase_data['start_time'] - start_time
            phase_end = phase_data['end_time'] - start_time
            phase_center = (phase_start + phase_end) / 2
            
            ax1.text(phase_center, max(memories) * 0.9, 
                    phase_data['phase'], ha='center', rotation=45, fontsize=9)
        
        # Memory peaks by phase
        phases = [p['phase'] for p in self.timeline_data]
        peaks = [p['memory_peak'] for p in self.timeline_data]
        
        bars = ax2.bar(phases, peaks, alpha=0.7, color='lightcoral')
        ax2.set_ylabel('Peak Memory (MB)')
        ax2.set_title('Peak Memory Usage by Phase')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add peak values on bars
        for bar, peak in zip(bars, peaks):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{peak:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'memory_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_dashboard(self):
        """Create comprehensive performance dashboard."""
        phases = list(self.phase_metrics.keys())
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Simulation Performance Dashboard ({self.particle_count:,} particles)', 
                     fontsize=16, fontweight='bold')
        
        # 1. Steps per second by phase
        computation_phases = [p for p in phases if self.phase_metrics[p].steps_completed > 0]
        if computation_phases:
            steps_per_sec = [self.phase_metrics[p].steps_per_second for p in computation_phases]
            ax1.bar(computation_phases, steps_per_sec, alpha=0.7, color='skyblue')
            ax1.set_ylabel('Steps per Second')
            ax1.set_title('Computational Performance')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
        
        # 2. Memory efficiency
        memory_deltas = [self.phase_metrics[p].memory_delta_mb for p in phases]
        ax2.bar(phases, memory_deltas, alpha=0.7, color='lightgreen')
        ax2.set_ylabel('Memory Delta (MB)')
        ax2.set_title('Memory Allocation by Phase')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # 3. CPU usage
        cpu_usage = [self.phase_metrics[p].cpu_usage_percent for p in phases]
        ax3.bar(phases, cpu_usage, alpha=0.7, color='orange')
        ax3.set_ylabel('CPU Usage (%)')
        ax3.set_title('CPU Utilization by Phase')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Garbage collection activity
        gc_total = [sum(self.phase_metrics[p].gc_collections.values()) for p in phases]
        ax4.bar(phases, gc_total, alpha=0.7, color='purple')
        ax4.set_ylabel('GC Collections')
        ax4.set_title('Garbage Collection Activity')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.dirs['plots'] / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_phase_analysis_plots(self):
        """Create detailed phase analysis plots."""
        # Performance vs Memory trade-off
        phases = list(self.phase_metrics.keys())
        performance = [self.phase_metrics[p].steps_per_second for p in phases 
                      if self.phase_metrics[p].steps_completed > 0]
        memory_peaks = [self.phase_metrics[p].peak_memory_mb for p in phases 
                       if self.phase_metrics[p].steps_completed > 0]
        phase_names = [p for p in phases if self.phase_metrics[p].steps_completed > 0]
        
        if performance and memory_peaks:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            scatter = ax.scatter(performance, memory_peaks, s=100, alpha=0.7, c=range(len(performance)), cmap='viridis')
            
            # Add phase labels
            for i, name in enumerate(phase_names):
                ax.annotate(name.replace('_', '\n'), (performance[i], memory_peaks[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax.set_xlabel('Performance (steps/second)')
            ax.set_ylabel('Peak Memory (MB)')
            ax.set_title('Performance vs Memory Trade-off')
            ax.grid(True, alpha=0.3)
            
            plt.colorbar(scatter, label='Phase Order')
            plt.tight_layout()
            plt.savefig(self.dirs['plots'] / 'performance_vs_memory.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def save_comprehensive_results(self):
        """Save all profiling results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save main results
        json_file = self.dirs['data'] / f'simulation_profile_{timestamp}.json'
        with open(json_file, 'w') as f:
            json.dump(self.profile_result.to_dict(), f, indent=2)
        
        # Save timeline data
        timeline_file = self.dirs['data'] / f'timeline_data_{timestamp}.json'
        with open(timeline_file, 'w') as f:
            json.dump(self.timeline_data, f, indent=2)
        
        # Create CSV summary
        csv_data = []
        for phase_name, metrics in self.phase_metrics.items():
            row = {'phase': phase_name}
            row.update(metrics.to_dict())
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_file = self.dirs['data'] / f'simulation_metrics_{timestamp}.csv'
        df.to_csv(csv_file, index=False)
        
        # Create summary report
        self.create_summary_report(timestamp)
        
        print(f"\nResults saved:")
        print(f"  Complete profile: {json_file}")
        print(f"  Timeline data: {timeline_file}")
        print(f"  Metrics CSV: {csv_file}")
        print(f"  Summary report: {self.dirs['reports'] / f'simulation_summary_{timestamp}.txt'}")
    
    def create_summary_report(self, timestamp: str):
        """Create comprehensive summary report."""
        report_file = self.dirs['reports'] / f'simulation_summary_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SARKAS COMPLETE SIMULATION PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Particles: {self.particle_count:,}\n")
            f.write(f"Equilibration Steps: {self.equilibration_steps:,}\n")
            f.write(f"Production Steps: {self.production_steps:,}\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Duration: {self.profile_result.total_duration:.2f} seconds\n")
            f.write(f"Peak Memory: {self.profile_result.peak_memory_usage:.2f} MB\n")
            f.write(f"Memory Allocated: {self.profile_result.total_memory_allocated:.2f} MB\n\n")
            
            # Phase breakdown
            f.write("PHASE BREAKDOWN\n")
            f.write("-" * 30 + "\n")
            f.write(f"{'Phase':<20} {'Duration':<12} {'Memory':<12} {'Performance':<15}\n")
            f.write("-" * 65 + "\n")
            
            for phase_name, metrics in self.phase_metrics.items():
                perf_str = f"{metrics.steps_per_second:.1f} step/s" if metrics.steps_completed > 0 else "N/A"
                f.write(f"{phase_name:<20} {metrics.duration_seconds:<12.2f} "
                       f"{metrics.peak_memory_mb:<12.1f} {perf_str:<15}\n")
            
            # Performance insights
            computational_phases = [(name, metrics) for name, metrics in self.phase_metrics.items() 
                                  if metrics.steps_completed > 0]
            
            if computational_phases:
                f.write(f"\nPERFORMANCE INSIGHTS\n")
                f.write("-" * 30 + "\n")
                
                # Fastest phase
                fastest_phase = max(computational_phases, key=lambda x: x[1].steps_per_second)
                f.write(f"Fastest Phase: {fastest_phase[0]} ({fastest_phase[1].steps_per_second:.1f} steps/s)\n")
                
                # Most memory intensive
                memory_intensive = max(self.phase_metrics.items(), key=lambda x: x[1].peak_memory_mb)
                f.write(f"Most Memory Intensive: {memory_intensive[0]} ({memory_intensive[1].peak_memory_mb:.1f} MB)\n")
                
                # Longest phase
                longest_phase = max(self.phase_metrics.items(), key=lambda x: x[1].duration_seconds)
                f.write(f"Longest Phase: {longest_phase[0]} ({longest_phase[1].duration_seconds:.1f} seconds)\n")
    
    def run_complete_simulation_profiling(self, detailed_profiling: bool = True):
        """Run complete simulation profiling suite."""
        print("Starting Complete Simulation Profiling")
        print("="*80)
        
        start_time = time.time()
        
        # 1. Profile complete simulation
        self.profile_complete_simulation(detailed_profiling)
        
        # 2. Create visualizations
        self.create_simulation_visualizations()
        
        # 3. Save all results
        self.save_comprehensive_results()
        
        total_time = time.time() - start_time
        
        print(f"\n" + "="*80)
        print("COMPLETE SIMULATION PROFILING FINISHED")
        print("="*80)
        print(f"Profiling overhead: {total_time - self.profile_result.total_duration:.1f} seconds")
        print(f"Output directory: {self.output_dir}")
        print(f"Phases profiled: {len(self.phase_metrics)}")

def main():
    """Main execution function."""
    profiler = SimulationProfiler(
        particle_count=10_000,
        equilibration_steps=2800,
        production_steps=2000,
        output_dir="complete_simulation_profiling"
    )
    
    profiler.run_complete_simulation_profiling(detailed_profiling=True)

if __name__ == "__main__":
    main()