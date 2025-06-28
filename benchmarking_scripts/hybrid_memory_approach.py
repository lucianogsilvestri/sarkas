"""
Hybrid approach: Keep AoS interface but provide SoA views for hot path operations.
This gives performance benefits without major code disruption.
"""

import numpy as np
from numba import jit

class OptimizedParticles:
    """
    Hybrid approach: AoS interface with SoA performance optimization.
    
    Strategy:
    1. Keep familiar AoS interface for user code
    2. Provide SoA views for performance-critical operations
    3. Lazy conversion between representations
    4. Automatic optimization for known hot paths
    """
    
    def __init__(self, N):
        # Primary storage: AoS (familiar interface)
        self.pos = np.zeros((N, 3))
        self.vel = np.zeros((N, 3))
        self.acc = np.zeros((N, 3))
        self.masses = np.zeros(N)
        self.charges = np.zeros(N)
        
        # Performance optimization: cached SoA views
        self._soa_cache = {}
        self._soa_dirty = True
        
    def _ensure_soa_cache(self):
        """Create SoA views when needed for performance operations."""
        if self._soa_dirty or 'pos_x' not in self._soa_cache:
            self._soa_cache = {
                'pos_x': self.pos[:, 0],    # View, not copy!
                'pos_y': self.pos[:, 1],
                'pos_z': self.pos[:, 2],
                'vel_x': self.vel[:, 0],
                'vel_y': self.vel[:, 1], 
                'vel_z': self.vel[:, 2],
                'acc_x': self.acc[:, 0],
                'acc_y': self.acc[:, 1],
                'acc_z': self.acc[:, 2],
            }
            self._soa_dirty = False
    
    def calculate_kinetic_energy(self, use_optimized=True):
        """
        Calculate kinetic energy with optional SoA optimization.
        
        Parameters
        ----------
        use_optimized : bool
            Use SoA layout for better performance
        """
        if use_optimized and len(self.masses) > 5000:  # Only for large systems
            self._ensure_soa_cache()
            return self._kinetic_energy_soa(
                self._soa_cache['vel_x'], 
                self._soa_cache['vel_y'],
                self._soa_cache['vel_z'], 
                self.masses
            )
        else:
            # Standard AoS calculation for small systems
            return self._kinetic_energy_aos(self.vel, self.masses)
    
    @staticmethod
    @jit(nopython=True)
    def _kinetic_energy_aos(vel, masses):
        """Standard AoS kinetic energy calculation."""
        ke = np.zeros(len(masses))
        for i in range(len(masses)):
            v2 = vel[i, 0]**2 + vel[i, 1]**2 + vel[i, 2]**2
            ke[i] = 0.5 * masses[i] * v2
        return ke
        
    @staticmethod
    @jit(nopython=True)
    def _kinetic_energy_soa(vel_x, vel_y, vel_z, masses):
        """Optimized SoA kinetic energy calculation."""
        ke = np.zeros(len(masses))
        for i in range(len(masses)):
            v2 = vel_x[i]**2 + vel_y[i]**2 + vel_z[i]**2
            ke[i] = 0.5 * masses[i] * v2
        return ke
    
    def update_positions_verlet(self, dt, use_optimized=True):
        """
        Verlet integration with optional SoA optimization.
        """
        if use_optimized and len(self.masses) > 5000:
            self._ensure_soa_cache()
            self._verlet_update_soa(
                self._soa_cache['pos_x'], self._soa_cache['pos_y'], self._soa_cache['pos_z'],
                self._soa_cache['vel_x'], self._soa_cache['vel_y'], self._soa_cache['vel_z'],
                self._soa_cache['acc_x'], self._soa_cache['acc_y'], self._soa_cache['acc_z'],
                self.masses, dt
            )
            self._soa_dirty = True  # Mark for potential re-sync
        else:
            self._verlet_update_aos(self.pos, self.vel, self.acc, self.masses, dt)
    
    @staticmethod
    @jit(nopython=True)
    def _verlet_update_aos(pos, vel, acc, masses, dt):
        """Standard AoS Verlet update."""
        for i in range(len(masses)):
            # Update velocities
            vel[i, 0] += 0.5 * dt * acc[i, 0]
            vel[i, 1] += 0.5 * dt * acc[i, 1]
            vel[i, 2] += 0.5 * dt * acc[i, 2]
            
            # Update positions
            pos[i, 0] += dt * vel[i, 0]
            pos[i, 1] += dt * vel[i, 1]
            pos[i, 2] += dt * vel[i, 2]
    
    @staticmethod
    @jit(nopython=True)
    def _verlet_update_soa(pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
                          acc_x, acc_y, acc_z, masses, dt):
        """Optimized SoA Verlet update."""
        for i in range(len(masses)):
            # Update velocities
            vel_x[i] += 0.5 * dt * acc_x[i]
            vel_y[i] += 0.5 * dt * acc_y[i]
            vel_z[i] += 0.5 * dt * acc_z[i]
            
            # Update positions
            pos_x[i] += dt * vel_x[i]
            pos_y[i] += dt * vel_y[i]
            pos_z[i] += dt * vel_z[i]
    
    def pppm_charge_assignment(self, mesh, use_optimized=True):
        """
        PPPM charge assignment with SoA optimization for large systems.
        """
        if use_optimized and len(self.charges) > 10000:
            self._ensure_soa_cache()
            return self._charge_assignment_soa(
                self._soa_cache['pos_x'],
                self._soa_cache['pos_y'], 
                self._soa_cache['pos_z'],
                self.charges, mesh
            )
        else:
            return self._charge_assignment_aos(self.pos, self.charges, mesh)
    
    @staticmethod
    @jit(nopython=True)
    def _charge_assignment_aos(pos, charges, mesh):
        """Standard AoS charge assignment."""
        mesh_size = mesh.shape[0]
        cell_size = 1.0 / mesh_size
        
        for i in range(len(charges)):
            ix = min(int(pos[i, 0] / cell_size), mesh_size - 1)
            iy = min(int(pos[i, 1] / cell_size), mesh_size - 1)
            iz = min(int(pos[i, 2] / cell_size), mesh_size - 1)
            mesh[iz, iy, ix] += charges[i]
        
        return mesh
    
    @staticmethod
    @jit(nopython=True)
    def _charge_assignment_soa(pos_x, pos_y, pos_z, charges, mesh):
        """Optimized SoA charge assignment."""
        mesh_size = mesh.shape[0]
        cell_size = 1.0 / mesh_size
        
        for i in range(len(charges)):
            ix = min(int(pos_x[i] / cell_size), mesh_size - 1)
            iy = min(int(pos_y[i] / cell_size), mesh_size - 1)
            iz = min(int(pos_z[i] / cell_size), mesh_size - 1)
            mesh[iz, iy, ix] += charges[i]
        
        return mesh


class AdaptiveParticleLayout:
    """
    Automatically choose optimal memory layout based on system size and operation.
    """
    
    THRESHOLDS = {
        'kinetic_energy': 5000,      # Use SoA for > 5k particles
        'force_calculation': 10000,   # Use SoA for > 10k particles  
        'integration': 5000,          # Use SoA for > 5k particles
        'charge_assignment': 8000,    # Use SoA for > 8k particles
    }
    
    def __init__(self, particles):
        self.particles = particles
        self.N = len(particles.masses)
        
    def should_use_soa(self, operation):
        """Decide whether to use SoA for given operation."""
        threshold = self.THRESHOLDS.get(operation, 10000)
        return self.N >= threshold
    
    def profile_and_adapt(self, operation, iterations=5):
        """
        Profile both AoS and SoA performance and adapt threshold.
        """
        import time
        
        # Benchmark AoS
        aos_times = []
        for _ in range(iterations):
            start = time.time()
            if operation == 'kinetic_energy':
                result_aos = self.particles.calculate_kinetic_energy(use_optimized=False)
            aos_times.append(time.time() - start)
        
        # Benchmark SoA
        soa_times = []
        for _ in range(iterations):
            start = time.time()
            if operation == 'kinetic_energy':
                result_soa = self.particles.calculate_kinetic_energy(use_optimized=True)
            soa_times.append(time.time() - start)
        
        aos_time = np.mean(aos_times)
        soa_time = np.mean(soa_times)
        speedup = aos_time / soa_time
        
        # Adapt threshold based on results
        if speedup > 1.1:  # SoA is significantly faster
            self.THRESHOLDS[operation] = min(self.THRESHOLDS[operation], self.N)
        elif speedup < 0.95:  # AoS is faster
            self.THRESHOLDS[operation] = max(self.THRESHOLDS[operation], self.N * 2)
        
        return {
            'aos_time': aos_time,
            'soa_time': soa_time, 
            'speedup': speedup,
            'new_threshold': self.THRESHOLDS[operation]
        }


# Backward compatible interface
class Particles(OptimizedParticles):
    """
    Drop-in replacement for current Particles class.
    
    Provides exact same interface as current Sarkas Particles,
    but with automatic performance optimization for large systems.
    """
    
    def calculate_kinetic_energy(self):
        """Same interface as current Sarkas, but automatically optimized."""
        # Automatically choose best implementation
        use_soa = len(self.masses) > 5000
        self.kinetic_energy = super().calculate_kinetic_energy(use_optimized=use_soa)
    
    def calculate_species_kinetic_temperature(self):
        """Current Sarkas interface with automatic optimization."""
        # Calculate per-particle kinetic energy (optimized)
        self.calculate_kinetic_energy()
        
        # Species aggregation (same as current code)
        self.species_kinetic_energy = self._aggregate_by_species(self.kinetic_energy)
        
        # Temperature calculation
        self.species_temperature = (2.0 * self.species_kinetic_energy / 
                                   (self.dimensions * self.species_num * self.kB))
    
    def _aggregate_by_species(self, per_particle_quantity):
        """Helper method for species aggregation."""
        species_totals = np.zeros(self.num_species)
        sp_start = 0
        for sp in range(self.num_species):
            sp_end = sp_start + self.species_num[sp]
            species_totals[sp] = per_particle_quantity[sp_start:sp_end].sum()
            sp_start = sp_end
        return species_totals


def benchmark_hybrid_approach():
    """
    Benchmark the hybrid approach to show benefits without major disruption.
    """
    import time
    
    print("=== Hybrid Approach Benchmark ===\n")
    
    # Test different system sizes
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    for N in sizes:
        print(f"Testing {N} particles:")
        
        # Current approach (pure AoS)
        particles_aos = OptimizedParticles(N)
        
        # Time kinetic energy calculation
        times_aos = []
        times_hybrid = []
        
        for _ in range(10):  # Average over multiple runs
            # Pure AoS
            start = time.time()
            ke_aos = particles_aos.calculate_kinetic_energy(use_optimized=False)
            times_aos.append(time.time() - start)
            
            # Hybrid (automatic optimization)
            start = time.time()
            ke_hybrid = particles_aos.calculate_kinetic_energy(use_optimized=True)
            times_hybrid.append(time.time() - start)
        
        aos_time = np.mean(times_aos) * 1000  # Convert to ms
        hybrid_time = np.mean(times_hybrid) * 1000
        speedup = aos_time / hybrid_time
        
        print(f"  AoS time: {aos_time:.3f} ms")
        print(f"  Hybrid time: {hybrid_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        
        # Verify correctness
        np.testing.assert_allclose(ke_aos, ke_hybrid, rtol=1e-10)
        print(f"  ✅ Results identical\n")


if __name__ == "__main__":
    benchmark_hybrid_approach()