"""
Implementation of boundary condition classes.

This module contains the actual implementations of different boundary conditions
using optimized Numba functions for maximum performance.
"""

import numpy as zeros, floor
from numba import jit
from numba.types import bool_
from .base import BoundaryConditionBase


@jit(nopython=True, cache=True)
def _enforce_pbc_numba(pos, cntr, box_lengths):
    """
    Numba'd function to enforce periodic boundary conditions.

    Parameters
    ----------
    pos : numpy.ndarray
        Particles' positions. Shape (N, D) where D is dimensions

    cntr : numpy.ndarray
        Counter for the number of times each particle gets folded back 
        into the main simulation box. Shape (N, D)

    box_lengths : numpy.ndarray
        Box dimensions. Shape (D,)
    """
    # Get the number of particles and dimensions
    N = pos.shape[0]
    D = pos.shape[1]
    
    for i in range(N):
        for j in range(D):
            L = box_lengths[j]
            # Calculate how many times this particle crosses the boundary
            fold_count = int(floor(pos[i, j] / L))
            cntr[i, j] += fold_count
            # Apply the boundary condition in-place
            pos[i, j] -= L * fold_count


@jit(nopython=True, cache=True)
def _enforce_abc_numba(pos, vel, acc, charges, box_lengths):
    """
    Optimized Numba function to enforce absorbing boundary conditions.

    Parameters
    ----------
    pos : numpy.ndarray
        Particles' positions. Shape (N, D)

    vel : numpy.ndarray
        Particles' velocities. Shape (N, D)

    acc : numpy.ndarray
        Particles' accelerations. Shape (N, D)

    charges : numpy.ndarray
        Charge of each particle. Shape (N,)

    box_lengths : numpy.ndarray
        Box dimensions. Shape (D,)
    """
    # Get dimensions for later use
    n_particles = pos.shape[0]
    n_dims = pos.shape[1]
    
    # Pre-allocate masks for particles that need modification
    outside_box = zeros(n_particles, dtype=bool_)
    
    # First, identify all particles that are outside the box in any dimension
    for d in range(n_dims):
        for p in range(n_particles):
            if pos[p, d] >= box_lengths[d] or pos[p, d] <= 0.0:
                outside_box[p] = True
                
                # Fix positions at boundary
                if pos[p, d] >= box_lengths[d]:
                    pos[p, d] = box_lengths[d]
                else:
                    pos[p, d] = 0.0
    
    # Now apply changes only to particles that need it
    for p in range(n_particles):
        if outside_box[p]:
            # Reset velocity, acceleration, and charge
            for d in range(n_dims):
                vel[p, d] = 0.0
                acc[p, d] = 0.0
            charges[p] = 0.0


@jit(nopython=True, cache=True)
def _enforce_rbc_numba(pos, vel, box_lengths, dt):
    """
    Optimized Numba function to enforce reflecting boundary conditions.

    Parameters
    ----------
    pos : numpy.ndarray
        Particles' positions. Shape (N, D)

    vel : numpy.ndarray
        Particles' velocities. Shape (N, D)

    box_lengths : numpy.ndarray
        Box dimensions. Shape (D,)

    dt : float
        Timestep.
    """
    # Get dimensions for later use
    n_particles = pos.shape[0]
    n_dims = pos.shape[1]
    
    # Pre-allocate arrays to avoid recreating them in loops
    outside_particles = zeros(n_particles, dtype=bool_)
    
    # For each dimension, find particles outside boundaries and reflect them
    for d in range(n_dims):
        # Reset the outside particle flags for this dimension
        for p in range(n_particles):
            outside_particles[p] = False
            
        # First, identify particles outside the box in this dimension
        for p in range(n_particles):
            if pos[p, d] > box_lengths[d] or pos[p, d] < 0.0:
                outside_particles[p] = True
        
        # Then, apply reflections only to those particles
        for p in range(n_particles):
            if outside_particles[p]:
                # Reverse velocity component
                vel[p, d] *= -1.0
                # Restore previous position assuming Verlet algorithm
                pos[p, d] += vel[p, d] * dt


class PeriodicBC(BoundaryConditionBase):
    """
    Periodic boundary conditions implementation.
    
    Particles crossing boundaries reappear on the opposite side.
    Maintains a counter of boundary crossings for each particle.
    
    Attributes
    ----------
    _counter_initialized : bool
        Whether the crossing counter has been initialized
    """
    
    _aliases = ['pbc', 'periodic_bc', 'periodic']
    
    def __init__(self, box_lengths, dimensions, **kwargs):
        super().__init__(box_lengths, dimensions, **kwargs)
        
        # Set capability flags
        self.requires_velocities = False
        self.requires_accelerations = False
        self.requires_charges = False
        self.requires_timestep = False
        
        # Set modification flags
        self.modifies_positions = True
        self.modifies_velocities = False
        self.modifies_accelerations = False
        self.modifies_charges = False
        
        # Internal state
        self._counter_initialized = False
    
    def enforce(self, ptcls):
        """
        Enforce periodic boundary conditions.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to apply boundary conditions to
        """

        # Apply periodic boundary conditions
        _enforce_pbc_numba(ptcls.pos, ptcls.pbc_cntr, self.box_lengths)


class AbsorbingBC(BoundaryConditionBase):
    """
    Absorbing boundary conditions implementation.
    
    Particles hitting boundaries are effectively removed by setting
    their charge to zero and resetting their properties.
    """
    
    _aliases = ['abc', 'absorbing_bc', 'absorbing', 'sink']
    
    def __init__(self, box_lengths, dimensions, **kwargs):
        super().__init__(box_lengths, dimensions, **kwargs)
        
        # Set capability flags
        self.requires_velocities = True
        self.requires_accelerations = True
        self.requires_charges = True
        self.requires_timestep = False
        
        # Set modification flags
        self.modifies_positions = True
        self.modifies_velocities = True
        self.modifies_accelerations = True
        self.modifies_charges = True
    
    def enforce(self, ptcls):
        """
        Enforce absorbing boundary conditions.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to apply boundary conditions to
        """

        # Apply absorbing boundary conditions
        _enforce_abc_numba(ptcls.pos, ptcls.vel, ptcls.acc, ptcls.charges, self.box_lengths)

class ReflectingBC(BoundaryConditionBase):
    """
    Reflecting boundary conditions implementation.
    
    Particles hitting boundaries are reflected back into the simulation box.
    """
    
    _aliases = ['rbc', 'reflecting_bc', 'reflecting']
    
    def __init__(self, box_lengths, dimensions **kwargs):
        super().__init__(box_lengths, dimensions, **kwargs)
        
        # Set capability flags
        self.requires_velocities = True
        self.requires_accelerations = False
        self.requires_charges = False
        self.requires_timestep = True
        
        # Set modification flags
        self.modifies_positions = True
        self.modifies_velocities = True
        self.modifies_accelerations = False
        self.modifies_charges = False
    
    def enforce(self, ptcls):
        """
        Enforce reflecting boundary conditions.
        
        Parameters
        ----------
        ptcls : Particles
            Particle data to apply boundary conditions to
        """

        # Apply reflecting boundary conditions
        _enforce_rbc_numba(ptcls.pos, ptcls.vel, self.box_lengths, ptcls.timestep)