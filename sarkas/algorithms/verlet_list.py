"""
Module for Verlet neighbor list algorithm for efficient particle interaction computation.
"""

from numba import jit
from numba.typed import List
from numpy import arange, sqrt, zeros, zeros_like, array, empty, int32
from .base import InteractionSolverBase
from .cell_list import LinkedCellList


class VerletList(InteractionSolverBase):
    """
    Verlet neighbor list algorithm for computing particle interactions.
    
    This algorithm maintains explicit neighbor lists for each particle, which are
    updated periodically rather than every timestep. The neighbor lists include
    particles within a "skin" distance larger than the cutoff radius, allowing
    the lists to remain valid for multiple timesteps.
    
    Key advantages:
    - Much faster force computation (only checks actual neighbors)
    - Better cache performance (neighbors stored contiguously)
    - Excellent for dense systems
    - Significantly reduces distance calculations
    
    The algorithm uses LinkedCellList internally for efficient neighbor finding
    during list construction.
    
    Attributes
    ----------
    cutoff_radius : float
        Interaction cutoff radius
    skin_thickness : float
        Additional distance beyond cutoff for neighbor lists
    neighbor_cutoff : float
        Total distance for neighbor list (cutoff + skin)
    rebuild_frequency : int
        Number of timesteps between neighbor list rebuilds
    neighbor_lists : list
        List of neighbor arrays for each particle
    max_displacement : float
        Maximum particle displacement since last rebuild
    """

    def __init__(self):
        """Initialize the Verlet list solver."""
        super().__init__()
        self.type = 'verlet_list'
        self.cutoff_radius = None
        self.skin_thickness = None
        self.neighbor_cutoff = None
        self.rebuild_frequency = 20
        self.neighbor_lists = []
        self.max_neighbors_per_particle = 100
        self.timestep_counter = 0
        self.last_positions = None
        self.max_displacement = 0.0
        
        # Use LinkedCellList for efficient neighbor finding
        self.cell_list = LinkedCellList()

    def setup(self, params, **kwargs):
        """
        Initialize the Verlet list with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions
            - cutoff_radius : float, interaction cutoff
            - Other parameters passed to cell list
        **kwargs : dict
            Verlet-specific parameters:
            - skin_thickness : float, skin distance (default: 0.3 * cutoff)
            - rebuild_frequency : int, timesteps between rebuilds (default: 20)
            - max_neighbors : int, maximum neighbors per particle (default: 100)
            - displacement_check : bool, rebuild based on displacement (default: True)
        """
        self.box_lengths = params.box_lengths
        self.cutoff_radius = params.cutoff_radius
        
        # Set skin thickness (typical: 20-50% of cutoff radius)
        self.skin_thickness = kwargs.get('skin_thickness', 0.3 * self.cutoff_radius)
        self.neighbor_cutoff = self.cutoff_radius + self.skin_thickness
        
        # Set rebuild parameters
        self.rebuild_frequency = kwargs.get('rebuild_frequency', 20)
        self.max_neighbors_per_particle = kwargs.get('max_neighbors', 100)
        self.displacement_check = kwargs.get('displacement_check', True)
        
        # Setup cell list with neighbor cutoff
        # Temporarily modify params for cell list setup
        original_cutoff = params.cutoff_radius
        params.cutoff_radius = self.neighbor_cutoff
        self.cell_list.setup(params)
        params.cutoff_radius = original_cutoff  # Restore original

        # Initialize other attributes
        self.dimensions = params.dimensions
        self.total_num_density = params.total_num_density
        self.units_dict = params.units_dict
        self.a_ws = params.a_ws

    def needs_rebuild(self, pos=None):
        """
        Determine if neighbor lists need to be rebuilt.
        
        Parameters
        ----------
        pos : numpy.ndarray, optional
            Current particle positions for displacement check
            
        Returns
        -------
        bool
            True if lists should be rebuilt
        """
        # Always rebuild on first call
        if len(self.neighbor_lists) == 0:
            return True
            
        # Rebuild based on timestep frequency
        if self.timestep_counter % self.rebuild_frequency == 0:
            return True
            
        # Rebuild based on particle displacement
        if self.displacement_check and pos is not None and self.last_positions is not None:
            self.update_max_displacement(pos)
            # Rebuild if particles moved more than half the skin thickness
            if self.max_displacement > 0.5 * self.skin_thickness:
                return True
                
        return False

    def update_max_displacement(self, pos):
        """
        Update maximum particle displacement since last rebuild.
        
        Parameters
        ----------
        pos : numpy.ndarray
            Current particle positions
        """
        if self.last_positions is not None:
            displacements = pos - self.last_positions
            
            # Apply minimum image correction for periodic boundaries
            for dim in range(3):
                if self.box_lengths[dim] > 0:
                    half_box = 0.5 * self.box_lengths[dim]
                    displacements[:, dim] = displacements[:, dim] - self.box_lengths[dim] * (
                        (displacements[:, dim] > half_box).astype(int) - 
                        (displacements[:, dim] < -half_box).astype(int)
                    )
            
            # Calculate displacement magnitudes
            displacement_magnitudes = sqrt((displacements**2).sum(axis=1))
            self.max_displacement = max(self.max_displacement, displacement_magnitudes.max())

    @staticmethod
    @jit(nopython=True)
    def build_neighbor_lists_numba(pos, box_lengths, neighbor_cutoff, max_neighbors,
                                   cells_per_dim, cell_length_per_dim):
        """
        Build neighbor lists using cell list algorithm (numba-optimized).
        
        Parameters
        ----------
        pos : numpy.ndarray
            Particle positions
        box_lengths : numpy.ndarray
            Box dimensions
        neighbor_cutoff : float
            Neighbor list cutoff distance
        max_neighbors : int
            Maximum neighbors per particle
        cells_per_dim : numpy.ndarray
            Number of cells per dimension
        cell_length_per_dim : numpy.ndarray
            Cell sizes
            
        Returns
        -------
        neighbor_lists : numba.typed.List
            List of neighbor arrays for each particle
        neighbor_counts : numpy.ndarray
            Number of neighbors for each particle
        """
        N = pos.shape[0]
        
        # Create head and list arrays for cell list
        ls = arange(N)
        Ncell = cells_per_dim[cells_per_dim > 0].prod()
        head = arange(Ncell)
        empty = -50
        head.fill(empty)
        
        # Assign particles to cells
        for i in range(N):
            cx = int(pos[i, 0] / (1 * (cell_length_per_dim[0] == 0.0) + cell_length_per_dim[0]))
            cy = int(pos[i, 1] / (1 * (cell_length_per_dim[1] == 0.0) + cell_length_per_dim[1]))
            cz = int(pos[i, 2] / (1 * (cell_length_per_dim[2] == 0.0) + cell_length_per_dim[2]))
            
            c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
            ls[i] = head[c]
            head[c] = i

        # Initialize neighbor lists
        neighbor_lists = zeros((N, max_neighbors), dtype=int32)
        neighbor_counts = zeros(N, dtype=int32)
        
        # Cell loop parameters
        d3_min = min(cells_per_dim[2], 1)
        d3_max = max(cells_per_dim[2], 1)
        d2_min = min(cells_per_dim[1], 1)
        d2_max = max(cells_per_dim[1], 1)
        d1_min = min(cells_per_dim[0], 1)
        d1_max = max(cells_per_dim[0], 1)
        
        rshift = zeros(3)
        neighbor_cutoff_sq = neighbor_cutoff * neighbor_cutoff
        
        # Loop over all cells
        for cz in range(d3_max):
            for cy in range(d2_max):
                for cx in range(d1_max):
                    c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]
                    
                    # Loop over neighboring cells
                    for cz_N in range(cz - 1, (cz + 2) * d3_min):
                        cz_shift = 0 + d3_max * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                        rshift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2] * (cz_N >= cells_per_dim[2])
                        
                        for cy_N in range(cy - 1, (cy + 2) * d2_min):
                            cy_shift = 0 + d2_max * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                            rshift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim[1])
                            
                            for cx_N in range(cx - 1, (cx + 2) * d1_min):
                                cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                                rshift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim[0])
                                
                                c_N = (
                                    (cx_N + cx_shift) + 
                                    (cy_N + cy_shift) * cells_per_dim[0] + 
                                    (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                                )
                                
                                i = head[c]
                                while i >= 0:
                                    j = head[c_N]
                                    while j >= 0:
                                        if i != j:  # Don't include self
                                            # Calculate distance with periodic boundaries
                                            dx = pos[i, 0] - (pos[j, 0] + rshift[0])
                                            dy = pos[i, 1] - (pos[j, 1] + rshift[1])
                                            dz = pos[i, 2] - (pos[j, 2] + rshift[2])
                                            
                                            r_sq = dx*dx + dy*dy + dz*dz
                                            
                                            # Add to neighbor list if within cutoff
                                            if r_sq < neighbor_cutoff_sq:
                                                if neighbor_counts[i] < max_neighbors:
                                                    neighbor_lists[i, neighbor_counts[i]] = j
                                                    neighbor_counts[i] += 1
                                        
                                        j = ls[j]
                                    i = ls[i]
        
        return neighbor_lists, neighbor_counts

    def rebuild_neighbor_lists(self, pos):
        """
        Rebuild neighbor lists for all particles.
        
        Parameters
        ----------
        pos : numpy.ndarray
            Current particle positions
        """
        # Update cell list structure if needed
        self.cell_list.create_cells_array()
        
        # Build neighbor lists using optimized numba function
        neighbor_arrays, neighbor_counts = self.build_neighbor_lists_numba(
            pos, self.box_lengths, self.neighbor_cutoff, self.max_neighbors_per_particle,
            self.cell_list.cells_per_dim, self.cell_list.cell_length_per_dim
        )
        
        # Convert to list of arrays for each particle
        self.neighbor_lists = []
        for i in range(pos.shape[0]):
            count = neighbor_counts[i]
            if count > 0:
                self.neighbor_lists.append(neighbor_arrays[i, :count].copy())
            else:
                self.neighbor_lists.append(array([], dtype=int32))
        
        # Update tracking variables
        self.last_positions = pos.copy()
        self.max_displacement = 0.0
        self.timestep_counter = 0

    @staticmethod
    @jit(nopython=True)
    def calculate_forces_from_neighbors(pos, vel, p_mass, p_id, rdf_hist, 
                                       potential_matrix, force, box_lengths, cutoff_radius,
                                       neighbor_lists_flat, neighbor_offsets):
        """
        Compute forces using pre-built neighbor lists (numba-optimized).
        
        Parameters
        ----------
        Parameters
        ----------
        pos: numpy.ndarray
            Particles' positions. Shape (N, 3) where N is the number of particles.
        vel: numpy.ndarray
            Particles' positions. Shape (N, 3) where N is the number of particles.
        p_mass: numpy.ndarray
            Mass of each particle. Shape (N,).
        p_id: numpy.ndarray
            Id of each particle. Shape (N,). 
        rdf_hist : numpy.ndarray
            RDF histogram
        potential_matrix : numpy.ndarray
            Potential parameters
        force : callable
            Force function
        box_lengths : numpy.ndarray
            Box dimensions
        cutoff_radius : float
            Interaction cutoff
        neighbor_lists_flat : numpy.ndarray
            Flattened neighbor lists
        neighbor_offsets : numpy.ndarray
            Offsets for each particle's neighbors
            
        Returns
        -------
        tuple
            (potential_energy, accelerations, virial_tensor, heat_flux_tensor)
        """
        N = pos.shape[0]
        
        # Initialize output arrays
        ptcl_pot_energy = zeros(N)
        acc_s_r = zeros_like(pos)
        # energy current
        j_e = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3))
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3, 3))

        # RDF parameters
        rdf_nbins = rdf_hist.shape[0]
        dr_rdf = cutoff_radius / float(rdf_nbins)
        
        # Half box lengths for minimum image
        half_box = 0.5 * box_lengths
        
        # Loop over all particles
        for i in range(N):
            start_idx = neighbor_offsets[i]
            end_idx = neighbor_offsets[i + 1]
            
            # Loop over neighbors of particle i
            for neighbor_idx in range(start_idx, end_idx):
                j = neighbor_lists_flat[neighbor_idx]
                
                if i < j:  # Only compute each pair once
                    # Calculate relative position with periodic boundaries
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dz = pos[i, 2] - pos[j, 2]
                    
                    # Apply minimum image convention
                    dx = dx - box_lengths[0] * ((dx > half_box[0]) - (dx < -half_box[0]))
                    dy = dy - box_lengths[1] * ((dy > half_box[1]) - (dy < -half_box[1]))
                    dz = dz - box_lengths[2] * ((dz > half_box[2]) - (dz < -half_box[2]))
                    
                    # Calculate distance
                    r_squared = dx*dx + dy*dy + dz*dz
                    r_in = sqrt(r_squared)
                    
                    # Get species information
                    id_i = p_id[i]
                    id_j = p_id[j]
                    p_matrix = potential_matrix[id_i, id_j]
                    
                    # Handle singularities
                    rs = p_matrix[-1]
                    r = r_in * (r_in >= rs) + rs * (r_in < rs)
                    
                    # Update RDF
                    rdf_bin = int(r / dr_rdf)
                    if rdf_bin < rdf_nbins:
                        rdf_hist[rdf_bin, id_i, id_j] += 1
                    
                    # Compute forces if within cutoff
                    # If below the cutoff radius, compute the force
                    if r < cutoff_radius:
                        p_matrix = potential_matrix[id_i, id_j]
                        # neighbors[i, j] = j

                        # Compute the short-ranged force
                        pot, fr = force(r, p_matrix)
                        fr /= r
                        # Need to add the same pot to each particle pair.
                        # The factor of 1/2 is to account for the fact that we are counting each pair twice
                        # The total potential energy will be calculated from the sum of the potential energy of each particle (ptcls_pot_energy = ptcls.potential_energy)
                        # The total potential energy is 1/2 * \sum_{i = 1}^N \sum_{j = 1, \\ j \neq i}^N U(r_ij)
                        ptcl_pot_energy[i] += 0.5 * pot
                        ptcl_pot_energy[j] += 0.5 * pot

                        # Update the acceleration for i particles in each dimension

                        acc_s_r[i, 0] += dx * fr / p_mass[i]
                        acc_s_r[i, 1] += dy * fr / p_mass[i]
                        acc_s_r[i, 2] += dz * fr / p_mass[i]

                        # Apply Newton's 3rd law to update acceleration on j particles
                        acc_s_r[j, 0] -= dx * fr / p_mass[j]
                        acc_s_r[j, 1] -= dy * fr / p_mass[j]
                        acc_s_r[j, 2] -= dz * fr / p_mass[j]

                        # Since we have the info already calculate the virial_species_tensor
                        # This factor is to avoid double counting in the case of same species
                        factor = 0.5  # * (id_i != id_j) + 0.25*( id_i == id_j)
                        virial_species_tensor[id_i, id_j, 0, 0] += factor * dx * dx * fr
                        virial_species_tensor[id_i, id_j, 0, 1] += factor * dx * dy * fr
                        virial_species_tensor[id_i, id_j, 0, 2] += factor * dx * dz * fr
                        virial_species_tensor[id_i, id_j, 1, 0] += factor * dy * dx * fr
                        virial_species_tensor[id_i, id_j, 1, 1] += factor * dy * dy * fr
                        virial_species_tensor[id_i, id_j, 1, 2] += factor * dy * dz * fr
                        virial_species_tensor[id_i, id_j, 2, 0] += factor * dz * dx * fr
                        virial_species_tensor[id_i, id_j, 2, 1] += factor * dz * dy * fr
                        virial_species_tensor[id_i, id_j, 2, 2] += factor * dz * dz * fr
                        # This is where the double counting could happen.
                        virial_species_tensor[id_j, id_i, 0, 0] += factor * dx * dx * fr
                        virial_species_tensor[id_j, id_i, 0, 1] += factor * dx * dy * fr
                        virial_species_tensor[id_j, id_i, 0, 2] += factor * dx * dz * fr
                        virial_species_tensor[id_j, id_i, 1, 0] += factor * dy * dx * fr
                        virial_species_tensor[id_j, id_i, 1, 1] += factor * dy * dy * fr
                        virial_species_tensor[id_j, id_i, 1, 2] += factor * dy * dz * fr
                        virial_species_tensor[id_j, id_i, 2, 0] += factor * dz * dx * fr
                        virial_species_tensor[id_j, id_i, 2, 1] += factor * dz * dy * fr
                        virial_species_tensor[id_j, id_i, 2, 2] += factor * dz * dz * fr

                        fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                        # For this further factor of 1/2 see eq.(5) in https://doi.org/10.1016/j.cpc.2013.01.008
                        factor *= 0.5

                        j_e[id_i, id_j, 0] += factor * dx * fij_vij
                        j_e[id_i, id_j, 1] += factor * dy * fij_vij
                        j_e[id_i, id_j, 2] += factor * dz * fij_vij

                        j_e[id_j, id_i, 0] += factor * dx * fij_vij
                        j_e[id_j, id_i, 1] += factor * dy * fij_vij
                        j_e[id_j, id_i, 2] += factor * dz * fij_vij
        
        # Add the ideal term of the energy current
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[id_i, id_i, 0] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[id_i, id_i, 1] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[id_i, id_i, 2] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        
        return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e

    def flatten_neighbor_lists(self):
        """
        Convert neighbor lists to flat arrays for numba compatibility.
        
        Returns
        -------
        tuple
            (flat_neighbors, offsets) - flattened neighbor array and offset indices
        """
        total_neighbors = sum(len(neighbors) for neighbors in self.neighbor_lists)
        flat_neighbors = empty(total_neighbors, dtype=int32)
        offsets = zeros(len(self.neighbor_lists) + 1, dtype=int32)
        
        current_offset = 0
        for i, neighbors in enumerate(self.neighbor_lists):
            start = current_offset
            end = current_offset + len(neighbors)
            flat_neighbors[start:end] = neighbors
            offsets[i + 1] = end
            current_offset = end
            
        return flat_neighbors, offsets

    def update(self, ptcls, potential):
        """
        Calculate particle interactions using Verlet neighbor lists.
        
        Parameters
        ----------
        ptcls : object
            Particles data containing positions, velocities, masses, etc.
        potential : object
            Potential class containing force functions and parameters
        """
        # Check if neighbor lists need rebuilding
        if self.needs_rebuild(ptcls.pos):
            self.rebuild_neighbor_lists(ptcls.pos)
        
        # Flatten neighbor lists for numba
        flat_neighbors, offsets = self.flatten_neighbor_lists()
        
        # Compute forces using neighbor lists
        (ptcls.potential_energy,
         ptcls.acc,
         ptcls.virial_species_tensor,
         ptcls.heat_flux_species_tensor) = self.calculate_forces_from_neighbors(
            ptcls.pos, ptcls.vel, ptcls.masses, ptcls.id,
            ptcls.rdf_hist, potential.matrix, potential.force,
            self.box_lengths, self.cutoff_radius, flat_neighbors, offsets
        )
        
        # Update displacement tracking
        if self.displacement_check:
            self.update_max_displacement(ptcls.pos)
        
        # Increment timestep counter
        self.timestep_counter += 1

    def pretty_print(self):
        """Print algorithm information and parameters."""
        msg = f"\nINTERACTION SOLVER: Verlet Neighbor Lists\n"
        
        if self.cutoff_radius is not None:
            msg += f"Interaction cutoff: {self.cutoff_radius:.6e}\n"
            msg += f"Skin thickness: {self.skin_thickness:.6e}\n"
            msg += f"Neighbor cutoff: {self.neighbor_cutoff:.6e}\n"
            
        msg += f"Rebuild frequency: {self.rebuild_frequency} timesteps\n"
        msg += f"Max neighbors per particle: {self.max_neighbors_per_particle}\n"
        msg += f"Displacement checking: {'Enabled' if self.displacement_check else 'Disabled'}\n"
        
        if self.neighbor_lists:
            avg_neighbors = sum(len(neighbors) for neighbors in self.neighbor_lists) / len(self.neighbor_lists)
            msg += f"Average neighbors per particle: {avg_neighbors:.1f}\n"
            
        msg += "Computational complexity: O(N) per timestep + O(N²) for rebuilds\n"
        msg += "Suitable for: Dense systems, frequent force evaluations\n"
        
        return msg

    def get_neighbor_statistics(self):
        """
        Get statistics about current neighbor lists.
        
        Returns
        -------
        dict
            Dictionary containing neighbor list statistics
        """
        if not self.neighbor_lists:
            return {'status': 'not_built'}
            
        neighbor_counts = [len(neighbors) for neighbors in self.neighbor_lists]
        
        return {
            'status': 'built',
            'total_particles': len(self.neighbor_lists),
            'total_neighbors': sum(neighbor_counts),
            'avg_neighbors': sum(neighbor_counts) / len(neighbor_counts),
            'min_neighbors': min(neighbor_counts) if neighbor_counts else 0,
            'max_neighbors': max(neighbor_counts) if neighbor_counts else 0,
            'max_displacement': self.max_displacement,
            'timesteps_since_rebuild': self.timestep_counter,
            'memory_usage_mb': self.estimate_memory_usage()
        }

    def estimate_memory_usage(self):
        """
        Estimate memory usage of neighbor lists in MB.
        
        Returns
        -------
        float
            Memory usage in megabytes
        """
        if not self.neighbor_lists:
            return 0.0
            
        total_neighbors = sum(len(neighbors) for neighbors in self.neighbor_lists)
        # Each neighbor is an int32 (4 bytes)
        bytes_per_mb = 1024 * 1024
        return (total_neighbors * 4) / bytes_per_mb