"""
Module for Minimum Image Convention algorithm for particle interactions in periodic systems.
"""

from numba import jit
from numpy import pi, sqrt, zeros, zeros_like

from .base import InteractionSolverBase


class MinimumImage(InteractionSolverBase):
    """
    Minimum Image Convention algorithm for computing particle interactions.

    This algorithm implements the minimum image convention for periodic boundary
    conditions, where each particle interacts only with the nearest periodic image
    of every other particle. The cutoff radius is automatically set to half the
    smallest box dimension to ensure proper minimum image behavior.

    This is essentially a brute force O(N²) algorithm but with proper handling
    of periodic boundary conditions using the minimum image convention.

    Attributes
    ----------
    cutoff : numpy.ndarray
        Cutoff distances set to half box lengths in each dimension
    """

    def __init__(self):
        """Initialize the Minimum Image solver."""
        super().__init__()
        self.type = "minimum_image"
        self.cutoff_radius = None
        self.box_lengths = zeros(3, dtype=float)
        self.dimensions = None 
        self.a_ws = None
        self.units_dict = None

    def setup(self, params, **kwargs):
        """
        Initialize the minimum image solver with simulation parameters.

        The cutoff radius is automatically set to half the box length in each
        dimension to ensure proper minimum image convention behavior.

        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions
        **kwargs : dict
            Additional parameters (unused for minimum image)
        """
        self.box_lengths = params.box_lengths
        # Set cutoff to half box length for proper minimum image convention
        self.cutoff_radius = self.box_lengths.min() * 0.5
        self.dimensions = params.dimensions
        self.total_num_density = params.total_num_density
        self.units_dict = params.units_dict
        self.a_ws = params.a_ws

        

    @staticmethod
    @jit(nopython=True)
    def calculate_rdf_hist(
        pos,
        p_ids,
        pair_index_map,
        cutoff,
        hist_array,
        box_lengths,
    ):
        """
        Calculate PDF histogram using linked cell-list algorithm.

        Parameters
        ----------
        pos : numpy.ndarray
            Particles' positions.
        p_ids : numpy.ndarray
            Species ID of each particle (integer).
        pair_index_map : numpy.ndarray
            Map from (species_i, species_j) to pair index. Shape: (num_species, num_species).
        cutoffs : numpy.ndarray
            Cutoff distances for each coordinate.
        hist_array : numpy.ndarray
            Histogram array with shape (num_pairs, bins_u, bins_v, bins_w).
        box_lengths : numpy.ndarray
            Array of box sides' length.
        
        Returns
        -------
        hist_array : numpy.ndarray
            Updated histogram array.
        """
        # Pre-compute constants for efficiency
        Lh = 0.5 * box_lengths  # Half box lengths for minimum image
        N = pos.shape[0]
        rdf_nbins = hist_array.shape[-1]  # Assuming shape (num_pairs, rdf_bins)
        # RDF parameters - use minimum cutoff for consistency
        dr_rdf = cutoff / float(rdf_nbins)

        # Double loop over all particle pairs
        for i in range(N):
            for j in range(i + 1, N):

                # Calculate relative position
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]

                # Calculate distance
                r_squared = dx * dx + dy * dy + dz * dz
                r_in = sqrt(r_squared)

                # Get species IDs and look up pair index
                sp1_id = p_ids[i]
                sp2_id = p_ids[j]
                pair_idx = pair_index_map[sp1_id, sp2_id]

                # Update RDF histogram
                
                if r_in < cutoff:
                    rdf_bin = int(r_in / dr_rdf)
                    hist_array[pair_idx, rdf_bin] += 1

        return hist_array

    @staticmethod
    @jit(nopython=True)
    def particles_interaction_loop(pos, vel, p_mass, p_id, potential_matrix, force, rdf_hist, box_lengths):
        """
        Compute forces and energies using minimum image convention.

        This method implements the minimum image convention where each particle
        interacts with the nearest periodic image of every other particle.
        The algorithm has O(N²) computational complexity.

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
        potential_matrix: numpy.ndarray
            Potential parameters. Shape (num_species, num_species, num_params).
        force: func
            Potential and force values.
        rdf_hist : numpy.ndarray
            Radial Distribution function array. Shape (nbins, num_species, num_species).
        box_lengths: numpy.ndarray
            Array of box sides' length. Shape (3,).

        Returns
        -------
        ptcl_pot_energy : numpy.ndarray
            Per-particle potential energies. Shape (N,).
        acc_s_r : numpy.ndarray
            Particle accelerations. Shape (N, 3).
        virial_species_tensor : numpy.ndarray
            Virial tensor for each species pair. Shape (3, 3, num_species, num_species).
        j_e : numpy.ndarray
            Heat flux tensor for each species pair. Shape (3, num_species, num_species).

        """
        # Pre-compute constants for efficiency
        Lh = 0.5 * box_lengths  # Half box lengths for minimum image
        N = pos.shape[0]

        # Initialize output arrays
        ptcl_pot_energy = zeros(N)
        acc_s_r = zeros(pos.shape)

        # Virial terms
        virial_xx_sr = zeros(pos.shape[0])
        virial_xy_sr = zeros(pos.shape[0])
        virial_xz_sr = zeros(pos.shape[0])
        virial_yy_sr = zeros(pos.shape[0])
        virial_yz_sr = zeros(pos.shape[0])
        virial_zz_sr = zeros(pos.shape[0])

        # heat current
        j_e = zeros_like(pos)

        # RDF parameters - use minimum cutoff for consistency
        rdf_nbins = rdf_hist.shape[0]
        min_cutoff = Lh[Lh > 0].min()  # Handle zero dimensions properly
        dr_rdf = min_cutoff / float(rdf_nbins)

        # Double loop over all particle pairs
        for i in range(N):
            for j in range(i + 1, N):

                # Calculate relative position
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]

                # Apply minimum image convention using more efficient method
                dx2 = box_lengths[0] - dx * (dx >= Lh[0]) + dx * (dx <= -Lh[0])
                dy2 = box_lengths[1] - dy * (dy >= Lh[1]) + dy * (dy <= -Lh[1])
                dz2 = box_lengths[2] - dz * (dz >= Lh[2]) + dz * (dz <= -Lh[2])


                # Calculate distance
                r_squared = dx2 * dx2 + dy2 * dy2 + dz2 * dz2

                r_in = sqrt(r_squared)

                # Get particle species information
                id_i = p_id[i]
                id_j = p_id[j]

                # Handle potential singularities with short-range cutoff
                p_matrix = potential_matrix[id_i, id_j]
                rs = p_matrix[-1]  # Short-range cutoff to avoid division by zero
                r = r_in * (r_in >= rs) + rs * (r_in < rs)  # Branchless programming

                # Update RDF histogram
                rdf_bin = int(r_in/ dr_rdf)
                if rdf_bin < rdf_nbins:
                    rdf_hist[id_i, id_j, rdf_bin] += 1

                # For minimum image, we compute all pairs (no cutoff check needed)
                # But we can add early termination for very distant pairs
                if r_in < min_cutoff:
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

                    fx = dx * fr
                    fy = dy * fr
                    fz = dz * fr

                    # Update the acceleration for i particles in each dimension
                    acc_s_r[i, 0] += fx / p_mass[i]
                    acc_s_r[i, 1] += fy / p_mass[i]
                    acc_s_r[i, 2] += fz / p_mass[i]

                    # Apply Newton's 3rd law to update acceleration on j particles
                    acc_s_r[j, 0] -= fx / p_mass[j]
                    acc_s_r[j, 1] -= fy / p_mass[j]
                    acc_s_r[j, 2] -= fz / p_mass[j]

                    # Since we have the info already calculate the virial_species_tensor
                    virial_xx_sr[i] += 0.5 * dx * fx
                    virial_xy_sr[i] += 0.5 * dx * fy
                    virial_xz_sr[i] += 0.5 * dx * fz
                    virial_yy_sr[i] += 0.5 * dy * fy
                    virial_yz_sr[i] += 0.5 * dy * fz
                    virial_zz_sr[i] += 0.5 * dz * fz

                    virial_xx_sr[j] += 0.5 * dx * fx
                    virial_xy_sr[j] += 0.5 * dx * fy
                    virial_xz_sr[j] += 0.5 * dx * fz
                    virial_yy_sr[j] += 0.5 * dy * fy
                    virial_yz_sr[j] += 0.5 * dy * fz
                    virial_zz_sr[j] += 0.5 * dz * fz

                    # Heat current
                    vij_x = vel[i, 0] + vel[j, 0]
                    vij_y = vel[i, 1] + vel[j, 1]
                    vij_z = vel[i, 2] + vel[j, 2]
                    fij_vij = vij_x * fx + vij_y * fy + vij_z * fz

                    j_e[i, 0] += 0.25 * (vij_x * pot - dx * fij_vij)
                    j_e[i, 1] += 0.25 * (vij_y * pot - dy * fij_vij)
                    j_e[i, 2] += 0.25 * (vij_z * pot - dz * fij_vij)
                    j_e[j, 0] += 0.25 * (vij_x * pot - dx * fij_vij)
                    j_e[j, 1] += 0.25 * (vij_y * pot - dy * fij_vij)
                    j_e[j, 2] += 0.25 * (vij_z * pot - dz * fij_vij)


        # Add the ideal term of the energy current
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[id_i, id_i, 0] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[id_i, id_i, 1] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[id_i, id_i, 2] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        return (
            ptcl_pot_energy,
            acc_s_r,
            virial_xx_sr,
            virial_yy_sr,
            virial_zz_sr,
            virial_xy_sr,
            virial_xz_sr,
            virial_yz_sr,
            j_e,
        )

    def update(self, ptcls, potential):
        """
        Calculate particle interactions using minimum image convention.

        This method computes all pairwise interactions using the minimum image
        convention for periodic boundary conditions. Each particle interacts
        with the nearest periodic image of every other particle.

        Parameters
        ----------
        ptcls : object
            Particles data containing positions, velocities, masses, etc.
        potential : object
            Potential class containing force functions and parameters

        Notes
        -----
        - Computational complexity: O(N²)
        - Suitable for small to medium system sizes
        - Automatically handles periodic boundary conditions
        - Cutoff is set to half the smallest box dimension
        """
        U_s_r, acc_s_r, virial_xx, virial_yy, virial_zz, virial_xy, virial_xz, virial_yz, j_e = self.particles_interaction_loop(
            ptcls.pos,
            ptcls.vel,
            ptcls.mass,
            ptcls.id,
            potential.matrix,
            potential.force,
            ptcls.rdf_hist,
            self.box_lengths,
        )

        ptcls.potential_energy = U_s_r
        ptcls.acceleration = acc_s_r
        ptcls.virial_xx = virial_xx
        ptcls.virial_xy = virial_xy
        ptcls.virial_xz = virial_xz
        ptcls.virial_yy = virial_yy
        ptcls.virial_yz = virial_yz
        ptcls.virial_zz = virial_zz
        ptcls.heat_flux = j_e

    def pretty_print(self):
        """Print algorithm information and parameters."""
        msg = f"\nINTERACTION SOLVER: Minimum Image Convention\n"

        
        msg += f"Cutoff distances (L/2): {self.cutoff_radius/ self.a_ws:.4f} a_ws = {self.cutoff_radius:.6e} {self.units_dict['length']}\n"

        return msg