"""
Module for Brute Force algorithm for particle interactions.
"""

from numba import jit
from numpy import zeros, zeros_like, sqrt, pi
from .base import InteractionSolverBase


class BruteForce(InteractionSolverBase):
    """
    Brute Force algorithm for computing all pairwise particle interactions.
    
    This algorithm computes interactions between all particle pairs without
    any spatial optimization or cutoff considerations. It provides the most
    straightforward O(N²) implementation and is primarily useful for:
    - Small systems where optimization overhead isn't worth it
    - Reference calculations to validate other algorithms
    - Systems with long-range interactions where cutoffs aren't appropriate
    - Debugging and testing purposes
    
    Unlike minimum image, this algorithm doesn't apply any periodic boundary
    conditions automatically - it computes the direct particle-particle
    interactions based on their actual positions.
    """

    def __init__(self):
        """Initialize the Brute Force solver."""
        super().__init__()
        self.type = 'brute_force'

    def setup(self, params, **kwargs):
        """
        Initialize the brute force solver with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions (for RDF calculations)
        **kwargs : dict
            Additional parameters:
            - apply_cutoff : bool, whether to apply a distance cutoff (default: False)
            - cutoff_radius : float, distance cutoff if apply_cutoff is True
        """
        self.box_lengths = params.box_lengths
    
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
        j_e = zeros((3, potential_matrix.shape[0], potential_matrix.shape[0]))
        virial_species_tensor = zeros((3, 3, potential_matrix.shape[0], potential_matrix.shape[0]))

        # RDF parameters - use minimum cutoff for consistency
        rdf_nbins = rdf_hist.shape[0]
        min_cutoff = Lh[Lh > 0].min()  # Handle zero dimensions properly
        dr_rdf = min_cutoff / float(rdf_nbins)

        # Double loop over all particle pairs
        for i in range(N):
            for j in range(i + 1, N):
                
                # Calculate relative velocity
                vx = vel[i, 0] + vel[j, 0]
                vy = vel[i, 1] + vel[j, 1]
                vz = vel[i, 2] + vel[j, 2]


                # Calculate relative position
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]


                # Calculate distance
                r_squared = dx * dx + dy * dy + dz * dz
                r_in = sqrt(r_squared)

                # Get particle species information
                id_i = p_id[i]
                id_j = p_id[j]

                # Handle potential singularities with short-range cutoff
                p_matrix = potential_matrix[id_i, id_j]
                rs = p_matrix[-1]  # Short-range cutoff to avoid division by zero
                r = r_in * (r_in >= rs) + rs * (r_in < rs)  # Branchless programming

                # Update RDF histogram
                rdf_bin = int(r / dr_rdf)
                if rdf_bin < rdf_nbins:
                    rdf_hist[rdf_bin, id_i, id_j] += 1

                # For minimum image, we compute all pairs (no cutoff check needed)
                # But we can add early termination for very distant pairs
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

    def update(self, ptcls, potential):
        """
        Calculate particle interactions using brute force algorithm.

        This method computes all pairwise interactions without any spatial
        optimization. It's most suitable for small systems or as a reference
        for validating other algorithms.

        Parameters
        ----------
        ptcls : object
            Particles data containing positions, velocities, masses, etc.
        potential : object
            Potential class containing force functions and parameters

        Notes
        -----
        - Computational complexity: O(N²)
        - No spatial optimization or neighbor lists
        - Suitable for small systems or reference calculations
        - Can optionally apply distance cutoff for efficiency
        """
        # Compute interactions
        (ptcls.potential_energy, 
         ptcls.acc, 
         ptcls.virial_species_tensor, 
         ptcls.heat_flux_species_tensor) = self.particles_interaction_loop(
            ptcls.pos,
            ptcls.vel,
            ptcls.masses,
            ptcls.id,
            potential.matrix,
            potential.force,
            ptcls.rdf_hist,
            self.box_lengths
        )

    def pretty_print(self):
        """Print algorithm information and parameters."""
        msg = f"\nINTERACTION SOLVER: Brute Force\n"
        
        if self.box_lengths is not None:
            msg += f"Box lengths: {self.box_lengths}\n"
                 
        msg += "Computational complexity: O(N²)\n"
        msg += "Spatial optimization: None\n"
        msg += "Suitable for: Small systems, reference calculations, debugging\n"
        
        return msg
