"""
Module for Minimum Image Convention algorithm for particle interactions in periodic systems.
"""

from numba import jit
from numpy import zeros, zeros_like, sqrt, pi
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
        self.type = 'minimum_image'
        self.cutoff = None

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
        self.cutoff = self.box_lengths * 0.5

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

                # Apply minimum image convention using more efficient method
                # Use numpy's sign function equivalent for branchless programming
                dx = dx - box_lengths[0] * ((dx > Lh[0]) - (dx < -Lh[0]))
                dy = dy - box_lengths[1] * ((dy > Lh[1]) - (dy < -Lh[1]))
                dz = dz - box_lengths[2] * ((dz > Lh[2]) - (dz < -Lh[2]))

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
        msg = f"\nINTERACTION SOLVER: Minimum Image Convention\n"
        
        if self.box_lengths is not None:
            msg += f"Box lengths: {self.box_lengths}\n"
            msg += f"Cutoff distances (L/2): {self.cutoff}\n"
            msg += f"Minimum cutoff: {self.cutoff.min():.6e}\n"
            
        msg += "Computational complexity: O(N²)\n"
        msg += "Periodic boundary conditions: Minimum image convention\n"
        msg += "Suitable for: Small to medium systems with periodic boundaries\n"
        
        return msg

    def estimate_computational_cost(self, num_particles):
        """
        Estimate computational cost for minimum image algorithm.
        
        Parameters
        ----------
        num_particles : int
            Number of particles in the system
            
        Returns
        -------
        dict
            Dictionary containing cost estimates:
            - 'pair_evaluations' : number of pair interactions computed
            - 'complexity_factor' : O(N²) scaling factor
            - 'relative_cost' : cost relative to N=1000 system
        """
        pair_evaluations = num_particles * (num_particles - 1) // 2
        complexity_factor = num_particles ** 2
        
        # Relative cost compared to 1000-particle system
        reference_n = 1000
        relative_cost = (num_particles / reference_n) ** 2
        
        return {
            'pair_evaluations': pair_evaluations,
            'complexity_factor': complexity_factor,
            'relative_cost': relative_cost
        }

    def validate_box_dimensions(self):
        """
        Validate that box dimensions are suitable for minimum image convention.
        
        Raises
        ------
        ValueError
            If any box dimension is zero or negative
        Warning
            If box dimensions are very different (highly anisotropic)
        """
        if self.box_lengths is None:
            raise ValueError("Box lengths not set - call setup() first")
            
        if (self.box_lengths <= 0).any():
            raise ValueError("All box dimensions must be positive for minimum image")
            
        # Check for highly anisotropic boxes
        min_length = self.box_lengths.min()
        max_length = self.box_lengths.max()
        anisotropy_ratio = max_length / min_length
        
        if anisotropy_ratio > 10:
            print(f"Warning: Highly anisotropic box (ratio: {anisotropy_ratio:.1f})")
            print("Consider using a different algorithm for better efficiency")

    def get_effective_cutoff(self):
        """
        Get the effective cutoff radius for interactions.
        
        Returns
        -------
        float
            Effective cutoff radius (minimum of half box lengths)
        """
        if self.cutoff is None:
            return None
        return self.cutoff.min()