"""
Module for linked cell list algorithm for efficient particle interaction computation.
"""

from numba import jit
from numba.core.types import float64, int64
from numpy import arange, arccos, atan2, sqrt, zeros, zeros_like, array2string, pi
from .base import InteractionSolverBase


class LinkedCellList(InteractionSolverBase):
    """
    Linked cell list algorithm for computing short-range particle interactions.
    
    This algorithm divides the simulation box into cells with size roughly equal
    to the cutoff radius, allowing for efficient O(N) neighbor finding instead
    of O(N²) brute force searches.
    
    Attributes
    ----------
    cutoff_radius : float
        Short-range interaction cutoff radius
    cells_per_dim : numpy.ndarray
        Number of cells in each spatial dimension
    cell_length_per_dim : numpy.ndarray
        Length of cells in each spatial dimension
    """

    def __init__(self):
        """Initialize the linked cell list solver."""
        super().__init__()
        self.type = 'linked_cell_list'
        self.cutoff_radius = None
        self.cells_per_dim = zeros(3, dtype=int)
        self.cell_length_per_dim = zeros(3, dtype=float)

    def setup(self, params, **kwargs):
        """
        Initialize the cell list with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions
            - cutoff_radius : float, interaction cutoff
            - dimensions : int, spatial dimensions
            - total_num_density : float, particle number density
            - units_dict : dict, unit system
            - a_ws : float, Wigner-Seitz radius
        **kwargs : dict
            Additional parameters (unused for cell list)
        """
        self.box_lengths = params.box_lengths
        self.cutoff_radius = params.cutoff_radius
        self.dimensions = params.dimensions
        self.total_num_density = params.total_num_density
        self.units_dict = params.units_dict
        self.a_ws = params.a_ws
        self.create_cells_array()

    @staticmethod
    @jit(nopython=True)
    def calculate_pdf_hist(
        pos, p_names, cutoffs, pdf_bins, hist_dict, key_string, head, ls_array, cells_per_dim, box_lengths, coord_system='cartesian'
    ):
        """
        Update the force on the particles based on a linked cell-list (LCL) algorithm.

        Parameters
        ----------
        pos: numpy.ndarray
            Particles' positions.

        vel: numpy.ndarray
            Particles' positions.

        p_mass: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        potential_matrix: numpy.ndarray
            Potential parameters.

        rc: float
            Cut-off radius.

        measure : bool
            Boolean for rdf calculation.

        force: func
            Potential and force values.

        rdf_hist : numpy.ndarray
            Radial Distribution function array.

        head: numpy.ndarray
            Head array of the linked cell list algorithm.

        ls_array: numpy.ndarray
            List array of the linked cell list algorithm.

        cells_per_dim: numpy.ndarray
            Number of cells per dimension.

        box_lengths: numpy.ndarray
            Array of box sides' length.

        Returns
        -------
        ptcl_pot_energy : numpy.ndarray
            Short-ranged component of the potential energy of each particle. Shape = `tot_num_ptcls`.

        acc_s_r : numpy.ndarray
            Short-ranged component of the acceleration for the particles.

        virial_species_tensor : numpy.ndarray
            Virial term of each particle. \n
            Shape = (3, 3, pos.shape[0])

        j_e : numpy.ndarray
            Energy current of each particle. Shape=((3,N)

        Notes
        -----
        Here the "short-ranged component" refers to the Ewald decomposition of the
        short and long ranged interactions. See the wikipedia article:
        https://en.wikipedia.org/wiki/Ewald_summation or
        "Computer Simulation of Liquids by Allen and Tildesley" for more information.

        """

        # Declare parameters
        rshift = zeros(3)  # Shifts for array flattening
        
        delta_u = cutoffs[0] / float(pdf_bins[0])
        delta_v = cutoffs[1] / float(pdf_bins[1])
        delta_w = cutoffs[2] / float(pdf_bins[2])

        d3_min = min(cells_per_dim[2], 1)
        d3_max = max(cells_per_dim[2], 1)
        d2_min = min(cells_per_dim[1], 1)
        d2_max = max(cells_per_dim[1], 1)
        d1_min = min(cells_per_dim[0], 1)
        d1_max = max(cells_per_dim[0], 1)

        # Dev Note: the array neighbors should be used for testing. This array is used to see if all the particles interact
        # with each other. The array is a NxN matrix initialized to empty. If two particles interact (r < rc) then the
        # matrix element (p1, p2) will be updated with p2. You can use the same array for checking if the loops go over
        # every particle in the case of small rc. If two particle see each other than the p1,p2 position is updated to -1.
        # neighbors = zeros((N, N), dtype=int64)
        # neighbors.fill(-50)


        # Loop over all cells in x, y, and z direction
        for cz in range(d3_max):
            for cy in range(d2_max):
                for cx in range(d1_max):
                    # Compute the cell in 3D volume
                    c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]

                    # Loop over all cell pairs (N-1 and N+1)
                    for cz_N in range(cz - 1, (cz + 2) * d3_min):
                        # if d3_min = 0 -> range( -1, 0). This ensures that when the z-dimension is 0 we only loop once here

                        # z cells
                        # Check periodicity: needed for 0th cell
                        # if cz_N < 0:
                        #     cz_shift = cells_per_dim[2]
                        #     rshift[2] = -box_lengths[2]
                        # # Check periodicity: needed for Nth cell
                        # elif cz_N >= cells_per_dim[2]:
                        #     cz_shift = -cells_per_dim[2]
                        #     rshift[2] = box_lengths[2]
                        # else:
                        #     cz_shift = 0
                        #     rshift[2] = 0.0
                        cz_shift = 0 + d3_max * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                        rshift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2] * (cz_N >= cells_per_dim[2])
                        # Note: In lower dimension systems (2D, 1D)
                        # cz_shift will be 1, 0, -1. This will cancel later on when cz_N + cz_shift = (-1 + 1, 0 + 0, 1 - 1)
                        # Similarly rshift[2] = 0.0 in all cases since box_lengths[2] == 0

                        for cy_N in range(cy - 1, (cy + 2) * d2_min):
                            # y cells
                            # Check periodicity
                            # if cy_N < 0:
                            #     cy_shift = cells_per_dim[1]
                            #     rshift[1] = -box_lengths[1]
                            # elif cy_N >= cells_per_dim[1]:
                            #     cy_shift = -cells_per_dim[1]
                            #     rshift[1] = box_lengths[1]
                            # else:
                            #     cy_shift = 0
                            #     rshift[1] = 0.0

                            cy_shift = 0 + d2_max * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                            rshift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim[1])

                            for cx_N in range(cx - 1, (cx + 2) * d1_min):
                                # x cells
                                # Check periodicity
                                # if cx_N < 0:
                                #     cx_shift = cells_per_dim[0]
                                #     rshift[0] = -box_lengths[0]
                                # elif cx_N >= cells_per_dim[0]:
                                #     cx_shift = -cells_per_dim[0]
                                #     rshift[0] = box_lengths[0]
                                # else:
                                #     cx_shift = 0
                                #     rshift[0] = 0.0

                                cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                                rshift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim[0])

                                # Compute the location of the N-th cell based on shifts
                                c_N = (
                                    (cx_N + cx_shift)
                                    + (cy_N + cy_shift) * cells_per_dim[0]
                                    + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                                )

                                i = head[c]
                                # print(cx_N, cy_N, cz_N, "head cell", c, "p1", i)
                                # First compute interaction of head particle with neighboring cell head particles
                                # Then compute interactions of head particle within a specific cell
                                while i >= 0:

                                    # Check neighboring head particle interactions
                                    j = head[c_N]

                                    while j >= 0:
                                        # print("cell", c, "p1", i, "cell", c_N, "p2", j)

                                        # Only compute particles beyond i-th particle (Newton's 3rd Law)
                                        if i < j:
                                            # neighbors[i, j] = -1
                                            # print("         rshift", rshift)

                                            # Compute the difference in positions for the i-th and j-th particles
                                            dx = pos[i, 0] - (pos[j, 0] + rshift[0])
                                            dy = pos[i, 1] - (pos[j, 1] + rshift[1])
                                            dz = pos[i, 2] - (pos[j, 2] + rshift[2])

                                            if coord_system == 'cylindrical':
                                                # Convert to cylindrical coordinates
                                                du = sqrt(dx**2 + dy**2)
                                                dv = atan2(dy, dx)
                                                dw = abs(dz)
                                            elif coord_system == 'spherical':
                                                # Convert to spherical coordinates
                                                du = sqrt(dx**2 + dy**2 + dz**2)
                                                dv = arccos(dz / du)  # angle with respect to z-axis
                                                dw = atan2(dy, dx)
                                            else:
                                                # Cartesian coordinates
                                                du = abs(dx)
                                                dv = abs(dy)
                                                dw = abs(dz)

                                            # Calculate the bin indices
                                            u_bin = int(du / delta_u)
                                            v_bin = int(dv / delta_v)
                                            w_bin = int(dw / delta_w)

                                            sp1 = p_names[i]
                                            sp2 = p_names[j]
                                            key = f"{sp1}-{sp2}" + key_string
                                            # These definitions are needed due to numba
                                            # see https://github.com/numba/numba/issues/5881

                                            hist_dict[key][u_bin, v_bin, w_bin] += (u_bin < pdf_bins[0]) * (v_bin < pdf_bins[1]) * (w_bin < pdf_bins[2])

                                        # Move down list (ls) of particles for cell interactions with a head particle
                                        j = ls_array[j]

                                    # Move to next particle in cell
                                    i = ls_array[i]

        return hist_dict
    
    @jit(nopython=True)
    def particles_interaction_loop(
        pos, vel, p_mass, p_id, potential_matrix, rc, measure, force, rdf_hist, head, ls_array, cells_per_dim, box_lengths
    ):
        """
        Update the force on the particles based on a linked cell-list (LCL) algorithm.

        Parameters
        ----------
        pos: numpy.ndarray
            Particles' positions.

        vel: numpy.ndarray
            Particles' positions.

        p_mass: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        potential_matrix: numpy.ndarray
            Potential parameters.

        rc: float
            Cut-off radius.

        measure : bool
            Boolean for rdf calculation.

        force: func
            Potential and force values.

        rdf_hist : numpy.ndarray
            Radial Distribution function array.

        head: numpy.ndarray
            Head array of the linked cell list algorithm.

        ls_array: numpy.ndarray
            List array of the linked cell list algorithm.

        cells_per_dim: numpy.ndarray
            Number of cells per dimension.

        box_lengths: numpy.ndarray
            Array of box sides' length.

        Returns
        -------
        ptcl_pot_energy : numpy.ndarray
            Short-ranged component of the potential energy of each particle. Shape = `tot_num_ptcls`.

        acc_s_r : numpy.ndarray
            Short-ranged component of the acceleration for the particles.

        virial_species_tensor : numpy.ndarray
            Virial term of each particle. \n
            Shape = (3, 3, pos.shape[0])

        j_e : numpy.ndarray
            Energy current of each particle. Shape=((3,N)

        Notes
        -----
        Here the "short-ranged component" refers to the Ewald decomposition of the
        short and long ranged interactions. See the wikipedia article:
        https://en.wikipedia.org/wiki/Ewald_summation or
        "Computer Simulation of Liquids by Allen and Tildesley" for more information.

        """

        # Declare parameters
        rshift = zeros(3)  # Shifts for array flattening
        acc_s_r = zeros_like(pos)
        # energy current
        j_e = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3))
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3, 3))
        # Initialize
        ptcl_pot_energy = zeros(pos.shape[0])  # Short-ranges potential energy of each particle
        # Pair distribution function

        rdf_nbins = rdf_hist.shape[-1]
        dr_rdf = rc / float(rdf_nbins)

        d3_min = min(cells_per_dim[2], 1)
        d3_max = max(cells_per_dim[2], 1)
        d2_min = min(cells_per_dim[1], 1)
        d2_max = max(cells_per_dim[1], 1)
        d1_min = min(cells_per_dim[0], 1)
        d1_max = max(cells_per_dim[0], 1)

        # Dev Note: the array neighbors should be used for testing. This array is used to see if all the particles interact
        # with each other. The array is a NxN matrix initialized to empty. If two particles interact (r < rc) then the
        # matrix element (p1, p2) will be updated with p2. You can use the same array for checking if the loops go over
        # every particle in the case of small rc. If two particle see each other than the p1,p2 position is updated to -1.
        # neighbors = zeros((N, N), dtype=int64)
        # neighbors.fill(-50)

        # Loop over all cells in x, y, and z direction
        for cz in range(d3_max):
            for cy in range(d2_max):
                for cx in range(d1_max):
                    # Compute the cell in 3D volume
                    c = cx + cy * cells_per_dim[0] + cz * cells_per_dim[0] * cells_per_dim[1]

                    # Loop over all cell pairs (N-1 and N+1)
                    for cz_N in range(cz - 1, (cz + 2) * d3_min):
                        # if d3_min = 0 -> range( -1, 0). This ensures that when the z-dimension is 0 we only loop once here

                        # z cells
                        # Check periodicity: needed for 0th cell
                        # if cz_N < 0:
                        #     cz_shift = cells_per_dim[2]
                        #     rshift[2] = -box_lengths[2]
                        # # Check periodicity: needed for Nth cell
                        # elif cz_N >= cells_per_dim[2]:
                        #     cz_shift = -cells_per_dim[2]
                        #     rshift[2] = box_lengths[2]
                        # else:
                        #     cz_shift = 0
                        #     rshift[2] = 0.0
                        cz_shift = 0 + d3_max * (cz_N < 0) - cells_per_dim[2] * (cz_N >= cells_per_dim[2])
                        rshift[2] = 0.0 - box_lengths[2] * (cz_N < 0) + box_lengths[2] * (cz_N >= cells_per_dim[2])
                        # Note: In lower dimension systems (2D, 1D)
                        # cz_shift will be 1, 0, -1. This will cancel later on when cz_N + cz_shift = (-1 + 1, 0 + 0, 1 - 1)
                        # Similarly rshift[2] = 0.0 in all cases since box_lengths[2] == 0

                        for cy_N in range(cy - 1, (cy + 2) * d2_min):
                            # y cells
                            # Check periodicity
                            # if cy_N < 0:
                            #     cy_shift = cells_per_dim[1]
                            #     rshift[1] = -box_lengths[1]
                            # elif cy_N >= cells_per_dim[1]:
                            #     cy_shift = -cells_per_dim[1]
                            #     rshift[1] = box_lengths[1]
                            # else:
                            #     cy_shift = 0
                            #     rshift[1] = 0.0

                            cy_shift = 0 + d2_max * (cy_N < 0) - cells_per_dim[1] * (cy_N >= cells_per_dim[1])
                            rshift[1] = 0.0 - box_lengths[1] * (cy_N < 0) + box_lengths[1] * (cy_N >= cells_per_dim[1])

                            for cx_N in range(cx - 1, (cx + 2) * d1_min):
                                # x cells
                                # Check periodicity
                                # if cx_N < 0:
                                #     cx_shift = cells_per_dim[0]
                                #     rshift[0] = -box_lengths[0]
                                # elif cx_N >= cells_per_dim[0]:
                                #     cx_shift = -cells_per_dim[0]
                                #     rshift[0] = box_lengths[0]
                                # else:
                                #     cx_shift = 0
                                #     rshift[0] = 0.0

                                cx_shift = 0 + cells_per_dim[0] * (cx_N < 0) - cells_per_dim[0] * (cx_N >= cells_per_dim[0])
                                rshift[0] = 0.0 - box_lengths[0] * (cx_N < 0) + box_lengths[0] * (cx_N >= cells_per_dim[0])

                                # Compute the location of the N-th cell based on shifts
                                c_N = (
                                    (cx_N + cx_shift)
                                    + (cy_N + cy_shift) * cells_per_dim[0]
                                    + (cz_N + cz_shift) * cells_per_dim[0] * cells_per_dim[1]
                                )

                                i = head[c]
                                # print(cx_N, cy_N, cz_N, "head cell", c, "p1", i)
                                # First compute interaction of head particle with neighboring cell head particles
                                # Then compute interactions of head particle within a specific cell
                                while i >= 0:

                                    # Check neighboring head particle interactions
                                    j = head[c_N]

                                    while j >= 0:
                                        # print("cell", c, "p1", i, "cell", c_N, "p2", j)

                                        # Only compute particles beyond i-th particle (Newton's 3rd Law)
                                        if i < j:
                                            # neighbors[i, j] = -1
                                            # print("         rshift", rshift)

                                            # Compute the difference in positions for the i-th and j-th particles
                                            dx = pos[i, 0] - (pos[j, 0] + rshift[0])
                                            dy = pos[i, 1] - (pos[j, 1] + rshift[1])
                                            dz = pos[i, 2] - (pos[j, 2] + rshift[2])
                                            # print("         distances", dx, dy, dz)

                                            vx = vel[i, 0] + vel[j, 0]
                                            vy = vel[i, 1] + vel[j, 1]
                                            vz = vel[i, 2] + vel[j, 2]

                                            # Compute distance between particles i and j
                                            r = sqrt(dx**2 + dy**2 + dz**2)
                                            rdf_bin = int(r / dr_rdf)
                                            id_i = p_id[i]
                                            id_j = p_id[j]

                                            # These definitions are needed due to numba
                                            # see https://github.com/numba/numba/issues/5881

                                            # if measure and rdf_bin < rdf_nbins:
                                            rdf_hist[id_i, id_j,rdf_bin] += measure * (rdf_bin < rdf_nbins)

                                            # If below the cutoff radius, compute the force
                                            if r < rc:
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

                                        # Move down list (ls) of particles for cell interactions with a head particle
                                        j = ls_array[j]

                                    # Check if head particle interacts with other cells
                                    i = ls_array[i]
        # Add the ideal term of the energy current
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[id_i, id_i, 0] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[id_i, id_i, 1] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[id_i, id_i, 2] += (0.5 * p_mass[i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e

    @staticmethod
    @jit(nopython=True)
    def create_cells_array(box_lengths, cutoff):
        """
        Calculate the number of cells per dimension and their lengths.

        Parameters
        ----------
        box_lengths: numpy.ndarray
            Length of each box side.

        cutoff: float
            Short range potential cutoff

        Returns
        -------
        cells_per_dim : numpy.ndarray, numba.int32
            No. of cells per dimension. There is only 1 cell for the non-dimension.

        cell_lengths_per_dim: numpy.ndarray, numba.float64
            Length of each cell per dimension.

        """
        # actual_dimensions = len(box_lengths.nonzero()[0])

        cells_per_dim = zeros(3, dtype=int64)

        # The number of cells in each dimension.
        # Note that the branchless programming is to take care of the 1D and 2D case, in which we should have at least 1 cell
        # so that we can enter the loops below
        cells_per_dim[0] = int(box_lengths[0] / cutoff)  # * (box_lengths[0] > 0.0) + 1 * (actual_dimensions < 1)
        cells_per_dim[1] = int(box_lengths[1] / cutoff)  # * (box_lengths[1] > 0.0) + 1 * (actual_dimensions < 2)
        cells_per_dim[2] = int(box_lengths[2] / cutoff)  # * (box_lengths[2] > 0.0) + 1 * (actual_dimensions < 3)

        # Branchless programming to avoid the division by zero later on
        cell_length_per_dim = zeros(3, dtype=float64)
        cell_length_per_dim[0] = box_lengths[0] / (1 * (cells_per_dim[0] == 0) + cells_per_dim[0])  # avoid division by zero
        cell_length_per_dim[1] = box_lengths[1] / (1 * (cells_per_dim[1] == 0) + cells_per_dim[1])  # avoid division by zero
        cell_length_per_dim[2] = box_lengths[2] / (1 * (cells_per_dim[2] == 0) + cells_per_dim[2])  # avoid division by zero

        return cells_per_dim, cell_length_per_dim

    @staticmethod
    @jit(nopython=True)
    def create_head_list_arrays(pos, cell_lengths, cells):
        # Loop over all particles and place them in cells
        ls = arange(pos.shape[0])  # List of particle indices in a given cell
        Ncell = cells[cells > 0].prod()
        head = arange(Ncell)  # List of head particles
        empty = -50  # value for empty list and head arrays
        head.fill(empty)  # Make head list empty until population

        for i in range(pos.shape[0]):
            # Determine what cell, in each direction, the i-th particle is in
            cx = int(pos[i, 0] / (1 * (cell_lengths[0] == 0.0) + cell_lengths[0]))  # X cell, avoid division by zero
            cy = int(pos[i, 1] / (1 * (cell_lengths[1] == 0.0) + cell_lengths[1]))  # Y cell, avoid division by zero
            cz = int(pos[i, 2] / (1 * (cell_lengths[2] == 0.0) + cell_lengths[2]))  # Z cell, avoid division by zero

            # Determine cell in 3D volume for i-th particle
            c = cx + cy * cells[0] + cz * cells[0] * cells[1]

            # List of particle indices occupying a given cell
            ls[i] = head[c]

            # The last particle found to lie in cell c (head particle)
            head[c] = i

        return head, ls

    def pretty_print(self):
        """Print algorithm information and computational parameters."""
        msg = f"\nINTERACTION SOLVER: Linked Cell List\n"
        
        ptcls_in_loop = int(self.total_num_density * (self.dimensions * self.cutoff_radius) ** self.dimensions)
        dim_const = (self.dimensions + 1) / 3.0 * pi
        pp_neighbors = int(self.total_num_density * dim_const * self.cutoff_radius**self.dimensions)

        msg += (
            f"rcut = {self.cutoff_radius / self.a_ws:.4f} a_ws = {self.cutoff_radius:.6e} {self.units_dict['length']}\n"
            f"No. of PP cells per dimension = {array2string(self.cells_per_dim)}\n"
            f"No. of particles in PP loop = {ptcls_in_loop}\n"
            f"No. of PP neighbors per particle = {pp_neighbors}\n"
        )
        
        return msg
    
    def update(self, ptcls, potential):
        """
        Calculate particle interactions using linked cell list algorithm.

        Parameters
        ----------
        ptcls : object
            Particles data containing positions, velocities, masses, etc.
        potential : object
            Potential class containing force functions and parameters
        """
        # Create cell lists
        head, ls_array = self.create_head_list_arrays(
            ptcls.pos, self.cell_length_per_dim, self.cells_per_dim
        )

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
            self.cutoff_radius,
            potential.measure,
            potential.force,
            ptcls.rdf_hist,
            head,
            ls_array,
            self.cells_per_dim,
            self.box_lengths
        )