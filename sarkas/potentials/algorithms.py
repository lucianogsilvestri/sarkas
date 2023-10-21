"""
Module for handling Particle-Particle interaction.
"""

from numba import jit
from numba.core.types import float64, int64, Tuple
from numpy import arange, sqrt, zeros, zeros_like, ndarray, array2string, pi
from fmm3dpy import hfmm3d, lfmm3d


class LinkedCellList:

    def __init__(self):
        self.type = 'linked_cell_list'
        self.box_lengths = None
        self.cutoff_radius = None

        self.cells_per_dim = zeros(3, dtype=int)
        self.cell_length_per_dim = zeros(3, dtype=float)

    def setup(self, params):
        self.box_lengths = params.box_lengths
        self.cutoff_radius = params.cutoff_radius
        self.dimensions = params.dimensions
        self.total_num_density = params.total_num_density
        self.units_dict = params.units_dict
        self.a_ws = params.a_ws
        self.create_cells_array()

    def create_cells_array(self):
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

        # The number of cells in each dimension.
        # Note that the branchless programming is to take care of the 1D and 2D case, in which we should have at least 1 cell
        # so that we can enter the loops below
        self.cells_per_dim[0] = int(self.box_lengths[0] / self.cutoff_radius)  # * (box_lengths[0] > 0.0) + 1 * (actual_dimensions < 1)
        self.cells_per_dim[1] = int(self.box_lengths[1] / self.cutoff_radius)  # * (box_lengths[1] > 0.0) + 1 * (actual_dimensions < 2)
        self.cells_per_dim[2] = int(self.box_lengths[2] / self.cutoff_radius)  # * (box_lengths[2] > 0.0) + 1 * (actual_dimensions < 3)

        # Branchless programming to avoid the division by zero later on
        self.cell_length_per_dim[0] = self.box_lengths[0] / (1 * (self.cells_per_dim[0] == 0) + self.cells_per_dim[0])  # avoid division by zero
        self.cell_length_per_dim[1] = self.box_lengths[1] / (1 * (self.cells_per_dim[1] == 0) + self.cells_per_dim[1])  # avoid division by zero
        self.cell_length_per_dim[2] = self.box_lengths[2] / (1 * (self.cells_per_dim[2] == 0) + self.cells_per_dim[2])  # avoid division by zero

    @staticmethod
    @jit(Tuple((int64[:], int64[:]))(float64[:, :], float64[:], int64[:]), nopython=True)
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

    @staticmethod
    @jit(nopython=True)
    def calculate_lcl(pos, vel, p_id, species_masses, rdf_hist, potential_matrix, force, box_lengths, rc, cells_per_dim, head, ls_array):
        """
        Update the force on the particles based on a linked cell-list (LCL) algorithm.

        Parameters
        ----------
        pos: numpy.ndarray
            Particles' positions.

        vel: numpy.ndarray
            Particles' positions.

        species_masses: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        potential_matrix: numpy.ndarray
            Potential parameters.

        rc: float
            Cut-off radius.

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
            Virial tensor of species pairs. \n
            Shape = ( (3, 3, `number_of_species`, `number_of_species`)))

        j_e : numpy.ndarray
            Heat flux tensor of each species pair. Shape = ( (3, `number_of_species`, `number_of_species`))

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
        # heat flux
        j_e = zeros((3, potential_matrix.shape[0], potential_matrix.shape[0]))
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((3, 3, potential_matrix.shape[0], potential_matrix.shape[0]))
        # Initialize
        ptcl_pot_energy = zeros(pos.shape[0])  # Short-ranges potential energy of each particle
        # Pair distribution function

        rdf_nbins = rdf_hist.shape[0]
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
                                            r_in = sqrt(dx**2 + dy**2 + dz**2)
                                            id_i = p_id[i]
                                            id_j = p_id[j]
                                            rs = potential_matrix[id_i, id_j, -1] # Short-range cutoff to avoid division by zero.
                                            # Avoid division by zero.
                                            r = r_in * (r_in >= rs) + rs * (r_in < rs) # Branchless programming. Note that if rs == 0, then this is useless.

                                            rdf_bin = int(r / dr_rdf)
                                            
                                            # These definitions are needed due to numba
                                            # see https://github.com/numba/numba/issues/5881

                                            if rdf_bin < rdf_nbins:
                                                rdf_hist[rdf_bin, id_i, id_j] += 1

                                            # If below the cutoff radius, compute the force
                                            if r < rc:
                                                p_matrix = potential_matrix[id_i, id_j]
                                                # neighbors[i, j] = j

                                                # Compute the short-ranged force
                                                pot, fr = force(r, p_matrix)
                                                fr /= r
                                                # Need to add the same pot to each particle pair.
                                                ptcl_pot_energy[i] += 0.5 * pot
                                                ptcl_pot_energy[j] += 0.5 * pot

                                                # Update the acceleration for i particles in each dimension

                                                acc_s_r[i, 0] += dx * fr / species_masses[id_i]
                                                acc_s_r[i, 1] += dy * fr / species_masses[id_i]
                                                acc_s_r[i, 2] += dz * fr / species_masses[id_i]

                                                # Apply Newton's 3rd law to update acceleration on j particles
                                                acc_s_r[j, 0] -= dx * fr / species_masses[id_j]
                                                acc_s_r[j, 1] -= dy * fr / species_masses[id_j]
                                                acc_s_r[j, 2] -= dz * fr / species_masses[id_j]

                                                # Since we have the info already calculate the virial_species_tensor
                                                # This factor is to avoid double counting in the case of same species
                                                factor = 0.5  # * (id_i != id_j) + 0.25*( id_i == id_j)
                                                virial_species_tensor[0, 0, id_i, id_j] += factor * dx * dx * fr
                                                virial_species_tensor[0, 1, id_i, id_j] += factor * dx * dy * fr
                                                virial_species_tensor[0, 2, id_i, id_j] += factor * dx * dz * fr
                                                virial_species_tensor[1, 0, id_i, id_j] += factor * dy * dx * fr
                                                virial_species_tensor[1, 1, id_i, id_j] += factor * dy * dy * fr
                                                virial_species_tensor[1, 2, id_i, id_j] += factor * dy * dz * fr
                                                virial_species_tensor[2, 0, id_i, id_j] += factor * dz * dx * fr
                                                virial_species_tensor[2, 1, id_i, id_j] += factor * dz * dy * fr
                                                virial_species_tensor[2, 2, id_i, id_j] += factor * dz * dz * fr
                                                # This is where the double counting could happen.
                                                virial_species_tensor[0, 0, id_j, id_i] += factor * dx * dx * fr
                                                virial_species_tensor[0, 1, id_j, id_i] += factor * dx * dy * fr
                                                virial_species_tensor[0, 2, id_j, id_i] += factor * dx * dz * fr
                                                virial_species_tensor[1, 0, id_j, id_i] += factor * dy * dx * fr
                                                virial_species_tensor[1, 1, id_j, id_i] += factor * dy * dy * fr
                                                virial_species_tensor[1, 2, id_j, id_i] += factor * dy * dz * fr
                                                virial_species_tensor[2, 0, id_j, id_i] += factor * dz * dx * fr
                                                virial_species_tensor[2, 1, id_j, id_i] += factor * dz * dy * fr
                                                virial_species_tensor[2, 2, id_j, id_i] += factor * dz * dz * fr

                                                fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                                                # For this further factor of 1/2 see eq.(5) in https://doi.org/10.1016/j.cpc.2013.01.008
                                                factor *= 0.5

                                                j_e[0, id_i, id_j] += factor * dx * fij_vij
                                                j_e[1, id_i, id_j] += factor * dy * fij_vij
                                                j_e[2, id_i, id_j] += factor * dz * fij_vij

                                                j_e[0, id_j, id_i] += factor * dx * fij_vij
                                                j_e[1, id_j, id_i] += factor * dy * fij_vij
                                                j_e[2, id_j, id_i] += factor * dz * fij_vij

                                        # Move down list (ls) of particles for cell interactions with a head particle
                                        j = ls_array[j]

                                    # Check if head particle interacts with other cells
                                    i = ls_array[i]
        # Add the ideal term of the heat flux
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[0, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[1, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[2, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e
    
    @staticmethod
    @jit(nopython = True)
    def calculate_rdf(pos, p_id, rdf_hist, box_lengths, rc, cells_per_dim, head, ls_array):
        
        # Declare parameters
        rshift = zeros(3)  # Shifts for array flattening

        rdf_nbins = rdf_hist.shape[0]
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

                                            # Compute distance between particles i and j
                                            r = sqrt(dx**2 + dy**2 + dz**2)
                                            id_i = p_id[i]
                                            id_j = p_id[j]
                                            
                                            rdf_bin = int(r / dr_rdf)
                                            
                                            # These definitions are needed due to numba
                                            # see https://github.com/numba/numba/issues/5881

                                            if rdf_bin < rdf_nbins:
                                                rdf_hist[rdf_bin, id_i, id_j] += 1

                                        # Move down list (ls) of particles for cell interactions with a head particle
                                        j = ls_array[j]

                                    # Check if head particle interacts with other cells
                                    i = ls_array[i]

    @staticmethod
    @jit(nopython = True)
    def calculate_virial(pos, vel, p_id, species_masses, rdf_hist, potential_matrix, force, box_lengths, rc, cells_per_dim, head, ls_array):
        """
        Update the force on the particles based on a linked cell-list (LCL) algorithm.

        Parameters
        ----------
        pos: numpy.ndarray
            Particles' positions.

        vel: numpy.ndarray
            Particles' positions.

        species_masses: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        potential_matrix: numpy.ndarray
            Potential parameters.

        rc: float
            Cut-off radius.

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
            Virial tensor of species pairs. \n
            Shape = ( (3, 3, `number_of_species`, `number_of_species`)))

        j_e : numpy.ndarray
            Heat flux tensor of each species pair. Shape = ( (3, `number_of_species`, `number_of_species`))

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
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((3, 3, potential_matrix.shape[0], potential_matrix.shape[0]))
        # Initialize
        ptcl_pot_energy = zeros(pos.shape[0])  # Short-ranges potential energy of each particle
        # Pair distribution function

        rdf_nbins = rdf_hist.shape[0]
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
                                            r_in = sqrt(dx**2 + dy**2 + dz**2)
                                            id_i = p_id[i]
                                            id_j = p_id[j]
                                            rs = potential_matrix[id_i, id_j, -1] # Short-range cutoff to avoid division by zero.
                                            # Avoid division by zero.
                                            r = r_in * (r_in >= rs) + rs * (r_in < rs) # Branchless programming. Note that if rs == 0, then this is useless.

                                            rdf_bin = int(r / dr_rdf)
                                            
                                            # These definitions are needed due to numba
                                            # see https://github.com/numba/numba/issues/5881

                                            if rdf_bin < rdf_nbins:
                                                rdf_hist[rdf_bin, id_i, id_j] += 1

                                            # If below the cutoff radius, compute the force
                                            if r < rc:
                                                p_matrix = potential_matrix[id_i, id_j]
                                                # neighbors[i, j] = j

                                                # Compute the short-ranged force
                                                pot, fr = force(r, p_matrix)
                                                fr /= r
                                                # Need to add the same pot to each particle pair.
                                                ptcl_pot_energy[i] += 0.5 * pot
                                                ptcl_pot_energy[j] += 0.5 * pot

                                                # Update the acceleration for i particles in each dimension

                                                acc_s_r[i, 0] += dx * fr / species_masses[id_i]
                                                acc_s_r[i, 1] += dy * fr / species_masses[id_i]
                                                acc_s_r[i, 2] += dz * fr / species_masses[id_i]

                                                # Apply Newton's 3rd law to update acceleration on j particles
                                                acc_s_r[j, 0] -= dx * fr / species_masses[id_j]
                                                acc_s_r[j, 1] -= dy * fr / species_masses[id_j]
                                                acc_s_r[j, 2] -= dz * fr / species_masses[id_j]

                                                # Since we have the info already calculate the virial_species_tensor
                                                # This factor is to avoid double counting in the case of same species
                                                factor = 0.5  # * (id_i != id_j) + 0.25*( id_i == id_j)
                                                virial_species_tensor[0, 0, id_i, id_j] += factor * dx * dx * fr
                                                virial_species_tensor[0, 1, id_i, id_j] += factor * dx * dy * fr
                                                virial_species_tensor[0, 2, id_i, id_j] += factor * dx * dz * fr
                                                virial_species_tensor[1, 0, id_i, id_j] += factor * dy * dx * fr
                                                virial_species_tensor[1, 1, id_i, id_j] += factor * dy * dy * fr
                                                virial_species_tensor[1, 2, id_i, id_j] += factor * dy * dz * fr
                                                virial_species_tensor[2, 0, id_i, id_j] += factor * dz * dx * fr
                                                virial_species_tensor[2, 1, id_i, id_j] += factor * dz * dy * fr
                                                virial_species_tensor[2, 2, id_i, id_j] += factor * dz * dz * fr
                                                # This is where the double counting could happen.
                                                virial_species_tensor[0, 0, id_j, id_i] += factor * dx * dx * fr
                                                virial_species_tensor[0, 1, id_j, id_i] += factor * dx * dy * fr
                                                virial_species_tensor[0, 2, id_j, id_i] += factor * dx * dz * fr
                                                virial_species_tensor[1, 0, id_j, id_i] += factor * dy * dx * fr
                                                virial_species_tensor[1, 1, id_j, id_i] += factor * dy * dy * fr
                                                virial_species_tensor[1, 2, id_j, id_i] += factor * dy * dz * fr
                                                virial_species_tensor[2, 0, id_j, id_i] += factor * dz * dx * fr
                                                virial_species_tensor[2, 1, id_j, id_i] += factor * dz * dy * fr
                                                virial_species_tensor[2, 2, id_j, id_i] += factor * dz * dz * fr

                                        # Move down list (ls) of particles for cell interactions with a head particle
                                        j = ls_array[j]

                                    # Check if head particle interacts with other cells
                                    i = ls_array[i]

        return ptcl_pot_energy, acc_s_r, virial_species_tensor
    
    def pretty_print(self):
        """Print algorithm information."""

        msg = f"\nALGORITHM: Linked Cell List\n"
        
        ptcls_in_loop = int(self.total_num_density * (self.dimensions * self.cutoff_radius) ** self.dimensions)
        dim_const = (self.dimensions + 1) / 3.0 * pi
        pp_neighbors = int(self.total_num_density * dim_const * self.cutoff_radius**self.dimensions)

        msg += (
            f"rcut = {self.cutoff_radius / self.a_ws:.4f} a_ws = {self.cutoff_radius:.6e} {self.units_dict['length']}\n"
            f"No. of PP cells per dimension = {array2string(self.cells_per_dim)}\n"
            f"No. of particles in PP loop = {ptcls_in_loop}\n"
            f"No. of PP neighbors per particle = {pp_neighbors}\n"
        )

        print(msg)

    def update(self, ptcls, potential):
        """
        Calculate the pp part of the acceleration.

        Parameters
        ----------
        ptcls : :class:`sarkas.particles.Particles`
            Particles data.

        potential: :class:`sarkas.potentials.core.Potential`
            Potential class.
        
        """
        head, ls_array = self.create_head_list_arrays(ptcls.pos, self.cell_length_per_dim, self.cells_per_dim)

        ptcls.potential_energy, ptcls.acc, ptcls.virial_species_tensor, ptcls.heat_flux_species_tensor  = self.calculate_lcl(
            ptcls.pos, 
            ptcls.vel,
            ptcls.id, 
            ptcls.species_masses, 
            ptcls.rdf_hist, 
            potential.matrix, 
            potential.force,  
            self.box_lengths,
            self.cutoff_radius, 
            self.cells_per_dim, 
            head, 
            ls_array
        )

    def update_rdf(self, pos, p_id, rdf_hist):

        head, ls_array = self.create_head_list_arrays(pos, self.cell_length_per_dim, self.cells_per_dim)

        self.calculate_rdf(pos, p_id, rdf_hist, self.box_lengths, self.cutoff_radius, self.cells_per_dim, head, ls_array)

class MinimumImage:

    def __init__(self) -> None:
        self.type = 'minimum_image'
        self.box_lengths = None
        self.cutoff = None
    
    def setup(self, params):
        self.box_lengths = params.box_lengths
        self.cutoff = self.box_lengths * 0.5
    
    @staticmethod
    @jit(nopython=True)
    def calculate_mi(pos, vel, p_id, species_masses, rdf_hist, potential_matrix, force,  box_lengths):
        """
        Updates particles' accelerations when the cutoff radius :math:`r_c` is half the box's length, :math:`r_c = L/2`
        For no sub-cell. All ptcls within :math:`r_c = L/2` participate for force calculation. Cost ~ O(N^2)

        Parameters
        ----------
        force: func
            Potential and force values.

        potential_matrix: numpy.ndarray
            Potential parameters.

        box_lengths: numpy.ndarray
            Array of box sides' length.

        species_masses: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        pos: numpy.ndarray
            Particles' positions.

        measure : bool
            Boolean for rdf calculation.

        rdf_hist : numpy.ndarray
            Radial Distribution function array.

        Returns
        -------
        U_s_r : float
            Short-ranged component of the potential energy of the system.

        acc_s_r : numpy.ndarray
            Short-ranged component of the acceleration for the particles.

        virial : numpy.ndarray
            Virial term of each particle. \n
            Shape = (3, 3, pos.shape[0])

        """
        # L = Lv[0]
        actual_dimensions = len(box_lengths.nonzero()[0])
        Lh = 0.5 * box_lengths
        N = pos.shape[0]  # Number of particles

        ptcl_pot_energy = zeros(N)  # Short-ranges potential energy of each particle
        acc_s_r = zeros(pos.shape)  # Vector of accelerations

        # heat flux
        j_e = zeros((3, potential_matrix.shape[0], potential_matrix.shape[0]))
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((3, 3, potential_matrix.shape[0], potential_matrix.shape[0]))

        rdf_nbins = rdf_hist.shape[0]
        dr_rdf = Lh[:actual_dimensions].prod() ** (1.0 / actual_dimensions) / float(rdf_nbins)

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]

                vx = vel[i, 0] + vel[j, 0]
                vy = vel[i, 1] + vel[j, 1]
                vz = vel[i, 2] + vel[j, 2]

                # Minimum Image
                if dx >= Lh[0]:
                    dx = dx - box_lengths[0]
                elif dx <= -Lh[0]:
                    dx = box_lengths[0] + dx
                
                if dy >= Lh[1]:
                    dy = - box_lengths[1] + dy
                elif dy <= -Lh[1]:
                    dy = box_lengths[1] + dy
                
                if dz >= Lh[2]:
                    dz = -box_lengths[2] + dz
                elif dz <= -Lh[2]:
                    dz = box_lengths[2] + dz

                # Compute distance between particles i and j
                r_in = sqrt(dx * dx + dy * dy + dz * dz)
                
                id_i = p_id[i]
                id_j = p_id[j]

                rs = p_matrix[id_i, id_j, -1] # Short-range cutoff to avoid division by zero.

                # Avoid division by zero.
                r = r_in * (r_in >= rs) + rs * (r_in < rs) # Branchless programming. Note that if rs == 0, then this is useless.

                rdf_bin = int(r / dr_rdf)
                # These definitions are needed due to numba
                # see https://github.com/numba/numba/issues/5881
                rdf_hist[rdf_bin, id_j, id_j] += 1 * (rdf_bin < rdf_nbins)

                if 0.0 < r < Lh:
                    mass_i = species_masses[id_i]
                    mass_j = species_masses[id_j]

                    p_matrix = potential_matrix[id_i, id_j,:]
                    # Compute the short-ranged force
                    pot, fr = force(r, p_matrix)
                    fr /= r

                    # Update the acceleration for i particles in each dimension
                    acc_ix = dx * fr / mass_i
                    acc_iy = dy * fr / mass_i
                    acc_iz = dz * fr / mass_i

                    acc_jx = dx * fr / mass_j
                    acc_jy = dy * fr / mass_j
                    acc_jz = dz * fr / mass_j

                    acc_s_r[i, 0] += acc_ix
                    acc_s_r[i, 1] += acc_iy
                    acc_s_r[i, 2] += acc_iz

                    # Apply Newton's 3rd law to update acceleration on j particles
                    acc_s_r[j, 0] -= acc_jx
                    acc_s_r[j, 1] -= acc_jy
                    acc_s_r[j, 2] -= acc_jz

                    # Need to add the same pot to each particle pair.
                    ptcl_pot_energy[i] += 0.5 * pot
                    ptcl_pot_energy[j] += 0.5 * pot

                    # Since we have the info already calculate the virial_species_tensor
                    # This factor is to avoid double counting in the case of same species
                    factor = 0.5  # * (id_i != id_j) + 0.25*( id_i == id_j)
                    virial_species_tensor[0, 0, id_i, id_j] += factor * dx * dx * fr
                    virial_species_tensor[0, 1, id_i, id_j] += factor * dx * dy * fr
                    virial_species_tensor[0, 2, id_i, id_j] += factor * dx * dz * fr
                    virial_species_tensor[1, 0, id_i, id_j] += factor * dy * dx * fr
                    virial_species_tensor[1, 1, id_i, id_j] += factor * dy * dy * fr
                    virial_species_tensor[1, 2, id_i, id_j] += factor * dy * dz * fr
                    virial_species_tensor[2, 0, id_i, id_j] += factor * dz * dx * fr
                    virial_species_tensor[2, 1, id_i, id_j] += factor * dz * dy * fr
                    virial_species_tensor[2, 2, id_i, id_j] += factor * dz * dz * fr
                    # This is where the double counting could happen.
                    virial_species_tensor[0, 0, id_j, id_i] += factor * dx * dx * fr
                    virial_species_tensor[0, 1, id_j, id_i] += factor * dx * dy * fr
                    virial_species_tensor[0, 2, id_j, id_i] += factor * dx * dz * fr
                    virial_species_tensor[1, 0, id_j, id_i] += factor * dy * dx * fr
                    virial_species_tensor[1, 1, id_j, id_i] += factor * dy * dy * fr
                    virial_species_tensor[1, 2, id_j, id_i] += factor * dy * dz * fr
                    virial_species_tensor[2, 0, id_j, id_i] += factor * dz * dx * fr
                    virial_species_tensor[2, 1, id_j, id_i] += factor * dz * dy * fr
                    virial_species_tensor[2, 2, id_j, id_i] += factor * dz * dz * fr

                    fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                    # For this further factor of 1/2 see eq.(5) in https://doi.org/10.1016/j.cpc.2013.01.008
                    factor *= 0.5

                    j_e[0, id_i, id_j] += factor * dx * fij_vij
                    j_e[1, id_i, id_j] += factor * dy * fij_vij
                    j_e[2, id_i, id_j] += factor * dz * fij_vij

                    j_e[0, id_j, id_i] += factor * dx * fij_vij
                    j_e[1, id_j, id_i] += factor * dy * fij_vij
                    j_e[2, id_j, id_i] += factor * dz * fij_vij

        # Add the ideal term of the heat flux
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[0, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[1, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[2, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e

    def update(self, ptcls, potential):
        """
        Calculate particles' acceleration and potential brutally.

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc, ptcls.virial_species_tensor, ptcls.heat_flux_species_tensor = self.calculate_mi(
            ptcls.pos,
            ptcls.vel,
            ptcls.id,
            ptcls.species_masses,
            ptcls.rdf_hist,
            potential.matrix,
            potential.force,
            self.box_lengths,
        )
        # if self.type != "lj":
        #     # Mie Energy of charged systems
        #     # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
        #     dipole = ptcls.charges @ ptcls.pos
        #     ptcls.total_potential_energy += 2.0 * pi * (dipole**2).sum() / (3.0 * self.box_volume * self.fourpie0)

class BruteForce:

    def __init__(self):
        self.type = 'brute'
        self.box_lengths = None
    
    def setup(self, params):
        self.box_lengths = params.box_lengths

    @staticmethod
    @jit(nopython=True)
    def calculate_brute(pos, vel, p_id, species_masses, rdf_hist, potential_matrix, force, box_lengths):
        """
        Updates particles' accelerations when the cutoff radius :math:`r_c` is half the box's length, :math:`r_c = L/2`
        For no sub-cell. All ptcls within :math:`r_c = L/2` participate for force calculation. Cost ~ O(N^2)

        Parameters
        ----------
        force: func
            Potential and force values.

        potential_matrix: numpy.ndarray
            Potential parameters.

        rc: float
            Cut-off radius.

        box_lengths: numpy.ndarray
            Array of box sides' length.

        species_masses: numpy.ndarray
            Mass of each particle.

        p_id: numpy.ndarray
            Id of each particle

        pos: numpy.ndarray
            Particles' positions.

        measure : bool
            Boolean for rdf calculation.

        rdf_hist : numpy.ndarray
            Radial Distribution function array.

        Returns
        -------
        U_s_r : float
            Short-ranged component of the potential energy of the system.

        acc_s_r : numpy.ndarray
            Short-ranged component of the acceleration for the particles.

        virial : numpy.ndarray
            Virial term of each particle. \n
            Shape = (3, 3, pos.shape[0])

        """
        # L = Lv[0]
        actual_dimensions = len(box_lengths.nonzero()[0])
        N = pos.shape[0]  # Number of particles

        ptcl_pot_energy = zeros(N)  # Short-ranges potential energy of each particle
        acc_s_r = zeros(pos.shape)  # Vector of accelerations

        # heat flux
        j_e = zeros((3, potential_matrix.shape[0], potential_matrix.shape[0]))
        # Virial term for the viscosity calculation
        virial_species_tensor = zeros((3, 3, potential_matrix.shape[0], potential_matrix.shape[0]))

        rdf_nbins = rdf_hist.shape[0]
        dr_rdf = box_lengths[:actual_dimensions].prod() ** (1.0 / actual_dimensions) / float(rdf_nbins)

        for i in range(N):
            for j in range(i + 1, N):
                        
                id_i = p_id[i]
                id_j = p_id[j]
                mass_i = species_masses[id_i]
                mass_j = species_masses[id_j]

                p_matrix = potential_matrix[id_i, id_j]
                
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dz = pos[i, 2] - pos[j, 2]

                vx = vel[i, 0] + vel[j, 0]
                vy = vel[i, 1] + vel[j, 1]
                vz = vel[i, 2] + vel[j, 2]

                # Compute distance between particles i and j
                r_in = sqrt(dx**2 + dy**2 + dz**2)
                
                rs = p_matrix[-1] # Short-range cutoff to avoid division by zero.
                # Avoid division by zero.
                r = r_in * (r_in >= rs) + rs * (r_in < rs) # Branchless programming. Note that if rs == 0, then this is useless.

                rdf_bin = int(r / dr_rdf)

                # These definitions are needed due to numba
                # see https://github.com/numba/numba/issues/5881
            
                if rdf_bin < rdf_nbins:
                    rdf_hist[rdf_bin, id_i, id_j] += 1 * (rdf_bin < rdf_nbins)

                # Compute the short-ranged force
                pot, fr = force(r, p_matrix)
                fr /= r

                # Update the acceleration for i particles in each dimension
                acc_ix = dx * fr / mass_i
                acc_iy = dy * fr / mass_i
                acc_iz = dz * fr / mass_i

                acc_jx = dx * fr / mass_j
                acc_jy = dy * fr / mass_j
                acc_jz = dz * fr / mass_j

                acc_s_r[i, 0] += acc_ix
                acc_s_r[i, 1] += acc_iy
                acc_s_r[i, 2] += acc_iz

                # Apply Newton's 3rd law to update acceleration on j particles
                acc_s_r[j, 0] -= acc_jx
                acc_s_r[j, 1] -= acc_jy
                acc_s_r[j, 2] -= acc_jz

                # Need to add the same pot to each particle pair.
                ptcl_pot_energy[i] += 0.5 * pot
                ptcl_pot_energy[j] += 0.5 * pot

                
                # Since we have the info already calculate the virial_species_tensor
                # This factor is to avoid double counting in the case of same species
                factor = 0.5  # * (id_i != id_j) + 0.25*( id_i == id_j)
                virial_species_tensor[0, 0, id_i, id_j] += factor * dx * dx * fr
                virial_species_tensor[0, 1, id_i, id_j] += factor * dx * dy * fr
                virial_species_tensor[0, 2, id_i, id_j] += factor * dx * dz * fr
                virial_species_tensor[1, 0, id_i, id_j] += factor * dy * dx * fr
                virial_species_tensor[1, 1, id_i, id_j] += factor * dy * dy * fr
                virial_species_tensor[1, 2, id_i, id_j] += factor * dy * dz * fr
                virial_species_tensor[2, 0, id_i, id_j] += factor * dz * dx * fr
                virial_species_tensor[2, 1, id_i, id_j] += factor * dz * dy * fr
                virial_species_tensor[2, 2, id_i, id_j] += factor * dz * dz * fr
                # This is where the double counting could happen.
                virial_species_tensor[0, 0, id_j, id_i] += factor * dx * dx * fr
                virial_species_tensor[0, 1, id_j, id_i] += factor * dx * dy * fr
                virial_species_tensor[0, 2, id_j, id_i] += factor * dx * dz * fr
                virial_species_tensor[1, 0, id_j, id_i] += factor * dy * dx * fr
                virial_species_tensor[1, 1, id_j, id_i] += factor * dy * dy * fr
                virial_species_tensor[1, 2, id_j, id_i] += factor * dy * dz * fr
                virial_species_tensor[2, 0, id_j, id_i] += factor * dz * dx * fr
                virial_species_tensor[2, 1, id_j, id_i] += factor * dz * dy * fr
                virial_species_tensor[2, 2, id_j, id_i] += factor * dz * dz * fr

                fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                # For this further factor of 1/2 see eq.(5) in https://doi.org/10.1016/j.cpc.2013.01.008
                factor *= 0.5

                j_e[0, id_i, id_j] += factor * dx * fij_vij
                j_e[1, id_i, id_j] += factor * dy * fij_vij
                j_e[2, id_i, id_j] += factor * dz * fij_vij

                j_e[0, id_j, id_i] += factor * dx * fij_vij
                j_e[1, id_j, id_i] += factor * dy * fij_vij
                j_e[2, id_j, id_i] += factor * dz * fij_vij

        # Add the ideal term of the heat flux
        for i in range(pos.shape[0]):
            id_i = p_id[i]
            j_e[0, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 0]
            j_e[1, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 1]
            j_e[2, id_i, id_i] += (0.5 * species_masses[id_i] * (vel[i] ** 2).sum() + ptcl_pot_energy[i]) * vel[i, 2]

        return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e

    def update(self, ptcls, potential):
        """
        Calculate particles' acceleration and potential brutally.

        Parameters
        ----------
        ptcls: :class:`sarkas.particles.Particles`
            Particles data.

        """
        ptcls.potential_energy, ptcls.acc, ptcls.virial_species_tensor, ptcls.heat_flux_species_tensor = self.calculate_brute(
            ptcls.pos,
            ptcls.vel,
            ptcls.id,
            ptcls.species_masses,
            ptcls.rdf_hist,
            self.box_lengths,
            potential.matrix,
            potential.force,
        )
        # if self.type != "lj":
        #     # Mie Energy of charged systems
        #     # J-M.Caillol, J Chem Phys 101 6080(1994) https: // doi.org / 10.1063 / 1.468422
        #     dipole = ptcls.charges @ ptcls.pos
        #     ptcls.total_potential_energy += 2.0 * pi * (dipole**2).sum() / (3.0 * self.box_volume * self.fourpie0)

class FastMultipoles:

    def __init__(self) -> None:
        self.type = 'fast_multipoles'
        self.box_lengths = None
        self.precision = 1e-5
    
    def setup(self, params, **kwargs):
        self.box_lengths = params.box_lenghts.copy()
        self.precision = kwargs["precision"]

    def update(self, ptcls, potential):

        if potential.type == "coulomb":
            out_fmm = lfmm3d(eps=self.precision, sources=ptcls.pos.transpose(), charges=ptcls.charges, pg=2)
        else:
            out_fmm = hfmm3d(eps=self.precision, zk=1j / potential.screening_length, sources=ptcls.pos.transpose(),charges=ptcls.charges,pg=2)

        ptcls.potential_energy = ptcls.charges * out_fmm.pot.real / potential.fourpie0
        acc = -(ptcls.charges * out_fmm.grad.real / ptcls.masses) / potential.fourpie0
        ptcls.acc = acc.transpose().copy()