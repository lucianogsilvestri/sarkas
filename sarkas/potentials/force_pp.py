"""
Module for handling Particle-Particle interaction.
"""

from numba import jit
from numba.core.types import float64, int64, Tuple
from numpy import arange, sqrt, zeros, zeros_like


@jit(nopython=True)
def update_0D(pos, vel, p_id, p_mass, box_lengths, rc, potential_matrix, force, measure, rdf_hist):
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

    p_mass: numpy.ndarray
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

    virial_species_tensor : numpy.ndarray
        Virial term of each particle. \n
        Shape = (3, 3, pos.shape[0])

    """
    # L = Lv[0]
    actual_dimensions = len(box_lengths.nonzero()[0])
    Lh = 0.5 * box_lengths
    N = pos.shape[0]  # Number of particles

    ptcl_pot_energy = zeros(N)  # Short-ranges potential energy of each particle
    acc_s_r = zeros(pos.shape)  # Vector of accelerations

    # Energy current
    j_e = zeros((N, 3))
    # Virial term for the viscosity calculation
    virial_species_tensor = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3, 3))

    rdf_nbins = rdf_hist.shape[-1]
    dr_rdf = Lh[:actual_dimensions].prod() ** (1.0 / actual_dimensions) / float(rdf_nbins)

    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]

            vx = vel[i, 0] + vel[j, 0]
            vy = vel[i, 1] + vel[j, 1]
            vz = vel[i, 2] + vel[j, 2]

            dx2 = box_lengths[0] - dx * (dx >= Lh[0]) + dx * (dx <= -Lh[0])
            dy2 = box_lengths[1] - dy * (dy >= Lh[1]) + dy * (dy <= -Lh[1])
            dz2 = box_lengths[2] - dz * (dz >= Lh[2]) + dz * (dz <= -Lh[2])

            # if dy >= Lh[1]:
            #     dy = Lv[1] - dy
            # elif dy <= -Lh:
            #     dy = Lv[1] + dy
            #
            # if dz >= Lh[2]:
            #     dz = Lv[2] - dz
            # elif dz <= -Lh:
            #     dz = Lv[2] + dz

            # Compute distance between particles i and j
            r = sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2)
            rdf_bin = int(r / dr_rdf)
            id_i = p_id[i]
            id_j = p_id[j]
            # These definitions are needed due to numba
            # see https://github.com/numba/numba/issues/5881
            if measure and rdf_bin < rdf_nbins:
                rdf_hist[id_j, id_j, rdf_bin] += 1

            if 0.0 < r < rc:
                mass_i = p_mass[i]
                mass_j = p_mass[j]

                p_matrix = potential_matrix[id_i, id_j]
                # Compute the short-ranged force
                pot, fr = force(r, p_matrix)
                fr /= r

                # Update the acceleration for i particles in each dimension
                acc_ix = dx2 * fr / mass_i
                acc_iy = dy2 * fr / mass_i
                acc_iz = dz2 * fr / mass_i

                acc_jx = dx2 * fr / mass_j
                acc_jy = dy2 * fr / mass_j
                acc_jz = dz2 * fr / mass_j

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
                virial_species_tensor[id_i, id_j, 0, 0] += dx2 * dx2 * fr
                virial_species_tensor[id_i, id_j, 0, 1] += dx2 * dy2 * fr
                virial_species_tensor[id_i, id_j, 0, 2] += dx2 * dz2 * fr
                virial_species_tensor[id_i, id_j, 1, 0] += dy2 * dx2 * fr
                virial_species_tensor[id_i, id_j, 1, 1] += dy2 * dy2 * fr
                virial_species_tensor[id_i, id_j, 1, 2] += dy2 * dz2 * fr
                virial_species_tensor[id_i, id_j, 2, 0] += dz2 * dx2 * fr
                virial_species_tensor[id_i, id_j, 2, 1] += dz2 * dy2 * fr
                virial_species_tensor[id_i, id_j, 2, 2] += dz2 * dz2 * fr

                virial_species_tensor[id_j, id_i, 0, 0] += dx2 * dx2 * fr
                virial_species_tensor[id_j, id_i, 0, 1] += dx2 * dy2 * fr
                virial_species_tensor[id_j, id_i, 0, 2] += dx2 * dz2 * fr
                virial_species_tensor[id_j, id_i, 1, 0] += dy2 * dx2 * fr
                virial_species_tensor[id_j, id_i, 1, 1] += dy2 * dy2 * fr
                virial_species_tensor[id_j, id_i, 1, 2] += dy2 * dz2 * fr
                virial_species_tensor[id_j, id_i, 2, 0] += dz2 * dx2 * fr
                virial_species_tensor[id_j, id_i, 2, 1] += dz2 * dy2 * fr
                virial_species_tensor[id_j, id_i, 2, 2] += dz2 * dz2 * fr
                fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                j_e[id_i, id_j, 0] += 0.25 * (vx * pot + dx * fij_vij)
                j_e[id_i, id_j, 1] += 0.25 * (vy * pot + dy * fij_vij)
                j_e[id_i, id_j, 2] += 0.25 * (vz * pot + dz * fij_vij)

                j_e[id_j, id_i, 0] += 0.25 * (vx * pot + dx * fij_vij)
                j_e[id_j, id_i, 1] += 0.25 * (vy * pot + dy * fij_vij)
                j_e[id_j, id_i, 2] += 0.25 * (vz * pot + dz * fij_vij)

    # Add the first term of the energy current
    for i in range(pos.shape[0]):
        j_e[id_i, id_i, 0] += (0.5 * p_mass[i] * (vel[i, :] ** 2).sum(axis=-1) + ptcl_pot_energy[i]) * vel[i, 0]
        j_e[id_i, id_i, 1] += (0.5 * p_mass[i] * (vel[i, :] ** 2).sum(axis=-1) + ptcl_pot_energy[i]) * vel[i, 1]
        j_e[id_i, id_i, 2] += (0.5 * p_mass[i] * (vel[i, :] ** 2).sum(axis=-1) + ptcl_pot_energy[i]) * vel[i, 2]

    return ptcl_pot_energy, acc_s_r, virial_species_tensor, j_e


@jit(nopython=True)
def update(pos, vel, p_id, p_mass, box_lengths, rc, potential_matrix, force, measure, rdf_hist):
    """
    Update the force on the particles based on a linked cell-list (LCL) algorithm.

    Parameters
    ----------
    pos: numpy.ndarray
        Particles' positions.

    vel: numpy.ndarray
        Velocity of each particle.

    p_id: numpy.ndarray
        Id of each particle

    force: func
        Potential and force values.

    potential_matrix: numpy.ndarray
        Potential parameters.

    rc: float
        Cut-off radius.

    box_lengths: numpy.ndarray
        Array of box sides' length.

    p_mass: numpy.ndarray
        Mass of each particle.


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

    virial_species_tensor : numpy.ndarray
        Virial term of each particle. \n
        Shape = (3, 3, pos.shape[0])

    """

    # Total number of cells in volume
    cells_per_dim, cell_lengths = create_cells_array(box_lengths, rc)

    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells_per_dim)

    U_s_r, acc_s_r, virial_species_tensor, heat_flux_species_tensor = particles_interaction_loop(
        pos, vel, p_mass, p_id, potential_matrix, rc, measure, force, rdf_hist, head, ls_array, cells_per_dim, box_lengths
    )

    return U_s_r, acc_s_r, virial_species_tensor, heat_flux_species_tensor


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


@jit(Tuple((int64[:], float64[:]))(float64[:], float64), nopython=True)
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


@jit(nopython=True)
def calculate_virial(pos, p_id, box_lengths, rc, potential_matrix, force):
    """
    Update the force on the particles based on a linked cell-list (LCL) algorithm.

    Parameters
    ----------
    force: tuple, float
        Potential and force values.

    potential_matrix: array
        Potential parameters.

    rc: float
        Cut-off radius.

    box_lengths: array
        Array of box sides' length.

    p_id: array
        Id of each particle

    pos: array
        Particles' positions.

    Returns
    -------

    """
    # Declare parameters
    
    rshift = zeros(3)  # Shifts for array flattening
    
    # Virial term for the viscosity calculation
    virial_species_tensor = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3, 3))

    d3_min = min(cells_per_dim[2], 1)
    d3_max = max(cells_per_dim[2], 1)
    d2_min = min(cells_per_dim[1], 1)
    d2_max = max(cells_per_dim[1], 1)
    d1_min = min(cells_per_dim[0], 1)
    d1_max = max(cells_per_dim[0], 1)

    # Total number of cells in volume
    cells_per_dim, cell_lengths = create_cells_array(box_lengths, rc)

    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells_per_dim)

    # Loop over all cells in x, y, and z direction
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
                                    
                                        # Compute distance between particles i and j
                                        r = sqrt(dx**2 + dy**2 + dz**2)

                                        # These definitions are needed due to numba
                                        # see https://github.com/numba/numba/issues/5881
                                        id_i = p_id[i]
                                        id_j = p_id[j]

                                        # If below the cutoff radius, compute the force
                                        if r < rc:
                                            p_matrix = potential_matrix[id_i, id_j]
                                            # neighbors[i, j] = j

                                            # Compute the short-ranged force
                                            _, fr = force(r, p_matrix)
                                            fr /= r
                                            
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

                                    # Move down list (ls) of particles for cell interactions with a head particle
                                    j = ls_array[j]

                                # Check if head particle interacts with other cells
                                i = ls_array[i]
    return virial_species_tensor


@jit(nopython=True)
def calculate_heat_flux(pos, vel, p_mass, p_id, box_lengths, rc, potential_matrix, force):
    """
    Update the force on the particles based on a linked cell-list (LCL) algorithm.

    Parameters
    ----------
    pos: numpy.ndarray
        Particles' positions. Shape = ((N,3)).

    vel: array
        Velocity of each particle. Shape = ((N,3)).

    p_id: numpy.ndarray
        Id of each particle

    box_lengths: array
        Array of box sides' length.

    rc: float
        Cut-off radius.

    potential_matrix: array
        Potential parameters.

    force: tuple, float
        Potential and force values.

    Returns
    -------
    j_e : numpy.ndarray
        Part of the energy current. Shape = ((3, N))

    """
    # Declare parameters
    rshift = zeros(3)  # Shifts for array flattening

    # energy current
    j_e = zeros((potential_matrix.shape[0], potential_matrix.shape[0], 3))

    # Initialize
    ptcl_pot_energy = zeros(pos.shape[0])  # Short-ranges potential energy of each particle
    
    # Total number of cells in volume
    cells_per_dim, cell_lengths = create_cells_array(box_lengths, rc)

    head, ls_array = create_head_list_arrays(pos, cell_lengths, cells_per_dim)

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

                                        # These definitions are needed due to numba
                                        # see https://github.com/numba/numba/issues/5881
                                        id_i = p_id[i]
                                        id_j = p_id[j]

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


                                            # Since we have the info already calculate the virial_species_tensor
                                            # This factor is to avoid double counting in the case of same species

                                            fij_vij = dx * fr * vx + dy * fr * vy + dz * fr * vz

                                            # For this further factor of 1/2 see eq.(5) in https://doi.org/10.1016/j.cpc.2013.01.008
                                            factor = 0.25

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

    return j_e
