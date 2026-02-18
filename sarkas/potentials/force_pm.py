"""
Module for handling the Particle-Mesh part of the force and potential calculation.
"""

from numba import float64, int64, jit
from numba.core.types import float64, int64
from numpy import arange, array, exp, mod, pi, rint, sin, sqrt, zeros, zeros_like
from numpy.fft import fftshift, ifftshift

# from pyfftw.builders import fftn, ifftn
from pyfftw import empty_aligned, FFTW


class FFTWObjects:
    """Optimized FFT objects for reuse across timesteps"""

    def __init__(self, mesh_sizes, threads=None):
        self.shape = (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0])
        self.threads = threads

        # Create aligned arrays for optimal SIMD performance
        self.fft_input = empty_aligned(self.shape, dtype=complex)
        self.fft_output = empty_aligned(self.shape, dtype=complex)

        # Prepare kwargs for FFTW creation
        fftw_kwargs = {"flags": ["FFTW_MEASURE"], "axes": (0, 1, 2)}  # FFT over all axes like fftn
        if threads is not None:
            fftw_kwargs["threads"] = threads

        # Create forward FFT object (for charge density)
        self.forward = FFTW(self.fft_input, self.fft_output, direction="FFTW_FORWARD", **fftw_kwargs)

        # Create backward FFT object (for electric fields and potential)
        self.backward = FFTW(self.fft_input, self.fft_output, direction="FFTW_BACKWARD", **fftw_kwargs)

    def forward_transform(self, input_array, output_array):
        """Perform forward FFT with data copy.

        Parameters
        ----------
        input_array : np.ndarray
            Input array containing data in real space. Shape must match the FFTW object.
        output_array : np.ndarray
            Output array to store the result in k-space. Must match the FFTW object shape.

        Returns
        -------
        np.ndarray
            The output array containing the transformed data in k-space.
        """
        self.fft_input[:] = input_array
        result = self.forward()  # result is a VIEW of self.fft_output
        output_array[:] = result  # Copy data INTO user's existing array

    def backward_transform(self, input_array, output_array):
        """Perform backward FFT with data copy.

        Parameters
        ----------
        input_array : np.ndarray
            Input array containing data in k-space. Shape must match the FFTW object.
        output_array : np.ndarray
            Output array to store the result in real space. Must match the FFTW object shape.

        Returns
        -------
        np.ndarray
            The output array containing the transformed data in real space.
        """
        self.fft_input[:] = input_array
        result = self.backward()  # result is a VIEW of self.fft_output
        output_array[:] = result  # Copy data INTO user's existing array


@jit(nopython=True)
def assgnmnt_func(cao, x):
    """
    Calculate the charge assignment function as given in Ref.:cite:`Deserno1998`

    Parameters
    ----------
    cao : int
        Charge assignment order.

    x : float
        Distance to the closest mesh point.

    Returns
    ------
    W : numpy.ndarray
        Charge Assignment Function. Each element is the fraction of the charge on each of the `cao` mesh points
        starting from the far left.

    """
    W = zeros(cao)

    if cao == 1:
        W[0] = 1.0

    elif cao == 2:
        W[0] = 0.5 * (1.0 - 2.0 * x)
        W[1] = 0.5 * (1.0 + 2.0 * x)

    elif cao == 3:
        W[0] = (1.0 - 4.0 * x + 4.0 * x**2) / 8.0
        W[1] = (3.0 - 4.0 * x**2) / 4.0
        W[2] = (1.0 + 4.0 * x + 4.0 * x**2) / 8.0

    elif cao == 4:
        W[0] = (1.0 - 6.0 * x + 12.0 * x**2 - 8.0 * x**3) / 48.0
        W[1] = (23.0 - 30.0 * x - 12.0 * x**2 + 24.0 * x**3) / 48.0
        W[2] = (23.0 + 30.0 * x - 12.0 * x**2 - 24.0 * x**3) / 48.0
        W[3] = (1.0 + 6.0 * x + 12.0 * x**2 + 8.0 * x**3) / 48.0

    elif cao == 5:
        W[0] = (1.0 - 8.0 * x + 24.0 * x**2 - 32.0 * x**3 + 16.0 * x**4) / 384.0
        W[1] = (19.0 - 44.0 * x + 24.0 * x**2 + 16.0 * x**3 - 16.0 * x**4) / 96.0
        W[2] = (115.0 - 120.0 * x**2 + 48.0 * x**4) / 192.0
        W[3] = (19.0 + 44.0 * x + 24.0 * x**2 - 16.0 * x**3 - 16.0 * x**4) / 96.0
        W[4] = (1.0 + 8.0 * x + 24.0 * x**2 + 32.0 * x**3 + 16.0 * x**4) / 384.0

    elif cao == 6:
        W[0] = (1.0 - 10.0 * x + 40.0 * x**2 - 80.0 * x**3 + 80.0 * x**4 - 32.0 * x**5) / 3840.0
        W[1] = (237.0 - 750.0 * x + 840.0 * x**2 - 240.0 * x**3 - 240.0 * x**4 + 160.0 * x**5) / 3840.0
        W[2] = (841.0 - 770.0 * x - 440.0 * x**2 + 560.0 * x**3 + 80.0 * x**4 - 160.0 * x**5) / 1920.0
        W[3] = (841.0 + 770.0 * x - 440.0 * x**2 - 560.0 * x**3 + 80.0 * x**4 + 160.0 * x**5) / 1920.0
        W[4] = (237.0 + 750.0 * x + 840.0 * x**2 + 240.0 * x**3 - 240.0 * x**4 - 160.0 * x**5) / 3840.0
        W[5] = (1.0 + 10.0 * x + 40.0 * x**2 + 80.0 * x**3 + 80.0 * x**4 + 32.0 * x**5) / 3840.0

    elif cao == 7:
        W[0] = (
            1.0 - 12.0 * x + 60.0 * x**2 - 160.0 * x**3 + 240.0 * x**4 - 192.0 * x**5 + 64.0 * x**6
        ) / 46080.0

        W[1] = (
            361.0 - 1416.0 * x + 2220.0 * x**2 - 1600.0 * x**3 + 240.0 * x**4 + 384.0 * x**5 - 192.0 * x**6
        ) / 23040.0

        W[2] = (
            10543.0 - 17340.0 * x + 4740.0 * x**2 + 6880.0 * x**3 - 4080.0 * x**4 - 960.0 * x**5 + 960.0 * x**6
        ) / 46080.0

        W[3] = (5887.0 - 4620.0 * x**2 + 1680.0 * x**4 - 320.0 * x**6) / 11520.0

        W[4] = (
            10543.0 + 17340.0 * x + 4740.0 * x**2 - 6880.0 * x**3 - 4080.0 * x**4 + 960.0 * x**5 + 960.0 * x**6
        ) / 46080.0

        W[5] = (
            361.0 + 1416.0 * x + 2220.0 * x**2 + 1600.0 * x**3 + 240.0 * x**4 - 384.0 * x**5 - 192.0 * x**6
        ) / 23040.0

        W[6] = (
            1.0 + 12.0 * x + 60.0 * x**2 + 160.0 * x**3 + 240.0 * x**4 + 192.0 * x**5 + 64.0 * x**6
        ) / 46080.0

    return W


@jit(nopython=True)
def calc_acc_pm(E_x_r, E_y_r, E_z_r, mesh_pos, mesh_points, q_over_m, cao, mesh_sz, mid, pshift):
    """
    Calculates the long range part of particles' accelerations.

    Parameters
    ----------
    E_x_r : numpy.ndarray
        Electric field along x-axis.

    E_y_r : numpy.ndarray
        Electric field along y-axis.

    E_z_r : numpy.ndarray
        Electric field along z-axis.

    mesh_pos: numpy.ndarray
        Particles' positions relative to the mesh.

    mesh_points: numpy.ndarray
        Particles' positions on the mesh.

    q_over_m : numpy.ndarray
        Particles' charges divided by their masses.

    cao : int
        Charge assignment order.

    mesh_sz: numpy.ndarray
        Mesh points per direction.

    mid: numpy.ndarray
        Midpoint flag for the three directions.

    pshift: numpy.ndarray
        Midpoint shift in each direction.

    Returns
    -------

    acc : numpy.ndarray
          Acceleration from Electric Field.

    """
    E_x_p = zeros_like(q_over_m)
    E_y_p = zeros_like(q_over_m)
    E_z_p = zeros_like(q_over_m)

    acc = zeros_like(mesh_pos)

    for ipart, q_m in enumerate(q_over_m):
        ix = mesh_points[ipart, 0]
        x = mesh_pos[ipart, 0] - (ix + mid[0])

        iy = mesh_points[ipart, 1]
        y = mesh_pos[ipart, 1] - (iy + mid[1])

        iz = mesh_points[ipart, 2]
        z = mesh_pos[ipart, 2] - (iz + mid[2])

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        izn = iz - pshift[2]  # min. index along z-axis

        for g in range(cao[2]):
            #
            # if izn < 0:
            #     r_g = izn + mesh_sz[2]
            # elif izn > (mesh_sz[2] - 1):
            #     r_g = izn - mesh_sz[2]
            # else:
            #     r_g = izn

            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > (mesh_sz[2] - 1))

            iyn = iy - pshift[1]  # min. index along y-axis

            for i in range(cao[1]):
                # if iyn < 0:
                #     r_i = iyn + mesh_sz[1]
                # elif iyn > (mesh_sz[1] - 1):
                #     r_i = iyn - mesh_sz[1]
                # else:
                #     r_i = iyn
                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > (mesh_sz[1] - 1))

                ixn = ix - pshift[0]  # min. index along x-axis

                for j in range(cao[0]):
                    r_j = ixn + mesh_sz[0] * (ixn < 0) - mesh_sz[0] * (ixn > (mesh_sz[0] - 1))

                    # if ixn < 0:
                    #     r_j = ixn + mesh_sz[0]
                    # elif ixn > (mesh_sz[0] - 1):
                    #     r_j = ixn - mesh_sz[0]
                    # else:
                    #     r_j = ixn

                    # q_over_m = charges[ipart] / masses[ipart]
                    E_x_p[ipart] += q_m * E_x_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_y_p[ipart] += q_m * E_y_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]
                    E_z_p[ipart] += q_m * E_z_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]

                    ixn += 1

                iyn += 1

            izn += 1

    acc[:, 0] = E_x_p
    acc[:, 1] = E_y_p
    acc[:, 2] = E_z_p

    return acc


@jit(nopython=True)
def calc_charge_dens(mesh_pos, mesh_points, charges, cao, mesh_sz, mid, pshift):
    """
    Assigns charges to mesh points using periodic boundary conditions.

    Parameters
    ----------
    mesh_pos : numpy.ndarray
        Particles' positions relative to the mesh.
    mesh_points : numpy.ndarray
        Particles' positions on the mesh.
    charges : numpy.ndarray
        Particles' charges.
    cao : numpy.ndarray
        Charge assignment order for each dimension.
    mesh_sz : numpy.ndarray
        Number of mesh points per direction.
    mid : numpy.ndarray
        Midpoint flag for each direction.
    pshift : numpy.ndarray
        Shift to the closest mesh point for each direction.

    Returns
    -------
    rho_r : numpy.ndarray
        Charge density distributed on the mesh.
    """
    rho_r = zeros((mesh_sz[2], mesh_sz[1], mesh_sz[0]), dtype=float64)

    for ipart in range(len(charges)):
        # Determine the left-most mesh index and the offset for each direction.
        ix = mesh_points[ipart, 0]
        iy = mesh_points[ipart, 1]
        iz = mesh_points[ipart, 2]

        delta_x = mesh_pos[ipart, 0] - (ix + mid[0])
        delta_y = mesh_pos[ipart, 1] - (iy + mid[1])
        delta_z = mesh_pos[ipart, 2] - (iz + mid[2])

        # Calculate weights using the charge assignment function.
        wx = assgnmnt_func(cao[0], delta_x)
        wy = assgnmnt_func(cao[1], delta_y)
        wz = assgnmnt_func(cao[2], delta_z)

        # Use modulo (%) for periodic boundary conditions.
        # Calculate starting indices in each dimension.
        base_z = (iz - pshift[2]) % mesh_sz[2]
        base_y = (iy - pshift[1]) % mesh_sz[1]
        base_x = (ix - pshift[0]) % mesh_sz[0]

        for g in range(cao[2]):
            r_g = (base_z + g) % mesh_sz[2]
            for i in range(cao[1]):
                r_i = (base_y + i) % mesh_sz[1]
                for j in range(cao[0]):
                    r_j = (base_x + j) % mesh_sz[0]
                    rho_r[r_g, r_i, r_j] += charges[ipart] * wz[g] * wy[i] * wx[j]

    return rho_r


@jit(nopython=True)
def calc_field(phi_k, kx_v, ky_v, kz_v):
    """
    Numba'd function that calculates the Electric field in Fourier space.

    Parameters
    ----------
    phi_k : numpy.ndarray, numba.complex128
        3D array of the Potential.

    kx_v : numpy.ndarray, numba.float64
        2D array containing the values of kx.

    ky_v : numpy.ndarray, numba.float64
        2D array containing the values of ky.

    kz_v : numpy.ndarray, numba.float64
        3D array containing the values of kz.

    Returns
    -------
    E_kx : numpy.ndarray, numba.complex128
       Electric Field along kx-axis.

    E_ky : numpy.ndarray, numba.complex128
       Electric Field along ky-axis.

    E_kz : numpy.ndarray, numba.complex128
       Electric Field along kz-axis.

    """

    E_kx = -1j * kx_v * phi_k
    E_ky = -1j * ky_v * phi_k
    E_kz = -1j * kz_v * phi_k

    return E_kx, E_ky, E_kz


@jit(nopython=True)
def calc_mesh_coord(pos, h_array, cao):
    """
    Calculate the particles positions with respect to the mesh and their closest point on the mesh.

    Parameters
    ----------
    pos: numpy.ndarray
        Particles' positions.

    h_array: numpy.ndarray
        Width of the mesh cells.

    cao: numpy.ndarray
        Charge assignment order.

    Returns
    -------
    mesh_pos: numpy.ndarray
        Particles' positions relative to the mesh, i.e. pos/h_array.

    mesh_points: numpy.ndarray
        Particles' positions on the mesh.

    """
    # Avoid division by zero. if mesh_sz[i] == 0 then there is no mesh in that direction, h_array = 0
    non_zero_hs = h_array.copy()
    non_zero_hs += 1.0 * (h_array == 0)

    # Calculate the particles' coordinates relative to the mesh
    mesh_pos = pos / non_zero_hs
    # Calculate the particles' closest grid points (if cao is odd) or closest mid points if cao is even
    mesh_points = rint(mesh_pos - 0.5 * (cao % 2 == 0))

    return mesh_pos, mesh_points.astype(int64)


@jit(nopython=True)
def calc_pot_pm(phi_r, mesh_pos, mesh_points, charges, cao, mesh_sz, mid, pshift):
    """
    Calculates the long range part of particles' potential energies.

    Parameters
    ----------
    phi_r : numpy.ndarray
        Potential energy at mesh points.

    mesh_pos: numpy.ndarray
        Particles' positions relative to the mesh.

    mesh_points: numpy.ndarray
        Particles' positions on the mesh.

    charges : numpy.ndarray
        Particles' charges.

    cao : int
        Charge assignment order.

    mesh_sz: numpy.ndarray
        Mesh points per direction.

    mid: numpy.ndarray
        Midpoint flag for the three directions.

    pshift: numpy.ndarray
        Midpoint shift in each direction.

    Returns
    -------
    pot_p : numpy.ndarray
          Potential energy of each particle.

    """
    pot_p = zeros_like(charges)  # potential energy for each particle

    for ipart, q in enumerate(charges):
        ix = mesh_points[ipart, 0]
        x = mesh_pos[ipart, 0] - (ix + mid[0])

        iy = mesh_points[ipart, 1]
        y = mesh_pos[ipart, 1] - (iy + mid[1])

        iz = mesh_points[ipart, 2]
        z = mesh_pos[ipart, 2] - (iz + mid[2])

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        # Use modulo for periodic boundary conditions - consistent with calc_charge_dens
        base_z = (iz - pshift[2]) % mesh_sz[2]
        base_y = (iy - pshift[1]) % mesh_sz[1]
        base_x = (ix - pshift[0]) % mesh_sz[0]

        for g in range(cao[2]):
            r_g = (base_z + g) % mesh_sz[2]

            for i in range(cao[1]):
                r_i = (base_y + i) % mesh_sz[1]

                for j in range(cao[0]):
                    r_j = (base_x + j) % mesh_sz[0]

                    pot_p[ipart] += 0.5 * q * phi_r[r_g, r_i, r_j] * wz[g] * wy[i] * wx[j]

    return pot_p


@jit(nopython=True)
def calc_virial_pm_per_particle(vg0, vg1, vg2, vg3, vg4, vg5, mesh_sz, mesh_pos, mesh_points, charges, cao, mid, pshift):
    """
    Interpolate 6 virial tensor components (in real-space) to each particle.

    Parameters
    ----------
    rho_k : complex ndarray
        Charge density in Fourier space.

    G_k : ndarray
        Green's function in Fourier space.

    vg0..vg5 : 3D ndarrays
        Virial coefficient fields: V_xx, V_yy, ..., V_yz

    Returns
    -------
    virial_p : ndarray (N, 6)
        Per-particle virial tensor components.
    """

    virial_xx = zeros_like(charges)
    virial_yy = zeros_like(charges)
    virial_zz = zeros_like(charges)
    virial_xy = zeros_like(charges)
    virial_xz = zeros_like(charges)
    virial_yz = zeros_like(charges)

    for ipart, q in enumerate(charges):
        ix = mesh_points[ipart, 0]
        x = mesh_pos[ipart, 0] - (ix + mid[0])

        iy = mesh_points[ipart, 1]
        y = mesh_pos[ipart, 1] - (iy + mid[1])

        iz = mesh_points[ipart, 2]
        z = mesh_pos[ipart, 2] - (iz + mid[2])

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        izn = iz - pshift[2]
        for g in range(cao[2]):
            r_g = izn + mesh_sz[2] * (izn < 0) - mesh_sz[2] * (izn > mesh_sz[2] - 1)

            iyn = iy - pshift[1]
            for j_ in range(cao[1]):
                r_i = iyn + mesh_sz[1] * (iyn < 0) - mesh_sz[1] * (iyn > mesh_sz[1] - 1)

                ixn = ix - pshift[0]
                for k_ in range(cao[0]):
                    r_j = ixn + mesh_sz[0] * (ixn < 0) - mesh_sz[0] * (ixn > mesh_sz[0] - 1)

                    weight = wz[g] * wy[j_] * wx[k_]

                    # Virial components
                    virial_xx[ipart] += 0.5 * q * vg0[r_g, r_i, r_j] * weight
                    virial_yy[ipart] += 0.5 * q * vg1[r_g, r_i, r_j] * weight
                    virial_zz[ipart] += 0.5 * q * vg2[r_g, r_i, r_j] * weight
                    virial_xy[ipart] += 0.5 * q * vg3[r_g, r_i, r_j] * weight
                    virial_xz[ipart] += 0.5 * q * vg4[r_g, r_i, r_j] * weight
                    virial_yz[ipart] += 0.5 * q * vg5[r_g, r_i, r_j] * weight

                    ixn += 1
                iyn += 1
            izn += 1

    return virial_xx, virial_yy, virial_zz, virial_xy, virial_xz, virial_yz


@jit(nopython=True)
def create_k_aliases(aliases, mesh_sizes, non_zero_box_lengths):
    """Calculate the alias arrays of the reciprocal space arrays for anti-aliasing.

    Parameters
    ----------
    aliases : numpy.ndarray, numba.int64
        Number of aliases per dimension.

    mesh_sizes : numpy.ndarray, numba.int64
        Number of mesh points in x,y,z.

    non_zero_box_lengths : numpy.ndarray, numba.float64
        Length of simulation's box in each direction. Note that no element should be equal to 0.0.
        If the dimensionality of the problem is lower than 3, then use 1.0 as the box length for those dimensions.
        Example: 2D non_zero_box_lengths = [Lx, Ly, 1.0].

    Returns
    -------
    kx_M : numpy.ndarray
       Array of aliases for each kx value. Shape=( mesh_size[0], 2 * aliases[0] + 1)

    ky_M : numpy.ndarray
       Array of aliases for each ky value. Shape=( mesh_size[1], 2 * aliases[1] + 1)

    kz_M : numpy.ndarray
       Array of aliases for each kz value. Shape=( mesh_size[2], 2 * aliases[2] + 1)

    """

    nz_mid = mesh_sizes[2] / 2 if mod(mesh_sizes[2], 2) == 0 else (mesh_sizes[2] - 1) / 2
    ny_mid = mesh_sizes[1] / 2 if mod(mesh_sizes[1], 2) == 0 else (mesh_sizes[1] - 1) / 2
    nx_mid = mesh_sizes[0] / 2 if mod(mesh_sizes[0], 2) == 0 else (mesh_sizes[0] - 1) / 2

    two_pi = 2.0 * pi

    kx_M = zeros((mesh_sizes[0], 2 * aliases[0] + 1), dtype=float64)
    ky_M = zeros((mesh_sizes[1], 2 * aliases[1] + 1), dtype=float64)
    kz_M = zeros((mesh_sizes[2], 2 * aliases[2] + 1), dtype=float64)

    for nz in range(mesh_sizes[2]):
        nz_sh = nz - nz_mid
        for mz in range(-aliases[2], aliases[2] + 1):
            kz_M[nz, mz + aliases[2]] = two_pi * (nz_sh + mz * mesh_sizes[2]) / non_zero_box_lengths[2]

    for ny in range(mesh_sizes[1]):
        ny_sh = ny - ny_mid
        for my in range(-aliases[1], aliases[1] + 1):
            ky_M[ny, my + aliases[1]] = two_pi * (ny_sh + my * mesh_sizes[1]) / non_zero_box_lengths[1]

    for nx in range(mesh_sizes[0]):
        nx_sh = nx - nx_mid
        for mx in range(-aliases[0], aliases[0] + 1):
            kx_M[nx, mx + aliases[0]] = two_pi * (nx_sh + mx * mesh_sizes[0]) / non_zero_box_lengths[0]

    return kx_M, ky_M, kz_M


@jit(nopython=True)
def create_k_arrays(mesh_sizes, non_zero_box_lengths):
    """Calculate the reciprocal space arrays.

    Parameters
    ----------
    non_zero_box_lengths : numpy.ndarray
        Length of simulation's box in each direction. Note that no element should be equal to 0.0.
        If the dimensionality of the problem is lower than 3, then use 1.0 as the box length for those dimensions.
        Example: 2D non_zero_box_lengths = [Lx, Ly, 1.0].

    mesh_sizes : numpy.ndarray
        Number of mesh points in x,y,z.

    Returns
    -------
    kx_v : numpy.ndarray
       Array of reciprocal space vectors along the x-axis

    ky_v : numpy.ndarray
       Array of reciprocal space vectors along the y-axis

    kz_v : numpy.ndarray
       Array of reciprocal space vectors along the z-axis

    """
    nz_mid = mesh_sizes[2] / 2 if mod(mesh_sizes[2], 2) == 0 else (mesh_sizes[2] - 1) / 2
    ny_mid = mesh_sizes[1] / 2 if mod(mesh_sizes[1], 2) == 0 else (mesh_sizes[1] - 1) / 2
    nx_mid = mesh_sizes[0] / 2 if mod(mesh_sizes[0], 2) == 0 else (mesh_sizes[0] - 1) / 2

    # nx_v = arange(mesh_sizes[0]).reshape((1, mesh_sizes[0]))
    # ny_v = arange(mesh_sizes[1]).reshape((mesh_sizes[1], 1))
    # nz_v = arange(mesh_sizes[2]).reshape((mesh_sizes[2], 1, 1))
    # Dev Note:
    # The above three lines where giving a problem with Numba in Windows only.
    # I replaced them with the ones below. I don't know why it was giving a problem.
    nx_v = zeros((1, mesh_sizes[0]), dtype=int64)
    nx_v[0, :] = arange(mesh_sizes[0])

    ny_v = zeros((mesh_sizes[1], 1), dtype=int64)
    ny_v[:, 0] = arange(mesh_sizes[1])

    nz_v = zeros((mesh_sizes[2], 1, 1), dtype=int64)
    nz_v[:, 0, 0] = arange(mesh_sizes[2])

    two_pi = 2.0 * pi
    kx_v = two_pi * (nx_v - nx_mid) / non_zero_box_lengths[0]
    ky_v = two_pi * (ny_v - ny_mid) / non_zero_box_lengths[1]
    kz_v = two_pi * (nz_v - nz_mid) / non_zero_box_lengths[2]

    return kx_v, ky_v, kz_v


@jit(nopython=True)
def sum_over_aliases(kx, ky, kz, kx_M, ky_M, kz_M, h_array, p, four_pi, alpha_sq, kappa_sq):
    """
    Perform the sum over aliases in each direction.

    Parameters
    ----------
    kx : float
        Value of the k_x wavenumber.

    ky : float
        Value of the k_y wavenumber.

    kz : float
        Value of the k_z wavenumber.

    kx_M : numpy.ndarray
       Array of aliases for each kx value. Shape=( 2 * aliases[0] + 1)

    ky_M : numpy.ndarray
       Array of aliases for each ky value. Shape=(2 * aliases[1] + 1)

    kz_M : numpy.ndarray
       Array of aliases for each kz value. Shape=(2 * aliases[2] + 1)

    h_array : numpy.ndarray
        Mesh spacings.

    p : numpy.ndarray
        Charge assignment order for each direction. i.e. cao_x, cao_y, cao_z

    four_pi: float
        Multiplier constant. :math:`4 \\pi` if cgs units or :math:`4 \\pi \\eplison_0` if mks units.

    alpha_sq: float
        Ewald parameter squared, :math:`\\alpha^2`.

    kappa_sq: float
        Screening parameter squared. It is equal to 0 (zero) in case of Coulomb interaction.

    Returns
    -------
    U_G_k : float
        Product of the Green's function and the FFT of the B-splines squared. i.e. The numerator of eq.(31) in :cite:`Dharuman2017`.

    U_k_sq : float
        Sqared sum of the FFT of the B-spline. i.e. The denominator (without the :math:`|k_n|^2`:) in cite:`Dharuman2017`.

    """
    U_k_sq = 0.0
    U_G_k = 0.0

    # Sum over the aliases
    for mz, kzm in enumerate(kz_M):
        kz_M_arg = 0.5 * kzm * h_array[2]
        kzm_sq = kzm * kzm
        U_kz_M = (sin(kz_M_arg) / kz_M_arg) ** p[2] if kz_M_arg != 0.0 else 1.0

        for my, kym in enumerate(ky_M):
            ky_M_arg = 0.5 * kym * h_array[1]
            kym_sq = kym * kym
            U_ky_M = (sin(ky_M_arg) / ky_M_arg) ** p[1] if ky_M_arg != 0.0 else 1.0

            for mx, kxm in enumerate(kx_M):
                kx_M_arg = 0.5 * kxm * h_array[0]
                kxm_sq = kxm * kxm
                U_kx_M = (sin(kx_M_arg) / kx_M_arg) ** p[0] if kx_M_arg != 0.0 else 1.0

                k_M_sq = kxm_sq + kym_sq + kzm_sq

                U_k_M = U_kx_M * U_ky_M * U_kz_M
                U_k_M_sq = U_k_M * U_k_M

                G_k_M = four_pi * exp(-0.25 * (kappa_sq + k_M_sq) / alpha_sq) / (kappa_sq + k_M_sq)

                k_dot_k_M = kx * kxm + ky * kym + kz * kzm

                U_G_k += U_k_M_sq * G_k_M * k_dot_k_M
                U_k_sq += U_k_M_sq

    return U_G_k, U_k_sq


@jit(nopython=True)
def force_optimized_green_function(box_lengths, h_array, mesh_sizes, aliases, p, constants):
    """
    Calculate the optimized Green's function for the PPPM method.

    This function computes the optimized Green's function used in solving Poisson's equation
    efficiently in Fourier space for the PPPM method.

    Parameters
    ----------
    box_lengths : ndarray
        Length of simulation's box in each direction, shape (3,).
    h_array : ndarray
        Mesh spacings, shape (3,).
    mesh_sizes : ndarray
        Number of mesh points in x, y, z, shape (3,).
    aliases : ndarray
        Number of aliases in each direction, shape (3,).
    p : ndarray
        Charge assignment order (cao) for each dimension, shape (3,).
    constants : ndarray
        Array containing [screening parameter, Ewald parameter, 4πε₀], shape (3,).

    Returns
    -------
    G_k : ndarray
        Optimized Green's function, shape (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]).
    kx_v : ndarray
        Array of kx values, shape (1, mesh_sizes[0]).
    ky_v : ndarray
        Array of ky values, shape (mesh_sizes[1], 1).
    kz_v : ndarray
        Array of kz values, shape (mesh_sizes[2], 1, 1).
    PM_err : float
        Estimated error in the force calculation due to the optimized Green's function.

    Notes
    -----
    The optimized Green's function is derived from the work of Hockney and Eastwood,
    with improvements by Deserno and Holm. It aims to minimize the error in the force
    calculation for a given charge assignment scheme and mesh size.

    The function handles both 3D and lower-dimensional systems. The aliasing sum is
    crucial for improving accuracy, especially for higher-order charge assignment schemes.

    This function is typically called once at the beginning of a simulation to set up
    the Green's function. The PM error estimate can be used to assess the accuracy of
    the method for given parameters.

    References
    ----------
    .. [1] Hockney, R. W., & Eastwood, J. W. (1988). Computer simulation using particles. CRC Press.
    .. [2] Deserno, M., & Holm, C. (1998). How to mesh up Ewald sums. I. A theoretical and
           numerical comparison of various particle mesh routines. The Journal of Chemical Physics,
           109(18), 7678-7693.
    """
    kappa = constants[0]
    Gew = constants[1]
    fourpie0 = constants[2]

    four_pi = 4.0 * pi if fourpie0 == 1.0 else 4.0 * pi / fourpie0

    mask = box_lengths.nonzero()
    non_zero_box_lengths = array([1.0, 1.0, 1.0])
    non_zero_box_lengths[mask] = box_lengths[mask].copy()

    kappa_sq = kappa * kappa
    Gew_sq = Gew * Gew

    G_k = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))

    PM_err = 0.0

    kx_v, ky_v, kz_v = create_k_arrays(mesh_sizes, non_zero_box_lengths)

    kx_M, ky_M, kz_M = create_k_aliases(aliases, mesh_sizes, non_zero_box_lengths)

    for nz, kz in enumerate(kz_v[:, 0, 0]):
        kz_sq = kz * kz
        for ny, ky in enumerate(ky_v[:, 0]):
            ky_sq = ky * ky
            for nx, kx in enumerate(kx_v[0, :]):
                kx_sq = kx * kx
                k_sq = kx_sq + ky_sq + kz_sq
                if k_sq != 0.0:
                    # eq.(22) of Ref.[Dharuman2017]_
                    U_G_k, U_k_sq = sum_over_aliases(
                        kx, ky, kz, kx_M[nx], ky_M[ny], kz_M[nz], h_array, p, four_pi, Gew_sq, kappa_sq
                    )

                    G_k[nz, ny, nx] = U_G_k / ((U_k_sq**2) * k_sq)

                    Gk_hat = four_pi * exp(-0.25 * (kappa_sq + k_sq) / Gew_sq) / (kappa_sq + k_sq)

                    # eq.(28) of Ref.[Dharuman2017]_
                    PM_err += Gk_hat * Gk_hat * k_sq - U_G_k**2 / ((U_k_sq**2) * k_sq)

    PM_err = sqrt(abs(PM_err)) / non_zero_box_lengths.prod() ** (1.0 / len(box_lengths.nonzero()[0]))

    return G_k, kx_v, ky_v, kz_v, PM_err


@jit(nopython=True)
def calc_virial_coefficients(kx_v, ky_v, kz_v, mesh_sizes, constants):
    """
    Compute the 6 virial coefficient arrays from k-space vectors.

    Parameters
    ----------
    kx_v, ky_v, kz_v : ndarray
        k-space vectors (kx[1,nx], ky[ny,1], kz[nz,1,1]).

    mesh_sizes : ndarray
        Number of mesh points in x, y, z, shape (3,).

    constants : ndarray
        Array containing [screening parameter, Ewald parameter, 4πε₀], shape (3,).

    Returns
    -------
    vg0..vg5 : 3D arrays
        Virial coefficient fields corresponding to:
        V_xx, V_yy, V_zz, V_xy, V_xz, V_yz
    """

    kappa = constants[0]
    Gew = constants[1]
    fourpie0 = constants[2]

    # four_pi = 4.0 * pi if fourpie0 == 1.0 else 4.0 * pi / fourpie0

    vg0 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))
    vg1 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))
    vg2 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))
    vg3 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))
    vg4 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))
    vg5 = zeros((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]))

    kx = kx_v[0]
    ky = ky_v[:, 0]
    kz = kz_v[:, 0, 0]

    for i in range(mesh_sizes[2]):
        kz_i = kz[i]
        kz_i_sq = kz_i * kz_i
        for j in range(mesh_sizes[1]):
            ky_j = ky[j]
            ky_j_sq = ky_j * ky_j
            for k in range(mesh_sizes[0]):
                kx_k = kx[k]
                kx_k_sq = kx_k * kx_k
                sqk = kx_k_sq + ky_j_sq + kz_i_sq + kappa**2

                if sqk == 0.0:
                    continue
                vterm = -2.0 * (1.0 / sqk + 0.25 / (Gew**2))

                vg0[i, j, k] = 1.0 + vterm * kx_k * kx_k  # V_xx
                vg1[i, j, k] = 1.0 + vterm * ky_j * ky_j  # V_yy
                vg2[i, j, k] = 1.0 + vterm * kz_i * kz_i  # V_zz
                vg3[i, j, k] = vterm * kx_k * ky_j  # V_xy
                vg4[i, j, k] = vterm * kx_k * kz_i  # V_xz
                vg5[i, j, k] = vterm * ky_j * kz_i  # V_yz

    return vg0, vg1, vg2, vg3, vg4, vg5


@jit(nopython=True)
def mesh_point_shift(cao):
    """
    Calculate the required shift based on the parity of the charge assignment orders.

    Parameters
    ----------
    cao: numpy.ndarray
        Charge assignment order per direction.

    Returns
    -------
    mid: numpy.ndarray
        Midpoint shift if cao is even, otherwise no shift.

    pshift: numpy.ndarray
        Shift to the closest mesh point.

    """
    pshift = zeros(len(cao), dtype=int64)
    mid = zeros(len(cao), dtype=float64)

    # Mid point calculation
    for ic, p in enumerate(cao):
        # Choose the midpoint between the two closest mesh point to the particle's position if cao is even otherwise
        # take the closest mesh-point
        mid[ic] = 0.5 * (p % 2 == 0)
        pshift[ic] = int(0.5 * p - 1 * (p % 2 == 0))

    return mid, pshift


@jit(nopython=True)
def calc_pm_all(
    E_x_r,
    E_y_r,
    E_z_r,
    phi_r,
    vg0,
    vg1,
    vg2,
    vg3,
    vg4,
    vg5,
    mesh_pos,
    mesh_points,
    charges,
    masses,
    cao,
    mesh_sz,
    mid,
    pshift,
):
    """
    Interpolate the electric field, potential energy, and virial coefficients to each particle.

    Parameters
    ----------
    E_x_r : ndarray
        Electric field in x-direction at mesh points, shape (mesh_sz[2], mesh_sz[1], mesh_sz[0]).
    E_y_r : ndarray
        Electric field in y-direction at mesh points, shape (mesh_sz[2], mesh_sz[1], mesh_sz[0]).
    E_z_r : ndarray
        Electric field in z-direction at mesh points, shape (mesh_sz[2], mesh_sz[1], mesh_sz[0]).
    phi_r : ndarray
        Potential energy at mesh points, shape (mesh_sz[2], mesh_sz[1], mesh_sz[0]).
    mesh_pos : ndarray
        Particles' positions relative to the mesh, shape (N, 3).
    mesh_points : ndarray
        Particles' positions on the mesh, shape (N, 3).
    charges : ndarray
        Particles' charges, shape (N,).
    masses : ndarray
        Particles' masses, shape (N,).
    cao : ndarray
        Charge assignment order for each direction, shape (3,).
    mesh_sz : ndarray
        Number of mesh points per direction, shape (3,).
    mid : ndarray
        Midpoint flag for the three directions, shape (3,).
    pshift : ndarray
        Midpoint shift in each direction, shape (3,).

    Returns
    -------
    acc : ndarray
        Particle accelerations, shape (N, 3).
    pot : ndarray
        Particle potential energies, shape (N,).
    virial_xx : ndarray
        Virial xx components, shape (N,).
    virial_yy : ndarray
        Virial yy components, shape (N,).
    virial_zz : ndarray
        Virial zz components, shape (N,).
    virial_xy : ndarray
        Virial xy components, shape (N,).
    virial_xz : ndarray
        Virial xz components, shape (N,).
    virial_yz : ndarray
        Virial yz components, shape (N,).
    """

    N = charges.shape[0]
    acc = zeros_like(mesh_pos)  # Particle accelerations, shape (N, 3)
    pot = zeros_like(charges)
    q_over_m = charges / masses

    virial_xx = zeros_like(charges)
    virial_yy = zeros_like(charges)
    virial_zz = zeros_like(charges)
    virial_xy = zeros_like(charges)
    virial_xz = zeros_like(charges)
    virial_yz = zeros_like(charges)

    for ipart in range(N):
        ix = mesh_points[ipart, 0]
        x = mesh_pos[ipart, 0] - (ix + mid[0])

        iy = mesh_points[ipart, 1]
        y = mesh_pos[ipart, 1] - (iy + mid[1])

        iz = mesh_points[ipart, 2]
        z = mesh_pos[ipart, 2] - (iz + mid[2])

        wx = assgnmnt_func(cao[0], x)
        wy = assgnmnt_func(cao[1], y)
        wz = assgnmnt_func(cao[2], z)

        q = charges[ipart]
        q_m = q_over_m[ipart]

        # Use modulo for periodic boundary conditions - consistent with calc_charge_dens
        base_z = (iz - pshift[2]) % mesh_sz[2]
        base_y = (iy - pshift[1]) % mesh_sz[1]
        base_x = (ix - pshift[0]) % mesh_sz[0]

        for g in range(cao[2]):
            r_g = (base_z + g) % mesh_sz[2]

            for i in range(cao[1]):
                r_i = (base_y + i) % mesh_sz[1]

                for j in range(cao[0]):
                    r_j = (base_x + j) % mesh_sz[0]

                    weight = wz[g] * wy[i] * wx[j]

                    # Acceleration
                    acc[ipart, 0] += q_m * E_x_r[r_g, r_i, r_j] * weight
                    acc[ipart, 1] += q_m * E_y_r[r_g, r_i, r_j] * weight
                    acc[ipart, 2] += q_m * E_z_r[r_g, r_i, r_j] * weight

                    # Potential energy
                    pot[ipart] += 0.5 * q * phi_r[r_g, r_i, r_j] * weight

                    # Virial components
                    virial_xx[ipart] += 0.5 * q * vg0[r_g, r_i, r_j] * weight
                    virial_yy[ipart] += 0.5 * q * vg1[r_g, r_i, r_j] * weight
                    virial_zz[ipart] += 0.5 * q * vg2[r_g, r_i, r_j] * weight
                    virial_xy[ipart] += 0.5 * q * vg3[r_g, r_i, r_j] * weight
                    virial_xz[ipart] += 0.5 * q * vg4[r_g, r_i, r_j] * weight
                    virial_yz[ipart] += 0.5 * q * vg5[r_g, r_i, r_j] * weight

    return acc, pot, virial_xx, virial_yy, virial_zz, virial_xy, virial_xz, virial_yz


def update(
    pos,
    charges,
    masses,
    mesh_sizes,
    mesh_spacings,
    mesh_volume,
    box_volume,
    G_k,
    kx_v,
    ky_v,
    kz_v,
    cao,
    vg0,
    vg1,
    vg2,
    vg3,
    vg4,
    vg5,
    fft_objects,
):
    """
    Calculate the long range part of particles' accelerations using the Particle-Mesh method.

    This function implements the core of the Particle-Mesh (PM) component of the PPPM algorithm.
    It computes the long-range contributions to particle accelerations and potential energies.

    Parameters
    ----------
    pos : ndarray
        Particles' positions, shape (N, 3).
    charges : ndarray
        Particles' charges, shape (N,).
    masses : ndarray
        Particles' masses, shape (N,).
    mesh_sizes : ndarray
        Number of mesh points per direction, shape (3,).
    mesh_spacings : ndarray
        Width of the mesh cells, shape (3,).
    mesh_volume : float
        Volume of a single mesh cell.
    box_volume : float
        Total volume of the simulation box.
    G_k : ndarray
        Optimized Green's function in Fourier space, shape (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]).
    kx_v : ndarray
        Array of kx values, shape (1, mesh_sizes[0]).
    ky_v : ndarray
        Array of ky values, shape (mesh_sizes[1], 1).
    kz_v : ndarray
        Array of kz values, shape (mesh_sizes[2], 1, 1).
    cao : ndarray
        Charge assignment order for each direction, shape (3,).
    vg0, vg1, vg2, vg3, vg4, vg5 : ndarray
        Virial coefficient fields corresponding to V_xx, V_yy, V_zz, V_xy, V_xz, V_yz, shape (mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]).
    fft_objects : FFTWObjects
        Pre-created FFT objects for reuse across timesteps

    Returns
    -------
    pot_particle : ndarray
        Long range part of the potential for each particle, shape (N,).
    acc_f : ndarray
        Long range part of particles' accelerations, shape (N, 3).

    Notes
    -----
    This function uses FFTW through pyfftw for efficient Fourier transforms. It assumes
    periodic boundary conditions and requires pre-computed optimized Green's function.

    The algorithm follows these main steps:
    1. Assign charges to the mesh
    2. Perform forward FFT on the charge density
    3. Solve Poisson's equation in Fourier space
    4. Compute the electric field in Fourier space
    5. Perform inverse FFT to get real-space electric field
    6. Interpolate mesh values to particle positions

    The function is optimized using Numba's JIT compilation.

    """

    # Calculate the necessary shifts
    mid, pshift = mesh_point_shift(cao)

    # Calculate particles' position relative to the mesh points
    mesh_pos, mesh_points = calc_mesh_coord(pos, mesh_spacings, cao)

    # Calculate charge density on mesh
    rho_r = calc_charge_dens(mesh_pos, mesh_points, charges, cao, mesh_sizes, mid, pshift)

    # Allocate memory for the output arrays of the FFT
    rho_k = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_x = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_y = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    E_z = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    phi_r = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xx = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_yy = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_zz = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xy = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_xz = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)
    virial_r_yz = empty_aligned((mesh_sizes[2], mesh_sizes[1], mesh_sizes[0]), dtype=complex)

    # Use reusable FFT object
    fft_objects.forward_transform(rho_r, rho_k)

    # Shift the DC value at the center of the ndarray
    rho_k = fftshift(rho_k)

    # Potential from Poisson eq.
    phi_k = G_k * rho_k

    # Calculate the Electric field's components on the mesh
    E_kx, E_ky, E_kz = calc_field(phi_k, kx_v, ky_v, kz_v)

    # Prepare for IFFT and compute - using reusable objects
    fft_objects.backward_transform(ifftshift(E_kx), E_x)
    fft_objects.backward_transform(ifftshift(E_ky), E_y)
    fft_objects.backward_transform(ifftshift(E_kz), E_z)

    # FFT normalization
    E_x /= mesh_volume
    E_y /= mesh_volume
    E_z /= mesh_volume

    # Calculate the potential energy of each particle
    fft_objects.backward_transform(ifftshift(phi_k), phi_r)
    phi_r /= mesh_volume

    # Virial tensor components - all using the same reusable IFFT
    fft_objects.backward_transform(ifftshift(phi_k * vg0), virial_r_xx)
    virial_r_xx /= mesh_volume

    fft_objects.backward_transform(ifftshift(phi_k * vg1), virial_r_yy)
    virial_r_yy /= mesh_volume

    fft_objects.backward_transform(ifftshift(phi_k * vg2), virial_r_zz)
    virial_r_zz /= mesh_volume

    fft_objects.backward_transform(ifftshift(phi_k * vg3), virial_r_xy)
    virial_r_xy /= mesh_volume

    fft_objects.backward_transform(ifftshift(phi_k * vg4), virial_r_xz)
    virial_r_xz /= mesh_volume

    fft_objects.backward_transform(ifftshift(phi_k * vg5), virial_r_yz)
    virial_r_yz /= mesh_volume

    # Interpolate all the fields to the particles
    acc_f, pot_particle, virial_xx, virial_yy, virial_zz, virial_xy, virial_xz, virial_yz = calc_pm_all(
        E_x.real,
        E_y.real,
        E_z.real,
        phi_r.real,
        virial_r_xx.real,
        virial_r_yy.real,
        virial_r_zz.real,
        virial_r_xy.real,
        virial_r_xz.real,
        virial_r_yz.real,
        mesh_pos,
        mesh_points,
        charges,
        masses,
        cao,
        mesh_sizes,
        mid,
        pshift,
    )

    return pot_particle, acc_f, virial_xx, virial_yy, virial_zz, virial_xy, virial_xz, virial_yz
