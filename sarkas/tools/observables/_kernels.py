"""
Numba-JIT compiled kernels for sarkas observables.

All @njit functions live exclusively here so Numba compiles each kernel
exactly once regardless of how many modules import it.
"""

import numpy as np
from numpy import zeros, exp, sqrt, real, pi
from numpy.linalg import lstsq
from numba import njit
from numpy import complex128


@njit
def calc_nk_numba(pos_data, k_list, species_np):
    """Calculate n(k) for all species using a Python loop over k-vectors.

    Parameters
    ----------
    pos_data : numpy.ndarray
        Positions of all particles. Shape (total_particles, 3).
    k_list : numpy.ndarray
        k-vectors. Shape (no_k_vectors, 3).
    species_np : numpy.ndarray
        Number of particles per species. Shape (no_species,).

    Returns
    -------
    nk : numpy.ndarray, complex128
        Shape (no_species, no_k_vectors).
    """
    n_species = len(species_np)
    nk = zeros((n_species, len(k_list)), dtype=complex128)
    sp_start = 0
    for i, sp in enumerate(species_np):
        sp_end = sp_start + sp
        for ik, k_vec in enumerate(k_list):
            kr_i = (
                2.0
                * pi
                * (
                    k_vec[0] * pos_data[sp_start:sp_end, 0]
                    + k_vec[1] * pos_data[sp_start:sp_end, 1]
                    + k_vec[2] * pos_data[sp_start:sp_end, 2]
                )
            )
            nk[i, ik] = exp(-1j * kr_i).sum()
        sp_start = sp_end
    return nk


def calc_vk_reference(pos_data, vel_data, k_list, species_np):
    """Reference (loop) implementation of MicroscopicVelocity v(k) per species.

    Computes for each species A and each k-vector:
        v_{A,d}(k) = sum_{j in A} v_{j,d} * exp(-i k . r_j)

    Used as a reference for testing the vectorised version.

    Parameters
    ----------
    pos_data : numpy.ndarray
        Positions. Shape (total_particles, 3).
    vel_data : numpy.ndarray
        Velocities. Shape (total_particles, 3).
    k_list : numpy.ndarray
        k-vectors. Shape (no_k, 3).
    species_np : numpy.ndarray
        Particles per species. Shape (no_species,).

    Returns
    -------
    vk : numpy.ndarray, complex128
        Shape (no_species, 3, no_k).
    """
    n_species = len(species_np)
    no_k = len(k_list)
    vk = zeros((n_species, 3, no_k), dtype=complex128)
    sp_start = 0
    for i, sp in enumerate(species_np):
        sp_end = sp_start + sp
        pos_sp = pos_data[sp_start:sp_end, :]  # (sp, 3)
        vel_sp = vel_data[sp_start:sp_end, :]  # (sp, 3)
        for ik, k_vec in enumerate(k_list):
            kr = 2.0 * pi * (k_vec[0] * pos_sp[:, 0] + k_vec[1] * pos_sp[:, 1] + k_vec[2] * pos_sp[:, 2])
            phase = exp(-1j * kr)  # (sp,)
            for d in range(3):
                vk[i, d, ik] = (vel_sp[:, d] * phase).sum()
        sp_start = sp_end
    return vk


@njit
def calc_Sk(nkt, k_list, k_counts, species_np, no_dumps):
    """Calculate S_ij(k) at each saved timestep.

    Parameters
    ----------
    nkt : numpy.ndarray, complex
        Density fluctuations. Shape (no_species, no_dumps, no_ka_values).
    k_list : numpy.ndarray
        k indices with magnitude and index. Shape (no_ka_values, 5).
    k_counts : numpy.ndarray
        Occurrences of each k magnitude.
    species_np : numpy.ndarray
        Particles per species.
    no_dumps : int

    Returns
    -------
    Sk_all : numpy.ndarray
        Shape (no_Sk, no_ka_values, no_dumps).
    """
    no_sk = int(len(species_np) * (len(species_np) + 1) / 2)
    Sk_raw = zeros((no_sk, len(k_counts), no_dumps))
    pair_indx = 0
    for ip, si in enumerate(species_np):
        for jp in range(ip, len(species_np)):
            sj = species_np[jp]
            dens_const = 1.0 / sqrt(si * sj)
            for it in range(no_dumps):
                for ik, ka in enumerate(k_list):
                    indx = int(ka[-1])
                    nk_i = nkt[ip, it, ik]
                    nk_j = nkt[jp, it, ik]
                    Sk_raw[pair_indx, indx, it] += real(nk_i.conjugate() * nk_j) * dens_const / k_counts[indx]
            pair_indx += 1
    return Sk_raw


@njit
def calc_elec_current(vel, sp_charge, sp_num):
    """Calculate total electric current and per-species current.

    Parameters
    ----------
    vel : numpy.ndarray
        Velocities. Shape (no_dumps, total_particles, no_dim).
    sp_charge : numpy.ndarray
        Charge of each species.
    sp_num : numpy.ndarray
        Number of particles per species.

    Returns
    -------
    Js : numpy.ndarray
        Electric current per species. Shape (no_dumps, no_species, no_dim).
    """
    no_dumps = vel.shape[0]
    no_dim = vel.shape[-1]
    Js = zeros((no_dumps, sp_num.shape[0], no_dim))
    sp_start = 0
    sp_end = 0
    for s, (q_sp, n_sp) in enumerate(zip(sp_charge, sp_num)):
        sp_end += n_sp
        Js[:, s, :] = q_sp * vel[:, sp_start:sp_end, :].sum(axis=1)
        sp_start += n_sp
    return Js


@njit
def calc_statistical_efficiency(observable, run_avg, run_std, max_no_divisions, no_dumps):
    """Calculate statistical efficiency via block averaging.

    Parameters
    ----------
    observable : numpy.ndarray
    run_avg : float
    run_std : float
    max_no_divisions : int
    no_dumps : int

    Returns
    -------
    tau_blk, sigma2_blk, statistical_efficiency : numpy.ndarray
    """
    tau_blk = zeros(max_no_divisions)
    sigma2_blk = zeros(max_no_divisions)
    statistical_efficiency = zeros(max_no_divisions)
    for i in range(2, max_no_divisions):
        tau_blk[i] = int(no_dumps / i)
        for j in range(i):
            t_start = int(j * tau_blk[i])
            t_end = int((j + 1) * tau_blk[i])
            blk_avg = observable[t_start:t_end].mean()
            sigma2_blk[i] += (blk_avg - run_avg) ** 2
        sigma2_blk[i] /= i - 1
        statistical_efficiency[i] = tau_blk[i] * sigma2_blk[i] / run_std ** 2
    return tau_blk, sigma2_blk, statistical_efficiency


@njit
def calc_vk(pos_data, vel_data, k_list):
    """Calculate instantaneous longitudinal and transverse velocity fluctuations.

    Parameters
    ----------
    pos_data : numpy.ndarray
        Positions. Shape (no_particles, 3).
    vel_data : numpy.ndarray
        Velocities. Shape (no_particles, 3).
    k_list : numpy.ndarray
        k-vectors with magnitude/index. Shape (no_ka_values, 5).

    Returns
    -------
    vk : numpy.ndarray, complex128
        Longitudinal velocity fluctuations. Shape (no_k,).
    vk_i, vk_j, vk_k : numpy.ndarray, complex128
        Transverse velocity fluctuations. Shape (no_k,) each.
    """
    vk = zeros(len(k_list), dtype=complex128)
    vk_i = zeros(len(k_list), dtype=complex128)
    vk_j = zeros(len(k_list), dtype=complex128)
    vk_k = zeros(len(k_list), dtype=complex128)

    for ik, k_vec in enumerate(k_list):
        kr_i = 2.0 * pi * (k_vec[0] * pos_data[:, 0] + k_vec[1] * pos_data[:, 1] + k_vec[2] * pos_data[:, 2])
        k_dot_v = 2.0 * pi * (k_vec[0] * vel_data[:, 0] + k_vec[1] * vel_data[:, 1] + k_vec[2] * vel_data[:, 2])
        k_cross_v_i = 2.0 * pi * (k_vec[1] * vel_data[:, 2] - k_vec[2] * vel_data[:, 1])
        k_cross_v_j = -2.0 * pi * (k_vec[0] * vel_data[:, 2] - k_vec[2] * vel_data[:, 0])
        k_cross_v_k = 2.0 * pi * (k_vec[0] * vel_data[:, 1] - k_vec[1] * vel_data[:, 0])
        vk[ik] = (k_dot_v * exp(-1j * kr_i)).sum()
        vk_i[ik] = (k_cross_v_i * exp(-1j * kr_i)).sum()
        vk_j[ik] = (k_cross_v_j * exp(-1j * kr_i)).sum()
        vk_k[ik] = (k_cross_v_k * exp(-1j * kr_i)).sum()
    return vk, vk_i, vk_j, vk_k


@njit
def remove_linear_trend(y, time_array):
    """Remove linear trend from a time series via simple linear regression.

    Parameters
    ----------
    y : numpy.ndarray
    time_array : numpy.ndarray

    Returns
    -------
    detrended : numpy.ndarray
    intercept : float
    slope : float
    """
    n = len(y)
    X = zeros((n, 2))
    X[:, 0] = 1.0
    X[:, 1] = time_array
    coeffs = lstsq(X, y)[0]
    intercept = coeffs[0]
    slope = coeffs[1]
    trend = intercept + slope * time_array
    detrended = y - trend
    return detrended, intercept, slope


def acf_batch(data: np.ndarray) -> np.ndarray:
    """Compute the normalised autocorrelation function for a batch of signals.

    Uses a single FFT pass for all N signals simultaneously.

    Parameters
    ----------
    data : np.ndarray
        Shape ``(N, T)`` — N signals each of length T.

    Returns
    -------
    acf : np.ndarray
        Shape ``(N, T)`` — normalised ACF for each signal, positive lags only.
        Normalisation divides lag tau by ``(T - tau)`` so that ``acf[:, 0]``
        equals the zero-lag (unnormalised) value divided by T.

    Notes
    -----
    Zero-padding to ``2*T`` prevents circular correlation artifacts.
    """
    from scipy.fft import fft, ifft
    N, T = data.shape
    nfft = 2 * T
    F = fft(data, n=nfft, axis=-1)           # (N, nfft)
    power = F * np.conj(F)                    # (N, nfft)
    corr = np.real(ifft(power, axis=-1))[:, :T]  # (N, T)
    norm = np.arange(T, 0, -1, dtype=float)  # (T,)
    return corr / norm
