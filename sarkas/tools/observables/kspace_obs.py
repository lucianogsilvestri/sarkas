"""
K-space observable classes for sarkas post-processing.

All classes inherit from KspaceObservable, which manages the shared HDF5
cache (nkt, vkt, k-arrays) and the Observable HDF5 result store.

Storage convention
------------------
All pair-indexed data uses a flat ``species_pair`` dimension containing
upper-triangular pair labels (e.g. ``['H-H', 'H-He', 'He-He']``), consistent
with :class:`RadialDistributionFunction`.  This halves storage compared to a
full (no_sp x no_sp) matrix for multi-species systems.

The pair list and index map are provided by
:meth:`Observable._build_species_pairs` (called in ``setup_init``) as
``self.species_pairs`` and ``self.pair_index_map``.

Classes
-------
MicroscopicDensity
MicroscopicVelocity
MicroscopicCurrent
DynamicStructureFactor
CurrentCorrelationFunction
StaticStructureFactor
"""

import warnings
from os.path import exists as os_path_exists

import h5py
import numpy as np
from scipy.fft import fft as sp_fft, fftshift
from tqdm import tqdm

from .base import _HDF5_COMPRESSION, _HDF5_COMPRESSION_OPTS, compute_doc, setup_doc
from .kspace import KspaceObservable, _calc_nk_vectorised, _calc_vk_vectorised


# ---------------------------------------------------------------------------
# Module-level kernel helpers
# ---------------------------------------------------------------------------

def _calc_Skw_pairs(
    nkt_sl: np.ndarray,
    species_num: np.ndarray,
    pair_index_map: np.ndarray,
    num_pairs: int,
    block_length: int,
    dt: float,
    dump_step: int,
) -> np.ndarray:
    """Compute S_{AB}(k,ω) for all upper-triangular pairs via batch FFT.

    Parameters
    ----------
    nkt_sl        : complex128 ndarray, shape (no_species, no_k, block_length)
    species_num   : ndarray, shape (no_species,)
    pair_index_map : ndarray, shape (no_species, no_species), dtype int
    num_pairs     : int
    block_length  : int
    dt, dump_step : float

    Returns
    -------
    Skw : ndarray, shape (num_pairs, no_k, block_length)
        Pairs ordered upper-triangular, consistent with ``self.species_pairs``.
    """
    no_sp = len(species_num)
    no_k  = nkt_sl.shape[1]
    norm  = dt / np.sqrt(block_length * dt * dump_step)

    nkw = sp_fft(nkt_sl, axis=-1) * norm     # (no_sp, no_k, block_length)
    Skw = np.zeros((num_pairs, no_k, block_length))

    for i in range(no_sp):
        for j in range(i, no_sp):
            dens = 1.0 / np.sqrt(species_num[i] * species_num[j])
            pidx = pair_index_map[i, j]
            Skw[pidx] = fftshift(
                (nkw[i].conj() * nkw[j]).real * dens, axes=-1
            )
    return Skw


# ---------------------------------------------------------------------------
# MicroscopicDensity
# ---------------------------------------------------------------------------


class MicroscopicDensity(KspaceObservable):
    """Microscopic number density in k-space.

    n_A(k,t) = Σ_{j∈A} exp(-i k·r_j(t))

    Primary observable. The **shared k-space cache** is the sole HDF5 store
    for this class — no separate result file is written.  This avoids storing
    nkt twice: once in the cache and once in an observable-specific file.

    The ``/nkt`` group in ``kspace_cache_<job_id>.h5`` contains everything
    needed for both user access (:meth:`get_nkt`, :meth:`read_dataset`) and
    downstream derived observables (DSF, SSF).

    Shared Cache Layout (relevant groups)
    --------------------------------------
    ``/nkt/data``     — complex128, shape (no_species, no_k, no_dumps)
    ``/nkt/species``  — UTF-8 species names
    ``/nkt`` attrs    — ``no_dumps``, ``h5md_path`` (cache invalidation)
    ``/time``         — shape (no_dumps,)
    ``/k_arrays/...`` — k-vector arrays (written by :meth:`_compute_k_arrays`)
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "mic_density"
        self.__long_name__ = "Microscopic Density"
        self.kw_observable = False

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    # ------------------------------------------------------------------
    # Store validity — delegates to the shared cache, not hdf_store_path
    # ------------------------------------------------------------------

    def _store_is_valid(self) -> bool:
        """Valid when the shared cache contains a current ``/nkt`` group."""
        return self._nkt_is_valid()

    def _invalidate_store(self):
        """Invalidate by deleting only the ``/nkt`` group from the shared cache."""
        self._invalidate_nkt()

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    @compute_doc
    def compute(self):
        """Compute n(k,t) and write it to the shared cache (single copy)."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self._compute_nkt()     # writes /nkt + /time to shared cache — done.
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    # ------------------------------------------------------------------
    # User-facing data access
    # ------------------------------------------------------------------

    def get_nkt(self) -> tuple:
        """Return n(k,t) and time, computing if necessary.

        Returns
        -------
        nkt  : numpy.ndarray, complex128, shape (no_species, no_k, no_dumps)
        time : numpy.ndarray,             shape (no_dumps,)
        """
        self.compute()          # no-op if cache is valid
        return self._get_nkt()  # reads from shared cache

    def read_dataset(self):
        """Return n(k,t) as an ``xr.Dataset``, reading from the shared cache.

        Returns
        -------
        xr.Dataset
            Variable ``nkt``, dims ``['species', 'k', 'time']``.
        """
        import xarray as xr
        self.compute()
        with self._open_kspace_cache() as f:
            species = list(f["nkt/species"].asstr()[:])
            nkt     = f["nkt/data"][:]
            time    = f["time"][:]
        no_k = nkt.shape[1]
        da = xr.DataArray(
            nkt, dims=["species", "k", "time"],
            coords={"species": species, "k": np.arange(no_k), "time": time},
        )
        return xr.Dataset({"nkt": da})


# ---------------------------------------------------------------------------
# MicroscopicVelocity
# ---------------------------------------------------------------------------


class MicroscopicVelocity(KspaceObservable):
    """Microscopic velocity field in k-space.

    v_{A,d}(k,t) = Σ_{j∈A} v_{j,d}(t) exp(-i k·r_j(t))

    Primary observable. The **shared k-space cache** is the sole HDF5 store —
    no separate result file is written, mirroring :class:`MicroscopicDensity`.

    Shared Cache Layout (relevant groups)
    --------------------------------------
    ``/vkt/data``       — complex128, shape (no_species, dim, no_k, no_dumps)
    ``/vkt/species``    — UTF-8 species names
    ``/vkt/component``  — UTF-8 component labels (e.g. ``['X','Y','Z']``)
    ``/vkt`` attrs      — ``no_dumps``, ``h5md_path`` (cache invalidation)
    ``/time``           — shape (no_dumps,)
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "mic_velocity"
        self.__long_name__ = "Microscopic Velocity"
        self.kw_observable = False

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    # ------------------------------------------------------------------
    # Store validity — delegates to the shared cache
    # ------------------------------------------------------------------

    def _store_is_valid(self) -> bool:
        """Valid when the shared cache contains a current ``/vkt`` group."""
        return self._vkt_is_valid()

    def _invalidate_store(self):
        """Invalidate by deleting only the ``/vkt`` group from the shared cache."""
        self._invalidate_vkt()

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    @compute_doc
    def compute(self):
        """Compute v(k,t) and write it to the shared cache (single copy)."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self._compute_vkt()     # writes /vkt + /time to shared cache — done.
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    # ------------------------------------------------------------------
    # User-facing data access
    # ------------------------------------------------------------------

    def get_vkt(self) -> tuple:
        """Return v(k,t) and time, computing if necessary.

        Returns
        -------
        vkt  : numpy.ndarray, complex128, shape (no_species, dim, no_k, no_dumps)
        time : numpy.ndarray,             shape (no_dumps,)
        """
        self.compute()
        return self._get_vkt()

    def read_dataset(self):
        """Return v(k,t) as an ``xr.Dataset``, reading from the shared cache.

        Returns
        -------
        xr.Dataset
            Variable ``vkt``, dims ``['species', 'component', 'k', 'time']``.
        """
        import xarray as xr
        self.compute()
        with self._open_kspace_cache() as f:
            species    = list(f["vkt/species"].asstr()[:])
            components = list(f["vkt/component"].asstr()[:])
            vkt        = f["vkt/data"][:]
            time       = f["time"][:]
        no_k = vkt.shape[2]
        da = xr.DataArray(
            vkt, dims=["species", "component", "k", "time"],
            coords={"species": species, "component": components,
                    "k": np.arange(no_k), "time": time},
        )
        return xr.Dataset({"vkt": da})


# ---------------------------------------------------------------------------
# MicroscopicCurrent
# ---------------------------------------------------------------------------


class MicroscopicCurrent(KspaceObservable):
    """Microscopic electric current in k-space.

    J_{A,d}(k,t) = q_A * v_{A,d}(k,t)

    Primary observable. Does **not** recompute v(k,t) if the shared cache
    already holds a valid ``/vkt`` group.  The total current
    J_total = Σ_A J_A is appended as an additional species row labelled
    ``'Total'``.

    HDF5 Result Store Layout
    -------------------------
    ``/coords/species``     — species names + ``'Total'``
    ``/coords/component``   — velocity components
    ``/coords/k``           — k-vector index
    ``/coords/time``        — simulation time
    ``/data/jkt_real``      — Re[J(k,t)], shape (no_species+1, dim, no_k, no_dumps)
    ``/data/jkt_imag``      — Im[J(k,t)], shape (no_species+1, dim, no_k, no_dumps)
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "mic_current"
        self.__long_name__ = "Microscopic Current"
        self.kw_observable = False

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    @compute_doc
    def compute(self):
        """Compute J(k,t) from v(k,t) and save to the result store."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        vkt, time = self._get_vkt()     # cache-aware: won't recompute if valid

        no_sp, dim, no_k, no_dumps = vkt.shape
        jkt = np.zeros((no_sp + 1, dim, no_k, no_dumps), dtype=np.complex128)
        for isp, q in enumerate(self.species_charges):
            jkt[isp] = q * vkt[isp]
        jkt[-1] = jkt[:no_sp].sum(axis=0)

        self._write_result_store(jkt, time)
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    def _write_result_store(self, jkt: np.ndarray, time: np.ndarray):
        _, dim, no_k, _ = jkt.shape
        species_all = list(self.species_names) + ["Total"]
        dt_str = h5py.string_dtype()

        with h5py.File(self.hdf_store_path, "w") as f:
            cg = f.require_group("coords")
            cg.create_dataset("species",   data=np.array(species_all, dtype=object), dtype=dt_str)
            cg.create_dataset("component", data=np.array(self.dim_labels[:dim], dtype=object), dtype=dt_str)
            cg.create_dataset("k",    data=np.arange(no_k))
            cg.create_dataset("time", data=time)
            for ds in cg.values():
                ds.attrs["_is_coord"] = True

            dg = f.require_group("data")
            for name, arr in [("jkt_real", jkt.real), ("jkt_imag", jkt.imag)]:
                ds = dg.create_dataset(name, data=arr,
                                       compression=_HDF5_COMPRESSION,
                                       compression_opts=_HDF5_COMPRESSION_OPTS)
                ds.attrs["_DIMS"] = ["species", "component", "k", "time"]

            f.attrs.update({
                "no_slices":    self.no_slices,
                "block_length": self.block_length,
                "dumps_shift":  self.dumps_shift,
                "h5md_path":    self.h5md_filepath,
                "no_dumps":     self.no_dumps,
            })

    def _store_is_valid(self) -> bool:
        if not super()._store_is_valid():
            return False
        try:
            with h5py.File(self.hdf_store_path, "r") as f:
                return (
                    f.attrs.get("no_dumps") == self.no_dumps
                    and "data/jkt_real" in f
                )
        except Exception:
            return False

    def read_dataset(self):
        """Return J(k,t) as an ``xr.Dataset``."""
        import xarray as xr
        with h5py.File(self.hdf_store_path, "r") as f:
            species    = list(f["coords/species"].asstr()[:])
            components = list(f["coords/component"].asstr()[:])
            k_idx      = f["coords/k"][:]
            time       = f["coords/time"][:]
            jkt        = f["data/jkt_real"][:] + 1j * f["data/jkt_imag"][:]
        da = xr.DataArray(jkt, dims=["species", "component", "k", "time"],
                          coords={"species": species, "component": components,
                                  "k": k_idx, "time": time})
        return xr.Dataset({"jkt": da})


# ---------------------------------------------------------------------------
# DynamicStructureFactor
# ---------------------------------------------------------------------------


class DynamicStructureFactor(KspaceObservable):
    """Dynamic Structure Factor S_{AB}(k,ω).

    S_{AB}(k,ω) = (1/√(N_A N_B)) FT[〈n_A(k,t) n_B(-k,0)〉]

    Computed slice-by-slice from the shared nkt cache to keep RAM bounded.
    Results are stored with a flat ``species_pair`` dimension.

    HDF5 Result Store Layout
    -------------------------
    ``/coords/species_pair`` — pair labels,   shape (num_pairs,)
    ``/coords/ka``           — ka values,     shape (no_ka,)
    ``/coords/frequency``    — frequencies,   shape (no_freq,)
    ``/coords/slice``        — slice indices, shape (no_slices,)
    ``/data/Skw``            — shape (num_pairs, no_ka, no_freq, no_slices)
    ``/data/mean_Skw``       — shape (num_pairs, no_ka, no_freq)
    ``/data/std_Skw``        — shape (num_pairs, no_ka, no_freq)

    Notes
    -----
    ``no_ka`` refers to the number of unique |k| magnitudes (angle-averaged
    bins), not the total number of k-vectors ``no_k``.  The binning is
    performed inside the slice loop using ``k_counts``.
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "dsf"
        self.__long_name__ = "Dynamic Structure Factor"
        self.kw_observable = True

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    @compute_doc
    def compute(self):
        """Compute S(k,ω) slice-by-slice and append mean/std to the result store."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        if not self._nkt_is_valid():
            self._compute_nkt()
        self._calc_slices()
        self._write_mean_std(["Skw"], ddof=min(1, self.no_slices - 1))
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    def _calc_slices(self):
        no_ka   = self.no_ka_values
        no_freq = self.block_length
        # k_bin maps each k-vector index to its ka-bin index
        k_bin   = self.k_list[:, -1].astype(int)

        coords = {
            "species_pair": self.species_pairs,
            "ka":           self.ka_values,
            "frequency":    self.frequencies,
            "slice":        np.arange(self.no_slices),
        }
        self._preallocate_store(
            variable_shapes={"Skw": (self.num_pairs, no_ka, no_freq, self.no_slices)},
            coords=coords,
            attrs={
                "no_slices":    self.no_slices,
                "block_length": self.block_length,
                "dumps_shift":  self.dumps_shift,
                "h5md_path":    self.h5md_filepath,
            },
        )

        with self._open_kspace_cache() as cache:
            nkt_ds = cache["nkt/data"]   # (no_sp, no_k, no_dumps) lazy

            for isl in tqdm(range(self.no_slices), desc="DSF slices",
                            disable=not self.verbose):
                start  = isl * self.dumps_shift
                end    = start + self.block_length
                nkt_sl = nkt_ds[:, :, start:end]   # (no_sp, no_k, block_length)

                # Skw_kv: (num_pairs, no_k, no_freq) — all k-vectors
                Skw_kv = _calc_Skw_pairs(
                    nkt_sl, self.species_num, self.pair_index_map,
                    self.num_pairs, self.block_length, self.dt, self.dump_step,
                )

                # Bin from no_k → no_ka using angle-averaging
                Skw_ka = np.zeros((self.num_pairs, no_ka, no_freq))
                counts = np.bincount(k_bin, minlength=no_ka).clip(min=1)
                for p in range(self.num_pairs):
                    for f in range(no_freq):
                        Skw_ka[p, :, f] = (
                            np.bincount(k_bin, weights=Skw_kv[p, :, f],
                                        minlength=no_ka) / counts
                        )

                self._write_slice({"Skw": Skw_ka}, isl)


# ---------------------------------------------------------------------------
# CurrentCorrelationFunction
# ---------------------------------------------------------------------------


class CurrentCorrelationFunction(KspaceObservable):
    """Longitudinal and transverse current correlation functions.

    L_{AB}(k,ω) ∝ FT[〈J_{A,∥}(k,t) J_{B,∥}(-k,0)〉]
    T_{AB}(k,ω) ∝ FT[〈J_{A,⊥}(k,t) J_{B,⊥}(-k,0)〉]

    Results are stored with a flat ``species_pair`` dimension.

    .. note::
        The L/T decomposition along k̂ is a placeholder; both currently
        store the full Cartesian sum.  This is marked ``# TODO``.

    HDF5 Result Store Layout
    -------------------------
    ``/coords/species_pair`` — pair labels
    ``/coords/ka``           — ka values
    ``/coords/frequency``    — frequencies
    ``/coords/slice``        — slice indices
    ``/data/longitudinal``   — shape (num_pairs, no_ka, no_freq, no_slices)
    ``/data/transverse``     — shape (num_pairs, no_ka, no_freq, no_slices)
    ``/data/mean_longitudinal``, ``/data/std_longitudinal``
    ``/data/mean_transverse``,   ``/data/std_transverse``
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "ccf"
        self.__long_name__ = "Current Correlation Function"
        self.kw_observable = True

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    @compute_doc
    def compute(self):
        """Compute L(k,ω) and T(k,ω) slice-by-slice."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        if not self._vkt_is_valid():
            self._compute_vkt()
        self._calc_slices()
        self._write_mean_std(
            ["longitudinal", "transverse"],
            ddof=min(1, self.no_slices - 1),
        )
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    def _calc_slices(self):
        no_ka   = self.no_ka_values
        no_freq = self.block_length
        norm    = self.dt / np.sqrt(self.block_length * self.dt * self.dump_step)
        k_bin   = self.k_list[:, -1].astype(int)
        counts  = np.bincount(k_bin, minlength=no_ka).clip(min=1)

        coords = {
            "species_pair": self.species_pairs,
            "ka":           self.ka_values,
            "frequency":    self.frequencies,
            "slice":        np.arange(self.no_slices),
        }
        self._preallocate_store(
            variable_shapes={
                "longitudinal": (self.num_pairs, no_ka, no_freq, self.no_slices),
                "transverse":   (self.num_pairs, no_ka, no_freq, self.no_slices),
            },
            coords=coords,
            attrs={
                "no_slices":    self.no_slices,
                "block_length": self.block_length,
                "dumps_shift":  self.dumps_shift,
                "h5md_path":    self.h5md_filepath,
            },
        )

        no_sp = len(self.species_names)

        with self._open_kspace_cache() as cache:
            vkt_ds = cache["vkt/data"]   # (no_sp, dim, no_k, no_dumps) lazy

            for isl in tqdm(range(self.no_slices), desc="CCF slices",
                            disable=not self.verbose):
                start  = isl * self.dumps_shift
                end    = start + self.block_length
                vkt_sl = vkt_ds[:, :, :, start:end]   # (no_sp, dim, no_k, T)
                vkw    = sp_fft(vkt_sl, axis=-1) * norm

                # Accumulate cross-spectra into flat pair array, then bin to ka
                Lkw_kv = np.zeros((self.num_pairs, len(self.k_list), no_freq))
                Tkw_kv = np.zeros_like(Lkw_kv)

                for i in range(no_sp):
                    for j in range(i, no_sp):
                        dens = 1.0 / np.sqrt(self.species_num[i] * self.species_num[j])
                        pidx = self.pair_index_map[i, j]
                        cross = np.zeros((len(self.k_list), no_freq))
                        for d in range(self.dimensions):
                            cross += np.real(vkw[i, d].conj() * vkw[j, d]) * dens
                        # TODO: proper L/T decomposition along k̂
                        Lkw_kv[pidx] = fftshift(cross, axes=-1)
                        Tkw_kv[pidx] = fftshift(cross, axes=-1)

                # Bin k-vectors → ka-bins
                Lkw_ka = np.zeros((self.num_pairs, no_ka, no_freq))
                Tkw_ka = np.zeros_like(Lkw_ka)
                for p in range(self.num_pairs):
                    for fi in range(no_freq):
                        Lkw_ka[p, :, fi] = np.bincount(
                            k_bin, weights=Lkw_kv[p, :, fi], minlength=no_ka
                        ) / counts
                        Tkw_ka[p, :, fi] = np.bincount(
                            k_bin, weights=Tkw_kv[p, :, fi], minlength=no_ka
                        ) / counts

                self._write_slice({"longitudinal": Lkw_ka, "transverse": Tkw_ka}, isl)


# ---------------------------------------------------------------------------
# StaticStructureFactor
# ---------------------------------------------------------------------------


class StaticStructureFactor(KspaceObservable):
    """Static Structure Factor S_{AB}(k).

    S_{AB}(k) = (1/N_dumps) Σ_t Re[n_A(k,t) n_B*(k,t)] / √(N_A N_B)

    Each dump is an independent sample.  Mean and std are computed over
    the dump axis.  Results use a flat ``species_pair`` dimension.

    HDF5 Result Store Layout
    -------------------------
    ``/coords/species_pair`` — pair labels,        shape (num_pairs,)
    ``/coords/ka``           — ka values,          shape (no_ka,)
    ``/coords/slice``        — dump indices,       shape (no_dumps,)
    ``/data/Sk``             — shape (num_pairs, no_ka, no_dumps)
    ``/data/mean_Sk``        — shape (num_pairs, no_ka)
    ``/data/std_Sk``         — shape (num_pairs, no_ka)
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "ssf"
        self.__long_name__ = "Static Structure Factor"
        self.kw_observable = False

    @setup_doc
    def setup(self, params, phase=None, no_slices=None, independent_slices=None,
              timesteps_per_slice=None, timesteps_shift=None,
              plasma_periods_per_slice=None, plasma_periods_shift=None, **kwargs):
        super().setup_init(
            params, phase=phase, no_slices=no_slices,
            independent_slices=independent_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift, **kwargs,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    @compute_doc
    def compute(self):
        """Compute S(k) dump-by-dump and append mean/std to the result store."""
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        if not self._nkt_is_valid():
            self._compute_nkt()
        self._calc_dumps()
        self._write_mean_std(["Sk"], ddof=min(1, self.no_slices - 1))
        self.save_state()
        tend = self.timer.current()
        from ...utilities.timing import time_stamp
        time_stamp(self.log_file, self.__long_name__ + " Calculation",
                   self.timer.time_division(tend - t0), self.verbose)

    def _store_is_valid(self) -> bool:
        if not super()._store_is_valid():
            return False
        try:
            with h5py.File(self.hdf_store_path, "r") as f:
                return f.attrs.get("no_slices") == self.no_slices
        except Exception:
            return False

    def _calc_dumps(self):
        """Compute S(k) for each dump independently and write slice-by-slice."""
        no_ka  = self.no_ka_values
        k_bin  = self.k_list[:, -1].astype(int)
        counts = np.bincount(k_bin, minlength=no_ka).clip(min=1)

        coords = {
            "species_pair": self.species_pairs,
            "ka":           self.ka_values,
            "slice":        np.arange(self.no_slices),
        }
        self._preallocate_store(
            variable_shapes={"Sk": (self.num_pairs, no_ka, self.no_slices)},
            coords=coords,
            attrs={
                "no_slices":    self.no_slices,
                "block_length": self.block_length,
                "dumps_shift":  self.dumps_shift,
                "h5md_path":    self.h5md_filepath,
                "no_dumps":     self.no_dumps,
            },
        )

        no_sp = len(self.species_names)

        with self._open_kspace_cache() as cache:
            nkt_ds = cache["nkt/data"]   # (no_sp, no_k, no_dumps) lazy

            for isl in tqdm(range(self.no_slices), desc="SSF slices", disable=not self.verbose):
                start = isl * self.dumps_shift
                end   = start + self.block_length
                Sk = np.zeros((self.num_pairs, no_ka))

                for it in tqdm(range(start, end), desc="SSF dumps", disable=not self.verbose, leave=False):
                    nk = nkt_ds[:, :, it]   # (no_sp, no_k) complex128

                    for i in range(no_sp):
                        for j in range(i, no_sp):
                            dens = 1.0 / np.sqrt(self.species_num[i] * self.species_num[j])
                            pidx = self.pair_index_map[i, j]
                            sk_kv = np.real(nk[i] * nk[j].conj()) * dens  # (no_k,)
                            Sk[pidx] += np.bincount(
                                k_bin, weights=sk_kv, minlength=no_ka
                            ) / counts
                self._write_slice({"Sk": Sk/(end - start)}, isl)