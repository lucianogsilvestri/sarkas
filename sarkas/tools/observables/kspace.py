"""
KspaceObservable — intermediate base class for all k-space observables.

Inherits from Observable and adds:
  - k-vector resolution and storage (HDF5, replaces npz)
  - shared n(k,t) / v(k,t) cache (single HDF5 file, shared by all derived classes)
  - frequency-axis setup for kw observables
  - overridden create_dirs_filenames, update_finish, pretty_print_msg

Classes that inherit from KspaceObservable
------------------------------------------
  MicroscopicDensity
  MicroscopicVelocity
  MicroscopicCurrent
  DynamicStructureFactor
  CurrentCorrelationFunction
  StaticStructureFactor
"""

import warnings
from os import mkdir
from os.path import exists as os_path_exists
from os.path import join as os_path_join

import h5py
import numpy as np
from scipy.fft import fftfreq, fftshift
from scipy.linalg import norm

from .base import Observable, _HDF5_COMPRESSION, _HDF5_COMPRESSION_OPTS, setup_doc


class KspaceObservable(Observable):
    """Intermediate base class for all k-space observables.

    Extends :class:`Observable` with k-vector management, a shared HDF5 cache
    for n(k,t) and v(k,t), and frequency-axis setup for kw observables.

    All k-space file I/O uses a single shared HDF5 file located at::

        <k_space_dir>/kspace_cache_<job_id>.h5

    K-vector HDF5 Layout
    --------------------
    ``/k_arrays/k_list``          — shape (no_k, 5): [kx, ky, kz, |k|, k_index]
    ``/k_arrays/k_harmonics``     — shape (no_k, 4): [nx, ny, nz, |n|]  int
    ``/k_arrays/k_counts``        — shape (no_ka,):  multiplicity per unique |k|
    ``/k_arrays/k_values``        — shape (no_ka,):  unique |k| values (1/m)
    ``/k_arrays/ka_values``       — shape (no_ka,):  unique ka = 2π|k|a_ws
    ``/k_arrays`` attrs           — angle_averaging, max_k_harmonics,
                                    max_aa_harmonics — used for cache invalidation

    n(k,t) / v(k,t) HDF5 Layout
    ----------------------------
    ``/nkt/data``                 — shape (no_species, no_k, no_dumps)  complex128
    ``/nkt/species``              — species name strings
    ``/nkt`` attrs                — no_dumps, h5md_path — cache invalidation
    ``/vkt/data``                 — shape (no_species, 3, no_k, no_dumps)  complex128
    ``/vkt/species``              — species name strings
    ``/vkt`` attrs                — no_dumps, h5md_path — cache invalidation
    ``/time``                     — shape (no_dumps,)
    ``/k_arrays/...``             — same k-vector layout as above
    """

    def __init__(self):
        super().__init__()
        # These replace the k_observable / kw_observable flags that lived in
        # Observable.__init__. KspaceObservable is always a k-observable.
        self.k_observable  = True
        self.kw_observable = False   # overridden to True by DSF, CCF

        # K-vector parameters — set by _resolve_k_harmonics
        self.angle_averaging  = "principal_axis"
        self.max_k_harmonics  = None
        self.max_aa_harmonics = None
        self.max_aa_ka_value  = None
        self.max_ka_value     = None
        self.ka_values        = None
        self.k_values         = None
        self.no_ka_values     = None
        self.k_list           = None
        self.k_harmonics      = None
        self.k_counts         = None

        # Frequency axis — set by update_finish when kw_observable is True
        self.frequencies = None
        self.w_min       = None
        self.w_max       = None

        # File paths — set by _create_kspace_dirs via create_dirs_filenames
        self.k_space_dir       = None
        self.kspace_cache_path = None   # single HDF5 for nkt, vkt, k-arrays

    # ------------------------------------------------------------------
    # Directory and path setup
    # ------------------------------------------------------------------

    def _create_kspace_dirs(self):
        """Create the k-space directory and set cache file path.

        Called from :meth:`create_dirs_filenames` after the base class has
        set up the observable-specific saving directory.
        """
        self.k_space_dir = os_path_join(
            self.directory_tree["postprocessing"]["path"], "k_space_data"
        )
        if not os_path_exists(self.k_space_dir):
            mkdir(self.k_space_dir)

        # Single shared HDF5 file for ALL k-space intermediate data.
        # Using job_id makes it unique to a simulation run, so multiple
        # observables computed from the same run share the same nkt/vkt.
        self.kspace_cache_path = os_path_join(
            self.k_space_dir, f"kspace_cache_{self.job_id}.h5"
        )

    def create_dirs_filenames(self):
        """Extend base directory setup with k-space paths."""
        super().create_dirs_filenames()
        self._create_kspace_dirs()

    # ------------------------------------------------------------------
    # update_finish override
    # ------------------------------------------------------------------

    def update_finish(self):
        """Extend base ``update_finish`` with k-data loading and frequency setup.

        Order of operations:

        1. Base class creates directories, saves state, writes log.
        2. Load or compute k-vectors via :meth:`_load_or_compute_k_arrays`.
        3. If ``kw_observable``, compute the frequency axis.
        """
        super().update_finish()
        self._load_or_compute_k_arrays()
        if self.kw_observable:
            dt_r             = self.dt * self.dump_step
            self.w_min       = 2.0 * np.pi / (self.block_length * dt_r)
            self.w_max       = np.pi / dt_r
            self.frequencies = fftshift(
                2.0 * np.pi * fftfreq(self.block_length, dt_r)
            )

    # ------------------------------------------------------------------
    # setup_init override
    # ------------------------------------------------------------------

    def setup_init(self, params, **kwargs):
        """Extend base ``setup_init`` with k-harmonic resolution.

        The k-harmonic resolution block is moved here from ``Observable``
        so that the base class has zero k-space awareness.
        """
        super().setup_init(params, **kwargs)
        self._resolve_k_harmonics()

    # ------------------------------------------------------------------
    # K-harmonic resolution
    # ------------------------------------------------------------------

    def _resolve_k_harmonics(self):
        """Compute ``max_k_harmonics`` and ``max_aa_harmonics`` from user inputs.

        Supports two mutually exclusive input styles:

        * **Harmonic count** — user supplies ``max_k_harmonics`` (int or array).
        * **ka-value** — user supplies ``max_ka_value`` (float); harmonics are
          derived from the box geometry.

        All four ``angle_averaging`` modes are handled:
        ``'full'``, ``'principal_axis'``, ``'custom'``.

        Raises
        ------
        AttributeError
            If neither ``max_k_harmonics`` nor ``max_ka_value`` is set, or if
            ``'custom'`` mode is requested without ``max_aa_ka_value`` /
            ``max_aa_harmonics``.
        """
        aa = self.angle_averaging

        # ---- validate inputs --------------------------------------------
        if aa in ("full", "principal_axis"):
            if self.max_k_harmonics is None and self.max_ka_value is None:
                raise AttributeError(
                    "Either max_k_harmonics or max_ka_value must be set."
                )
        elif aa == "custom":
            if self.max_aa_ka_value is None and self.max_aa_harmonics is None:
                raise AttributeError(
                    "Either max_aa_harmonics or max_aa_ka_value must be set "
                    "for angle_averaging='custom'."
                )
        else:
            raise ValueError(
                f"angle_averaging='{aa}' is not recognised. "
                "Choose from ['full', 'principal_axis', 'custom']."
            )

        # ---- ensure max_k_harmonics is a 3-element array ----------------
        if self.max_k_harmonics is not None:
            if not isinstance(self.max_k_harmonics, np.ndarray):
                self.max_k_harmonics = np.array(
                    [self.max_k_harmonics] * 3, dtype=int
                )
            if self.dimensions < 3:
                self.max_k_harmonics[2] = 0

        # ---- branch: user supplied max_k_harmonics ----------------------
        if self.max_k_harmonics is not None:
            if aa == "full":
                self.max_aa_harmonics = self.max_k_harmonics.copy()

            elif aa == "custom":
                if self.max_aa_ka_value is not None:
                    self.max_aa_harmonics = self._ka_to_harmonics(
                        self.max_aa_ka_value, mode="full"
                    )
                # else max_aa_harmonics already set by the user

            elif aa == "principal_axis":
                self.max_aa_harmonics = np.array([0, 0, 0], dtype=int)

        # ---- branch: user supplied max_ka_value -------------------------
        else:
            if aa == "full":
                self.max_k_harmonics  = self._ka_to_harmonics(
                    self.max_ka_value, mode="full"
                )
                self.max_aa_harmonics = self.max_k_harmonics.copy()

            elif aa == "custom":
                self.max_k_harmonics = self._ka_to_harmonics(
                    self.max_ka_value, mode="principal"
                )
                if self.max_aa_ka_value is not None:
                    self.max_aa_harmonics = self._ka_to_harmonics(
                        self.max_aa_ka_value, mode="full"
                    )
                # else max_aa_harmonics already set by the user

            elif aa == "principal_axis":
                self.max_k_harmonics  = self._ka_to_harmonics(
                    self.max_ka_value, mode="principal"
                )
                self.max_aa_harmonics = np.array([0, 0, 0], dtype=int)

        # ---- derive scalar ka limits ------------------------------------
        if aa == "full":
            self.max_ka_value    = (
                2.0 * np.pi * self.a_ws
                * norm(self.max_k_harmonics / self.box_lengths)
            )
            self.max_aa_ka_value = self.max_ka_value

        elif aa == "principal_axis":
            self.max_ka_value    = (
                2.0 * np.pi * self.a_ws
                * self.max_k_harmonics[0] / self.box_lengths[0]
            )
            self.max_aa_ka_value = 0.0

        elif aa == "custom":
            self.max_ka_value    = (
                2.0 * np.pi * self.a_ws
                * self.max_k_harmonics[0] / self.box_lengths[0]
            )
            self.max_aa_ka_value = (
                2.0 * np.pi * self.a_ws
                * norm(self.max_aa_harmonics / self.box_lengths)
            )

    def _ka_to_harmonics(self, ka_value: float, mode: str) -> np.ndarray:
        """Convert a maximum ka value to an integer harmonic count array.

        Parameters
        ----------
        ka_value : float
            Maximum ka value in units of ``1/a_ws``.
        mode : str
            ``'full'``  — divide by sqrt(3) to account for the body diagonal.
            ``'principal'`` — no sqrt(3) factor (principal axis only).

        Returns
        -------
        harmonics : numpy.ndarray, shape (3,), dtype int
        """
        factor = np.sqrt(3.0) if mode == "full" else 1.0
        denom  = 2.0 * np.pi * self.a_ws * factor
        nx = int(ka_value * self.box_lengths[0] / denom)
        ny = int(ka_value * self.box_lengths[1] / denom)
        nz = int(ka_value * self.box_lengths[2] / denom)
        harmonics = np.array([nx, ny, nz], dtype=int)
        if self.dimensions < 3:
            harmonics[2] = 0
        return harmonics

    # ------------------------------------------------------------------
    # K-array HDF5 I/O
    # ------------------------------------------------------------------

    def _k_arrays_are_valid(self) -> bool:
        """Check whether the k-arrays in the cache match the current settings.

        Returns
        -------
        bool
        """
        if not os_path_exists(self.kspace_cache_path):
            return False
        try:
            with h5py.File(self.kspace_cache_path, "r") as f:
                if "k_arrays" not in f:
                    return False
                grp = f["k_arrays"]
                return (
                    grp.attrs.get("angle_averaging") == self.angle_averaging
                    and np.array_equal(
                        grp.attrs.get("max_k_harmonics"),
                        self.max_k_harmonics,
                    )
                    and np.array_equal(
                        grp.attrs.get("max_aa_harmonics"),
                        self.max_aa_harmonics,
                    )
                )
        except Exception:
            return False

    def _save_k_arrays(self):
        """Write k-vector arrays to the shared HDF5 cache.

        Opens the file in append mode so existing ``/nkt`` or ``/vkt`` groups
        are untouched.  The ``/k_arrays`` group is deleted first if it exists
        so that a stale set of k-vectors is never silently reused.
        """
        mode = "a" if os_path_exists(self.kspace_cache_path) else "w"
        with h5py.File(self.kspace_cache_path, mode) as f:
            # Remove stale k-arrays before rewriting
            if "k_arrays" in f:
                del f["k_arrays"]

            grp = f.require_group("k_arrays")

            # Numeric arrays
            grp.create_dataset(
                "k_list",
                data=self.k_list,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "k_harmonics",
                data=self.k_harmonics[:, :3].astype(int),
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "k_counts",
                data=self.k_counts,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "k_values",
                data=self.k_values,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.create_dataset(
                "ka_values",
                data=self.ka_values,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )

            # Cache-invalidation attrs
            grp.attrs["angle_averaging"]   = self.angle_averaging
            grp.attrs["max_k_harmonics"]   = self.max_k_harmonics
            grp.attrs["max_aa_harmonics"]  = self.max_aa_harmonics

    def _load_k_arrays(self):
        """Load k-vector arrays from the shared HDF5 cache into instance attrs."""
        with h5py.File(self.kspace_cache_path, "r") as f:
            grp              = f["k_arrays"]
            self.k_list      = grp["k_list"][:]
            self.k_harmonics = grp["k_harmonics"][:]
            self.k_counts    = grp["k_counts"][:]
            self.k_values    = grp["k_values"][:]
            self.ka_values   = grp["ka_values"][:]
        self.no_ka_values = len(self.ka_values)

    def _compute_k_arrays(self):
        """Compute k-vector arrays from scratch and save them to the cache."""
        self.k_list, self.k_counts, k_unique, self.k_harmonics = kspace_setup(
            self.box_lengths,
            self.angle_averaging,
            self.max_k_harmonics,
            self.max_aa_harmonics,
        )
        self.ka_values   = 2.0 * np.pi * k_unique * self.a_ws
        self.k_values    = 2.0 * np.pi * k_unique
        self.no_ka_values = len(self.ka_values)
        self._save_k_arrays()

    def _load_or_compute_k_arrays(self):
        """Load k-arrays from cache if valid, otherwise compute and cache them."""
        if self._k_arrays_are_valid():
            self._load_k_arrays()
        else:
            self._compute_k_arrays()

    # ------------------------------------------------------------------
    # n(k,t) cache
    # ------------------------------------------------------------------

    def _nkt_is_valid(self) -> bool:
        """Check whether the cached n(k,t) matches the current simulation.

        Returns
        -------
        bool
        """
        if not os_path_exists(self.kspace_cache_path):
            return False
        try:
            with h5py.File(self.kspace_cache_path, "r") as f:
                if "nkt" not in f:
                    return False
                grp = f["nkt"]
                # Check simulation identity and k-vector consistency
                return (
                    grp.attrs.get("no_dumps")  == self.no_dumps
                    and grp.attrs.get("h5md_path") == self.h5md_filepath
                    and grp["data"].shape[1]       == len(self.k_list)
                )
        except Exception:
            return False

    def _save_nkt(self, nkt: np.ndarray, time: np.ndarray):
        """Write n(k,t) to the shared HDF5 cache.

        Parameters
        ----------
        nkt : numpy.ndarray, complex128, shape (no_species, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        mode = "a" if os_path_exists(self.kspace_cache_path) else "w"
        with h5py.File(self.kspace_cache_path, mode) as f:
            if "nkt" in f:
                del f["nkt"]

            grp = f.require_group("nkt")
            grp.create_dataset(
                "data",
                data=nkt,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.attrs["_DIMS"]     = ["species", "k", "time"]
            grp.attrs["no_dumps"]  = self.no_dumps
            grp.attrs["h5md_path"] = self.h5md_filepath

            # Species names as variable-length UTF-8 strings
            dt = h5py.string_dtype()
            grp.create_dataset(
                "species",
                data=np.array(self.species_names, dtype=object),
                dtype=dt,
            )

            # Write shared time axis if not already present
            if "time" not in f:
                f.create_dataset("time", data=time)

    def _compute_nkt(self) -> tuple:
        """Compute n(k,t) from the h5md file and save to the cache.

        Returns
        -------
        nkt : numpy.ndarray, complex128, shape (no_species, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        from tqdm import tqdm as _tqdm

        no_k  = len(self.k_list)
        k_xyz = self.k_list[:, :3]
        nkt   = np.zeros(
            (len(self.species_names), no_k, self.no_dumps), dtype=np.complex128
        )
        time = np.zeros(self.no_dumps)

        with h5py.File(self.h5md_filepath, "r") as h5:
            for it in _tqdm(
                range(self.no_dumps),
                desc="Computing n(k,t)",
                disable=not self.verbose,
            ):
                pos       = h5["particles/pos"][it, :, :]
                nkt[:, :, it] = _calc_nk_vectorised(pos, k_xyz, self.species_num)
                time[it]  = h5["particles/time"][it]

        self._save_nkt(nkt, time)
        return nkt, time

    def _get_nkt(self) -> tuple:
        """Return n(k,t) and time, computing and caching if necessary.

        Returns
        -------
        nkt : numpy.ndarray, complex128, shape (no_species, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        if self._nkt_is_valid():
            with h5py.File(self.kspace_cache_path, "r") as f:
                return f["nkt/data"][:], f["time"][:]
        return self._compute_nkt()

    # ------------------------------------------------------------------
    # v(k,t) cache
    # ------------------------------------------------------------------

    def _vkt_is_valid(self) -> bool:
        """Check whether the cached v(k,t) matches the current simulation.

        Returns
        -------
        bool
        """
        if not os_path_exists(self.kspace_cache_path):
            return False
        try:
            with h5py.File(self.kspace_cache_path, "r") as f:
                if "vkt" not in f:
                    return False
                grp = f["vkt"]
                return (
                    grp.attrs.get("no_dumps")  == self.no_dumps
                    and grp.attrs.get("h5md_path") == self.h5md_filepath
                    and grp["data"].shape[2]       == len(self.k_list)
                )
        except Exception:
            return False

    def _save_vkt(self, vkt: np.ndarray, time: np.ndarray):
        """Write v(k,t) to the shared HDF5 cache.

        Parameters
        ----------
        vkt : numpy.ndarray, complex128, shape (no_species, 3, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        mode = "a" if os_path_exists(self.kspace_cache_path) else "w"
        with h5py.File(self.kspace_cache_path, mode) as f:
            if "vkt" in f:
                del f["vkt"]

            grp = f.require_group("vkt")
            grp.create_dataset(
                "data",
                data=vkt,
                compression=_HDF5_COMPRESSION,
                compression_opts=_HDF5_COMPRESSION_OPTS,
            )
            grp.attrs["_DIMS"]     = ["species", "component", "k", "time"]
            grp.attrs["no_dumps"]  = self.no_dumps
            grp.attrs["h5md_path"] = self.h5md_filepath

            dt = h5py.string_dtype()
            grp.create_dataset(
                "species",
                data=np.array(self.species_names, dtype=object),
                dtype=dt,
            )
            comp_labels = self.dim_labels[: self.dimensions]
            grp.create_dataset(
                "component",
                data=np.array(comp_labels, dtype=object),
                dtype=dt,
            )

            if "time" not in f:
                f.create_dataset("time", data=time)

    def _compute_vkt(self) -> tuple:
        """Compute v(k,t) from the h5md file and save to the cache.

        Returns
        -------
        vkt : numpy.ndarray, complex128, shape (no_species, 3, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        from tqdm import tqdm as _tqdm

        no_k  = len(self.k_list)
        k_xyz = self.k_list[:, :3]
        vkt   = np.zeros(
            (len(self.species_names), self.dimensions, no_k, self.no_dumps),
            dtype=np.complex128,
        )
        time = np.zeros(self.no_dumps)

        with h5py.File(self.h5md_filepath, "r") as h5:
            for it in _tqdm(
                range(self.no_dumps),
                desc="Computing v(k,t)",
                disable=not self.verbose,
            ):
                pos = h5["particles/pos"][it, :, :]
                vel = h5["particles/vel"][it, :, :]
                vkt[:, :, :, it] = _calc_vk_vectorised(
                    pos, vel, k_xyz, self.species_num
                )
                time[it] = h5["particles/time"][it]

        self._save_vkt(vkt, time)
        return vkt, time

    def _get_vkt(self) -> tuple:
        """Return v(k,t) and time, computing and caching if necessary.

        Returns
        -------
        vkt : numpy.ndarray, complex128, shape (no_species, 3, no_k, no_dumps)
        time : numpy.ndarray, shape (no_dumps,)
        """
        if self._vkt_is_valid():
            with h5py.File(self.kspace_cache_path, "r") as f:
                return f["vkt/data"][:], f["time"][:]
        return self._compute_vkt()

    # ------------------------------------------------------------------
    # Direct cache access (slice-by-slice reads for derived observables)
    # ------------------------------------------------------------------

    def _open_kspace_cache(self, mode: str = "r") -> h5py.File:
        """Open the shared k-space HDF5 cache file.

        Derived observables (DSF, CCF, SSF) use this to read nkt/vkt
        one slice at a time without loading the full array into RAM.

        Parameters
        ----------
        mode : str
            h5py file mode, default ``'r'``.

        Returns
        -------
        h5py.File
            Caller is responsible for closing (use as context manager).

        Example
        -------
        >>> with self._open_kspace_cache() as f:
        ...     nkt_slice = f["nkt/data"][:, :, start:end]
        """
        return h5py.File(self.kspace_cache_path, mode)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _invalidate_nkt(self):
        """Delete only the ``/nkt`` group from the shared cache."""
        if os_path_exists(self.kspace_cache_path):
            with h5py.File(self.kspace_cache_path, "a") as f:
                if "nkt" in f:
                    del f["nkt"]

    def _invalidate_vkt(self):
        """Delete only the ``/vkt`` group from the shared cache."""
        if os_path_exists(self.kspace_cache_path):
            with h5py.File(self.kspace_cache_path, "a") as f:
                if "vkt" in f:
                    del f["vkt"]

    def _invalidate_k_arrays(self):
        """Delete the ``/k_arrays`` group from the shared cache."""
        if os_path_exists(self.kspace_cache_path):
            with h5py.File(self.kspace_cache_path, "a") as f:
                if "k_arrays" in f:
                    del f["k_arrays"]

    # ------------------------------------------------------------------
    # Recalculation hook
    # ------------------------------------------------------------------

    def _on_recalculate(self, **kwargs):
        """Extend base recalculation hook with k-space cache invalidation.

        If k-vector parameters change, the k-arrays and all derived caches
        (nkt, vkt) are invalidated and recomputed.  If only slicing parameters
        change, only the result store is invalidated (handled by the base class).
        """
        super()._on_recalculate(**kwargs)

        k_params = {
            "angle_averaging", "max_k_harmonics", "max_ka_value",
            "max_aa_harmonics", "max_aa_ka_value",
        }
        if set(kwargs) & k_params:
            self._resolve_k_harmonics()
            self._invalidate_k_arrays()
            self._invalidate_nkt()
            self._invalidate_vkt()
            self._compute_k_arrays()

    # ------------------------------------------------------------------
    # Deprecated legacy methods
    # ------------------------------------------------------------------

    def calc_k_data(self):
        """.. deprecated::
            Use :meth:`_compute_k_arrays` instead.
            K-vectors are now stored in HDF5, not npz.
        """
        warnings.warn(
            "calc_k_data() is deprecated. "
            "K-vectors are now stored in HDF5 via _compute_k_arrays().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._compute_k_arrays()

    def parse_k_data(self):
        """.. deprecated::
            Use :meth:`_load_or_compute_k_arrays` instead.
        """
        warnings.warn(
            "parse_k_data() is deprecated. "
            "Use _load_or_compute_k_arrays() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._load_or_compute_k_arrays()

    # ------------------------------------------------------------------
    # pretty_print_msg override
    # ------------------------------------------------------------------

    def pretty_print_msg(self) -> str:
        """Append k-space information to the base observable summary."""
        msg = super().pretty_print_msg()
        msg += (
            f"\nK-space cache: {self.kspace_cache_path}\n"
            f"\nWave vector parameters:\n"
            f"  angle_averaging : {self.angle_averaging}\n"
            f"  max_k_harmonics : {self.max_k_harmonics}\n"
        )
        if self.ka_values is not None:
            msg += (
                f"  k_min = {self.ka_values[0]:.6e} / a_ws\n"
                f"  k_max = {self.ka_values[-1]:.6e} / a_ws\n"
                f"  no_ka_values = {self.no_ka_values}\n"
            )
        if self.kw_observable and self.w_min is not None:
            msg += (
                f"\nFrequency axis:\n"
                f"  dw    = {self.w_min / self.total_plasma_frequency:.6e} w_p\n"
                f"  w_max = {self.w_max / self.total_plasma_frequency:.6e} w_p\n"
            )
        return msg


# ---------------------------------------------------------------------------
# Vectorised kernel helpers
# ---------------------------------------------------------------------------


def _calc_nk_vectorised(
    pos_data: np.ndarray,
    k_list: np.ndarray,
    species_np: np.ndarray,
) -> np.ndarray:
    """Compute n_A(k) = Σ_{j∈A} exp(-i k·r_j) for all species and k-vectors.

    Parameters
    ----------
    pos_data : numpy.ndarray, shape (total_particles, 3)
    k_list   : numpy.ndarray, shape (no_k, 3)   — raw k-vectors (not 2π-scaled)
    species_np : numpy.ndarray, shape (no_species,)  — particle counts per species

    Returns
    -------
    nk : numpy.ndarray, complex128, shape (no_species, no_k)
    """
    no_k     = len(k_list)
    n_species = len(species_np)
    nk       = np.zeros((n_species, no_k), dtype=np.complex128)
    sp_start = 0
    for i, sp in enumerate(species_np):
        sp_end    = sp_start + sp
        pos_sp    = pos_data[sp_start:sp_end, :]          # (sp, 3)
        kr        = 2.0 * np.pi * (pos_sp @ k_list.T)        # (sp, no_k)
        nk[i]     = np.exp(-1j * kr).sum(axis=0)          # (no_k,)
        sp_start  = sp_end
    return nk


def _calc_vk_vectorised(
    pos_data: np.ndarray,
    vel_data: np.ndarray,
    k_list: np.ndarray,
    species_np: np.ndarray,
) -> np.ndarray:
    """Compute v_{A,d}(k) = Σ_{j∈A} v_{j,d} exp(-i k·r_j) for all species.

    Parameters
    ----------
    pos_data   : numpy.ndarray, shape (total_particles, 3)
    vel_data   : numpy.ndarray, shape (total_particles, 3)
    k_list     : numpy.ndarray, shape (no_k, 3)
    species_np : numpy.ndarray, shape (no_species,)

    Returns
    -------
    vk : numpy.ndarray, complex128, shape (no_species, 3, no_k)
    """
    no_k      = len(k_list)
    n_species  = len(species_np)
    vk        = np.zeros((n_species, 3, no_k), dtype=np.complex128)
    sp_start  = 0
    for i, sp in enumerate(species_np):
        sp_end   = sp_start + sp
        pos_sp   = pos_data[sp_start:sp_end, :]                       # (sp, 3)
        vel_sp   = vel_data[sp_start:sp_end, :]                       # (sp, 3)
        phase    = np.exp(-1j * 2.0 * np.pi * (pos_sp @ k_list.T))      # (sp, no_k)
        vk[i]    = vel_sp.T @ phase                                    # (3, no_k)
        sp_start = sp_end
    return vk


# ---------------------------------------------------------------------------
# kspace_setup — moved here from base.py (no longer needed in Observable)
# ---------------------------------------------------------------------------


def kspace_setup(
    box_lengths: np.ndarray,
    angle_averaging: str,
    max_k_harmonics: np.ndarray,
    max_aa_harmonics: np.ndarray,
) -> tuple:
    """Calculate all allowed k vectors for the given geometry and averaging scheme.

    Parameters
    ----------
    box_lengths      : numpy.ndarray, shape (3,)
    angle_averaging  : str — ``'full'``, ``'principal_axis'``, or ``'custom'``
    max_k_harmonics  : numpy.ndarray, shape (3,), dtype int
    max_aa_harmonics : numpy.ndarray, shape (3,), dtype int

    Returns
    -------
    k_arr     : numpy.ndarray, shape (no_k, 5)
                Columns: [kx, ky, kz, |k|, k_index]
    k_counts  : numpy.ndarray, shape (no_ka,)
    k_unique  : numpy.ndarray, shape (no_ka,)
    harmonics : numpy.ndarray, shape (no_k, 4)
                Columns: [nx, ny, nz, k_index]
    """
    if angle_averaging == "full":
        first_non_zero = 1
        k_arr = [
            np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]   
        harmonics = [
            np.array([i, j, k], dtype=int)
            for i in range(max_k_harmonics[0] + 1)
            for j in range(max_k_harmonics[1] + 1)
            for k in range(max_k_harmonics[2] + 1)
        ]

    elif angle_averaging == "principal_axis":
        first_non_zero = 0
        k_arr = [
            np.array([i / box_lengths[0], 0.0, 0.0])
            for i in range(1, max_k_harmonics[0] + 1)
        ]
        harmonics = [
            np.array([i, 0, 0], dtype=int)
            for i in range(1, max_k_harmonics[0] + 1)
        ]
        k_arr = np.append(
            k_arr,
            [np.array([0.0, i / box_lengths[1], 0.0])
             for i in range(1, max_k_harmonics[1] + 1)],
            axis=0,
        )
        harmonics = np.append(
            harmonics,
            [np.array([0, i, 0], dtype=int)
             for i in range(1, max_k_harmonics[1] + 1)],
            axis=0,
        )
        k_arr = np.append(
            k_arr,
            [np.array([0.0, 0.0, i / box_lengths[2]])
             for i in range(1, max_k_harmonics[2] + 1)],
            axis=0,
        )
        harmonics = np.append(
            harmonics,
            [np.array([0, 0, i], dtype=int)
             for i in range(1, max_k_harmonics[2] + 1)],
            axis=0,
        )

    elif angle_averaging == "custom":
        first_non_zero = 1
        k_arr = [
            np.array([i / box_lengths[0], j / box_lengths[1], k / box_lengths[2]])
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]
        harmonics = [
            np.array([i, j, k], dtype=int)
            for i in range(max_aa_harmonics[0] + 1)
            for j in range(max_aa_harmonics[1] + 1)
            for k in range(max_aa_harmonics[2] + 1)
        ]
        # Append principal-axis-only k-vectors beyond the aa range
        for axis, box_l, aa_max, k_max in zip(
            range(3),
            box_lengths,
            max_aa_harmonics,
            max_k_harmonics,
        ):
            extra_k   = []
            extra_h   = []
            for i in range(aa_max + 1, k_max + 1):
                v    = [0.0, 0.0, 0.0]
                h    = [0,   0,   0  ]
                v[axis] = i / box_l
                h[axis] = i
                extra_k.append(np.array(v))
                extra_h.append(np.array(h, dtype=int))
            if extra_k:
                k_arr     = np.append(k_arr,     extra_k, axis=0)
                harmonics = np.append(harmonics, extra_h, axis=0)

    else:
        raise ValueError(f"angle_averaging='{angle_averaging}' not recognised.")

    k_arr     = np.array(k_arr)
    harmonics = np.array(harmonics)

    k_mag    = np.sqrt((k_arr ** 2).sum(axis=1, keepdims=True))
    harm_mag = np.sqrt((harmonics.astype(float) ** 2).sum(axis=1, keepdims=True))

    # Collapse near-equal magnitudes to avoid spurious unique values
    for i in range(len(k_mag) - 1):
        if abs(k_mag[i] - k_mag[i + 1]) < 2.0e-5:
            k_mag[i + 1] = k_mag[i]

    k_arr     = np.concatenate([k_arr,     k_mag],    axis=1)   # (no_k, 4)
    harmonics = np.concatenate([harmonics, harm_mag], axis=1)   # (no_k, 4)

    ind       = np.argsort(k_arr[:, -1])
    k_arr     = k_arr[ind]
    harmonics = harmonics[ind]

    k_unique, k_counts = np.unique(
        k_arr[first_non_zero:, -1], return_counts=True
    )
    k_index   = np.repeat(range(len(k_counts)), k_counts)[:, np.newaxis]

    k_arr     = np.concatenate(
        [k_arr[first_non_zero:, :], k_index], axis=1
    )                                                            # (no_k, 5)
    harmonics = np.concatenate(
        [harmonics[first_non_zero:, :3].astype(int), k_index], axis=1
    )                                                            # (no_k, 4)

    return k_arr, k_counts, k_unique, harmonics