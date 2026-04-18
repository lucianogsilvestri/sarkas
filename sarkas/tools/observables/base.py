"""
Observable base class for sarkas post-processing.

All I/O uses HDF5 (h5py) directly. Result stores live at:
    <saving_dir>/<job_id>.h5
"""

import inspect
import warnings
import xarray as xr
from copy import deepcopy
from IPython import get_ipython
from os import mkdir, remove as os_remove
from os.path import exists as os_path_exists
from os.path import join as os_path_join

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import h5py
import numpy as np
from numpy import (
    array,
    ndarray,
    pi,
    sqrt,
    trapz,
)

from ...utilities.io import PortableStateSaver, print_to_logger
from ...utilities.timing import datetime_stamp, SarkasTimer

UNITS = [
    {
        "Energy": "J",
        "Heat Flux": "J/s",
        "Time": "s",
        "Length": "m",
        "Charge": "C",
        "Temperature": "K",
        "ElectronVolt": "eV",
        "Mass": "kg",
        "Magnetic Field": "T",
        "Current": "A",
        "Power": "erg/s",
        "Pressure": "Pa",
        "Electrical Conductivity": "S/m",
        "Diffusion": r"m$^2$/s",
        "InterDiffusion": r"m$^2$/s",
        "Viscosity": r"kg/m-s",
        "Thermal Conductivity": r"J/m-s-K",
        "none": "",
    },
    {
        "Energy": "erg",
        "Heat Flux": "erg/s",
        "Time": "s",
        "Length": "m",
        "Charge": "esu",
        "Temperature": "K",
        "ElectronVolt": "eV",
        "Mass": "g",
        "Magnetic Field": "G",
        "Current": "esu/s",
        "Power": "erg/s",
        "Pressure": "Ba",
        "Electrical Conductivity": "mho/m",
        "Diffusion": r"m$^2$/s",
        "InterDiffusion": r"m$^2$/s",
        "Viscosity": r"g/cm-s",
        "Thermal Conductivity": r"erg/cm-s-K",
        "none": "",
    },
]

PREFIXES = {
    "y": 1.0e-24,
    "z": 1.0e-21,
    "a": 1.0e-18,
    "f": 1.0e-15,
    "p": 1.0e-12,
    "n": 1.0e-9,
    r"$\mu$": 1.0e-6,
    "m": 1.0e-3,
    "c": 1.0e-2,
    "": 1.0,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
    "E": 1e18,
    "Z": 1e21,
    "Y": 1e24,
}

# HDF5 compression settings used throughout
_HDF5_COMPRESSION = "gzip"
_HDF5_COMPRESSION_OPTS = 5


def compute_doc(func):
    func.__doc__ = (
        "Routine for computing the observable. See class doc for exact quantities.\n\n"
        "Parameters\n----------\ncalculate_acf : bool\n    Calculate ACF. Default = True.\n"
    )
    return func


def compute_acf_doc(func):
    func.__doc__ = "Routine for computing the observable's autocorrelation function."
    return func


def calc_slices_doc(func):
    func.__doc__ = "Calculate the observable for each slice."
    return func


def calc_acf_slices_doc(func):
    func.__doc__ = "Calculate the observable ACF for each slice."
    return func


def avg_slices_doc(func):
    func.__doc__ = "Average observable over slices."
    return func


def avg_acf_slices_doc(func):
    func.__doc__ = "Average observable ACF over slices."
    return func


def setup_doc(func):
    func.__doc__ = (
        "Assign attributes from simulation parameters.\n\n"
        "Parameters\n----------\nparams : sarkas.core.Parameters\n"
        "phase : str, optional\nno_slices : int, optional\n**kwargs\n"
    )
    return func


def arg_update_doc(func):
    func.__doc__ = "Update observable specific attributes and call update_finish."
    return func


class Observable:
    """Parent class of all observables. Uses HDF5 (h5py) for all I/O.

    HDF5 Store Layout
    -----------------
    Each observable writes to ``<saving_dir>/<job_id>.h5`` with the following
    internal structure::

        /coords/
            <dim_name>      — coordinate arrays (one dataset per dimension)
        /data/
            <var_name>      — data arrays, each with a ``_DIMS`` attribute
                              listing its dimension names in order
        attrs (file-level) — cache-invalidation parameters
                             (no_slices, block_length, dumps_shift, h5md_path, ...)

    String coordinates (e.g. species names) are stored as HDF5 variable-length
    UTF-8 strings using ``h5py.string_dtype()``.

    xarray Integration
    ------------------
    Use :meth:`read_dataset` to open the store as an ``xr.Dataset``.
    Dimension labels are reconstructed from the ``_DIMS`` attribute on each
    dataset, so no xarray-specific metadata needs to be written during
    calculation.
    """

    def __init__(self):
        self.postprocessing_dir = None
        self.mag_no_dumps = None
        self.eq_no_dumps = None
        self.prod_no_dumps = None
        self.no_obs = None
        self.filename_hdf_acf = None
        self.species_index_start = None
        self.filename_hdf_acf_slices = None
        self.filename_hdf_slices = None
        self.filename_hdf = None
        self.__long_name__ = None
        self.__name__ = None
        self.saving_dir = None
        self.phase = "production"
        self.multi_run_average = False
        self.dimensional_average = False
        self.runs = 1
        self.no_slices = 1
        self.block_length = None
        self.timesteps_per_plasma_period = None
        self.timesteps_per_slice = None
        self.independent_slices = True
        self.timesteps_shift = None
        self.plasma_period = None
        self.plasma_periods_per_slice = None
        self.plasma_periods_shift = None
        self.screen_output = True
        self.timer = SarkasTimer()

        self.dim_labels = ["X", "Y", "Z"]
        self.acf_observable = False
        # HDF5 store path (set by create_dirs_filenames)
        self.hdf_store_path = None
        # Frozen setup-time attributes — populated at end of setup_init.
        # recalculate() refuses to modify any of these.
        self._setup_frozen_attrs: set = set()

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = "Observable( " + self.__class__.__name__ + "\n"
        exclude = {"hdf_store_path", "_setup_frozen_attrs"}
        for key, value in sortedDict.items():
            if key not in exclude:
                disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    # ------------------------------------------------------------------
    # HDF5 I/O helpers
    # ------------------------------------------------------------------

    def _preallocate_store(
        self,
        variable_shapes: dict,
        coords: dict,
        attrs: dict = None,
    ):
        """Create the HDF5 result file with full coordinates and NaN-filled data.

        The file is written once before the slice loop begins.  Each call to
        :meth:`_write_slice` then fills in one slice without touching the rest
        of the file, keeping peak RAM to a single slice.

        Parameters
        ----------
        variable_shapes : dict
            Mapping of ``{var_name: shape}`` where *shape* does **not** include
            the ``"slice"`` dimension.  Example: ``{"rdf": (num_pairs, no_bins)}``.
        coords : dict
            Ordered mapping of ``{dim_name: coordinate_array}`` for **all**
            dimensions including ``"slice"``.  The order of keys defines the
            axis order of each data variable.
        attrs : dict, optional
            File-level attributes written for cache invalidation
            (e.g. ``no_slices``, ``cutoff_radius``, ``h5md_path``).

        Notes
        -----
        String coordinates are stored as HDF5 variable-length UTF-8 strings.
        All numeric coordinates are stored as float64.
        Data variables are chunked ``(*shape, 1)`` — one chunk per slice —
        and pre-filled with ``np.nan`` so unwritten slices are distinguishable
        from computed zeros.
        """
        with h5py.File(self.hdf_store_path, "w") as f:
            # ---- coordinates ------------------------------------------------
            coord_grp = f.require_group("coords")
            for dim, values in coords.items():
                vals = np.asarray(values)
                if vals.dtype.kind in ("U", "O"):
                    # Variable-length UTF-8 strings
                    dt = h5py.string_dtype()
                    coord_grp.create_dataset(
                        dim, data=np.array(values, dtype=object), dtype=dt
                    )
                else:
                    coord_grp.create_dataset(dim, data=vals)
                coord_grp[dim].attrs["_is_coord"] = True

            # ---- data variables ---------------------------------------------
            data_grp = f.require_group("data")
            dim_names = list(coords.keys())
            for var_name, shape in variable_shapes.items():
                full_shape = shape
                chunk_shape = shape
                ds = data_grp.create_dataset(
                    var_name,
                    shape=full_shape,
                    dtype="float64",
                    chunks=chunk_shape,
                    compression=_HDF5_COMPRESSION,
                    compression_opts=_HDF5_COMPRESSION_OPTS,
                    fillvalue=np.nan,
                )
                # _DIMS allows read_dataset to reconstruct xarray dimensions
                ds.attrs["_DIMS"] = dim_names

            # ---- file-level cache-invalidation attrs ------------------------
            if attrs:
                for k, v in attrs.items():
                    f.attrs[k] = v

    def _write_slice(self, slice_data: dict, isl: int):
        """Write one slice of data into the pre-allocated HDF5 store.

        Must be called **after** :meth:`_preallocate_store`.  The file is
        opened in append mode so only the target chunk is touched.

        Parameters
        ----------
        slice_data : dict
            Mapping of ``{var_name: np.ndarray}`` where each array has the
            shape specified in :meth:`_preallocate_store` (i.e. **without**
            the slice dimension).
        isl : int
            Index along the ``"slice"`` axis to write into.
        """
        with h5py.File(self.hdf_store_path, "a") as f:
            for var_name, arr in slice_data.items():
                f["data"][var_name][..., isl] = arr

    def _write_mean_std(self, var_names: list, ddof: int = 1):
        """Compute and append mean/std over the slice dimension to the HDF5 store.

        Opens the store with :meth:`read_dataset`, computes statistics with
        xarray, then writes the results back into the ``/data`` group.

        Parameters
        ----------
        var_names : list of str
            Variables in the store for which mean and std should be computed.
        ddof : int
            Delta degrees of freedom passed to ``xr.DataArray.std``.  Use
            ``min(1, no_slices - 1)`` to handle the single-slice case.
        """
        ds = self.read_dataset()
        with h5py.File(self.hdf_store_path, "a") as f:
            grp = f["data"]
            for var in var_names:
                if "slice" not in ds[var].dims:
                    continue
                mean_arr = ds[var].mean("slice").values
                std_arr  = ds[var].std( "slice", ddof=ddof).values
                # Dimension names without "slice"
                dims_no_slice = [d for d in ds[var].dims if d != "slice"]
                for stat_name, stat_arr in [
                    (f"mean_{var}", mean_arr),
                    (f"std_{var}",  std_arr),
                ]:
                    if stat_name in grp:
                        del grp[stat_name]
                    out = grp.create_dataset(
                        stat_name,
                        data=stat_arr,
                        compression=_HDF5_COMPRESSION,
                        compression_opts=_HDF5_COMPRESSION_OPTS,
                    )
                    out.attrs["_DIMS"] = dims_no_slice

    def read_dataset(self) -> xr.Dataset:
        """Open the observable's HDF5 store as an ``xr.Dataset``.

        Reconstructs dimension labels and coordinates from the ``/coords``
        group and the ``_DIMS`` attribute stored on each dataset in ``/data``.
        String coordinates are decoded to Python ``str``.

        Returns
        -------
        xr.Dataset
            All variables in ``/data`` with proper named dimensions and
            coordinate arrays attached.
        """
        variables = {}
        with h5py.File(self.hdf_store_path, "r") as f:
            # Load coordinates first
            coords = {}
            for dim, ds in f["coords"].items():
                raw = ds[:]
                # Decode variable-length strings to Python str
                if h5py.check_string_dtype(ds.dtype):
                    raw = np.array([v.decode() if isinstance(v, bytes) else v for v in raw])
                coords[dim] = raw

            # Build DataArrays from data variables
            for var_name, ds in f["data"].items():
                dims = list(ds.attrs["_DIMS"])
                coord_subset = {d: coords[d] for d in dims if d in coords}
                variables[var_name] = xr.DataArray(
                    ds[:],
                    dims=dims,
                    coords=coord_subset,
                )

        return xr.Dataset(variables)

    def _store_is_valid(self) -> bool:
        """Return True if the HDF5 store exists and its attrs match current parameters.

        Subclasses that add cache-invalidation parameters (e.g. ``cutoff_radius``
        for the RDF) should override this method, call ``super()._store_is_valid()``,
        and then check their own attrs.

        Returns
        -------
        bool
        """
        if not os_path_exists(self.hdf_store_path):
            return False
        try:
            with h5py.File(self.hdf_store_path, "r") as f:
                return (
                    f.attrs.get("no_slices")    == self.no_slices
                    and f.attrs.get("block_length") == self.block_length
                    and f.attrs.get("dumps_shift")  == self.dumps_shift
                    and f.attrs.get("h5md_path")    == self.h5md_filepath
                )
        except Exception:
            return False

    def _invalidate_store(self):
        """Delete the HDF5 store file if it exists."""
        if os_path_exists(self.hdf_store_path):
            os_remove(self.hdf_store_path)

    # ------------------------------------------------------------------
    # Recalculation
    # ------------------------------------------------------------------

    def _on_recalculate(self, **kwargs):
        """Recalculate the observable with new parameters.

        Any attribute that was not frozen at setup time can be updated here.
        After updating, :meth:`_on_recalculate` is called so subclasses can
        recompute derived quantities, then the store is invalidated and
        :meth:`compute` is called.

        Parameters
        ----------
        **kwargs
            Attribute name / new value pairs.  Attempting to change a
            setup-frozen attribute raises ``ValueError`` — call
            :meth:`setup` again instead.

        Raises
        ------
        ValueError
            If any key in *kwargs* is in ``_setup_frozen_attrs``.
        """
        frozen_attempts = set(kwargs) & self._setup_frozen_attrs
        if frozen_attempts:
            raise ValueError(
                f"Cannot recalculate with setup-time parameters: {frozen_attempts}. "
                f"Call setup() again instead."
            )
        self.__dict__.update(kwargs)
        slicing_keys = {
            "no_slices", "timesteps_per_slice", "timesteps_shift",
            "plasma_periods_per_slice", "plasma_periods_shift",
            "independent_slices",
        }
        if set(kwargs) & slicing_keys:
            self.update_block_attributes(**{k: kwargs[k] for k in set(kwargs) & slicing_keys})

        self._invalidate_store()

    # ------------------------------------------------------------------
    # Slicing / block attributes
    # ------------------------------------------------------------------

    def update_block_attributes(
        self,
        independent_slices=None,
        no_slices=None,
        timesteps_per_slice=None,
        timesteps_shift=None,
        plasma_periods_per_slice=None,
        plasma_periods_shift=None,
    ):
        """Compute all slice/block attributes from the user's slicing parameters.

        Parameters
        ----------
        independent_slices : bool, optional
        no_slices : int, optional
        timesteps_per_slice : int, optional
        timesteps_shift : int, optional
        plasma_periods_per_slice : float, optional
        plasma_periods_shift : float, optional
        """
        if independent_slices is not None:
            self.independent_slices = independent_slices
        if no_slices is not None:
            self.no_slices = no_slices
        if timesteps_per_slice is not None:
            self.timesteps_per_slice = timesteps_per_slice
        if timesteps_shift is not None:
            self.timesteps_shift = timesteps_shift
        if plasma_periods_per_slice is not None:
            self.plasma_periods_per_slice = plasma_periods_per_slice
        if plasma_periods_shift is not None:
            self.plasma_periods_shift = plasma_periods_shift

        if self.independent_slices:
            if (
                self.timesteps_shift is not None
                or self.plasma_periods_shift is not None
            ):
                raise AttributeError(
                    "timesteps_shift and plasma_periods_shift must be None for "
                    "independent slices — the shift is always equal to the window size."
                )
            if self.no_slices == 1:
                self.timesteps_per_slice = self.no_steps
            elif self.no_slices > 1:
                if self.no_slices >= self.no_dumps:
                    raise AttributeError(
                        f"no_slices ({self.no_slices}) must be less than "
                        f"no_dumps ({self.no_dumps})."
                    )
                if (
                    self.timesteps_per_slice is None
                    and self.plasma_periods_per_slice is None
                ):
                    self.timesteps_per_slice = self.no_steps // self.no_slices
                elif self.timesteps_per_slice is None:
                    self.timesteps_per_slice = int(
                        self.plasma_periods_per_slice
                        * self.timesteps_per_plasma_period
                    )
            else:
                raise AttributeError("no_slices must be a positive integer.")

            self.timesteps_shift = self.timesteps_per_slice

        else:
            if self.timesteps_per_slice is not None:
                pass
            elif self.plasma_periods_per_slice is not None:
                self.timesteps_per_slice = int(
                    self.plasma_periods_per_slice * self.timesteps_per_plasma_period
                )
            else:
                raise AttributeError(
                    "timesteps_per_slice or plasma_periods_per_slice must be defined "
                    "for sliding window slicing."
                )

            if self.timesteps_shift is not None:
                pass
            elif self.plasma_periods_shift is not None:
                self.timesteps_shift = int(
                    self.plasma_periods_shift * self.timesteps_per_plasma_period
                )
            else:
                raise AttributeError(
                    "timesteps_shift or plasma_periods_shift must be defined "
                    "for sliding window slicing."
                )

            if self.timesteps_shift > self.timesteps_per_slice:
                raise AttributeError(
                    f"timesteps_shift ({self.timesteps_shift}) cannot be larger "
                    f"than timesteps_per_slice ({self.timesteps_per_slice})."
                )

            if self.timesteps_per_slice > self.no_steps:
                raise AttributeError(
                    f"timesteps_per_slice ({self.timesteps_per_slice}) exceeds "
                    f"the total number of timesteps ({self.no_steps})."
                )

            self.no_slices = int(
                (self.no_steps - self.timesteps_per_slice) // self.timesteps_shift + 1
            )

            if self.no_slices < 1:
                raise AttributeError(
                    "The combination of timesteps_per_slice and timesteps_shift "
                    "produces zero valid slices."
                )

        self.block_length      = self.timesteps_per_slice // self.dump_step
        self.dumps_shift       = self.timesteps_shift // self.dump_step
        self.dumps_per_slice   = self.block_length

        self.plasma_periods_per_slice = (
            self.timesteps_per_slice / self.timesteps_per_plasma_period
        )
        self.plasma_periods_shift = (
            self.timesteps_shift / self.timesteps_per_plasma_period
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def copy_params(self, params):
        """Copy all non-method attributes from *params* into self."""
        for i, val in params.__dict__.items():
            if not inspect.ismethod(val):
                if isinstance(val, dict):
                    self.__dict__[i] = deepcopy(val)
                elif isinstance(val, ndarray):
                    self.__dict__[i] = val.copy()
                else:
                    self.__dict__[i] = val

    def from_dict(self, input_dict: dict):
        """Update instance attributes from a dictionary."""
        self.__dict__.update(input_dict)

    def create_dirs_filenames(self):
        """Create directories and set the HDF5 store path."""
        self.setup_multirun_dirs()
        long_name_no_spaces = self.__long_name__.replace(" ", "")
        saving_dir = os_path_join(
            self.directory_tree["postprocessing"]["path"], long_name_no_spaces
        )
        if not os_path_exists(saving_dir):
            mkdir(saving_dir)

        self.saving_dir = os_path_join(saving_dir, self.phase.capitalize())
        if not os_path_exists(self.saving_dir):
            mkdir(self.saving_dir)

        self.log_file = os_path_join(
            self.saving_dir, f"{long_name_no_spaces}_log_file.out"
        )

        # Primary HDF5 result store — one file per (observable, job_id)
        self.hdf_store_path = os_path_join(
            self.saving_dir, f"{self.job_id}.h5"
        )

    def setup_multirun_dirs(self):
        """Set up dump directory list for single or multi-run averaging."""
        self.dump_dirs_list = []
        if self.multi_run_average:
            for r in range(self.runs):
                dump_dir = os_path_join(
                    f"run{r}",
                    os_path_join(
                        "Simulation", os_path_join(self.phase.capitalize(), "dumps")
                    ),
                )
                dump_dir = os_path_join(self.md_simulations_dir, dump_dir)
                self.dump_dirs_list.append(dump_dir)
            self.postprocessing_dir = os_path_join(
                self.md_simulations_dir, "PostProcessing"
            )
            if not os_path_exists(self.postprocessing_dir):
                mkdir(self.postprocessing_dir)
        else:
            self.dump_dirs_list = [self.dump_dir]

    def setup_init(
        self,
        params,
        phase=None,
        independent_slices=None,
        no_slices=None,
        timesteps_per_slice=None,
        timesteps_shift=None,
        plasma_periods_per_slice=None,
        plasma_periods_shift=None,
        multi_run_average=None,
        dimensional_average=None,
        runs=None,
        **kwargs,
    ):
        """Initialise the observable from simulation parameters.

        This is called by every subclass's ``setup()`` method as the first
        step.  It copies all parameters from *params*, determines phase-specific
        quantities, computes block attributes, and then populates
        ``_setup_frozen_attrs`` so that :meth:`recalculate` knows which
        attributes cannot be changed without a full ``setup()`` call.

        Parameters
        ----------
        params : sarkas.core.Parameters
        phase : str, optional
        independent_slices : bool, optional
        no_slices : int, optional
        timesteps_per_slice : int, optional
        timesteps_shift : int, optional
        plasma_periods_per_slice : float, optional
        plasma_periods_shift : float, optional
        multi_run_average : bool, optional
        dimensional_average : bool, optional
        runs : int, optional
        **kwargs
            Passed through to subclass ``update_args``.
        """
        if phase:
            self.phase = phase.lower()
        if multi_run_average:
            self.multi_run_average = multi_run_average
        if dimensional_average:
            self.dimensional_average = dimensional_average
        if runs:
            self.runs = runs

        name      = self.__name__
        long_name = self.__long_name__
        self.copy_params(params)

        if self.dimensions == 3:
            self.dim_labels = ["X", "Y", "Z"]
        elif self.dimensions == 2:
            self.dim_labels = ["X", "Y"]

        self.__name__      = name
        self.__long_name__ = long_name

        self.dump_dir      = self.directory_tree["postprocessing"][self.phase]["dumps"]["path"]
        self.h5md_filepath = self.h5md_filenames_tree["postprocessing"][self.phase]

        self.prod_no_dumps = params.production_steps // params.prod_dump_step + 1 # +1 to include step 0
        self.eq_no_dumps   = params.equilibration_steps // params.eq_dump_step + 1

        if self.magnetized and self.electrostatic_equilibration:
            self.mag_no_dumps = params.magnetization_steps // params.mag_dump_step + 1

        if self.phase == "equilibration":
            self.no_dumps  = self.eq_no_dumps
            self.dump_step = self.eq_dump_step
            self.no_steps  = self.equilibration_steps + 1
        elif self.phase == "production":
            self.no_dumps  = self.prod_no_dumps
            self.dump_step = self.prod_dump_step
            self.no_steps  = self.production_steps + 1
        elif self.phase == "magnetization":
            self.no_dumps  = self.mag_no_dumps
            self.dump_step = self.mag_dump_step
            self.no_steps  = self.magnetization_steps + 1
            
        self.plasma_period = 2.0 * pi / self.total_plasma_frequency
        self.timesteps_per_plasma_period = int(
            self.plasma_period // self.dt
        )

        self.update_block_attributes(
            independent_slices=independent_slices,
            no_slices=no_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift,
        )

        self.species_index_start = array(
            [0, *self.species_num.cumsum()], dtype=int
        )

        # Build the flat upper-triangular species-pair list and index map.
        # Available to every subclass immediately after setup_init returns.
        self.species_pairs, self.pair_index_map = self._build_species_pairs()
        self.num_pairs = len(self.species_pairs)

        # Freeze setup-time attributes so recalculate() can protect them.
        # These are quantities derived directly from params or from the phase
        # that cannot be changed without re-reading the simulation parameters.
        self._setup_frozen_attrs = {
            "phase",
            "h5md_filepath",
            "dump_dir",
            "box_lengths",
            "box_volume",
            "total_num_ptcls",
            "species_names",
            "species_num",
            "num_species",
            "dt",
            "dump_step",
            "no_dumps",
            "no_steps",
            "plasma_period",
            "timesteps_per_plasma_period",
        }

    def _build_species_pairs(self) -> tuple:
        """Build the flat upper-triangular species-pair list and index map.
 
        Called automatically at the end of :meth:`setup_init` so that every
        subclass has ``self.species_pairs``, ``self.pair_index_map``, and
        ``self.num_pairs`` available without repeating the loop.
 
        Returns
        -------
        species_pairs : list of str
            Upper-triangular pair labels in row-major order,
            e.g. ``['H-H', 'H-He', 'He-He']``.
        pair_index_map : numpy.ndarray, shape (num_species, num_species), dtype int
            Symmetric mapping: ``pair_index_map[i, j]`` → linear pair index.
 
        Notes
        -----
        This is the canonical source of pair labels for **all** observables.
        Subclasses that previously built their own ``_species_pairs`` /
        ``_pair_index_map`` should use ``self.species_pairs`` and
        ``self.pair_index_map`` instead.
        """
        n = len(self.species_names)
        pairs     = []
        index_map = np.zeros((n, n), dtype=int)
        k = 0
        for i in range(n):
            for j in range(i, n):
                pairs.append(f"{self.species_names[i]}-{self.species_names[j]}")
                index_map[i, j] = k
                index_map[j, i] = k
                k += 1
        return pairs, index_map
    
    def update_finish(self):
        """Create directories, parse k-data, write frequencies, and save state.

        Called at the end of every subclass's ``update_args`` method.
        """
        self.create_dirs_filenames()

        self.save_state()

        datetime_stamp(self.log_file)
        msg = self.pretty_print_msg()
        print_to_logger(msg, self.log_file, self.verbose)

    def get_save_format(self):
        """Return the format of any existing saved config file."""
        config_filename  = os_path_join(
            self.saving_dir,
            self.__long_name__.replace(" ", "") + "_config.json",
        )
        pickle_filename  = os_path_join(
            self.saving_dir,
            self.__long_name__.replace(" ", "") + ".pickle",
        )
        if os_path_exists(config_filename):
            return "portable_json"
        elif os_path_exists(pickle_filename):
            return "legacy_pickle"
        return "none"

    def from_json(self):
        """Restore observable state from a JSON config file."""
        config_filename = os_path_join(
            self.saving_dir,
            self.__long_name__.replace(" ", "") + "_config.json",
        )
        if os_path_exists(config_filename):
            saver  = PortableStateSaver()
            config = saver.load_observable_config(config_filename)
            saver.restore_observable_from_config(config, self)
        else:
            old_filename = os_path_join(
                self.saving_dir,
                self.__long_name__.replace(" ", "") + ".pickle",
            )
            if os_path_exists(old_filename):
                import pickle
                with open(old_filename, "rb") as pkl_data:
                    data = pickle.load(pkl_data)
                self.from_dict(data.__dict__)
            else:
                raise FileNotFoundError(
                    f"Neither config file {config_filename} nor pickle file found"
                )

    def save_state(self):
        """Persist observable configuration to a JSON file."""
        saver = PortableStateSaver()
        config_filename = os_path_join(
            self.saving_dir,
            self.__long_name__.replace(" ", "") + "_config.json",
        )
        saver.save_observable(self, config_filename)
        self.filename_config = config_filename

    # ------------------------------------------------------------------
    # Deprecated methods
    # ------------------------------------------------------------------

    def grab_sim_data(self, pva="vel"):
        """.. deprecated::
            Read the necessary data directly from the h5md file or use read_dataset() to read from the observable's HDF5 store.
        """
        warnings.warn(
            "grab_sim_data() is deprecated and has no effect. "
            "Read the necessary data directly from the h5md file or use read_dataset() to read from the observable's HDF5 store.",
            DeprecationWarning,
            stacklevel=2,
        )

    def parse(self, acf_data=False):
        """.. deprecated::
            Use :meth:`read_dataset` instead.
        """
        warnings.warn(
            "parse() is deprecated. Use read_dataset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def load_hdf(self, acf_data=False):
        """.. deprecated::
            Use :meth:`read_dataset` instead.
        """
        warnings.warn(
            "load_hdf() is deprecated. Use read_dataset() instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def load_simulation_data(self):
        """.. deprecated::
            Read directly from the h5md file.
        """
        warnings.warn(
            "load_simulation_data() is deprecated. "
            "Read from h5md directly or use read_dataset() to read from the observable's HDF5 store.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, scaling=None, acf=False, figname=None, show=False, **kwargs):
        """.. deprecated::
            Plotting is now handled by the user with the data read from :meth:`read_dataset`.
        """

    def pretty_print_msg(self):
        """Return a formatted summary string for the log file."""
        name = " " + self.__long_name__ + " "
        msg  = f"\n\n{name:=^70}\n"
        if self.hdf_store_path:
            msg += f"Data saved in: \n {self.hdf_store_path}\n"

        dtau    = self.dt * self.dump_step
        tau     = self.dt * self.timesteps_per_slice
        tau_wp  = int(round(tau / self.plasma_period))
        msg += (
            f"\nTime Series Data:\n"
            f"No. of blocks = {self.no_slices}\n"
            f"No. dumps per block = {int(self.block_length)}\n"
            f"Total time per block: T = {tau:.4e} {self.units_dict['time']}"
            f" ~ {tau_wp} plasma periods\n"
            f"Time interval: dt = {dtau:.4e} {self.units_dict['time']}"
            f" ~ {dtau / self.plasma_period:.1e} plasma period"
        )

        return msg

    def integrate_normalized_acf_squared(self, time, data):
        """Integrate the square of the normalised ACF."""
        data_0 = data[0]
        tau_2  = 2.0 * trapz((data / data_0) ** 2, x=time)
        return tau_2


# ---------------------------------------------------------------------------
# Module-level utility functions
# ---------------------------------------------------------------------------



def load_from_restart(fldr, it):
    """..deprecated::
        Use :meth:`read_dataset` to read from the observable's HDF5 store instead.
    """
    warnings.warn(
        "load_from_restart() is deprecated. Use read_dataset() to read from the observable's HDF5 store instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    

def plot_labels(xdata, ydata, xlbl, ylbl, units):
    """Create plot labels with correct SI prefixes.

    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    xlbl : str
    ylbl : str
    units : str  ``'cgs'`` or ``'mks'``

    Returns
    -------
    xmultiplier, ymultiplier, xprefix, yprefix, xlabel, ylabel
    """
    units_idx = 1 if units.lower() == "cgs" else 0

    def _best_prefix(data, label):
        magnitude = abs(data).max() if len(data) > 0 else 1.0
        best_p = ""
        best_v = 1.0
        for p, v in PREFIXES.items():
            if v <= magnitude and v > best_v:
                best_p = p
                best_v = v
        unit = UNITS[units_idx].get(label, "")
        return best_v, best_p, f"[{best_p}{unit}]"

    xmult, xpref, xlabel = _best_prefix(xdata, xlbl)
    ymult, ypref, ylabel = _best_prefix(ydata, ylbl)
    return xmult, ymult, xpref, ypref, xlabel, ylabel


def run_thermalization_tests(
    data, time_array, adf_significance=0.05, kpss_significance=0.05
):
    """Run comprehensive stationarity tests for thermalization verification.

    Parameters
    ----------
    data : numpy.ndarray
    time_array : numpy.ndarray
    adf_significance : float
    kpss_significance : float

    Returns
    -------
    dict
    """
    from arch.unitroot import ADF, KPSS
    import pymannkendall as mk
    from ._kernels import remove_linear_trend

    detrended, intercept, slope = remove_linear_trend(data, time_array)

    linear_fit = intercept + slope * time_array
    mse_fit    = ((data - linear_fit) ** 2).sum() / len(data)
    rmse       = sqrt(mse_fit)

    delta_yt = detrended[1:] - detrended[:-1]
    epsilon  = delta_yt.var(ddof=1)

    adftest    = ADF(detrended, trend="c")
    adf_passed = (
        adftest.pvalue < adf_significance
        and adftest.stat < adftest.critical_values["5%"]
    )

    kpsstest    = KPSS(detrended, trend="c")
    kpss_passed = (
        kpsstest.pvalue > kpss_significance
        and kpsstest.stat < kpsstest.critical_values["5%"]
    )

    mk_test   = mk.original_test(data)
    mk_passed = bool(~mk_test.h)

    return {
        "detrended": detrended,
        "intercept": intercept,
        "slope":     slope,
        "rmse":      rmse,
        "epsilon":   epsilon,
        "adf": {
            "statistic":      adftest.stat,
            "pvalue":         adftest.pvalue,
            "critical_value": adftest.critical_values["5%"],
            "passed":         adf_passed,
        },
        "kpss": {
            "statistic":      kpsstest.stat,
            "pvalue":         kpsstest.pvalue,
            "critical_value": kpsstest.critical_values["5%"],
            "passed":         kpss_passed,
        },
        "mann_kendall": {
            "s":      mk_test.s,
            "pvalue": mk_test.p,
            "tau":    mk_test.Tau,
            "h":      mk_test.h,
            "trend":  mk_test.trend,
            "passed": mk_passed,
        },
        "all_conditions": [adf_passed, kpss_passed, mk_passed],
        "verdict":        all([adf_passed, kpss_passed, mk_passed]),
    }