"""
Spatial observables for sarkas post-processing.

Classes
-------
RadialDistributionFunction : g(r), the pair correlation function
PairDistributionFunction   : 3D pair distribution function (cartesian/cylindrical/spherical)
"""

import warnings
import h5py
import numpy as np
from scipy.special import factorial
from tqdm import tqdm

from .base import (
    Observable,
    calc_slices_doc,
    compute_doc,
    setup_doc,
)
from ...utilities.timing import time_stamp


class RadialDistributionFunction(Observable):
    """Radial Distribution Function g(r).

    Attributes
    ----------
    no_bins : int
        Number of histogram bins.
    dr_rdf : float
        Width of each radial bin.
    ra_values : numpy.ndarray
        Radial bin centres normalised by the Wigner-Seitz radius.
    bin_vol : numpy.ndarray
        Volume (hyper-sphere shell) of each bin.

    HDF5 Store Layout
    -----------------
    ``/coords/r``             — bin-centre distances (m),  shape (no_bins,)
    ``/coords/ra``            — bin-centres / a_ws,        shape (no_bins,)
    ``/coords/species_pair``  — pair labels UTF-8,         shape (num_pairs,)
    ``/coords/slice``         — slice indices,             shape (no_slices,)
    ``/data/rdf``             — shape (num_pairs, no_bins, no_slices)
    ``/data/mean_rdf``        — shape (num_pairs, no_bins)
    ``/data/std_rdf``         — shape (num_pairs, no_bins)
    file attrs                — cache-invalidation metadata
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "rdf"
        self.__long_name__ = "Radial Distribution Function"
        self.rdf_nbins     = None
        self.cutoff_radius = None
        self.rc            = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @setup_doc
    def setup(
        self,
        params,
        phase: str = None,
        independent_slices: bool = None,
        no_slices: int = None,
        timesteps_per_slice: int = None,
        timesteps_shift: int = None,
        plasma_periods_per_slice: int = None,
        plasma_periods_shift: int = None,
        **kwargs,
    ):
        super().setup_init(
            params,
            phase=phase,
            independent_slices=independent_slices,
            no_slices=no_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift,
            **kwargs,
        )
        self._init_rdf_args(**kwargs)

    def _init_rdf_args(self, **kwargs):
        """Initialise RDF-specific attributes and derived quantities.

        Called once from :meth:`setup` and again from :meth:`_on_recalculate`
        whenever RDF-specific parameters change.  This replaces the old
        ``update_args`` pattern with an explicit, named method.

        Parameters
        ----------
        **kwargs
            Any RDF-specific keyword arguments (e.g. ``cutoff_radius``,
            ``rdf_nbins``) passed through from :meth:`setup`.
        """
        self.__dict__.update(kwargs)
        self.cutoff_radius = kwargs.get("cutoff_radius", self.cutoff_radius)

        if self.rdf_nbins is None:
            # Try to infer from an existing h5md checkpoint
            try:
                with h5py.File(self.h5md_filepath, "r") as h5md_file:
                    self.rdf_nbins = h5md_file["observables"]["rdf_hist"][
                        "value"
                    ].shape[-1]
            except KeyError:
                warnings.warn(
                    "rdf_hist not found in HDF5 file. "
                    "Setting rdf_nbins to 0.05 of total number of particles."
                )
                self.rdf_nbins = int(0.05 * self.total_num_particles)

        self.dr_rdf  = self.cutoff_radius / self.rdf_nbins

        self.update_finish()

        # Build species-ID array (integer per particle, used by LinkedCellList)
        self._particles_ids = np.zeros(self.total_num_ptcls, dtype="int64")
        s0 = 0
        for i, n in enumerate(self.species_num):
            self._particles_ids[s0 : s0 + n] = i
            s0 += n

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _store_is_valid(self) -> bool:
        """Check the HDF5 store against all RDF-specific parameters.

        Extends the base class check with ``no_bins`` and ``cutoff_radius``
        so that :meth:`recalculate` automatically invalidates stale results.
        """
        if not super()._store_is_valid():
            return False
        try:
            with h5py.File(self.hdf_store_path, "r") as f:
                return (
                    f.attrs.get("rdf_nbins")       == self.rdf_nbins
                    and f.attrs.get("cutoff_radius") == self.cutoff_radius
                )
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Recalculation hook
    # ------------------------------------------------------------------

    def recalculate(self, **kwargs):
        """Recompute RDF-derived quantities after :meth:`recalculate` updates attrs.

        Handles both RDF-specific parameters (``cutoff_radius``, ``rdf_nbins``)
        and base-class slicing parameters (delegated to
        :meth:`~Observable.update_block_attributes`).
        """
        # Let the base class handle any slicing parameter changes first
        super()._on_recalculate(**kwargs)

        # Recompute RDF-specific derived quantities
        self.cutoff_radius = kwargs.get("cutoff_radius", self.cutoff_radius)
        self.rdf_nbins = kwargs.get("rdf_nbins", self.rdf_nbins)
        self.dr_rdf  = self.cutoff_radius / self.rdf_nbins
        self.compute(from_trajectory=True)  # Recompute slices and averages with new parameters

    # ------------------------------------------------------------------
    # Top-level compute entry point
    # ------------------------------------------------------------------

    @compute_doc
    def compute(self, from_trajectory: bool = False):
        """Compute the RDF and save to the HDF5 store.

        Parameters
        ----------
        from_trajectory : bool
            If ``True``, compute histograms directly from the raw particle
            trajectories in the h5md file (positions only).  Useful when the
            simulator did not accumulate ``rdf_hist`` on-the-fly.
            If ``False`` (default), read pre-accumulated histograms from the
            ``observables/rdf_hist`` group in the h5md file.
        """
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        if from_trajectory:
            self._calc_slices_from_trajectory()
        else:
            self._calc_slices_from_histogram()
        self._average_slices()
        self.save_state()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _calculate_bin_volumes(self) -> np.ndarray:
        """Volume of each spherical-shell bin.

        Uses the N-dimensional hyper-sphere shell formula::

            V_shell(r1, r2) = π^(D/2) / Γ(D/2 + 1) * (r2^D - r1^D)

        Returns
        -------
        bin_vol : numpy.ndarray, shape (no_bins,)
        """
        sphere_shell_const = (np.pi ** (self.dimensions / 2.0)) / factorial(self.dimensions / 2.0)
        bin_vol    = np.zeros(self.rdf_nbins)
        bin_vol[0] = sphere_shell_const * self.dr_rdf ** self.dimensions
        ir         = np.arange(1, self.rdf_nbins)
        r1         = ir * self.dr_rdf
        r2         = (ir + 1) * self.dr_rdf
        bin_vol[1:] = sphere_shell_const * (
            r2 ** self.dimensions - r1 ** self.dimensions
        )
        return bin_vol

    def _calculate_bin_centers(self) -> np.ndarray:
        """Radial bin centre positions.

        Returns
        -------
        r_values : numpy.ndarray, shape (no_bins,)
        """
        r_values    = np.zeros(self.rdf_nbins)
        r_values[0] = 0.5 * self.dr_rdf
        r_values[1:] = (np.arange(1, self.rdf_nbins) + 0.5) * self.dr_rdf
        return r_values

    def _calculate_pair_density_matrix(self) -> np.ndarray:
        """Number of pairs per unit volume for each species combination.

        Used to normalise raw histogram counts into g(r).

        Returns
        -------
        pair_density : numpy.ndarray, shape (num_species, num_species)
        """
        num_species  = len(self.species_names)
        pair_density = np.zeros((num_species, num_species))
        for i, sp1 in enumerate(self.species_num):
            pair_density[i, i] = 0.5 * sp1 * (sp1 - 1) / self.box_volume
            if num_species > 1:
                for j, sp2 in enumerate(self.species_num[i + 1 :], i + 1):
                    pair_density[i, j] = 0.5 * sp1 * sp2 / self.box_volume
                    pair_density[j, i] = pair_density[i, j]
        return pair_density

    def _normalize_single_slice(
        self, hist_array: np.ndarray, timesteps: int
    ) -> np.ndarray:
        """Normalise a raw histogram count array into g(r) for one slice.

        Parameters
        ----------
        hist_array : numpy.ndarray, shape (num_pairs, no_bins)
            Raw histogram counts for the slice.
        timesteps : int
            Number of timesteps accumulated in this slice.

        Returns
        -------
        rdf_slice : numpy.ndarray, shape (num_pairs, no_bins)
        """
        pair_density = self._calculate_pair_density_matrix()
        bin_vol      = self._calculate_bin_volumes()
        rdf_slice    = np.zeros(hist_array.shape, dtype=float)

        for k, pair_name in enumerate(self.species_pairs):
            sp1, sp2 = pair_name.split("-")
            i = list(self.species_names).index(sp1)
            j = list(self.species_names).index(sp2)
            symmetry = 2.0 if i != j else 1.0
            norm     = pair_density[i, j] * timesteps * bin_vol * symmetry
            norm     = np.where(norm > 0, norm, 1.0)
            rdf_slice[k] = hist_array[k] / norm

        return rdf_slice

    # ------------------------------------------------------------------
    # Shared store-preparation helper
    # ------------------------------------------------------------------

    def _prepare_store(self, r_values: np.ndarray) -> dict:
        """Pre-allocate the HDF5 store and return the coords dict.

        Parameters
        ----------
        r_values : numpy.ndarray
            Bin-centre distances in metres.

        Returns
        -------
        coords : dict
            The coordinate dict passed to :meth:`_preallocate_store`, kept
            so callers can reference the slice coordinate without rebuilding.
        """
        coords = {
            "species_pair": self.species_pairs,
            "r":            r_values,
            "slice":        np.arange(self.no_slices, dtype=int),
        }
        attrs = {
            "rdf_nbins":     self.rdf_nbins,
            "cutoff_radius": self.cutoff_radius,
            "no_slices":     self.no_slices,
            "block_length":  self.block_length,
            "dumps_shift":   self.dumps_shift,
            "h5md_path":     self.h5md_filepath,
        }
        self._preallocate_store(
            variable_shapes={"rdf": (self.num_pairs, self.rdf_nbins, self.no_slices)},
            coords=coords,
            attrs=attrs,
        )
        return coords

    # ------------------------------------------------------------------
    # Per-slice calculation — from raw trajectories
    # ------------------------------------------------------------------

    @calc_slices_doc
    def _calc_slices_from_trajectory(self):
        """Calculate the RDF slice-by-slice from raw particle positions.

        Reads positions one timestep at a time from the h5md file so that
        peak RAM equals one histogram array regardless of simulation size.
        """
        from ...core import Parameters

        r_values = self._calculate_bin_centers()
        self.bin_vol   = self._calculate_bin_volumes()
        self.ra_values = r_values / self.a_ws
        self._prepare_store(r_values)

        # ---- set up the distance-calculation algorithm ------------------
        
        from ...algorithms.cell_list import LinkedCellList

        method_class = LinkedCellList()
        params = Parameters()
        params.from_dict(
            {
                "box_lengths":        self.box_lengths,
                "cutoff_radius":      self.cutoff_radius,
                "dimensions":         self.dimensions,
                "total_num_density":  self.total_num_density,
                "a_ws":               self.a_ws,
                "units_dict":         self.units_dict,
            }
        )
        method_class.setup(params)
        cells_per_dim, cell_length_per_dim = method_class.create_cells_array(
            self.box_lengths, self.cutoff_radius
        )
    
        # ---- slice loop — O(slice) peak RAM per iteration ---------------
        with h5py.File(self.h5md_filepath, "r") as h5:
            pos_ds = h5["particles"]["pos"]  # lazy — nothing loaded yet

            for isl in tqdm(
                range(self.no_slices),
                desc="Calculating RDF from trajectory",
                disable=not self.verbose,
            ):
                start = isl * self.dumps_shift
                end   = start + self.block_length
                hist  = np.zeros((self.num_pairs, self.rdf_nbins))

                for it in tqdm(range(start, end), desc=f"Processing slice {isl}",
                               disable=not self.verbose, leave=False):
                    positions = pos_ds[it, :, :]
                    
                    head, ls_array = method_class.create_head_list_arrays(
                        positions, cell_length_per_dim, cells_per_dim
                    )
                    hist = method_class.calculate_rdf_hist(
                        pos=positions,
                        p_ids=self._particles_ids,
                        pair_index_map=self.pair_index_map,
                        cutoff=self.cutoff_radius,
                        rdf_bins=self.rdf_nbins,
                        hist_array=hist,
                        head=head,
                        ls_array=ls_array,
                        cells_per_dim=cells_per_dim,
                        box_lengths=self.box_lengths,
                    )
                    
                rdf_slice = self._normalize_single_slice(hist, end - start)
                self._write_slice({"rdf": rdf_slice}, isl)
    
    # ------------------------------------------------------------------
    # Per-slice calculation — from pre-accumulated histograms
    # ------------------------------------------------------------------

    @calc_slices_doc
    def _calc_slices_from_histogram(self):
        """Calculate the RDF from pre-accumulated ``rdf_hist`` in the h5md file.

        Reads only the first and last dump of each slice, so memory usage is
        independent of slice length.
        """
        r_values = self._calculate_bin_centers()
        self.bin_vol   = self._calculate_bin_volumes()
        self.ra_values = r_values / self.a_ws
        self._prepare_store(r_values)

        step      = self.dumps_shift
        dump_init = 0
        dump_end  = self.block_length - 1

        with h5py.File(self.h5md_filepath, "r") as h5md_file:
            rdf_hist_ds = h5md_file["observables"]["rdf_hist"]["value"]

            for isl in tqdm(
                range(self.no_slices),
                desc="Calculating RDF from histogram",
                disable=not self.verbose,
            ):
                data_init = rdf_hist_ds[dump_init, :, :, :]
                data_end  = rdf_hist_ds[dump_end,  :, :, :]

                hist_data = np.zeros((self.num_pairs, self.rdf_nbins), dtype=float)
                for i, sp1 in enumerate(self.species_names):
                    for j, sp2 in enumerate(self.species_names[i:], i):
                        # The off-diagonal counts already account for symmetry,
                        # so divide by 2 only for same-species pairs.
                        symmetry    = 2.0 if i == j else 1.0
                        rdf_hist_init = (data_init[i, j, :] + data_init[j, i, :]) / symmetry
                        rdf_hist_end  = (data_end[i, j, :] + data_end[j, i, :]) / symmetry
                        rdf_hist_slc = rdf_hist_end - rdf_hist_init
                        pair_index   = self.pair_index_map[i, j]
                        hist_data[pair_index, :] = rdf_hist_slc

                rdf_slice = self._normalize_single_slice(
                    hist_data, self.timesteps_per_slice
                )
                self._write_slice({"rdf": rdf_slice}, isl)

                dump_init += step
                dump_end  += step

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------

    def _average_slices(self):
        """Compute mean and std over the slice dimension and append to the store."""
        ddof = min(1, self.no_slices - 1)
        self._write_mean_std(["rdf"], ddof=ddof)

    # ------------------------------------------------------------------
    # Deprecated public API
    # ------------------------------------------------------------------

    def update_args(self, **kwargs):
        """.. deprecated::
            Use :meth:`recalculate` to change parameters after setup,
            or pass keyword arguments directly to :meth:`setup`.
        """
        warnings.warn(
            "update_args() is deprecated. Pass kwargs to setup() directly, "
            "or use recalculate() to change parameters after setup.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._init_rdf_args(**kwargs)

    def calc_slices_data(self):
        """.. deprecated::
            Use :meth:`compute` with ``from_trajectory=False`` instead.
        """
        warnings.warn(
            "calc_slices_data() is deprecated. "
            "Call compute(from_trajectory=False) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._calc_slices_from_histogram()

    def average_slices_data(self):
        """.. deprecated::
            Averaging is now performed automatically by :meth:`compute`.
        """
        warnings.warn(
            "average_slices_data() is deprecated. "
            "Averaging is now handled automatically inside compute().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._average_slices()
        
# ---------------------------------------------------------------------------
# PairDistributionFunction
# ---------------------------------------------------------------------------

# Coordinate system metadata — axis names and default cutoffs per system.
# Keeps the coordinate-system logic in one place rather than scattered
# across __init__, _init_pdf_args, and _calculate_bin_volumes.
_COORD_SYSTEMS = {
    "cartesian": {
        "axes":         ("x",     "y",      "z"),
        "default_bins": (100,     100,      100),
        "default_cuts": (5.0,     5.0,      5.0),
    },
    "cylindrical": {
        "axes":         ("rho",   "theta",  "z"),
        "default_bins": (100,     45,       100),
        "default_cuts": (5.0,     np.pi,    5.0),
    },
    "spherical": {
        "axes":         ("r",     "theta",  "phi"),
        "default_bins": (100,     45,       90),
        "default_cuts": (5.0,     np.pi/2,  2*np.pi),
    },
}


class PairDistributionFunction(Observable):
    """3D Pair Distribution Function g(u, v, w).

    Computes the pair distribution function on a 3-dimensional grid in
    Cartesian, cylindrical, or spherical coordinates using the linked-cell
    list algorithm.  The computation follows the same slice-by-slice,
    O(1-slice) peak-RAM pattern as :class:`RadialDistributionFunction`.

    Parameters
    ----------
    coord_system : str
        One of ``'cartesian'``, ``'cylindrical'``, ``'spherical'``.
        Default: ``'cartesian'``.
    pdf_bins : array-like of int, optional
        Number of bins in each of the three spatial dimensions.
        Defaults per coordinate system:

        * cartesian   — ``(100, 100, 100)``
        * cylindrical — ``(100,  45, 100)``
        * spherical   — ``(100,  45,  90)``

    cutoffs : array-like of float, optional
        Upper cutoff in each dimension (simulation units).
        Defaults per coordinate system:

        * cartesian   — ``(5.0, 5.0, 5.0)``
        * cylindrical — ``(5.0, π,   5.0)``
        * spherical   — ``(5.0, π/2, 2π)``

    cutoff_radius : float, optional
        Linked-cell list interaction cutoff radius.  Defaults to the first
        element of *cutoffs* (the radial cutoff) if not supplied.

    HDF5 Store Layout
    -----------------
    ``/coords/species_pair``  — pair labels UTF-8,     shape (num_pairs,)
    ``/coords/<axis0>``       — bin-centre values,     shape (pdf_bins[0],)
    ``/coords/<axis1>``       — bin-centre values,     shape (pdf_bins[1],)
    ``/coords/<axis2>``       — bin-centre values,     shape (pdf_bins[2],)
    ``/coords/slice``         — slice indices,         shape (no_slices,)
    ``/data/pdf``             — shape (num_pairs, bins0, bins1, bins2, no_slices)
    ``/data/mean_pdf``        — shape (num_pairs, bins0, bins1, bins2)
    ``/data/std_pdf``         — shape (num_pairs, bins0, bins1, bins2)
    file attrs                — cache-invalidation metadata
    """

    def __init__(
        self,
        coord_system: str = "cartesian",
        pdf_bins: np.ndarray = None,
        cutoffs: np.ndarray = None,
        cutoff_radius: float = None,
    ):
        super().__init__()
        self.__name__      = "pdf"
        self.__long_name__ = "Pair Distribution Function"

        if coord_system not in _COORD_SYSTEMS:
            raise ValueError(
                f"coord_system must be one of {list(_COORD_SYSTEMS)}, "
                f"got '{coord_system}'."
            )
        self.coord_system = coord_system

        meta = _COORD_SYSTEMS[coord_system]
        self.pdf_bins = (
            np.array(meta["default_bins"], dtype=int)
            if pdf_bins is None
            else np.asarray(pdf_bins, dtype=int)
        )
        self.cutoffs = (
            np.array(meta["default_cuts"], dtype=float)
            if cutoffs is None
            else np.asarray(cutoffs, dtype=float)
        )
        # Radial cutoff for the linked-cell list: default to the first cutoff
        # (the radial dimension is always axis 0 in all three systems).
        self.cutoff_radius = (
            float(self.cutoffs[0]) if cutoff_radius is None else float(cutoff_radius)
        )

        # Derived geometry — recomputed in _init_pdf_args after setup_init
        self.deltas_pdf = self.cutoffs / self.pdf_bins

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @setup_doc
    def setup(
        self,
        params,
        phase: str = None,
        independent_slices: bool = None,
        no_slices: int = None,
        timesteps_per_slice: int = None,
        timesteps_shift: int = None,
        plasma_periods_per_slice: int = None,
        plasma_periods_shift: int = None,
        **kwargs,
    ):
        super().setup_init(
            params,
            phase=phase,
            independent_slices=independent_slices,
            no_slices=no_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift,
            **kwargs,
        )
        self._init_pdf_args(**kwargs)

    def _init_pdf_args(self, **kwargs):
        """Initialise PDF-specific attributes and derived quantities.

        Called once from :meth:`setup` and again from :meth:`_on_recalculate`
        whenever PDF-specific parameters change.

        Parameters
        ----------
        **kwargs
            Any PDF-specific keyword arguments passed through from
            :meth:`setup` (e.g. ``cutoff_radius``, ``pdf_bins``, ``cutoffs``).
        """
        self.__dict__.update(kwargs)

        # Recompute bin widths whenever cutoffs or pdf_bins change
        self.deltas_pdf = self.cutoffs / self.pdf_bins

        self.update_finish()

        # Build species-ID array (integer per particle, used by LinkedCellList)
        self._particles_ids = np.zeros(self.total_num_ptcls, dtype="int64")
        s0 = 0
        for i, n in enumerate(self.species_num):
            self._particles_ids[s0 : s0 + n] = i
            s0 += n

    def pretty_print_msg(self):
        """Return a formatted summary string for the log file."""
        
        msg = super().pretty_print_msg()
        msg += f"\n  Coordinate system: {self.coord_system}"
        msg += f"\n  PDF bins: {self.pdf_bins}"
        msg += f"\n  Cutoffs:"
        if self.coord_system == "cartesian":
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][0]}: {self.cutoffs[0]/self.a_ws:.4e} a_ws, dx = {self.deltas_pdf[0]/self.a_ws:.4e} a_ws"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][1]}: {self.cutoffs[1]/self.a_ws:.4e} a_ws, dy = {self.deltas_pdf[1]/self.a_ws:.4e} a_ws"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][2]}: {self.cutoffs[2]/self.a_ws:.4e} a_ws, dz = {self.deltas_pdf[2]/self.a_ws:.4e} a_ws"
        elif self.coord_system == "cylindrical":
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][0]}: {self.cutoffs[0]/self.a_ws:.4e} a_ws, d{_COORD_SYSTEMS[self.coord_system]['axes'][0]} = {self.deltas_pdf[0]/self.a_ws:.4e} a_ws"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][1]}: {self.cutoffs[1]/(np.pi):.4e} π, d{_COORD_SYSTEMS[self.coord_system]['axes'][1]} = {self.deltas_pdf[1]/(np.pi):.4e} π"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][2]}: {self.cutoffs[2]/self.a_ws:.4e} a_ws, d{_COORD_SYSTEMS[self.coord_system]['axes'][2]} = {self.deltas_pdf[2]/self.a_ws:.4e} a_ws"
        else:  # spherical
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][0]}: {self.cutoffs[0]/self.a_ws:.4e} a_ws, d{_COORD_SYSTEMS[self.coord_system]['axes'][0]} = {self.deltas_pdf[0]/self.a_ws:.4e} a_ws"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][1]}: {self.cutoffs[1]/(np.pi):.4e} π, d{_COORD_SYSTEMS[self.coord_system]['axes'][1]} = {self.deltas_pdf[1]/(np.pi):.4e} π"
            msg += f"\n\t  {_COORD_SYSTEMS[self.coord_system]['axes'][2]}: {self.cutoffs[2]/(np.pi):.4e} π, d{_COORD_SYSTEMS[self.coord_system]['axes'][2]} = {self.deltas_pdf[2]/(np.pi):.4e} π"
        return msg
    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    def _store_is_valid(self) -> bool:
        """Extend the base check with PDF-specific cache-invalidation parameters.

        Checks ``cutoff_radius``, ``pdf_bins``, ``cutoffs``, and
        ``coord_system`` in addition to the base-class slicing parameters.
        """
        if not super()._store_is_valid():
            return False
        try:
            with h5py.File(self.hdf_store_path, "r") as f:
                return (
                    f.attrs.get("coord_system")  == self.coord_system
                    and np.array_equal(f.attrs.get("pdf_bins"), self.pdf_bins)
                    and np.array_equal(f.attrs.get("cutoffs"),  self.cutoffs)
                )
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Recalculation hook
    # ------------------------------------------------------------------

    def _on_recalculate(self, **kwargs):
        """Recompute PDF-derived quantities after :meth:`recalculate` updates attrs.

        Handles PDF-specific parameters (``cutoff_radius``, ``pdf_bins``,
        ``cutoffs``, ``coord_system``) and delegates slicing parameters to
        the base class.
        """
        super()._on_recalculate(**kwargs)

        if "coord_system" in kwargs and kwargs["coord_system"] not in _COORD_SYSTEMS:
            raise ValueError(
                f"coord_system must be one of {list(_COORD_SYSTEMS)}, "
                f"got '{kwargs['coord_system']}'."
            )
        # Recompute derived geometry
        self.deltas_pdf = self.cutoffs / self.pdf_bins

    # ------------------------------------------------------------------
    # Compute entry point
    # ------------------------------------------------------------------

    @compute_doc
    def compute(self):
        """Compute the PDF slice-by-slice and save to the HDF5 store.

        Always computes from raw particle trajectories using the linked-cell
        list algorithm.  The PDF does not have a pre-accumulated histogram
        equivalent (unlike the RDF), so no ``from_trajectory`` flag is needed.
        """
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self._calc_slices_from_trajectory()
        self._average_slices()
        self.save_state()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _calculate_bin_centers(self) -> tuple:
        """Bin-centre coordinates for each spatial axis.

        Returns
        -------
        centers : tuple of three numpy.ndarray
            Each array has shape ``(pdf_bins[i],)`` and contains the
            physical coordinate at the centre of each bin along axis *i*.
        """
        centers = tuple(
            (np.arange(int(self.pdf_bins[i])) + 0.5) * self.deltas_pdf[i]
            for i in range(3)
        )
        return centers

    def _calculate_bin_volumes(self) -> np.ndarray:
        """Volume element for each bin in the current coordinate system.

        Returns
        -------
        bin_vols : numpy.ndarray, shape (pdf_bins[0], pdf_bins[1], pdf_bins[2])
        """
        b0, b1, b2 = int(self.pdf_bins[0]), int(self.pdf_bins[1]), int(self.pdf_bins[2])
        d0, d1, d2 = self.deltas_pdf
        bin_vols   = np.zeros((b0, b1, b2))

        if self.coord_system == "cartesian":
            # Uniform Cartesian volume element
            bin_vols[:] = d0 * d1 * d2

        elif self.coord_system == "cylindrical":
            # V = π(r_out² - r_in²) · dz · (dφ / 2π)
            # The φ factor accounts for the fraction of the annulus covered.
            for i in range(b0):
                r_in  = i * d0
                r_out = (i + 1) * d0
                bin_vols[i] = (
                    np.pi * (r_out ** 2 - r_in ** 2) * d2 * (d1 / (2.0 * np.pi))
                )

        else:  # spherical
            # V_shell = (4/3)π(r_out³ - r_in³)
            # Polar fraction: sin(θ_c) dθ / 2
            # Azimuthal fraction: dφ / (2π)
            for i in range(b0):
                r_in    = i * d0
                r_out   = (i + 1) * d0
                v_shell = (4.0 / 3.0) * np.pi * (r_out ** 3 - r_in ** 3)
                for j in range(b1):
                    theta_c        = (j + 0.5) * d1
                    bin_vols[i, j] = (
                        v_shell
                        * np.sin(theta_c) * d1 / 2.0
                        * d2 / (2.0 * np.pi)
                    )

        return bin_vols

    def _calculate_pair_density_matrix(self) -> np.ndarray:
        """Number of pairs per unit volume for each species combination.

        Returns
        -------
        pair_density : numpy.ndarray, shape (num_species, num_species)
        """
        num_species  = len(self.species_names)
        pair_density = np.zeros((num_species, num_species))
        for i, n_i in enumerate(self.species_num):
            pair_density[i, i] = 0.5 * n_i * (n_i - 1) / self.box_volume
            if num_species > 1:
                for j, n_j in enumerate(self.species_num[i + 1:], i + 1):
                    pair_density[i, j] = 0.5 * n_i * n_j / self.box_volume
                    pair_density[j, i] = pair_density[i, j]
        return pair_density

    def _normalize_single_slice(
        self, hist_array: np.ndarray, timesteps: int
    ) -> np.ndarray:
        """Normalise a raw 3D histogram count array into g(u,v,w) for one slice.

        Parameters
        ----------
        hist_array : numpy.ndarray, shape (num_pairs, bins0, bins1, bins2)
            Raw histogram counts for the slice.
        timesteps : int
            Number of timesteps accumulated in this slice.

        Returns
        -------
        pdf_slice : numpy.ndarray, shape (num_pairs, bins0, bins1, bins2)
        """
        pair_density = self._calculate_pair_density_matrix()
        bin_vols     = self._calculate_bin_volumes()  # (bins0, bins1, bins2)
        pdf_slice    = np.zeros(hist_array.shape, dtype=float)

        for k, pair_name in enumerate(self.species_pairs):
            sp1, sp2 = pair_name.split("-")
            i = list(self.species_names).index(sp1)
            j = list(self.species_names).index(sp2)
            symmetry = 2.0 if i != j else 1.0
            # norm broadcasts over (bins0, bins1, bins2)
            norm = pair_density[i, j] * timesteps * bin_vols * symmetry
            norm = np.where(norm > 0, norm, 1.0)
            pdf_slice[k] = hist_array[k] / norm

        return pdf_slice

    # ------------------------------------------------------------------
    # Store preparation
    # ------------------------------------------------------------------

    def _prepare_store(self) -> dict:
        """Pre-allocate the HDF5 store and return the coords dict.

        Returns
        -------
        coords : dict
            The coordinate dict passed to :meth:`_preallocate_store`.
        """
        axis_names   = _COORD_SYSTEMS[self.coord_system]["axes"]
        bin_centers  = self._calculate_bin_centers()

        coords = {
            "species_pair": self.species_pairs,
            axis_names[0]:  bin_centers[0],
            axis_names[1]:  bin_centers[1],
            axis_names[2]:  bin_centers[2],
            "slice":        np.arange(self.no_slices),
        }
        attrs = {
            "coord_system":  self.coord_system,
            "cutoffs":       self.cutoffs,
            "pdf_bins":      self.pdf_bins,
            "no_slices":     self.no_slices,
            "block_length":  self.block_length,
            "dumps_shift":   self.dumps_shift,
            "h5md_path":     self.h5md_filepath,
        }
        self._preallocate_store(
            variable_shapes={
                "pdf": (
                    self.num_pairs,
                    int(self.pdf_bins[0]),
                    int(self.pdf_bins[1]),
                    int(self.pdf_bins[2]),
                    self.no_slices,
                )
            },
            coords=coords,
            attrs=attrs,
        )
        return coords

    # ------------------------------------------------------------------
    # Per-slice calculation
    # ------------------------------------------------------------------

    @calc_slices_doc
    def _calc_slices_from_trajectory(self):
        """Calculate the PDF slice-by-slice from raw particle positions.

        Reads positions one timestep at a time from the h5md file so that
        peak RAM equals one histogram array regardless of simulation size.
        Only the linked-cell list algorithm is supported for the 3D PDF
        because the minimum-image algorithm does not implement 3D binning.
        """
        from ...algorithms.cell_list import LinkedCellList
        from ...core import Parameters

        self._prepare_store()

        # ---- set up the linked-cell list --------------------------------
        lcl    = LinkedCellList()
        params = Parameters()
        params.from_dict({
            "box_lengths":       self.box_lengths,
            "cutoff_radius":     self.cutoff_radius,
            "dimensions":        self.dimensions,
            "total_num_density": self.total_num_density,
            "a_ws":              self.a_ws,
            "units_dict":        self.units_dict,
        })
        lcl.setup(params)
        cells_per_dim, cell_length_per_dim = lcl.create_cells_array(
            self.box_lengths, self.cutoff_radius
        )

        # ---- slice loop — O(slice) peak RAM per iteration ---------------
        with h5py.File(self.h5md_filepath, "r") as h5:
            pos_ds = h5["particles"]["pos"]   # lazy — nothing loaded yet

            for isl in tqdm(
                range(self.no_slices),
                desc="Calculating PDF from trajectory",
                disable=not self.verbose,
            ):
                start = isl * self.dumps_shift
                end   = start + self.block_length
                hist  = np.zeros(
                    (
                        self.num_pairs,
                        int(self.pdf_bins[0]),
                        int(self.pdf_bins[1]),
                        int(self.pdf_bins[2]),
                    ),
                    dtype="int64",
                )

                for it in tqdm(range(start, end), desc=f"Timestep", disable=not self.verbose, leave=False, position=1):
                    positions        = pos_ds[it, :, :]
                    head, ls_array   = lcl.create_head_list_arrays(
                        positions, cell_length_per_dim, cells_per_dim
                    )
                    hist = lcl.calculate_pdf_hist(
                        pos=positions,
                        p_ids=self._particles_ids,
                        pair_index_map=self.pair_index_map,
                        cutoffs=self.cutoffs,
                        pdf_bins=self.pdf_bins,
                        hist_array=hist,
                        head=head,
                        ls_array=ls_array,
                        cells_per_dim=cells_per_dim,
                        box_lengths=self.box_lengths,
                        coord_system=self.coord_system,
                    )

                pdf_slice = self._normalize_single_slice(hist, end - start)
                self._write_slice({"pdf": pdf_slice}, isl)

    # ------------------------------------------------------------------
    # Averaging
    # ------------------------------------------------------------------

    def _average_slices(self):
        """Compute mean and std over the slice dimension and append to the store."""
        ddof = min(1, self.no_slices - 1)
        self._write_mean_std(["pdf"], ddof=ddof)

    # ------------------------------------------------------------------
    # Convenience reducers
    # ------------------------------------------------------------------

    def reduce_to_2d(self, species_pair: str, axis: int = 2):
        """Return ``mean_pdf`` for one pair averaged over one spatial axis.

        Parameters
        ----------
        species_pair : str
            Pair label, e.g. ``'H-He'``.
        axis : int
            Spatial axis to average over (0, 1, or 2).  Default: 2.

        Returns
        -------
        xr.DataArray
            Shape ``(bins_a, bins_b)`` — the two remaining spatial axes.
        """
        ds        = self.read_dataset()
        mean_pdf  = ds["mean_pdf"].sel(species_pair=species_pair)
        axis_name = mean_pdf.dims[axis]
        return mean_pdf.mean(dim=axis_name)

    def reduce_to_1d(self, species_pair: str, axes: tuple = (1, 2)):
        """Return ``mean_pdf`` for one pair averaged over two spatial axes.

        Parameters
        ----------
        species_pair : str
            Pair label, e.g. ``'H-He'``.
        axes : tuple of int
            Two spatial axes to average over.  Default: ``(1, 2)``.

        Returns
        -------
        xr.DataArray
            Shape ``(bins_remaining,)``.
        """
        ds       = self.read_dataset()
        mean_pdf = ds["mean_pdf"].sel(species_pair=species_pair)
        dims     = [mean_pdf.dims[a] for a in axes]
        return mean_pdf.mean(dim=dims)

    def get_slice(self, species_pair: str, slice_idx: int):
        """Return the raw 3D PDF for one pair and one slice.

        Reads directly from the HDF5 store without loading the full array.

        Parameters
        ----------
        species_pair : str
            Pair label, e.g. ``'H-He'``.
        slice_idx : int
            Slice index (0-based).

        Returns
        -------
        numpy.ndarray, shape (bins0, bins1, bins2)
        """
        pair_idx = self.species_pairs.index(species_pair)
        with h5py.File(self.hdf_store_path, "r") as f:
            return f["data/pdf"][pair_idx, :, :, :, slice_idx]

    # ------------------------------------------------------------------
    # Deprecated public API
    # ------------------------------------------------------------------

    def update_args(self, **kwargs):
        """.. deprecated::
            Use :meth:`recalculate` or pass kwargs directly to :meth:`setup`.
        """
        warnings.warn(
            "update_args() is deprecated. Pass kwargs to setup() directly, "
            "or use recalculate() to change parameters after setup.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._init_pdf_args(**kwargs)

    def calc_slices_data(self):
        """.. deprecated::
            Use :meth:`compute` instead.
        """
        warnings.warn(
            "calc_slices_data() is deprecated. Call compute() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._calc_slices_from_trajectory()

    def average_slices_data(self):
        """.. deprecated::
            Averaging is now performed automatically by :meth:`compute`.
        """
        warnings.warn(
            "average_slices_data() is deprecated. "
            "Averaging is handled automatically inside compute().",
            DeprecationWarning,
            stacklevel=2,
        )
        self._average_slices()