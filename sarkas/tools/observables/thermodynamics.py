"""
Thermodynamic and pressure-tensor observables for sarkas post-processing.

All I/O uses HDF5 (h5py) directly via the Observable base-class helpers
(_preallocate_store, _write_slice, _write_mean_std, read_dataset).

Thermodynamics HDF5 store layout
---------------------------------
/coords/
    species     — species names (+ "Total" for multi-species)
    quantity    — thermodynamic quantity labels
    time        — time values for one block
    slice       — slice indices
/data/
    therm                       — shape (no_species, no_quantities, block_length, no_slices)
    mean_therm / std_therm      — shape (no_species, no_quantities, block_length)
    acf_therm                   — shape (no_species, no_quantities, block_length, no_slices)
    mean_acf_therm / std_acf_therm — shape (no_species, no_quantities, block_length)
file attrs — no_slices, block_length, dumps_shift, h5md_path

PressureTensor HDF5 store layout
----------------------------------
/coords/
    species         — species names (+ "Total")
    component       — upper-triangular tensor labels (e.g. "XX","XY",...)
    time            — time values for one block
    slice           — slice indices
/data/
    pressure                        — shape (no_species, block_length, no_slices)
    tensor                          — shape (no_species, no_components, block_length, no_slices)
    mean_pressure / std_pressure    — shape (no_species, block_length)
    mean_tensor / std_tensor        — shape (no_species, no_components, block_length)
    acf_pressure                    — shape (no_species, block_length, no_slices)
    acf_tensor                      — shape (no_species, no_comp, no_comp, block_length, no_slices)
    mean_acf_pressure / std_acf_pressure — shape (no_species, block_length)
    mean_acf_tensor / std_acf_tensor     — shape (no_species, no_comp, no_comp, block_length)
file attrs — no_slices, block_length, dumps_shift, h5md_path
"""

import warnings

import h5py
import numpy as np
import xarray as xr
from tqdm import tqdm

from .base import (
    Observable,
    arg_update_doc,
    avg_acf_slices_doc,
    avg_slices_doc,
    calc_acf_slices_doc,
    calc_slices_doc,
    compute_acf_doc,
    compute_doc,
    setup_doc,
)
from ...utilities.maths import correlationfunction
from ...utilities.misc import calculate_beta
from ...utilities.timing import time_stamp


# ---------------------------------------------------------------------------
# Thermodynamics
# ---------------------------------------------------------------------------


class Thermodynamics(Observable):
    """Thermodynamic functions.

    Computes per-species and total thermodynamic quantities (temperature,
    kinetic / potential / total energy, …) from the simulation H5MD file
    and stores results slice-by-slice in the observable's HDF5 store.

    Attributes
    ----------
    beta_slices : numpy.ndarray
        Inverse temperature for each slice.  Populated by
        :meth:`calculate_beta_slices`.
    beta : float
        Inverse temperature from the full simulation run.  Populated by
        :meth:`calculate_beta_simulation`.
    specific_heat_volume_slice : numpy.ndarray
        Specific heat at constant volume for each slice.
    specific_heat_volume : float
        Specific heat from the full simulation run.
    """

    def __init__(self):
        super().__init__()
        self.__name__      = "therm"
        self.__long_name__ = "Thermodynamics"
        self.acf_observable           = True
        self.beta_slices              = None
        self.beta                     = None
        self.specific_heat_volume_slice = None
        self.specific_heat_volume     = None

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

        self.restart_sim = params.load_method[:-7] == "restart"
        self.__dict__.update(kwargs)
        self.update_finish()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _species_list(self) -> list:
        """Species labels used as the first dimension of all stored arrays.

        For a single-species run this is just ``[species_name]``.
        For multi-species runs ``'Total'`` is appended.
        """
        names = list(self.species_names)
        return [*names, "Total"] if len(names) > 1 else names

    @property
    def _quantity_labels(self) -> list:
        """Display-friendly quantity labels (title-case, space-separated)."""
        return [self._capitalize_words(q) for q in self.thermodynamics_list]

    @staticmethod
    def _capitalize_words(s: str) -> str:
        """``'kinetic_energy'`` → ``'Kinetic Energy'``."""
        return " ".join(w.capitalize() for w in s.split("_"))

    def _standard_attrs(self) -> dict:
        """Cache-invalidation attributes written to every store."""
        return {
            "no_slices":    self.no_slices,
            "block_length": self.block_length,
            "dumps_shift":  self.dumps_shift,
            "h5md_path":    self.h5md_filepath,
        }

    # ------------------------------------------------------------------
    # Public compute entry points
    # ------------------------------------------------------------------

    @compute_doc
    def compute(self, calculate_acf: bool = False):
        """Compute thermodynamic quantities for each slice.

        Parameters
        ----------
        calculate_acf : bool
            If ``True``, also compute the autocorrelation functions after
            the main calculation.  Default ``False``.
        """
        if self._store_is_valid():
            self.calculate_beta_slices()
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self._calc_slices()
        self._average_slices()
        self.calculate_beta_slices()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self):
        """Compute autocorrelation functions for all thermodynamic quantities."""
        t0 = self.timer.current()
        self._calc_acf_slices()
        self._average_acf_slices()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " ACF Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # Core slice computation
    # ------------------------------------------------------------------

    @calc_slices_doc
    def _calc_slices(self):
        """Read thermodynamic data from H5MD and write one slice at a time."""
        species_list  = self._species_list
        qty_labels    = self._quantity_labels
        no_sp         = len(species_list)
        no_q          = len(qty_labels)

        # ---- read all raw data from H5MD once ---------------------------
        raw      = {}
        time_arr = None
        with h5py.File(self.h5md_filepath, "r") as h5:
            for sp in self.species_names:
                raw[sp] = {}
                grp = h5["observables"][sp]
                for qty in self.thermodynamics_list:
                    try:
                        raw[sp][qty] = grp[qty]["value"][:]
                        if time_arr is None:
                            time_arr = grp[qty]["time"][:]
                    except KeyError:
                        raw[sp][qty] = np.zeros(self.no_dumps)

        # ---- derive "Total" for multi-species ---------------------------
        if len(self.species_names) > 1:
            raw["Total"] = {}
            sp_names = list(self.species_names)
            for qty in self.thermodynamics_list:
                total = np.zeros(len(time_arr))
                for sp in sp_names:
                    if qty.lower() == "temperature":
                        frac   = self.species_num[sp_names.index(sp)] / self.total_num_ptcls
                        total += raw[sp][qty] * frac
                    else:
                        total += raw[sp][qty]
                raw["Total"][qty] = total

        # ---- pre-allocate HDF5 store ------------------------------------
        coords = {
            "species":  species_list,
            "quantity": qty_labels,
            "time":     time_arr[:self.block_length],
            "slice":    np.arange(self.no_slices),
        }
        self._preallocate_store(
            variable_shapes={"therm": (no_sp, no_q, self.block_length, self.no_slices)},
            coords=coords,
            attrs=self._standard_attrs(),
        )

        # ---- fill one slice at a time — O(slice) peak RAM ---------------
        for isl in tqdm(
            range(self.no_slices),
            desc="Thermodynamics slices",
            disable=not self.verbose,
        ):
            start = isl * self.dumps_shift
            end   = start + self.block_length
            arr   = np.zeros((no_sp, no_q, self.block_length))
            for isp, sp in enumerate(species_list):
                for iq, qty in enumerate(self.thermodynamics_list):
                    arr[isp, iq, :] = raw[sp][qty][start:end]
            self._write_slice({"therm": arr}, isl)

    @avg_slices_doc
    def _average_slices(self):
        """Compute mean and std of thermodynamic quantities over slices."""
        self._write_mean_std(["therm"], ddof=min(1, self.no_slices - 1))

    # ------------------------------------------------------------------
    # ACF slice computation
    # ------------------------------------------------------------------

    @calc_acf_slices_doc
    def _calc_acf_slices(self):
        """Compute per-quantity ACF for each slice and append to the store."""
        ds       = self.read_dataset()
        therm    = ds["therm"].values   # (no_sp, no_q, block_length, no_slices)
        no_sp, no_q, bl, no_sl = therm.shape

        # ACF shares coords with therm but uses "lag" instead of "time"
        # We append to the existing store, which already has coords written.
        # Use h5py directly to add the new dataset without touching coords.
        ddof = min(1, no_sl - 1)

        acf_arr = np.np.zeros((no_sp, no_q, bl, no_sl))
        for isl in range(no_sl):
            for isp in range(no_sp):
                for iq in range(no_q):
                    d = therm[isp, iq, :, isl]
                    acf_arr[isp, iq, :, isl] = correlationfunction(d, d)

        # Append acf_therm to the existing HDF5 file.
        # The "lag" axis shares the same values as "time" so we reuse it.
        with h5py.File(self.hdf_store_path, "a") as f:
            dg  = f.require_group("data")
            cg  = f["coords"]

            # Add "lag" coordinate if not already present (same values as time)
            if "lag" not in cg:
                cg.create_dataset("lag", data=cg["time"][:])
                cg["lag"].attrs["_is_coord"] = True

            for name, arr, dims in [
                ("acf_therm",      acf_arr,            ["species", "quantity", "lag", "slice"]),
                ("mean_acf_therm", acf_arr.mean(-1),   ["species", "quantity", "lag"]),
                ("std_acf_therm",  acf_arr.std(-1, ddof=ddof), ["species", "quantity", "lag"]),
            ]:
                if name in dg:
                    del dg[name]
                ds_out = dg.create_dataset(
                    name, data=arr,
                    compression="gzip", compression_opts=5,
                )
                ds_out.attrs["_DIMS"] = dims

    @avg_acf_slices_doc
    def _average_acf_slices(self):
        """Mean and std of the thermodynamic ACFs are written in _calc_acf_slices."""
        # Already computed inside _calc_acf_slices to avoid a second full read.
        pass

    # ------------------------------------------------------------------
    # Thermodynamic derived quantities
    # ------------------------------------------------------------------

    def calculate_beta_slices(self, ensemble: str = "NVE"):
        """Calculate the inverse temperature for each slice.

        Reads temperature from the HDF5 store and populates
        :attr:`beta_slices` (shape ``(no_slices,)``).

        Parameters
        ----------
        ensemble : str
            ``'NVE'`` (default) or ``'NVT'``.
        """
        if ensemble == "NVE":
            self.beta_slices = np.zeros(self.no_slices)
            ds       = self.read_dataset()
            sp_label = "Total" if len(self.species_names) > 1 else self.species_names[0]
            qty_lbl  = self._capitalize_words("temperature")
            # shape: (block_length, no_slices)
            temp     = ds["therm"].sel(species=sp_label, quantity=qty_lbl).values
            for isl in range(self.no_slices):
                self.beta_slices[isl] = calculate_beta(
                    float(temp[:, isl].mean()), k_B=self.kB
                )
        else:
            self.beta_slices = np.ones(self.no_slices) * calculate_beta(
                self.T_desired, k_B=self.kB
            )

    def calculate_heat_capacity_slices(self, ensemble: str = "NVE"):
        """Calculate the specific heat at constant volume for each slice.

        Populates :attr:`specific_heat_volume_slice` (shape ``(no_slices,)``).
        Calls :meth:`calculate_beta_slices` internally.

        Parameters
        ----------
        ensemble : str
            ``'NVE'`` (default) or ``'NVT'``.
        """
        self.calculate_beta_slices(ensemble=ensemble)
        self.specific_heat_volume_slice = np.zeros(self.no_slices)
        ds = self.read_dataset()

        if ensemble == "NVE":
            sp_label = "Total" if len(self.species_names) > 1 else self.species_names[0]
            kin_qty  = self._capitalize_words("kinetic_energy")
            kin      = ds["therm"].sel(species=sp_label, quantity=kin_qty).values
            # shape: (block_length, no_slices)
            for isl in range(self.no_slices):
                kin_2  = kin[:, isl].std() ** 2
                denom  = (
                    1.0
                    - 2.0 * self.beta_slices[isl] ** 2 * kin_2
                    / (self.dimensions * self.total_num_ptcls)
                )
                self.specific_heat_volume_slice[isl] = (
                    0.5 * self.dimensions * self.kB * self.total_num_ptcls / denom
                )
        else:
            sp_labels = list(self.species_names)
            te_qty    = self._capitalize_words("total_energy")
            te        = ds["therm"].sel(species=sp_labels, quantity=te_qty).values
            # shape: (no_sp, block_length, no_slices)
            for isl in range(self.no_slices):
                total_e = te[:, :, isl].sum(axis=0)   # (block_length,)
                delta_e_2 = total_e.std() ** 2
                self.specific_heat_volume_slice[isl] = (
                    delta_e_2 * self.beta_slices[isl] ** 2 * self.kB
                )

    def calculate_beta_simulation(self, ensemble: str = "NVE"):
        """Calculate the inverse temperature from the full simulation run.

        Reads directly from the H5MD file and populates :attr:`beta`.

        Parameters
        ----------
        ensemble : str
            ``'NVE'`` (default) or ``'NVT'``.
        """
        if ensemble == "NVE":
            temps = []
            with h5py.File(self.h5md_filepath, "r") as h5:
                for sp in self.species_names:
                    temps.append(
                        float(h5[f"observables/{sp}/temperature/value"][:].mean())
                    )
            self.beta = calculate_beta(float(np.mean(temps)), k_B=self.kB)
        else:
            self.beta = calculate_beta(self.T_desired, k_B=self.kB)

    def calculate_heat_capacity_simulation(self, ensemble: str = "NVE"):
        """Calculate the specific heat at constant volume from the full run.

        Calls :meth:`calculate_beta_simulation` first, then populates
        :attr:`specific_heat_volume`.

        Parameters
        ----------
        ensemble : str
            ``'NVE'`` (default) or ``'NVT'``.

        Notes
        -----
        NVE formula:

        .. math::
            C_V = \\frac{dN k_B}{2}
            \\left(1 - \\frac{2\\beta^2}{dN}\\langle K^2 \\rangle\\right)^{-1}

        NVT formula:

        .. math::
            C_V = k_B \\beta^2 \\langle \\Delta E^2 \\rangle
        """
        self.calculate_beta_simulation(ensemble=ensemble)

        if ensemble == "NVE":
            with h5py.File(self.h5md_filepath, "r") as h5:
                kin_total = sum(
                    h5[f"observables/{sp}/kinetic_energy/value"][:]
                    for sp in self.species_names
                )
            kin_2  = kin_total.std() ** 2
            denom  = (
                1.0
                - 2.0 * self.beta ** 2 * kin_2
                / (self.dimensions * self.total_num_ptcls)
            )
            self.specific_heat_volume = (
                0.5 * self.dimensions * self.kB * self.total_num_ptcls / denom
            )
        else:
            with h5py.File(self.h5md_filepath, "r") as h5:
                total_e = sum(
                    h5[f"observables/{sp}/total_energy/value"][:]
                    for sp in self.species_names
                )
            self.specific_heat_volume = total_e.std() ** 2 * self.beta ** 2 * self.kB

    # ------------------------------------------------------------------
    # RDF-based thermodynamics
    # ------------------------------------------------------------------

    def compute_from_rdf(self, rdf, potential):
        """Calculate correlational energy and pressure from the RDF.

        Parameters
        ----------
        rdf : RadialDistributionFunction
        potential : sarkas.potentials.core.Potential

        Returns
        -------
        nkT : float
        u_hartree : numpy.ndarray
        u_corr : numpy.ndarray
        p_hartree : numpy.ndarray
        p_corr : numpy.ndarray
        """
        hartrees, corrs = rdf.compute_sum_rule_integrals(potential)
        u_hartree = self.box_volume * hartrees[:, 0]
        u_corr    = self.box_volume * corrs[:, 0]
        p_hartree = -hartrees[:, 1] / 3.0
        p_corr    = -corrs[:, 1] / 3.0
        nkT       = self.total_num_density / self.beta_slices.mean()
        return nkT, u_hartree, u_corr, p_hartree, p_corr

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------

    def temp_energy_plot(
        self,
        process,
        phase: str = "production",
        info_list: list = None,
        show: bool = False,
        publication: bool = False,
        figname: str = None,
    ):
        """Plot temperature and total energy as a function of time.

        Parameters
        ----------
        process : sarkas.processes.Process
        phase : str
        info_list : list of str, optional
        show : bool
        publication : bool
        figname : str, optional

        Returns
        -------
        fig : matplotlib.figure.Figure
        T_axes : dict  — keys ``'main_plot'``, ``'hist_plot'``, ``'delta_plot'``
        E_axes : dict  — keys ``'main_plot'``, ``'hist_plot'``, ``'delta_plot'``
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from math import sqrt
        from os.path import join as os_path_join

        import scipy.stats as scp_stats
        import astropy.units as ast_u

        try:
            from seaborn import histplot as sns_histplot
        except ImportError:
            sns_histplot = None

        from .base import plot_labels

        # ---- 1. Load raw time-series from H5MD --------------------------
        with h5py.File(self.h5md_filepath, "r") as h5:
            time_arr = h5[
                f"observables/{self.species_names[0]}/temperature/time"
            ][:]
            temp_raw  = {}
            tot_e_raw = {}
            for sp in self.species_names:
                grp = h5["observables"][sp]
                temp_raw[sp]  = grp["temperature"]["value"][:]
                tot_e_raw[sp] = grp["total_energy"]["value"][:]

        sp_names    = list(self.species_names)
        temperature = np.zeros(len(time_arr))
        for isp, sp in enumerate(sp_names):
            frac         = self.species_num[isp] / self.total_num_ptcls
            temperature += temp_raw[sp] * frac
        tot_energy = sum(tot_e_raw[sp] for sp in sp_names)

        # ---- 2. Unit conversions ----------------------------------------
        K2eV     = ast_u.K.to(ast_u.eV, equivalencies=ast_u.temperature_energy())
        enrg_2eV = ast_u.erg.to(ast_u.eV) if self.units == "cgs" else ast_u.J.to(ast_u.eV)

        temperature_eV = temperature * K2eV
        tot_energy_eV  = tot_energy  * enrg_2eV

        time_mul, temp_mul, _, _, time_lbl, temp_lbl = plot_labels(
            time_arr, temperature_eV, "Time", "ElectronVolt", self.units
        )
        _, energy_mul, _, _, _, energy_lbl = plot_labels(
            time_arr, tot_energy_eV, "Time", "ElectronVolt", self.units
        )

        time        = time_mul  * time_arr
        Temperature = temp_mul  * temperature_eV
        Energy      = energy_mul * tot_energy_eV
        T_desired   = temp_mul  * self.T_desired * K2eV

        # ---- 3. Rolling cumulative averages (pure numpy) ----------------
        def _cumavg(arr):
            return np.cumsum(arr) / np.arange(1, len(arr) + 1)

        T_cumavg       = _cumavg(Temperature)
        E_cumavg       = _cumavg(Energy)
        Delta_T        = (Temperature - T_desired) * 100.0 / T_desired
        Delta_T_cumavg = _cumavg(Delta_T)
        Delta_E        = (Energy - Energy[0]) * 100.0 / abs(Energy[0])
        Delta_E_cumavg = _cumavg(Delta_E)

        # ---- 4. Theoretical distributions -------------------------------
        ensemble = "NVE" if phase == "production" else "NVT"
        self.calculate_beta_simulation(ensemble=ensemble)
        self.calculate_heat_capacity_simulation(ensemble=ensemble)

        dN = self.total_num_ptcls * self.dimensions
        if phase == "production":
            dN_2      = 0.5 * dN
            term      = 1.0 - dN_2 * (self.kB / self.specific_heat_volume)
            T_std     = T_desired * sqrt(max(term, 0.0) / dN_2)
            delta_E2  = (
                dN / (self.beta ** 2)
                * (1.0 - dN_2 * (self.kB / self.specific_heat_volume))
            )
        else:
            T_std    = T_desired * sqrt(2.0 / dN)
            delta_E2 = self.specific_heat_volume / (self.beta ** 2 * self.kB)

        delta_E_scaled  = sqrt(max(delta_E2, 0.0)) * enrg_2eV * energy_mul
        T_dist_desired  = scp_stats.norm(loc=T_desired,       scale=T_std)
        T_dist_actual   = scp_stats.norm(loc=Temperature.mean(), scale=T_std)
        E_dist_desired  = scp_stats.norm(loc=Energy.mean(),   scale=delta_E_scaled)
        E_dist_actual   = scp_stats.norm(loc=Energy.mean(),   scale=Energy.std())

        # ---- 5. Build figure --------------------------------------------
        fsz = 16
        fig = plt.figure(figsize=(20, 8))
        current_rcParams = plt.rcParams.copy()
        plt.rc("font",   size=fsz)
        plt.rc("axes",   titlesize=fsz, labelsize=fsz)
        plt.rc("xtick",  labelsize=fsz - 2)
        plt.rc("ytick",  labelsize=fsz - 2)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        if publication:
            plt.style.use("PUBstyle")
            gs           = GridSpec(3, 7)
            T_delta_plot = fig.add_subplot(gs[0,   0:2])
            T_main_plot  = fig.add_subplot(gs[1:3, 0:2])
            T_hist_plot  = fig.add_subplot(gs[1:3, 2])
            E_delta_plot = fig.add_subplot(gs[0,   4:6])
            E_main_plot  = fig.add_subplot(gs[1:3, 4:6])
            E_hist_plot  = fig.add_subplot(gs[1:3, 6])
        else:
            gs           = GridSpec(3, 8)
            Info_plot    = fig.add_subplot(gs[0:4, 0:2])
            T_delta_plot = fig.add_subplot(gs[0,   2:4])
            T_main_plot  = fig.add_subplot(gs[1:4, 2:4])
            T_hist_plot  = fig.add_subplot(gs[1:4, 4])
            E_delta_plot = fig.add_subplot(gs[0,   5:7])
            E_main_plot  = fig.add_subplot(gs[1:4, 5:7])
            E_hist_plot  = fig.add_subplot(gs[1:4, 7])

        # Temperature panels
        T_main_plot.plot(time, Temperature, alpha=0.7)
        T_main_plot.plot(time, T_cumavg, label="Rolling Average")
        T_main_plot.axhline(T_desired, ls="--", c="r", alpha=0.7, label="Desired T")
        T_main_plot.legend(loc="best")
        T_main_plot.set(ylabel="Temperature" + temp_lbl, xlabel="Time" + time_lbl)
        if phase == "equilibration":
            T_main_plot.set(ylim=(T_desired * 0.85, T_desired * 1.15))

        T_delta_plot.plot(time, Delta_T, alpha=0.5)
        T_delta_plot.plot(time, Delta_T_cumavg, alpha=0.8)
        T_delta_plot.set(xticks=[], ylabel=r"Deviation [%]")

        T_sorted = np.sort(Temperature)
        if sns_histplot is not None:
            sns_histplot(y=Temperature, bins="fd", stat="density", alpha=0.75, ax=T_hist_plot)
        else:
            T_hist_plot.hist(Temperature, bins="fd", density=True, alpha=0.75, orientation="horizontal")
        T_hist_plot.plot(T_dist_desired.pdf(T_sorted), T_sorted, ls="--", color="r",   alpha=0.7)
        T_hist_plot.plot(T_dist_actual.pdf(T_sorted),  T_sorted,           color=colors[1])
        T_hist_plot.set(ylabel=None, xlabel=None, xticks=[], yticks=[], ylim=T_main_plot.get_ylim())

        # Energy panels
        E_main_plot.plot(time, Energy, alpha=0.7)
        E_main_plot.plot(time, E_cumavg, label="Rolling Average")
        E_main_plot.axhline(Energy.mean(), ls="--", c="r", alpha=0.7, label="Avg")
        E_main_plot.legend(loc="best")
        E_main_plot.set(ylabel="Total Energy" + energy_lbl, xlabel="Time" + time_lbl)

        E_delta_plot.plot(time, Delta_E, alpha=0.5)
        E_delta_plot.plot(time, Delta_E_cumavg, alpha=0.8)
        E_delta_plot.set(xticks=[], ylabel=r"Deviation [%]")

        E_sorted = np.sort(Energy)
        if sns_histplot is not None:
            sns_histplot(y=Energy, bins="fd", stat="density", alpha=0.75, ax=E_hist_plot)
        else:
            E_hist_plot.hist(Energy, bins="fd", density=True, alpha=0.75, orientation="horizontal")
        E_hist_plot.plot(E_dist_desired.pdf(E_sorted), E_sorted, alpha=0.7, ls="--", color="r")
        E_hist_plot.plot(E_dist_actual.pdf(E_sorted),  E_sorted,                     color=colors[1])
        E_hist_plot.set(ylabel=None, xlabel=None, ylim=E_main_plot.get_ylim(), xticks=[], yticks=[])

        # Info panel (non-publication)
        if not publication:
            from math import sqrt as msqrt
            dt_mul, _, _, _, dt_lbl, _ = plot_labels(
                np.atleast_1d(process.integrator.dt), tot_energy_eV, "Time", "Energy", self.units
            )
            delta_t        = dt_mul * process.integrator.dt
            completed_steps = self.dump_step * (len(time_arr) - 1)

            Info_plot.axis([0, 10, 0, 10])
            Info_plot.grid(False)
            Info_plot.text(0.0, 10,  f"Job ID: {self.job_id}")
            Info_plot.text(0.0, 9.5, f"Phase: {phase.capitalize()}")
            Info_plot.text(0.0, 9.0, f"No. of species = {len(self.species_num)}")
            y = 8.5
            for isp, sp in enumerate(process.species):
                if sp.name != "electron_background":
                    Info_plot.text(0.0, y,       f"Species {isp + 1} : {sp.name}")
                    Info_plot.text(0.0, y - 0.5, f"  No. of particles = {sp.num}")
                    Info_plot.text(0.0, y - 1.0,
                        f"  Temperature = {temp_mul * sp.temperature * K2eV:.2f} {temp_lbl}")
                    y -= 1.5
            y -= 0.25

            if info_list is None:
                int_type = {
                    "equilibration": process.integrator.equilibration_type,
                    "magnetization": process.integrator.magnetization_type,
                    "production":    process.integrator.production_type,
                }
                eq_cycles   = int(process.parameters.equilibration_steps
                                  * process.integrator.dt / self.plasma_period)
                prod_cycles = int(process.parameters.production_steps
                                  * process.integrator.dt / self.plasma_period)
                info_list = [
                    f"Total N = {process.parameters.total_num_ptcls}",
                ]
                if process.integrator.thermalization:
                    info_list += [
                        f"Thermostat: {process.integrator.thermostat_type}",
                        f"  Berendsen rate = {process.integrator.thermalization_rate:.2f}",
                    ]
                info_list += [
                    f"Equilibration cycles = {eq_cycles}",
                    f"Potential: {process.potential.type}",
                    f"  Tot Force Error = {process.potential.force_error:.2e}",
                    f"Integrator: {int_type[phase]}",
                ]
                if int_type[phase] == "langevin":
                    info_list.append(
                        f"Langevin gamma = {process.integrator.langevin_gamma:.4e}"
                    )
                tau_str = (
                    r"  Plasma period $\tau_{\omega_p}$ = "
                    + f"{self.plasma_period * dt_mul:.2f} {dt_lbl}"
                )
                info_list += [
                    tau_str,
                    f"  dt = {delta_t:.2f} {dt_lbl} = "
                    + f"{delta_t / self.plasma_period / dt_mul:.2e} tau_wp",
                    f"Completed steps = {completed_steps}",
                    f"Total steps = {self.no_steps}",
                    f"{100 * completed_steps / self.no_steps:.2f} % Completed",
                    f"Production time = "
                    + f"{self.no_steps * delta_t / dt_mul * time_mul:.2f} {time_lbl}",
                    f"Production cycles = {prod_cycles}",
                ]

            for text_str in info_list:
                Info_plot.text(0.0, y, text_str)
                y -= 0.5
            Info_plot.axis("off")
            fig.tight_layout()

        # ---- 7. Save ----------------------------------------------------
        stem = figname if figname else "Plot_EnsembleCheck"
        from os.path import join as os_path_join
        fig.savefig(os_path_join(self.saving_dir, f"{stem}_{self.job_id}.png"))
        if show:
            fig.show()

        plt.rcParams = current_rcParams
        return (
            fig,
            {"main_plot": T_main_plot, "hist_plot": T_hist_plot, "delta_plot": T_delta_plot},
            {"main_plot": E_main_plot, "hist_plot": E_hist_plot, "delta_plot": E_delta_plot},
        )

    # ------------------------------------------------------------------
    # Deprecated public API
    # ------------------------------------------------------------------

    def update_args(self, **kwargs):
        """.. deprecated:: Pass kwargs directly to :meth:`setup`."""
        warnings.warn(
            "update_args() is deprecated. Pass kwargs to setup() directly.",
            DeprecationWarning, stacklevel=2,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    def calc_slices_data(self):
        """.. deprecated:: Use :meth:`compute` instead."""
        warnings.warn(
            "calc_slices_data() is deprecated. Call compute() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self._calc_slices()

    def average_slices_data(self):
        """.. deprecated:: Averaging is handled automatically by :meth:`compute`."""
        warnings.warn(
            "average_slices_data() is deprecated. "
            "Averaging is handled automatically inside compute().",
            DeprecationWarning, stacklevel=2,
        )
        self._average_slices()

    def calc_acf_slices_data(self):
        """.. deprecated:: Use :meth:`compute_acf` instead."""
        warnings.warn(
            "calc_acf_slices_data() is deprecated. Call compute_acf() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self._calc_acf_slices()

    def average_acf_slices_data(self):
        """.. deprecated:: ACF averaging is handled automatically by :meth:`compute_acf`."""
        warnings.warn(
            "average_acf_slices_data() is deprecated. "
            "ACF averaging is handled automatically inside compute_acf().",
            DeprecationWarning, stacklevel=2,
        )


# ---------------------------------------------------------------------------
# PressureTensor
# ---------------------------------------------------------------------------


class PressureTensor(Observable):
    """Pressure Tensor.

    Computes the total pressure, its fluctuations, and the individual
    pressure-tensor elements from the simulation H5MD file.

    Attributes
    ----------
    kinetic_potential_division : bool
        Whether kinetic and potential contributions are stored separately.
        Currently unused — reserved for future expansion.
    """

    def __init__(self):
        super().__init__()
        self.__name__                  = "pressure_tensor"
        self.__long_name__             = "Pressure Tensor"
        self.acf_observable            = True
        self.kinetic_potential_division = False

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
        self.__dict__.update(kwargs)
        self.update_finish()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _species_list(self) -> list:
        names = list(self.species_names)
        return [*names, "Total"] if len(names) > 1 else names

    @property
    def _tensor_labels(self) -> list:
        """Upper-triangular component labels, e.g. ``['XX','XY','XZ','YY','YZ','ZZ']``."""
        idx = np.triu_indices(self.dimensions)
        return [
            f"{self.dim_labels[i]}{self.dim_labels[j]}"
            for i, j in zip(idx[0], idx[1])
        ]

    def _standard_attrs(self) -> dict:
        return {
            "no_slices":    self.no_slices,
            "block_length": self.block_length,
            "dumps_shift":  self.dumps_shift,
            "h5md_path":    self.h5md_filepath,
        }

    # ------------------------------------------------------------------
    # Public compute entry points
    # ------------------------------------------------------------------

    @compute_doc
    def compute(
        self,
        calculate_acf: bool = False,
        kin_pot_division: bool = False,
    ):
        """Compute pressure and pressure tensor for each slice.

        Parameters
        ----------
        calculate_acf : bool
            Also compute ACFs after the main calculation.  Default ``False``.
        kin_pot_division : bool
            Reserved for kinetic/potential decomposition.  Default ``False``.
        """
        self.kinetic_potential_division = kin_pot_division
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self._calc_slices()
        self._average_slices()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )
        if calculate_acf:
            self.compute_acf(kin_pot_division=kin_pot_division)

    @compute_acf_doc
    def compute_acf(self, kin_pot_division: bool = False):
        """Compute autocorrelation functions for pressure and pressure tensor."""
        self.kinetic_potential_division = kin_pot_division
        t0 = self.timer.current()
        self._calc_acf_slices()
        self._average_acf_slices()
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            self.__long_name__ + " ACF Calculation",
            self.timer.time_division(tend - t0),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # Core slice computation
    # ------------------------------------------------------------------

    @calc_slices_doc
    def _calc_slices(self):
        """Read pressure-tensor data from H5MD and write one slice at a time."""
        species_list  = self._species_list
        tensor_labels = self._tensor_labels
        no_sp         = len(species_list)
        no_comp       = len(tensor_labels)
        idx           = np.triu_indices(self.dimensions)

        # ---- read full pressure-tensor arrays from H5MD (once) ----------
        with h5py.File(self.h5md_filepath, "r") as h5:
            first_sp = self.species_names[0]
            time_arr = h5[f"observables/{first_sp}/pressure_tensor/time"][:]
            sp_pt    = [
                h5[f"observables/{sp}/pressure_tensor/value"][:].astype(float)
                for sp in self.species_names
            ]
        total_pt = sum(sp_pt)   # (no_dumps, D, D)
        all_pt   = [*sp_pt, total_pt]

        # ---- pre-allocate HDF5 store (two variables share the same file) -
        coords = {
            "species":   species_list,
            "component": tensor_labels,
            "time":      time_arr[:self.block_length],
            "slice":     np.arange(self.no_slices),
        }
        # pressure dims: (species, time, slice)
        # tensor  dims: (species, component, time, slice)
        # _preallocate_store uses the full coords dict for _DIMS — we need
        # to store two variables with different dim subsets, so we manage
        # the HDF5 file directly here.
        with h5py.File(self.hdf_store_path, "w") as f:
            cg = f.require_group("coords")
            dt_str = h5py.string_dtype()
            cg.create_dataset("species",   data=np.array(species_list,  dtype=object), dtype=dt_str)
            cg.create_dataset("component", data=np.array(tensor_labels, dtype=object), dtype=dt_str)
            cg.create_dataset("time",  data=time_arr[:self.block_length])
            cg.create_dataset("slice", data=np.arange(self.no_slices))
            for ds in cg.values():
                ds.attrs["_is_coord"] = True

            dg = f.require_group("data")
            p_ds = dg.create_dataset(
                "pressure",
                shape=(no_sp, self.block_length, self.no_slices),
                dtype="float64",
                chunks=(no_sp, self.block_length, 1),
                compression="gzip", compression_opts=5,
                fillvalue=np.nan,
            )
            p_ds.attrs["_DIMS"] = ["species", "time", "slice"]

            t_ds = dg.create_dataset(
                "tensor",
                shape=(no_sp, no_comp, self.block_length, self.no_slices),
                dtype="float64",
                chunks=(no_sp, no_comp, self.block_length, 1),
                compression="gzip", compression_opts=5,
                fillvalue=np.nan,
            )
            t_ds.attrs["_DIMS"] = ["species", "component", "time", "slice"]

            for k, v in self._standard_attrs().items():
                f.attrs[k] = v

        # ---- fill one slice at a time -----------------------------------
        for isl in tqdm(
            range(self.no_slices),
            desc="Pressure tensor slices",
            disable=not self.verbose,
        ):
            start  = isl * self.dumps_shift
            end    = start + self.block_length
            p_arr  = np.zeros((no_sp, self.block_length))
            t_arr  = np.zeros((no_sp, no_comp, self.block_length))
            for isp, pt in enumerate(all_pt):
                block          = pt[start:end]              # (block_length, D, D)
                p_arr[isp]     = np.trace(block, axis1=-2, axis2=-1) / self.dimensions
                t_arr[isp]     = block[:, idx[0], idx[1]].T  # (no_comp, block_length)

            self._write_slice({"pressure": p_arr, "tensor": t_arr}, isl)

    @avg_slices_doc
    def _average_slices(self):
        """Compute mean and std of pressure and tensor over slices."""
        self._write_mean_std(
            ["pressure", "tensor"],
            ddof=min(1, self.no_slices - 1),
        )

    # ------------------------------------------------------------------
    # ACF slice computation
    # ------------------------------------------------------------------

    @calc_acf_slices_doc
    def _calc_acf_slices(self):
        """Compute bulk pressure ACF and full tensor ACF for each slice."""
        ds        = self.read_dataset()
        p_arr     = ds["pressure"].values   # (no_sp, block_length, no_slices)
        t_arr     = ds["tensor"].values     # (no_sp, no_comp, block_length, no_slices)
        sp_vals   = list(ds["pressure"].coords["species"].values)
        comp_vals = list(ds["tensor"].coords["component"].values)
        no_sp     = len(sp_vals)
        no_comp   = len(comp_vals)
        bl        = self.block_length
        ddof      = min(1, self.no_slices - 1)

        bulk_acf   = np.zeros((no_sp, bl, self.no_slices))
        tensor_acf = np.zeros((no_sp, no_comp, no_comp, bl, self.no_slices))

        for isl in tqdm(
            range(self.no_slices),
            desc="Pressure tensor ACF slices",
            disable=not self.verbose,
        ):
            for isp in range(no_sp):
                delta_p = p_arr[isp, :, isl]
                delta_p = delta_p - delta_p.mean()
                bulk_acf[isp, :, isl] = correlationfunction(delta_p, delta_p)
                for ic in range(no_comp):
                    d1 = t_arr[isp, ic, :, isl]
                    for jc in range(ic, no_comp):
                        d2  = t_arr[isp, jc, :, isl]
                        acf = correlationfunction(d1, d2)
                        tensor_acf[isp, ic, jc, :, isl] = acf
                        if ic != jc:
                            tensor_acf[isp, jc, ic, :, isl] = acf

        # Append ACF datasets to the existing HDF5 file
        with h5py.File(self.hdf_store_path, "a") as f:
            dg = f.require_group("data")
            cg = f["coords"]

            # Add "lag" coord (same values as "time") if not already present
            if "lag" not in cg:
                cg.create_dataset("lag", data=cg["time"][:])
                cg["lag"].attrs["_is_coord"] = True

            for name, arr, dims in [
                ("acf_pressure",
                 bulk_acf,
                 ["species", "lag", "slice"]),
                ("mean_acf_pressure",
                 bulk_acf.mean(-1),
                 ["species", "lag"]),
                ("std_acf_pressure",
                 bulk_acf.std(-1, ddof=ddof),
                 ["species", "lag"]),
                ("acf_tensor",
                 tensor_acf,
                 ["species", "component_row", "component_col", "lag", "slice"]),
                ("mean_acf_tensor",
                 tensor_acf.mean(-1),
                 ["species", "component_row", "component_col", "lag"]),
                ("std_acf_tensor",
                 tensor_acf.std(-1, ddof=ddof),
                 ["species", "component_row", "component_col", "lag"]),
            ]:
                if name in dg:
                    del dg[name]
                out = dg.create_dataset(name, data=arr,
                                        compression="gzip", compression_opts=5)
                out.attrs["_DIMS"] = dims

            # component_row and component_col share the same values as component
            for coord_name in ("component_row", "component_col"):
                if coord_name not in cg:
                    cg.create_dataset(
                        coord_name,
                        data=cg["component"][:],
                        dtype=cg["component"].dtype,
                    )
                    cg[coord_name].attrs["_is_coord"] = True

    @avg_acf_slices_doc
    def _average_acf_slices(self):
        """Mean and std are computed inside :meth:`_calc_acf_slices`."""
        pass

    # ------------------------------------------------------------------
    # Sum rule
    # ------------------------------------------------------------------

    def sum_rule(self, beta: float, rdf, potential):
        r"""Calculate pressure-tensor sum rule integrals from the RDF.

        .. math::

            \sigma_{zzzz} = \frac{n}{\beta^2}
            \left[3 + \frac{2\beta}{15}I^{(1)} + \frac{\beta}{5}I^{(2)}\right]

            \sigma_{zzxx} = \frac{n}{\beta^2}
            \left[1 - \frac{2\beta}{5}I^{(1)} + \frac{\beta}{15}I^{(2)}\right]

            \sigma_{xyxy} = \frac{n}{\beta^2}
            \left[1 + \frac{4\beta}{15}I^{(1)} + \frac{\beta}{15}I^{(2)}\right]

        Parameters
        ----------
        beta : float
            Inverse temperature. Use :attr:`Thermodynamics.beta`.
        rdf : RadialDistributionFunction
        potential : sarkas.potentials.core.Potential

        Returns
        -------
        sigma_zzzz, sigma_zzxx, sigma_xyxy : float
        """
        hartrees, corrs = rdf.compute_sum_rule_integrals(potential)
        I_1  = hartrees[:, 1].sum() + corrs[:, 1].sum()
        I_2  = hartrees[:, 2].sum() + corrs[:, 2].sum()
        nkT  = self.total_num_density / beta

        sigma_zzzz = 3.0 * nkT + (2.0 / 15.0) * I_1 + I_2 / 5.0
        sigma_zzxx = nkT       - (2.0 / 5.0)  * I_1 + I_2 / 15.0
        sigma_xyxy = nkT       + (4.0 / 15.0) * I_1 + I_2 / 15.0
        return sigma_zzzz, sigma_zzxx, sigma_xyxy

    # ------------------------------------------------------------------
    # Deprecated public API
    # ------------------------------------------------------------------

    def update_args(self, **kwargs):
        """.. deprecated:: Pass kwargs directly to :meth:`setup`."""
        warnings.warn(
            "update_args() is deprecated. Pass kwargs to setup() directly.",
            DeprecationWarning, stacklevel=2,
        )
        self.__dict__.update(kwargs)
        self.update_finish()

    def calc_slices_data(self):
        """.. deprecated:: Use :meth:`compute` instead."""
        warnings.warn(
            "calc_slices_data() is deprecated. Call compute() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self._calc_slices()

    def average_slices_data(self):
        """.. deprecated:: Averaging is handled automatically by :meth:`compute`."""
        warnings.warn(
            "average_slices_data() is deprecated. "
            "Averaging is handled automatically inside compute().",
            DeprecationWarning, stacklevel=2,
        )
        self._average_slices()

    def calc_acf_slices_data(self):
        """.. deprecated:: Use :meth:`compute_acf` instead."""
        warnings.warn(
            "calc_acf_slices_data() is deprecated. Call compute_acf() instead.",
            DeprecationWarning, stacklevel=2,
        )
        self._calc_acf_slices()

    def average_acf_slices_data(self):
        """.. deprecated:: ACF averaging is handled automatically by :meth:`compute_acf`."""
        warnings.warn(
            "average_acf_slices_data() is deprecated. "
            "ACF averaging is handled automatically inside compute_acf().",
            DeprecationWarning, stacklevel=2,
        )