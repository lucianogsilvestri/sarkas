"""
Transport Module.

All I/O uses xarray + zarr (no pandas).  Each transport-coefficient class
stores its results in an xarray-compatible zarr store at::

    <saving_dir>/<name>_<job_id>.zarr

Store layout::

    <quantity>      (slice, time)   — raw per-slice integrals
    mean_<quantity> (time,)         — slice mean
    std_<quantity>  (time,)         — slice std
"""

import inspect
from copy import deepcopy
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

import numpy as np
import xarray as xr
from matplotlib.pyplot import subplots
from numpy import array, column_stack, ndarray, pi, rint
from os import mkdir as os_mkdir
from os.path import exists as os_path_exists
from os.path import join as os_path_join
from scipy.integrate import cumulative_trapezoid
from warnings import warn

import zarr
import numcodecs

from ..utilities.io import print_to_logger
from ..utilities.timing import datetime_stamp, SarkasTimer

# Sarkas Modules
from .observables import plot_labels, Thermodynamics

_COMPRESSOR = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)


class TransportCoefficients:
    """Transport Coefficients parent."""

    def __init__(self):
        self.time_array = None
        self.saving_dir = None
        self.zarr_store_path = None
        #
        # To be copied from parameters class
        self.postprocessing_dir = None
        self.units = None
        self.job_id = None
        self.verbose = None
        self.dt = None
        self.units_dict = None
        self.total_plasma_frequency = None
        self.dimensions = None
        self.box_volume = None
        self.pbox_volume = None
        self.phase = None
        self.no_slices = None
        self.dump_step = None

        self.kB = 0.0
        self.beta_slices = 0.0
        #
        self.log_file = None
        self.timer = SarkasTimer()

    def setup(self, params, observable, thermodynamics):
        """
        Set up the necessary parameters and structures for computing transport coefficients.

        Parameters
        ----------
        params : :class:`sarkas.core.Parameters`
        observable : :class:`sarkas.tools.observables.Observable`
        thermodynamics : :class:`sarkas.tools.observables.Thermodynamics`
        """
        self.copy_params(params=params)
        self.postprocessing_dir = self.directory_tree["postprocessing"]["path"]
        self.get_observable_data(observable)

        thermodynamics.calculate_beta_slices()
        self.beta_slices = thermodynamics.beta_slices.copy()

        self.make_directories()
        self.zarr_store_path = os_path_join(
            self.saving_dir, f"{self.__name__}_{self.job_id}.zarr"
        )
        self.log_file = os_path_join(self.saving_dir, f"{self.__name__}_logfile.out")

        datetime_stamp(self.log_file)
        self.pretty_print()

    def copy_params(self, params):
        for i, val in params.__dict__.items():
            if not inspect.ismethod(val):
                if isinstance(val, dict):
                    self.__dict__[i] = deepcopy(val)
                elif isinstance(val, ndarray):
                    self.__dict__[i] = val.copy()
                else:
                    self.__dict__[i] = val

    def __repr__(self):
        sortedDict = dict(sorted(self.__dict__.items(), key=lambda x: x[0].lower()))
        disp = f"{self.__name__}( \n"
        for key, value in sortedDict.items():
            disp += "\t{} : {}\n".format(key, value)
        disp += ")"
        return disp

    def get_observable_data(self, observable):
        """Copy slice/block metadata from observable.

        Parameters
        ----------
        observable : :class:`sarkas.tools.observables.Observable`
        """
        self.phase = observable.phase
        self.no_slices = observable.no_slices
        self.block_length = observable.block_length
        self.timesteps_per_slice = observable.timesteps_per_slice
        self.timesteps_per_plasma_period = observable.timesteps_per_plasma_period
        self.plasma_period = observable.plasma_period
        self.dump_step = observable.dump_step
        self.observable_zarr_path = observable.zarr_store_path

        # Build time axis from xarray-compatible zarr store
        try:
            ds = xr.open_zarr(observable.zarr_store_path)
            # Try to find a 'time' or 'lag' dimension coordinate
            for var in ds.data_vars:
                arr = ds[var]
                for dim in ("time", "lag"):
                    if dim in arr.coords:
                        self.time_array = arr.coords[dim].values
                        break
                else:
                    continue
                break
            else:
                self.time_array = (
                    np.arange(observable.block_length) * observable.dt * observable.dump_step
                )
        except Exception:
            self.time_array = (
                np.arange(observable.block_length) * observable.dt * observable.dump_step
            )

    def _write_transport_zarr(self, quantities: dict, mode: str = "a"):
        """Write transport coefficient arrays to the xarray-compatible zarr store.

        Parameters
        ----------
        quantities : dict
            Mapping of ``{name: array}`` where each array has shape
            ``(no_slices, block_length)``.
        mode : str
            Zarr write mode; ``'w'`` to create/overwrite, ``'a'`` to append.
        """
        from numcodecs import Blosc
        compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.BITSHUFFLE)
        ddof = min(1, self.no_slices - 1)
        variables = {}
        for name, slices_arr in quantities.items():
            da = xr.DataArray(
                slices_arr,
                dims=["slice", "time"],
                coords={
                    "slice": np.arange(self.no_slices),
                    "time": self.time_array,
                },
            )
            variables[name] = da
            variables[f"mean_{name}"] = da.mean("slice").compute() if hasattr(da, "compute") else xr.DataArray(
                slices_arr.mean(axis=0), dims=["time"], coords={"time": self.time_array}
            )
            variables[f"std_{name}"] = xr.DataArray(
                slices_arr.std(axis=0, ddof=ddof), dims=["time"], coords={"time": self.time_array}
            )
        ds = xr.Dataset(variables)
        encoding = {}
        for var in ds.data_vars:
            arr = ds[var]
            chunks = [1 if d == "slice" else arr.sizes[d] for d in arr.dims]
            encoding[var] = {"compressor": compressor, "chunks": chunks}
        ds.to_zarr(self.zarr_store_path, mode=mode, encoding=encoding)

    def save_zarr(self, quantities: dict):
        """Alias for backward compatibility; calls :meth:`_write_transport_zarr`."""
        mode = "w" if not os_path_exists(self.zarr_store_path) else "a"
        self._write_transport_zarr(quantities, mode=mode)

    def parse(self):
        """Return the xarray Dataset containing the transport coefficients.

        Returns
        -------
        xr.Dataset
        """
        return xr.open_zarr(self.zarr_store_path)

    def make_directories(self):
        transport_dir = os_path_join(self.postprocessing_dir, "TransportCoefficients")
        if not os_path_exists(transport_dir):
            os_mkdir(transport_dir)

        coeff_dir = os_path_join(transport_dir, self.__name__)
        if not os_path_exists(coeff_dir):
            os_mkdir(coeff_dir)

        self.saving_dir = os_path_join(coeff_dir, self.phase.capitalize())
        if not os_path_exists(self.saving_dir):
            os_mkdir(self.saving_dir)

    # -----------------------------------------------------------------------
    # Deprecated stub methods
    # -----------------------------------------------------------------------

    def diffusion(self, *args, **kwargs):
        warn("Deprecated. Use the Diffusion class.", DeprecationWarning)

    def electrical_conductivity(self, *args, **kwargs):
        warn("Deprecated. Use the ElectricalConductivity class.", DeprecationWarning)

    def interdiffusion(self, *args, **kwargs):
        warn("Deprecated. Use the InterDiffusion class.", DeprecationWarning)

    def viscosity(self, *args, **kwargs):
        warn("Deprecated. Use the Viscosity class.", DeprecationWarning)

    # -----------------------------------------------------------------------
    # Plotting
    # -----------------------------------------------------------------------

    def plot_tc(self, time, acf_data, tc_data, acf_name, tc_name, figname, show: bool = False):
        """Dual plot of ACF and running transport coefficient."""
        fig, (ax1, ax2) = subplots(1, 2, figsize=(16, 7))
        ax3 = ax1.twiny()
        ax4 = ax2.twiny()

        xmul, ymul, _, _, xlbl, ylbl = plot_labels(time, tc_data[:, 0], "Time", self.__long_name__, self.units)

        ax1.plot(xmul * time, acf_data[:, 0] / acf_data[0, 0])
        ax1.fill_between(
            xmul * time,
            (acf_data[:, 0] - acf_data[:, 1]) / (acf_data[0, 0] - acf_data[0, 1]),
            (acf_data[:, 0] + acf_data[:, 1]) / (acf_data[0, 0] + acf_data[0, 1]),
            alpha=0.2,
        )

        ax2.plot(xmul * time, ymul * tc_data[:, 0])
        ax2.fill_between(
            xmul * time,
            ymul * (tc_data[:, 0] - tc_data[:, 1]),
            ymul * (tc_data[:, 0] + tc_data[:, 1]),
            alpha=0.2,
        )

        xlims = (xmul * time[1], xmul * time[-1] * 1.5)
        ax1.set(xlim=xlims, xscale="log", ylim=(-0.5, 1.1), ylabel=acf_name, xlabel=r"Time difference" + xlbl)
        xlims = (xmul * time[1], xmul * time[-1] * 1.05)
        ax2.set(xlim=xlims, ylim=(-0.05, ax2.get_ylim()[1]), ylabel=tc_name + ylbl, xlabel=r"$\tau$" + xlbl, xscale="log")

        ax3.set(xlim=(1, len(time) * 1.5), xscale="log")
        ax4.set(xlim=(1, len(time) * 1.5), xscale="log")
        for axi in [ax3, ax4]:
            axi.grid(alpha=0.1)
            axi.set(xlabel="Index")

        fig.tight_layout()
        fig.savefig(os_path_join(self.saving_dir, figname))

        if show:
            fig.show()

        return fig, (ax1, ax2, ax3, ax4)

    def pretty_print_msg(self, info: str = None, append_info: str = None):
        tc_name = f" {self.__long_name__} "
        dtau = self.dt * self.dump_step
        tau = dtau * self.block_length
        tau_wp = rint(tau / self.plasma_period).astype(int)
        if info:
            msg = info
        else:
            msg = (
                f"\n\n{tc_name:=^70}\n"
                f"Data saved in: \n {self.zarr_store_path}\n"
                f"No. of slices = {self.no_slices}\n"
                f"No. dumps per block = {self.block_length}\n"
                f"Total time interval of autocorrelation function: tau = {tau:.4e} {self.units_dict['time']} ~ {tau_wp} plasma periods\n"
                f"Time interval step: dtau = {dtau:.4e} ~ {dtau / self.plasma_period:.4e} plasma periods"
            )
            if append_info:
                msg += append_info
        return msg

    def pretty_print(self, info: str = None, append_info: str = None):
        msg = self.pretty_print_msg(info, append_info)
        print_to_logger(message=msg, log_file=self.log_file, print_to_screen=self.verbose)

    def time_stamp(self, message: str, timing: tuple):
        import sys
        screen = sys.stdout
        f_log = open(self.log_file, "a+")
        repeat = 2 if self.verbose else 1
        t_hrs, t_min, t_sec, t_msec, t_usec, t_nsec = timing
        sys.stdout = f_log
        while repeat > 0:
            if t_hrs == 0 and t_min == 0 and t_sec <= 2:
                print(f"\n{message} Time: {int(t_sec)} sec {int(t_msec)} msec {int(t_usec)} usec {int(t_nsec)} nsec")
            else:
                print(f"\n{message} Time: {int(t_hrs)} hrs {int(t_min)} min {int(t_sec)} sec")
            repeat -= 1
            sys.stdout = screen
        f_log.close()


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


class Diffusion(TransportCoefficients):
    """Self-diffusion coefficient from the Green-Kubo formula.

    Reads from the :class:`~sarkas.tools.observables.VelocityAutoCorrelationFunction`
    zarr store.  The VACF layout is::

        acf  (no_species, D+1, block_length, no_slices)

    Access pattern: ``acf[isp, -1, :, isl]`` — isotropic VACF for species
    ``isp``, slice ``isl``.
    """

    def __init__(self):
        self.__name__ = "Diffusion"
        self.__long_name__ = "Diffusion"
        self.required_observable = "Velocity Autocorrelation Function"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """Calculate the diffusion coefficient from the Green-Kubo formula.

        Parameters
        ----------
        observable : :class:`~sarkas.tools.observables.VelocityAutoCorrelationFunction`
        plot : bool
        display_plot : bool
        """
        t0 = self.timer.current()
        const = 1.0 / self.dimensions

        ds_obs = xr.open_zarr(observable.zarr_store_path)
        acf_da = ds_obs["acf"]  # (species, component, lag, slice)

        quantities = {}
        if not observable.magnetized:
            for sp in observable.species_names:
                slices_arr = np.zeros((self.no_slices, self.block_length))
                for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
                    integrand = acf_da.sel(species=sp, component="Total").isel(slice=isl).values
                    slices_arr[isl] = const * cumulative_trapezoid(integrand, x=self.time_array, initial=0.0)
                quantities[f"{sp}_Diffusion"] = slices_arr
        else:
            D = observable.dimensions
            dim_labels = list(observable.dim_labels)
            for sp in observable.species_names:
                sl_par  = np.zeros((self.no_slices, self.block_length))
                sl_perp = np.zeros((self.no_slices, self.block_length))
                for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
                    integrand_par = acf_da.sel(species=sp, component=dim_labels[D - 1]).isel(slice=isl).values
                    sl_par[isl] = cumulative_trapezoid(integrand_par, x=self.time_array, initial=0.0)
                    integrand_perp = 0.5 * (
                        acf_da.sel(species=sp, component=dim_labels[0]).isel(slice=isl).values
                        + acf_da.sel(species=sp, component=dim_labels[1]).isel(slice=isl).values
                    )
                    sl_perp[isl] = cumulative_trapezoid(integrand_perp, x=self.time_array, initial=0.0)
                quantities[f"{sp}_Diffusion_Parallel"]      = sl_par
                quantities[f"{sp}_Diffusion_Perpendicular"] = sl_perp

        tend = self.timer.current()
        self.time_stamp("Diffusion Calculation", self.timer.time_division(tend - t0))
        self.save_zarr(quantities)

        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, scaling: tuple = (1.0, 1.0), display_plot: bool = False, **kwargs):
        """Dual plot of VACF and diffusion coefficient."""
        ds = xr.open_zarr(self.zarr_store_path)
        ds_obs = xr.open_zarr(observable.zarr_store_path)

        if not isinstance(scaling, tuple):
            scaling = (scaling, 1.0)

        figs = {}
        axes = {}

        if not observable.magnetized:
            for sp in observable.species_names:
                acf_mean = ds_obs["mean_acf"].sel(species=sp, component="Total").values
                acf_std  = ds_obs["std_acf"].sel(species=sp, component="Total").values
                tc_mean  = ds[f"mean_{sp}_Diffusion"].values
                tc_std   = ds[f"std_{sp}_Diffusion"].values

                fig, (ax1, ax2, ax3, ax4) = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_mean / scaling[0], acf_std / scaling[0])),
                    tc_data=column_stack((tc_mean / scaling[1], tc_std / scaling[1])),
                    acf_name=f"{sp} VACF",
                    tc_name=f"{sp} Diffusion",
                    figname=f"{sp}_Diffusion_Plot.png",
                    show=display_plot,
                )
                figs[sp] = fig
                axes[sp] = (ax1, ax2, ax3, ax4)
        else:
            D = observable.dimensions
            dim_labels = list(observable.dim_labels)
            for sp in observable.species_names:
                for comp, dim_lbl, key in (
                    ("Parallel",     dim_labels[D - 1], f"{sp}_Diffusion_Parallel"),
                    ("Perpendicular", dim_labels[0],    f"{sp}_Diffusion_Perpendicular"),
                ):
                    tc_mean  = ds[f"mean_{key}"].values
                    tc_std   = ds[f"std_{key}"].values
                    acf_mean = ds_obs["mean_acf"].sel(species=sp, component=dim_lbl).values
                    acf_std  = ds_obs["std_acf"].sel(species=sp, component=dim_lbl).values
                    fig, axes_tuple = self.plot_tc(
                        time=self.time_array,
                        acf_data=column_stack((acf_mean, acf_std)),
                        tc_data=column_stack((tc_mean, tc_std)),
                        acf_name=f"{sp} VACF {comp}",
                        tc_name=f"{sp} Diffusion {comp}",
                        figname=f"{sp}_{comp}_Diffusion_Plot.png",
                        show=display_plot,
                    )
                    figs.setdefault(sp, {})[comp] = fig
                    axes.setdefault(sp, {})[comp] = axes_tuple

        return figs, axes


# ---------------------------------------------------------------------------
# InterDiffusion
# ---------------------------------------------------------------------------


class InterDiffusion(TransportCoefficients):
    """Interdiffusion coefficient from the Green-Kubo formula.

    Reads from the :class:`~sarkas.tools.observables.DiffusionFlux` zarr store.
    Layout::

        acf  (no_fluxes, no_fluxes, D+1, block_length, no_slices)

    Access pattern: ``acf[i, j, -1, :, isl]`` — isotropic cross-correlation
    of flux ``i`` and flux ``j``, slice ``isl``.
    """

    def __init__(self):
        self.__name__ = "InterDiffusion"
        self.__long_name__ = "InterDiffusion"
        self.required_observable = "Diffusion Flux"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """Calculate the interdiffusion coefficient.

        Parameters
        ----------
        observable : :class:`~sarkas.tools.observables.DiffusionFlux`
        """
        t0 = self.timer.current()

        no_fluxes = observable.no_fluxes
        const = 1.0 / (3.0 * observable.total_num_ptcls * observable.species_concentrations.prod())

        ds_obs = xr.open_zarr(observable.zarr_store_path)
        acf_da = ds_obs["acf"]  # (flux_row, flux_col, component, lag, slice)
        flux_labels = list(acf_da.coords["flux_row"].values)

        quantities = {}
        for i in range(no_fluxes):
            for j in range(no_fluxes):
                fi = flux_labels[i]
                fj = flux_labels[j]
                slices_arr = np.zeros((self.no_slices, self.block_length))
                for isl in tqdm(range(self.no_slices), disable=not self.verbose):
                    integrand = acf_da.sel(flux_row=fi, flux_col=fj, component="Total").isel(slice=isl).values
                    slices_arr[isl] = const * cumulative_trapezoid(integrand, self.time_array, initial=0.0)
                quantities[f"InterDiffusion_{i}{j}"] = slices_arr

        tend = self.timer.current()
        self.time_stamp("Interdiffusion Calculation", self.timer.time_division(tend - t0))
        self.save_zarr(quantities)

        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        no_fluxes = observable.no_fluxes
        ds = xr.open_zarr(self.zarr_store_path)
        ds_obs = xr.open_zarr(observable.zarr_store_path)
        flux_labels = list(ds_obs["acf"].coords["flux_row"].values)

        figs = {}
        axes = {}
        for i in range(no_fluxes):
            fi = flux_labels[i]
            key = f"InterDiffusion_{i}{i}"
            acf_mean = ds_obs["mean_acf"].sel(flux_row=fi, flux_col=fi, component="Total").values
            acf_std  = ds_obs["std_acf"].sel(flux_row=fi, flux_col=fi, component="Total").values
            tc_mean  = ds[f"mean_{key}"].values
            tc_std   = ds[f"std_{key}"].values

            fig, axes_tuple = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_mean, acf_std)),
                tc_data=column_stack((tc_mean, tc_std)),
                acf_name=f"Diffusion Flux ACF {i}",
                tc_name=f"InterDiffusion Flux {i}",
                figname=f"InterDiffusion_Flux{i}_Plot.png",
                show=display_plot,
            )
            figs[f"Flux_{i}"] = fig
            axes[f"Flux_{i}"] = axes_tuple

        return figs, axes


# ---------------------------------------------------------------------------
# Viscosity
# ---------------------------------------------------------------------------


class Viscosity(TransportCoefficients):
    """Viscosity coefficients from the Green-Kubo formula.

    Reads from the :class:`~sarkas.tools.observables.PressureTensor` zarr store.
    Layout::

        total/acf_bulk    (block_length, no_slices)
        total/acf_tensor  (no_comp, no_comp, block_length, no_slices)
        coords/components (no_comp, 2)  int8

    Bulk viscosity uses ``total/acf_bulk[:, isl]``.
    Shear viscosity uses ``total/acf_tensor[ic, ic, :, isl]`` for each
    off-diagonal component index ``ic`` (i.e. where ``components[ic, 0] !=
    components[ic, 1]``).
    """

    def __init__(self):
        self.__name__ = "Viscosities"
        self.__long_name__ = "Viscosity"
        self.required_observable = "Pressure Tensor"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """Calculate bulk and shear viscosity.

        Parameters
        ----------
        observable : :class:`~sarkas.tools.observables.PressureTensor`
        """
        t0 = self.timer.current()

        ds_obs = xr.open_zarr(observable.zarr_store_path)
        # acf_bulk: (species, lag, slice), acf_tensor: (species, component_row, component_col, lag, slice)
        bulk_acf_da = ds_obs["acf_bulk"].sel(species="Total")
        tensor_acf_da = ds_obs["acf_tensor"].sel(species="Total")
        comp_labels = list(tensor_acf_da.coords["component_row"].values)
        # Off-diagonal component labels: those where row != col (e.g. "XY", "XZ", "YZ")
        off_diag_labels = [c for c in comp_labels if c[0] != c[-1]]

        quantities = {}

        # ----- Bulk viscosity -----
        sl_bulk = np.zeros((self.no_slices, self.block_length))
        for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
            const = observable.box_volume * self.beta_slices[isl]
            integrand = bulk_acf_da.isel(slice=isl).values
            sl_bulk[isl] = const * cumulative_trapezoid(integrand, x=self.time_array, initial=0.0)
        quantities["Bulk_Viscosity"] = sl_bulk

        # ----- Shear viscosity elements -----
        sl_shear_list = []
        for comp_label in off_diag_labels:
            sl = np.zeros((self.no_slices, self.block_length))
            for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
                const = observable.box_volume * self.beta_slices[isl]
                integrand = tensor_acf_da.sel(component_row=comp_label, component_col=comp_label).isel(slice=isl).values
                sl[isl] = const * cumulative_trapezoid(integrand, x=self.time_array, initial=0.0)
            quantities[f"Shear_Viscosity_{comp_label}"] = sl
            sl_shear_list.append(sl)

        if sl_shear_list:
            quantities["Shear_Viscosity"] = np.mean(sl_shear_list, axis=0)

        tend = self.timer.current()
        self.time_stamp("Viscosities Calculation", self.timer.time_division(tend - t0))
        self.save_zarr(quantities)

        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        ds = xr.open_zarr(self.zarr_store_path)
        ds_obs = xr.open_zarr(observable.zarr_store_path)
        comp_labels = list(ds_obs["acf_tensor"].coords["component_row"].values)
        off_diag_labels = [c for c in comp_labels if c[0] != c[-1]]

        figs = []
        axes_list = []

        # Bulk viscosity
        bulk_acf_mean = ds_obs["mean_acf_bulk"].sel(species="Total").values
        bulk_acf_std  = ds_obs["std_acf_bulk"].sel(species="Total").values
        tc_mean = ds["mean_Bulk_Viscosity"].values
        tc_std  = ds["std_Bulk_Viscosity"].values
        fig, ax_tuple = self.plot_tc(
            time=self.time_array,
            acf_data=column_stack((bulk_acf_mean, bulk_acf_std)),
            tc_data=column_stack((tc_mean, tc_std)),
            acf_name="Pressure Bulk ACF",
            tc_name="Bulk Viscosity",
            figname="Bulk_Viscosity_Plot.png",
            show=display_plot,
        )
        figs.append(fig)
        axes_list.append(ax_tuple)

        # Shear viscosity
        if "mean_Shear_Viscosity" in ds:
            acf_mean = np.mean([
                ds_obs["mean_acf_tensor"].sel(species="Total", component_row=c, component_col=c).values
                for c in off_diag_labels
            ], axis=0)
            acf_std = np.mean([
                ds_obs["std_acf_tensor"].sel(species="Total", component_row=c, component_col=c).values
                for c in off_diag_labels
            ], axis=0)
            tc_mean = ds["mean_Shear_Viscosity"].values
            tc_std  = ds["std_Shear_Viscosity"].values
            fig, ax_tuple = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_mean, acf_std)),
                tc_data=column_stack((tc_mean, tc_std)),
                acf_name="Shear Stress ACF",
                tc_name="Shear Viscosity",
                figname="Shear_Viscosity_Plot.png",
                show=display_plot,
            )
            figs.append(fig)
            axes_list.append(ax_tuple)

        return figs, axes_list


# ---------------------------------------------------------------------------
# ElectricalConductivity
# ---------------------------------------------------------------------------


class ElectricalConductivity(TransportCoefficients):
    """Electrical conductivity from the Green-Kubo formula.

    Reads from the :class:`~sarkas.tools.observables.ElectricCurrent` zarr store.
    Layout::

        acf  (no_sp+1, no_sp+1, D, block_length, no_slices)

    Access: ``acf[-1, -1, :, :, isl].sum(axis=0)`` — total current
    auto-correlation summed over spatial dimensions.
    """

    def __init__(self):
        self.__name__ = "ElectricalConductivity"
        self.__long_name__ = "Electrical Conductivity"
        self.required_observable = "Electric Current"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """Calculate the electrical conductivity.

        Parameters
        ----------
        observable : :class:`~sarkas.tools.observables.ElectricCurrent`
        """
        t0 = self.timer.current()

        ds_obs = xr.open_zarr(observable.zarr_store_path)
        acf_da = ds_obs["acf"]  # (species_row, species_col, component, lag, slice)
        D = observable.dimensions
        dim_labels = list(observable.dim_labels)

        quantities = {}

        if not observable.magnetized:
            sl = np.zeros((self.no_slices, self.block_length))
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                const = self.beta_slices[isl] / observable.box_volume
                # Total-total ACF averaged over spatial dimensions
                integrand = acf_da.sel(species_row="Total", species_col="Total").isel(slice=isl).mean("component").values
                sl[isl] = const * cumulative_trapezoid(integrand, self.time_array, initial=0.0)
            quantities["ElectricalConductivity"] = sl
        else:
            sl_par  = np.zeros((self.no_slices, self.block_length))
            sl_perp = np.zeros((self.no_slices, self.block_length))
            for isl in tqdm(range(observable.no_slices), disable=not observable.verbose):
                const = self.beta_slices[isl] / observable.box_volume
                integrand_par = acf_da.sel(species_row="Total", species_col="Total", component=dim_labels[D - 1]).isel(slice=isl).values
                sl_par[isl] = const * cumulative_trapezoid(integrand_par, self.time_array, initial=0.0)
                integrand_perp = 0.5 * (
                    acf_da.sel(species_row="Total", species_col="Total", component=dim_labels[0]).isel(slice=isl).values
                    + acf_da.sel(species_row="Total", species_col="Total", component=dim_labels[1]).isel(slice=isl).values
                )
                sl_perp[isl] = const * cumulative_trapezoid(integrand_perp, self.time_array, initial=0.0)
            quantities["ElectricalConductivity_Parallel"]     = sl_par
            quantities["ElectricalConductivity_Perpendicular"] = sl_perp

        tend = self.timer.current()
        self.time_stamp(f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0))
        self.save_zarr(quantities)

        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        ds = xr.open_zarr(self.zarr_store_path)
        ds_obs = xr.open_zarr(observable.zarr_store_path)
        D = observable.dimensions
        dim_labels = list(observable.dim_labels)

        figs = []
        axes_list = []

        if not observable.magnetized:
            acf_mean = ds_obs["mean_acf"].sel(species_row="Total", species_col="Total").mean("component").values
            acf_std  = ds_obs["std_acf"].sel(species_row="Total", species_col="Total").mean("component").values
            tc_mean  = ds["mean_ElectricalConductivity"].values
            tc_std   = ds["std_ElectricalConductivity"].values
            fig, ax_tuple = self.plot_tc(
                time=self.time_array,
                acf_data=column_stack((acf_mean, acf_std)),
                tc_data=column_stack((tc_mean, tc_std)),
                acf_name="Electric Current ACF",
                tc_name="Electrical Conductivity",
                figname="ElectricalConductivity_Plot.png",
                show=display_plot,
            )
            figs.append(fig)
            axes_list.append(ax_tuple)
        else:
            for comp, dim_lbl, key in (
                ("Parallel",     dim_labels[D - 1], "ElectricalConductivity_Parallel"),
                ("Perpendicular", dim_labels[0],    "ElectricalConductivity_Perpendicular"),
            ):
                acf_mean = ds_obs["mean_acf"].sel(species_row="Total", species_col="Total", component=dim_lbl).values
                acf_std  = ds_obs["std_acf"].sel(species_row="Total", species_col="Total", component=dim_lbl).values
                tc_mean  = ds[f"mean_{key}"].values
                tc_std   = ds[f"std_{key}"].values
                fig, ax_tuple = self.plot_tc(
                    time=self.time_array,
                    acf_data=column_stack((acf_mean, acf_std)),
                    tc_data=column_stack((tc_mean, tc_std)),
                    acf_name=f"Electric Current ACF {comp}",
                    tc_name=f"Electrical Conductivity {comp}",
                    figname=f"ElectricalConductivity_{comp}_Plot.png",
                    show=display_plot,
                )
                figs.append(fig)
                axes_list.append(ax_tuple)

        return figs, axes_list


# ---------------------------------------------------------------------------
# ThermalConductivity
# ---------------------------------------------------------------------------


class ThermalConductivity(TransportCoefficients):
    """Thermal conductivity from the Green-Kubo formula.

    Reads from the :class:`~sarkas.tools.observables.HeatFlux` zarr store.
    Layout::

        acf  (no_species, no_species, D+1, block_length, no_slices)

    For single-species: ``acf[0, 0, -1, :, isl]``.
    For multi-species: sum over all ``(i, j)`` cross-correlations.
    """

    def __init__(self):
        self.__name__ = "ThermalConductivity"
        self.__long_name__ = "Thermal Conductivity"
        self.required_observable = "Heat Flux"
        super().__init__()

    def compute(self, observable, plot: bool = True, display_plot: bool = False):
        """Calculate the thermal conductivity.

        Parameters
        ----------
        observable : :class:`~sarkas.tools.observables.HeatFlux`
        """
        t0 = self.timer.current()

        ds_obs = xr.open_zarr(observable.zarr_store_path)
        acf_da = ds_obs["acf"]  # (species_row, species_col, component, lag, slice)

        sl = np.zeros((self.no_slices, self.block_length))
        for isl in tqdm(range(self.no_slices), disable=not observable.verbose):
            const = self.kB * self.beta_slices[isl] ** 2 / observable.box_volume
            # Total heat flux ACF = sum over species pairs, isotropic component ("Total")
            integrand = acf_da.sel(component="Total").isel(slice=isl).sum(["species_row", "species_col"]).values
            sl[isl] = const * cumulative_trapezoid(integrand, self.time_array, initial=0.0)

        tend = self.timer.current()
        self.time_stamp(f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0))
        self.save_zarr({"ThermalConductivity": sl})

        if plot:
            _, _ = self.plot(observable, display_plot=display_plot)

    def plot(self, observable, display_plot: bool = False):
        ds = xr.open_zarr(self.zarr_store_path)
        ds_obs = xr.open_zarr(observable.zarr_store_path)

        acf_mean = ds_obs["mean_acf"].sel(component="Total").sum(["species_row", "species_col"]).values
        acf_std  = ds_obs["std_acf"].sel(component="Total").sum(["species_row", "species_col"]).values
        tc_mean  = ds["mean_ThermalConductivity"].values
        tc_std   = ds["std_ThermalConductivity"].values

        fig, ax_tuple = self.plot_tc(
            time=self.time_array,
            acf_data=column_stack((acf_mean, acf_std)),
            tc_data=column_stack((tc_mean, tc_std)),
            acf_name="Heat Flux ACF",
            tc_name="Thermal Conductivity",
            figname=f"{self.__name__}_Plot.png",
            show=display_plot,
        )
        return fig, ax_tuple
