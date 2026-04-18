"""Transport observables: VACF, ElectricCurrent, HeatFlux, DiffusionFlux.

All I/O uses xarray + zarr (no pandas).  Every zarr store is xarray-compatible:
labels are real array dimensions with string coordinate values.

VACF store layout::
    acf          (species, component, lag, slice)
    mean_acf / std_acf   — no-slice versions

ElectricCurrent store layout::
    current      (species, component, time, slice)   — species includes "Total"
    acf          (species_row, species_col, component, lag, slice)
    mean_current / std_current / mean_acf / std_acf

HeatFlux store layout::
    flux         (species, component, time, slice)
    acf          (species_row, species_col, component, lag, slice)
    mean_flux / std_flux / mean_acf / std_acf

DiffusionFlux store layout::
    flux         (flux_index, component, time, slice)
    acf          (flux_row, flux_col, component, lag, slice)
    mean_flux / std_flux / mean_acf / std_acf
"""

import h5py
import numpy as np
import xarray as xr
from numpy import zeros, sort, ones_like, array
from numpy.random import default_rng
from tqdm import tqdm

from ...utilities.maths import correlationfunction
from ...utilities.timing import time_stamp
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


# ---------------------------------------------------------------------------
# VelocityAutoCorrelationFunction
# ---------------------------------------------------------------------------


class VelocityAutoCorrelationFunction(Observable):
    """Velocity Auto-correlation function.

    Reads per-particle velocities from the h5md dump file, then computes the
    VACF averaged over a randomly-selected subset of particles and over
    independent time slices.
    """

    def __init__(self):
        super(VelocityAutoCorrelationFunction, self).__init__()
        self.__name__ = "vacf"
        self.__long_name__ = "Velocity AutoCorrelation Function"
        self.no_ptcls_per_species = [10]
        self.particles_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.acf_observable = True

    @setup_doc
    def setup(
        self,
        params,
        phase: str = None,
        no_ptcls_per_species: list = None,
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

        if no_ptcls_per_species:
            self.select_random_indices(no_ptcls_per_species)

        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        self.__dict__.update(kwargs.copy())
        self.update_finish()

        if "no_ptcls_per_species" in kwargs.keys():
            self.select_random_indices(kwargs["no_ptcls_per_species"])
        else:
            self.select_random_indices()

    @compute_doc
    def compute(self, calculate_acf_data: bool = True, no_ptcls_per_species=None):
        raise DeprecationWarning("VACF does not have a `compute` method anymore. Use `compute_acf()`.")

    @compute_acf_doc
    def compute_acf(self):
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_acf_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, f"{self.__long_name__} Calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_slices_doc
    def calc_slices_data(self):
        from ._kernels import acf_batch

        D = self.dimensions
        comp_labels = [*self.dim_labels, "Total"]
        no_sp = len(self.species_names)

        sp_offsets = [0]
        for n in self.no_ptcls_per_species:
            sp_offsets.append(sp_offsets[-1] + int(n))

        acf_arr = np.zeros((no_sp, D + 1, self.block_length, self.no_slices))

        # Read slice-at-a-time from h5md to avoid loading entire trajectory
        with h5py.File(self.h5md_filepath, "r") as h5:
            vel_ds = h5["particles/vel"]
            time_all = h5["particles/time"][:]

            for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__}", disable=not self.verbose):
                start = isl * self.dumps_shift
                end = start + self.block_length
                vel_slice = vel_ds[start:end, self.particles_id, :]  # (T, N_selected, D)
                T = vel_slice.shape[0]

                for isp in range(no_sp):
                    s0 = sp_offsets[isp]
                    s1 = sp_offsets[isp + 1]
                    vel_sp = vel_slice[:, s0:s1, :]  # (T, N_sp, D)
                    N_sp = s1 - s0
                    batch = vel_sp.transpose(1, 2, 0).reshape(N_sp * D, T)
                    acf_all = acf_batch(batch).reshape(N_sp, D, T)
                    acf_per_dim = acf_all.mean(axis=0)    # (D, T)
                    acf_total = acf_per_dim.mean(axis=0)  # (T,)
                    acf_arr[isp, :D, :, isl] = acf_per_dim
                    acf_arr[isp,  D, :, isl] = acf_total

        acf = xr.DataArray(
            acf_arr,
            dims=["species", "component", "lag", "slice"],
            coords={
                "species": list(self.species_names),
                "component": comp_labels,
                "lag": time_all[:self.block_length],
                "slice": np.arange(self.no_slices),
            },
        )
        self._write_zarr_dataset(
            {"acf": acf},
            mode="w",
            attrs={
                "no_slices": self.no_slices,
                "block_length": self.block_length,
                "dumps_shift": self.dumps_shift,
                "h5md_path": self.h5md_filepath,
            },
        )

    @avg_acf_slices_doc
    def average_acf_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")

    def grab_sim_data(self, start_dump_no, end_dump_no, vel, time):
        """Grab velocities from simulation dump files (legacy helper).

        Parameters
        ----------
        start_dump_no : int
        end_dump_no : int
        vel : numpy.ndarray
            Shape ``(dimensions, n_selected_particles, 2 * block_length)``.
        time : numpy.ndarray
            Shape ``(2 * block_length,)``.
        """
        for it, dump in enumerate(
            tqdm(
                range(start_dump_no, end_dump_no, self.dump_step),
                desc="Reading data",
                disable=not self.verbose,
                position=1,
                leave=False,
            )
        ):
            from ...utilities.io import load_from_restart

            datap = load_from_restart(self.dump_dir, dump)
            time[it] = datap["time"]
            for d in range(self.dimensions):
                vel[d, :, it] = datap["vel"][self.particles_id, d]

    def select_random_indices(self, no_ptcls_per_species=None):
        """Randomly select particle indices for VACF averaging.

        Parameters
        ----------
        no_ptcls_per_species : list or int, optional
            Number of particles to select per species.  Defaults to
            ``self.no_ptcls_per_species`` (``[10]`` on construction).

        Raises
        ------
        ValueError
            If the requested number exceeds the available particles.
        """
        if no_ptcls_per_species:
            if isinstance(no_ptcls_per_species, int):
                self.no_ptcls_per_species = [no_ptcls_per_species]

        rng = default_rng()

        if len(self.no_ptcls_per_species) != len(self.species_num):
            self.no_ptcls_per_species = min(self.no_ptcls_per_species) * ones_like(self.species_num)

        combined_random_indices = []
        species_start = 0
        species_end = 0
        for ip, num_ptcls in enumerate(self.no_ptcls_per_species):
            species_end += self.species_num[ip]
            if num_ptcls > self.species_num[ip]:
                raise ValueError(
                    f"Species {self.species_names[ip]}: the chosen random number of particles, "
                    f"{num_ptcls}, is less than its total species number of particles, "
                    f"{self.species_num[ip]}"
                )
            random_indices = sort(rng.choice(range(species_start, species_end), size=num_ptcls, replace=False))
            combined_random_indices.extend(random_indices)
            species_start += self.species_num[ip]

        self.particles_id = combined_random_indices

    def calculate_vacf(self, vel):
        """Calculate VACF from a velocity array (legacy helper).

        Parameters
        ----------
        vel : numpy.ndarray
            Shape ``(D, Np, Nt)``.

        Returns
        -------
        vacf : numpy.ndarray
            Shape ``(num_species, D + 1, block_length)``.
        """
        no_dim = vel.shape[0]
        vacf = zeros((self.num_species, no_dim + 1, self.block_length))
        species_vacf = zeros(self.block_length)
        ptcl_vacf = zeros(self.block_length)

        for d in tqdm(range(no_dim), desc="Dimension", position=1, disable=not self.verbose, leave=False):
            species_start = 0
            species_end = 0
            for sp, np_sp in enumerate(
                tqdm(self.no_ptcls_per_species, desc="Species", position=2, disable=not self.verbose, leave=False)
            ):
                species_end += np_sp
                for ptcl in range(species_start, species_end):
                    for it in range(self.block_length):
                        v = vel[d, ptcl, : self.block_length + it]
                        ptcl_vacf[it] = correlationfunction(v, v)[it]
                    species_vacf += ptcl_vacf

                vacf[sp, d, :] = species_vacf / np_sp
                vacf[sp, -1, :] += species_vacf / np_sp
                species_start += np_sp

        return vacf


# ---------------------------------------------------------------------------
# ElectricCurrent
# ---------------------------------------------------------------------------


class ElectricCurrent(Observable):
    """Electric Current and its Auto-correlation function."""

    def __init__(self):
        super().__init__()
        self.__name__ = "ec"
        self.__long_name__ = "Electric Current"
        self.__hdf_key__ = "electric_current"
        self.acf_observable = True

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
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    @compute_doc
    def compute(self, calculate_acf: bool = False):
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " calculation", self.timer.time_division(tend - t0), self.verbose)
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self):
        t0 = self.timer.current()
        self.calc_acf_slices_data()
        self.average_acf_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " ACF calculation", self.timer.time_division(tend - t0), self.verbose)

    def calculate_observable_from_dumps(self):
        """Calculate species electric current from particle velocities and write to h5md."""
        if not __import__("os.path", fromlist=["exists"]).exists(self.h5md_filepath):
            raise FileNotFoundError(f"The H5MD file {self.h5md_filepath} does not exist.")

        with h5py.File(self.h5md_filepath, "a") as h5file:
            num_dumps = h5file["particles/vel"].shape[0]
            time_ = h5file["particles/time"][:]
            step_ = h5file["particles/step"][:]

            sp_start = 0
            sp_end = self.species_num[0]
            for isp, sp in enumerate(self.species_names):
                group_name = f"observables/{sp}/{self.__hdf_key__}"

                if group_name not in h5file:
                    obs_group = h5file.create_group(group_name)
                    maxshape = (None, self.dimensions)
                    dtype = "f8"
                    obs_group.create_dataset(
                        "value",
                        shape=(num_dumps, self.dimensions),
                        maxshape=maxshape,
                        chunks=True,
                        dtype=dtype,
                    )
                    obs_group.create_dataset(
                        "time",
                        shape=(num_dumps,),
                        maxshape=(None,),
                        chunks=True,
                        dtype=dtype,
                    )
                    obs_group.create_dataset(
                        "step",
                        shape=(num_dumps,),
                        maxshape=(None,),
                        chunks=True,
                        dtype="i8",
                    )
                else:
                    obs_group = h5file[group_name]
                    if obs_group["value"].shape[0] < num_dumps:
                        obs_group["value"].resize((num_dumps, self.num_species, self.dimensions))
                        obs_group["time"].resize((num_dumps,))
                        obs_group["step"].resize((num_dumps,))

                vel = h5file["particles/vel"][:, sp_start:sp_end, :].sum(axis=1)
                current = self.species_charges[isp] * vel
                obs_group = h5file[group_name]
                obs_group["value"][:, :] = current
                obs_group["time"][:] = time_
                obs_group["step"][:] = step_
                sp_start = sp_end
                sp_end += self.species_num[isp + 1] if isp + 1 < len(self.species_num) else 0

    @calc_slices_doc
    def calc_slices_data(self):
        D = self.dimensions
        no_sp = len(self.species_names)
        species_all = [*list(self.species_names), "Total"]
        comp_labels = list(self.dim_labels)

        with h5py.File(self.h5md_filepath, "r") as h5:
            first_sp = self.species_names[0]
            no_dumps = h5[f"observables/{first_sp}/electric_current/value"].shape[0]
            time = h5[f"observables/{first_sp}/electric_current/time"][:]
            raw = np.zeros((no_dumps, no_sp, D))
            for isp, sp in enumerate(self.species_names):
                raw[:, isp, :] = h5[f"observables/{sp}/electric_current/value"][:]

        # shape (no_sp+1, D, block_length, no_slices)
        curr_arr = np.zeros((no_sp + 1, D, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__}", disable=not self.verbose):
            start = isl * self.dumps_shift
            end = start + self.block_length
            for isp in range(no_sp):
                curr_arr[isp, :, :, isl] = raw[start:end, isp, :].T
            curr_arr[no_sp, :, :, isl] = raw[start:end, :, :].sum(axis=1).T

        current = xr.DataArray(
            curr_arr,
            dims=["species", "component", "time", "slice"],
            coords={
                "species": species_all,
                "component": comp_labels,
                "time": time[:self.block_length],
                "slice": np.arange(self.no_slices),
            },
        )
        self._write_zarr_dataset(
            {"current": current},
            mode="w",
            attrs={
                "no_slices": self.no_slices,
                "block_length": self.block_length,
                "dumps_shift": self.dumps_shift,
                "h5md_path": self.h5md_filepath,
            },
        )

    @avg_slices_doc
    def average_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")

    @calc_acf_slices_doc
    def calc_acf_slices_data(self):
        D = self.dimensions
        no_sp = len(self.species_names)
        n_entries = no_sp + 1
        species_all = [*list(self.species_names), "Total"]
        comp_labels = list(self.dim_labels)

        ds = xr.open_zarr(self.zarr_store_path)
        curr_arr = ds["current"].values  # (n_entries, D, block_length, no_slices)
        time_vals = ds["current"].coords["time"].values
        slice_vals = ds["current"].coords["slice"].values

        acf_arr = np.zeros((n_entries, n_entries, D, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__} ACF", disable=not self.verbose):
            for i in range(n_entries):
                ci = curr_arr[i, :, :, isl]  # (D, block_length)
                for j in range(i, n_entries):
                    cj = curr_arr[j, :, :, isl]
                    for d in range(D):
                        sig_i = ci[d] - ci[d].mean()
                        sig_j = cj[d] - cj[d].mean()
                        acf_val = correlationfunction(sig_i, sig_j)
                        acf_arr[i, j, d, :, isl] = acf_val
                        if i != j:
                            acf_arr[j, i, d, :, isl] = acf_val

        acf = xr.DataArray(
            acf_arr,
            dims=["species_row", "species_col", "component", "lag", "slice"],
            coords={
                "species_row": species_all,
                "species_col": species_all,
                "component": comp_labels,
                "lag": time_vals,
                "slice": slice_vals,
            },
        )
        self._write_zarr_dataset({"acf": acf}, mode="a")

    @avg_acf_slices_doc
    def average_acf_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")


# ---------------------------------------------------------------------------
# HeatFlux
# ---------------------------------------------------------------------------


class HeatFlux(Observable):
    """Heat Flux and its Auto-correlation function."""

    def __init__(self):
        super().__init__()
        self.__name__ = "heat_flux_species_tensor"
        self.__long_name__ = "Heat Flux"
        self.acf_observable = True

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
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        self.__dict__.update(kwargs.copy())
        self.update_finish()

    @compute_doc
    def compute(self, calculate_acf: bool = False):
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self.calc_slices_data()
        self.average_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " calculation", self.timer.time_division(tend - t0), self.verbose)
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self):
        t0 = self.timer.current()
        self.calc_acf_slices_data()
        self.average_acf_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " ACF calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_slices_doc
    def calc_slices_data(self):
        D = self.dimensions
        no_sp = len(self.species_names)
        comp_labels = [*list(self.dim_labels), "Total"]

        with h5py.File(self.h5md_filepath, "r") as h5:
            first_sp = self.species_names[0]
            no_dumps = h5[f"observables/{first_sp}/heat_flux/value"].shape[0]
            time = h5[f"observables/{first_sp}/heat_flux/time"][:]
            raw = np.zeros((no_dumps, no_sp, D))
            for isp, sp in enumerate(self.species_names):
                raw[:, isp, :] = h5[f"observables/{sp}/heat_flux/value"][:]

        flux_arr = np.zeros((no_sp, D + 1, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__}", disable=not self.verbose):
            start = isl * self.dumps_shift
            end = start + self.block_length
            for isp in range(no_sp):
                flux_arr[isp, :D, :, isl] = raw[start:end, isp, :].T
                flux_arr[isp,  D, :, isl] = raw[start:end, isp, :].mean(axis=1)

        flux = xr.DataArray(
            flux_arr,
            dims=["species", "component", "time", "slice"],
            coords={
                "species": list(self.species_names),
                "component": comp_labels,
                "time": time[:self.block_length],
                "slice": np.arange(self.no_slices),
            },
        )
        self._write_zarr_dataset(
            {"flux": flux},
            mode="w",
            attrs={
                "no_slices": self.no_slices,
                "block_length": self.block_length,
                "dumps_shift": self.dumps_shift,
                "h5md_path": self.h5md_filepath,
            },
        )

    @avg_slices_doc
    def average_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")

    @calc_acf_slices_doc
    def calc_acf_slices_data(self):
        D = self.dimensions
        no_sp = len(self.species_names)
        comp_labels = [*self.dim_labels, "Total"]
        sp_labels = list(self.species_names)

        ds = xr.open_zarr(self.zarr_store_path)
        flux_arr = ds["flux"].values  # (no_sp, D, block_length, no_slices)
        time_vals = ds["flux"].coords["time"].values
        slice_vals = ds["flux"].coords["slice"].values

        acf_arr = np.zeros((no_sp, no_sp, D + 1, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__} ACF", disable=not self.verbose):
            for i in range(no_sp):
                fi = flux_arr[i, :, :, isl]  # (D, block_length)
                for j in range(i, no_sp):
                    fj = flux_arr[j, :, :, isl]
                    acf_dims = zeros((D, self.block_length))
                    for d in range(D):
                        sig_i = fi[d] - fi[d].mean()
                        sig_j = fj[d] - fj[d].mean()
                        acf_dims[d] = correlationfunction(sig_i, sig_j)
                    acf_arr[i, j, :D, :, isl] = acf_dims
                    acf_arr[i, j,  D, :, isl] = acf_dims.mean(axis=0)
                    if i != j:
                        acf_arr[j, i, :D, :, isl] = acf_dims
                        acf_arr[j, i,  D, :, isl] = acf_dims.mean(axis=0)

        acf = xr.DataArray(
            acf_arr,
            dims=["species_row", "species_col", "component", "lag", "slice"],
            coords={
                "species_row": sp_labels,
                "species_col": sp_labels,
                "component": comp_labels,
                "lag": time_vals,
                "slice": slice_vals,
            },
        )
        self._write_zarr_dataset({"acf": acf}, mode="a")

    @avg_acf_slices_doc
    def average_acf_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")


# ---------------------------------------------------------------------------
# DiffusionFlux
# ---------------------------------------------------------------------------


class DiffusionFlux(Observable):
    """Diffusion Fluxes and their Auto-correlation functions.

    The :math:`\\alpha` diffusion flux :math:`\\mathbf J_{\\alpha}(t)` is
    calculated from eq. (3.5) in :cite:`Zhou1996`.
    """

    def __init__(self):
        super().__init__()
        self.__name__ = "diff_flux"
        self.__long_name__ = "Diffusion Flux"
        self.acf_observable = True

    @setup_doc
    def setup(
        self,
        params,
        phase: str = None,
        plasma_periods_per_slice: int = None,
        plasma_periods_shift: int = None,
        **kwargs,
    ):
        super().setup_init(
            params,
            phase,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift,
            **kwargs,
        )
        self.update_args(**kwargs)

    @arg_update_doc
    def update_args(self, **kwargs):
        self.__dict__.update(kwargs.copy())
        self.no_fluxes = self.num_species - 1
        self.no_fluxes_acf = int(self.no_fluxes * self.no_fluxes)
        self.update_finish()

    @compute_doc
    def compute(self, from_pva :bool = False, calculate_acf: bool = False):
        if self._store_is_valid():
            return
        self._invalidate_store()
        t0 = self.timer.current()
        self.calc_slices_data(from_pva=from_pva)
        self.average_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " calculation", self.timer.time_division(tend - t0), self.verbose)
        if calculate_acf:
            self.compute_acf()

    @compute_acf_doc
    def compute_acf(self):
        t0 = self.timer.current()
        self.calc_acf_slices_data()
        self.average_acf_slices_data()
        tend = self.timer.current()
        time_stamp(self.log_file, self.__long_name__ + " ACF calculation", self.timer.time_division(tend - t0), self.verbose)

    @calc_slices_doc
    def calc_slices_data(self, from_pva: bool = False):
        D = self.dimensions
        no_fluxes = self.no_fluxes  # = no_species - 1
        comp_labels = [*list(self.dim_labels), "Total"]
        flux_labels = [self.species_names[i] for i in range(no_fluxes)]

        if from_pva:
            from ...particles import calc_diffusion_fluxes

            with h5py.File(self.h5md_filepath, "r") as h5:
                first_sp = self.species_names[0]
                no_dumps = h5[f"particles/vel"].shape[0]
                time = h5[f"particles/time"][:]
                raw = np.zeros((no_dumps, no_fluxes, D))

                for i in range(no_dumps):
                    vel = h5[f"particles/vel"][i, :, :]  # (N, D)
                    raw[i] = calc_diffusion_fluxes(vel, self.species_masses, self.species_num)

        else:
            with h5py.File(self.h5md_filepath, "r") as h5:
                time = h5[f"observables/diffusion_fluxes/time"][:]
                no_dumps = len(time)
                raw = np.zeros((no_dumps, no_fluxes, D))
                raw[:, :, :] = h5[f"observables/diffusion_fluxes/value"][:]

        flux_arr = np.zeros((no_fluxes, D + 1, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__}", disable=not self.verbose):
            start = isl * self.dumps_shift
            end = start + self.block_length
            for i in range(no_fluxes):
                flux_arr[i, :D, :, isl] = raw[start:end, i, :].T
                flux_arr[i,  D, :, isl] = raw[start:end, i, :].mean(axis=1)

        flux = xr.DataArray(
            flux_arr,
            dims=["flux_index", "component", "time", "slice"],
            coords={
                "flux_index": flux_labels,
                "component": comp_labels,
                "time": time[:self.block_length],
                "slice": np.arange(self.no_slices),
            },
        )
        self._write_zarr_dataset(
            {"flux": flux},
            mode="w",
            attrs={
                "no_slices": self.no_slices,
                "block_length": self.block_length,
                "dumps_shift": self.dumps_shift,
                "h5md_path": self.h5md_filepath,
            },
        )

    @avg_slices_doc
    def average_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")

    @calc_acf_slices_doc
    def calc_acf_slices_data(self):
        D = self.dimensions
        no_fluxes = self.no_fluxes
        comp_labels = [*self.dim_labels, "Total"]
        flux_labels = [self.species_names[i] for i in range(no_fluxes)]

        ds = xr.open_zarr(self.zarr_store_path)
        flux_arr = ds["flux"].values  # (no_fluxes, D, block_length, no_slices)
        time_vals = ds["flux"].coords["time"].values
        slice_vals = ds["flux"].coords["slice"].values

        acf_arr = np.zeros((no_fluxes, no_fluxes, D + 1, self.block_length, self.no_slices))
        for isl in tqdm(range(self.no_slices), desc=f"Calculating {self.__long_name__} ACF", disable=not self.verbose):
            for i in range(no_fluxes):
                fi = flux_arr[i, :, :, isl]
                for j in range(i, no_fluxes):
                    fj = flux_arr[j, :, :, isl]
                    acf_dims = zeros((D, self.block_length))
                    for d in range(D):
                        sig_i = fi[d] - fi[d].mean()
                        sig_j = fj[d] - fj[d].mean()
                        acf_dims[d] = correlationfunction(sig_i, sig_j)
                    acf_arr[i, j, :D, :, isl] = acf_dims
                    acf_arr[i, j,  D, :, isl] = acf_dims.mean(axis=0)
                    if i != j:
                        acf_arr[j, i, :D, :, isl] = acf_dims
                        acf_arr[j, i,  D, :, isl] = acf_dims.mean(axis=0)

        acf = xr.DataArray(
            acf_arr,
            dims=["flux_row", "flux_col", "component", "lag", "slice"],
            coords={
                "flux_row": flux_labels,
                "flux_col": flux_labels,
                "component": comp_labels,
                "lag": time_vals,
                "slice": slice_vals,
            },
        )
        self._write_zarr_dataset({"acf": acf}, mode="a")

    @avg_acf_slices_doc
    def average_acf_slices_data(self):
        ds = xr.open_zarr(self.zarr_store_path)
        ddof = min(1, self.no_slices - 1)
        to_write = {}
        for var in ds.data_vars:
            if "slice" in ds[var].dims:
                to_write[f"mean_{var}"] = ds[var].mean("slice").compute()
                to_write[f"std_{var}"] = ds[var].std("slice", ddof=ddof).compute()
        self._write_zarr_dataset(to_write, mode="a")
