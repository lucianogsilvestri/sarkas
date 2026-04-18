"""
Velocity distribution observable for sarkas post-processing.

All I/O uses zarr (no pandas).
"""

import h5py
import numpy as np
from numpy import zeros, array, histogram, trapz, sqrt, pi, exp
from numpy.polynomial import hermite_e
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import shapiro
from tqdm import tqdm

from .base import Observable, setup_doc, arg_update_doc
from ...utilities.timing import time_stamp
from ..fit_functions import exponential, gaussian


# ---------------------------------------------------------------------------
# Module-level helpers (mirrored from observables.py)
# ---------------------------------------------------------------------------


def calc_moments(dist, max_moment, species_index_start):
    """Calculate the moments of the (velocity) distribution.

    Parameters
    ----------
    dist : numpy.ndarray
        Distribution of each time step.
        Shape = (``no_dumps``, ``dim``, ``runs * inv_dim * total_num_ptcls``)

    max_moment : int
        Maximum moment to calculate.

    species_index_start : numpy.ndarray
        Array containing the start index of each species.
        The last value is equivalent to ``dist.shape[-1]``.

    Returns
    -------
    moments : numpy.ndarray
        Moments of the distribution.
        Shape = (``no_species``, ``no_dumps``, ``dim``, ``max_moment``)

    ratios : numpy.ndarray
        Ratios of each moment with respect to the expected Maxwellian value.
        Shape = (``no_species``, ``no_dumps``, ``no_dim``, ``max_moment - 1``)

    Notes
    -----
    See `Wikipedia – Normal distribution moments
    <https://en.wikipedia.org/wiki/Normal_distribution#Moments>`_.
    """
    from scipy.stats import moment as scp_moment
    from scipy.special import gamma as scp_gamma

    no_species = len(species_index_start) - 1
    no_dumps = dist.shape[0]
    dim = dist.shape[1]
    moments = zeros((no_species, no_dumps, dim, max_moment))
    ratios = zeros((no_species, no_dumps, dim, max_moment))

    for indx, sp_start in enumerate(species_index_start[:-1]):
        sp_end = species_index_start[indx + 1]
        for mom in range(max_moment):
            moments[indx, :, :, mom] = scp_moment(dist[:, :, sp_start:sp_end], moment=mom + 1, axis=-1)

    # sqrt( <v^2> ) = standard deviation = moments[:, :, :, 1] ** (1/2)
    for mom in range(max_moment):
        pwr = mom + 1
        const = 2.0 ** (pwr / 2) * scp_gamma((pwr + 1) / 2) / sqrt(pi)
        ratios[:, :, :, mom] = moments[:, :, :, mom] / (const * moments[:, :, :, 1] ** (pwr / 2.0))

    return moments, ratios


def grad_expansion(x, rms, h_coeff):
    """Calculate the Grad expansion as given by eq.(5.97) in Liboff.

    Parameters
    ----------
    x : numpy.ndarray
        Array of the scaled velocities.

    rms : float
        RMS width of the Gaussian.

    h_coeff : numpy.ndarray
        Hermite coefficients without the division by factorial.

    Returns
    -------
    numpy.ndarray
        Grad expansion evaluated at *x*.
    """
    gauss = exp(-0.5 * (x / rms) ** 2) / (sqrt(2.0 * pi * rms ** 2))
    herm_coef = h_coeff / array([factorial(i) for i in range(len(h_coeff))])
    hermite_series = hermite_e.hermeval(x, herm_coef)
    return gauss * hermite_series


def calculate_herm_coeff(v, distribution, maxpower):
    r"""Calculate Hermite coefficients by integrating the velocity distribution.

    .. math::
        a_i = \int_{-\infty}^{\infty} dv \, He_i(v) f(v)

    Parameters
    ----------
    v : numpy.ndarray
        Range of velocities.

    distribution : numpy.ndarray
        Velocity histogram.

    maxpower : int
        Hermite order.

    Returns
    -------
    coeff : numpy.ndarray
        Coefficients :math:`a_i`.
    """
    coeff = zeros(maxpower + 1)
    for i in range(maxpower + 1):
        hc = zeros(1 + i)
        hc[-1] = 1.0
        Hp = hermite_e.hermeval(v, hc)
        coeff[i] = trapz(distribution * Hp, x=v)
    return coeff


# ---------------------------------------------------------------------------
# VelocityDistribution
# ---------------------------------------------------------------------------


class VelocityDistribution(Observable):
    """Moments of the velocity distributions defined as

    .. math::
        \\langle v^{\\alpha} \\rangle = \\int_{-\\infty}^{\\infty} d v \\, f(v) v^{2 \\alpha}.

    Attributes
    ----------
    no_bins : int
        Number of bins used to calculate the velocity distribution.

    plots_dir : str
        Directory in which to store Hermite coefficients plots.

    species_plots_dirs : list of str
        Directory for each species where to save Hermite coefficients plots.

    max_no_moment : int
        Maximum number of moments = :math:`\\alpha`. Default = 6.
    """

    def __init__(self):
        super(VelocityDistribution, self).__init__()
        self.max_no_moment = None
        self.__name__ = "vd"
        self.__long_name__ = "Velocity Distribution"

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
        hist_kwargs: dict = None,
        max_no_moment: int = None,
        multi_run_average: bool = None,
        dimensional_average: bool = None,
        runs: int = 1,
        curve_fit_kwargs: dict = None,
        **kwargs,
    ):
        """Assign attributes from simulation's parameters.

        Parameters
        ----------
        params : sarkas.core.Parameters
            Simulation's parameters.

        phase : str, optional
            Phase to compute. Default = ``'production'``.

        independent_slices : bool, optional
            Flag for independent time slices.

        no_slices : int, optional
            Number of time slices.

        timesteps_per_slice : int, optional
            Number of timesteps per slice.

        timesteps_shift : int, optional
            Shift in timesteps between slices.

        plasma_periods_per_slice : int, optional
            Number of plasma periods per slice.

        plasma_periods_shift : int, optional
            Shift in plasma periods between slices.

        hist_kwargs : dict, optional
            Dictionary of keyword arguments to pass to ``numpy.histogram``
            for the calculation of the distributions.

        max_no_moment : int, optional
            Maximum number of moments to calculate. Default = 6.

        multi_run_average : bool, optional
            Flag for averaging over multiple runs.

        dimensional_average : bool, optional
            Flag for averaging over dimensions.

        runs : int, optional
            Number of runs. Default = 1.

        curve_fit_kwargs : dict, optional
            Dictionary of keyword arguments to pass to
            ``scipy.optimize.curve_fit`` for fitting of Hermite coefficients.

        **kwargs
            These will overwrite any :class:`sarkas.core.Parameters`
            or default :class:`sarkas.tools.observables.Observable`
            attributes and/or add new ones.
        """
        super().setup_init(
            params,
            phase=phase,
            independent_slices=independent_slices,
            no_slices=no_slices,
            timesteps_per_slice=timesteps_per_slice,
            timesteps_shift=timesteps_shift,
            plasma_periods_per_slice=plasma_periods_per_slice,
            plasma_periods_shift=plasma_periods_shift,
            multi_run_average=multi_run_average,
            dimensional_average=dimensional_average,
            runs=runs,
        )
        self.update_args(hist_kwargs, max_no_moment, curve_fit_kwargs, **kwargs)

    @arg_update_doc
    def update_args(
        self,
        hist_kwargs: dict = None,
        max_no_moment: int = None,
        curve_fit_kwargs: dict = None,
        **kwargs,
    ):
        """Update observable specific attributes and call update_finish."""

        if curve_fit_kwargs:
            self.curve_fit_kwargs = curve_fit_kwargs
        elif not hasattr(self, "curve_fit_kwargs"):
            self.curve_fit_kwargs = {}

        # Check on hist_kwargs
        if hist_kwargs:
            if not isinstance(hist_kwargs, dict):
                raise TypeError("hist_kwargs not a dictionary. Please pass a dictionary.")
            # Ensure each value is a list with one entry per species
            for key, value in hist_kwargs.items():
                if not isinstance(hist_kwargs[key], list):
                    hist_kwargs[key] = [value for _ in range(self.num_species)]
            self.hist_kwargs = hist_kwargs
        elif not hasattr(self, "hist_kwargs"):
            self.hist_kwargs = {}

        # Default number of moments to calculate
        if max_no_moment:
            self.max_no_moment = max_no_moment
        else:
            self.max_no_moment = 6

        # Update with remaining keyword arguments
        self.__dict__.update(kwargs.copy())
        self.update_finish()

        # Hermite-related defaults
        if hasattr(self, "max_hermite_order") and not hasattr(self, "hermite_rms_tol"):
            self.hermite_rms_tol = 0.05

        self.species_plots_dirs = None

        # 2nd dimension of the raw velocity array
        self.dim = 1 if self.dimensional_average else self.dimensions
        # range(inv_dim) for the loop over dimensions
        self.inv_dim = self.dimensions if self.dimensional_average else 1

        self.save_state()

    # ------------------------------------------------------------------
    # compute
    # ------------------------------------------------------------------

    def compute(self, compute_moments: bool = False, compute_Grad_expansion: bool = False):
        """Calculate the velocity distribution and associated quantities.

        Parameters
        ----------
        compute_moments : bool, optional
            Whether to compute velocity moments. Default = False.

        compute_Grad_expansion : bool, optional
            Whether to compute the Hermite/Grad expansion. Default = False.
        """
        # Grab simulation data
        time, vel_raw = self.grab_sim_data(pva="vel")

        # Normality test
        self.normality_tests(time=time, vel_data=vel_raw)

        # Calculate velocity moments
        if compute_moments:
            self.compute_moments(parse_data=False, vel_raw=vel_raw, time=time)

        if compute_Grad_expansion:
            self.compute_hermite_expansion(compute_moments=False)

    # ------------------------------------------------------------------
    # normality_tests
    # ------------------------------------------------------------------

    def normality_tests(self, time, vel_data):
        """Calculate the Shapiro-Wilks test for each timestep.

        Performs the Shapiro-Wilks test per timestep, per species, per axis
        from the raw velocity data and stores the results into zarr arrays.

        Parameters
        ----------
        time : numpy.ndarray
            One-dimensional array with time data.

        vel_data : numpy.ndarray
            Array with shape (``no_dumps``, ``dim``,
            ``runs * inv_dim * total_num_ptcls``).
            ``dim`` = 1 if ``dimensional_average`` is True, otherwise equals
            the number of dimensions. ``runs`` is the number of runs to be
            averaged over. ``inv_dim`` is the complement of ``dim``.
        """
        tinit = self.timer.current()

        no_dim = vel_data.shape[1]
        no_dumps = len(time)

        # Storage: shape (no_species, no_dim, no_dumps) for W and p separately
        shapiro_W = zeros((self.num_species, no_dim, no_dumps))
        shapiro_p = zeros((self.num_species, no_dim, no_dumps))

        for it in tqdm(range(no_dumps), desc="Normality tests", disable=not self.verbose):
            for d in range(no_dim):
                for sp, sp_start in enumerate(self.species_index_start[:-1]):
                    sp_end = self.species_index_start[sp + 1]
                    stat, p_value = shapiro(vel_data[it, d, sp_start:sp_end])
                    shapiro_W[sp, d, it] = stat
                    shapiro_p[sp, d, it] = p_value

        # Store as instance attributes
        self.normality_W = shapiro_W
        self.normality_p = shapiro_p
        self.normality_time = time

        # Persist to zarr
        species_attr = {"species": list(self.species_names)}
        self.save_zarr(shapiro_W, "normality/shapiro_W", attrs=species_attr)
        self.save_zarr(shapiro_p, "normality/shapiro_p", attrs=species_attr)
        self.save_zarr(time, "normality/time")

        tend = self.timer.current()
        time_stamp(
            self.log_file,
            "Normality tests (Shapiro-Wilks)",
            self.timer.time_division(tend - tinit),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # prepare_histogram_args
    # ------------------------------------------------------------------

    def prepare_histogram_args(self):
        """Initialise histogram arguments for each species."""
        if not hasattr(self, "hist_kwargs"):
            self.hist_kwargs = {"density": [], "bins": [], "range": []}

        bin_width = 0.05
        wid = 5
        no_bins = int(2.0 * wid / bin_width)

        # Thermal speed from temperature / energy data
        try:
            from os.path import exists as _exists
            energy_fle = self.prod_energy_filename if self.phase == "production" else self.eq_energy_filename
            if not _exists(energy_fle):
                raise FileNotFoundError
            # Read via numpy (no pandas)
            energy_data = np.genfromtxt(energy_fle, delimiter=",", names=True)
            if self.num_species > 1:
                vth = zeros(self.num_species)
                for sp, (sp_mass, sp_name) in enumerate(zip(self.species_masses, self.species_names)):
                    col = "{}_Temperature".format(sp_name)
                    vth[sp] = sqrt(float(energy_data[col].mean()) * self.kB / sp_mass)
            else:
                vth = sqrt(float(energy_data["Temperature"].mean()) * self.kB / self.species_masses)
        except (FileNotFoundError, (ValueError, KeyError)):
            vth = sqrt(self.kB * self.T_desired / self.species_masses)

        self.vth = vth.copy() if hasattr(vth, "copy") else array([vth])

        default_hist_kwargs = {"density": [], "bins": [], "range": []}
        for sp in range(self.num_species):
            default_hist_kwargs["density"].append(True)
            default_hist_kwargs["bins"].append(no_bins)
            default_hist_kwargs["range"].append((-wid * self.vth[sp], wid * self.vth[sp]))

        must_have_keys = ["bins", "range", "density"]
        for key in must_have_keys:
            try:
                if len(self.hist_kwargs[key]) == 0:
                    self.hist_kwargs[key] = default_hist_kwargs[key]
            except KeyError:
                self.hist_kwargs[key] = default_hist_kwargs[key]

        self.list_hist_kwargs = []
        for indx in range(self.num_species):
            another_dict = {}
            for key, values in self.hist_kwargs.items():
                another_dict[key] = values[indx]
            self.list_hist_kwargs.append(another_dict)

    # ------------------------------------------------------------------
    # compute_moments
    # ------------------------------------------------------------------

    def compute_moments(
        self,
        parse_data: bool = False,
        vel_raw=None,
        time=None,
        **kwargs,
    ):
        """Calculate and save moments of the velocity distribution.

        Parameters
        ----------
        parse_data : bool, optional
            If True, read velocity data from simulation dumps before computing.
            Default = False. If False, *vel_raw* and *time* must be supplied.

        vel_raw : numpy.ndarray, optional
            Container of particles' velocity at each time step.

        time : numpy.ndarray, optional
            Time array.

        **kwargs
            Additional keyword arguments (unused; accepted for API compatibility).
        """
        if parse_data:
            time, vel_raw = self.grab_sim_data(pva="vel")

        tinit = self.timer.current()
        moments, ratios = calc_moments(vel_raw, self.max_no_moment, self.species_index_start)
        tend = self.timer.current()
        time_stamp(
            self.log_file,
            "Velocity moments calculation",
            self.timer.time_division(tend - tinit),
            self.verbose,
        )

        # Store as instance attributes: shape (no_species, no_dumps, dim, max_no_moment)
        self.moments_data = moments
        self.moments_ratios = ratios
        self.moments_time = time

        # Save time array to zarr once
        self.save_zarr(time, "moments/time")

        # Save per species, per axis
        for i, sp_name in enumerate(self.species_names):
            for d, ds in enumerate(self.dim_labels[: self.dim]):
                path_mom = "moments/sp_{}_{}".format(sp_name, ds)
                path_rat = "moments/ratios_sp_{}_{}".format(sp_name, ds)
                # shape (no_dumps, max_no_moment)
                self.save_zarr(
                    moments[i, :, d, :],
                    path_mom,
                    attrs={"species": sp_name, "axis": ds, "quantity": "moments"},
                )
                self.save_zarr(
                    ratios[i, :, d, :],
                    path_rat,
                    attrs={"species": sp_name, "axis": ds, "quantity": "moment_ratios"},
                )

    # ------------------------------------------------------------------
    # compute_hermite_expansion
    # ------------------------------------------------------------------

    def compute_hermite_expansion(self, compute_moments: bool = True, **kwargs):
        """Calculate and save Hermite (Grad) expansion coefficients.

        Parameters
        ----------
        compute_moments : bool, optional
            If True, compute velocity moments first (required input for the
            Hermite fit). Default = True.

        **kwargs
            Additional keyword arguments forwarded to ``compute_moments``.

        Notes
        -----
        This method is still in development. The iterative procedure finds the
        underlying Gaussian in a non-equilibrium distribution before computing
        the Hermite coefficients via :func:`calculate_herm_coeff`.
        """
        if compute_moments:
            self.compute_moments(parse_data=True, **kwargs)

        if not hasattr(self, "hermite_rms_tol"):
            self.hermite_rms_tol = 0.05

        if not hasattr(self, "max_hermite_order"):
            raise AttributeError(
                "max_hermite_order is not set. Set it before calling compute_hermite_expansion()."
            )

        no_dumps = len(self.moments_time)

        self.hermite_sigmas = zeros((self.num_species, self.dim, no_dumps))
        self.hermite_epochs = zeros((self.num_species, self.dim, no_dumps))
        # shape (no_species, dim, max_hermite_order+1, no_dumps)
        hermite_coeff = zeros((self.num_species, self.dim, self.max_hermite_order + 1, no_dumps))

        print("\nCalculating Hermite coefficients ...")
        tinit = self.timer.current()

        for sp, sp_name in enumerate(tqdm(self.species_names, desc="Species")):
            for it in tqdm(range(no_dumps), desc="Time", leave=False):
                # Grab the thermal speed from the 2nd moment (variance)
                vrms = float(self.moments_data[sp, it, 0, 1]) ** 0.5

                for d, ds in zip(range(self.dim), self.dim_labels):
                    # Retrieve the velocity distribution for this species/axis/timestep
                    # from the zarr store
                    dist = self.load_zarr("distributions/sp_{}_{}".format(sp_name, ds))
                    v_bins = self.load_zarr("distributions/bins_{}_{}".format(sp_name, ds))

                    dist_t = dist[it]
                    cntrl = True
                    j = 0

                    while cntrl:
                        norm_val = trapz(dist_t, x=v_bins / vrms)

                        h_coeff = calculate_herm_coeff(
                            v_bins / vrms,
                            dist_t / norm_val,
                            self.max_hermite_order,
                        )

                        res, _ = curve_fit(
                            lambda x, rms: grad_expansion(x, rms, h_coeff),
                            v_bins / vrms,
                            dist_t / norm_val,
                            **self.curve_fit_kwargs,
                        )

                        vrms *= res[0]

                        if abs(1.0 - res[0]) < self.hermite_rms_tol:
                            cntrl = False
                            self.hermite_sigmas[sp, d, it] = vrms
                            self.hermite_epochs[sp, d, it] = j
                            hermite_coeff[sp, d, :, it] = h_coeff
                        j += 1

        tend = self.timer.current()

        # Save per species, per axis
        for sp, sp_name in enumerate(self.species_names):
            for d, ds in zip(range(self.dim), self.dim_labels):
                path = "hermite/sp_{}_{}".format(sp_name, ds)
                # shape (max_hermite_order+1, no_dumps)
                self.save_zarr(
                    hermite_coeff[sp, d, :, :],
                    path,
                    attrs={
                        "species": sp_name,
                        "axis": ds,
                        "max_hermite_order": self.max_hermite_order,
                    },
                )
                self.save_zarr(
                    self.hermite_sigmas[sp, d, :],
                    "hermite/sigmas_{}_{}".format(sp_name, ds),
                    attrs={"species": sp_name, "axis": ds, "quantity": "rms_width"},
                )
                self.save_zarr(
                    self.hermite_epochs[sp, d, :],
                    "hermite/epochs_{}_{}".format(sp_name, ds),
                    attrs={"species": sp_name, "axis": ds, "quantity": "convergence_epochs"},
                )

        # Store full coefficient array as instance attribute for convenience
        self.hermite_coeff = hermite_coeff

        time_stamp(
            self.log_file,
            "Hermite expansion calculation",
            self.timer.time_division(tend - tinit),
            self.verbose,
        )

    # ------------------------------------------------------------------
    # pretty_print
    # ------------------------------------------------------------------

    def pretty_print(self):
        """Print information in a user-friendly way."""
        print("\n\n{:=^70} \n".format(" " + self.__long_name__ + " "))
        print("Zarr store: ", self.zarr_store_path)
        print("\nMulti run average: ", self.multi_run_average)
        print("No. of runs: ", self.runs)
        print(
            "Size of the parsed velocity array: {} x {} x {}".format(
                self.no_dumps,
                self.dim,
                self.runs * self.inv_dim * self.total_num_ptcls,
            )
        )

        if hasattr(self, "max_no_moment"):
            print("\nMoments Information:")
            print("Zarr group: moments/")
            print("Highest moment to calculate: {}".format(self.max_no_moment))

        if hasattr(self, "max_hermite_order"):
            print("\nGrad Expansion Information:")
            print("Zarr group: hermite/")
            print("Highest order to calculate: {}".format(self.max_hermite_order))
            print("RMS Tolerance: {:.3f}".format(self.hermite_rms_tol))
