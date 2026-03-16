"""
Module handling stages of an MD run: PreProcessing, Simulation, PostProcessing.
"""

from importlib import import_module
from IPython import get_ipython

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import trange
else:
    from tqdm import tqdm, trange

import h5py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import LogNorm
from numpy import (
    arange,
    array,
    full,
    int64,
    linspace,
    log2,
    log10,
    logspace,
    meshgrid,
    pi,
    quantile,
    sqrt,
    zeros,
)
from os import listdir, mkdir
from os import remove as os_remove
from os import stat as os_stat
from os.path import exists, join
from pandas import DataFrame, read_csv
from plotly.subplots import make_subplots
from seaborn import scatterplot
from warnings import warn

# Sarkas modules
from .core import Parameters
from .particles import Particles
from .plasma import Species
from .plotting.styles import get_msu_colors
from .potentials.core import Potential
from .pppm_bayesian_optimization import BayesianPPPMOptimizer
from .time_evolution.integrators import Integrator
from .tools.observables import run_thermalization_tests
from .utilities.io import InputOutput, print_to_logger
from .utilities.maths import force_error_analytic_pp, force_error_approx_pppm
from .utilities.timing import SarkasTimer


class Process:
    """Parent class for :class:`sarkas.process.PreProcess`, :class:`sarkas.process.Simulation`, and
    :class:`sarkas.process.PostProcess`.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file. Default = `None`

    Attributes
    ----------
    potential : :class:`sarkas.potential.base.Potential`
        Class handling the interaction between particles.

    integrator : :class:`sarkas.time_evolution.integrators.Integrator`
        Class handling the integrator.

    particles: :class:`sarkas.particles.Particles`
        Class handling particles properties.

    parameters : :class:`sarkas.core.Parameters`
        Class handling simulation's parameters.

    species : list
        List of :class:`sarkas.plasma.Species` classes.

    input_file : str
        Path to YAML input file.

    timer : :class:`sarkas.utilities.timing.SarkasTimer`
        Class handling the timing of processes.

    io : :class:`sarkas.utilities.io.InputOutput`
        Class handling the IO in Sarkas.

    """

    def __init__(
        self,
        input_file: str = None,
        potential_class: Potential = None,
        integrator_class: Integrator = None,
        particles_class: Particles = None,
        parameters_class: Parameters = None,
        io_class: InputOutput = None,
        species: list = None,
    ):
        if potential_class is not None:
            self.potential = potential_class
        else:
            self.potential = Potential()

        if integrator_class is not None:
            self.integrator = integrator_class
        else:
            self.integrator = Integrator()

        if particles_class is not None:
            self.particles = particles_class
        else:
            self.particles = Particles()

        if parameters_class is not None:
            self.parameters = parameters_class
        else:
            self.parameters = Parameters()

        if io_class is not None:
            self.io = io_class
        else:
            self.io = InputOutput(process=self.__name__)

        if species is not None:
            self.species = species
        else:
            self.species = []

        self.observables_dict = {}
        self.transport_dict = {}

        self.input_file = input_file
        self.timer = SarkasTimer()

    def common_parser(self, filename: str = None):
        """
        Parse simulation parameters from a YAML file.

        Parameters
        ----------
        filename : str, optional
            Path to the YAML input file. If not provided, the input file path specified during object initialization will be used.

        Returns
        -------
        dict
            A nested dictionary containing the parsed simulation parameters.

        Notes
        -----
        This method reads the simulation parameters from a YAML file and returns them as a nested dictionary.
        It uses the :meth:`sarkas.utilities.io.InputOutput.from_yaml` to read the YAML file.

        If the `filename` parameter is provided, it will override the input file path specified during object initialization.

        Examples
        --------
        >>> process = Process(input_file='/path/to/input.yaml')
        >>> params_dict = process.common_parser()

        """
        if filename:
            self.input_file = filename

        params_dict = self.io.from_yaml(self.input_file)

        return params_dict

    def update_subclasses_from_dict(self, nested_dict: dict):
        """Update the subclasses parameters using a dictionary.

        Parameters
        ----------
        nested_dict : dict
            Nested dictionary. See example for format.

        """
        for lkey, vals in nested_dict.items():
            if lkey not in ["Particles", "Observables", "TransportCoefficients"]:
                self.__dict__[lkey.lower()].__dict__.update(vals)
            elif lkey in ["Particles", "Plasma"]:
                # Remember Particles should be a list of dict
                # example:
                # args = {"Particles" : [ { "Species" : { "name": "O" } } ] }

                # Check if you already have a non-empty dict of species
                if len(self.species) > 0:
                    # If so do you want to replace or update?
                    # Update species attributes

                    for sp, species in enumerate(vals):
                        spec = Species(species["Species"])
                        if hasattr(spec, "replace"):
                            self.species[sp].__dict__.update(spec.__dict__)
                        else:
                            self.species.append(spec)
                else:
                    # Append new species
                    for sp, species in enumerate(vals):
                        spec = Species(species["Species"])
                        self.species.append(spec)

            elif lkey in ["Observables"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".observables", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        if inst.__long_name__ in self.observables_dict.keys():
                            self.observables_dict[inst.__long_name__].__dict__.update(params)
                        else:
                            self.observables_dict[inst.__long_name__] = inst
                            self.observables_dict[inst.__long_name__].from_dict(params)

            elif lkey in ["TransportCoefficients"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".transport", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        if inst.__long_name__ in self.transport_dict.keys():
                            self.transport_dict[inst.__long_name__].__dict__.update(params)
                        else:
                            self.transport_dict[inst.__long_name__] = inst
                            self.transport_dict[inst.__long_name__].from_dict(params)

        # electron properties has been moved to the Parameters class. Therefore I need to put this here.
        if hasattr(self.potential, "electron_temperature"):
            self.parameters.electron_temperature = self.potential.electron_temperature
        elif hasattr(self.potential, "electron_temperature_eV"):
            self.parameters.electron_temperature_eV = self.potential.electron_temperature_eV

    def instantiate_subclasses_from_dict(self, nested_dict: dict):
        """Instantiate the process subclasses from a dictionary."""

        for lkey, vals in nested_dict.items():
            if lkey not in ["Particles", "Observables", "TransportCoefficients"]:
                # self.__dict__[lkey.lower()].__dict__.update(vals)

                if lkey == "Potential":
                    self.potential.from_dict(nested_dict[lkey])

                elif lkey == "Integrator":
                    self.integrator.from_dict(nested_dict[lkey])

                elif lkey == "Parameters":
                    self.parameters.from_dict(nested_dict[lkey])

            elif lkey in ["Particles", "Plasma"]:
                for sp, species in enumerate(vals):
                    spec = Species(species["Species"])
                    self.species.append(spec)

            elif lkey in ["Observables"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".observables", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        self.observables_dict[inst.__long_name__] = inst
                        self.observables_dict[inst.__long_name__].from_dict(params)

            elif lkey in ["TransportCoefficients"]:
                for obs_dict in vals:
                    for obs, params in obs_dict.items():
                        module = import_module(".transport", "sarkas.tools")
                        class_ = getattr(module, obs)
                        inst = class_()
                        self.transport_dict[inst.__long_name__] = inst
                        self.transport_dict[inst.__long_name__].from_dict(params)

        # electron properties has been moved to the Parameters class. Therefore I need to put this here.
        if hasattr(self.potential, "electron_temperature"):
            self.parameters.electron_temperature = self.potential.electron_temperature
        elif hasattr(self.potential, "electron_temperature_eV"):
            self.parameters.electron_temperature_eV = self.potential.electron_temperature_eV

    def directory_sizes(self):
        """Calculate the size of the dumps directories and print them to logger."""
        # Estimate size of dump folder
        if self.__name__ == "preprocessing":
            if self.parameters.equilibration_phase:
                eq_dump_size = self.io.estimate_existing_file_size("equilibration")
            else:
                eq_dump_size = 0

            prod_dump_size = self.io.estimate_existing_file_size("production")

            sizes = array([eq_dump_size, prod_dump_size])

            if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
                mag_dump_size = self.io.estimate_existing_file_size("magnetization")
                sizes = array([eq_dump_size, prod_dump_size, mag_dump_size])
        else:
            # Grab one file from the dump directory and get the size of it.
            if self.parameters.equilibration_phase:
                if not listdir(self.io.eq_dump_dir):
                    raise FileNotFoundError(
                        "Could not estimate the size of the equilibration phase dumps"
                        " because there are no dumps in the equilibration directory."
                        "Re-run .time_n_space_estimate(loops) with loops > eq_dump_step"
                    )
                else:
                    eq_dump_size = os_stat(join(self.io.eq_dump_dir, listdir(self.io.eq_dump_dir)[0])).st_size
                    # eq_dump_fldr_size = eq_dump_size # * (self.parameters.equilibration_steps / self.parameters.eq_dump_step)
            else:
                eq_dump_size = 0

            if not listdir(self.io.directory_tree[self.__name__]["production"]["path"]):
                raise FileNotFoundError(
                    "Could not estimate the size of the production phase because there are no files in the production directory."
                    "Re-run .time_n_space_estimate(loops) with loops > prod_dump_step"
                )

            # Grab one file from the dump directory and get the size of it.
            prod_dump_size = os_stat(join(self.io.prod_dump_dir, listdir(self.io.prod_dump_dir)[0])).st_size

            # Prepare arguments to pass for print out
            sizes = array([eq_dump_size, prod_dump_size])
            # Check for electrostatic equilibration
            if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
                if not listdir(self.io.mag_dump_dir):
                    raise FileNotFoundError(
                        "Could not estimate the size of the magnetization phase dumps because"
                        " there are no dumps in the production directory."
                        "Re-run .time_n_space_estimate(loops) with loops > mag_dump_step"
                    )
                # dump = self.parameters.mag_dump_step
                mag_dump_size = os_stat(join(self.io.mag_dump_dir, listdir(self.io.mag_dump_dir)[0])).st_size
                sizes = array([eq_dump_size, prod_dump_size, mag_dump_size])
        self.io.directory_size_report(sizes, process=self.__name__)

    def evolve(self, phase, thermalization, it_start, it_end, dump_step):
        """
        Evolve the system forward in time.

        Parameters
        ----------
        phase: str
            Indicates the stage of the simulation used for saving dumps in the right directory. \n
            Choices = ("equilibration", "production", "magnetization")

        thermalization : bool
            Indicates whether to apply the thermostat or not.

        it_start: int
            Initial timestep of the loop.

        it_end: int
            Final timestep of the loop.

        dump_step: int
            Interval for dumping data.

        """

        for it in trange(it_start, it_end, disable=not self.parameters.verbose):
            # Calculate the Potential energy and update particles' data

            self.integrator.update(self.particles)

            if (it + 1) % dump_step == 0:
                self.particles.calculate_observables()

                # self.io.dump(phase, self.particles, it + 1)
                time = self.integrator.dt * (it + 1)
                self.io.save_timestep_data(it + 1, dump_step, time, self.particles)

            if thermalization and (it + 1 >= self.integrator.thermalization_timestep):
                self.particles.calculate_species_kinetic_temperature()
                self.integrator.thermostate(self.particles)

    def initialization(self):
        """Initialize all classes."""

        # initialize the directories and filenames
        self.io.setup()

        # Copy relevant subsclasses attributes into parameters class. This is needed for post-processing.
        # it updates parameters' dictionary with filenames and directories
        self.parameters.copy_io_attrs(self.io)

        self.parameters.potential_type = self.potential.type.lower()
        self.parameters.setup(self.species)

        # Initialize particles
        t0 = self.timer.current()
        self.particles.setup(self.parameters, self.species)
        time_ptcls = self.timer.current()
        self.parameters.particles_initialization_time = time_ptcls - t0

        # Initialize potential and calculate initial potential
        self.potential.setup(self.parameters, self.species)
        self.potential.calc_acc_pot(self.particles)
        time_pot = self.timer.current()
        self.parameters.cutoff_radius = self.potential.rc

        # Initialize Integrator
        self.integrator.setup(self.parameters, self.potential)
        # Copy needed parameters for pretty print
        self.parameters.dt = self.integrator.dt
        self.parameters.equilibration_integrator = self.integrator.equilibration_type
        self.parameters.production_integrator = self.integrator.production_type
        if self.parameters.magnetized:
            self.parameters.magnetization_integrator = self.integrator.magnetization_type

        # Copy some parameters needed for saving data
        self.io.copy_params(self.parameters)
        # For restart and backups.
        self.io.setup_checkpoint(self.parameters, self.particles, phase="equilibration")
        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            self.io.setup_checkpoint(self.parameters, self.particles, phase="magnetization")
        self.io.setup_checkpoint(self.parameters, self.particles, phase="production")

        self.io.save_simulation_state(self)

        # Print Process summary to file and screen
        self.io.simulation_summary(self)
        time_end = self.timer.current()

        # self.evolve = self.evolve_loop_threading if self.parameters.threading else self.evolve_loop

        # Print timing
        self.io.time_stamp("Particles Initialization", self.timer.time_division(time_ptcls - t0))
        self.io.time_stamp("Potential Initialization", self.timer.time_division(time_pot - time_ptcls))
        self.io.time_stamp("Total Simulation Initialization", self.timer.time_division(time_end - t0))

        self.print_initial_state()

    def print_initial_state(self):
        """Print the initial energies of the system."""

        init_eng = " Initial Energies "
        msg = f"\n\n{init_eng:-^70}\n" f"Initial temperature and kinetic energy of each species\n"

        self.particles.calculate_species_kinetic_temperature()
        self.particles.calculate_species_potential_energy()

        factor = self.parameters.J2erg if self.parameters.units == "mks" else 1.0 / self.parameters.J2erg

        for sp, kp, tp, pot_sp in zip(
            self.species,
            self.particles.species_kinetic_energy,
            self.particles.species_temperature,
            self.particles.species_potential_energy,
        ):
            sp_msg = (
                f"Species {sp.name} :\n"
                f"\tTemperature = {tp:.6e} {self.parameters.units_dict['temperature']} = {tp * self.parameters.eV2K:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tKinetic Energy = {kp:.6e} {self.parameters.units_dict['energy']} = {kp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tPotential Energy = {pot_sp:.6e} {self.parameters.units_dict['energy']} = {pot_sp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            )
            msg += sp_msg

        tot_kin_e = self.particles.species_kinetic_energy.sum()
        tot_pot_e = self.particles.species_potential_energy.sum()
        tot_e = tot_kin_e + tot_pot_e

        msg += (
            f"Initial total kinetic energy = {tot_kin_e:.6e} {self.parameters.units_dict['energy']} = {tot_kin_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total potential energy = {tot_pot_e:.6e} {self.parameters.units_dict['energy']} = {tot_pot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total energy = {tot_e:.6e} {self.parameters.units_dict['energy']} = {tot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
        )
        self.io.write_to_logger(msg)

    def setup(self, read_yaml: bool = True, input_file: str = None, other_inputs: dict = None):
        """
        Setup simulations' subclasses by reading the YAML input file and update them with `other_inputs`.

        Parameters
        ----------
        read_yaml: bool
            Flag for reading YAML input file. Default = True.

        input_file: str (optional)
            Path to YAML file with inputs.

        other_inputs: dict (optional)
            Nested dictionary with additional simulations options. This is called after reading the YAML file.

        """
        if input_file:
            self.input_file = input_file

        if read_yaml:
            yaml_dict = self.common_parser()
            self.instantiate_subclasses_from_dict(yaml_dict)

        if other_inputs:
            if not isinstance(other_inputs, dict):
                raise TypeError("Wrong input type. other_inputs should be a nested dictionary")
            self.update_subclasses_from_dict(other_inputs)

        if self.__name__ == "postprocessing":
            # Create the file paths without creating directories and redefining io attributes
            self.io.make_directory_tree()
            self.io.make_directories()
            self.io.make_files_tree()

            # Read the parameters and species classes from saved files
            self.io.read_simulation_state(self, self.io.directory_tree["simulation"]["path"])
            self.io.copy_params(self.parameters)

            # DEV NOTE: the potential setup could take a long time if the optimal green function need be calculated
            self.potential.setup(self.parameters, self.species)
            self.integrator.setup(self.parameters, self.potential)

            # Print parameters to log file
            if not exists(self.io.log_file):
                # if the file exists do not print the file header
                self.io.file_header()
                self.io.simulation_summary(self)

            self.io.datetime_stamp()

            if self.grab_last_step:
                # Initialize the Particles class attributes by reading the last step
                old_method = self.parameters.load_method
                self.parameters.load_method = "production_restart"
                last_step = self.parameters.production_steps
                self.parameters.restart_step = last_step
                self.particles.setup(self.parameters, self.species)

                # Restore the original value for future use
                self.parameters.load_method = old_method
                # Update the log file. It is set to the simulation log in the parameters class, but it is correct in the IO class.
                self.parameters.log_file = self.io.log_file
        else:
            self.initialization()

        if self.parameters.plot_style:
            plt.style.use(self.parameters.plot_style)

    def setup_from_dict(self, input_dict: dict):
        """Setup simulations' subclasses from a nested dictionary.

        Parameters
        ----------
        input_dict: dict
            Nested dictionary with all necessary simulations parameters.

        Note
        ----
        This method does the same as:meth:`setup` but without reading a yaml file.
        If you want to update the attributes of this class use :meth:`update_subclasses_from_dict`.
        """

        self.instantiate_subclasses_from_dict(input_dict)

        if self.__name__ == "postprocessing":
            # Create the file paths without creating directories and redefining io attributes
            self.io.make_directory_tree()
            self.io.make_directories()
            self.io.make_files_tree()

            # Read the parameters and species classes from saved files
            self.io.read_simulation_state(self, self.io.directory_tree["simulation"]["path"])
            self.io.copy_params(self.parameters)

            # DEV NOTE: the potential setup could take a long time if the optimal green function need be calculated
            self.potential.setup(self.parameters, self.species)
            self.integrator.setup(self.parameters, self.potential)

            # Print parameters to log file
            if not exists(self.io.log_file):
                # if the file exists do not print the file header
                self.io.file_header()
                self.io.simulation_summary(self)

            self.io.datetime_stamp()

            if self.grab_last_step:
                # Initialize the Particles class attributes by reading the last step
                old_method = self.parameters.load_method
                self.parameters.load_method = "production_restart"
                last_step = self.parameters.production_steps
                self.parameters.restart_step = last_step
                self.particles.setup(self.parameters, self.species)

                # Restore the original value for future use
                self.parameters.load_method = old_method
                # Update the log file. It is set to the simulation log in the parameters class, but it is correct in the IO class.
                self.parameters.log_file = self.io.log_file
        else:
            self.initialization()

        if self.parameters.plot_style:
            plt.style.use(self.parameters.plot_style)


class PostProcess(Process):
    """
    Class handling the post-processing stage of a simulation.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    """

    def __init__(self, input_file: str = None, grab_last_step: bool = False):
        self.__name__ = "postprocessing"
        self.grab_last_step = grab_last_step

        super().__init__(input_file)

    def run(self):
        """Calculate all the observables from the YAML input file."""

        if len(self.observables_dict.keys()) == 0:
            print("No observables found in observables_dict")
        else:
            for obs_key, obs_class in self.observables_dict.items():
                obs_class.setup(self.parameters)
                msg = obs_class.pretty_print_msg()
                self.io.write_to_logger(msg)

                if obs_key == "Thermodynamics":
                    # Make Temperature and Energy plots
                    obs_class.temp_energy_plot(self)
                else:
                    obs_class.compute()

        if len(self.transport_dict.keys()) == 0:
            print("No transport coefficients found in tranport_dict")
        else:
            for obs_key, obs_class in self.transport_dict.items():
                obs_class.setup(self.parameters)
                msg = obs_class.pretty_print_msg()
                self.io.write_to_logger(msg)
                obs_class.compute(observable=self.observables_dict[obs_class.required_observable])

    def setup_from_simulation(self, simulation):
        """
        Setup postprocess' subclasses by (shallow) copying them from simulation object.

        Parameters
        ----------
        simulation: :class:`sarkas.core.processes.Simulation`
            Simulation object

        """
        self.parameters = simulation.parameters.__copy__()
        self.integrator = simulation.integrator.__copy__()
        self.potential = simulation.potential.__copy__()
        self.species = simulation.species.copy()
        self.io = simulation.io.__copy__()
        self.io.process = "postprocess"


class PreProcess(Process, BayesianPPPMOptimizer):
    """
    Wrapper class handling the estimation of time and best parameters of a simulation.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    Attributes
    ----------
    loops: int
        Number of timesteps to run for time and size estimates. Default = 10

    estimate: bool
        Run an estimate for the best PPPM parameters in the simulation. Default=False.

    pm_meshes: numpy.ndarray
        Array of mesh sizes used in the PPPM parameters estimation.

    pp_cells: numpy.ndarray
        Array of simulations box cells used in the PPPM parameters estimation.

    kappa: float
        Screening parameter. Calculated from :meth:`sarkas.potentials.core.Potential.matrix`.

    """

    def __init__(self, input_file: str = None):
        self.__name__ = "preprocessing"
        self.estimate = False
        self.pm_meshes = logspace(3, 7, 12, base=2, dtype=int64)
        # array([16, 24, 32, 48, 56, 64, 72, 88, 96, 112, 128], dtype=int64)
        self.pm_caos = arange(1, 8, dtype=int64)
        self.pp_cells = arange(3, 16, dtype=int64)
        super().__init__(input_file)

    def analytical_approx_pppm(self, rcuts=None, alphas=None, rlims=None, alims=None, mesh_size=None, cao=None):
        """Calculate the total force error as given in :cite:`Dharuman2017`.
        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances in real units, i.e. cm or m.
            If None, it will be calculated from rlims or from the potential cutoff radius.
        alphas: numpy.ndarray
            Ewald parameters in real units, i.e. cm^-1 or m^-1.
            If None, it will be calculated from alims or from the potential Ewald parameter.

        rlims: tuple
            Min and max cut off distances in real units, i.e. cm or m.
        alims: tuple
            Min and max Ewald parameters in real units, i.e. cm^-1 or m^-1.
        mesh_size: int
            Mesh size for the PPPM part.
        cao: int
            Cells per box length for the PP part.
        Returns
        -------
        total_force_error: numpy.ndarray
            Force error matrix.
        pp_force_error: numpy.ndarray
            Force error matrix for the PP part.
        pm_force_error: numpy.ndarray
            Force error array for the PM part.
        rcuts: numpy.ndarray
            Cut off distances in dimensionless units, i.e. in Wigner-Seitz radius.
        alphas: numpy.ndarray
            Ewald parameters in dimensionless units, i.e. in Wigner-Seitz radius^-1.
        """
        if rcuts is None:
            if rlims:
                r_min, r_max = rlims
            else:
                r_min = self.potential.rc * 0.5
                r_max = self.potential.rc * 2.0
            rcuts = linspace(r_min, r_max, 101)
        else:
            r_min = rcuts.min()
            r_max = rcuts.max()

        if alphas is None:
            if alims:
                a_min, a_max = alims
            else:
                a_min = self.potential.pppm_alpha_ewald * 0.25
                a_max = self.potential.pppm_alpha_ewald * 2.0

            alphas = linspace(a_min, a_max, 101)

        else:
            a_min = alphas.min()
            a_max = alphas.max()

        # Create the meshgrids
        pm_force_error = zeros(len(alphas))
        pp_force_error = zeros((len(alphas), len(rcuts)))
        total_force_error = zeros((len(alphas), len(rcuts)))

        if mesh_size is not None:
            # if mesh_size is float convert to int
            pppm_mesh = full(3, int(mesh_size), dtype=int64)
        else:
            pppm_mesh = self.potential.pppm_mesh.copy()

        if cao is not None:
            cao = int(cao)
        else:
            cao = self.potential.pppm_cao[0]

        lambda_k = self.potential.screening_length / self.potential.a_ws
        ha = self.potential.box_lengths / pppm_mesh / self.potential.a_ws
        rescaling_constant = self.potential.QFactor / (self.parameters.total_num_ptcls) * sqrt(3.0 / (4.0 * pi))
        rescaling_constant /= self.potential.matrix[0, 0, 0]  # Rescale by the first species charges

        for ia, alpha in enumerate(alphas):
            for ir, rc in enumerate(rcuts):
                tot_err, pm_err, pp_err = force_error_approx_pppm(
                    screening_length=lambda_k,
                    cutoff_radius=rc / self.potential.a_ws,
                    alpha_ewald=alpha * self.potential.a_ws,
                    mesh_discretization=ha[0],
                    cao=cao,
                    rescaling_constant=rescaling_constant,
                )
                total_force_error[ia, ir] = tot_err
                pm_force_error[ia] = pm_err
                pp_force_error[ia, ir] = pp_err

        return (
            total_force_error,
            pp_force_error,
            pm_force_error,
            rcuts / self.potential.a_ws,
            alphas * self.potential.a_ws,
        )

    def green_function_timer(self):
        """Time Potential setup."""

        self.timer.start()
        self.potential.pppm_setup()

        return self.timer.stop()

    def make_color_map(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot a color map of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.

        Raises
        ------
          DeprecationWarning

        """
        warn(
            f"The function has been renamed make_pppm_color_map. make_color_map will be removed in v2.0.0.",
            DeprecationWarning,
        )
        # Line Plot
        self.make_pppm_color_map(rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error)

    @staticmethod
    def make_fit_plot(pp_xdata, pm_xdata, pp_times, pm_times, pp_opt, pm_opt, pp_xlabels, pm_xlabels, fig_path):
        """
        Make a dual plot of the fitted functions.
        """
        fig, ax = plt.subplots(1, 2, figsize=(12, 7))
        ax[0].plot(pm_xdata, pm_times.mean(axis=-1), "o", label="Measured times")
        # ax[0].plot(pm_xdata, quadratic(pm_xdata, *pm_opt), '--r', label="Fit $f(x) = a + b x + c x^2$")
        ax[1].plot(pp_xdata, pp_times.mean(axis=-1), "o", label="Measured times")
        # ax[1].plot(pp_xdata, linear(pp_xdata, *pp_opt), '--r', label="Fit $f(x) = a x$")

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")

        ax[1].set_xscale("log")
        ax[1].set_yscale("log")

        ax[0].legend()
        ax[1].legend()

        ax[0].set_xticks(pm_xdata)
        ax[0].set_xticklabels(pm_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[1].set_xticks(pp_xdata[0:-1:3])
        ax[1].set_xticklabels(pp_xlabels)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax[0].set_title("PM calculation")
        ax[1].set_title("PP calculation")

        ax[0].set_xlabel("Mesh sizes")
        ax[1].set_xlabel(r"$r_c / a_{ws}$")
        fig.tight_layout()
        fig.savefig(join(fig_path, "Timing_Fit.png"))

    def make_line_plot(self, rcuts, alphas, chosen_alpha, chosen_rcut, total_force_error):
        """
        Plot selected values of the total force error approximation.

        Parameters
        ----------
        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.

        total_force_error: numpy.ndarray
            Force error matrix.

        Raises
        ------
            : DeprecationWarning
        """
        warn(
            f"The function has been renamed make_pppm_line_plot. make_line_plot will be removed in v2.0.0.",
            DeprecationWarning,
        )
        # Line Plot
        self.make_pppm_line_plot(
            total_force_error=total_force_error,
            rcuts=rcuts,
            alphas=alphas,
            chosen_alpha=chosen_alpha,
            chosen_rcut=chosen_rcut,
            chosen_mesh=self.potential.pppm_mesh[0],
            chosen_cao=self.potential.pppm_cao[0],
        )

    def make_pppm_line_plot(
        self, total_force_error, rcuts, alphas, chosen_alpha=None, chosen_rcut=None, chosen_mesh=None, chosen_cao=None
    ):
        """
        Plot selected values of the total force error approximation.

        Parameters
        ----------
        total_force_error: numpy.ndarray
            Force error matrix.

        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float, optional
            Chosen Ewald parameter.

        chosen_rcut: float, optional
            Chosen cut off radius.

        chosen_mesh: int, optional
            Chosen mesh size.

        chosen_cao: int, optional
            Chosen Spline order per box length.
        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=(19, 7))
        linestyles = [(0, (5, 10)), "dashed", "solid", "dashdot", (0, (3, 10, 1, 10))]
        # Indexes is quantiles of the alphas and rcuts arrays
        r_indexes = quantile(arange(len(rcuts)), [0.3, 0.4, 0.5, 0.6, 0.7], method="nearest").astype(int)
        a_indexes = quantile(arange(len(alphas)), [0.3, 0.4, 0.5, 0.6, 0.7], method="nearest").astype(int)

        if chosen_alpha is not None:
            a_index = (abs(alphas - chosen_alpha)).argmin()
            a_indexes[2] = a_index
        if chosen_rcut is not None:
            r_index = (abs(rcuts - chosen_rcut)).argmin()
            r_indexes[2] = r_index

        for lns, i, j in zip(linestyles, r_indexes, a_indexes):
            min_rc = rcuts[total_force_error[j, :].argmin()]
            rc_lbl = (
                r"$\alpha a_{ws} = "
                + "{:.2f}$".format(alphas[j])
                + r" min @ $r_c = "
                + "{:.2f}".format(min_rc)
                + r" a_{\rm ws}$"
            )
            ax[0].plot(rcuts, total_force_error[j, :], ls=lns, label=rc_lbl)

            min_a = alphas[total_force_error[:, i].argmin()]
            a_lbl = (
                r"$r_c = {:.2f}".format(rcuts[i])
                + " a_{ws}$"
                + r" min @ $\alpha_{\rm min} a_{ws} = "
                + "{:.2f}$".format(min_a)
            )
            ax[1].plot(alphas, total_force_error[:, i], ls=lns, label=a_lbl)

        ax[0].set(ylabel=r"$\Delta F^{approx}_{tot}$", xlabel=r"$r_c/a_{ws}$", yscale="log")
        ax[1].set(xlabel=r"$\alpha \; a_{ws}$", yscale="log")

        if chosen_rcut is not None and chosen_alpha is not None:
            ax[0].axvline(chosen_rcut, ls="--", c="k")
            ax[1].axvline(chosen_alpha, ls="--", c="k")

        ax[0].axhline(self.potential.force_error, ls="--", c="k", label="Actual Force Error")
        ax[1].axhline(self.potential.force_error, ls="--", c="k", label="Actual Force Error")

        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            ax[0].axvline(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws, c="r", label=r"$L/2$")

        for a in ax:
            a.grid(True, alpha=0.3)
            a.legend(loc="best")

        fig.suptitle(
            r"Parameters  $N = {}, \quad M = {}, \quad p = {}, \quad \kappa = {:.2f}$".format(
                self.parameters.total_num_ptcls,
                chosen_mesh if chosen_mesh is not None else self.potential.pppm_mesh[0],
                chosen_cao if chosen_cao is not None else self.potential.pppm_cao[0],
                self.parameters.a_ws / self.potential.screening_length,
            )
        )
        fig.savefig(join(fig_path, "LinePlot_ForceError_" + self.io.job_id + ".png"))

    def make_pppm_line_plot_interactive(
        self, total_force_error, rcuts, alphas, chosen_alpha=None, chosen_rcut=None, chosen_mesh=None, chosen_cao=None
    ):
        """
        Create interactive line plots of the total force error approximation.

        Parameters
        ----------
        total_force_error: numpy.ndarray
            Force error matrix.
        rcuts: numpy.ndarray
            Cut off distances.
        alphas: numpy.ndarray
            Ewald parameters.
        chosen_alpha: float, optional
            Chosen Ewald parameter.
        chosen_rcut: float, optional
            Chosen cut off radius.
        chosen_mesh: int, optional
            Chosen mesh size.
        chosen_cao: int, optional
            Chosen Spline order per box length.
        """

        # Create subplots
        fig = make_subplots(
            rows=1, cols=2, subplot_titles=("Force Error vs r<sub>c</sub>", "Force Error vs α"), horizontal_spacing=0.12
        )

        # Get MSU colors
        msu_colors = get_msu_colors()

        # Line styles
        dash_styles = ["dot", "dash", "solid", "dashdot", "longdash"]
        # Indexes is quantiles of the alphas and rcuts arrays
        r_indexes = quantile(arange(len(rcuts)), [0.3, 0.4, 0.5, 0.6, 0.7], method="nearest").astype(int)
        a_indexes = quantile(arange(len(alphas)), [0.3, 0.4, 0.5, 0.6, 0.7], method="nearest").astype(int)

        if chosen_alpha is not None:
            a_index = (abs(alphas - chosen_alpha)).argmin()
            a_indexes[2] = a_index
        if chosen_rcut is not None:
            r_index = (abs(rcuts - chosen_rcut)).argmin()
            r_indexes[2] = r_index

        # Left plot: Force error vs r_c for different alpha values
        for idx, (i, j, color, dash) in enumerate(zip(r_indexes, a_indexes, msu_colors[:5], dash_styles)):
            rc_lbl = f"αa<sub>ws</sub> = {alphas[j]:.2f}"  # min @ r<sub>c</sub> = {min_rc:.2f} a<sub>ws</sub>"

            fig.add_trace(
                go.Scatter(
                    x=rcuts,
                    y=total_force_error[j, :],
                    mode="lines",
                    name=rc_lbl,
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate="r<sub>c</sub>/a<sub>ws</sub>: %{x:.4e}<br>ΔF: %{y:.4e}<extra></extra>",
                    legendgroup=f"group{idx}",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        # Right plot: Force error vs alpha for different r_c values
        for idx, (i, j, color, dash) in enumerate(zip(r_indexes, a_indexes, msu_colors[:5], dash_styles)):
            a_lbl = f"r<sub>c</sub> = {rcuts[i]:.2f} a<sub>ws</sub>"  # min @ α<sub>min</sub>a<sub>ws</sub> = {min_a:.2f}"

            fig.add_trace(
                go.Scatter(
                    x=alphas,
                    y=total_force_error[:, i],
                    mode="lines",
                    name=a_lbl,
                    line=dict(color=color, dash=dash, width=2),
                    hovertemplate="αa<sub>ws</sub>: %{x:.4e}<br>ΔF: %{y:.4e}<extra></extra>",
                    legendgroup=f"group{idx}",
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

        # Add reference lines
        if chosen_rcut is not None and chosen_alpha is not None:
            fig.add_vline(
                x=chosen_rcut, line_dash="dash", line_color="black", row=1, col=1, annotation_text="chosen r<sub>c</sub>"
            )
            fig.add_vline(x=chosen_alpha, line_dash="dash", line_color="black", row=1, col=2, annotation_text="chosen α")

        # fig.add_hline(y=self.potential.force_error, line_dash="dash", line_color="black",
        #             row=1, col=1, annotation_text="Actual Force Error")
        # fig.add_hline(y=self.potential.force_error, line_dash="dash", line_color="black",
        #             row=1, col=2, annotation_text="Actual Force Error")

        # Add L/2 reference line if applicable
        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            l_half = 0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws
            fig.add_vline(x=l_half, line_color="red", row=1, col=1, annotation_text="L/2", annotation_position="top")

        # Update axes
        fig.update_xaxes(title_text="r<sub>c</sub>/a<sub>ws</sub>", row=1, col=1)
        fig.update_xaxes(
            title_text="α a<sub>ws</sub>",
            row=1,
            col=2,
        )
        fig.update_yaxes(title_text="ΔF<sub>tot</sub><sup>approx</sup>", type="log", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=2)

        # Update layout with MSUstyle template
        title_text = (
            f"Parameters  N = {self.parameters.total_num_ptcls}, "
            f"M = {chosen_mesh if chosen_mesh is not None else self.potential.pppm_mesh[0]}, "
            f"p = {chosen_cao if chosen_cao is not None else self.potential.pppm_cao[0]}, "
            f"κ = {self.parameters.a_ws / self.potential.screening_length:.2f}"
        )

        fig.update_layout(
            template="MSUstyle",  # Use MSUstyle template
            title_text=title_text,
            height=600,
            width=1400,
            hovermode="closest",
            showlegend=True,
            legend=dict(x=1.05, y=1, xanchor="left", yanchor="top"),
        )
        return fig

    def make_pppm_color_map_interactive(
        self, total_force_error, rcuts, alphas, chosen_alpha=None, chosen_rcut=None, chosen_mesh=None, chosen_cao=None
    ):
        """
        Create an interactive color map of the total force error approximation.

        Parameters
        ----------
        total_force_error: numpy.ndarray
            Force error matrix.
        rcuts: numpy.ndarray
            Cut off distances.
        alphas: numpy.ndarray
            Ewald parameters.
        chosen_alpha: float, optional
            Chosen Ewald parameter.
        chosen_rcut: float, optional
            Chosen cut off radius.
        chosen_mesh: int, optional
            Chosen mesh size.
        chosen_cao: int, optional
            Chosen Spline order per box length.
        """

        # Create figure
        fig = go.Figure()

        # Add heatmap with log scale
        fig.add_trace(
            go.Heatmap(
                x=alphas,
                y=rcuts,
                z=log10(total_force_error + 1e-20).T,
                colorscale="Viridis",
                colorbar=dict(
                    title=dict(text="log<sub>10</sub>ΔF<sub>tot</sub><sup>approx</sup>(r<sub>c</sub>,α)", side="right"),
                    exponentformat="e",
                    tickformat=".2e",
                ),
                # hovertemplate='α a<sub>ws</sub>: %{x:.2f}<br>r<sub>c</sub>/a<sub>ws</sub>: %{y:.2f}<br>ΔF: %{z:.2e}<extra></extra>',
                showlegend=False,
            )
        )

        # Add contour lines
        fig.add_trace(
            go.Contour(
                x=alphas,
                y=rcuts,
                z=log10(total_force_error + 1e-20).T,
                showscale=False,
                contours=dict(showlabels=True, labelfont=dict(size=12, color="white"), coloring="none"),
                line=dict(color="white", width=2),
                ncontours=10,
                hoverinfo="skip",
                showlegend=False,
            )
        )

        # Add chosen point
        if chosen_alpha is not None and chosen_rcut is not None:
            fig.add_trace(
                go.Scatter(
                    x=[chosen_alpha],
                    y=[chosen_rcut],
                    mode="markers",
                    marker=dict(size=15, color="black", symbol="circle", line=dict(width=2, color="white")),
                    # name='Chosen parameters',
                    # hovertemplate='Chosen: α=%{x:.2f}, r<sub>c</sub>=%{y:.2f}<extra></extra>'
                    showlegend=False,
                )
            )

        # Add L/2 reference line if applicable
        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            l_half = 0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws
            fig.add_hline(y=l_half, line_color="red", line_width=2, annotation_text="L/2", annotation_position="right")

        # Update layout with MSUstyle template
        title_text = (
            f"Parameters  N = {self.parameters.total_num_ptcls}, "
            f"M = {chosen_mesh if chosen_mesh is not None else self.potential.pppm_mesh[0]}, "
            f"p = {chosen_cao if chosen_cao is not None else self.potential.pppm_cao[0]}, "
            f"κ = {self.parameters.a_ws / self.potential.screening_length:.2f}"
        )

        fig.update_layout(
            template="MSUstyle",  # Use MSUstyle template
            title_text=title_text,
            xaxis_title="α a<sub>ws</sub>",
            yaxis_title="r<sub>c</sub>/a<sub>ws</sub>",
            height=700,
            width=900,
            hovermode="closest",
        )

        return fig

    def make_pppm_color_map(
        self, total_force_error, rcuts, alphas, chosen_alpha=None, chosen_rcut=None, chosen_mesh=None, chosen_cao=None
    ):
        """
        Plot a color map of the total force error approximation.

        Parameters
        ----------
        total_force_error: numpy.ndarray
            Force error matrix.

        rcuts: numpy.ndarray
            Cut off distances.

        alphas: numpy.ndarray
            Ewald parameters.

        chosen_alpha: float
            Chosen Ewald parameter.

        chosen_rcut: float
            Chosen cut off radius.
        chosen_mesh: int
            Chosen mesh size.
        chosen_cao: int
            Chosen Spline order per box length.
        """
        # Plot the results
        fig_path = self.pppm_plots_dir

        if chosen_rcut is None:
            chosen_rcut = self.potential.rc / self.potential.a_ws
        if chosen_alpha is None:
            chosen_alpha = self.potential.pppm_alpha_ewald * self.potential.a_ws
        if chosen_mesh is None:
            chosen_mesh = self.potential.pppm_mesh[0]
        if chosen_cao is None:
            chosen_cao = self.potential.pppm_cao[0]

        r_mesh, a_mesh = meshgrid(rcuts, alphas)
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        CS = ax.pcolormesh(a_mesh, r_mesh, total_force_error, shading="auto", norm=LogNorm())
        CS2 = ax.contour(a_mesh, r_mesh, total_force_error, levels=10, colors="w", norm=LogNorm())
        ax.clabel(CS2, fmt="%1.0e", colors="w")

        ax.scatter(chosen_alpha, chosen_rcut, s=200, c="k")

        if rcuts[-1] * self.parameters.a_ws > 0.5 * self.parameters.box_lengths.min():
            ax.axhline(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws, c="r", label=r"$L/2$")
        # ax.tick_parameters(labelsize=fsz)
        ax.set_xlabel(r"$\alpha \;a_{ws}$")
        ax.set_ylabel(r"$r_c/a_{ws}$")
        ax.set_title(
            r"Parameters  $N = {}, \quad M = {}, \quad p = {}, \quad \kappa = {:.2f}$".format(
                self.parameters.total_num_ptcls,
                chosen_mesh,
                chosen_cao,
                self.parameters.a_ws / self.potential.screening_length,
            )
        )
        clb = fig.colorbar(CS)
        clb.set_label(r"$\Delta F^{approx}_{tot}(r_c,\alpha)$", va="bottom", rotation=270)
        fig.tight_layout()
        fig.savefig(join(fig_path, "ClrMap_ForceError_" + self.io.job_id + ".png"))

    def make_timing_plots(self, data_df: DataFrame = None):
        """
        Makes a figure with three subplots of the CPU times vs PPPM parameters.\n
        The first plot is PP acc time vs the number of cells at different Mesh sizes.\n
        The second plot is the PM acc time vs the mesh size at different charge assignment orders.\n
        The third plot is the time for the calculation of the optimal green's function
        for different charge asssignment orders.

        Parameters
        ----------

        data_df : pandas.DataFrame, Optional
            Timing study data. If `None` it will look for previously saved data, otherwise it will run
            :meth:`sarkas.processes.PreProcess.timing_study_calculation` to calculate the data. Default is `None`.

        """

        if not data_df:
            try:
                data_df = read_csv(
                    join(self.io.directory_tree["preprocessing"]["path"], f"TimingStudy_data_{self.io.job_id}.csv"),
                    index_col=False,
                )
                self.dataframe = data_df
            except FileNotFoundError:
                print(f"I could not find the data from the timing study. Running the timing study now.")
                self.timing_study_calculation()
        else:
            data_df = self.dataframe

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        scatterplot(data=data_df, x="pp_cells", y="pp_acc_time [s]", hue="M_x", s=100, palette="viridis", ax=ax[0])

        scatterplot(data=data_df, x="M_x", y="pm_acc_time [s]", hue="pppm_cao_x", s=150, palette="viridis", ax=ax[1])

        scatterplot(data=data_df, x="M_x", y="G_k time [s]", hue="pppm_cao_x", s=150, palette="viridis", ax=ax[2])
        # ax[0].legend(ncol = 2)
        ax[0].set(yscale="log", xlabel="LCL Cells", ylabel="PP Time [s]")
        ax[1].set(yscale="log", xlabel="Mesh", ylabel="PM Time [s]")
        ax[2].set(yscale="log", xlabel="Mesh", ylabel="Green Function Time [s]")
        ax[1].set_xscale("log", base=2)
        ax[2].set_xscale("log", base=2)
        fig_path = self.pppm_plots_dir
        fig.savefig(join(fig_path, f"PPPM_Times_{self.io.job_id}.png"))

        msg = f"\nFigures can be found in {self.pppm_plots_dir}"
        self.io.write_to_logger(msg)

    def make_force_v_timing_plot(self, data_df: DataFrame = None):
        """Make contour maps of the force error and total acc time as functions of LCL cells and PM meshes for each
        charge assignment order sequence.

        Parameters
        ----------
        data_df : pandas.DataFrame, Optional
            Timing study data. If `None` it will look for previously saved data, otherwise it will run
            :meth:`sarkas.processes.PreProcess.timing_study_calculation` to calculate the data. Default is `None`.

        """
        from scipy.interpolate import griddata

        fig_path = self.pppm_plots_dir
        if not data_df:
            try:
                data_df = read_csv(
                    join(self.io.directory_tree["preprocessing"]["path"], f"TimingStudy_data_{self.io.job_id}.csv"),
                    index_col=False,
                )
                self.dataframe = data_df.copy()
            except FileNotFoundError:
                print(f"I could not find the data from the timing study. Running the timing study now.")
                self.timing_study_calculation()
        else:
            data_df = self.dataframe.copy(deep=True)

        # Plot the results
        for _, cao in enumerate(self.pm_caos):
            mask = self.dataframe["pppm_cao_x"] == cao
            df = data_df[mask][
                ["M_x", "pp_cells", "force error [measured]", "pp_acc_time [s]", "pm_acc_time [s]", "tot_acc_time [s]"]
            ]

            # 2D-arrays from DataFrame
            n_meshes = len(df["M_x"].unique())
            x1 = logspace(log2(df["M_x"].min()), log2(df["M_x"].max()), 5 * n_meshes, base=2)
            n_cells = len(df["pp_cells"].unique())
            y1 = linspace(df["pp_cells"].min(), df["pp_cells"].max(), 5 * n_cells)

            m_mesh, c_mesh = meshgrid(x1, y1)

            # Interpolate unstructured D-dimensional data.
            tot_time_map = griddata((df["M_x"], df["pp_cells"]), df["tot_acc_time [s]"], (m_mesh, c_mesh))
            force_error_map = griddata((df["M_x"], df["pp_cells"]), df["force error [measured]"], (m_mesh, c_mesh))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
            if force_error_map.min() == 0.0:
                minv = 1e-120
            else:
                minv = force_error_map.min()

            maxt = force_error_map.max()
            nlvl = 12
            lvls = logspace(log10(minv), log10(maxt), nlvl)

            luxmap = get_cmap("viridis", nlvl)
            luxnorm = LogNorm(vmin=minv, vmax=maxt)
            CS = ax1.contourf(m_mesh, c_mesh, force_error_map, levels=lvls, cmap=luxmap, norm=luxnorm)
            clb = fig.colorbar(ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax1)
            clb.set_label(r"Force Error [$Q^2/ a_{\rm ws}^2$]", rotation=270, va="bottom")
            CS2 = ax1.contour(CS, colors="w")
            ax1.clabel(CS2, fmt="%1.0e", colors="w")

            if cao == self.potential.pppm_cao[0]:
                input_Nc = int(self.potential.box_lengths[0] / self.potential.rc)
                ax1.scatter(self.potential.pppm_mesh[0], input_Nc, s=200, c="w")

            ax1.set_xscale("log", base=2)
            ax1.set(xlabel="Mesh size", ylabel=r"LCL Cells", title=f"Force Error Map @ cao = {cao}")

            # Timing Plot
            maxt = tot_time_map.max()
            mint = tot_time_map.min()
            # nlvl = 13
            lvls = logspace(log10(mint), log10(maxt), nlvl)
            luxmap = get_cmap("viridis", nlvl)
            luxnorm = LogNorm(vmin=minv, vmax=maxt)

            CS = ax2.contourf(m_mesh, c_mesh, tot_time_map, levels=lvls, cmap=luxmap)
            CS2 = ax2.contour(CS, colors="w", levels=lvls)
            ax2.clabel(CS2, fmt="%.2e", colors="w")
            # fig.colorbar(, ax = ax2)
            clb = fig.colorbar(ScalarMappable(norm=luxnorm, cmap=luxmap), ax=ax2)
            clb.set_label("CPU Time [s]", rotation=270, va="bottom")
            if cao == self.potential.pppm_cao[0]:
                input_Nc = int(self.potential.box_lengths[0] / self.potential.rc)
                ax2.scatter(self.potential.pppm_mesh[0], input_Nc, s=200, c="k")

            ax2.set_xscale("log", base=2)
            ax2.set(xlabel="Mesh size", title=f"Timing Map @ cao = {cao}")
            fig.savefig(join(fig_path, f"ForceErrorMap_v_Timing_cao_{cao}_{self.io.job_id}.png"))

    def postproc_estimates(self):
        # POST- PROCESSING
        self.io.postprocess_info(self, observable="header")
        # Header of process
        process_title = f"{'PostProcessing':^80}"
        msg = f"{'':*^80}\n {process_title} \n{'':*^80}"
        print_to_logger(msg, self.parameters.log_file, self.parameters.verbose)

        for _, obs_class in self.observables_dict.items():
            obs_class.setup(self.parameters)
            msg += self.rdf.pretty_print_msg()

        print_to_logger(msg, self.parameters.log_file, self.parameters.verbose)

    def make_pppm_plots_dir(self):
        self.pppm_plots_dir = join(self.io.directory_tree["preprocessing"]["path"], "PPPM_Plots")

        if not exists(self.pppm_plots_dir):
            mkdir(self.pppm_plots_dir)

    def pppm_approximation(self, rcuts=None, alphas=None, rlims=None, alims=None):
        """
        Calculate the force error for a PPPM simulation using analytical approximations.\n
        Plot the force error in the parameter space.

        Parameters
        ----------
        rcuts: array-like
            Array of cutoff radii used in the PPPM approximation.
        alphas: array-like
            Array of Ewald splitting parameters used in the PPPM approximation.
        rlims: tuple
            Limits for the cutoff radius axis in the plots (min, max).
        alims: tuple
            Limits for the Ewald parameter axis in the plots (min, max).

        """

        self.make_pppm_plots_dir()

        # Calculate Force error from analytic approximation given in Dharuman et al. J Chem Phys 2017
        total_force_error, _, _, rcuts, alphas = self.analytical_approx_pppm(
            rcuts=rcuts, alphas=alphas, rlims=rlims, alims=alims
        )

        chosen_alpha = self.potential.pppm_alpha_ewald * self.parameters.a_ws
        chosen_rcut = self.potential.rc / self.parameters.a_ws
        chosen_mesh = self.potential.pppm_mesh[0]
        chosen_cao = self.potential.pppm_cao[0]

        # Color Map
        self.make_pppm_color_map(
            total_force_error=total_force_error,
            rcuts=rcuts,
            alphas=alphas,
            chosen_alpha=chosen_alpha,
            chosen_rcut=chosen_rcut,
            chosen_mesh=chosen_mesh,
            chosen_cao=chosen_cao,
        )

        # Line Plot
        self.make_pppm_line_plot(total_force_error, rcuts, alphas, chosen_alpha, chosen_rcut, chosen_mesh, chosen_cao)

        msg = f"\nFigures can be found in {self.pppm_plots_dir}"

        self.io.write_to_logger(msg)

    def remove_preproc_dumps(self):
        # Delete dumps created during the estimation runs
        for npz in listdir(self.io.eq_dump_dir):
            os_remove(join(self.io.eq_dump_dir, npz))

        for npz in listdir(self.io.prod_dump_dir):
            os_remove(join(self.io.prod_dump_dir, npz))

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            os_remove(self.io.mag_energy_filename)
            # Remove dumps
            for npz in listdir(self.io.mag_dump_dir):
                os_remove(join(self.io.mag_dump_dir, npz))

    def run(
        self,
        loops: int = 10,
        timing: bool = True,
        timing_study: bool = False,
        pppm_estimate: bool = False,
        pppm_estimate_args: dict = {},
        postprocessing: bool = False,
        remove: bool = False,
    ):
        """
        Estimate the time of the simulation and best parameters if wanted.

        Parameters
        ----------
        loops : int
            Number of loops over which to average the acceleration calculation. Default = 10.

        timing : bool
            Flag for estimating simulation times. Default =True.

        timing_study : bool
            Flag for estimating time for simulation parameters.

        pppm_estimate : bool
            Flag for showing the force error plots in case of pppm algorithm.

        postprocessing : bool
            Flag for calculating Post processing parameters.

        remove : bool
            Flag for removing energy files and dumps created during times estimation. Default = False.

        """

        # Clean everything
        plt.close("all")

        if timing:
            self.time_n_space_estimates(loops=loops)

        if remove:
            self.remove_preproc_dumps()

        if pppm_estimate:
            self.pppm_approximation(**pppm_estimate_args)

        if timing_study:
            self.timing_study_calculation()
            self.make_timing_plots()
            self.make_force_v_timing_plot()
            print(f"\nFigures can be found in {self.pppm_plots_dir}")

        if postprocessing:
            self.postproc_estimates()

    def time_acceleration(self, loops: int = 11):
        """
        Run loops number of acceleration calculations for timing estimate.


        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 11.

        """

        if self.potential.linked_list_on:
            self.pp_acc_time = zeros(loops)
            for i in trange(loops, desc="PP acceleration timer", disable=not self.parameters.verbose):
                self.timer.start()
                self.potential.update_linked_list(self.particles)
                self.pp_acc_time[i] = self.timer.stop()

            # Calculate the mean excluding the first value because that time include numba compilation time
            pp_mean_time = self.timer.time_division(self.pp_acc_time[1:].mean())

            self.io.preprocess_timing("PP", pp_mean_time, loops)

        # PM acceleration
        if self.potential.pppm_on:
            self.pm_acc_time = zeros(loops)
            for i in trange(loops, desc="PM acceleration timer", disable=not self.parameters.verbose):
                self.timer.start()
                self.potential.update_pm(self.particles)
                self.pm_acc_time[i] = self.timer.stop()
            pm_mean_time = self.timer.time_division(self.pm_acc_time[1:].mean())
            self.io.preprocess_timing("PM", pm_mean_time, loops)

        if self.potential.method == "fmm":
            self.fmm_acc_time = zeros(loops)

            for i in range(loops):
                self.timer.start()
                self.integrator.update_accelerations(self.particles)
                self.fmm_acc_time[i] = self.timer.stop()
            fmm_mean_time = self.timer.time_division(self.fmm_acc_time[:].mean())
            self.io.preprocess_timing("FMM", fmm_mean_time, loops)

    def time_evolution_loop(self, loops: int = 11):
        """Run several loops of the equilibration and production phase to estimate the total time of the simulation.

        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 11.

        """

        msg = f"\nRunning {loops} steps for each phase to estimate simulation times\n"
        self.io.write_to_logger(msg)

        # Run few equilibration steps to estimate the equilibration time
        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            self.integrator.update = self.integrator.type_setup(self.integrator.equilibration_type)
            self.io.open_h5md_file(phase="equilibration")
            self.timer.start()
            self.evolve("equilibration", self.integrator.thermalization, 0, loops, self.parameters.eq_dump_step)
            self.io.close_h5md_file()
            # Print the average equilibration & production times
            self.eq_mean_time = self.timer.stop() / loops
            self.io.preprocess_timing("Equilibration", self.timer.time_division(self.eq_mean_time), loops)

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            self.integrator.update = self.integrator.type_setup(self.integrator.magnetization_type)
            self.io.open_h5md_file(phase="magnetization")
            self.timer.start()
            self.evolve("magnetization", self.integrator.thermalization, 0, loops, self.parameters.mag_dump_step)
            self.io.close_h5md_file()
            self.mag_mean_time = self.timer.stop() / loops
            # Print the average equilibration & production times
            self.io.preprocess_timing("Magnetization", self.timer.time_division(self.mag_mean_time), loops)

        # Run few production steps to estimate the equilibration time
        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)
        self.potential_measure = True
        self.io.open_h5md_file(phase="production")
        self.timer.start()
        self.evolve("production", False, 0, loops, self.parameters.prod_dump_step)
        self.io.close_h5md_file()
        self.prod_mean_time = self.timer.stop() / loops
        self.io.preprocess_timing("Production", self.timer.time_division(self.prod_mean_time), loops)

        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            # Print the estimate for the full run
            eq_prediction = self.eq_mean_time * self.parameters.equilibration_steps
            self.io.time_stamp("Equilibration", self.timer.time_division(eq_prediction))
        else:
            eq_prediction = 0.0

        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            mag_prediction = self.mag_mean_time * self.parameters.magnetization_steps
            self.io.time_stamp("Magnetization", self.timer.time_division(mag_prediction))
            eq_prediction += mag_prediction

        prod_prediction = self.prod_mean_time * self.parameters.production_steps
        self.io.time_stamp("Production", self.timer.time_division(prod_prediction))

        tot_time = eq_prediction + prod_prediction
        self.io.time_stamp("Total Run", self.timer.time_division(tot_time))

    def time_n_space_estimates(self, loops: int = 10):
        """Estimate simulation times and space

        Parameters
        ----------
        loops: int
            Number of simulation steps to run. Default = 10.

        """

        if loops:
            loops += 1

        self.io.preprocess_timing("header", [0, 0, 0, 0, 0, 0], 0)
        if self.potential.pppm_on:
            green_time = self.timer.time_division(self.green_function_timer())
            self.io.preprocess_timing("GF", green_time, 0)

        self.time_acceleration(loops)

        self.time_evolution_loop(loops)

        self.directory_sizes()

    def timing_study_calculation(
        self, target_error=1e-5, pp_cells=None, pm_meshes=None, pm_caos=None, method="brute_force", **kwargs
    ):
        """
        Estimate optimal PPPM parameters balancing accuracy and performance.

        Parameters
        ----------
        target_error : float, optional
            Target force error tolerance. If provided, will find fastest configuration meeting this error. Default is 1e-5.
        pp_cells : numpy.ndarray, optional
            Array of cells for PP calculations. If None uses the attribute :attr:`PreProcess.pp_cells`.
        pm_meshes : numpy.ndarray, optional
            Array of mesh sizes for PM calculations. If None uses the attribute :attr:`PreProcess.pm_meshes`.
        pm_caos : numpy.ndarray, optional
            Array of charge assignment orders. If None uses the attribute :attr:`PreProcess.pm_caos`.
        method : str, optional
            Method for parameter optimization: "brute_force" or "automated". Default is "brute_force".
        **kwargs
            Additional keyword arguments for parameter optimization.

        Returns
        -------
        dict
            Dictionary containing the optimal parameters and Pareto-optimal configurations.

        Notes
        -----
        User-provided parameters are saved as attributes (self.user_pp_cells,
        self.user_pm_meshes, self.user_pm_caos) and are respected without modification.
        """
        # Setup directories for outputs
        self.pppm_plots_dir = join(self.io.directory_tree["preprocessing"]["path"], "PPPM_Plots")
        if not exists(self.pppm_plots_dir):
            mkdir(self.pppm_plots_dir)

        msg = "\n\n{:=^70} \n".format(f" PPPM Parameter Optimization ({method}) ")
        self.io.write_to_logger(msg)

        # Store original values to restore later
        self.input_rc = self.potential.rc
        self.input_mesh = self.potential.pppm_mesh.copy()
        self.input_alpha = self.potential.pppm_alpha_ewald
        self.input_cao = self.potential.pppm_cao.copy()
        self.input_aliases = self.potential.pppm_aliases.copy()

        # Calculate maximum allowed cells based on minimum particle separation
        max_cells = int(0.5 * self.parameters.box_lengths.min() / self.parameters.a_ws)

        # Rescaling constant for PP force error calculation
        rescaling_constant = (
            sqrt(self.potential.total_num_ptcls) * self.potential.a_ws**2 / sqrt(self.potential.pbox_volume)
        )

        if method.lower() == "automated":
            return self._automated_parameter_selection(target_error, rescaling_constant, max_cells)
        elif method.lower() == "bayesian":
            return self._bayesian_parameter_selection(
                target_error=target_error,
                pp_cells=pp_cells,
                pm_meshes=pm_meshes,
                pm_caos=pm_caos,
                warm_start_from=kwargs.pop("warm_start_from", None),
                **kwargs,
            )
        else:
            return self._brute_force_parameter_selection(pp_cells, pm_meshes, pm_caos, target_error, max_cells)

    def _brute_force_parameter_selection(
        self, pp_cells=None, pm_meshes=None, pm_caos=None, target_error=None, max_cells=None
    ):
        """
        Perform brute force parameter sweep to find optimal PPPM parameters.

        Parameters are the same as timing_study_calculation.
        """
        # Save user inputs in self attributes for reference
        if pp_cells is not None:
            self.user_pp_cells = pp_cells.copy() if hasattr(pp_cells, "copy") else pp_cells
        if pm_meshes is not None:
            self.user_pm_meshes = pm_meshes.copy() if hasattr(pm_meshes, "copy") else pm_meshes
        if pm_caos is not None:
            self.user_pm_caos = pm_caos.copy() if hasattr(pm_caos, "copy") else pm_caos

        # Use provided parameters or defaults
        if pp_cells is None:
            # If no user input, calculate based on max_cells, but respect original defaults
            if max_cells is not None and max_cells > self.pp_cells[-1]:
                pp_cells = arange(3, max_cells, dtype=int)
            else:
                pp_cells = self.pp_cells
        # User input is respected - we don't modify it based on max_cells

        if pm_meshes is None:
            pm_meshes = self.pm_meshes
        if pm_caos is None:
            pm_caos = self.pm_caos

        # Data collection for all parameter combinations
        data = []

        # Progress tracking for all parameter combinations
        total_combinations = len(pm_meshes) * len(pm_caos) * len(pp_cells)
        progress = tqdm(
            total=total_combinations, desc="Testing PPPM parameter combinations", disable=not self.parameters.verbose
        )

        # Start the parameter sweep
        for _, m in enumerate(pm_meshes):
            # Setup PM params
            self.potential.pppm_mesh = full(3, m, dtype=int)
            self.potential.pppm_alpha_ewald = 0.3 * m / self.potential.box_lengths.min()
            self.potential.pppm_h_array = self.potential.box_lengths / self.potential.pppm_mesh

            for _, cao in enumerate(pm_caos):
                self.potential.pppm_cao = full(3, cao, dtype=int)

                # Update potential parameters
                self.potential.pot_update_params(self.potential, self.species)

                # Calculate Green's function and PM error
                green_time = self.green_function_timer()

                # Measure PM acceleration time (average of 3 runs)
                pm_acc_time = 0.0
                for it in range(3):
                    self.timer.start()
                    self.potential.update_pm(self.particles)
                    pm_acc_time += self.timer.stop() / 3.0

                # For each cutoff radius option
                for _, cell in enumerate(pp_cells):
                    # Update progress bar
                    progress.update(1)

                    # Set cutoff radius based on cell size
                    self.potential.rc = self.potential.box_lengths.min() / cell

                    # Calculate PPPM error approximation
                    self.potential.calculate_force_error()

                    # Measure PP acceleration time (average of 3 runs)
                    pp_acc_time = 0.0
                    for it in range(3):
                        self.timer.start()
                        self.potential.update_linked_list(self.particles)
                        pp_acc_time += self.timer.stop() / 3.0

                    # Total acceleration time
                    total_acc_time = pp_acc_time + pm_acc_time

                    # Error metrics
                    pp_pm_ratio = self.potential.pppm_pp_err / self.potential.pppm_pm_err

                    # Store all the data
                    data_row = [
                        cell,
                        self.potential.rc,
                        self.potential.pppm_alpha_ewald,
                        self.potential.pppm_cao[0],
                        self.potential.pppm_cao[1],
                        self.potential.pppm_cao[2],
                        self.potential.pppm_mesh[0],
                        self.potential.pppm_mesh[1],
                        self.potential.pppm_mesh[2],
                        self.potential.pppm_mesh.prod(),
                        self.potential.pppm_h_array[0],
                        self.potential.pppm_h_array[1],
                        self.potential.pppm_h_array[2],
                        self.potential.pppm_h_array.prod(),
                        self.potential.pppm_h_array[0] * self.potential.pppm_alpha_ewald,
                        self.potential.pppm_h_array[1] * self.potential.pppm_alpha_ewald,
                        self.potential.pppm_h_array[2] * self.potential.pppm_alpha_ewald,
                        self.potential.pppm_h_array.prod() * self.potential.pppm_alpha_ewald**3,
                        green_time * 1.0e-9,
                        pp_acc_time * 1.0e-9,
                        pm_acc_time * 1.0e-9,
                        total_acc_time * 1.0e-9,
                        self.potential.pppm_pp_err,
                        self.potential.pppm_pm_err,
                        self.potential.force_error,
                        self.potential.pppm_pm_err_approx,
                        self.potential.force_error_approx,
                        pp_pm_ratio,  # Added PP/PM error ratio
                    ]
                    data.append(data_row)

        # Close progress bar
        progress.close()

        # Create DataFrame with all results
        column_names = [
            "pp_cells",
            "r_cut",
            "pppm_alpha_ewald",
            "pppm_cao_x",
            "pppm_cao_y",
            "pppm_cao_z",
            "M_x",
            "M_y",
            "M_z",
            "Mesh volume",
            "h_x",
            "h_y",
            "h_z",
            "h_M volume",
            "h_x alpha",
            "h_y alpha",
            "h_z alpha",
            "h_M a_ws^3",
            "G_k time [s]",
            "pp_acc_time [s]",
            "pm_acc_time [s]",
            "tot_acc_time [s]",
            "pppm_pp_error [measured]",
            "pppm_pm_error [measured]",
            "force error [measured]",
            "pppm_pm_error [approx]",
            "force error [approx]",
            "pp_pm_error_ratio",  # Added PP/PM error ratio
        ]

        self.dataframe = DataFrame(data, columns=column_names)
        csv_location = join(self.io.directory_tree["preprocessing"]["path"], f"TimingStudy_data_{self.io.job_id}.csv")
        self.dataframe.to_csv(csv_location, index=False)

        # Find and save Pareto-optimal configurations
        pareto_points, best_point = self.find_pareto_optimal_configs(target_error=target_error)

        # Run the pppm_estimate for the best parameters
        self.potential.rc = best_point["r_cut"]
        self.potential.pppm_mesh = best_point[["M_x", "M_y", "M_z"]].values.astype(int)
        self.potential.pppm_alpha_ewald = best_point["pppm_alpha_ewald"]
        self.potential.pppm_cao = best_point[["pppm_cao_x", "pppm_cao_y", "pppm_cao_z"]].values.astype(int)
        self.potential.estimate_parameters = False
        self.pppm_approximation()

        # Reset to original values
        self.potential.rc = self.input_rc
        self.potential.pppm_mesh = self.input_mesh.copy()
        self.potential.pppm_alpha_ewald = self.input_alpha
        self.potential.pppm_cao = self.input_cao.copy()
        # Set up potential with original parameters
        self.potential.estimate_parameters = False
        self.potential.setup(self.parameters, self.species)

        # Report file locations
        msg = (
            f"\nResults saved to:\n"
            f"  Full parameter sweep data: {csv_location}\n"
            f"  Pareto-optimal configurations: {join(self.io.directory_tree['preprocessing']['path'], f'Pareto_optimal_PPPM_{self.io.job_id}.csv')}\n"
            f"  Visualizations: {self.pppm_plots_dir}"
        )
        if self.parameters.verbose:
            print(msg)
        self.io.write_to_logger(msg)

    def _automated_parameter_selection(self, target_error=1e-5, rescaling_constant=None, max_cells=None):
        """
        Perform automated parameter optimization using a directed search approach.

        Instead of testing all combinations, this method uses iterative refinement and
        theoretical relationships to quickly converge on optimal parameters.

        Parameters are the same as timing_study_calculation.
        """
        from scipy.optimize import minimize

        self.io.write_to_logger(f"\nRunning automated parameter optimization (target error: {target_error:.2e})")

        # Define parameter bounds
        cao_bounds = (1, 7)
        mesh_bounds = (8, 256)
        alpha_factor_bounds = (0.2, 0.5)  # Alpha typically = factor * mesh / box_length
        rc_factor_bounds = (0.4, 2.0)  # rc typically = box_length / (factor * mesh)

        # Initialize data collection
        data = []

        # First, determine optimal charge assignment order (CAO)
        # CAO primarily affects accuracy vs setup cost of PM
        cao_options = [3, 5, 7]
        cao_results = []

        for cao in cao_options:
            # Use a medium mesh for testing
            mesh = 32
            self.potential.pppm_mesh = full(3, mesh, dtype=int)
            self.potential.pppm_cao = full(3, cao, dtype=int)
            self.potential.pppm_alpha_ewald = 0.3 * mesh / self.potential.box_lengths.min()
            self.potential.pppm_h_array = self.potential.box_lengths / self.potential.pppm_mesh

            # Measure setup time (Green's function calculation)
            self.potential.pot_update_params(self.potential, self.species)

            green_time = self.green_function_timer()

            # Measure PM time
            pm_acc_time = 0.0
            for it in range(3):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_acc_time += self.timer.stop() / 3.0

            # Store results
            cao_results.append(
                {
                    "cao": cao,
                    "green_time": green_time * 1.0e-9,
                    "pm_time": pm_acc_time * 1.0e-9,
                    "pm_error": self.potential.pppm_pm_err,
                }
            )

        # Find best CAO based on error/time tradeoff
        for result in cao_results:
            result["score"] = result["pm_error"] * result["pm_time"]

        best_cao_result = min(cao_results, key=lambda x: x["score"])
        best_cao = best_cao_result["cao"]

        msg = f"\nSelected optimal CAO: {best_cao}"

        self.io.write_to_logger(msg)

        # Now define the objective function for the optimizer
        def objective_function(params):
            """
            Objective function for parameter optimization.

            Parameters
            ----------
            params : array-like
                [mesh_size, alpha_factor, rc_factor]

            Returns
            -------
            float
                Weighted combination of time and error, or penalty if error exceeds target.
            """
            mesh_size, alpha_factor, rc_factor = params

            # Convert parameters to actual values
            mesh = int(mesh_size)  # Round to nearest integer
            if mesh < mesh_bounds[0]:
                mesh = mesh_bounds[0]
            if mesh > mesh_bounds[1]:
                mesh = mesh_bounds[1]

            alpha = alpha_factor * mesh / self.potential.box_lengths.min()
            rc = self.potential.box_lengths.min() / (rc_factor * mesh)

            # Set parameters in potential
            self.potential.pppm_mesh = full(3, mesh, dtype=int)
            self.potential.pppm_alpha_ewald = alpha
            self.potential.pppm_cao = full(3, best_cao, dtype=int)
            self.potential.rc = rc
            self.potential.pppm_h_array = self.potential.box_lengths / self.potential.pppm_mesh

            # Update potential
            self.potential.pot_update_params(self.potential, self.species)
            green_time = self.green_function_timer() * 1.0e-9

            # Calculate error
            pp_err = force_error_analytic_pp(
                self.potential.type,
                self.potential.rc,
                self.potential.screening_length,
                self.potential.pppm_alpha_ewald,
                rescaling_constant,
            )

            # Calculate total force error
            total_err = sqrt(pp_err**2 + self.potential.pppm_pm_err**2)

            # Measure performance
            pm_acc_time = 0.0
            for it in range(3):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_acc_time += self.timer.stop() / 3.0
            pm_acc_time *= 1.0e-9

            pp_acc_time = 0.0
            for it in range(3):
                self.timer.start()
                self.potential.update_linked_list(self.particles)
                pp_acc_time += self.timer.stop() / 3.0
            pp_acc_time *= 1.0e-9

            total_time = pm_acc_time + pp_acc_time

            # Store the data for this evaluation
            data_row = [
                int(self.potential.box_lengths.min() / rc),  # pp_cells
                rc,
                alpha,
                best_cao,
                best_cao,
                best_cao,
                mesh,
                mesh,
                mesh,
                mesh**3,
                self.potential.pppm_h_array[0],
                self.potential.pppm_h_array[1],
                self.potential.pppm_h_array[2],
                self.potential.pppm_h_array.prod(),
                self.potential.pppm_h_array[0] * alpha,
                self.potential.pppm_h_array[1] * alpha,
                self.potential.pppm_h_array[2] * alpha,
                self.potential.pppm_h_array.prod() * alpha**3,
                green_time,
                pp_acc_time,
                pm_acc_time,
                total_time,
                pp_err,
                self.potential.pppm_pm_err,
                total_err,
                pp_err / self.potential.pppm_pm_err,  # PP/PM error ratio
            ]
            data.append(data_row)

            # Return objective value
            if total_err <= target_error:
                # If we meet the error target, minimize time
                return total_time
            else:
                # If we don't meet the error target, heavily penalize
                return total_time + 1000 * (total_err / target_error - 1)

        # Initial guess: balanced configuration
        initial_guess = [32, 0.3, 5.0]  # [mesh_size, alpha_factor, rc_factor]

        # Set up bounds
        bounds = [mesh_bounds, alpha_factor_bounds, rc_factor_bounds]

        # Run the optimization
        if self.parameters.verbose:
            print("\nOptimizing mesh, alpha, and rc parameters...")

        result = minimize(objective_function, initial_guess, method="L-BFGS-B", bounds=bounds, options={"maxiter": 20})

        # Get optimized parameters
        opt_mesh, opt_alpha_factor, opt_rc_factor = result.x

        # Convert to actual values
        opt_mesh = int(round(opt_mesh))
        if opt_mesh < mesh_bounds[0]:
            opt_mesh = mesh_bounds[0]
        if opt_mesh > mesh_bounds[1]:
            opt_mesh = mesh_bounds[1]

        opt_alpha = opt_alpha_factor * opt_mesh / self.potential.box_lengths.min()
        opt_rc = self.potential.box_lengths.min() / (opt_rc_factor * opt_mesh)

        # Set optimal parameters and measure final performance
        self.potential.pppm_mesh = full(3, opt_mesh, dtype=int)
        self.potential.pppm_alpha_ewald = opt_alpha
        self.potential.pppm_cao = full(3, best_cao, dtype=int)
        self.potential.rc = opt_rc
        self.potential.pppm_h_array = self.potential.box_lengths / self.potential.pppm_mesh

        # Update potential
        self.potential.pot_update_params(self.potential, self.species)
        green_time = self.green_function_timer() * 1.0e-9

        # Calculate error
        pp_err = force_error_analytic_pp(
            self.potential.type,
            self.potential.rc,
            self.potential.screening_length,
            self.potential.pppm_alpha_ewald,
            rescaling_constant,
        )

        # Calculate total force error
        total_err = sqrt(pp_err**2 + self.potential.pppm_pm_err**2)

        # Measure final performance
        pm_acc_time = 0.0
        for it in range(3):
            self.timer.start()
            self.potential.update_pm(self.particles)
            pm_acc_time += self.timer.stop() / 3.0
        pm_acc_time *= 1.0e-9

        pp_acc_time = 0.0
        for it in range(3):
            self.timer.start()
            self.potential.update_linked_list(self.particles)
            pp_acc_time += self.timer.stop() / 3.0
        pp_acc_time *= 1.0e-9

        total_time = pm_acc_time + pp_acc_time

        # Create DataFrame with all results
        column_names = [
            "pp_cells",
            "r_cut",
            "pppm_alpha_ewald",
            "pppm_cao_x",
            "pppm_cao_y",
            "pppm_cao_z",
            "M_x",
            "M_y",
            "M_z",
            "Mesh volume",
            "h_x",
            "h_y",
            "h_z",
            "h_M volume",
            "h_x alpha",
            "h_y alpha",
            "h_z alpha",
            "h_M a_ws^3",
            "G_k time [s]",
            "pp_acc_time [s]",
            "pm_acc_time [s]",
            "tot_acc_time [s]",
            "pppm_pp_error [measured]",
            "pppm_pm_error [measured]",
            "force error [measured]",
            "pp_pm_error_ratio",  # Added PP/PM error ratio
        ]

        self.dataframe = DataFrame(data, columns=column_names)
        csv_location = join(self.io.directory_tree["preprocessing"]["path"], f"AutomatedPPPM_data_{self.io.job_id}.csv")
        self.dataframe.to_csv(csv_location, index=False)

        # Reset to original values
        self.potential.rc = self.input_rc
        self.potential.pppm_mesh = self.input_mesh.copy()
        self.potential.pppm_alpha_ewald = self.input_alpha
        self.potential.pppm_cao = self.input_cao.copy()
        # Set up potential with original parameters
        self.potential.estimate_parameters = False
        self.potential.setup(self.parameters, self.species)

        # Report optimal configuration
        msg = (
            f"\nOPTIMAL PPPM CONFIGURATION (AUTOMATED):\n"
            f"  Mesh: {opt_mesh} | CAO: {best_cao} | rc: {opt_rc:.4e}\n"
            f"  Ewald alpha: {opt_alpha:.4e} | Force Error: {total_err:.4e}\n"
            f"  PP Time: {pp_acc_time:.4e} s | PM Time: {pm_acc_time:.4e} s\n"
            f"  Total Time: {total_time:.4e} s"
        )
        if self.parameters.verbose:
            print(msg)
        self.io.write_to_logger(msg)

        # Report file locations
        msg = f"\nResults saved to:\n" f"  Optimization data: {csv_location}\n" f"  Visualizations: {self.pppm_plots_dir}"
        if self.parameters.verbose:
            print(msg)
        self.io.write_to_logger(msg)

    def find_pareto_optimal_configs(self, configuration_df=None, target_error=1e-5, show_plot=True):
        """
        Find the Pareto-optimal configurations (those where error or time cannot
        be improved without worsening the other).

        Parameters
        ----------
        configuration_df : pandas.DataFrame, optional
            DataFrame containing the configuration data. If None, uses self.dataframe.

        target_error : float
            Target force error tolerance. Default is 1e-5.

        show_plot : bool
            Whether to show the Pareto frontier plot. Default is True.
            If False, only returns the Pareto points without plotting.

        Returns
        -------
        list
            List of Pareto-optimal parameter configurations
        """
        if configuration_df is None:
            configuration_df = self.dataframe

        pareto_points = []
        for _, row in configuration_df.iterrows():
            # Check if this point is dominated by any other point
            dominated = False
            for _, other_row in configuration_df.iterrows():
                # Point i is dominated by point j if j has better (lower) time AND error
                # Or if one is equal and the other is better
                if (
                    other_row["tot_acc_time [s]"] <= row["tot_acc_time [s]"]
                    and other_row["force error [measured]"] < row["force error [measured]"]
                    and (
                        other_row["tot_acc_time [s]"] < row["tot_acc_time [s]"]
                        or other_row["force error [measured]"] <= row["force error [measured]"]
                    )
                ):
                    dominated = True
                    break

            if not dominated:
                pareto_points.append(row)

        # Sort by error
        pareto_points = sorted(pareto_points, key=lambda x: x["force error [measured]"])

        # Find absolute best configuration
        if len(pareto_points) > 0:
            # Find the best configuration by finding the points which are less the force_error and then choosing the points with the smallest time
            force_points = [point for point in pareto_points if point["force error [measured]"] <= target_error]
            best_point = min(force_points, key=lambda x: x["tot_acc_time [s]"])

            # Report only the best configuration
            msg = (
                f"\nOPTIMAL PPPM CONFIGURATION:\n"
                f"  Target Error: {target_error:.4e}\n"
                f"  Mesh: {int(best_point['M_x'])} | CAO: {int(best_point['pppm_cao_x'])} | rc: {best_point['r_cut']:.4e}\n"
                f"  Ewald alpha: {best_point['pppm_alpha_ewald']:.4e} | Force Error: {best_point['force error [measured]']:.4e}\n"
                f"  PP Time: {best_point['pp_acc_time [s]']:.4e} s | PM Time: {best_point['pm_acc_time [s]']:.4e} s\n"
                f"  Total Time: {best_point['tot_acc_time [s]']:.4e} s"
            )
            if self.parameters.verbose:
                print(msg)
            self.io.write_to_logger(msg)
        else:
            print("Unable to find a configuration that meets the target error.")
            best_point = None

        # If no Pareto points found, return empty list and None
        if not pareto_points:
            print("No Pareto-optimal configurations found.")
            return None, None

        # Create DataFrame from Pareto points and save to CSV
        pareto_df = DataFrame(pareto_points)
        pareto_csv_path = join(
            self.io.directory_tree["preprocessing"]["path"], f"Pareto_optimal_PPPM_{self.io.job_id}.csv"
        )
        pareto_df.to_csv(pareto_csv_path, index=False)
        self.pareto_points_df = DataFrame(pareto_points)

        return pareto_points, best_point

    def plot_error_vs_performance(self, configuration_df, pareto_points, best_point):
        """
        Generate a Pareto frontier visualization from the configuration DataFrame and Pareto points.

        Parameters
        ----------
        configuration_df : pandas.DataFrame
            DataFrame containing all tested configurations.

        pareto_points : list
            List of Pareto-optimal configurations.

        best_point : dict
            Dictionary containing the best configuration found.

        """
        if configuration_df is None:
            configuration_df = self.dataframe
        # Generate Pareto frontier visualization
        fig, ax = plt.subplots(figsize=(12, 8))

        scatterplot(
            data=configuration_df,
            x="force error [measured]",
            y="tot_acc_time [s]",
            size="pp_cells",
            sizes=(50, 200),  # Adjust size range for better visibility
            style="pppm_cao_x",
            palette="Dark2",  # Use a color palette
            alpha=0.7,
            hue="M_x",
            ax=ax,
        )

        if pareto_points is not None and len(pareto_points) > 0:
            # Sort Pareto points by force error for consistent plotting

            # Highlight Pareto-optimal points
            pareto_errors = [p["force error [measured]"] for p in pareto_points]
            pareto_times = [p["tot_acc_time [s]"] for p in pareto_points]

            ax.plot(
                pareto_errors,
                pareto_times,
                linewidth=2,
                markersize=8,
                alpha=0.4,
                zorder=2,
                markeredgecolor="black",
                markerfacecolor="red",
                linestyle="--",
                marker="o",
                label="Pareto Frontier",
            )

            if best_point is not None:
                ax.scatter(
                    best_point["force error [measured]"],
                    best_point["tot_acc_time [s]"],
                    color="orange",
                    s=200,
                    marker="*",
                    alpha=0.5,
                    label="Best Configuration",
                    zorder=2.5,
                )

        # Set labels and title
        ax.set(
            xscale="log",
            yscale="log",
            xlabel=r"Force Error $[Q^2/a_{ws}^2]$",
            ylabel="Computation Time (s)",
            title="Error vs Performance",
        )

        # Put the legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        fig.tight_layout()
        fig.savefig(join(self.pppm_plots_dir, f"pareto_frontier_{self.io.job_id}.png"))

    def plot_error_balance(self):
        """
        Create visualization showing how balanced the PP and PM errors are
        for different parameter combinations.
        """
        fig, ax = plt.subplots()

        # Group by mesh size
        for mesh in self.dataframe["M_x"].unique():
            subset = self.dataframe[self.dataframe["M_x"] == mesh]

            # For consistent cao
            cao_filter = subset["pppm_cao_x"] == 4  # Choose a representative cao
            if cao_filter.any():
                filtered = subset[cao_filter]

                # Sort by rc
                filtered = filtered.sort_values("r_cut")

                # Plot PP/PM error ratio vs rc
                ax.plot(
                    filtered["r_cut"] / self.parameters.a_ws,  # Normalize rc by a_ws
                    filtered["pp_pm_error_ratio"],
                    label=f"Mesh={int(mesh)}",
                )

                # Find where PP error ≈ PM error (ratio ≈ 1)
                optimal_idx = (filtered["pp_pm_error_ratio"] - 1).abs().idxmin()
                optimal_row = filtered.loc[optimal_idx]

                ax.scatter(optimal_row["r_cut"] / self.parameters.a_ws, optimal_row["pp_pm_error_ratio"], marker="o")

        # Add reference line for balanced errors
        ax.axhline(y=1.0, color="k", linestyle="--", alpha=0.5, label="Balanced PP and PM Errors")

        ax.set(
            xlabel="Cutoff Radius (rc/a_ws)",
            ylabel="PP Error / PM Error Ratio",
            yscale="log",
            title="Error Balance: Optimal rc for Each Mesh Size",
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(join(self.pppm_plots_dir, f"error_balance_{self.io.job_id}.png"))

    def plot_parameter_sensitivity(self):
        """
        Create visualizations showing how sensitive performance and error
        are to each parameter.

        Only creates the visualizations without detailed logging.
        """
        # Get baseline parameters
        try:
            baseline_mesh = self.input_mesh[0]
            baseline_cao = self.input_cao[0]
            baseline_rc = self.input_rc
        except:
            # If no baseline exists, use middle values
            baseline_mesh = self.dataframe["M_x"].median()
            baseline_cao = self.dataframe["pppm_cao_x"].median()
            baseline_rc = self.dataframe["r_cut"].median()

        # Analyze mesh sensitivity
        # Filter data for constant cao and rc (closest to baseline)
        rc_filter = abs(self.dataframe["r_cut"] - baseline_rc) < baseline_rc * 0.1
        cao_filter = self.dataframe["pppm_cao_x"] == baseline_cao
        mesh_sensitivity = self.dataframe[rc_filter & cao_filter].sort_values("M_x")

        if not mesh_sensitivity.empty:
            # Create figure with two subplots
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

            # Plot time sensitivity
            ax[0].plot(mesh_sensitivity["M_x"], mesh_sensitivity["tot_acc_time [s]"], "o-")
            ax[0].set(
                xscale="log",
                yscale="log",
                xlabel="Mesh Size",
                ylabel="Computation Time (s)",
                title="Time Sensitivity to Mesh Size",
            )

            # Set x-ticks to actual mesh sizes
            ax[0].set_xticks(mesh_sensitivity["M_x"].values)
            ax[0].set_xticklabels(mesh_sensitivity["M_x"].values.astype(int))
            ax[0].grid(True, alpha=0.3)

            # Plot error sensitivity
            ax[1].plot(mesh_sensitivity["M_x"], mesh_sensitivity["force error [measured]"], "o-")
            ax[1].set(
                xscale="log",
                yscale="log",
                xlabel="Mesh Size",
                ylabel="Force Error",
                title="Error Sensitivity to Mesh Size",
            )

            # Set x-ticks to actual mesh sizes
            ax[1].set_xticks(mesh_sensitivity["M_x"].values)
            ax[1].set_xticklabels(mesh_sensitivity["M_x"].values.astype(int))
            ax[1].grid(True, alpha=0.3)

            # Adjust layout and save
            fig.tight_layout()
            fig.savefig(join(self.pppm_plots_dir, f"sensitivity_mesh_{self.io.job_id}.png"))


class Simulation(Process):
    """
    Sarkas simulation wrapper. This class manages the entire simulation and its small moving parts.

    Parameters
    ----------
    input_file : str
        Path to the YAML input file.

    """

    def __init__(self, input_file: str = None):
        self.__name__ = "simulation"
        super().__init__(input_file)

    def check_restart(self, phase):
        """
        Check if the simulation is a restart.

        Parameters
        ----------
        phase: str
            Simulation phase, e.g. equilibration, Magnetization, production.

        Returns
        -------
        it_start: int
            Restart step.

        """

        if self.parameters.verbose:
            print(f"\n\n{phase.capitalize():-^70} \n")

        # Check if this is restart
        if self.parameters.load_method[:2] == phase[:2] and self.parameters.load_method[-7:] == "restart":
            it_start = self.parameters.restart_step
        else:
            it_start = 0
            dump_step = self.parameters.eq_dump_step if phase == "equilibration" else self.parameters.prod_dump_step
            self.io.save_timestep_data(it_start, dump_step, self.integrator.dt * it_start, self.particles)
        return it_start

    def equilibrate(self):
        """
        Run the time integrator with the thermostat to evolve the system to its thermodynamics equilibrium state.
        """

        self.io.open_h5md_file(phase="equilibration")
        it_start = self.check_restart(phase="equilibration")
        self.integrator.update = self.integrator.type_setup(self.integrator.equilibration_type)
        # Start timer, equilibrate, and print run time.
        self.timer.start()
        self.evolve(
            "equilibration",
            self.integrator.thermalization,
            it_start,
            self.parameters.equilibration_steps,
            self.parameters.eq_dump_step,
        )
        time_eq = self.timer.stop()
        self.io.close_h5md_file()
        self.io.time_stamp("Equilibration", self.timer.time_division(time_eq))

    def magnetize(self):
        self.io.open_h5md_file(phase="magnetization")
        # Check for magnetization phase
        it_start = self.check_restart(phase="magnetization")
        # Update integrator
        self.integrator.update = self.integrator.type_setup(self.integrator.magnetization_type)
        # Start timer, magnetize, and print run time.
        self.timer.start()
        self.evolve(
            "magnetization",
            self.integrator.thermalization,
            it_start,
            self.parameters.magnetization_steps,
            self.parameters.mag_dump_step,
        )
        time_eq = self.timer.stop()
        self.io.close_h5md_file()
        self.io.time_stamp("Magnetization", self.timer.time_division(time_eq))

    def produce(self):
        self.io.open_h5md_file(phase="production")
        it_start = self.check_restart(phase="production")
        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)
        # Update measurement flag for rdf.
        self.potential.measure = True
        self.timer.start()
        self.evolve("production", False, it_start, self.parameters.production_steps, self.parameters.prod_dump_step)
        time_eq = self.timer.stop()
        self.io.close_h5md_file()
        self.io.time_stamp("Production", self.timer.time_division(time_eq))

    def run(self):
        """Run the simulation."""
        time0 = self.timer.current()

        if self.parameters.equilibration_phase and self.parameters.electrostatic_equilibration:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()
            self.equilibrate()

        if self.parameters.magnetization_phase:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()
            self.magnetize()

        if self.parameters.production_phase:
            if self.parameters.remove_initial_drift:
                self.particles.remove_drift()

            self.produce()

        time_tot = self.timer.current()
        self.io.time_stamp("Total", self.timer.time_division(time_tot - time0))

        self.directory_sizes()

    def adaptive_thermalize(self):
        """
        Run adaptive thermalization using statistical tests to verify equilibration.

        This method alternates between NVT (thermostat) and NVE (microcanonical) cycles
        until statistical tests confirm proper thermalization, or max_cycles is reached.

        The NVT steps are automatically set to production_steps // 2 for reheating,
        while NVE steps (for testing) must be specified in the YAML configuration.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing thermalization statistics and test results

        Examples
        --------
        >>> sim = Simulation(input_file='input.yaml')
        >>> sim.setup(read_yaml=True)
        >>> therm_results = sim.adaptive_thermalize()
        >>> sim.produce()

        Notes
        -----
        YAML configuration:

        .. code-block:: yaml

            Parameters:
            adaptive_thermalization:
                max_cycles: 10 # Optional, default 10
                nve_steps: 50000 # Required for testing thermalization
                nvt_steps: 50000 # Optional, defaults to equilibration_steps
                dump_step: 10 # Optional, defaults to eq_dump_step
                observable: "temperature"
                max_mae: 0.01 # Optional, default 0.01
                adf_significance: 0.05 # Optional, default 0.05
                kpss_significance: 0.05 # Optional, default 0.05
        """
        # Check if adaptive thermalization is configured
        if self.parameters.adaptive_thermalization is None:
            raise AttributeError(
                "Adaptive thermalization not configured. "
                "Add 'adaptive_thermalization' section to Parameters in YAML file. See documentation for details."
            )

        config = self.parameters.adaptive_thermalization

        if "nve_steps" not in config or config["nve_steps"] is None:
            config["nve_steps"] = self.parameters.equilibration_steps

        if "nvt_steps" not in config or config["nvt_steps"] is None:
            config["nvt_steps"] = self.parameters.equilibration_steps

        if "dump_step" not in config or config["dump_step"] is None:
            config["dump_step"] = self.parameters.eq_dump_step

        if "max_cycles" not in config or config["max_cycles"] is None:
            config["max_cycles"] = 10
        if "max_mae" not in config or config["max_mae"] is None:
            config["max_mae"] = 0.01
        if "adf_significance" not in config or config["adf_significance"] is None:
            config["adf_significance"] = 0.05
        if "kpss_significance" not in config or config["kpss_significance"] is None:
            config["kpss_significance"] = 0.05

        # Log configuration
        msg = f"\n{'='*60}"
        msg += f"\nAdaptive Thermalization Configuration:"
        msg += f"\n  Observable: {config['observable']}"
        msg += f"\n  NVT steps: {config['nvt_steps']}"
        msg += f"\n  NVE steps: {config['nve_steps']}"
        msg += f"\n  Dump step: {config['dump_step']}"
        msg += f"\n  Max cycles: {config['max_cycles']}"
        msg += f"\n  Max MAE: {config['max_mae']}"
        msg += f"\n  ADF significance: {config['adf_significance']}"
        msg += f"\n  KPSS significance: {config['kpss_significance']}"
        msg += f"\n{'='*60}\n"
        self.io.write_to_logger(msg)

        # Initialize thermalization data storage
        self._init_thermalization_data_dict(config["observable"])

        # Prepare simulation
        self._prepare_adaptive_thermalization(config)

        # Start timer
        self.timer.start()

        # Run initial NVE phase
        self._run_initial_nve(config["nve_steps"], config["dump_step"], config)

        # Continue with NVT-NVE cycles until thermalized
        cycle_counter = 0
        while not self._thermalization_data["Verdict"][-1] and cycle_counter < config["max_cycles"]:
            self._run_nvt_thermalization_cycle(config["nvt_steps"], config["dump_step"])
            self._run_nve_thermalization_cycle(config["nve_steps"], config["dump_step"], config)
            self._save_thermalization_results()
            cycle_counter += 1

        time_eq = self.timer.stop()
        self.io.close_h5md_file()
        self.io.time_stamp("Adaptive Equilibration", self.timer.time_division(time_eq))

        # Finalize
        self._finalize_adaptive_thermalization(cycle_counter, config)

        return DataFrame(self._thermalization_data)

    def _init_thermalization_data_dict(self, observable_name):
        """Initialize dictionary for storing thermalization results."""
        obs_name = observable_name.replace("_", " ").title()

        self._thermalization_data = {
            "Completed steps": [],
            "NVT start": [],
            "NVT end": [],
            "NVE start": [],
            "NVE end": [],
            "Cycle": [],
            f"Average {obs_name} Deviation": [],
            f"Mean {obs_name}": [],
            f"Std {obs_name}": [],
            f"MAE {obs_name}": [],
            "Linear slope": [],
            "Linear intercept": [],
            "Linear rmse_fit": [],
            "Epsilon": [],
            "ADF Test": [],
            "ADF p-value": [],
            "ADF Critical Value": [],
            "KPSS Test": [],
            "KPSS p-value": [],
            "KPSS Critical Value": [],
            "MK Test": [],
            "MK p-value": [],
            "MK Tau": [],
            "MK h": [],
            "MK Trend": [],
            "Conditions": [],
            "Verdict": [],
        }

        self._therm_step_counter = 0
        self._therm_dump_counter = 0

    def _read_thermalization_observable(self, observable_name, start_dump, end_dump):
        """
        Read observable data from H5MD file for thermalization check.

        Parameters
        ----------
        observable_name : str
            Name of the observable to read
        start_dump : int
            Starting dump index
        end_dump : int
            Ending dump index

        Returns
        -------
        tuple
            (time_data, observable_data) - weighted average across species
        """

        with h5py.File(self.io.h5md_filepath, "r") as file:
            time_data = None
            observable_data = 0.0

            # Compute weighted average across species using concentrations
            for sp_name, concentration in zip(self.parameters.species_names, self.parameters.species_concentrations):
                path_base = f"observables/{sp_name}/{observable_name}"

                if time_data is None:
                    time_data = file[f"{path_base}/time"][start_dump:end_dump]

                species_data = file[f"{path_base}/value"][start_dump:end_dump]
                observable_data += concentration * species_data

        return time_data, observable_data

    def _check_thermalization_statistics(self, observable_name, start_dump, end_dump, stats_config):
        """
        Check if system is thermalized using statistical tests.

        Parameters
        ----------
        observable_name : str
            Name of the observable to check
        start_dump : int
            Starting dump index
        end_dump : int
            Ending dump index
        stats_config : dict
            Configuration for statistical tests
        """
        obs_name = observable_name.replace("_", " ").title()

        # Get target value
        target_value = self.parameters.T_desired

        # Read observable data
        time_data, observable_data = self._read_thermalization_observable(observable_name, start_dump, end_dump)

        # Calculate basic statistics
        mean_obs = observable_data.mean()
        std_obs = observable_data.std()
        relative_deviation = abs(mean_obs - target_value) / target_value
        mae = abs(observable_data - target_value).mean() / target_value

        self._thermalization_data[f"Average {obs_name} Deviation"].append(relative_deviation)
        self._thermalization_data[f"Mean {obs_name}"].append(mean_obs)
        self._thermalization_data[f"Std {obs_name}"].append(std_obs)
        self._thermalization_data[f"MAE {obs_name}"].append(mae)

        # Normalize time
        time_normalized = time_data / self.parameters.total_plasma_frequency

        # Run statistical tests
        test_results = run_thermalization_tests(
            observable_data,
            time_normalized,
            adf_significance=stats_config.get("adf_significance", 0.05),
            kpss_significance=stats_config.get("kpss_significance", 0.05),
        )

        # Check MAE condition
        mae_condition = mae < stats_config.get("max_mae", 0.01)

        # Store results
        self._thermalization_data["Linear intercept"].append(test_results["intercept"])
        self._thermalization_data["Linear slope"].append(test_results["slope"])
        self._thermalization_data["Linear rmse_fit"].append(test_results["rmse"])
        self._thermalization_data["Epsilon"].append(test_results["epsilon"])

        self._thermalization_data["ADF Test"].append(test_results["adf"]["statistic"])
        self._thermalization_data["ADF p-value"].append(test_results["adf"]["pvalue"])
        self._thermalization_data["ADF Critical Value"].append(test_results["adf"]["critical_value"])

        self._thermalization_data["KPSS Test"].append(test_results["kpss"]["statistic"])
        self._thermalization_data["KPSS p-value"].append(test_results["kpss"]["pvalue"])
        self._thermalization_data["KPSS Critical Value"].append(test_results["kpss"]["critical_value"])

        self._thermalization_data["MK Test"].append(test_results["mann_kendall"]["s"])
        self._thermalization_data["MK p-value"].append(test_results["mann_kendall"]["pvalue"])
        self._thermalization_data["MK Tau"].append(test_results["mann_kendall"]["tau"])
        self._thermalization_data["MK h"].append(test_results["mann_kendall"]["h"])
        self._thermalization_data["MK Trend"].append(test_results["mann_kendall"]["trend"])

        # Overall verdict: all statistical tests + MAE condition
        all_conditions = test_results["all_conditions"] + [mae_condition]
        self._thermalization_data["Conditions"].append(all_conditions)
        self._thermalization_data["Verdict"].append(all(all_conditions))

        msg = f"\nThermalization Check Results:"
        msg += f"\n  {obs_name} Mean: {mean_obs:.4f}, Std: {std_obs:.4f}, Rel. Deviation: {relative_deviation:.4e}, MAE: {mae:.4e}"
        # msg += f"\n  Linear Fit - Slope: {test_results['slope']:.4e}, Intercept: {test_results['intercept']:.4f}, RMSE: {test_results['rmse']:.4e}, Epsilon: {test_results['epsilon']:.4e}"
        msg += f"\n  ADF Test - Statistic: {test_results['adf']['statistic']:.4f}, p-value: {test_results['adf']['pvalue']:.4f}, Critical Value: {test_results['adf']['critical_value']:.44f}"
        msg += f"\n  KPSS Test - Statistic: {test_results['kpss']['statistic']:.4f}, p-value: {test_results['kpss']['pvalue']:.4f}, Critical Value: {test_results['kpss']['critical_value']:.4f}"
        msg += f"\n  Mann-Kendall Test - S: {test_results['mann_kendall']['s']}, p-value: {test_results['mann_kendall']['pvalue']:.4f}, Tau: {test_results['mann_kendall']['tau']:.4f}, h: {test_results['mann_kendall']['h']}, Trend: {test_results['mann_kendall']['trend']}"
        msg += f"\n  MAE Condition (< {stats_config.get('max_mae', 0.01)}): {'Passed' if mae_condition else 'Failed'}"
        msg += f"\n  Overall Verdict: {'Thermalized' if all(all_conditions) else 'Not Thermalized'}\n"
        self.io.write_to_logger(msg)

    def _prepare_adaptive_thermalization(self, config):
        """Prepare simulation for adaptive thermalization."""
        self.io.open_h5md_file(phase="equilibration")
        self.potential.measure = True
        # Save the initial configuration
        it_start = 0
        if config.get("restart_step", None) is not None:
            it_start = config["restart_step"]
        self.io.save_timestep_data(it_start, config["dump_step"], self.integrator.dt * it_start, self.particles)

    def _resize_thermalization_h5md(self, new_steps):
        """Resize H5MD file for additional thermalization steps."""
        self.io.close_h5md_file()
        self.parameters.equilibration_steps = new_steps
        self.io.setup_checkpoint(self.parameters, self.particles, phase="equilibration")
        self.io.open_h5md_file(phase="equilibration")
        self.potential.measure = True

    def _run_initial_nve(self, nve_steps, dump_step, config):
        """Run initial NVE phase for thermalization.

        Parameters
        ----------
        nve_steps : int
            Number of NVE steps to run.
        dump_step : int
            Dump step interval.
        config : dict
            Configuration for statistical tests.
        """

        nve_dumps = nve_steps // dump_step

        msg = f"\nRunning initial NVE phase"
        msg += f"  Steps: {nve_steps}, Dumps: {nve_dumps}"
        self.io.write_to_logger(msg)

        self._thermalization_data["NVT start"].append(0)
        self._thermalization_data["NVT end"].append(0)
        self._thermalization_data["NVE start"].append(0)
        self._thermalization_data["NVE end"].append(nve_dumps)
        self._thermalization_data["Cycle"].append(0)

        # Update integrator to NVE
        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)

        # Run NVE
        self.evolve("equilibration", False, self._therm_step_counter, nve_steps, dump_step)

        self.particles.remove_drift()

        # Check thermalization
        self._check_thermalization_statistics(config["observable"], 0, nve_dumps, config)

        # Update counters
        self._therm_step_counter += nve_steps
        self._therm_dump_counter = nve_dumps
        self._thermalization_data["Completed steps"].append(self._therm_step_counter)

        # Save results
        self._save_thermalization_results()

    def _run_nvt_thermalization_cycle(self, nvt_steps, dump_step):
        """Run NVT (thermostat) cycle.

        Parameters
        ----------
        nvt_steps : int
            Number of NVT steps to run.
        dump_step : int
            Dump step interval.
        """

        cycle_num = self._thermalization_data["Cycle"][-1] + 1

        # Log NVT cycle info
        msg = f"\nCycle {cycle_num}: Running NVT phase\n"
        msg += f"  Steps: {nvt_steps}, Dumps: {nvt_steps // dump_step}\n"
        msg += f"  Total equilibration steps so far: {self._therm_step_counter}"
        self.io.write_to_logger(msg)

        # Record starting dump index
        self._thermalization_data["NVT start"].append(self._therm_step_counter // dump_step)

        # Resize H5MD file for additional NVT steps
        new_total_steps = self._therm_step_counter + nvt_steps
        self._resize_thermalization_h5md(new_total_steps)

        # Update integrator to NVT
        self.integrator.update = self.integrator.type_setup(self.integrator.equilibration_type)

        # Run NVT
        self.evolve("equilibration", self.integrator.thermalization, self._therm_step_counter, new_total_steps, dump_step)

        self.particles.remove_drift()

        # Update counters
        self._therm_step_counter = new_total_steps
        self._therm_dump_counter = new_total_steps // dump_step
        self._thermalization_data["NVT end"].append(self._therm_dump_counter)

    def _run_nve_thermalization_cycle(self, nve_steps, dump_step, config):
        """Run NVE (microcanonical) cycle.

        Parameters
        ----------
        nve_steps : int
            Number of NVE steps to run.
        dump_step : int
            Dump step interval.
        config : dict
            Configuration for statistical tests.
        """

        cycle_num = self._thermalization_data["Cycle"][-1] + 1

        msg = f"Cycle {cycle_num}: Running NVE phase\n"
        msg += f"  Steps: {nve_steps}, Dumps: {nve_steps // dump_step}\n"
        self.io.write_to_logger(msg)

        self._thermalization_data["NVE start"].append(self._therm_dump_counter + 1)

        end_nve_steps = self._therm_step_counter + nve_steps

        # Update integrator to NVE
        self.integrator.update = self.integrator.type_setup(self.integrator.production_type)

        # Resize H5MD file
        self._resize_thermalization_h5md(end_nve_steps)

        # Run NVE
        self.evolve("equilibration", False, self._therm_step_counter, end_nve_steps, dump_step)

        self.particles.remove_drift()

        # Calculate ending dump index
        end_nve_dumps = end_nve_steps // dump_step
        self._thermalization_data["NVE end"].append(end_nve_dumps)

        # Check thermalization
        self._check_thermalization_statistics(config["observable"], self._therm_dump_counter + 1, end_nve_dumps, config)

        # Update counters
        self._therm_step_counter += nve_steps
        self._thermalization_data["Completed steps"].append(self._therm_step_counter)

        # Increment cycle counter
        self._thermalization_data["Cycle"].append(cycle_num)

    def _save_thermalization_results(self):
        """Save thermalization results to CSV."""
        output_path = join(self.parameters.directory_tree["simulation"]["path"], f"AdaptiveThermalizationData.csv")
        DataFrame(self._thermalization_data).to_csv(output_path, index=False)

    def _finalize_adaptive_thermalization(self, cycle_counter, config):
        """Finalize adaptive thermalization."""
        obs_name = config["observable"].replace("_", " ").title()
        if self._thermalization_data["Verdict"][-1]:
            msg = f"\n{'='*60}"
            msg += f"\n  System thermalized after {cycle_counter} cycles"
            msg += f"\n  Observable: {obs_name}"
            msg += f"\n  Final MAE: {self._thermalization_data[f'MAE {obs_name}'][-1]:.6f}"
            msg += f"\n{'='*60}\n"
            self.io.write_to_logger(msg)
        else:
            msg = f"\n{'='*60}"
            msg += f"\n  Maximum cycles ({config['max_cycles']}) reached"
            msg += f"\n  Observable: {obs_name}"
            msg += f"\n  Final MAE: {self._thermalization_data[f'MAE {obs_name}'][-1]:.6f}"
            msg += f"\n  Consider adjusting parameters or running more cycles"
            msg += f"\n{'='*60}\n"
            self.io.write_to_logger(msg)
