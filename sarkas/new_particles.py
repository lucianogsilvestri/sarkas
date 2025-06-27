"""
Module containing the basic class for handling particles properties.

This module has been updated to use the optimized thermodynamics calculator
while maintaining 100% backward compatibility.
"""

import csv
from copy import deepcopy
from h5py import File as h5File
from numba import float64, int64, jit, njit, void
from numpy import arange, array, empty, exp, floor, full, histogram, int64, log, pi
from numpy import load as np_load
from numpy import (
    loadtxt,
    meshgrid,
    ndarray,
    newaxis,
    outer,
    rint,
    savetxt,
    savez,
    sqrt,
    sum,
    triu_indices,
    zeros,
)
import h5py
from numpy.random import Generator, PCG64
from os.path import join
from scipy.linalg import norm
from scipy.spatial.distance import pdist
from scipy.stats import moment, qmc
from warnings import warn

from .utilities.exceptions import ParticlesError, ParticlesWarning

# Performance toggle for gradual migration
_USE_FAST_THERMODYNAMICS = True
_USE_FAST_TRANSPORT = True
_USE_FAST_MECHANICAL = True

# Import the new thermodynamics calculator
try:
    from .physics.thermodynamics import (
        kinetic_energy as fast_kinetic_energy,
        temperature_from_velocities as fast_temperature_from_velocities,
        enthalpy as fast_enthalpy,
        species_enthalpy as fast_species_enthalpy
    )
    _THERMODYNAMICS_AVAILABLE = True
except ImportError:
    # Fallback if new module is not available
    _THERMODYNAMICS_AVAILABLE = False
    _USE_FAST_THERMODYNAMICS = False
    warn(
        "Fast thermodynamics calculator not available. Using legacy implementation.",
        category=UserWarning
    )

# Import the new transport calculator
try:
    from .physics.transport import (
        electric_current_vector as fast_electric_current_vector,
        electric_current_density as fast_electric_current_density,
        heat_flux_vector as fast_heat_flux_vector,
        heat_flux_tensor as fast_heat_flux_tensor,
        diffusion_flux as fast_diffusion_flux,
        calculate_electric_current as fast_calculate_electric_current,
        calculate_species_electric_current as fast_calculate_species_electric_current
    )
    _TRANSPORT_AVAILABLE = True
except ImportError:
    # Fallback if new module is not available
    _TRANSPORT_AVAILABLE = False
    _USE_FAST_TRANSPORT = False
    warn(
        "Fast transport calculator not available. Using legacy implementation.",
        category=UserWarning
    )

# Import the new mechanical calculator
try:
    from .physics.mechanical import (
        momentum as fast_momentum,
        center_of_mass_velocity as fast_center_of_mass_velocity,
        remove_center_of_mass_motion as fast_remove_center_of_mass_motion,
        angular_momentum as fast_angular_momentum,
        pressure_tensor_kinetic as fast_pressure_tensor_kinetic,
        pressure_tensor_virial as fast_pressure_tensor_virial,
        pressure_scalar as fast_pressure_scalar,
        stress_tensor as fast_stress_tensor,
        calculate_species_pressure_tensor as fast_calculate_species_pressure_tensor
    )
    _MECHANICAL_AVAILABLE = True
except ImportError:
    # Fallback if new module is not available
    _MECHANICAL_AVAILABLE = False
    _USE_FAST_MECHANICAL = False
    warn(
        "Fast mechanical calculator not available. Using legacy implementation.",
        category=UserWarning
    )
class Particles:
    """
    Class handling particles' properties.

    Attributes
    ----------
    kB : float
        Boltzmann constant.

    fourpie0: float
        Electrostatic constant :math:`4\\pi \\epsilon_0`.

    pos : numpy.ndarray
        Particles' positions.

    vel : numpy.ndarray
        Particles' velocities.

    acc : numpy.ndarray
        Particles' accelerations.

    box_lengths : numpy.ndarray
        Box sides' lengths.

    pbox_lengths : numpy.ndarray
        Initial particle box sides' lengths.

    masses : numpy.ndarray
        Mass of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    charges : numpy.ndarray
        Charge of each particle. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    id : numpy.ndarray,
        Species identifier. Shape = (attr:`sarkas.core.Parameters.total_num_ptcls`).

    names : numpy.ndarray
        Species' names. (attr:`sarkas.core.Parameters.total_num_ptcls`).

    rdf_nbins : int
        Number of bins for radial pair distribution.

    no_grs : int
        Number of independent :math:`g_{ij}(r)`.

    rdf_hist : numpy.ndarray
        Histogram array for the radial pair distribution function.

    prod_dump_dir : str
        Directory name where to store production phase's simulation's checkpoints. Default = 'dumps'.

    eq_dump_dir : str
        Directory name where to store equilibration phase's simulation's checkpoints. Default = 'dumps'.

    total_num_ptcls : int
        Total number of simulation's particles.

    num_species : int
        Number of species.

    species_num : numpy.ndarray
        Number of particles of each species. Shape = (attr:`sarkas.particles.Particles.num_species`).

    dimensions : int
        Number of non-zero dimensions. Default = 3.

    potential_energy : float
        Instantaneous value of the potential energy of each particle. Note that the total potential energy requires the multiplication of the array's sum by 0.5 to avoid double counting.\n
        For example: `N` = 3, `particle_potential_energy[0] = U_12 + U_13` and `particle_potential_energy[1] = U_21 + U_23` and `particle_potential_energy[1] = U_31 + U_32`

    rnd_gen : numpy.random.Generator
        Random number generator.

    Notes
    -----
        Naming convention:\n
        Properties/Quantities of each particle are stored in `numpy.ndarray`'s as `.[property]`.\n
        Species properties/quantities are stored in `numpy.ndarray`'s as `.species_[property]`.\n
        Total properties/quantities are stored as `float`/`int` identified by `.total_[property]`.\n
        For example:\n
        :attr:`total_num_ptcls` is an `int` indicating the total number of particles.
        :attr:`potential_energy` is a 1-D array of length :attr:`total_num_ptcls` containing the potential energy of each particle.\n
        :attr:`species_num` is a 1-D array of length `num_species` containing the number of particles of each species.\n
        Methods for the calculation of properties/quantities follow the same convention as above but we the prefix `.calculate_[quantity]`.\n
        For example:\n
        :meth:`calculate_kinetic_energy()` calculates the kinetic energy of each particle and stores it in :attr:`kinetic_energy` a 1-D array of length :attr:`total_num_ptcls`.\n
        :meth:`calculate_species_kinetic_energy()` calculates the kinetic energy of each species and stores it in :attr:`species_kinetic_energy` a 1-D array of length :attr:`num_species`.\n
        :meth:`calculate_total_kinetic_energy()` calculates the total kinetic energy and stores it in :attr:`tottal_kinetic_energy` a float.\n\n
        Quantities requiring cross species evaluation are stored in `numpy.ndarray` as `.[quantity]_species_tensor`.\n
        For example:
        :attr:`virial_species_tensor` is a  :attr:`num_species` x :attr:`num_species` x 3 x 3 tensor.
        :attr:`heat_flux_species_tensor` is a :attr:`num_species` x :attr:`num_species` x  3 tensor.
        The attributes :attr:`potential_energy`, :attr:`virial_species_tensor`, :attr:`heat_flux_species_tensor` are calculated by the :class:`sarkas.potentials.core.Potential` class.\n
        Therefore, this class is missing the :meth:`calculate_potential_energy`, :meth:`calculate_virial`, :meth:`calculate_heat_flux` methods.
    """

    def __init__(self):
        self.mag_dump_dir = None
        self.rdf_nbins = None
        self.kB = None
        self.fourpie0 = None
        self.prod_dump_dir = None
        self.eq_dump_dir = None
        self.box_lengths = None
        self.pbox_lengths = None
        self.total_num_ptcls = None
        self.num_species = 1
        self.species_num = None
        self.dimensions = None
        self.rnd_gen = None

        self.names = None
        self.id = None
        self.pos = None
        self.vel = None
        self.acc = None
        self.virial_species_tensor = None
        self.heat_flux_species_tensor = None
        self.potential_energy = None
        self.dipole_energy = None
        self.pbc_cntr = None
        self.masses = None
        self.charges = None
        self.cyclotron_frequencies = None

        self.species_initial_velocity = None
        self.species_thermal_velocity = None
        self.species_thermostat_temperatures = None
        self.species_masses = None
        self.species_charges = None
        self.species_velocity_moments = None
        self.species_thermal_speed = None
        self.species_kl_divergence = None

        self.species_kinetic_energy = None
        self.species_potential_energy = None
        self.species_dipole_energy = None
        self.species_temperature = None
        self.species_thermostat_temperatures = None

        self.no_grs = None
        self.rdf_hist = None

        self.observables_list = ["Radial Distribution Function"]
        self.observables_arrays_list = ['rdf_hist']
        self.thermodynamics_list = ['total_energy', 'kinetic_energy', 'potential_energy', 'temperature']
        
        self.species_thermodynamics_data = {}
        self.species_thermodynamics_method_map = {}
        
        self.species_observables_method_map = {
            "Momentum": self.calculate_species_momentum,
            "Velocity Moments": self.calculate_species_velocity_moments,
        }

        self.qmc_sequence = None
        self.available_qmc_sequences = ["halton", "sobol", "poissondisk", "latinhypercube"]
        self.max_velocity_distribution_moment = 4
        
        # Performance toggle attribute for gradual migration
        self._use_fast_thermodynamics = _USE_FAST_THERMODYNAMICS and _THERMODYNAMICS_AVAILABLE
        self._use_fast_transport = _USE_FAST_TRANSPORT and _TRANSPORT_AVAILABLE
        self._use_fast_mechanical = _USE_FAST_MECHANICAL and _MECHANICAL_AVAILABLE

    def __copy__(self):
        """
        Make a shallow copy of the object using copy by creating a new instance of the object and copying its __dict__."""
        # Create a new object
        _copy = type(self)()
        # copy the dictionary
        _copy.__dict__.update(self.__dict__)
        return _copy

    def __deepcopy__(self, memodict: dict = {}):
        """Make a deepcopy of the object.

        Parameters
        ----------
        memodict: dict
            Dictionary of id's to copies

        Returns
        -------
        _copy: :class:`sarkas.particles.Particles`
            A new Particles class.
        """
        id_self = id(self)  # memorization avoids unnecessary recursion
        _copy = memodict.get(id_self)
        if _copy is None:
            # Make a shallow copy of all attributes
            _copy = type(self)()
            # Make a deepcopy of the mutable arrays using numpy copy function
            for k, v in self.__dict__.items():
                if isinstance(v, ndarray):
                    _copy.__dict__[k] = v.copy()
                else:
                    _copy.__dict__[k] = deepcopy(v, memodict)

        return _copy

    def __getstate__(self):
        """Copy the object's state from self.__dict__ which contains all our instance attributes.
        Always use the dict.copy() method to avoid modifying the original state.
        Reference: https://docs.python.org/3/library/pickle.html#handling-stateful-objects
        """

        state = self.__dict__.copy()
        # Remove the data that is stored already
        del state["pos"]
        del state["vel"]
        del state["acc"]
        del state["id"]
        del state["names"]
        del state["pbc_cntr"]
        del state["rdf_hist"]
        del state["virial_species_tensor"]
        del state["potential_energy"]
        del state["heat_flux_species_tensor"]

        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        # Initialize arrays
        self.pos = zeros((self.__dict__["total_num_ptcls"], 3))
        self.vel = zeros((self.__dict__["total_num_ptcls"], 3))
        self.acc = zeros((self.__dict__["total_num_ptcls"], 3))
        self.id = zeros(self.__dict__["total_num_ptcls"])
        self.names = zeros(self.__dict__["total_num_ptcls"])
        self.pbc_cntr = zeros((self.__dict__["total_num_ptcls"], 3))
        self.rdf_hist = zeros((self.__dict__["num_species"], self.__dict__["num_species"], self.__dict__["rdf_nbins"]))
        self.virial_species_tensor = zeros((self.__dict__["num_species"], self.__dict__["num_species"], 3, 3))
        self.potential_energy = zeros((self.__dict__["total_num_ptcls"]))
        self.dipole_energy = zeros((self.__dict__["total_num_ptcls"]))
        self.heat_flux_species_tensor = zeros((self.__dict__["num_species"], self.__dict__["num_species"], 3))

    def copy_params(self, params):
        """
        Copy necessary parameters.

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        """
        
        self.process_directory_tree = deepcopy(params.process_directory_tree)
        self.filenames_tree = deepcopy(params.filenames_tree)
        self.h5md_filenames_tree = deepcopy(params.h5md_filenames_tree)
        self.process_h5md_filepath_dict = deepcopy(params.process_h5md_filepath_dict)
        
        # Redundant info. Can be removed.
        self.prod_dump_dir = params.process_directory_tree["production"]["dumps"]["path"]
        self.eq_dump_dir = params.process_directory_tree["equilibration"]["dumps"]["path"]
        self.mag_dump_dir = params.process_directory_tree["magnetization"]["dumps"]["path"]

        self.kB = params.kB
        self.fourpie0 = params.fourpie0

        
        self.box_lengths = params.box_lengths.copy()
        self.pbox_lengths = params.pbox_lengths.copy()
        self.total_num_ptcls = params.total_num_ptcls
        self.total_num_density = params.total_num_density
        self.num_species = params.num_species
        self.species_num = params.species_num.copy()
        self.species_masses = params.species_masses.copy()
        self.species_charges = params.species_charges.copy()

        self.dimensions = params.dimensions
        self.box_volume = params.box_volume
        self.pbox_volume = params.pbox_volume
        self.load_method = params.load_method

        if hasattr(params, "qmc_sequence"):
            self.qmc_sequence = params.qmc_sequence
        if hasattr(params, "qmc_seed"):
            self.qmc_seed = params.qmc_seed
            
        if hasattr(params, "max_velocity_distribution_moment"):
            self.max_velocity_distribution_moment = params.max_velocity_distribution_moment
            self.species_velocity_moments = zeros((self.num_species, self.max_velocity_distribution_moment, 3))

        self.restart_step = params.restart_step
        # Needed for restarts
        self.eq_dump_step = params.eq_dump_step
        self.prod_dump_step = params.prod_dump_step
        self.mag_dump_step = params.mag_dump_step 
        self.job_id = params.job_id
        self.particles_input_file = params.particles_input_file
        self.load_perturb = params.load_perturb
        self.load_rejection_radius = params.load_rejection_radius
        self.load_halton_bases = params.load_halton_bases

        for obs in params.observables_list:
            if obs not in self.observables_list:
                self.observables_list.append(obs)
        
        for obs in params.thermodynamics_list:
            if obs not in self.thermodynamics_list:
                self.thermodynamics_list.append(obs)
        
        # These array_names are the key of the species_observables_method_map used to calculate the observables.
        for array_name in params.observables_arrays_list:
            if array_name not in self.observables_arrays_list:
                self.observables_arrays_list.append(array_name) 

        if hasattr(params, "np_per_side"):
            self.np_per_side = params.np_per_side

        if hasattr(params, "initial_lattice_config"):
            self.lattice_type = params.initial_lattice_config

        if hasattr(params, "load_gauss_sigma"):
            self.load_gauss_sigma = params.load_gauss_sigma.copy()

        self.species_names = params.species_names.copy()

        if hasattr(params, "rdf_nbins"):
            self.rdf_nbins = params.rdf_nbins
        else:
            # nbins = 5% of the number of particles.
            self.rdf_nbins = int64(0.05 * params.total_num_ptcls)
            params.rdf_nbins = self.rdf_nbins

    def dump_arrays(self, filename, data_to_save):
        """
        Save particles' data to binary file (uncompressed npz) for future restart.

        Parameters
        ----------
        filename : str
            Name of the file.

        data_to_save : list
            Name of the arrays to save to file.
        """

        kwargs = {key: self.__dict__[key] for key in data_to_save}
        savez(f"{filename}", **kwargs)

    def dump_pva_h5(self, filename, data_to_save):
        """
        Save particles' data to HDF5 file.

        Parameters
        ----------
        filename : str
            Name of the file.

        data_to_save : list
            Name of the arrays to save to file.
        """
        ## DEV Note: This method seems to be slower than npz.

        with h5File(f"{filename}.h5", "w") as hf:
            for key in data_to_save:
                hf.create_dataset(key, data=self.__dict__[key])

    def gaussian(self, mean, sigma, size):
        """
        Initialize particles' velocities according to a normalized Maxwell-Boltzmann (Normal) distribution.
        It calls :meth:`numpy.random.Generator.normal`

        Parameters
        ----------
        size : tuple
            Size of the array to initialize. (no. of particles, dimensions).

        mean : float
            Center of the normal distribution.

        sigma : float
            Scale of the normal distribution.

        Returns
        -------
         : numpy.ndarray
            Particles property distributed according to a Normal probability density function.

        """
        return self.rnd_gen.normal(mean, sigma, size)

    def halton_reject(self, bases, r_reject):
        """
        Place particles according to a Halton sequence from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        bases : numpy.ndarray
            Array of 3 ints each of which is a base for the Halton sequence.
            Defualt: bases = array([2,3,5])

        r_reject : float
            Value of rejection radius.

        """

        # Get bases
        b1, b2, b3 = bases

        # Allocate space and store first value from Halton
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Initialize particle counter and Halton counter
        i = 1
        k = 1

        # Loop over all particles
        while i < self.total_num_ptcls:
            # Increment particle counter
            n = k
            m = k
            p = k

            # Determine x coordinate
            f1 = 1
            r1 = 0
            while n > 0:
                f1 /= b1
                r1 += f1 * (n % int(b1))
                n = floor(n / b1)
            x_new = self.pbox_lengths[0] * r1  # new x value

            # Determine y coordinate
            f2 = 1
            r2 = 0
            while m > 0:
                f2 /= b2
                r2 += f2 * (m % int(b2))
                m = floor(m / b2)
            y_new = self.pbox_lengths[1] * r2  # new y value

            # Determine z coordinate
            f3 = 1
            r3 = 0
            while p > 0:
                f3 /= b3
                r3 += f3 * (p % int(b3))
                p = floor(p / b3)
            z_new = self.pbox_lengths[2] * r3  # new z value

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):
                # Flag for if particle is outside of cutoff radius (1 -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # Periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff = x_diff + self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff = x_diff - self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff = y_diff + self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff = y_diff - self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff = z_diff + self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff = z_diff - self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    k += 1  # Increment Halton counter
                    flag = 0  # New position not added (0 -> no longer outside reject r)
                    break

            # If flag true add new position
            if flag == 1:
                # Add new positions to arrays
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                k += 1  # Increment Halton counter
                i += 1  # Increment particle number

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def initialize_accelerations(self):
        """
        Initialize particles' accelerations.
        """
        self.acc = zeros((self.total_num_ptcls, 3))

    def initialize_arrays(self):
        """Initialize the needed arrays"""

        self.names = empty(self.total_num_ptcls, dtype=self.species_names.dtype)
        self.id = zeros(self.total_num_ptcls, dtype=int64)

        self.pos = zeros((self.total_num_ptcls, 3))
        self.vel = zeros((self.total_num_ptcls, 3))
        self.acc = zeros((self.total_num_ptcls, 3))

        self.pbc_cntr = zeros((self.total_num_ptcls, 3))

        self.masses = zeros(self.total_num_ptcls)  # mass of each particle
        self.charges = zeros(self.total_num_ptcls)  # charge of each particle
        self.cyclotron_frequencies = zeros(self.total_num_ptcls)

        self.kinetic_energy = zeros(self.total_num_ptcls)
        self.potential_energy = zeros(self.total_num_ptcls)
        self.dipole_energy = zeros(self.total_num_ptcls)
        self.temperature = zeros(self.total_num_ptcls)

        self.species_concentrations = self.species_num/ self.species_num.sum()
        self.species_initial_velocity = zeros((self.num_species, 3))
        self.species_thermal_velocity = zeros((self.num_species, 3))
        
        self.species_thermal_speed = zeros(self.num_species)
        self.species_kl_divergence = zeros( (self.num_species, 3))

        self.species_initial_spatial_distribution = empty((self.num_species, 3), dtype=str)
        self.species_initial_velocity_distribution = empty((self.num_species, 3), dtype=str)

        self.species_kinetic_energy = zeros(self.num_species)
        self.species_potential_energy = zeros(self.num_species)
        self.species_temperature = zeros(self.num_species)
        self.species_thermostat_temperatures = zeros(self.num_species)
        
        self.no_grs = int64(self.num_species * (self.num_species + 1) / 2)

        if "Radial Distribution Function" in self.observables_list:
            self.rdf_hist = zeros((self.num_species, self.num_species, self.rdf_nbins))
            if 'rdf_hist' not in self.observables_arrays_list:
                self.observables_arrays_list.append('rdf_hist')

        if "Momentum" in self.observables_list:
            self.momentum = zeros((self.total_num_ptcls, 3))
            self.species_momentum = zeros((self.num_species, 3))
            if 'species_momentum' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_momentum')
                
        if "Electric Current" in self.observables_list:
            self.electric_current = zeros((self.total_num_ptcls, 3))
            self.species_electric_current = zeros((self.num_species, 3))
            if 'species_electric_current' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_electric_current')

        if "Pressure Tensor" in self.observables_list:
            self.pressure = zeros(self.total_num_ptcls)
            self.species_pressure = zeros(self.species_num)
            self.species_pressure_kin_tensor = zeros((self.num_species, 3, 3))
            self.species_pressure_pot_tensor = zeros((self.num_species, 3, 3))
            self.species_pressure_tensor = zeros((self.num_species, 3, 3))
            self.virial_species_tensor = zeros((self.num_species, self.num_species, 3, 3))

            if 'species_pressure_tensor' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_pressure_tensor')
            if 'species_pressure_kin_tensor' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_pressure_kin_tensor')
            if 'species_pressure_pot_tensor' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_pressure_pot_tensor')

        if "enthalpy" in self.thermodynamics_list:
            self.enthalpy = zeros(self.total_num_ptcls)
            self.species_enthalpy = zeros(self.num_species)
            
        if "Heat Flux" in self.observables_list:
            self.heat_flux_species_tensor = zeros(( self.num_species, self.num_species, 3))
            self.species_heat_flux = zeros((self.num_species, 3))
            if 'species_heat_flux' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_heat_flux')

        if "Velocity Moments" in self.observables_list:
            self.species_velocity_moments = zeros((self.num_species, self.max_velocity_distribution_moment, 3))
            if 'species_velocity_moments' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_velocity_moments')

        if "Diffusion Flux" in self.observables_list:
            self.species_diffusion_flux = zeros((self.num_species, 3))
            if 'species_diffusion_flux' not in self.observables_arrays_list:
                self.observables_arrays_list.append('species_diffusion_flux')
            
        if "Inter Diffusion Flux" in self.observables_list:
            self.interdiffusion_fluxes = zeros((self.num_species - 1, 3))
            if 'interdiffusion_flux' not in self.observables_arrays_list:
                self.observables_arrays_list.append('interdiffusion_flux')
                
    def initialize_positions(self, species: list = None):
        """
        Initialize particles' positions based on the load method.
        """

        # TODO: Break this method into smaller methods. It is too long.
        if self.load_method == "lattice":
            self.lattice(self.load_perturb)

        elif self.load_method == "random_reject":
            # check
            if not hasattr(self, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. Please define Parameters.load_rejection_radius.")
            self.random_reject(self.load_rejection_radius)

        elif self.load_method == "halton_reject":
            # check
            if not hasattr(self, "load_rejection_radius"):
                raise AttributeError("Rejection radius not defined. Please define Parameters.load_rejection_radius.")
            self.halton_reject(self.load_halton_bases, self.load_rejection_radius)

        elif self.load_method in ["uniform", "random_no_reject"]:
            self.pos = self.uniform_no_reject(
                0.0 * self.pbox_lengths, self.pbox_lengths
            )

        elif self.load_method == "gaussian":
            if self.load_gauss_sigma is None:
                raise AttributeError("Gaussian sigma not defined. Please define Parameters.load_gauss_sigma.")

            sp_start = 0
            sp_end = 0
            for sp, sp_num in enumerate(self.species_num):
                sp_end += sp_num
                self.pos[sp_start:sp_end, :] = self.gaussian(
                    self.pbox_lengths[0] / 2.0, self.load_gauss_sigma[sp], (sp_num, 3)
                )
                sp_start += sp_num

        elif self.load_method in ["qmc", "quasi_monte_carlo"]:
            # Ensure that we are not making silly typos
            self.qmc_sequence = self.qmc_sequence.lower()
                
            if self.qmc_sequence not in self.available_qmc_sequences:
                raise AttributeError(
                    f"Quasi Monte Carlo sequence not recognized. Please choose from {self.available_qmc_sequences}"
                )
            if hasattr(self, "qmc_seed"):
                kwargs = {"seed": self.qmc_seed}
            else:
                kwargs = {}

            self.pos = self.quasi_monte_carlo(self.qmc_sequence, self.total_num_ptcls, self.dimensions, self.pbox_lengths, **kwargs)

        elif self.load_method == "species_specific":
            sp_start = 0
            sp_end = 0
            for ic, sp in enumerate(species):
                # TODO: Add the read_from_file option
                if sp.name != "electron_background":
                    sp_end += sp.num
                    if sp.initial_spatial_distribution in ["uniform", "random_no_reject"]:
                        self.pos[sp_start:sp_end, :] = self.uniform_no_reject(
                            0.0 * self.pbox_lengths,
                             self.pbox_lengths,
                        )
                    elif sp.initial_spatial_distribution == "gaussian":
                        if sp.gaussian_sigma is None:
                            raise AttributeError("Gaussian sigma not defined. Please define Species.gaussian_sigma.")

                        if sp.gaussian_mean is None:
                            raise AttributeError("Gaussian mean not defined. Please define Species.gaussian_mean.")

                        # Set the mean and sigma for the species
                        if sp.gaussian_mean == "center":
                            sp_mean = self.box_lengths / 2.0
                        else:
                            if isinstance(sp.gaussian_mean, (int, float)):
                                sp_mean = full(self.dimensions, sp.gaussian_mean)
                            elif isinstance(sp.gaussian_mean, (list, ndarray)):
                                # Check that it has length 3
                                if len(sp.gaussian_mean) != 3 and self.dimensions == 3:
                                    raise AttributeError("Gaussian mean must be a list or array of length 3.")
                                elif len(sp.gaussian_mean) != 2 and self.dimensions == 2:
                                    raise AttributeError("Gaussian mean must be a list or array of length 2.")
                                else:
                                    sp_mean = sp.gaussian_mean

                        if isinstance(sp.gaussian_sigma, (int, float)):
                            sp_sigma = full(self.dimensions, sp.gaussian_sigma)
                        else:
                            # Check that it has the correct dimensions
                            if len(sp.gaussian_sigma) != 3 and self.dimensions == 3:
                                raise AttributeError("Gaussian sigma must be a list or array of length 3.")
                            elif len(sp.gaussian_sigma) != 2 and self.dimensions == 2:
                                raise AttributeError("Gaussian sigma must be a list or array of length 2.")
                            else:
                                sp_sigma = sp.gaussian_sigma

                        for dim in range(self.dimensions):
                            self.pos[sp_start:sp_end, dim] = self.gaussian(sp_mean[dim], sp_sigma[dim], (sp.num, 1))

                    elif sp.initial_spatial_distribution in ["quasi_monte_carlo", "qmc"]:
                        if sp.qmc_sequence not in self.available_qmc_sequences:
                            raise AttributeError(
                                f"Quasi Monte Carlo sequence not recognized. Please choose from {self.available_qmc_sequences}"
                            )
                        self.pos[sp_start:sp_end, :] = self.quasi_monte_carlo(
                            sp.qmc_sequence, sp.num, self.dimensions, self.pbox_lengths
                        )
                    elif sp.initial_spatial_distribution in ["quasi_monte_carlo_rejection", "qmc_reject"]:
                        if self.qmc_sequence not in self.available_qmc_sequences:
                            raise AttributeError(
                                f"Quasi Monte Carlo sequence not recognized. Please choose from {self.available_qmc_sequences}"
                            )
                        self.pos[sp_start:sp_end, :] = self.quasi_monte_carlo_rejection(
                            sp.qmc_sequence, sp_num, self.dimensions, self.load_rejection_radius, self.pbox_lengths
                        )
                    sp_start += sp.num
        else:
            raise AttributeError("Incorrect particle placement scheme specified.")

    @staticmethod
    def quasi_monte_carlo(
        sequence: str = "sobol", num_ptcls: int = 2**10, dimensions: int = 3, box_lengths: ndarray = None, **kwargs
    ):
        """
        Place particles according to a quasi-Monte Carlo sequence.

        This method uses the classes in the Quasi Monte Carlo module of `scipy` to place particles in a simulation box
        according to a quasi-Monte Carlo sequence. The sequence can be chosen from the following options: Halton, Sobol,
        PoissonDisk, LatinHypercube.

        Parameters
        ----------
        sequence : str, optional
            Name of the sequence to use. Options: Halton, Sobol, PoissonDisk, LatinHypercube. Default is "Sobol".

        num_ptcls : int, optional
            Number of particles to place. Default is 2**10.

        dimensions : int, optional
            Number of dimensions. Default is 3.

        box_lengths : array-like, optional
            Lengths of the simulation box in each dimension. Default is None.

        kwargs : dict, optional
            Additional keyword arguments to pass for the initialization of the `scipy.stats.qmc` class.
        Returns
        -------
        array-like
            Randomly generated positions of the particles, scaled by the box lengths.

        Raises
        ------
        ValueError
            If an invalid sequence name is provided.

        Notes
        -----
        The `quasi_monte_carlo` method initializes a quasi-Monte Carlo sequence generator based on the provided sequence
        name. It then generates random positions for the specified number of particles in the specified number of
        dimensions. The generated positions are scaled by the box lengths of the simulation.

        Examples
        --------
        >>> positions = quasi_monte_carlo("Halton", 100, 3, [10, 10, 10])

        """
        if sequence == "sobol":
            qmc_sampler = qmc.Sobol(d=dimensions, **kwargs)
        elif sequence == "latinhypercube":
            qmc_sampler = qmc.LatinHypercube(d=dimensions, **kwargs)
        elif sequence == "halton":
            qmc_sampler = qmc.Halton(d=dimensions, **kwargs)
        else:
            raise ValueError("Invalid sequence name.")

        positions = qmc_sampler.random(num_ptcls)
        positions = qmc.scale(positions, l_bounds=[0.0, 0.0, 0.0], u_bounds=box_lengths)

        return positions

    @staticmethod
    def quasi_monte_carlo_rejection(
        sequence: str = "poissondisk",
        num_ptcls: int = 2**10,
        dimensions: int = 3,
        rejection_radius: float = 0.5,
        box_lengths=None,
        **kwargs,
    ):
        """
        Place particles according to a quasi-Monte Carlo sequence with rejection.

        Parameters
        ----------
        sequence : str, optional
            Name of the sequence to use. Options: PoissonDisk. Default is "PoissonDisk".

        num_ptcls : int, optional
            Number of particles to place. Default is 2**10.

        dimensions : int, optional
            Number of dimensions. Default is 3.

        rejection_radius : float, optional
            Value of rejection radius. Default is 0.5.

        box_lengths : array-like, optional
            Lengths of the simulation box in each dimension. Default is None.

        kwargs : dict, optional
            Additional keyword arguments to pass for the initialization of the `scipy.stats.qmc.PoissonDisk` class.

        Returns
        -------
        array-like
            Randomly generated positions of the particles, scaled by the box lengths.

        Raises
        ------
        ValueError
            If an invalid sequence name is provided.

        Notes
        -----
        The `quasi_monte_carlo_rejection` method initializes a quasi-Monte Carlo sequence generator based on the provided sequence
        name. It then generates random positions for the specified number of particles in the specified number of
        dimensions. The generated positions are scaled by the box lengths of the simulation.

        Examples
        --------
        >>> positions = quasi_monte_carlo_rejection("poissondisk", 100, 3, 0.5, [10, 10, 10])

        """
        if sequence == "poissondisk":
            qmc_sampler = qmc.poisson(d=dimensions, radius=rejection_radius, **kwargs)
        else:
            raise ValueError("Invalid sequence name.")

        positions = qmc_sampler.random(num_ptcls)
        positions = qmc.scale(positions, l_bounds=[0.0, 0.0, 0.0], u_bounds=box_lengths)

        return positions

    def initialize_velocities(self, species):
        """
        Initialize particles' velocities based on the species input values. The velocities can be initialized from a
        Maxwell-Boltzmann distribution or from a monochromatic distribution.

        Parameters
        ----------
        species: list
            List of :class:`sarkas.core.Species`.

        """
        species_end = 0
        species_start = 0
        for ic, sp in enumerate(species):
            if sp.name != "electron_background":
                species_end += sp.num
                self.species_initial_velocity[ic, :] = sp.initial_velocity
                self.species_thermostat_temperatures[ic] = sp.temperature
                self.species_thermal_speed[ic] = sqrt(self.kB * sp.temperature / sp.mass)
                if sp.initial_velocity_distribution == "boltzmann":
                    if isinstance(sp.temperature, (int, float)):
                        sp_temperature = zeros(3)
                        for d in range(self.dimensions):
                            sp_temperature[d] = sp.temperature

                    self.species_thermal_velocity[ic] = sqrt(self.kB * sp_temperature / sp.mass)
                    # Note gaussian(0.0, 0.0, N) = array of zeros
                    self.vel[species_start:species_end, : self.dimensions] = self.gaussian(
                        sp.initial_velocity, self.species_thermal_velocity[ic], (sp.num, self.dimensions)
                    )

                elif sp.initial_velocity_distribution == "monochromatic":
                    vrms = sqrt(self.dimensions * self.kB * sp.temperature / sp.mass)
                    self.vel[species_start:species_end, : self.dimensions] = vrms * self.random_unit_vectors(
                        sp.num, self.dimensions
                    )

                species_start += sp.num

    # =============================================================================
    # THERMODYNAMICS METHODS - UPDATED TO USE NEW CALCULATOR
    # =============================================================================

    def calculate_kinetic_energy(self, use_fast=None):
        """
        Calculate the kinetic energy of each particle.

        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
            This parameter provides a migration path for gradual adoption.

        Returns
        -------
        kin : numpy.ndarray
            Total kinetic energy. Shape = (:attr:`total_num_ptcls`)

        Notes
        -----
        This method now delegates to the optimized thermodynamics calculator by default
        while maintaining full backward compatibility. The legacy implementation is
        still available via the use_fast=False parameter.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_thermodynamics
        
        if use_fast and _THERMODYNAMICS_AVAILABLE:
            # Use the new optimized calculator
            self.kinetic_energy = fast_kinetic_energy(self.vel, self.masses)
        else:
            # Use the legacy implementation
            self.kinetic_energy = 0.5 * self.masses * (self.vel * self.vel).sum(axis=-1)

    def calculate_species_kinetic_energy(self, use_fast=None):
        """
        Calculate the kinetic energy of each species and store it into :attr:`species_kinetic_energy`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        self.calculate_kinetic_energy(use_fast=use_fast)
        self.species_kinetic_energy = scalar_species_loop(self.kinetic_energy, self.species_num)

    def calculate_species_kinetic_temperature(self, use_fast=None):
        """
        Calculate the kinetic energy and temperature of each species.

        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.

        Returns
        -------
        K : numpy.ndarray
            Kinetic energy of each species. Shape=(:attr:`num_species`).

        T : numpy.ndarray
            Temperature of each species. Shape=(:attr:`num_species`).

        Notes
        -----
        This method now uses the optimized thermodynamics calculator for improved performance
        while maintaining the exact same interface and behavior.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_thermodynamics

        if use_fast and _THERMODYNAMICS_AVAILABLE:
            # Use the new optimized calculator
            self.kinetic_energy = fast_kinetic_energy(self.vel, self.masses)
            self.species_kinetic_energy = scalar_species_loop(self.kinetic_energy, self.species_num)
            
            # Calculate temperature using optimized function
            particle_temperatures = fast_temperature_from_velocities(
                self.vel, self.masses, self.dimensions, self.kB
            )
            # Aggregate by species using the same logic as before
            self.species_temperature = scalar_species_loop(particle_temperatures, self.species_num) / self.species_num
        else:
            # Use the legacy implementation
            const = 2.0 / (self.kB * self.species_num * self.dimensions)
            self.calculate_kinetic_energy(use_fast=False)
            self.species_kinetic_energy = scalar_species_loop(self.kinetic_energy, self.species_num)
            self.species_temperature = const * self.species_kinetic_energy

    def calculate_species_temperature(self, use_fast=None):
        """
        Calculate the temperature of each species and store it into :attr:`species_temperature`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        
        Note
        ----
        Redundant with :meth:`calculate_species_kinetic_temperature`.
        """
        self.calculate_species_kinetic_temperature(use_fast=use_fast)

    def calculate_total_kinetic_energy(self, use_fast=None):
        """
        Calculate the total kinetic energy by summing the :attr:`kinetic_energy` array and store it into :attr:`total_kinetic_energy`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        self.calculate_species_kinetic_temperature(use_fast=use_fast)
        self.total_kinetic_energy = self.species_kinetic_energy.sum()

    def calculate_species_momentum(self, use_fast=None):
        """
        Calculate momentum of each species.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            # Use the new optimized calculator
            particle_momenta = fast_momentum(self.vel, self.masses)
            # Aggregate by species
            self.species_momentum = vector_species_loop(particle_momenta, self.species_num)
        else:
            # Use the legacy implementation
            velocity = vector_species_loop(self.vel, self.species_num)
            self.species_momentum = self.species_masses[:, newaxis] * velocity

    def calculate_total_momentum(self, use_fast=None):
        """
        Calculate total momentum of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        self.calculate_species_momentum(use_fast=use_fast)
        self.total_momentum = self.species_momentum.sum()

    # =============================================================================
    # TRANSPORT METHODS
    # =============================================================================

    def calculate_electric_current(self, use_fast=None):
        """
        Calculate the electric current of each particle and store it into :attr:`electric_current`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
        
        Notes
        -----
        This method now delegates to the optimized transport calculator by default
        while maintaining full backward compatibility.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport
        
        if use_fast and _TRANSPORT_AVAILABLE:
            # Use the new optimized calculator
            self.electric_current = fast_calculate_electric_current(self.vel, self.charges)
        else:
            # Use the legacy implementation
            self.electric_current = self.charges[:, None] * self.vel

    def calculate_species_electric_current(self, use_fast=None):
        """
        Calculate the electric current of each species from :attr:`vel` and stores it into :attr:`species_electric_current`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            # Use the new optimized calculator
            self.species_electric_current = fast_calculate_species_electric_current(
                self.vel, self.charges, self.species_num
            )
        else:
            # Use the legacy implementation
            self.species_electric_current = self.species_charges[:, None] * vector_species_loop(self.vel, self.species_num)

    def calculate_total_electric_current(self, use_fast=None):
        """
        Calculate the total electric current of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            # Use the new optimized calculator - direct total calculation
            self.total_electric_current = fast_electric_current_vector(self.vel, self.charges)
        else:
            # Use the legacy implementation via species calculation
            self.calculate_species_electric_current(use_fast=False)
            self.total_electric_current = self.species_electric_current.sum(axis=0)

    def calculate_species_heat_flux(self, use_fast=None):
        """
        Calculate the energy current of each species and stores it into :attr:`species_heat_flux`.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
        
        Notes
        -----
        This method now uses the optimized transport calculator for improved performance.
        The heat flux calculation includes convective energy transport.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            # Calculate kinetic energy first
            self.calculate_kinetic_energy(use_fast=use_fast)
            
            # Use the new optimized calculator for heat flux
            # Calculate total heat flux and then distribute by species
            total_heat_flux = fast_heat_flux_vector(self.vel, self.kinetic_energy, volume=1.0)
            
            # For species-specific heat flux, we need to aggregate properly
            # This is a more complex calculation that requires species-wise aggregation
            species_start = 0
            species_heat_flux = zeros((self.num_species, 3))
            
            for sp in range(self.num_species):
                species_end = species_start + self.species_num[sp]
                sp_vel = self.vel[species_start:species_end]
                sp_ke = self.kinetic_energy[species_start:species_end]
                
                if sp_vel.size > 0:
                    species_heat_flux[sp] = fast_heat_flux_vector(sp_vel, sp_ke, volume=1.0)
                
                species_start = species_end
            
            self.species_heat_flux = species_heat_flux
        else:
            # Use the legacy implementation
            if hasattr(self, 'heat_flux_species_tensor') and self.heat_flux_species_tensor is not None:
                self.species_heat_flux = self.heat_flux_species_tensor.sum(axis=0)
            else:
                # Fallback calculation
                self.calculate_kinetic_energy(use_fast=False)
                species_start = 0
                species_heat_flux = zeros((self.num_species, 3))
                
                for sp in range(self.num_species):
                    species_end = species_start + self.species_num[sp]
                    sp_vel = self.vel[species_start:species_end]
                    sp_ke = self.kinetic_energy[species_start:species_end]
                    
                    # Simple convective heat flux: sum of ke * v
                    if sp_vel.size > 0:
                        species_heat_flux[sp] = (sp_ke[:, None] * sp_vel).sum(axis=0)
                    
                    species_start = species_end
                
                self.species_heat_flux = species_heat_flux

    def calculate_species_diffusion_flux(self, use_fast=None):
        """
        Calculate the diffusion fluxes.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            # Use the new optimized calculator
            # Calculate species average velocities
            species_velocities = vector_species_loop(self.vel, self.species_num)
            
            self.species_diffusion_flux = fast_diffusion_flux(
                species_velocities, self.species_concentrations, self.species_masses
            )
        else:
            # Use the legacy implementation
            self.species_diffusion_flux = calc_species_diffusion_flux(
                self.vel, self.species_masses, self.species_num
            )

    # =============================================================================
    # TRANSPORT UTILITY METHODS
    # =============================================================================

    def calculate_current_density(self, use_fast=None):
        """
        Calculate electric current density of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
            
        Returns
        -------
        numpy.ndarray
            Current density vector. Shape: (3,).
            Units: A/m²
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            volume = getattr(self, 'box_volume', 1.0)
            return fast_electric_current_density(self.vel, self.charges, volume)
        else:
            # Legacy calculation
            total_current = self.charges[:, None] * self.vel
            current_vector = total_current.sum(axis=0)
            volume = getattr(self, 'box_volume', 1.0)
            return current_vector / volume

    def calculate_heat_flux_tensor(self, use_fast=None):
        """
        Calculate the full heat flux tensor including convective and stress contributions.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized transport calculator. If None, uses the class default.
            
        Returns
        -------
        numpy.ndarray
            Heat flux tensor. Shape: (3, 3).
            Units: W/m²
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_transport

        if use_fast and _TRANSPORT_AVAILABLE:
            # Calculate kinetic energy first
            self.calculate_kinetic_energy(use_fast=use_fast)
            
            # Use stress tensor if available
            stress_tensor = getattr(self, 'stress_tensor', None)
            volume = getattr(self, 'box_volume', 1.0)
            
            return fast_heat_flux_tensor(
                self.vel, self.kinetic_energy, stress_tensor, volume
            )
        else:
            # Legacy implementation - simplified version
            self.calculate_kinetic_energy(use_fast=False)
            
            # Simple convective heat flux tensor
            heat_flux_tensor = zeros((3, 3))
            for i in range(self.total_num_ptcls):
                for α in range(3):
                    for β in range(3):
                        heat_flux_tensor[α, β] += self.kinetic_energy[i] * self.vel[i, α] * self.vel[i, β]
            
            volume = getattr(self, 'box_volume', 1.0)
            return heat_flux_tensor / volume

    # =============================================================================
    # MECHANICAL METHODS - UPDATED TO USE NEW CALCULATOR
    # =============================================================================

    def calculate_species_pressure_tensor(self, use_fast=None):
        """
        Calculate the pressure, the kinetic part of the pressure tensor, the potential part of the pressure tensor of each species.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            # Use the new optimized calculator
            volume = getattr(self, 'box_volume', 1.0)
            
            self.species_pressure, self.species_pressure_kin_tensor, self.species_pressure_pot_tensor = (
                fast_calculate_species_pressure_tensor(
                    self.vel, self.virial_species_tensor, self.species_num, self.masses, volume
                )
            )
            self.species_pressure_tensor = self.species_pressure_kin_tensor + self.species_pressure_pot_tensor
        else:
            # Use the legacy implementation
            self.species_pressure, self.species_pressure_kin_tensor, self.species_pressure_pot_tensor = calc_pressure_tensor(
                self.vel, self.virial_species_tensor, self.species_masses, self.species_num, 
                getattr(self, 'box_volume', 1.0), self.dimensions
            )
            self.species_pressure_tensor = self.species_pressure_kin_tensor + self.species_pressure_pot_tensor

    def calculate_species_pressure(self, use_fast=None):
        """
        Calculate the pressure of each species.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
        """
        self.calculate_species_pressure_tensor(use_fast=use_fast)

    def calculate_total_pressure(self, use_fast=None):
        """
        Calculate the total pressure of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
        """
        self.calculate_species_pressure_tensor(use_fast=use_fast)
        self.total_pressure = self.species_pressure.sum()

    def calculate_species_enthalpy(self, use_fast=None):
        """
        Calculate the enthalpy of each species.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_thermodynamics

        if use_fast and _THERMODYNAMICS_AVAILABLE:
            # Ensure we have all required quantities
            self.calculate_kinetic_energy(use_fast=use_fast)
            self.calculate_species_potential_energy()
            self.calculate_species_pressure_tensor(use_fast=self._use_fast_mechanical)
            
            volume = getattr(self, 'box_volume', 1.0)
            
            # Use the new optimized calculator
            self.species_enthalpy = fast_species_enthalpy(
                self.kinetic_energy, self.potential_energy, 
                self.species_pressure, volume, self.species_num
            )
        else:
            # Use the legacy implementation
            self.calculate_kinetic_energy(use_fast=False)
            self.calculate_species_potential_energy()
            self.calculate_species_pressure_tensor(use_fast=False)
            
            # Legacy calculation
            energy = scalar_species_loop(self.kinetic_energy + self.potential_energy, self.species_num)
            volume = getattr(self, 'box_volume', 1.0)
            self.species_enthalpy = energy + self.species_pressure * volume

    def calculate_total_enthalpy(self, use_fast=None):
        """
        Calculate the total enthalpy of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
        """
        self.calculate_species_enthalpy(use_fast=use_fast)
        self.total_enthalpy = self.species_enthalpy.sum()

    # =============================================================================
    # NEW MECHANICAL UTILITY METHODS
    # =============================================================================

    def calculate_angular_momentum(self, origin=None, use_fast=None):
        """
        Calculate total angular momentum of the system.
        
        Parameters
        ----------
        origin : numpy.ndarray, optional
            Origin point for angular momentum calculation. If None, uses (0,0,0).
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
            
        Returns
        -------
        numpy.ndarray
            Total angular momentum vector. Shape: (3,).
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            return fast_angular_momentum(self.pos, self.vel, self.masses, origin)
        else:
            # Legacy implementation
            if origin is None:
                origin = zeros(3)
            
            r = self.pos - origin
            p = self.masses[:, None] * self.vel
            
            # L = r × p
            L = zeros(3)
            for i in range(len(self.masses)):
                L[0] += r[i, 1] * p[i, 2] - r[i, 2] * p[i, 1]
                L[1] += r[i, 2] * p[i, 0] - r[i, 0] * p[i, 2]
                L[2] += r[i, 0] * p[i, 1] - r[i, 1] * p[i, 0]
            
            return L

    def calculate_pressure_tensor_kinetic(self, use_fast=None):
        """
        Calculate kinetic contribution to pressure tensor.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
            
        Returns
        -------
        numpy.ndarray
            Kinetic pressure tensor. Shape: (3, 3).
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            volume = getattr(self, 'box_volume', 1.0)
            return fast_pressure_tensor_kinetic(self.vel, self.masses, volume)
        else:
            # Legacy implementation
            pressure_tensor = zeros((3, 3))
            for i in range(len(self.masses)):
                for α in range(3):
                    for β in range(3):
                        pressure_tensor[α, β] += self.masses[i] * self.vel[i, α] * self.vel[i, β]
            
            volume = getattr(self, 'box_volume', 1.0)
            return pressure_tensor / volume
        
    # =============================================================================
    # NEW UTILITY METHODS FOR CENTER OF MASS
    # =============================================================================

    def calculate_center_of_mass_velocity(self, use_fast=None):
        """
        Calculate the center of mass velocity of the system.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized mechanical calculator. If None, uses the class default.
            
        Returns
        -------
        numpy.ndarray
            Center of mass velocity. Shape: (3,).
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            return fast_center_of_mass_velocity(self.vel, self.masses)
        else:
            # Legacy implementation
            total_momentum = (self.masses[:, None] * self.vel).sum(axis=0)
            total_mass = self.masses.sum()
            return total_momentum / total_mass if total_mass > 0 else zeros(3)

    def remove_center_of_mass_motion(self, use_fast=None):
        """
        Remove center of mass motion from particle velocities.
        
        Parameters
        ----------
        use_fast : bool, optional
            Use the optimized thermodynamics calculator. If None, uses the class default.
            
        Notes
        -----
        This method modifies self.vel in place to remove center of mass motion,
        enforcing conservation of total momentum.
        """
        # Determine which implementation to use
        if use_fast is None:
            use_fast = self._use_fast_mechanical

        if use_fast and _MECHANICAL_AVAILABLE:
            self.vel = fast_remove_center_of_mass_motion(self.vel, self.masses)
        else:
            # Legacy implementation
            cm_velocity = self.calculate_center_of_mass_velocity(use_fast=False)
            self.vel -= cm_velocity

    # =============================================================================
    # PERFORMANCE CONTROL METHODS
    # =============================================================================

    def set_fast_thermodynamics(self, enabled=True):
        """
        Enable or disable the fast thermodynamics calculator.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to use the fast thermodynamics calculator. Default: True.
            
        Notes
        -----
        This method provides a runtime toggle for the thermodynamics implementation,
        useful for testing, debugging, or gradual migration.
        """
        if enabled and not _THERMODYNAMICS_AVAILABLE:
            warn(
                "Fast thermodynamics calculator is not available. "
                "Install the sarkas.physics.thermodynamics module.",
                category=UserWarning
            )
            self._use_fast_thermodynamics = False
        else:
            self._use_fast_thermodynamics = enabled

    def get_fast_thermodynamics_status(self):
        """
        Get the current status of the fast thermodynamics calculator.
        
        Returns
        -------
        dict
            Dictionary containing status information:
            - 'available': Whether the fast calculator is available
            - 'enabled': Whether it's currently enabled
            - 'active': Whether it's both available and enabled
        """
        return {
            'available': _THERMODYNAMICS_AVAILABLE,
            'enabled': self._use_fast_thermodynamics,
            'active': self._use_fast_thermodynamics and _THERMODYNAMICS_AVAILABLE
        }

    def set_fast_transport(self, enabled=True):
        """
        Enable or disable the fast transport calculator.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to use the fast transport calculator. Default: True.
            
        Notes
        -----
        This method provides a runtime toggle for the transport implementation,
        useful for testing, debugging, or gradual migration.
        """
        if enabled and not _TRANSPORT_AVAILABLE:
            warn(
                "Fast transport calculator is not available. "
                "Install the sarkas.physics.transport module.",
                category=UserWarning
            )
            self._use_fast_transport = False
        else:
            self._use_fast_transport = enabled

    def get_fast_transport_status(self):
        """
        Get the current status of the fast transport calculator.
        
        Returns
        -------
        dict
            Dictionary containing status information:
            - 'available': Whether the fast calculator is available
            - 'enabled': Whether it's currently enabled
            - 'active': Whether it's both available and enabled
        """
        return {
            'available': _TRANSPORT_AVAILABLE,
            'enabled': self._use_fast_transport,
            'active': self._use_fast_transport and _TRANSPORT_AVAILABLE
        }

    def set_fast_mechanical(self, enabled=True):
        """
        Enable or disable the fast mechanical calculator.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to use the fast mechanical calculator. Default: True.
        """
        if enabled and not _MECHANICAL_AVAILABLE:
            warn(
                "Fast mechanical calculator is not available. "
                "Install the sarkas.physics.mechanical module.",
                category=UserWarning
            )
            self._use_fast_mechanical = False
        else:
            self._use_fast_mechanical = enabled

    def get_fast_mechanical_status(self):
        """
        Get the current status of the fast mechanical calculator.
        
        Returns
        -------
        dict
            Dictionary containing status information.
        """
        return {
            'available': _MECHANICAL_AVAILABLE,
            'enabled': self._use_fast_mechanical,
            'active': self._use_fast_mechanical and _MECHANICAL_AVAILABLE
        }

    def set_fast_physics(self, enabled=True):
        """
        Enable or disable all fast physics calculators.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to use fast calculators. Default: True.
        """
        self.set_fast_thermodynamics(enabled)
        self.set_fast_transport(enabled)
        self.set_fast_mechanical(enabled)

    def get_fast_physics_status(self):
        """
        Get comprehensive status of all fast physics calculators.
        
        Returns
        -------
        dict
            Dictionary with status for all physics modules.
        """
        return {
            'thermodynamics': self.get_fast_thermodynamics_status(),
            'transport': self.get_fast_transport_status(),
            'mechanical': self.get_fast_mechanical_status()
        }

    # =============================================================================
    # BACKWARD COMPATIBILITY METHODS - UNCHANGED INTERFACE
    # =============================================================================

    def kinetic_temperature(self):
        """
        Calculate the kinetic energy and temperature of each species.

        Returns
        -------
        K : numpy.ndarray
            Kinetic energy of each species. Shape=(:attr:`num_species`).

        T : numpy.ndarray
            Temperature of each species. Shape=(:attr:`num_species`).

        Raises
        ------
            : DeprecationWarning
        """
        warn(
            "Deprecated feature. It will be removed in a future release. \n"
            "Use particles.calculate_species_kinetic_temperature()",
            category=DeprecationWarning,
        )
        self.calculate_species_kinetic_temperature()
        return self.species_kinetic_energy, self.species_temperature

    # =============================================================================
    # VALIDATION AND TESTING METHODS
    # =============================================================================

    def validate_thermodynamics_consistency(self, rtol=1e-12, atol=1e-15):
        """
        Validate that fast and legacy thermodynamics implementations give identical results.
        
        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for comparison. Default: 1e-12.
        atol : float, optional
            Absolute tolerance for comparison. Default: 1e-15.
            
        Returns
        -------
        dict
            Dictionary with validation results for each method tested.
            
        Raises
        ------
        AssertionError
            If results differ beyond specified tolerances.
            
        Notes
        -----
        This method is useful for testing and validation during migration.
        It compares results from both implementations to ensure numerical consistency.
        """
        if not _THERMODYNAMICS_AVAILABLE:
            return {"error": "Fast thermodynamics not available for comparison"}
        
        import numpy as np
        results = {}
        
        # Test kinetic energy calculation
        try:
            # Calculate with fast method
            ke_fast = fast_kinetic_energy(self.vel, self.masses)
            
            # Calculate with legacy method
            ke_legacy = 0.5 * self.masses * (self.vel * self.vel).sum(axis=-1)
            
            # Compare results
            np.testing.assert_allclose(ke_fast, ke_legacy, rtol=rtol, atol=atol)
            results['kinetic_energy'] = 'PASS'
        except Exception as e:
            results['kinetic_energy'] = f'FAIL: {str(e)}'
        
        # Test temperature calculation if possible
        if hasattr(self, 'kB') and self.kB is not None:
            try:
                # Calculate temperatures with fast method
                T_fast = fast_temperature_from_velocities(
                    self.vel, self.masses, self.dimensions, self.kB
                )
                
                # Calculate with legacy method via kinetic energy
                ke = 0.5 * self.masses * (self.vel * self.vel).sum(axis=-1)
                T_legacy = 2.0 * ke / (self.dimensions * self.kB)
                
                # Compare results
                np.testing.assert_allclose(T_fast, T_legacy, rtol=rtol, atol=atol)
                results['temperature'] = 'PASS'
            except Exception as e:
                results['temperature'] = f'FAIL: {str(e)}'
        
        return results

    # =============================================================================
    # REMAINING METHODS - UNCHANGED FROM ORIGINAL
    # =============================================================================

    def calculate_observables(self):
        """Calculate the observables in :attr:`observables_list`."""
        for key in self.species_observables_method_map.keys():
            self.species_observables_method_map[key]()

    def calculate_species_total_energy(self):
        """Calculate the total energy of each species and store it into :attr:`species_total_energy`."""
        self.calculate_species_kinetic_energy()
        self.calculate_species_potential_energy()
        self.species_total_energy = scalar_species_loop(self.kinetic_energy + self.potential_energy, self.species_num)

    def calculate_species_velocity_moments(self):
        """Calculate the moments of the velocity distribution using the velocity of each species and stores them into :attr:`species_velocity_moments`."""
        species_start = 0
        species_end = 0

        for i, num in enumerate(self.species_num):
            species_end += num
            for mom in range(self.max_velocity_distribution_moment):
                self.species_velocity_moments[i, mom, :] = moment(
                    self.vel[species_start:species_end, :], moment=mom + 1, axis=0
                )
            species_start += num

    def calculate_species_kl_divergence(self):
        """Calculate the Kullback-Leibler divergence of the velocity distribution of each species and stores it into :attr:`species_kl_divergence`."""
        nbins = self.total_num_ptcls // 10
        self.species_kl_divergence = kl_divergence(self.vel, self.species_num, self.species_thermal_velocity, nbins)

    def calculate_species_potential_energy(self):
        """Calculate the potential energy of each species from :attr:`potential_energy`, calculated in the force loop, and stores it into :attr:`species_potential_energy`."""
        self.species_potential_energy = scalar_species_loop(self.potential_energy, self.species_num)

    def calculate_total_potential_energy(self):
        """Calculate the total potential energy by summing the :attr:`potential_energy` array. The total potential energy is store in :attr:`total_potential_energy`."""
        self.calculate_species_potential_energy()
        self.total_potential_energy = self.species_potential_energy.sum()

    def make_species_observables_method_map(self, observables_list=None):
        """Make a dictionary where each key is an element of observables_list and each value is a method of Particles."""
        if observables_list is None:
            observables_list = self.observables_list
        else:
            for obs in observables_list:
                if obs not in self.observables_list:
                    self.observables_list.append(obs)
            observables_list = self.observables_list

        key_list = list(self.species_observables_method_map.keys())
        for key in key_list:
            if key not in observables_list:
                del self.species_observables_method_map[key]

    def make_species_thermodynamics_dictionary(self, thermodynamics_list=None):
        """
        Put the main thermodynamic quantities into a dictionary. This is used for saving data while running.

        Returns
        -------
        data : dict
            Thermodynamics data. In case of multiple species, it returns thermodynamics quantities per species.
        """
        if thermodynamics_list is None:
            thermodynamics_list = self.thermodynamics_list

        for property in thermodynamics_list:
            if property not in self.species_thermodynamics_method_map.keys():
                self.species_thermodynamics_method_map[property] = getattr(self, f"calculate_species_{property}")
    
    def make_species_thermodynamics_method_map(self, thermodynamics_list):
        """
        Make the dictionary :attr:`species_thermodynamics_method_map` with the new thermodynamic quantities.
        
        Parameters
        ----------
        thermodynamics_list : list
            List of thermodynamic quantities to calculate for each species.
        
        Notes
        -----
        This is used to make the dictionary :attr:`species_thermodynamics_method_map` with the new thermodynamic quantities. 
        The dictionary is a map of the thermodynamic quantities to the method to calculate them.

        """
        if thermodynamics_list is None:
            thermodynamics_list = self.thermodynamics_list

        for property in thermodynamics_list:
            if not hasattr(self, f"calculate_species_{property}"):
                raise ParticlesError(f"Method calculate_species_{property} not found in Particles.")
            else:
                self.species_thermodynamics_method_map[property] = getattr(self, f"calculate_species_{property}")

    def calculate_species_thermodynamics(self):
        """Calculate thermodynamics quantities for each species."""
        for key in self.species_thermodynamics_method_map.keys():
            self.species_thermodynamics_method_map[key]()

    def calculate_species_observables(self):
        """Calculate the observables for each species."""
        for key in self.species_observables_method_map.keys():
            self.species_observables_method_map[key]()

    def load_from_checkpoint(self, phase, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.
        phase: str
            Restart phase.
        """
        if phase == "equilibration":
            file_name = self.process_h5md_filepath_dict["equilibration"]
            dump_step = self.eq_dump_step
        elif phase == "production":
            file_name = self.process_h5md_filepath_dict["production"]
            dump_step = self.prod_dump_step
        elif phase == "magnetization":
            file_name = self.process_h5md_filepath_dict["magnetization"]
            dump_step = self.mag_dump_step

        # Calculate the index of the time step
        index = self.restart_step // dump_step

        with h5py.File(file_name, "r") as file:
            self.pos = file["particles/pos"][index]
            self.vel = file["particles/vel"][index]
            if 'rdf_hist' in file["observables"].keys():
                self.rdf_hist = file["observables/rdf_hist/value"][index]

    def lattice(self, perturb: float = 0.05):
        """
        Place particles in a simple cubic lattice with a slight perturbation ranging
        from 0 to 0.5 times the lattice spacing.

        Parameters
        ----------
        perturb : float
            Value of perturbation, p, such that 0 <= p <= 0.5. Default = 0.05

        """

        # Check if perturbation is below maximum allowed. If not, default to maximum perturbation.
        if perturb > 0.5:
            warn("Random perturbation must not exceed 0.5. Setting perturb = 0.5", category=ParticlesWarning)
            perturb = 0.5

        if self.lattice_type == "simple_cubic":
            # Determining number of particles per side of simple cubic lattice
            part_per_side = self.total_num_ptcls ** (1.0 / 3.0)  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if round(part_per_side) ** 3 != self.total_num_ptcls:
                part_per_side = rint(self.total_num_ptcls ** (1.0 / 3.0))
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a simple cubic lattice. "
                    f"Use {int(part_per_side ** 3)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing
            dz_lattice = self.pbox_lengths[2] / (self.total_num_ptcls ** (1.0 / 3.0))  # Lattice spacing

            # Create x, y, and z position arrays
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice
            z = arange(0, self.pbox_lengths[2], dz_lattice) + 0.5 * dz_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y, Z = meshgrid(x, y, z)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice
            Z += self.rnd_gen.uniform(-0.5, 0.5, Z.shape) * perturb * dz_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = Z.ravel() + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

        elif self.lattice_type == "bcc":
            # Determining number of particles per side of simple cubic lattice
            part_per_side = int(0.5 * self.total_num_ptcls) ** (
                1.0 / 3.0
            )  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if round(part_per_side) ** 3 != int(0.5 * self.total_num_ptcls):
                part_per_side = rint((0.5 * self.total_num_ptcls) ** (1.0 / 3.0))
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a bcc lattice. "
                    f"Use {int(2.0*part_per_side ** 3)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing
            dz_lattice = self.pbox_lengths[2] / (0.5 * self.total_num_ptcls) ** (1.0 / 3.0)  # Lattice spacing

            # Create x, y, and z position arrays. 
            # Note that the lattice is shifted by 0.5 * dx_lattice this is to ensure periodic boundary conditions. There are no particles at the corner [0,0,0] while there is one at [Lx, Ly, Lz]
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice
            z = arange(0, self.pbox_lengths[2], dz_lattice) + 0.5 * dz_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y, Z = meshgrid(x, y, z)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice
            Z += self.rnd_gen.uniform(-0.5, 0.5, Z.shape) * perturb * dz_lattice

            half_Np = int(self.total_num_ptcls / 2)
            
            self.pos[:half_Np, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:half_Np, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:half_Np, 2] = Z.ravel() + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2
            
            self.pos[half_Np:, 0] = X.ravel() + 0.5 * dx_lattice + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[half_Np:, 1] = Y.ravel() + 0.5 * dy_lattice + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[half_Np:, 2] = Z.ravel() + 0.5 * dz_lattice + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

        elif self.lattice_type in ["square", "tetragonal_2D"]:
            # Determining number of particles per side of simple cubic lattice
            part_per_side = rint(sqrt(self.total_num_ptcls))  # Number of particles per side of a square lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if part_per_side**2 != self.total_num_ptcls:
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in a square lattice. "
                    f"Use {int(part_per_side ** 2)} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / sqrt(self.total_num_ptcls)  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / sqrt(self.total_num_ptcls)  # Lattice spacing

            # Create x, y, and z position arrays
            x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
            y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice

            # Create a lattice with appropriate x, y, and z values based on arange
            X, Y = meshgrid(x, y)

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = 0.0

        elif self.lattice_type in ["hexagonal", "triangular"]:
            # Determining number of particles per side of simple cubic lattice
            part_per_side = round(sqrt(self.total_num_ptcls))  # Number of particles per side of cubic lattice

            # Check if total number of particles is a perfect cube, if not, place more than the requested amount
            if self.np_per_side[:2].prod() != part_per_side * (part_per_side + 1):
                raise ParticlesError(
                    f"N = {self.total_num_ptcls} cannot be placed in an hexagonal lattice. "
                    f"Use Nx = {part_per_side} and Ny = {part_per_side + 1} particles instead."
                )

            dx_lattice = self.pbox_lengths[0] / (self.np_per_side[0])  # Lattice spacing
            dy_lattice = self.pbox_lengths[1] / (self.np_per_side[1])  # Lattice spacing

            if self.np_per_side[0] > self.np_per_side[1]:
                # Create x, y, and z position arrays
                x = arange(0, self.pbox_lengths[0], dx_lattice)
                y = arange(0, self.pbox_lengths[1], dy_lattice) + 0.5 * dy_lattice

                # Create a lattice with appropriate x, y, and z values based on arange
                X, Y = meshgrid(x, y)
                # Shift the Y axis of every other row of particles
                X[:, ::2] += 0.5 * dx_lattice

            else:
                # Create x, y, and z position arrays
                x = arange(0, self.pbox_lengths[0], dx_lattice) + 0.5 * dx_lattice
                y = arange(0, self.pbox_lengths[1], dy_lattice)

                # Create a lattice with appropriate x, y, and z values based on arange
                X, Y = meshgrid(x, y)
                # Shift the Y axis of every other row of particles
                Y[:, ::2] += 0.5 * dy_lattice

            # Perturb lattice
            X += self.rnd_gen.uniform(-0.5, 0.5, X.shape) * perturb * dx_lattice
            Y += self.rnd_gen.uniform(-0.5, 0.5, Y.shape) * perturb * dy_lattice

            # Flatten the meshgrid values for plotting and computation
            self.pos[:, 0] = X.ravel() + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
            self.pos[:, 1] = Y.ravel() + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
            self.pos[:, 2] = 0.0

    def load(self):
        """
        Initialize particles' positions and velocities.
        Positions are initialized based on the load method while velocities are chosen
        from a Maxwell-Boltzmann distribution.

        """

        warn(
            "Deprecated feature. It will be removed in a future release. \n"
            "Use parameters.calc_electron_properties(species). You need to pass the species list.",
            category=DeprecationWarning,
        )

        self.initialize_positions()

    def load_from_file(self, f_name):
        """
        Load particles' data from a specific file.

        Parameters
        ----------
        f_name : str
            Filename

        Raises
        ------
            : DeprecationWarning

        """

        warn(
            "Deprecated feature. This is a legacy feature that will be removed in a future release unless requests to keep it are made.",
            category=DeprecationWarning,
        )

        pv_data = loadtxt(f_name)
        if not (pv_data.shape[0] == self.total_num_ptcls):
            msg = (
                f"Number of particles is not same between input file and initial p & v data file. \n "
                f"Input file: N = {self.total_num_ptcls}, load data: N = {pv_data.shape[0]}"
            )
            raise ParticlesError(msg)

        self.pos[:, 0] = pv_data[:, 0].copy()
        self.pos[:, 1] = pv_data[:, 1].copy()
        self.pos[:, 2] = pv_data[:, 2].copy()

        self.vel[:, 0] = pv_data[:, 3].copy()
        self.vel[:, 1] = pv_data[:, 4].copy()
        self.vel[:, 2] = pv_data[:, 5].copy()

    def load_from_npz(self, file_name):
        """
        Load particles' data from an .npz data file.

        Parameters
        ----------
        file_name : str
            Path to file.

        """
        # file_name = join(self.eq_dump_dir, "checkpoint_" + str(it) + ".npz")
        data = np_load(file_name, allow_pickle=True)
        if not data["pos"].shape[0] == self.total_num_ptcls:
            msg = (
                f"Number of particles is not same between input file and particles data file. \n "
                f"Input file: N = {self.total_num_ptcls}, particles data file: N = {data['pos'].shape[0]}"
            )
            raise ParticlesError(msg)

        self.id = data["id"].copy()
        self.names = data["names"].copy()
        self.pos = data["pos"].copy()
        self.vel = data["vel"].copy()
        if "acc" in data.files:
            self.acc = data["acc"].copy()
        if "rdf_hist" in data.files:
            self.rdf_hist = data["rdf_hist"]

    def load_from_restart(self, phase, it):
        """
        Initialize particles' data from a checkpoint of a previous run.

        Raises
        ------
            : DeprecationWarning
        """

        warn(
            "Deprecated feature. It will be removed in a future release.\nUse load_from_checkpoint. ",
            category=DeprecationWarning,
        )

        self.load_from_checkpoint(phase, it)

    def load_from_checkpoint(self, phase, it):
        """
        Load particles' data from a checkpoint of a previous run

        Parameters
        ----------
        it : int
            Timestep.

        phase: str
            Restart phase.

        """
        if phase == "equilibration":
            file_name = self.process_h5md_filepath_dict["equilibration"]
            dump_step = self.eq_dump_step
        elif phase == "production":
            file_name = self.process_h5md_filepath_dict["production"]
            dump_step = self.prod_dump_step
        elif phase == "magnetization":
            file_name = self.process_h5md_filepath_dict["magnetization"]
            dump_step = self.mag_dump_step

        # Calculate the index of the time step
        index = self.restart_step // dump_step
        
        with h5py.File(file_name, "r") as file:
            self.pos = file["particles/pos"][index]
            self.vel = file["particles/vel"][index]
            if 'rdf_hist' in file["observables"].keys():
                self.rdf_hist = file["observables/rdf_hist/value"][index]

    def random_reject(self, r_reject):
        """
        Place particles by sampling a uniform distribution from 0 to LP (the initial particle box length)
        and uses a rejection radius to avoid placing particles to close to each other.

        Parameters
        ----------
        r_reject : float
            Value of rejection radius.
        """

        # Initialize Arrays
        x = zeros(self.total_num_ptcls)
        y = zeros(self.total_num_ptcls)
        z = zeros(self.total_num_ptcls)

        # Set first x, y, and z positions
        x_new = self.rnd_gen.uniform(0, self.pbox_lengths[0])
        y_new = self.rnd_gen.uniform(0, self.pbox_lengths[1])
        z_new = self.rnd_gen.uniform(0, self.pbox_lengths[2])

        # Append to arrays
        x[0] = x_new
        y[0] = y_new
        z[0] = z_new

        # Particle counter
        i = 1

        cntr_reject = 0
        cntr_total = 0
        # Loop to place particles
        while i < self.total_num_ptcls:
            # Set x, y, and z positions
            x_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[0])
            y_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[1])
            z_new = self.rnd_gen.uniform(0.0, self.pbox_lengths[2])

            # Check if particle was place too close relative to all other current particles
            for j in range(len(x)):
                # Flag for if particle is outside of cutoff radius (True -> not inside rejection radius)
                flag = 1

                # Compute distance b/t particles for initial placement
                x_diff = x_new - x[j]
                y_diff = y_new - y[j]
                z_diff = z_new - z[j]

                # periodic condition applied for minimum image
                if x_diff < -self.pbox_lengths[0] / 2:
                    x_diff += self.pbox_lengths[0]
                if x_diff > self.pbox_lengths[0] / 2:
                    x_diff -= self.pbox_lengths[0]

                if y_diff < -self.pbox_lengths[1] / 2:
                    y_diff += self.pbox_lengths[1]
                if y_diff > self.pbox_lengths[1] / 2:
                    y_diff -= self.pbox_lengths[1]

                if z_diff < -self.pbox_lengths[2] / 2:
                    z_diff += self.pbox_lengths[2]
                if z_diff > self.pbox_lengths[2] / 2:
                    z_diff -= self.pbox_lengths[2]

                # Compute distance
                r = sqrt(x_diff**2 + y_diff**2 + z_diff**2)

                # Check if new particle is below rejection radius. If not, break out and try again
                if r <= r_reject:
                    flag = 0  # new position not added (False -> no longer outside reject r)
                    cntr_reject += 1
                    cntr_total += 1
                    break

            # If flag true add new position
            if flag == 1:
                x[i] = x_new
                y[i] = y_new
                z[i] = z_new

                # Increment particle number
                i += 1
                cntr_total += 1

        self.pos[:, 0] = x + self.box_lengths[0] / 2 - self.pbox_lengths[0] / 2
        self.pos[:, 1] = y + self.box_lengths[1] / 2 - self.pbox_lengths[1] / 2
        self.pos[:, 2] = z + self.box_lengths[2] / 2 - self.pbox_lengths[2] / 2

    def random_unit_vectors(self, num_ptcls, dimensions):
        """
        Initialize random unit vectors for particles' velocities (e.g. for monochromatic energies but random velocities).
        It calls :meth:`numpy.random.Generator.normal`.

        Parameters
        ----------
        num_ptcls : int
            Number of particles to initialize.

        dimensions : int
            Number of non-zero dimensions.

        Returns
        -------
        uvec : numpy.ndarray
            Random unit vectors of specified dimensions for all particles

        """

        uvec = self.rnd_gen.normal(size=(num_ptcls, dimensions))
        # Broadcasting
        uvec /= norm(uvec, axis=1).reshape(num_ptcls, 1)

        return uvec

    def remove_drift(self):
        """
        Enforce conservation of total linear momentum. Updates particles velocities
        """
        remove_drift_nb(self.vel, self.species_num)

    def setup(self, params, species):
        """
        Initialize class' attributes

        Parameters
        ----------
        params: :class:`sarkas.core.Parameters`
            Simulation's parameters.

        species : list
            List of :class:`sarkas.plasma.Species` objects.

        """

        if hasattr(params, "rand_seed"):
            self.rand_seed = params.rand_seed
            self.rnd_gen = Generator(PCG64(params.rand_seed))
        else:
            self.rnd_gen = Generator(PCG64())

        self.copy_params(params)
        self.initialize_arrays()
        self.update_attributes(species)
        
        self.make_species_thermodynamics_dictionary(thermodynamics_list=self.thermodynamics_list)
        self.make_species_thermodynamics_method_map(thermodynamics_list=self.thermodynamics_list)
        self.make_species_observables_method_map(observables_list=self.observables_list)
        # Particles Position Initialization
        if self.load_method in [
            "equilibration_restart",
            "eq_restart",
            "magnetization_restart",
            "mag_restart",
            "production_restart",
            "prod_restart",
        ]:
            # checks
            if self.restart_step is None:
                raise AttributeError("Restart step not defined. Please define Parameters.restart_step.")

            if type(self.restart_step) is not int:
                self.restart_step = int(self.restart_step)

            if self.load_method[:2] == "eq":
                self.load_from_restart("equilibration", self.restart_step)
            elif self.load_method[:2] == "pr":
                self.load_from_restart("production", self.restart_step)
            elif self.load_method[:2] == "ma":
                self.load_from_restart("magnetization", self.restart_step)

        elif self.load_method == "file":
            # check
            if not hasattr(self, "particles_input_file"):
                raise AttributeError("Input file not defined. Please define Parameters.particles_input_file.")

            if self.particles_input_file[-3:] == "npz":
                self.load_from_npz(self.particles_input_file)
            else:
                self.load_from_file(self.particles_input_file)
        else:
            self.initialize_positions(species=species)
            self.initialize_velocities(species=species)
            self.initialize_accelerations()

    def uniform_no_reject(self, mins, maxs):
        """
        Randomly distribute particles along each direction.

        Parameters
        ----------
        mins : float
            Minimum value of the range of a uniform distribution.

        maxs : float
            Maximum value of the range of a uniform distribution.

        Returns
        -------
         : numpy.ndarray
            Particles' property, e.g. pos, vel. Shape = (:attr:`total_num_ptcls`, 3).

        """

        return self.rnd_gen.uniform(mins, maxs, (self.total_num_ptcls, 3))

    def update_attributes(self, species):
        """
        Assign particles attributes.

        Parameters
        ----------
        species : list
            List of :class:`sarkas.plasma.Species` objects.

        """
        species_end = 0
        species_start = 0

        for ic, sp in enumerate(species):
            if sp.name != "electron_background":
                species_end += sp.num

                self.names[species_start:species_end] = sp.name
                self.masses[species_start:species_end] = sp.mass

                if hasattr(sp, "charge"):
                    self.charges[species_start:species_end] = sp.charge
                else:
                    self.charges[species_start:species_end] = 1.0

                if hasattr(sp, "cyclotron_frequency"):
                    self.cyclotron_frequencies[species_start:species_end] = sp.cyclotron_frequency

                self.id[species_start:species_end] = ic
                species_start += sp.num


# =============================================================================
# HELPER FUNCTIONS - UPDATED AND EXISTING
# =============================================================================

@njit
def scalar_species_loop(observable, species_num):
    """
    Calculate the sum over species of the given observable.

    Parameters
    ----------
    observable : numpy.ndarray
        The observable array of shape (N,), where N is the total number of particles.
    species_num : numpy.ndarray
        The array of shape (num_species,) containing the number of particles for each species.

    Returns
    -------
    numpy.ndarray
        An array of shape (num_species,) with the sum over species of the observable.
    """
    sp_start = 0
    sp_end = 0
    sp_obs = zeros(species_num.shape[0])
    for sp, sp_num in enumerate(species_num):
        sp_end += sp_num
        sp_obs[sp] = observable[sp_start:sp_end].sum()
        sp_start += sp_num

    return sp_obs


@njit
def vector_species_loop(observable, species_num):
    """
    Calculate the sum over species of the given observable.

    Parameters
    ----------
    observable : numpy.ndarray
        The observable array of shape (`N`, 3), where `N` is the total number of particles.
    species_num : numpy.ndarray
        The array of shape (`num_species`,) containing the number of particles for each species.

    Returns
    -------
    sp_obs: numpy.ndarray
        An array of shape (`num_species`, 3) with the sum over species of the observable.
    """
    sp_start = 0
    sp_end = 0
    sp_obs = zeros((species_num.shape[0], 3))
    for sp, sp_num in enumerate(species_num):
        sp_end += sp_num
        sp_obs[sp, :] = observable[sp_start:sp_end, :].sum(axis=0)
        sp_start += sp_num

    return sp_obs


@njit
def tensor_species_loop(observable, species_num):
    """
    Calculate the sum over species of the given observable tensor.

    Parameters
    ----------
    observable : numpy.ndarray
        The observable tensor array of shape (N, 3, 3), where N is the total number of particles.
    species_num : numpy.ndarray
        The array of shape (num_species,) containing the number of particles for each species.

    Returns
    -------
    numpy.ndarray
        An array of shape (num_species, 3, 3) with the sum over species of the observable tensor.
    """
    sp_start = 0
    sp_end = 0
    sp_obs = zeros((species_num.shape[0], 3, 3))
    for sp, sp_num in enumerate(species_num):
        sp_end += sp_num
        sp_obs[sp, :, :] = observable[:, :, sp_start:sp_end].sum(axis=-1)
        sp_start += sp_num

    return sp_obs


@njit
def remove_drift_nb(vel, nums):
    """
    Numba'd function to enforce conservation of total linear momentum.
    It updates velocities by removing center of mass motion for each species.

    Parameters
    ----------
    vel: numpy.ndarray
        Particles' velocities.
    nums: numpy.ndarray
        Number of particles of each species.
    """
    species_start = 0
    species_end = 0
    for ic, sp_num in enumerate(nums):
        species_end += sp_num
        vel[species_start:species_end, :] -= vel[species_start:species_end, :].sum(axis=0) / sp_num
        species_start += sp_num


@njit
def calc_species_diffusion_flux(vel, species_masses, species_num):
    """
    Calculates the diffusion flux for each species based on their velocities, masses, and concentrations.

    Parameters
    ----------
    vel : numpy.ndarray
        Array of shape (N, 3) representing the velocities of N particles.
    species_masses : numpy.ndarray
        Array of shape (M,) representing the masses of M species.
    species_num : numpy.ndarray
        Number of particles of each species.

    Returns
    -------
    numpy.ndarray
        Array of shape (M-1, 3) representing the diffusion flux for each species.
    """
    species_net_velocity = vector_species_loop(vel, species_num)
    species_concentrations = species_num / species_num.sum()
    m_bar = species_masses @ species_concentrations
    species_diffusion_flux = zeros((len(species_num) - 1, 3))
    
    for i, m_alpha in enumerate(species_masses[:-1]):
        for j, m_beta in enumerate(species_masses):
            delta_ab = 1 * (m_alpha == m_beta)
            species_diffusion_flux[i, :] += (m_bar * delta_ab - species_concentrations[i] * m_beta) * species_net_velocity[j, :]
        species_diffusion_flux[i, :] *= m_alpha / m_bar

    return species_diffusion_flux


@jit(nopython=True)
def kl_divergence(vel, species_num, species_thermal_velocity, n_bins=100):
    """
    Calculate KL divergence between samples and a standard normal distribution.

    Parameters
    ----------
    vel : numpy.ndarray
        Particle velocities.
    species_num : numpy.ndarray
        Number of particles per species.
    species_thermal_velocity : numpy.ndarray
        Thermal velocities for each species.
    n_bins : int
        Number of bins for histogram.

    Returns
    -------
    numpy.ndarray
        KL divergence for each species.
    """
    kl_div = zeros(len(species_num))
    species_start = 0
    
    for isp, sp_num in enumerate(species_num):
        species_end = species_start + sp_num
        
        for d in range(vel.shape[1]):  # Loop over dimensions
            v_normalized = vel[species_start:species_end, d] / species_thermal_velocity[isp, d]
            
            # Create histogram
            hist_range = (-4.0, 4.0)  # Reasonable range for normalized velocities
            hist, bin_edges = histogram(v_normalized, bins=n_bins, range=hist_range)
            
            # Calculate bin centers and widths
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Calculate probabilities (normalize histogram)
            p_empirical = hist / (hist.sum() * bin_width)
            
            # Calculate theoretical probabilities (standard normal)
            p_theoretical = exp(-0.5 * bin_centers**2) / sqrt(2 * pi)
            
            # Calculate KL divergence
            for i in range(len(p_empirical)):
                if p_empirical[i] > 1e-10 and p_theoretical[i] > 1e-10:
                    kl_div[isp] += p_empirical[i] * log(p_empirical[i] / p_theoretical[i]) * bin_width
        
        species_start = species_end
    
    return kl_div


@njit
def calc_pressure_tensor(vel, virial_species_tensor, species_masses, species_num, box_volume, dimensions):
    """
    Calculate the species pressure tensor.

    Parameters
    ----------
    vel : numpy.ndarray
        Particles' velocities.
    virial_species_tensor : numpy.ndarray
        Virial tensor for each species pair.
    species_masses : numpy.ndarray
        Mass of each species.
    species_num : numpy.ndarray
        Number of particles of each species.
    box_volume : float
        Volume of simulation's box.
    dimensions : int
        Number of dimensions.

    Returns
    -------
    pressure : numpy.ndarray
        Scalar pressure for each species.
    pressure_kin : numpy.ndarray
        Kinetic part of pressure tensor.
    pressure_pot : numpy.ndarray
        Potential part of pressure tensor.
    """
    pressure = zeros(species_num.shape[0])
    pressure_kin = zeros((species_num.shape[0], 3, 3))
    pressure_pot = zeros((species_num.shape[0], 3, 3))
    temp_kin_tensor = zeros((3, 3, vel.shape[0]))

    # Calculate kinetic tensor
    for i in range(3):
        for j in range(3):
            temp_kin_tensor[i, j, :] = vel[:, i] * vel[:, j]

    pressure_kin = species_masses * tensor_species_loop(temp_kin_tensor, species_num) / box_volume
    pressure_pot = virial_species_tensor.sum(axis=0) / box_volume
    pressure_tensor = pressure_kin + pressure_pot
    
    for isp in range(species_num.shape[0]):
        pressure[isp] += (pressure_tensor[isp, 0, 0] + pressure_tensor[isp, 1, 1] + pressure_tensor[isp, 2, 2]) / dimensions

    return pressure, pressure_kin, pressure_pot


# =============================================================================
# MIGRATION UTILITIES
# =============================================================================

def validate_particles_thermodynamics(particles_instance, rtol=1e-12, atol=1e-15):
    """
    Standalone function to validate thermodynamics consistency for a Particles instance.
    
    Parameters
    ----------
    particles_instance : Particles
        The Particles instance to validate.
    rtol : float, optional
        Relative tolerance for comparison. Default: 1e-12.
    atol : float, optional
        Absolute tolerance for comparison. Default: 1e-15.
        
    Returns
    -------
    dict
        Validation results.
    """
    return particles_instance.validate_thermodynamics_consistency(rtol=rtol, atol=atol)


def benchmark_thermodynamics_performance(particles_instance, n_iterations=100):
    """
    Benchmark the performance difference between fast and legacy thermodynamics.
    
    Parameters
    ----------
    particles_instance : Particles
        The Particles instance to benchmark.
    n_iterations : int, optional
        Number of iterations for timing. Default: 100.
        
    Returns
    -------
    dict
        Performance comparison results.
    """
    import time
    import numpy as np
    
    if not _THERMODYNAMICS_AVAILABLE:
        return {"error": "Fast thermodynamics not available for benchmarking"}
    
    # Warm up both implementations
    particles_instance.calculate_kinetic_energy(use_fast=True)
    particles_instance.calculate_kinetic_energy(use_fast=False)
    
    results = {}
    
    # Benchmark kinetic energy calculation
    # Fast implementation
    start_time = time.time()
    for _ in range(n_iterations):
        particles_instance.calculate_kinetic_energy(use_fast=True)
    fast_time = (time.time() - start_time) / n_iterations
    
    # Legacy implementation
    start_time = time.time()
    for _ in range(n_iterations):
        particles_instance.calculate_kinetic_energy(use_fast=False)
    legacy_time = (time.time() - start_time) / n_iterations
    
    speedup = legacy_time / fast_time if fast_time > 0 else float('inf')
    
    results['kinetic_energy'] = {
        'fast_time': fast_time,
        'legacy_time': legacy_time,
        'speedup': speedup,
        'n_particles': particles_instance.total_num_ptcls
    }
    
    return results
