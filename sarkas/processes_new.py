"""
Refactored module handling stages of an MD run with improved architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Type
from importlib import import_module
from IPython import get_ipython
from threading import Thread

if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
    from tqdm import tqdm_notebook as tqdm
    from tqdm.notebook import trange
else:
    from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from numpy import zeros
from os import listdir
from os.path import exists

from .core import Parameters
from .particles import Particles
from .plasma import Species
from .potentials.core import Potential
from .time_evolution.integrators import Integrator
from .utilities.io import InputOutput, print_to_logger
from .utilities.timing import SarkasTimer


# ============================================================================
# Configuration and Validation
# ============================================================================

class ConfigurationValidator:
    """Validates simulation configuration parameters."""
    
    @staticmethod
    def validate_physics_parameters(params: Parameters) -> None:
        """Validate physics-related parameters."""
        if hasattr(params, 'dt') and params.dt <= 0:
            raise ValueError("Timestep (dt) must be positive")
        
        if hasattr(params, 'temperature') and params.temperature < 0:
            raise ValueError("Temperature cannot be negative")
        
        if hasattr(params, 'equilibration_steps') and params.equilibration_steps < 0:
            raise ValueError("Equilibration steps must be non-negative")
        
        if hasattr(params, 'production_steps') and params.production_steps < 0:
            raise ValueError("Production steps must be non-negative")

    @staticmethod
    def validate_io_parameters(io_config: Dict[str, Any]) -> None:
        """Validate I/O related parameters."""
        required_fields = ['md_simulations_dir', 'job_dir']
        for field in required_fields:
            if field not in io_config:
                raise ValueError(f"Required I/O field '{field}' is missing")


# ============================================================================
# Factory Patterns
# ============================================================================

class IntegratorFactory:
    """Factory for creating integrator instances."""
    
    _registry: Dict[str, Type] = {}
    _aliases: Dict[str, str] = {
        'verlet': 'velocity_verlet',
        'leapfrog': 'velocity_verlet',
    }
    
    @classmethod
    def register(cls, name: str, integrator_class: Type, aliases: Optional[List[str]] = None):
        """Register an integrator class."""
        name = name.lower()
        cls._registry[name] = integrator_class
        
        if aliases:
            for alias in aliases:
                cls._aliases[alias.lower()] = name
    
    @classmethod
    def create(cls, integrator_type: str, **kwargs) -> 'IntegratorBase':
        """Create an integrator instance."""
        integrator_type = integrator_type.lower()
        
        # Check aliases
        if integrator_type in cls._aliases:
            integrator_type = cls._aliases[integrator_type]
        
        if integrator_type not in cls._registry:
            available = list(cls._registry.keys())
            available_aliases = list(cls._aliases.keys())
            raise ValueError(
                f"Unknown integrator '{integrator_type}'. "
                f"Available: {available}. Aliases: {available_aliases}"
            )
        
        integrator_class = cls._registry[integrator_type]
        return integrator_class(**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available integrator types."""
        return list(cls._registry.keys())


class PotentialFactory:
    """Factory for creating potential instances."""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, potential_class: Type):
        """Register a potential class."""
        cls._registry[name.lower()] = potential_class
    
    @classmethod
    def create(cls, potential_type: str, method: str = 'pppm', **kwargs) -> 'PotentialBase':
        """Create a potential instance."""
        potential_type = potential_type.lower()
        
        if potential_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown potential '{potential_type}'. Available: {available}")
        
        potential_class = cls._registry[potential_type]
        potential = potential_class(**kwargs)
        
        # Configure the method if the potential supports it
        if hasattr(potential, 'set_method'):
            potential.set_method(method)
        
        return potential


class ThermostatFactory:
    """Factory for creating thermostat instances."""
    
    _registry: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, thermostat_class: Type):
        """Register a thermostat class."""
        cls._registry[name.lower()] = thermostat_class
    
    @classmethod
    def create(cls, thermostat_type: str, **kwargs) -> 'ThermostatBase':
        """Create a thermostat instance."""
        thermostat_type = thermostat_type.lower()
        
        if thermostat_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown thermostat '{thermostat_type}'. Available: {available}")
        
        thermostat_class = cls._registry[thermostat_type]
        return thermostat_class(**kwargs)


# ============================================================================
# Simulation State and Context
# ============================================================================

@dataclass
class SimulationState:
    """Encapsulates the complete state of a simulation."""
    
    parameters: Parameters
    particles: Particles
    integrator: Integrator
    potential: Potential
    io: InputOutput
    timer: SarkasTimer
    species: List[Species]
    
    def evolve(self, phase: str, thermalization: bool, it_start: int, 
               num_steps: int, dump_step: int) -> None:
        """Evolution method to be implemented based on original evolve logic."""
        # This would contain the actual MD evolution loop
        # For now, we'll keep the interface but note that the implementation
        # needs to be moved from the original Process class
        pass


# ============================================================================
# Simulation Phases
# ============================================================================

class SimulationPhase(ABC):
    """Abstract base class for simulation phases."""
    
    def __init__(self, name: str):
        self.name = name
        self.timer = SarkasTimer()
    
    @abstractmethod
    def should_run(self, parameters: Parameters) -> bool:
        """Determine if this phase should execute."""
        pass
    
    @abstractmethod
    def execute(self, simulation_state: SimulationState) -> None:
        """Execute this phase."""
        pass
    
    def prepare(self, simulation_state: SimulationState) -> None:
        """Common preparation steps for all phases."""
        if simulation_state.parameters.remove_initial_drift:
            simulation_state.particles.remove_drift()
    
    def check_restart(self, simulation_state: SimulationState, phase: str) -> int:
        """Check if the simulation is a restart and return starting iteration."""
        if simulation_state.parameters.verbose:
            print(f"\n\n{phase.capitalize():-^70} \n")

        # Check if this is restart
        if (simulation_state.parameters.load_method[:2] == phase[:2] and 
            simulation_state.parameters.load_method[-7:] == "restart"):
            it_start = simulation_state.parameters.restart_step
        else:
            it_start = 0
            dump_step = (simulation_state.parameters.eq_dump_step if phase == "equilibration" 
                        else simulation_state.parameters.prod_dump_step)
            simulation_state.io.save_timestep_data(
                it_start, dump_step, 
                simulation_state.integrator.dt * it_start, 
                simulation_state.particles
            )
        return it_start


class EquilibrationPhase(SimulationPhase):
    """Handles the equilibration phase of the simulation."""
    
    def __init__(self):
        super().__init__("equilibration")
    
    def should_run(self, parameters: Parameters) -> bool:
        """Check if equilibration phase should run."""
        return (parameters.equilibration_phase and 
                parameters.electrostatic_equilibration)
    
    def execute(self, simulation_state: SimulationState) -> None:
        """Execute the equilibration phase."""
        self.prepare(simulation_state)
        
        # Setup phase-specific integrator
        simulation_state.integrator.update = simulation_state.integrator.type_setup(
            simulation_state.integrator.equilibration_type
        )
        
        # Execute equilibration
        simulation_state.io.open_h5md_file(phase="equilibration")
        it_start = self.check_restart(simulation_state, "equilibration")
        
        self.timer.start()
        simulation_state.evolve(
            "equilibration",
            simulation_state.integrator.thermalization,
            it_start,
            simulation_state.parameters.equilibration_steps,
            simulation_state.parameters.eq_dump_step,
        )
        time_eq = self.timer.stop()
        
        simulation_state.io.close_h5md_file()
        simulation_state.io.time_stamp("Equilibration", self.timer.time_division(time_eq))


class MagnetizationPhase(SimulationPhase):
    """Handles the magnetization phase of the simulation."""
    
    def __init__(self):
        super().__init__("magnetization")
    
    def should_run(self, parameters: Parameters) -> bool:
        """Check if magnetization phase should run."""
        return parameters.magnetization_phase
    
    def execute(self, simulation_state: SimulationState) -> None:
        """Execute the magnetization phase."""
        self.prepare(simulation_state)
        
        simulation_state.io.open_h5md_file(phase="magnetization")
        it_start = self.check_restart(simulation_state, "magnetization")
        
        # Update integrator
        simulation_state.integrator.update = simulation_state.integrator.type_setup(
            simulation_state.integrator.magnetization_type
        )
        
        self.timer.start()
        simulation_state.evolve(
            "magnetization",
            simulation_state.integrator.thermalization,
            it_start,
            simulation_state.parameters.magnetization_steps,
            simulation_state.parameters.mag_dump_step,
        )
        time_mag = self.timer.stop()
        
        simulation_state.io.close_h5md_file()
        simulation_state.io.time_stamp("Magnetization", self.timer.time_division(time_mag))


class ProductionPhase(SimulationPhase):
    """Handles the production phase of the simulation."""
    
    def __init__(self):
        super().__init__("production")
    
    def should_run(self, parameters: Parameters) -> bool:
        """Check if production phase should run."""
        return parameters.production_phase
    
    def execute(self, simulation_state: SimulationState) -> None:
        """Execute the production phase."""
        self.prepare(simulation_state)
        
        simulation_state.io.open_h5md_file(phase="production")
        it_start = self.check_restart(simulation_state, "production")
        
        simulation_state.integrator.update = simulation_state.integrator.type_setup(
            simulation_state.integrator.production_type
        )
        
        # Update measurement flag for observables
        simulation_state.potential.measure = True
        
        self.timer.start()
        simulation_state.evolve(
            "production", 
            False, 
            it_start, 
            simulation_state.parameters.production_steps, 
            simulation_state.parameters.prod_dump_step
        )
        time_prod = self.timer.stop()
        
        simulation_state.io.close_h5md_file()
        simulation_state.io.time_stamp("Production", self.timer.time_division(time_prod))


# ============================================================================
# Base Process Class
# ============================================================================

class Process:
    """
    Refactored base class for simulation processes with improved architecture.
    
    This class now focuses on configuration management and setup,
    delegating specific execution logic to appropriate handlers.
    """

    def __init__(self, input_file: Optional[str] = None):
        # Core components
        self.potential: Optional[Potential] = None
        self.integrator: Optional[Integrator] = None
        self.particles: Optional[Particles] = None
        self.parameters: Optional[Parameters] = None
        self.species: List[Species] = []
        self.timer = SarkasTimer()
        self.io = InputOutput()
        
        # Configuration
        self.input_file = input_file
        
        # Observables and transport (for compatibility)
        self.observables_dict: Dict[str, Any] = {}
        self.transport_dict: Dict[str, Any] = {}
        
        # Initialize factories (in a real implementation, this might be done elsewhere)
        self._initialize_factories()
    
    def _initialize_factories(self):
        """Initialize the factory registries."""
        # This would normally be done through module imports or configuration
        # For now, we'll leave it as a placeholder
        pass
    
    def setup(self, read_yaml: bool = True, input_file: Optional[str] = None, 
              other_inputs: Optional[Dict[str, Any]] = None) -> None:
        """
        Setup simulation with improved error handling and validation.
        
        Parameters
        ----------
        read_yaml : bool, default=True
            Flag for reading YAML input file
        input_file : str, optional
            Path to YAML file with inputs
        other_inputs : dict, optional
            Additional simulation options applied after YAML reading
        """
        try:
            if input_file:
                self.input_file = input_file

            if read_yaml:
                if not self.input_file:
                    raise ValueError("No input file specified for YAML reading")
                
                yaml_dict = self.common_parser()
                self.instantiate_subclasses_from_dict(yaml_dict)

            if other_inputs:
                if not isinstance(other_inputs, dict):
                    raise TypeError("other_inputs must be a dictionary")
                self.update_subclasses_from_dict(other_inputs)

            # Validate configuration
            self._validate_configuration()
            
            # Initialize simulation components
            if self.__name__ != "postprocessing":
                self.initialization()

            # Apply plot style if specified
            if hasattr(self.parameters, 'plot_style') and self.parameters.plot_style:
                plt.style.use(self.parameters.plot_style)
                
        except Exception as e:
            self.io.write_to_logger(f"Setup failed: {str(e)}", level="ERROR")
            raise
    
    def _validate_configuration(self) -> None:
        """Validate the complete simulation configuration."""
        if self.parameters:
            ConfigurationValidator.validate_physics_parameters(self.parameters)
    
    def common_parser(self, filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Parse simulation parameters from YAML file.
        
        Parameters
        ----------
        filename : str, optional
            Path to YAML file. If not provided, uses self.input_file
            
        Returns
        -------
        dict
            Parsed simulation parameters
        """
        if filename:
            self.input_file = filename

        if not self.input_file:
            raise ValueError("No input file specified")
        
        if not exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")

        try:
            params_dict = self.io.from_yaml(self.input_file)
            return params_dict
        except Exception as e:
            raise ValueError(f"Failed to parse YAML file {self.input_file}: {str(e)}")

    def instantiate_subclasses_from_dict(self, nested_dict: Dict[str, Any]) -> None:
        """
        Instantiate simulation components using factory patterns.
        
        Parameters
        ----------
        nested_dict : dict
            Configuration dictionary from YAML file
        """
        try:
            for key, config in nested_dict.items():
                if key == "Parameters":
                    self.parameters = Parameters(config)
                    
                elif key == "Particles":
                    self._create_species(config)
                    
                elif key == "Integrator":
                    self.integrator = IntegratorFactory.create(**config)
                    
                elif key == "Potential":
                    self.potential = PotentialFactory.create(**config)
                    
                elif key == "Thermostat":
                    # Thermostat can be handled by integrator or separately
                    if hasattr(self.integrator, 'set_thermostat'):
                        thermostat = ThermostatFactory.create(**config)
                        self.integrator.set_thermostat(thermostat)
                    
                elif key == "Observables":
                    self._create_observables(config)
                    
                elif key == "TransportCoefficients":
                    self._create_transport_coefficients(config)
        
        except Exception as e:
            raise ValueError(f"Failed to instantiate components: {str(e)}")
    
    def _create_species(self, species_config: List[Dict[str, Any]]) -> None:
        """Create species from configuration."""
        self.species = []
        for species_dict in species_config:
            if "Species" in species_dict:
                species = Species(species_dict["Species"])
                self.species.append(species)
    
    def _create_observables(self, observables_config: List[Dict[str, Any]]) -> None:
        """Create observables from configuration."""
        for obs_dict in observables_config:
            for obs_name, params in obs_dict.items():
                try:
                    module = import_module(".observables", "sarkas.tools")
                    obs_class = getattr(module, obs_name)
                    obs_instance = obs_class()
                    obs_instance.from_dict(params)
                    self.observables_dict[obs_instance.__long_name__] = obs_instance
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not create observable {obs_name}: {e}")
    
    def _create_transport_coefficients(self, transport_config: List[Dict[str, Any]]) -> None:
        """Create transport coefficients from configuration."""
        for transport_dict in transport_config:
            for transport_name, params in transport_dict.items():
                try:
                    module = import_module(".transport", "sarkas.tools")
                    transport_class = getattr(module, transport_name)
                    transport_instance = transport_class()
                    transport_instance.from_dict(params)
                    self.transport_dict[transport_instance.__long_name__] = transport_instance
                except (ImportError, AttributeError) as e:
                    print(f"Warning: Could not create transport coefficient {transport_name}: {e}")

    def update_subclasses_from_dict(self, nested_dict: Dict[str, Any]) -> None:
        """
        Update existing simulation components from dictionary.
        
        Parameters
        ----------
        nested_dict : dict
            Dictionary with parameter updates
        """
        for key, values in nested_dict.items():
            if key not in ["Particles", "Observables", "TransportCoefficients"]:
                component = getattr(self, key.lower(), None)
                if component and hasattr(component, '__dict__'):
                    component.__dict__.update(values)
            # Handle special cases like Particles, Observables, etc.
            # (Implementation similar to original but with better error handling)

    def initialization(self) -> None:
        """
        Initialize simulation components with comprehensive setup.
        """
        try:
            # Setup I/O
            self.io.setup()
            self.parameters.copy_io_attrs(self.io)
            self.parameters.potential_type = self.potential.type.lower()
            self.parameters.setup(self.species)

            # Initialize particles
            t0 = self.timer.current()
            self.particles = Particles()
            self.particles.setup(self.parameters, self.species)
            time_ptcls = self.timer.current()
            self.parameters.particles_initialization_time = time_ptcls - t0
            
            # Initialize potential and calculate initial forces
            self.potential.setup(self.parameters, self.species)
            self.potential.calc_acc_pot(self.particles)
            time_pot = self.timer.current()
            self.parameters.cutoff_radius = self.potential.rc

            # Initialize integrator
            self.integrator.setup(self.parameters, self.potential)
            
            # Copy parameters for output
            self._copy_integrator_parameters()
            
            # Setup I/O for different phases
            self._setup_io_phases()
            
            # Save initial state
            self.io.save_pickle(self)
            
            # Print summary
            self.io.simulation_summary(self)
            time_end = self.timer.current()

            # Print timing information
            self._print_initialization_timing(t0, time_ptcls, time_pot, time_end)
            self.print_initial_state()
            
        except Exception as e:
            self.io.write_to_logger(f"Initialization failed: {str(e)}", level="ERROR")
            raise
    
    def _copy_integrator_parameters(self) -> None:
        """Copy integrator parameters to main parameters object."""
        self.parameters.dt = self.integrator.dt
        self.parameters.equilibration_integrator = self.integrator.equilibration_type
        self.parameters.production_integrator = self.integrator.production_type
        if self.parameters.magnetized:
            self.parameters.magnetization_integrator = self.integrator.magnetization_type
    
    def _setup_io_phases(self) -> None:
        """Setup I/O for different simulation phases."""
        self.io.copy_params(self.parameters)
        self.io.setup_checkpoint(self.parameters, self.particles, phase="equilibration")
        
        if self.parameters.magnetized and self.parameters.electrostatic_equilibration:
            self.io.setup_checkpoint(self.parameters, self.particles, phase="magnetization")
            
        self.io.setup_checkpoint(self.parameters, self.particles, phase="production")
    
    def _print_initialization_timing(self, t0: float, time_ptcls: float, 
                                   time_pot: float, time_end: float) -> None:
        """Print timing information for initialization phases."""
        self.io.time_stamp("Particles Initialization", 
                          self.timer.time_division(time_ptcls - t0))
        self.io.time_stamp("Potential Initialization", 
                          self.timer.time_division(time_pot - time_ptcls))
        self.io.time_stamp("Total Simulation Initialization", 
                          self.timer.time_division(time_end - t0))

    def print_initial_state(self) -> None:
        """Print the initial energies and state of the system."""
        init_eng = " Initial Energies "
        msg = f"\n\n{init_eng:-^70}\n"
        msg += "Initial temperature and kinetic energy of each species\n"

        self.particles.calculate_species_kinetic_temperature()
        self.particles.calculate_species_potential_energy()

        factor = (self.parameters.J2erg if self.parameters.units == "mks" 
                 else 1.0 / self.parameters.J2erg)

        for sp, kp, tp, pot_sp in zip(
            self.species,
            self.particles.species_kinetic_energy,
            self.particles.species_temperature,
            self.particles.species_potential_energy,
        ):
            sp_msg = (
                f"Species {sp.name} :\n"
                f"\tTemperature = {tp:.6e} {self.parameters.units_dict['temperature']} "
                f"= {tp * self.parameters.eV2K:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tKinetic Energy = {kp:.6e} {self.parameters.units_dict['energy']} "
                f"= {kp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
                f"\tPotential Energy = {pot_sp:.6e} {self.parameters.units_dict['energy']} "
                f"= {pot_sp * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            )
            msg += sp_msg

        tot_kin_e = self.particles.species_kinetic_energy.sum()
        tot_pot_e = self.particles.species_potential_energy.sum()
        tot_e = tot_kin_e + tot_pot_e

        msg += (
            f"Initial total kinetic energy = {tot_kin_e:.6e} {self.parameters.units_dict['energy']} "
            f"= {tot_kin_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total potential energy = {tot_pot_e:.6e} {self.parameters.units_dict['energy']} "
            f"= {tot_pot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
            f"Initial total energy = {tot_e:.6e} {self.parameters.units_dict['energy']} "
            f"= {tot_e * factor / self.parameters.eV2J:.6e} {self.parameters.units_dict['electron volt']}\n"
        )
        self.io.write_to_logger(msg)


# ============================================================================
# Refactored Simulation Class
# ============================================================================

class Simulation(Process):
    """
    Refactored Simulation class with improved architecture and modularity.
    
    This class now uses composition and delegation to manage simulation phases,
    making it more maintainable, testable, and extensible.
    """

    def __init__(self, input_file: Optional[str] = None):
        super().__init__(input_file)
        self.__name__ = "simulation"
        
        # Initialize simulation phases
        self.phases: List[SimulationPhase] = [
            EquilibrationPhase(),
            MagnetizationPhase(),
            ProductionPhase()
        ]
    
    def run(self) -> None:
        """
        Execute the complete simulation workflow.
        
        This method coordinates all simulation phases in a clean, 
        maintainable way using the strategy pattern.
        """
        try:
            time0 = self.timer.current()
            
            # Create simulation state object
            simulation_state = SimulationState(
                parameters=self.parameters,
                particles=self.particles,
                integrator=self.integrator,
                potential=self.potential,
                io=self.io,
                timer=self.timer,
                species=self.species
            )
            
            # Execute each phase that should run
            executed_phases = []
            for phase in self.phases:
                if phase.should_run(self.parameters):
                    self.io.write_to_logger(f"Starting {phase.name} phase...")
                    phase.execute(simulation_state)
                    executed_phases.append(phase.name)
            
            # Calculate and log total time
            time_total = self.timer.current()
            self.io.time_stamp("Total", self.timer.time_division(time_total - time0))
            
            # Log completion summary
            phase_list = ", ".join(executed_phases) if executed_phases else "none"
            self.io.write_to_logger(f"Simulation completed. Executed phases: {phase_list}")
            
            # Calculate directory sizes (if needed)
            if hasattr(self, 'directory_sizes'):
                self.directory_sizes()
                
        except Exception as e:
            self.io.write_to_logger(f"Simulation failed: {str(e)}", level="ERROR")
            raise
    
    def add_phase(self, phase: SimulationPhase) -> None:
        """
        Add a custom simulation phase.
        
        Parameters
        ----------
        phase : SimulationPhase
            Custom phase to add to the simulation
        """
        if not isinstance(phase, SimulationPhase):
            raise TypeError("Phase must be an instance of SimulationPhase")
        
        self.phases.append(phase)
    
    def remove_phase(self, phase_name: str) -> bool:
        """
        Remove a simulation phase by name.
        
        Parameters
        ----------
        phase_name : str
            Name of the phase to remove
            
        Returns
        -------
        bool
            True if phase was found and removed, False otherwise
        """
        for i, phase in enumerate(self.phases):
            if phase.name == phase_name:
                del self.phases[i]
                return True
        return False
    
    def get_phase(self, phase_name: str) -> Optional[SimulationPhase]:
        """
        Get a simulation phase by name.
        
        Parameters
        ----------
        phase_name : str
            Name of the phase to retrieve
            
        Returns
        -------
        SimulationPhase or None
            The requested phase, or None if not found
        """
        for phase in self.phases:
            if phase.name == phase_name:
                return phase
        return None


# ============================================================================
# Placeholder classes for compatibility
# ============================================================================

class PreProcess(Process):
    """
    Placeholder for PreProcess class.
    To be implemented with specific preprocessing logic.
    """
    
    def __init__(self, input_file: Optional[str] = None):
        super().__init__(input_file)
        self.__name__ = "preprocessing"


class PostProcess(Process):
    """
    Placeholder for PostProcess class.
    To be implemented with specific postprocessing logic.
    """
    
    def __init__(self, input_file: Optional[str] = None, grab_last_step: bool = False):
        super().__init__(input_file)
        self.__name__ = "postprocessing"
        self.grab_last_step = grab_last_step