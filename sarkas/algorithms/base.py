from abc import ABC, abstractmethod


class InteractionSolverBase(ABC):
    """
    Abstract base class for particle interaction solvers in molecular dynamics simulations.

    All interaction solvers must implement the setup() and update() methods to handle
    the computation of forces, energies, and other particle interaction properties.
    """

    def __init__(self):
        """Initialize the interaction solver."""
        self.type = None
        self.box_lengths = None

    @abstractmethod
    def setup(self, params, **kwargs):
        """
        Initialize the solver with simulation parameters.

        Parameters
        ----------
        params : object
            Simulation parameters object containing box_lengths, cutoff_radius, etc.
        **kwargs : dict
            Additional solver-specific parameters (e.g., precision for FMM,
            optimization flags for PPPM, etc.)
        """
        pass

    @abstractmethod
    def update(self, ptcls, potential):
        """
        Compute particle interactions and update particle properties.

        This method should compute and update the following particle properties:
        - ptcls.potential_energy : particle potential energies
        - ptcls.acc : particle accelerations
        - ptcls.virial_species_tensor : virial tensor (if applicable)
        - ptcls.heat_flux_species_tensor : heat flux tensor (if applicable)

        Parameters
        ----------
        ptcls : object
            Particles object containing positions, velocities, charges, masses, etc.
        potential : object
            Potential object containing force functions, parameters, etc.
        """
        pass

    def pretty_print(self):
        """
        Print solver information and parameters.

        Override this method in derived classes to provide solver-specific
        information about computational parameters, efficiency metrics, etc.
        """
        print(f"\nINTERACTION SOLVER: {self.type}")
        if self.box_lengths is not None:
            print(f"Box lengths: {self.box_lengths}")

    def validate_inputs(self, ptcls, potential):
        """
        Validate input parameters before computation.

        Parameters
        ----------
        ptcls : object
            Particles object
        potential : object
            Potential object

        Raises
        ------
        ValueError
            If required attributes are missing or have incompatible dimensions
        """
        required_ptcl_attrs = ["pos", "vel", "id", "species_masses"]
        for attr in required_ptcl_attrs:
            if not hasattr(ptcls, attr):
                raise ValueError(f"Particles object missing required attribute: {attr}")

        required_potential_attrs = ["force", "matrix"]
        for attr in required_potential_attrs:
            if not hasattr(potential, attr):
                raise ValueError(f"Potential object missing required attribute: {attr}")

        # Check dimensions
        if ptcls.pos.shape[0] != ptcls.vel.shape[0]:
            raise ValueError("Position and velocity arrays have incompatible dimensions")

        if ptcls.pos.shape[1] != 3:
            raise ValueError("Position array must have shape (N, 3)")

    def allocate_output_arrays(self, ptcls, potential):
        """
        Allocate output arrays for computed quantities.

        This is a utility method that derived classes can use to ensure
        consistent array allocation.

        Parameters
        ----------
        ptcls : object
            Particles object
        potential : object
            Potential object

        Returns
        -------
        dict
            Dictionary containing allocated arrays for:
            - 'potential_energy' : per-particle potential energies
            - 'acceleration' : per-particle accelerations
            - 'virial_species_tensor' : species-pair virial tensor
            - 'heat_flux_species_tensor' : species-pair heat flux tensor
        """
        from numpy import zeros

        N = ptcls.pos.shape[0]
        num_species = potential.matrix.shape[0]

        arrays = {
            "potential_energy": zeros(N),
            "acceleration": zeros((N, 3)),
            "virial_species_tensor": zeros((3, 3, num_species, num_species)),
            "heat_flux_species_tensor": zeros((3, num_species, num_species)),
        }

        return arrays
