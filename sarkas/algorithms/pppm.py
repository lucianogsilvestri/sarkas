"""
Module for Particle-Particle Particle-Mesh (PPPM) algorithm for efficient Coulomb interactions.
"""

from numpy import zeros, zeros_like, array, sqrt, pi, full, int64, int32, log2, round
from .base import InteractionSolverBase
from .cell_list import LinkedCellList
from .verlet_list import VerletList

# Import your existing PPPM functions directly
from .pppm_mesh_functions import (
    force_optimized_green_function,
    update as pppm_update
)


class PPPM(InteractionSolverBase):
    """
    Particle-Particle Particle-Mesh (PPPM) algorithm for Coulomb interactions.
    
    PPPM splits the Coulomb interaction into short-range and long-range parts:
    - Short-range: Computed using real-space methods (LinkedCellList or VerletList)  
    - Long-range: Computed using mesh-based Fourier methods
    
    This implementation uses your existing optimized PPPM mesh functions and
    allows users to choose between LinkedCellList (default) or VerletList
    for the short-range interactions.
    
    Attributes
    ----------
    short_range_solver : InteractionSolverBase
        Solver for short-range interactions (LinkedCellList or VerletList)
    pppm_alpha_ewald : float
        Ewald parameter for real/reciprocal space split
    pppm_mesh : numpy.ndarray
        Number of mesh points in each dimension
    pppm_h_array : numpy.ndarray
        Spacing between mesh points
    pppm_cao : numpy.ndarray
        Charge assignment order for each dimension
    pppm_green_function : numpy.ndarray
        Optimized Green's function in Fourier space
    """

    def __init__(self):
        """Initialize the PPPM solver."""
        super().__init__()
        self.type = 'pppm'
        
        # Short-range solver (composition)
        self.short_range_solver = None
        self.short_range_method = 'linked_cell_list'
        
        # PPPM parameters (using your naming convention)
        self.pppm_alpha_ewald = None
        self.pppm_mesh = None
        self.pppm_h_array = None
        self.pppm_h_volume = None
        self.pppm_cao = None
        self.pppm_aliases = None
        
        # Pre-computed arrays for mesh calculations (using your naming)
        self.pppm_green_function = None
        self.pppm_kx = None
        self.pppm_ky = None
        self.pppm_kz = None
        self.pppm_pm_err = None
        
        # System parameters
        self.rc = None
        self.force_error = 1e-5
        self.a_ws = None
        self.screening_length = None
        self.fourpie0 = None
        self.total_num_ptcls = None
        self.box_volume = None

    def setup(self, params, **kwargs):
        """
        Initialize the PPPM solver with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions
            - cutoff_radius : float, short-range cutoff (will be stored as rc)
            - a_ws : float, Wigner-Seitz radius
            - Other parameters for short-range solver
        **kwargs : dict
            PPPM-specific parameters:
            - short_range_method : str, 'linked_cell_list' or 'verlet_list' (default: 'linked_cell_list')
            - pppm_alpha_ewald : float, Ewald parameter (default: auto-estimate)
            - pppm_mesh : array-like, mesh dimensions (default: auto-estimate)
            - pppm_cao : array-like or int, charge assignment orders (default: auto-select)
            - pppm_aliases : array-like, anti-aliasing parameters (default: [3, 3, 3])
            - force_error : float, target force error (default: 1e-5)
        """
        self.box_lengths = params.box_lengths
        self.rc = params.cutoff_radius
        self.a_ws = params.a_ws
        self.box_volume = self.box_lengths.prod()
        
        # Get other parameters
        self.dimensions = params.dimensions
        self.total_num_density = params.total_num_density
        self.units_dict = params.units_dict
        self.force_error = kwargs.get('force_error', 1e-5)
        
        # Setup short-range solver
        self._setup_short_range_solver(params, **kwargs)
        
        # Setup PPPM parameters
        self._setup_pppm_parameters(params, **kwargs)
        
        # Run PPPM setup using your existing function
        self.pppm_setup()

    def _setup_short_range_solver(self, params, **kwargs):
        """Setup the short-range interaction solver."""
        self.short_range_method = kwargs.get('short_range_method', 'linked_cell_list')
        
        if self.short_range_method == 'linked_cell_list':
            self.short_range_solver = LinkedCellList()
        elif self.short_range_method == 'verlet_list':
            self.short_range_solver = VerletList()
        else:
            raise ValueError(f"Unsupported short-range method: {self.short_range_method}")
        
        # Setup the short-range solver with original parameters
        self.short_range_solver.setup(params, **kwargs)

    def _setup_pppm_parameters(self, params, **kwargs):
        """Setup PPPM-specific parameters using your parameter estimation."""
        # Use provided parameters or estimate them
        if 'pppm_mesh' in kwargs:
            self.pppm_mesh = array(kwargs['pppm_mesh'], dtype=int64)
        if 'pppm_alpha_ewald' in kwargs:
            self.pppm_alpha_ewald = kwargs['pppm_alpha_ewald']
        if 'pppm_cao' in kwargs:
            if isinstance(kwargs['pppm_cao'], int):
                self.pppm_cao = full(3, kwargs['pppm_cao'], dtype=int64)
            else:
                self.pppm_cao = array(kwargs['pppm_cao'], dtype=int64)
        if 'pppm_aliases' in kwargs:
            self.pppm_aliases = array(kwargs['pppm_aliases'], dtype=int64)
        
        # If not provided, use your estimation method
        if (self.pppm_mesh is None or self.pppm_alpha_ewald is None or 
            self.pppm_cao is None or self.pppm_aliases is None):
            self.estimate_pppm_parameters(params)

    def estimate_pppm_parameters(self, params):
        """
        Estimate optimal PPPM parameters based on your existing method.
        Simplified version of your parameter estimation.
        """
        # Initial mesh estimate: 
        pppm_h_array = full(3, 0.5 * self.a_ws, dtype=float)
        # Mesh size is power of 2 of L/h
        pppm_mesh = (self.box_lengths / pppm_h_array).astype(int32)
        # Find the closest power of 2
        self.pppm_mesh = 2**(round(log2(pppm_mesh))).astype(int64)
        self.pppm_h_array = self.box_lengths / self.pppm_mesh

        # First calculate alpha using the relation alpha = 0.3 * pppm_mesh[0] / box_length
        alpha_initial = 0.3 / min(self.pppm_h_array)
        
        # Then calculate rc from alpha_initial using pppm_alpha * rc = 3.6
        rc_initial = 3.6 / alpha_initial
        
        # Ensure rc is at least 3 times the Wigner-Seitz radius
        self.rc = max(rc_initial, 3 * self.a_ws)
        
        # Recalculate alpha using the constraint pppm_alpha * rc = 3.6
        self.ppmp_alpha_ewald = 3.6 / self.rc
        
        # Choose charge assignment order based on force_error
        if self.force_error <= 1e-7:
            self.pppm_cao = full(3, 7, dtype=int64)
        elif self.force_error <= 1e-5:
            self.pppm_cao = full(3, 6, dtype=int64)
        elif self.force_error <= 1e-4:
            self.pppm_cao = full(3, 4, dtype=int64)
        else:
            self.pppm_cao = full(3, 3, dtype=int64)
        
        self.pppm_aliases = array([3, 3, 3], dtype=int64)

    def pppm_setup(self):
        """Calculate the PPPM parameters using your existing setup function."""
        # Validate arrays
        if isinstance(self.pppm_mesh, list):
            self.pppm_mesh = array(self.pppm_mesh, dtype=int64)
        
        if isinstance(self.pppm_aliases, list):
            self.pppm_aliases = array(self.pppm_aliases, dtype=int64)
        
        if isinstance(self.pppm_cao, int):
            self.pppm_cao = full(3, self.pppm_cao, dtype=int64)
        elif isinstance(self.pppm_cao, list):
            self.pppm_cao = array(self.pppm_cao, dtype=int64)

        if self.pppm_cao.max() > 7:
            raise AttributeError("Charge assignment order > 7 not supported. Please choose <= 7")

        # Calculate mesh parameters
        self.pppm_h_array = self.box_lengths / self.pppm_mesh
        # To avoid division by zero
        mask = self.pppm_h_array == 0.0
        self.pppm_h_array[mask] = 1.0
        self.pppm_h_volume = self.pppm_h_array.prod()
        # To avoid unnecessary loops
        self.pppm_aliases[mask] = 0

        # Pack constants for Green's function calculation
        kappa = 1.0 / self.screening_length if hasattr(self, 'screening_length') and self.screening_length else 0.0
        constants = array([kappa, self.pppm_alpha_ewald, self.fourpie0])

        # Calculate the Optimized Green's Function using your function
        (self.pppm_green_function, self.pppm_kx, self.pppm_ky, 
         self.pppm_kz, self.pppm_pm_err) = force_optimized_green_function(
            self.box_lengths, self.pppm_h_array, self.pppm_mesh, 
            self.pppm_aliases, self.pppm_cao, constants
        )

        # Complete PM Force error calculation
        if hasattr(self, 'total_num_ptcls') and self.total_num_ptcls:
            self.pppm_pm_err *= sqrt(self.total_num_ptcls) * self.a_ws**2 * self.fourpie0
            self.pppm_pm_err /= self.box_volume ** (2.0 / 3.0)

    def update(self, ptcls, potential):
        """
        Calculate particle interactions using PPPM method.
        
        This method splits the Coulomb interaction into short-range and long-range
        parts, computing each with the appropriate method.
        
        Parameters
        ----------
        ptcls : object
            Particles data containing positions, velocities, charges, masses, etc.
        potential : object
            Potential class (must be Coulomb-type for PPPM)
        """
        # Step 1: Compute short-range interactions
        self._compute_short_range(ptcls, potential)
        
        # Step 2: Compute long-range interactions using your mesh functions
        self._compute_long_range(ptcls, potential)
        
        # Step 3: Combine results
        self._combine_results(ptcls)

    def _compute_short_range(self, ptcls, potential):
        """Compute short-range part using chosen real-space method."""
        # Use the chosen short-range solver with original potential
        # The short-range/long-range split is handled in the mesh calculation
        self.short_range_solver.update(ptcls, potential)
        
        # Store short-range results
        self.short_range_energy = ptcls.potential_energy.copy()
        self.short_range_forces = ptcls.acc.copy()
        self.short_range_virial = ptcls.virial_species_tensor.copy()
        self.short_range_heat_flux = ptcls.heat_flux_species_tensor.copy()

    def _compute_long_range(self, ptcls, potential):
        """Compute long-range part using your mesh functions."""
        # Use your existing PPPM update function
        pot_long_range, acc_long_range = pppm_update(
            ptcls.pos, ptcls.charges, ptcls.masses,
            self.pppm_mesh, self.pppm_h_array, self.pppm_h_volume, self.box_volume,
            self.pppm_green_function, self.pppm_kx, self.pppm_ky, self.pppm_kz, 
            self.pppm_cao
        )
        
        # Store long-range results
        self.long_range_energy = pot_long_range
        self.long_range_forces = acc_long_range

    def _combine_results(self, ptcls):
        """Combine short-range and long-range contributions."""

        # Ewald self-energy of each particle
        u_ewald -= ptcls.charges**2 * self.pppm_alpha_ewald / sqrt(pi) / self.fourpie0

        # J-M.Caillol, J Chem Phys 101 6080 (1994) https: // doi.org / 10.1063 / 1.468422
        dipoles = ptcls.charges[:, newaxis] *  ptcls.pos / sqrt(self.fourpie0)
        vol_const = 2.0 * pi / (3.0 * self.box_volume)
        ptcls.dipole_energy = vol_const * (dipoles**2).sum(axis = 1) 

        dipole_force = -vol_const *  ptcls.charges[:, newaxis] * dipoles.sum(axis = 0)  / sqrt(self.fourpie0)
        dipole_acc = dipole_force / ptcls.masses[:, newaxis]

        ptcls.potential_energy = self.short_range_energy + self.long_range_energy + u_ewald + ptcls.dipole_energy
        # Combine forces
        ptcls.acc = self.short_range_forces + self.long_range_forces + dipole_acc
        
        # Virial and heat flux (only short-range computed for now)
        ptcls.virial_species_tensor = self.short_range_virial
        ptcls.heat_flux_species_tensor = self.short_range_heat_flux

    def pretty_print(self):
        """Print algorithm information and parameters."""
        msg = f"\nINTERACTION SOLVER: Particle-Particle Particle-Mesh (PPPM)\n"
        
        msg += f"Short-range method: {self.short_range_method}\n"
        
        if self.pppm_alpha_ewald is not None:
            msg += f"Ewald parameter (alpha): {self.pppm_alpha_ewald:.6e}\n"
            
        if self.pppm_mesh is not None:
            msg += f"Mesh sizes: {self.pppm_mesh}\n"
            msg += f"Mesh spacings: {self.pppm_h_array}\n"
            
        if self.pppm_cao is not None:
            msg += f"Charge assignment orders: {self.pppm_cao}\n"
            
        if self.pppm_pm_err is not None:
            msg += f"Estimated PM error: {self.pppm_pm_err:.2e}\n"
            
        if self.rc is not None:
            msg += f"Short-range cutoff: {self.rc:.6e}\n"
            
        msg += "Computational complexity: O(N log N)\n"
        msg += "Suitable for: Large systems with long-range Coulomb interactions\n"
        
        return msg

    def get_pppm_statistics(self):
        """
        Get statistics about PPPM calculation.
        
        Returns
        -------
        dict
            Dictionary containing PPPM statistics
        """
        stats = {
            'short_range_method': self.short_range_method,
            'ppmp_alpha_ewald': self.pppm_alpha_ewald,
            'pppm_mesh': self.pppm_mesh.tolist() if self.pppm_mesh is not None else None,
            'pppm_cao': self.pppm_cao.tolist() if self.pppm_cao is not None else None,
            'pppm_pm_err': self.pppm_pm_err,
            'pppm_h_volume': self.pppm_h_volume,
            'total_mesh_points': self.pppm_mesh.prod() if self.pppm_mesh is not None else None,
            'rc': self.rc,
            'force_error_target': self.force_error
        }
        
        # Add short-range solver statistics if available
        if hasattr(self.short_range_solver, 'get_neighbor_statistics'):
            stats['short_range_stats'] = self.short_range_solver.get_neighbor_statistics()
        elif hasattr(self.short_range_solver, 'get_statistics'):
            stats['short_range_stats'] = self.short_range_solver.get_statistics()
        
        return stats

    def set_pppm_parameters(self, **kwargs):
        """
        Manually set PPPM parameters.
        
        Parameters
        ----------
        **kwargs : dict
            PPPM parameters to set:
            - pppm_alpha_ewald : float
            - pppm_mesh : array-like
            - pppm_cao : array-like or int
            - pppm_aliases : array-like
        """
        if 'pppm_alpha_ewald' in kwargs:
            self.pppm_alpha_ewald = kwargs['pppm_alpha_ewald']
            
        if 'pppm_mesh' in kwargs:
            self.pppm_mesh = array(kwargs['pppm_mesh'], dtype=int64)
            
        if 'pppm_cao' in kwargs:
            if isinstance(kwargs['pppm_cao'], int):
                self.pppm_cao = full(3, kwargs['pppm_cao'], dtype=int64)
            else:
                self.pppm_cao = array(kwargs['pppm_cao'], dtype=int64)
                
        if 'pppm_aliases' in kwargs:
            self.pppm_aliases = array(kwargs['pppm_aliases'], dtype=int64)
        
        # Re-run setup if parameters changed
        if any(param in kwargs for param in ['pppm_alpha_ewald', 'pppm_mesh', 'pppm_cao', 'pppm_aliases']):
            self.pppm_setup()

    def validate_pppm_parameters(self):
        """
        Validate PPPM parameters and provide recommendations.
        
        Returns
        -------
        dict
            Dictionary containing validation results
        """
        issues = []
        recommendations = []
        
        if self.pppm_mesh is not None:
            # Check if mesh sizes are powers of 2 (efficient for FFT)
            for i, mesh_size in enumerate(self.pppm_mesh):
                if mesh_size > 0 and (mesh_size & (mesh_size - 1)) != 0:
                    issues.append(f"Mesh size {mesh_size} in dimension {i} is not a power of 2")
                    recommendations.append("Use power of 2 mesh sizes for optimal FFT performance")
                    break
        
        if self.pppm_alpha_ewald is not None and self.rc is not None:
            alpha_rc = self.pppm_alpha_ewald * self.rc
            if alpha_rc < 2.0:
                issues.append("Alpha*rc < 2.0 may give poor accuracy")
                recommendations.append("Increase alpha or cutoff radius")
            elif alpha_rc > 5.0:
                issues.append("Alpha*rc > 5.0 may be inefficient")
                recommendations.append("Decrease alpha for better efficiency")
        
        if self.pppm_cao is not None:
            if self.pppm_cao.max() > 6:
                issues.append("Very high charge assignment order may be unnecessarily expensive")
                recommendations.append("Consider cao <= 6 for good accuracy/efficiency balance")
        
        return {
            'issues': issues,
            'recommendations': recommendations,
            'status': 'good' if not issues else 'needs_attention'
        }