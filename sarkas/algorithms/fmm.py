"""
Module for Fast Multipole Method (FMM) algorithm for efficient long-range interaction computation.
"""

from numpy import log10
from fmm3dpy import hfmm3d, lfmm3d
from .base import InteractionSolverBase


class FastMultipoles(InteractionSolverBase):
    """
    Fast Multipole Method for computing long-range particle interactions.
    
    This algorithm uses hierarchical multipole expansions to compute long-range
    Coulomb or screened Coulomb (Yukawa) interactions with O(N) or O(N log N)
    computational complexity instead of O(N²) direct summation.
    
    The implementation uses the fmm3dpy library which provides optimized
    implementations for both Laplace (Coulomb) and Helmholtz (screened) kernels.
    
    Attributes
    ----------
    precision : float
        Relative precision for FMM computation (default: 1e-5)
    """

    def __init__(self):
        """Initialize the Fast Multipole Method solver."""
        super().__init__()
        self.type = 'fast_multipoles'
        self.precision = 1e-5

    def setup(self, params, **kwargs):
        """
        Initialize the FMM solver with simulation parameters.
        
        Parameters
        ----------
        params : object
            Simulation parameters containing:
            - box_lengths : array-like, box dimensions (copied for FMM)
        **kwargs : dict
            Additional FMM-specific parameters:
            - precision : float, relative precision for FMM (default: 1e-5)
            - max_level : int, maximum tree depth (optional)
            - expansion_order : int, multipole expansion order (optional)
        """
        # Note: Original code had typo "box_lenghts" - keeping for compatibility
        # but should be "box_lengths"
        if hasattr(params, 'box_lengths'):
            self.box_lengths = params.box_lengths.copy()
        elif hasattr(params, 'box_lenghts'):
            # Handle the typo case for backward compatibility
            self.box_lengths = params.box_lenghts.copy()
        else:
            raise AttributeError("Parameters object must have 'box_lengths' attribute")
            
        # Set FMM-specific parameters
        self.precision = kwargs.get("precision", 1e-5)

    def update(self, ptcls, potential):
        """
        Calculate particle interactions using Fast Multipole Method.
        
        The method automatically selects between Laplace FMM (for pure Coulomb
        interactions) and Helmholtz FMM (for screened Coulomb/Yukawa interactions)
        based on the potential type.
        
        Parameters
        ----------
        ptcls : object
            Particles data containing:
            - pos : numpy.ndarray, particle positions (N, 3)
            - charges : numpy.ndarray, particle charges (N,)
            - masses : numpy.ndarray, particle masses (N,) or species masses
            - potential_energy : numpy.ndarray, output potential energies (N,)
            - acc : numpy.ndarray, output accelerations (N, 3)
        potential : object
            Potential class containing:
            - type : str, potential type ("coulomb" for Laplace, others for Helmholtz)
            - screening_length : float, screening length for Yukawa potential
            - fourpie0 : float, 4π×ε₀ in appropriate units
            
        Notes
        -----
        - For Coulomb potentials: uses Laplace FMM (lfmm3d)
        - For Yukawa/screened potentials: uses Helmholtz FMM (hfmm3d)
        - Only computes potential and gradient (forces), not higher-order derivatives
        - Virial and heat flux tensors are not computed (set to zero)
        """
  
        # Prepare FMM parameters
        fmm_kwargs = {
            'eps': self.precision,
            'sources': ptcls.pos.transpose(),  # FMM expects (3, N) format
            'charges': ptcls.charges,
            'pg': 2  # Compute potential and gradient
        }
        
        # Select appropriate FMM kernel based on potential type
        if potential.type == "coulomb":
            # Use Laplace FMM for pure Coulomb interactions
            out_fmm = lfmm3d(**fmm_kwargs)
        else:
            # Helmholtz parameter: zk = ikappa where kappa = 1/screening_length
            fmm_kwargs['zk'] = 1j / potential.screening_length
            out_fmm = hfmm3d(**fmm_kwargs)

        # Extract results and convert to physical units
        # Potential energy per particle
        ptcls.potential_energy = (ptcls.charges * out_fmm.pot.real / potential.fourpie0)
        
        # Acceleration = -force/mass = -(charge * gradient) / mass
        # FMM gradient has shape (3, N), transpose to (N, 3)
        forces = -(ptcls.charges * out_fmm.grad.real / potential.fourpie0)
        ptcls.acc = forces / ptcls.masses[:, None]


    def pretty_print(self):
        """Print FMM algorithm information and parameters."""
        msg = f"\nINTERACTION SOLVER: Fast Multipole Method\n"
        msg += f"Precision: {self.precision:.2e}\n"
        
        if self.box_lengths is not None:
            msg += f"Box lengths: {self.box_lengths}\n"
            
        if hasattr(self, 'max_level') and self.max_level is not None:
            msg += f"Maximum tree level: {self.max_level}\n"
            
        if hasattr(self, 'expansion_order') and self.expansion_order is not None:
            msg += f"Multipole expansion order: {self.expansion_order}\n"
            
        msg += "Computational complexity: O(N) to O(N log N)\n"
        msg += "Suitable for: Long-range Coulomb and Yukawa interactions\n"
        
        return msg

    def estimate_memory_usage(self, num_particles):
        """
        Estimate memory usage for FMM computation.
        
        Parameters
        ----------
        num_particles : int
            Number of particles in the system
            
        Returns
        -------
        dict
            Dictionary containing memory estimates in MB:
            - 'sources' : memory for particle positions and charges
            - 'tree' : estimated memory for FMM tree structure
            - 'expansions' : estimated memory for multipole expansions
            - 'total' : total estimated memory usage
        """
        # Rough memory estimates (in MB)
        bytes_per_mb = 1024 * 1024
        
        # Input data: positions (3*N) + charges (N) in double precision
        sources_mb = (4 * num_particles * 8) / bytes_per_mb
        
        # Tree structure: roughly 2*N nodes for well-balanced tree
        tree_mb = (2 * num_particles * 32) / bytes_per_mb  # Rough estimate
        
        # Multipole expansions: depends on precision and tree depth
        # Higher precision requires more terms in expansions
        expansion_factor = max(1, -int(log10(self.precision)))
        expansions_mb = (num_particles * expansion_factor * 16) / bytes_per_mb
        
        total_mb = sources_mb + tree_mb + expansions_mb
        
        return {
            'sources': sources_mb,
            'tree': tree_mb, 
            'expansions': expansions_mb,
            'total': total_mb
        }

    def set_precision(self, precision):
        """
        Set the FMM precision.
        
        Parameters
        ----------
        precision : float
            Relative precision for FMM computation.
            Smaller values give higher accuracy but slower computation.
            Typical range: 1e-3 to 1e-12
        """
        if precision <= 0:
            raise ValueError("Precision must be positive")
        if precision > 1e-1:
            print(f"Warning: Large precision value {precision} may give inaccurate results")
        
        self.precision = precision

    def supports_potential_type(self, potential_type):
        """
        Check if FMM supports the given potential type.
        
        Parameters
        ----------
        potential_type : str
            Type of potential ("coulomb", "yukawa", etc.)
            
        Returns
        -------
        bool
            True if potential type is supported by FMM
        """
        supported_types = ["coulomb", "yukawa", "screened_coulomb", "debye_huckel"]
        return potential_type.lower() in supported_types