"""Welcome to Sarkas: a fast pure Python molecular dynamics software for plasmas physics."""

# Explicit exports for better import control and IDE support
__all__ = [
    "__version__",
    "install_matplotlib_styles",
    # Uncomment below when ready to expose main classes
    # "Potential",
    # "Integrator", 
    # "Thermostat",
    # "Process",
    # "Simulation",
    # "PreProcess",
    # "PostProcess", 
    # "Particles",
    # "Parameters",
    # "Species",
    # "InputOutput",
]

# Enforce Python version check during package import.
# This is the same check as the one at the top of setup.py
import sys

if sys.version_info < (3, 7):
    raise ImportError("Sarkas requires Python 3.7 or later")

# Clear sys reference early to avoid holding onto it
del sys

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------

### This imports make the first import of Sarkas slow. That is why they are commented
# from .core import Parameters
# from .particles import Particles
# from .plasma import Species
# from .potentials.core import Potential
# from .processes import PostProcess, PreProcess, Process, Simulation
# from .time_evolution.integrators import Integrator
# from .time_evolution.thermostats import Thermostat
# from .utilities.io import InputOutput

# Lazy version detection to speed up imports
def _get_version():
    """Get package version with lazy loading."""
    try:
        # Use modern importlib.metadata instead of deprecated pkg_resources
        try:
            # Python 3.8+
            from importlib.metadata import version, PackageNotFoundError
        except ImportError:
            # Python < 3.8, fallback to importlib_metadata
            from importlib_metadata import version, PackageNotFoundError
        
        return version("sarkas")
    except PackageNotFoundError:
        # package is not installed, try setuptools_scm
        try:
            from setuptools_scm import get_version
            return get_version(root="..", relative_to=__file__, fallback_version="unknown")
        except (ModuleNotFoundError, LookupError):
            return "unknown"

# Use a property-like approach for lazy evaluation
class _VersionProperty:
    def __init__(self):
        self._version = None
        self._warned = False
    
    def __str__(self):
        if self._version is None:
            self._version = _get_version()
            if self._version == "unknown" and not self._warned:
                import warnings
                warnings.warn(
                    "sarkas.__version__ not generated (set to 'unknown'), "
                    "Sarkas is not an installed package or setuptools_scm is not available.",
                    RuntimeWarning,
                    stacklevel=2
                )
                self._warned = True
        return self._version
    
    def __repr__(self):
        return f"'{self}'"

#: Sarkas version string (lazy-loaded)
__version__ = _VersionProperty()

def install_matplotlib_styles():
    """
    Install Sarkas matplotlib styles to matplotlib's stylelib directory.
    
    This function can be called manually if the automatic installation
    during pip install didn't work properly.
    
    Examples
    --------
    >>> import sarkas
    >>> sarkas.install_matplotlib_styles()
    
    Returns
    -------
    bool
        True if installation was successful, False otherwise.
    """
    import os
    import glob
    
    try:
        import matplotlib as mpl
        import shutil
        
        # Get the path to this package
        package_dir = os.path.dirname(__file__)
        style_source_path = os.path.join(package_dir, "mplstyles")
        
        if not os.path.exists(style_source_path):
            print(f"Error: Matplotlib styles directory not found at {style_source_path}")
            return False
            
        # Get matplotlib's stylelib directory
        stylelib_path = os.path.join(mpl.get_data_path(), "stylelib")
        
        # Find all .mplstyle files
        style_files = glob.glob(os.path.join(style_source_path, "*.mplstyle"))
        
        if not style_files:
            print(f"Warning: No .mplstyle files found in {style_source_path}")
            return False
            
        # Ensure the matplotlib stylelib directory exists
        os.makedirs(stylelib_path, exist_ok=True)
        
        installed_count = 0
        for style_file in style_files:
            style_name = os.path.basename(style_file)
            dest_path = os.path.join(stylelib_path, style_name)
            
            try:
                shutil.copy2(style_file, dest_path)
                print(f"✓ Installed {style_name} to {dest_path}")
                installed_count += 1
            except (OSError, IOError) as e:
                print(f"✗ Failed to install {style_name}: {e}")
                
        if installed_count > 0:
            print(f"\nSuccessfully installed {installed_count} matplotlib style(s).")
            print("You can now use them with: plt.style.use('MSUstyle') or plt.style.use('PUBstyle')")
            return True
        else:
            print("No styles were installed successfully.")
            return False
            
    except ImportError:
        print("Error: matplotlib is not installed. Please install matplotlib first.")
        return False
    except Exception as e:
        print(f"Error during style installation: {e}")
        return False
