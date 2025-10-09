"""
Sarkas setup script with custom post-install commands.

This setup.py is kept for backward compatibility and to handle
custom installation steps (matplotlib and Plotly style installation).
The main configuration is now in pyproject.toml.
"""

import os
import shutil
import site
from pathlib import Path

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


def install_matplotlib_styles():
    """
    Install matplotlib style files to the matplotlib stylelib directory.
    """
    try:
        import matplotlib as mpl
        
        base_library_path = Path(mpl.get_data_path()) / "stylelib"
        style_path = Path.cwd() / "sarkas" / "plotting" / "mplstyles"
        style_files = list(style_path.glob("*.mplstyle"))
        
        if not style_files:
            print("Warning: No .mplstyle files found in sarkas/plotting/mplstyles/")
            return
        
        # Copy style files to matplotlib directory
        for style_file in style_files:
            dest = base_library_path / style_file.name
            shutil.copy2(style_file, dest)
            print(f"✓ {style_file.name} installed to matplotlib stylelib")
            
    except ImportError:
        print("Warning: matplotlib not found. Skipping matplotlib style installation.")
    except Exception as e:
        print(f"Warning: Could not install matplotlib styles: {e}")


def install_plotly_auto_registration():
    """
    Install a .pth file that automatically registers Plotly templates on Python startup.
    This is the cleanest way to make templates available without explicit imports.
    """
    try:
        # Get site-packages directory
        site_packages = Path(site.getsitepackages()[0])
        
        # Create .pth file that imports and registers templates
        pth_file = site_packages / "sarkas_plotly_templates.pth"
        
        # Content: silently register templates on Python startup
        pth_content = (
            "import sys; "
            "exec(\"try:\\n"
            "    from sarkas.plotting.styles import register_all_styles\\n"
            "    register_all_styles()\\n"
            "except: pass\\n\")"
        )
        
        with open(pth_file, 'w') as f:
            f.write(pth_content)
        
        print(f"✓ Plotly auto-registration installed")
        print(f"  Location: {pth_file}")
        print(f"  Templates will be available in all Python sessions")
        return True
        
    except Exception as e:
        print(f"⚠ Could not install auto-registration: {e}")
        print("  Plotly templates will be registered when you 'import sarkas'")
        return False


def verify_plotly_templates():
    """
    Verify that Plotly templates can be registered.
    """
    try:
        from sarkas.plotting.styles import register_all_styles
        register_all_styles()
        
        import plotly.io as pio
        if "MSUstyle" in pio.templates and "PUBstyle" in pio.templates:
            print("✓ Plotly templates verified: MSUstyle and PUBstyle")
            return True
        else:
            print("⚠ Warning: Templates registration may have failed")
            return False
            
    except ImportError:
        print("⚠ Plotly not installed. Templates will be available after installing plotly.")
        return False
    except Exception as e:
        print(f"⚠ Could not verify templates: {e}")
        return False


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    
    def run(self):
        develop.run(self)
        print("\n" + "="*70)
        print("Installing Sarkas custom styles...")
        print("="*70)
        
        # Install matplotlib styles
        install_matplotlib_styles()
        
        # Install Plotly auto-registration
        install_plotly_auto_registration()
        
        # Verify templates work
        verify_plotly_templates()
        
        print("="*70)
        print("✓ Installation complete!")
        print("\nUsage:")
        print("  Matplotlib: plt.style.use('MSUstyle') or plt.style.use('PUBstyle')")
        print("  Plotly:     fig.update_layout(template='MSUstyle') or 'PUBstyle'")
        print("="*70 + "\n")


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    
    def run(self):
        install.run(self)
        print("\n" + "="*70)
        print("Installing Sarkas custom styles...")
        print("="*70)
        
        # Install matplotlib styles
        install_matplotlib_styles()
        
        # Install Plotly auto-registration
        install_plotly_auto_registration()
        
        # Verify templates work
        verify_plotly_templates()
        
        print("="*70)
        print("✓ Installation complete!")
        print("\nUsage:")
        print("  Matplotlib: plt.style.use('MSUstyle') or plt.style.use('PUBstyle')")
        print("  Plotly:     fig.update_layout(template='MSUstyle') or 'PUBstyle'")
        print("\nNote: Plotly templates are automatically available in all Python")
        print("      sessions without needing to import sarkas first!")
        print("="*70 + "\n")


# Run setup with custom commands
# Main configuration is in pyproject.toml
setup(
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
)