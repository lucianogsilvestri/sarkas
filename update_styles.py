#!/usr/bin/env python
"""
Update Sarkas styles without full reinstallation.

This script updates matplotlib style files and re-registers Plotly templates
without needing to reinstall the entire package.

Usage:
    python update_styles.py
"""

import shutil
import sys
from pathlib import Path


def update_matplotlib_styles():
    """Update matplotlib style files."""
    print("\n📊 Updating matplotlib styles...")
    try:
        import matplotlib as mpl
        
        # Determine style source path
        if Path("sarkas", "plotting", "mplstyles").exists():
            style_source = Path("sarkas", "plotting", "mplstyles")
        elif Path("mplstyles").exists():
            style_source = Path("mplstyles")
        else:
            print("  ✗ Could not find style source directory")
            print("    Expected: sarkas/plotting/mplstyles/ or mplstyles/")
            return False
        
        style_dest = Path(mpl.get_data_path()) / "stylelib"
        
        # Get all .mplstyle files
        style_files = list(style_source.glob("*.mplstyle"))
        
        if not style_files:
            print(f"  ✗ No .mplstyle files found in {style_source}")
            return False
        
        # Copy each style file
        for style_file in style_files:
            dest = style_dest / style_file.name
            shutil.copy2(style_file, dest)
            print(f"  ✓ Updated {style_file.name}")
        
        print(f"  ✓ Successfully updated {len(style_files)} matplotlib style(s)")
        return True
        
    except ImportError:
        print("  ✗ matplotlib not installed")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def update_plotly_templates():
    """Re-register Plotly templates."""
    print("\n📈 Updating Plotly templates...")
    try:
        import plotly.io as pio
        
        # Remove existing templates if present
        if 'MSUstyle' in pio.templates:
            del pio.templates['MSUstyle']
        if 'PUBstyle' in pio.templates:
            del pio.templates['PUBstyle']
        
        # Re-register templates
        from sarkas.plotting.styles import register_all_styles
        register_all_styles()
        
        # Verify registration
        if 'MSUstyle' in pio.templates and 'PUBstyle' in pio.templates:
            print("  ✓ MSUstyle registered")
            print("  ✓ PUBstyle registered")
            return True
        else:
            print("  ✗ Template registration failed")
            return False
            
    except ImportError as e:
        print(f"  ✗ Could not import required modules: {e}")
        print("    Make sure plotly and sarkas are installed")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def verify_styles():
    """Verify that styles are available."""
    print("\n🔍 Verifying installation...")
    
    # Check matplotlib
    try:
        import matplotlib.pyplot as plt
        available_styles = plt.style.available
        
        msu_available = 'MSUstyle' in available_styles
        pub_available = 'PUBstyle' in available_styles
        
        print(f"  Matplotlib MSUstyle: {'✓' if msu_available else '✗'}")
        print(f"  Matplotlib PUBstyle: {'✓' if pub_available else '✗'}")
        
    except ImportError:
        print("  ⚠ matplotlib not available for verification")
    
    # Check Plotly
    try:
        import plotly.io as pio
        
        msu_available = 'MSUstyle' in pio.templates
        pub_available = 'PUBstyle' in pio.templates
        
        print(f"  Plotly MSUstyle:     {'✓' if msu_available else '✗'}")
        print(f"  Plotly PUBstyle:     {'✓' if pub_available else '✗'}")
        
    except ImportError:
        print("  ⚠ plotly not available for verification")


def main():
    """Main function."""
    print("="*60)
    print("Sarkas Styles Update Script")
    print("="*60)
    
    # Check if running from correct directory
    if not (Path("sarkas").exists() or Path("setup.py").exists()):
        print("\n⚠ Warning: Run this script from the package root directory")
        print("  (the directory containing setup.py)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Update styles
    mpl_success = update_matplotlib_styles()
    plotly_success = update_plotly_templates()
    
    # Verify
    verify_styles()
    
    # Summary
    print("\n" + "="*60)
    if mpl_success and plotly_success:
        print("✓ All styles updated successfully!")
        print("\nNote: Restart your Python session/kernel to see changes")
    elif mpl_success or plotly_success:
        print("⚠ Partial success - some styles updated")
        print("  Check messages above for details")
    else:
        print("✗ Update failed - check messages above")
    print("="*60)
    
    return 0 if (mpl_success and plotly_success) else 1


if __name__ == "__main__":
    sys.exit(main())