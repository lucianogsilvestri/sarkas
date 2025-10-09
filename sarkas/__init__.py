"""
Sarkas - A fast pure-python Molecular Dynamics suite for plasmas

Copyright (c) Murillo Group
License: MIT
"""

import sys

__version__ = "1.1.0"
__minimum_python_version__ = "3.9"

# Check Python version
if sys.version_info < tuple(int(val) for val in __minimum_python_version__.split(".")):
    raise RuntimeError(
        f"Sarkas requires Python {__minimum_python_version__} or later. "
        f"You are using Python {sys.version_info.major}.{sys.version_info.minor}."
    )

# Note: Plotly templates (MSUstyle and PUBstyle) are automatically registered
# via a .pth file installed during package installation. They are available
# in all Python sessions without needing to explicitly import sarkas.
# If you need to manually register them, use:
#   from sarkas.plotting.styles import register_all_styles
#   register_all_styles()

# Import main modules here
# from .core import ...
# from .tools import ...

__all__ = [
    "__version__",
]