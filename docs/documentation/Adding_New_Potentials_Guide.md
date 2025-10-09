# Adding New Potentials to Sarkas

This guide provides a comprehensive walkthrough for adding new potentials to the Sarkas molecular dynamics package. Sarkas supports both analytical and tabulated potentials, and this document covers the implementation process for both types.

## Table of Contents

1. [Overview of Sarkas Potential Architecture](#overview)
2. [Adding Analytical Potentials](#analytical-potentials)
3. [Adding Tabulated Potentials](#tabulated-potentials)
4. [Integration with Core Framework](#integration)
5. [Testing and Validation](#testing)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

## Overview of Sarkas Potential Architecture {#overview}

Sarkas uses a modular design where each potential is implemented as a separate module within the `sarkas/potentials/` directory. The core functionality is managed by the `Potential` class in `sarkas/potentials/core.py`.

### Key Components

- **Core Module** (`core.py`): Main `Potential` class that handles setup and coordination
- **Individual Potential Modules**: Each potential type has its own module (e.g., `yukawa.py`, `coulomb.py`)
- **Force Functions**: Numba-compiled functions for efficient force calculations
- **Update Functions**: Parameter setup and validation functions
- **Pretty Print Functions**: User-friendly parameter display

### Potential Matrix Structure

Each potential uses a multi-dimensional matrix to store parameters:
```python
potential.matrix = numpy.ndarray(shape=(num_species, num_species, num_parameters))
```

For species pair (i,j), `potential.matrix[i,j,:]` contains all parameters needed for force calculation.

## Adding Analytical Potentials {#analytical-potentials}

### Step 1: Create the Potential Module

Create a new file `sarkas/potentials/my_potential.py`. Each potential module requires four essential functions, each serving a specific purpose in the Sarkas framework:

#### Function Overview

1. **Force Function** (`my_potential_force`): JIT-compiled function used during simulation
   - This is the performance-critical function called millions of times during MD simulation
   - Must be decorated with `@jit` for optimal performance
   - Returns both potential energy and force magnitude for each particle pair
   - Implements the core physics of your potential model

2. **Derivatives Function** (`potential_derivatives`): Non-JIT function for analytical calculations
   - Used for post-processing analysis and thermodynamic calculations
   - Computes analytical derivatives of the potential (not forces)
   - Essential for calculating ensemble averages using integrals over the pair distribution function $g(r)$.
   - Not performance-critical. Do not JIT because this function is called by scipy.

3. **Pretty Print Function** (`pretty_print_info`): Human-readable parameter display
   - Provides user-friendly output of potential parameters
   - Called during simulation setup to display configuration
   - Should include physical units and key dimensionless parameters
   - Helps users verify their potential configuration is correct

4. **Update Parameters Function** (`update_params`): Core setup and parameter assignment
   - The most critical function that connects your potential to the Sarkas framework
   - Initializes the potential matrix with all necessary parameters
   - Assigns force and derivative functions to the potential object
   - Handles multi-species interactions and parameter validation
   - Called once during simulation initialization

Below is an example of a new potential. Read closely the docstrings for more details.

```python
r"""
Module for handling My Custom Potential.

Potential
*********

The My Custom potential between two charges :math:`q_i` and :math:`q_j` at distance :math:`r` is defined as

.. math::
    U_{ij}(r) = \frac{q_i q_j}{4 \pi \epsilon_0} \frac{A e^{-\alpha r}}{r^n}

where :math:`A` is the strength parameter, :math:`\alpha` is the decay parameter, and :math:`n` is the power.

Potential Attributes
********************

The elements of the :attr:`sarkas.potentials.core.Potential.matrix` are:

.. code-block:: python

    pot_matrix[0] = q_i*q_j/(4*pi*eps0)  # Charge factor
    pot_matrix[1] = A                     # Strength parameter
    pot_matrix[2] = alpha                 # Decay parameter
    pot_matrix[3] = n                     # Power parameter
    pot_matrix[4] = a_rs                  # Short-range cutoff

"""
from numba import jit
from numpy import exp, pi, sqrt, zeros
from ..utilities.maths import force_error_analytic_lcl


@jit(nopython=True)
def my_potential_force(r_in, pot_matrix):
    """
    JIT-compiled force function used during MD simulation.
    
    This is the heart of your potential implementation. It's called millions of times
    during a molecular dynamics simulation, so performance is critical. The function
    must be JIT-compiled with Numba's @jit decorator and follow strict type annotations.
    
    The function takes the distance between two particles and the potential parameters
    for their species pair, then returns both the potential energy and force magnitude.
    This dual return allows Sarkas to calculate both energetics and dynamics efficiently.
    
    Key Implementation Notes:
    - Use branchless programming for conditionals when possible
    - Avoid Python loops; use vectorized NumPy operations
    - Return positive force values for repulsive interactions
    - Handle short-range cutoffs to prevent divergences
    
    Parameters
    ----------
    r_in : float
        Distance between particles.
    pot_matrix : numpy.ndarray
        Potential parameters matrix for the specific species pair.
        
    Returns
    -------
    u_r : float
        Potential energy.
    f_r : float
        Force magnitude.
    """
    # Extract parameters
    q_factor = pot_matrix[0]    # q_i*q_j/(4*pi*eps0)
    A = pot_matrix[1]           # Strength parameter
    alpha = pot_matrix[2]       # Decay parameter
    n = pot_matrix[3]           # Power parameter
    a_rs = pot_matrix[4]        # Short-range cutoff
    
    # Apply short-range cutoff (branchless programming)
    r = r_in * (r_in >= a_rs) + a_rs * (r_in < a_rs)
    
    # Calculate potential
    exp_term = exp(-alpha * r)
    r_power = r**n
    u_r = q_factor * A * exp_term / r_power
    
    # Calculate force magnitude (positive for repulsive)
    # Note: Force = -dU/dr, but we return |F| with correct sign
    f_r = u_r * (n / r + alpha)
    
    return u_r, f_r


def potential_derivatives(r, pot_matrix):
    """
    Non-JIT function for calculating potential derivatives.
    
    This function serves a different purpose than the force function above. While the
    force function is optimized for speed during simulation, this function provides
    analytical derivatives needed for post-processing calculations and theoretical
    analysis.
    
    The derivatives are used in various thermodynamic calculations:
    - Radial distribution function g(r) analysis
    - Pressure calculations via the virial equation of state
    - Potential energy ensemble averages
    - Heat capacity and compressibility calculations
    
    Important: This function returns the actual mathematical derivative dU/dr, not
    the force magnitude. The relationship is F = -dU/dr, but here we return dU/dr
    directly for use in analytical expressions.
    
    Unlike the force function, this doesn't need JIT compilation since it's not
    called during the tight simulation loops. It's primarily used for analysis
    and initialization calculations.
    
    Parameters
    ----------
    r : float
        Distance between particles.
    pot_matrix : numpy.ndarray
        Potential parameters for the specific species pair.
        
    Returns
    -------
    u_r : float
        Potential energy U(r).
    dv_dr : float
        First derivative dU/dr (note: this is NOT the force).
    d2v_dr2 : float
        Second derivative d²U/dr².
    """
    # Extract parameters
    q_factor = pot_matrix[0]
    A = pot_matrix[1]
    alpha = pot_matrix[2]
    n = pot_matrix[3]
    
    # Potential
    exp_term = exp(-alpha * r)
    r_power = r**n
    u_r = q_factor * A * exp_term / r_power
    
    # First derivative (actual derivative, not force)
    dv_dr = -u_r * (n / r + alpha)
    
    # Second derivative
    d2v_dr2 = u_r * ((n / r + alpha)**2 - n / r**2)
    
    return u_r, dv_dr, d2v_dr2


def pretty_print_info(potential):
    """
    Display human-readable potential parameters.
    
    This function provides a user-friendly way to display the potential configuration
    during simulation setup. It's called automatically by Sarkas when initializing
    the simulation to help users verify their input parameters are correct.
    
    The function should display:
    - Key physical parameters with appropriate units
    - Dimensionless parameters (like coupling constants)
    - Any derived quantities that help characterize the system
    - Warning messages if parameters are in unusual ranges
    
    This is particularly important for complex potentials where users need to verify
    that their parameter choices make physical sense. The output appears in both
    the console and log files, making it valuable for debugging and documentation.
    
    Design your output to be concise but informative. Include units from the
    potential.units_dict to ensure consistency with the chosen unit system.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form with populated matrix.
    """
    msg = (
        f"Strength parameter A = {potential.matrix[0,0,1]:.6e}\n"
        f"Decay parameter α = {potential.matrix[0,0,2]:.6e} {potential.units_dict['inverse length']}\n"
        f"Power parameter n = {potential.matrix[0,0,3]:.2f}\n"
        f"Gamma_eff = {potential.coupling_constant:.2f}"
    )
    print(msg)


def update_params(potential, species):
    """
    Core function for setting up potential parameters.
    
    This is arguably the most important function in your potential module. It serves
    as the bridge between your potential implementation and the Sarkas framework,
    handling all the setup required for your potential to work within simulations.
    
    The function has several critical responsibilities:
    
    1. **Matrix Initialization**: Creates the potential.matrix with the correct
       dimensions to store parameters for all species pairs. The matrix structure
       is [num_species, num_species, num_parameters].
    
    2. **Parameter Population**: Fills the matrix with all parameters needed by
       your force function. This includes charge factors, potential-specific
       parameters, and any derived quantities.
    
    3. **Function Assignment**: Links your force and derivative functions to the
       potential object so Sarkas can call them during simulation and analysis.
    
    4. **Multi-species Handling**: Sets up different parameters for different
       species pairs (e.g., electron-electron vs electron-ion interactions).
    
    5. **Validation**: Performs sanity checks on parameters and calculates
       derived quantities like force errors for accuracy estimation.
    
    The function is called once during simulation initialization, after the
    potential type has been identified but before the simulation begins. Any
    errors in this function will prevent the simulation from starting.
    
    This function is linked to `potential.pot_update_params` and is called 
    during simulation setup.

    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.
    species : list
        List of species data (:class:`sarkas.plasma.Species`).
    """
    # Initialize matrix with correct dimensions
    # Shape: (num_species, num_species, num_parameters)
    potential.matrix = zeros((potential.num_species, potential.num_species, 5))
    
    # Set up the potential matrix for all species pairs
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            # Charge factor (always first element)
            potential.matrix[i, j, 0] = sp1.charge * sp2.charge / potential.fourpie0
            
            # Custom parameters (these should be set as attributes of potential)
            potential.matrix[i, j, 1] = potential.strength_parameter  # A
            potential.matrix[i, j, 2] = potential.decay_parameter     # alpha
            potential.matrix[i, j, 3] = potential.power_parameter     # n
            potential.matrix[i, j, 4] = potential.a_rs               # cutoff
    
    # Assign the force function (critical step!)
    potential.force = my_potential_force
    
    # Assign the derivatives function for analytical calculations
    potential.potential_derivatives = potential_derivatives
    
    # Calculate force error if using particle-particle method
    if potential.method == "pp":
        potential.force_error = force_error_analytic_lcl(
            "custom", potential.rc, potential.matrix, 
            sqrt(3.0 * potential.a_ws / (4.0 * pi))
        )
```

### Step 2: Register the Potential

Update `sarkas/potentials/__init__.py`:

```python
__all__ = [
    "Potential",
    "coulomb_force",
    "coulomb_force_pppm",
    "egs_force",
    "lj_force",
    "moliere_force",
    "deutsch_force",
    "kelbg_force",
    "yukawa_force",
    "yukawa_force_pppm",
    "my_potential_force",  # Add your force function
]

# Add import
from .my_potential import my_potential_force
```

### Step 3: Update Core Module

Add your potential to the `type_setup` method in `sarkas/potentials/core.py`:

```python
def type_setup(self, species):
    """Update potential-specific parameters."""
    
    # ... existing code ...
    
    elif self.type == "my_potential":
        # My Custom potential
        from .my_potential import pretty_print_info, update_params
        
        # Optional: calculate screening length if needed
        # self.calc_screening_length(species)
        
        self.pot_update_params = update_params
    
    # ... rest of existing code ...
    
    self.pot_pretty_print = pretty_print_info
    self.pot_update_params(self, species)
```

### Step 4: YAML Configuration

Users can now configure your potential:

```yaml
Potential:
    type: my_potential               # Your potential name
    method: pp                       # or pppm if supported
    rc: 1.0e-8                      # cutoff radius
    strength_parameter: 1.0          # A parameter
    decay_parameter: 5.0e8           # alpha parameter  
    power_parameter: 2.0             # n parameter
    force_error: 1.0e-6             # target accuracy
```

## Adding Tabulated Potentials {#tabulated-potentials}

Sarkas provides robust support for tabulated potentials through the `tabulated.py` module. This feature is invaluable when you have experimental data, ab initio calculations, or complex analytical expressions that are computationally expensive to evaluate on-the-fly.

Tabulated potentials work by pre-computing the potential energy, force, and force derivatives on a grid of distances, then using interpolation during the simulation to obtain values at arbitrary points. This approach offers several advantages:

- **Performance**: Avoids expensive function evaluations during simulation
- **Flexibility**: Can incorporate any potential form, including empirical fits
- **Accuracy**: Preserves numerical precision of original calculations
- **Data Integration**: Directly uses experimental or quantum mechanical results

The implementation supports multiple interpolation methods and handles the complexities of force continuity and boundary conditions automatically.

### Data Format Requirements

Tabulated potentials expect a CSV file with a specific structure that provides all necessary information for interpolation during simulation. The format is designed to ensure smooth force continuity and accurate energy conservation.

```csv
# r [length_units], U(r) [energy_units], F(r) [force_units], F'(r) [force_derivative_units]
1.0e-10, 1.5e-19, -3.0e-9, 6.0e1
2.0e-10, 7.5e-20, -1.5e-9, 3.0e1
3.0e-10, 3.75e-20, -7.5e-10, 1.5e1
# ... continue with your tabulated data
```

**Column Definitions:**
- **Column 1**: Distance `r` (in your chosen length units)
  - Should cover the full range from minimum to cutoff distance
  - Spacing can be uniform or non-uniform (logarithmic often works well, but you need to work in the interpolation)
  - Denser spacing needed where potential varies rapidly

- **Column 2**: Potential energy `U(r)` (in your chosen energy units)
  - The actual potential energy at each distance
  - Should be continuous and well-behaved
  - Consider zero-shifting to avoid large numbers

- **Column 3**: Force magnitude `F(r) = -dU/dr` (force units)
  - Note the negative sign: this is the actual force, not the derivative
  - Must be consistent with the potential energy column
  - Critical for accurate dynamics during simulation

- **Column 4**: Force derivative `F'(r) = d²U/dr²` (force derivative units)
  - Second derivative of the potential energy
  - Used for higher-order interpolation and error estimation
  - Helps ensure smooth interpolation between data points

**Important Considerations:**
- Data points should extend slightly beyond your intended cutoff radius
- The potential should go smoothly to zero at the cutoff
- Include enough points for accurate interpolation (typically 100-1000 points)
- Verify that F(r) = -dU/dr numerically for your data

### Basic Tabulated Potential Usage

#### 1. Prepare Your Data File

Create `my_tabulated_potential.csv`:

```csv
# r [cm], U(r) [erg], F(r) [dyne], F'(r) [dyne/cm]
1.0e-8, 1.44e-11, -2.88e-3, 5.76e5
2.0e-8, 7.2e-12, -1.44e-3, 2.88e5
3.0e-8, 4.8e-12, -9.6e-4, 1.92e5
4.0e-8, 3.6e-12, -7.2e-4, 1.44e5
5.0e-8, 2.88e-12, -5.76e-4, 1.152e5
```

#### 2. YAML Configuration

```yaml
Potential:
    type: tabulated                           # Use tabulated potential
    method: pp                                # Particle-particle method
    tabulated_file: my_tabulated_potential.csv  # Path to your data file
    interpolation_type: linear                # Interpolation method
    rc: 5.0e-8                               # Cutoff radius (should match data range)
    force_error: 1.0e-6                      # Target accuracy
```

### Advanced Tabulated Potential Implementation

For more control over data processing and interpolation, create a custom tabulated potential:

```python
# my_custom_tabulated.py
from sarkas.potentials.tabulated import (
    tab_force_lin_interp, 
    potential_derivatives,
    pretty_print_info
)
from numpy import loadtxt, zeros, where, log, exp
from warnings import warn

def custom_data_processor(filename):
    """
    Custom data processing function.
    
    Parameters
    ----------
    filename : str
        Path to data file
        
    Returns
    -------
    r : numpy.ndarray
        Distance array
    u : numpy.ndarray
        Potential array
    f : numpy.ndarray
        Force array
    f2 : numpy.ndarray
        Force derivative array
    """
    # Example: Handle different file formats
    if filename.endswith('.txt'):
        # Space-separated format
        data = loadtxt(filename, skiprows=2)
    elif filename.endswith('.csv'):
        # Comma-separated format
        data = loadtxt(filename, skiprows=1, delimiter=',')
    else:
        raise ValueError("Unsupported file format")
    
    r = data[:, 0]
    u = data[:, 1]
    f = data[:, 2]
    f2 = data[:, 3]
    
    # Apply custom processing
    # Example: Smooth the data, apply unit conversions, etc.
    
    return r, u, f, f2

def update_params(potential, species):
    """
    Custom tabulated potential with specific preprocessing.
    
    Parameters
    ----------
    potential : :class:`sarkas.potentials.core.Potential`
        Class handling potential form.
    species : list
        List of species data.
    """
    
    # Load data using custom processor
    r, u, f, f2 = custom_data_processor(potential.tabulated_file)
    
    # Apply any custom preprocessing
    # Example: Convert units, apply smoothing, etc.
    if hasattr(potential, 'unit_conversion_factor'):
        u *= potential.unit_conversion_factor
        f *= potential.unit_conversion_factor
    
    # Select data within cutoff radius
    mask = where(r <= potential.rc)[0]
    if len(mask) == 0:
        raise ValueError(f"No data points within cutoff radius {potential.rc}")
    
    params_len = len(r[mask])
    
    # Initialize matrix with extra dimension for tabulated data
    potential.matrix = zeros((potential.num_species, potential.num_species, 4, params_len))
    
    # Shift potential to zero at cutoff (optional but recommended)
    u_tab = u[mask] - (r[mask] - r[mask][-1]) * f[mask][-1] - u[mask][-1]
    f_tab = f[mask] - f[mask][-1]
    f2_tab = f2[mask] - f2[mask][-1]
    
    # Fill matrix for all species pairs
    for i in range(potential.num_species):
        for j in range(potential.num_species):
            potential.matrix[i, j, 0, :] = r[mask]    # Distance array
            potential.matrix[i, j, 1, :] = u_tab      # Potential array
            potential.matrix[i, j, 2, :] = f_tab      # Force array
            potential.matrix[i, j, 3, :] = f2_tab     # Force derivative array
    
    # Set interpolation method
    if hasattr(potential, "interpolation_type"):
        if potential.interpolation_type in ["linear", "lin"]:
            potential.force = tab_force_lin_interp
        elif potential.interpolation_type in ["nearest", "nn"]:
            from sarkas.potentials.tabulated import tab_force_nn
            potential.force = tab_force_nn
    else:
        # Default to linear interpolation
        potential.force = tab_force_lin_interp
    
    potential.potential_derivatives = potential_derivatives
    
    # Calculate force error for the tabulated potential
    beta = 1.0 / (potential.kB * potential.electron_temperature)
    potential.force_error = calc_force_error_quad(
        potential.a_ws, beta, potential.rc, potential.matrix[0, 0]
    )

def calc_force_error_quad(a_ws, beta, rc, pot_matrix):
    """Calculate force error for tabulated potential."""
    from scipy.integrate import quad
    from numpy import inf, pi, sqrt
    
    def force_error_integrand(r, pot_matrix):
        """Integrand for force error calculation."""
        _, dv_dr, _ = potential_derivatives(r, pot_matrix)
        return 4.0 * pi * r**2 * dv_dr**2
    
    # Rescale parameters
    params = pot_matrix.copy()
    params[0, :] /= a_ws
    params[1, :] *= beta
    params[2, :] *= beta
    
    r_c = rc / a_ws
    
    try:
        result, _ = quad(force_error_integrand, a=r_c, b=inf, args=(params,))
        f_err = sqrt(result)
    except:
        warn("Could not calculate force error for tabulated potential")
        f_err = 0.0
    
    return f_err
```

### Tabulated Potential Features

#### Interpolation Methods

Sarkas supports multiple interpolation methods, each with different trade-offs between accuracy and performance:

1. **Linear Interpolation** (`tab_force_lin_interp`):
   - **Best for**: Most general applications where accuracy is important
   - **Advantages**: Smooth interpolation, good accuracy between data points
   - **Disadvantages**: Slightly slower than nearest neighbor
   - **Recommended when**: Data spacing is reasonably uniform and accuracy is crucial

2. **Nearest Neighbor** (`tab_force_nn`):
   - **Best for**: Dense, uniform data grids where speed is critical
   - **Advantages**: Fastest interpolation method, minimal computational overhead
   - **Disadvantages**: Can introduce discontinuities and step artifacts
   - **Recommended when**: You have very dense data and need maximum performance

The choice of interpolation method affects both simulation accuracy and performance. Linear interpolation provides smoother forces and better energy conservation, while nearest neighbor can be significantly faster for large datasets. Consider your specific requirements when choosing.

#### Data Preprocessing

Common preprocessing steps include:

```python
# Unit conversion
u_converted = u * energy_conversion_factor
f_converted = f * force_conversion_factor

# Smoothing (using scipy)
from scipy.signal import savgol_filter
u_smooth = savgol_filter(u, window_length=5, polyorder=2)

# Zero-shifting at cutoff
u_shifted = u - (r - r[-1]) * f[-1] - u[-1]

# Extrapolation handling
if r[0] > desired_r_min:
    # Extrapolate to smaller distances
    # Use analytical form or constant extrapolation
```

## Integration with Core Framework {#integration}

### Registering New Potentials

After creating your potential module, integrate it with the core framework:

1. **Update `__init__.py`**:
```python
from .my_potential import my_potential_force
```

2. **Update `core.py`**:
```python
elif self.type == "my_potential":
    from .my_potential import pretty_print_info, update_params
    self.pot_update_params = update_params
```

### PPPM Algorithm Support

For long-range potentials, you may want to support the PPPM algorithm:

```python
@jit(nopython=True)
def my_potential_force_pppm(r_in, pot_matrix):
    """PPPM-compatible force function."""
    # Extract Ewald parameter
    alpha = pot_matrix[-2]  # Ewald parameter
    
    # Implement short-range part of potential
    # The long-range part is handled by the mesh
    
    # ... implementation details ...
    
    return u_r, f_r

def update_params(potential, species):
    """Update parameters with PPPM support."""
    
    # ... existing setup code ...
    
    if potential.method == "pppm":
        # Add PPPM-specific parameters
        potential.matrix[:, :, -2] = potential.pppm_alpha_ewald
        potential.force = my_potential_force_pppm
    else:
        potential.force = my_potential_force
```

## Testing and Validation {#testing}

### Basic Functionality Test

```python
#!/usr/bin/env python3
"""Test script for new potential."""

import numpy as np
import matplotlib.pyplot as plt
from sarkas.potentials.my_potential import my_potential_force

# Test parameters
pot_matrix = np.array([
    1.44e-7,    # q_factor (for e-e interaction in cgs)
    1.0,        # A 
    5.0e8,      # alpha (1/cm)
    2.0,        # n
    1.0e-10     # a_rs (cm)
])

# Test distance range
r_test = np.logspace(-10, -7, 1000)  # 1e-10 to 1e-7 cm
u_test = []
f_test = []

for r in r_test:
    u, f = my_potential_force(r, pot_matrix)
    u_test.append(u)
    f_test.append(f)

# Plot results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

ax1.loglog(r_test, np.abs(u_test), 'b-', label='Potential')
ax1.set_xlabel('Distance (cm)')
ax1.set_ylabel('|Potential| (erg)')
ax1.legend()
ax1.grid(True)

ax2.loglog(r_test, np.abs(f_test), 'r-', label='Force')
ax2.set_xlabel('Distance (cm)')
ax2.set_ylabel('|Force| (dyne)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Integration Test

```python
from sarkas.processes import PreProcess

# Create test YAML
test_yaml = """
Particles:
    - Species:
        name: e
        number_density: 1.0e23
        mass: 9.1093837015e-28
        num: 100
        Z: -1.0
        temperature_eV: 10.0

    - Species:
        name: p  
        number_density: 1.0e23
        mass: 1.67262192369e-24
        num: 100
        Z: 1.0
        temperature_eV: 10.0

Potential:
    type: my_potential
    method: pp
    rc: 1.0e-8
    strength_parameter: 1.0
    decay_parameter: 5.0e8
    power_parameter: 2.0

Integrator:
    type: verlet
    dt: 1.0e-17
    equilibration_steps: 100
    production_steps: 100

Parameters:
    units: cgs
    load_method: random_no_reject
    boundary_conditions: periodic

IO:
    verbose: True
"""

# Test preprocessing
with open('test_potential.yaml', 'w') as f:
    f.write(test_yaml)

pre = PreProcess('test_potential.yaml')
pre.setup(read_yaml=True)
print("Potential setup successful!")
```

### Validation Against Known Results

```python
def validate_potential():
    """Validate against analytical results."""
    
    # Test limiting cases
    # Example: At large distances, potential should match Coulomb
    r_large = 1.0e-6  # Large distance
    pot_matrix_test = np.array([1.44e-7, 1.0, 0.0, 1.0, 1.0e-10])  # alpha=0
    
    u_test, f_test = my_potential_force(r_large, pot_matrix_test)
    u_coulomb = pot_matrix_test[0] / r_large  # Pure Coulomb
    
    assert abs(u_test - u_coulomb) / u_coulomb < 1e-6, "Large distance limit failed"
    
    # Test energy conservation
    # ... additional tests ...
    
    print("All validation tests passed!")

validate_potential()
```

## Advanced Features {#advanced-features}

### Force Error Calculation

The function `calc_force_error_quad` implement accurate force error estimation by integrating (using Gaussian quadrature) the force error over the neglected volume. This function is not necessary and you can set your force error calculation as you prefer. 

```python
def calc_force_error_quad(potential):
    """Calculate force error using quadrature integration."""
    from scipy.integrate import quad
    from numpy import inf, pi, sqrt, gamma
    
    pot_matrix = potential.matrix[0, 0]  # Use representative pair
    r_c = potential.rc
    
    # Solid angle factor
    solid_angle = 2.0 * pi**(potential.dimensions / 2) / gamma(potential.dimensions / 2)
    
    # Define integrand for force error
    def integrand(r):
        _, force = my_potential_force(r, pot_matrix)
        return solid_angle * r**(potential.dimensions - 1) * force**2
    
    # Integrate from cutoff to infinity
    f_err_a, _ = quad(integrand, a=r_c, b=inf)
    
    # Apply appropriate scaling factors
    rescaling_constant = sqrt(3.0 * potential.a_ws / (4.0 * pi))
    constant = rescaling_constant * potential.a_ws**(potential.dimensions / 2.0 - 1)
    
    return constant * sqrt(f_err_a)
```

### Multi-Species Support

When creating the `update_params` function make sure to handle different species:

```python
def update_params(potential, species):
    """Handle multi-species interactions."""
    
    potential.matrix = zeros((potential.num_species, potential.num_species, num_params))
    
    for i, sp1 in enumerate(species):
        for j, sp2 in enumerate(species):
            # Species-specific parameters
            if sp1.name == "e" and sp2.name == "e":
                # Electron-electron interaction
                potential.matrix[i, j, 1] = potential.ee_strength
            elif (sp1.name == "e" and sp2.name != "e") or (sp1.name != "e" and sp2.name == "e"):
                # Electron-ion interaction
                potential.matrix[i, j, 1] = potential.ei_strength
            else:
                # Ion-ion interaction
                potential.matrix[i, j, 1] = potential.ii_strength
```

## Best Practices {#best-practices}

### Code Organization

1. **Modular Design**: Keep each potential in its own module
2. **Clear Documentation**: Include mathematical formulations and parameter descriptions
3. **Consistent Naming**: Follow Sarkas naming conventions
4. **Error Handling**: Add appropriate validation and error messages

### Performance Considerations

1. **Numba Optimization**: Use `@jit` decorators for force calculation functions
2. **Memory Efficiency**: Minimize array allocations in tight loops
3. **Algorithm Selection**: Choose appropriate methods based on system characteristics

### Testing Strategy

1. **Unit Tests**: Test individual functions with known inputs
2. **Integration Tests**: Test full workflow with simple systems
3. **Validation Tests**: Compare against analytical results or benchmarks
4. **Performance Tests**: Profile and optimize bottlenecks

### Documentation Requirements

1. **Mathematical Formulation**: Include LaTeX equations in docstrings
2. **Parameter Descriptions**: Document all matrix elements
3. **Usage Examples**: Provide YAML configuration examples
4. **Reference Citations**: Include relevant literature references

### Common Pitfalls to Avoid

1. **Sign Conventions**: Ensure force has correct sign (positive for repulsive)
2. **Unit Consistency**: Test with both CGS and MKS unit systems
3. **Numerical Stability**: Handle small distance divergences properly
4. **Matrix Indexing**: Ensure correct species pair mapping

This guide should enable you to successfully add new potentials to Sarkas, whether they are analytical expressions or tabulated data. The modular design makes it straightforward to extend the package while maintaining compatibility with existing features and ensuring optimal performance.
