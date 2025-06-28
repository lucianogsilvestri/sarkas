# Boundary Conditions Module

This module provides a comprehensive framework for implementing and managing boundary conditions in molecular dynamics simulations.

## Available Boundary Conditions

### 1. Periodic Boundary Conditions (PBC)
- **Name**: `'periodic'`, `'pbc'`, `'periodic_bc'`
- **Description**: Particles crossing boundaries reappear on the opposite side
- **Use case**: Bulk simulations, infinite systems
- **Requirements**: Positions
- **Modifies**: Positions, maintains a crossing counter

### 2. Absorbing Boundary Conditions (ABC)
- **Name**: `'absorbing'`, `'abc'`, `'absorbing_bc'`, `'sink'`
- **Description**: Particles hitting boundaries are removed (charge → 0)
- **Use case**: Open systems, particle sinks
- **Requirements**: Positions, velocities, accelerations, charges
- **Modifies**: Positions, velocities, accelerations, charges

### 3. Reflecting Boundary Conditions (RBC)
- **Name**: `'reflecting'`, `'rbc'`, `'reflecting_bc'`, `'mirror'`, `'elastic'`
- **Description**: Particles bounce off boundaries elastically
- **Use case**: Confined systems, containers
- **Requirements**: Positions, velocities, timestep
- **Modifies**: Positions, velocities

### 4. Open Boundary Conditions
- **Name**: `'open'`, `'none'`, `'free'`, `'open_bc'`
- **Description**: No boundary enforcement (particles move freely)
- **Use case**: Large systems, no confinement
- **Requirements**: None
- **Modifies**: Nothing

### 5. Mixed Boundary Conditions
- **Name**: `'mixed'`
- **Description**: Different BC types for different dimensions
- **Use case**: Complex geometries (e.g., periodic in x,y but reflecting in z)
- **Requirements**: All particle properties
- **Modifies**: All properties

## Usage

### Basic Usage
```python
from algorithms.boundary_conditions import get_boundary_condition_method

# Get a boundary condition method
enforce_bc = get_boundary_condition_method('periodic')

# Apply to particles (called by integrator)
enforce_bc(ptcls)
```

### Integration with Integrators
Boundary conditions are automatically applied by integrators:

```python
from algorithms.integrators import get_integrator

# Setup integrator
integrator = get_integrator('verlet')
integrator.setup(params, boundary_conditions='periodic')

# During integration, BC is automatically applied
integrator.update(ptcls)  # BC enforcement happens inside
```

### Available Methods
```python
from algorithms.boundary_conditions import (
    list_boundary_conditions,
    get_boundary_condition_info
)

# List all available boundary conditions
print(list_boundary_conditions())
# ['periodic', 'absorbing', 'reflecting', 'open', 'mixed']

# Get detailed information
info = get_boundary_condition_info()
print(info['periodic'])
```

## Implementation Details

### Performance
- All core functions are **Numba JIT-compiled** for maximum performance
- **Optimized algorithms** with minimal memory allocation
- **In-place operations** for efficiency

### Consistency
- **Unified interface** through wrapper functions
- **Automatic parameter extraction** from particle objects
- **Consistent error handling** and validation

### Extensibility
- **Easy to add new BC types** by following the pattern
- **Property flags** describe BC requirements and effects
- **Automatic registration** system

## Adding New Boundary Conditions

To add a new boundary condition:

1. **Implement the core function** (preferably Numba JIT-compiled):
```python
@jit(nopython=True, cache=True)
def _enforce_custom_bc_numba(pos, vel, box_lengths):
    # Your implementation here
    pass
```

2. **Create a wrapped version**:
```python
enforce_custom_bc = create_boundary_condition_wrapper(
    _enforce_custom_bc_numba,
    'custom',
    requires_velocities=True,
    requires_charges=False,
    modifies_positions=True,
    modifies_velocities=True
)
```

3. **Register it** in `__init__.py`:
```python
register_boundary_condition(
    'custom',
    enforce_custom_bc,
    aliases=['custom_bc', 'special']
)
```

## Technical Notes

### Periodic Boundary Conditions
- Maintains a **crossing counter** (`ptcls.pbc_cntr`) for particle tracking
- **Handles arbitrary box sizes** and dimensions
- **Preserves particle ordering**

### Absorbing Boundary Conditions
- **Sets charge to zero** to effectively "remove" particles
- **Resets all particle properties** to prevent artifacts
- **Clamps positions** to boundary values

### Reflecting Boundary Conditions
- **Reverses velocity components** for elastic reflection
- **Requires timestep** for position correction
- **Assumes Verlet-type integration** for position restoration

### Mixed Boundary Conditions
- **Dimension-specific BC types** via integer array
- **Supports all BC combinations** per dimension
- **Efficient single-pass implementation**

## Directory Structure
```
boundary_conditions/
├── __init__.py           # Factory and registry system
├── base.py               # Base classes and interfaces
├── implementations.py    # Core BC implementations
└── README.md            # This documentation
```