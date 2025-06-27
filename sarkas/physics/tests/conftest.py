"""
Pytest configuration for sarkas physics tests.
"""

import pytest
import warnings

def pytest_configure(config):
    """Configure pytest markers and settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", 
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", 
        "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", 
        "physics: marks tests that validate physics equations"
    )
    config.addinivalue_line(
        "markers", 
        "performance: marks tests that benchmark performance"
    )
    config.addinivalue_line(
        "markers", 
        "edge_case: marks tests for edge cases and error conditions"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add slow marker to performance tests
        if "benchmark" in item.name.lower() or "performance" in item.name.lower():
            item.add_marker(pytest.mark.slow)
        
        # Add physics marker to physics validation tests
        if "physics" in item.name.lower() or "conservation" in item.name.lower():
            item.add_marker(pytest.mark.physics)
        
        # Add edge_case marker to edge case tests
        if "edge" in item.name.lower() or "error" in item.name.lower():
            item.add_marker(pytest.mark.edge_case)

@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress common warnings during testing."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="pkg_resources")
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="sarkas")
        warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
        yield

@pytest.fixture
def sample_particles_small():
    """Create small sample particle data for testing."""
    import numpy as np
    np.random.seed(42)
    
    n_particles = 100
    velocities = np.random.randn(n_particles, 3) * 10.0
    masses = np.random.uniform(0.5, 2.0, n_particles)
    
    return {
        'velocities': velocities,
        'masses': masses,
        'n_particles': n_particles
    }

@pytest.fixture
def sample_particles_large():
    """Create large sample particle data for performance testing."""
    import numpy as np
    np.random.seed(42)
    
    n_particles = 10000
    velocities = np.random.randn(n_particles, 3) * 100.0
    masses = np.random.uniform(0.1, 10.0, n_particles)
    
    return {
        'velocities': velocities,
        'masses': masses,
        'n_particles': n_particles
    }

@pytest.fixture
def physical_constants():
    """Provide physical constants for testing."""
    return {
        'kB': 1.380649e-23,  # J/K
        'dimensions': 3,
        'tolerance': 1e-10,
        'relative_tolerance': 1e-12
    }
