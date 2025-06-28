algorithms/
├── integrators/
│   ├── __init__.py                 # Registry and factory
│   ├── base.py                     # IntegratorBase class
│   ├── README.md                   # How to add new integrators
│   ├── single_timestep/            # Standard integrators
│   │   ├── __init__.py
│   │   ├── verlet.py              # Verlet(IntegratorBase)
│   │   ├── langevin.py            # Langevin(IntegratorBase)
│   │   └── leapfrog.py            # Leapfrog(IntegratorBase)
│   ├── runge_kutta/                # Runge-Kutta family
│   │   ├── __init__.py
│   │   ├── base_rk.py             # RungeKuttaBase(IntegratorBase)
│   │   ├── rk2.py                 # RK2(RungeKuttaBase) - Midpoint
│   │   ├── rk4.py                 # RK4(RungeKuttaBase) - Classic RK4
│   │   ├── rk45.py                # RK45(RungeKuttaBase) - Adaptive
│   │   ├── dormand_prince.py      # DormandPrince(RungeKuttaBase)
│   │   └── butcher_tableau.py     # Generic Butcher tableau implementation
│   ├── magnetic/                   # Magnetic field integrators
│   │   ├── __init__.py
│   │   ├── magnetic_verlet.py     # MagneticVerlet(IntegratorBase)
│   │   ├── boris.py               # Boris(IntegratorBase)
│   │   └── cyclotronic.py         # Cyclotronic(IntegratorBase)
│   ├── multi_timestep/             # Multi-timestep integrators
│   │   ├── __init__.py
│   │   ├── base_mts.py            # MultiTimestepBase(IntegratorBase)
│   │   ├── respa.py               # RESPA-style
│   │   ├── forest_ruth.py         # Forest-Ruth algorithm
│   │   └── symplectic_splitting.py # Custom splitting methods
│   ├── advanced/                   # Advanced/specialized integrators
│   │   ├── __init__.py
│   │   ├── adaptive_timestep.py   # Adaptive timestep control
│   │   ├── constrained.py         # SHAKE/RATTLE for constraints
│   │   └── implicit.py            # Implicit integrators
│   └── tests/
│       ├── __init__.py
│       ├── test_single_timestep.py
│       ├── test_runge_kutta.py
│       ├── test_magnetic.py
│       ├── test_multi_timestep.py
│       └── test_advanced.py
│
├── thermostats/
│   ├── __init__.py                 # Registry and factory
│   ├── base.py                     # ThermostatBase class
│   ├── README.md                   # How to add new thermostats
│   ├── stochastic/                 # Stochastic thermostats
│   │   ├── __init__.py
│   │   ├── berendsen.py           # Berendsen(ThermostatBase)
│   │   ├── anderson.py            # Anderson(ThermostatBase)
│   │   └── bussi.py               # BussiThermostat(ThermostatBase)
│   ├── deterministic/              # Deterministic thermostats
│   │   ├── __init__.py
│   │   ├── nose_hoover.py         # NoseHoover(ThermostatBase)
│   │   ├── nose_hoover_chain.py   # NoseHooverChain(ThermostatBase)
│   │   ├── gaussian.py            # GaussianThermostat(ThermostatBase)
│   │   └── evans.py               # EvansThermostat(ThermostatBase)
│   ├── global_methods/             # Global temperature control
│   │   ├── __init__.py
│   │   ├── velocity_scaling.py    # VelocityScaling(ThermostatBase)
│   │   ├── isokinetic.py          # Isokinetic(ThermostatBase)
│   │   └── csvr.py                # CanonicalSampling(ThermostatBase)
│   ├── advanced/                   # Advanced/specialized thermostats
│   │   ├── __init__.py
│   │   ├── adaptive.py            # AdaptiveThermostat(ThermostatBase)
│   │   ├── local.py               # LocalThermostat(ThermostatBase)
│   │   └── configurational.py     # ConfigurationalThermostat(ThermostatBase)
│   └── tests/
│       ├── __init__.py
│       ├── test_stochastic.py
│       ├── test_deterministic.py
│       ├── test_global_methods.py
│       └── test_advanced.py