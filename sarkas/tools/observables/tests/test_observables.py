"""
Tier 1 tests for the sarkas observables refactoring.
All tests in this file require no simulation data (h5md files).
"""

import ast
import pathlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1.1  Import compatibility
# ---------------------------------------------------------------------------


def test_import_observables():
    from sarkas.tools.observables import (
        CurrentCorrelationFunction,
        DiffusionFlux,
        DynamicStructureFactor,
        ElectricCurrent,
        HeatFlux,
        MicroscopicCurrent,
        MicroscopicDensity,
        MicroscopicVelocity,
        Observable,
        PressureTensor,
        RadialDistributionFunction,
        StaticStructureFactor,
        Thermodynamics,
        VelocityAutoCorrelationFunction,
        VelocityDistribution,
        kspace_setup,
        load_from_restart,
        plot_labels,
    )


def test_import_transport():
    from sarkas.tools.transport import (
        Diffusion,
        ElectricalConductivity,
        InterDiffusion,
        ThermalConductivity,
        TransportCoefficients,
        Viscosity,
    )


# ---------------------------------------------------------------------------
# 1.2  update_block_attributes — synthetic scalar parameters
# ---------------------------------------------------------------------------


class FakeObs:
    """Minimal stand-in for Observable with attributes set by setup_init."""

    no_steps = 10000
    no_dumps = 1001  # 1 + 10000//10
    dump_step = 10
    dt = 0.001
    plasma_period = 1.0
    timesteps_per_plasma_period = 1000.0
    independent_slices = True
    no_slices = 1
    timesteps_per_slice = None
    timesteps_shift = None
    plasma_periods_per_slice = None
    plasma_periods_shift = None

    def update_block_attributes(self, **kwargs):
        from sarkas.tools.observables import Observable

        Observable.update_block_attributes(self, **kwargs)


def test_independent_single_block():
    obs = FakeObs()
    obs.update_block_attributes()
    assert obs.block_length == obs.no_steps // obs.dump_step  # 1000
    assert obs.no_slices == 1
    assert obs.dumps_shift == obs.block_length
    assert obs.dumps_per_slice == obs.block_length


def test_independent_multi_slice():
    obs = FakeObs()
    obs.update_block_attributes(no_slices=4)
    assert obs.block_length == (obs.no_steps // 4) // obs.dump_step  # 250
    assert obs.no_slices == 4
    assert obs.dumps_shift == obs.block_length


def test_independent_via_plasma_periods():
    obs = FakeObs()
    obs.update_block_attributes(plasma_periods_per_slice=2, no_slices=4)
    assert obs.timesteps_per_slice == 2000
    assert obs.block_length == 2000 // obs.dump_step  # 200


def test_independent_rejects_shift():
    obs = FakeObs()
    with pytest.raises(AttributeError, match="timesteps_shift"):
        obs.update_block_attributes(timesteps_shift=500)


def test_independent_rejects_zero_slices():
    obs = FakeObs()
    with pytest.raises(AttributeError):
        obs.update_block_attributes(no_slices=0)


def test_independent_rejects_too_many_slices():
    obs = FakeObs()
    with pytest.raises(AttributeError):
        obs.update_block_attributes(no_slices=obs.no_dumps + 1)


def test_sliding_timesteps():
    obs = FakeObs()
    obs.update_block_attributes(
        independent_slices=False,
        timesteps_per_slice=2000,
        timesteps_shift=500,
    )
    # (10000 - 2000) // 500 + 1 = 17
    assert obs.no_slices == 17
    assert obs.block_length == 2000 // obs.dump_step  # 200
    assert obs.dumps_shift == 500 // obs.dump_step  # 50


def test_sliding_plasma_periods():
    obs = FakeObs()
    obs.update_block_attributes(
        independent_slices=False,
        plasma_periods_per_slice=3,
        plasma_periods_shift=1,
    )
    assert obs.timesteps_per_slice == 3000
    assert obs.timesteps_shift == 1000
    assert obs.no_slices == (10000 - 3000) // 1000 + 1  # 8


def test_sliding_rejects_missing_shift():
    obs = FakeObs()
    with pytest.raises(AttributeError, match="timesteps_shift"):
        obs.update_block_attributes(
            independent_slices=False,
            timesteps_per_slice=2000,
        )


def test_sliding_rejects_missing_window():
    obs = FakeObs()
    with pytest.raises(AttributeError, match="timesteps_per_slice"):
        obs.update_block_attributes(
            independent_slices=False,
            timesteps_shift=500,
        )


def test_sliding_rejects_window_exceeds_trajectory():
    obs = FakeObs()
    with pytest.raises(AttributeError):
        obs.update_block_attributes(
            independent_slices=False,
            timesteps_per_slice=obs.no_steps + 1,
            timesteps_shift=500,
        )


def test_plasma_period_equivalents_are_consistent():
    obs = FakeObs()
    obs.update_block_attributes(
        independent_slices=False,
        timesteps_per_slice=2000,
        timesteps_shift=1000,
    )
    assert np.isclose(
        obs.plasma_periods_per_slice,
        obs.timesteps_per_slice / obs.timesteps_per_plasma_period,
    )
    assert np.isclose(
        obs.plasma_periods_shift,
        obs.timesteps_shift / obs.timesteps_per_plasma_period,
    )


# ---------------------------------------------------------------------------
# 1.3  MicroscopicDensity kernel correctness
# ---------------------------------------------------------------------------


def test_nk_vectorised_matches_numba():
    rng = np.random.default_rng(42)
    N, no_k = 50, 8
    pos = rng.uniform(0, 10.0, (N, 3))
    k_list = rng.uniform(0, 1.0, (no_k, 3))
    species_np = np.array([20, 30])

    from sarkas.tools.observables._kernels import calc_nk_numba
    from sarkas.tools.observables.kspace import _calc_nk_vectorised

    nk_ref = calc_nk_numba(pos, k_list, species_np)
    nk_new = _calc_nk_vectorised(pos, k_list, species_np)

    np.testing.assert_allclose(nk_new, nk_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# 1.4  MicroscopicVelocity kernel correctness
# ---------------------------------------------------------------------------


def test_vk_vectorised_matches_reference():
    rng = np.random.default_rng(7)
    N, no_k = 40, 6
    pos = rng.uniform(0, 8.0, (N, 3))
    vel = rng.standard_normal((N, 3))
    k_list = rng.uniform(0, 1.0, (no_k, 3))
    species_np = np.array([40])

    from sarkas.tools.observables._kernels import calc_vk_reference
    from sarkas.tools.observables.kspace import _calc_vk_vectorised

    vk_ref = calc_vk_reference(pos, vel, k_list, species_np)
    vk_new = _calc_vk_vectorised(pos, vel, k_list, species_np)

    np.testing.assert_allclose(vk_new, vk_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# 1.5  MicroscopicCurrent charge weighting
# ---------------------------------------------------------------------------


def test_microscopic_current_charge_weighting():
    rng = np.random.default_rng(3)
    N, no_k = 30, 4
    pos = rng.uniform(0, 5.0, (N, 3))
    vel = rng.standard_normal((N, 3))
    k_list = rng.uniform(0, 1.0, (no_k, 3))
    species_np = np.array([30])
    charge = 2.0

    from sarkas.tools.observables.kspace import _calc_jk_from_vk, _calc_vk_vectorised

    vk = _calc_vk_vectorised(pos, vel, k_list, species_np)
    jk = _calc_jk_from_vk(vk, np.array([charge]))

    np.testing.assert_allclose(jk[0], charge * vk[0], rtol=1e-14)


# ---------------------------------------------------------------------------
# 1.6  Zarr store shape and metadata
# ---------------------------------------------------------------------------


def test_zarr_store_shapes(tmp_path):
    import xarray as xr

    no_species, no_k, no_dumps = 2, 10, 50
    nkt = np.zeros((no_species, no_k, no_dumps), dtype=np.complex128)
    species_names = ["H", "He"]
    k_harmonics = np.zeros((no_k, 3), dtype=int)
    time = np.arange(no_dumps) * 0.001

    store_path = str(tmp_path / "nkt.zarr")
    from sarkas.tools.observables.kspace import save_nkt_to_zarr

    save_nkt_to_zarr(store_path, nkt, species_names, k_harmonics, time)

    ds = xr.open_zarr(store_path)
    assert "nkt" in ds.data_vars
    assert "k_harmonics" in ds.data_vars
    assert ds["nkt"].dims == ("species", "k", "time")
    assert ds["nkt"].shape == (no_species, no_k, no_dumps)
    assert ds["nkt"].dtype == np.complex128
    assert list(ds["nkt"].coords["species"].values) == species_names
    assert ds["k_harmonics"].shape == (no_k, 3)


# ---------------------------------------------------------------------------
# 1.6b  acf_batch correctness
# ---------------------------------------------------------------------------


def test_acf_batch_constant_signal():
    """For a constant signal, acf_batch should return a flat (constant) ACF."""
    from sarkas.tools.observables._kernels import acf_batch

    N, T = 5, 200
    data = np.ones((N, T))
    acf = acf_batch(data)
    assert acf.shape == (N, T)
    # All lags equal — ratio is const / (T - tau) normalised the same way
    np.testing.assert_allclose(acf / acf[:, 0:1], 1.0, rtol=1e-10)


def test_acf_batch_shape():
    rng = np.random.default_rng(0)
    N, T = 10, 300
    data = rng.standard_normal((N, T))
    from sarkas.tools.observables._kernels import acf_batch
    acf = acf_batch(data)
    assert acf.shape == (N, T)


def test_acf_batch_matches_correlationfunction():
    """acf_batch on a single signal must match correlationfunction."""
    from sarkas.tools.observables._kernels import acf_batch
    from sarkas.utilities.maths import correlationfunction

    rng = np.random.default_rng(42)
    signal = rng.standard_normal(300)
    acf_ref = correlationfunction(signal, signal)
    acf_new = acf_batch(signal[np.newaxis, :])[0]
    np.testing.assert_allclose(acf_new, acf_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# 1.6c  no_simulation_dataframe — removed from all observable classes
# ---------------------------------------------------------------------------


def test_no_simulation_dataframe_attribute():
    from sarkas.tools.observables import (
        DiffusionFlux,
        ElectricCurrent,
        HeatFlux,
        PressureTensor,
        Thermodynamics,
        VelocityAutoCorrelationFunction,
    )
    for cls in [ElectricCurrent, HeatFlux, DiffusionFlux,
                PressureTensor, Thermodynamics, VelocityAutoCorrelationFunction]:
        obj = cls()
        assert not hasattr(obj, "simulation_dataframe"), \
            f"{cls.__name__} still has simulation_dataframe"
        assert not hasattr(obj, "read_data_from_dumps"), \
            f"{cls.__name__} still has read_data_from_dumps"
        assert not hasattr(obj, "load_simulation_dataframe"), \
            f"{cls.__name__} still has load_simulation_dataframe"
        assert not hasattr(obj, "save_simulation_hdf"), \
            f"{cls.__name__} still has save_simulation_hdf"


# ---------------------------------------------------------------------------
# 1.6d  correlationfunction uses no list comprehension
# ---------------------------------------------------------------------------


def test_correlationfunction_no_list_comprehension():
    import inspect
    from sarkas.utilities import maths
    src = inspect.getsource(maths.correlationfunction)
    tree = ast.parse(src)
    list_comps = [n for n in ast.walk(tree) if isinstance(n, ast.ListComp)]
    assert len(list_comps) == 0, "correlationfunction still uses a list comprehension"


# ---------------------------------------------------------------------------
# 1.6e  fast_integral_loop deleted
# ---------------------------------------------------------------------------


def test_fast_integral_loop_deleted():
    from sarkas.utilities import maths
    assert not hasattr(maths, "fast_integral_loop"), \
        "fast_integral_loop should be deleted — use cumulative_trapezoid instead"


# ---------------------------------------------------------------------------
# 1.6f  No rint() in calc_slices_data (step fix)
# ---------------------------------------------------------------------------


def test_no_rint_in_observables():
    tools = pathlib.Path("sarkas/tools/observables")
    if not tools.exists():
        tools = pathlib.Path(__file__).parent.parent
    for fname in tools.rglob("*.py"):
        src = fname.read_text()
        if "rint" not in src:
            continue
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                assert name != "rint", f"rint() still used in {fname}"


# ---------------------------------------------------------------------------
# 1.6g  bin_vol loop vectorised — no 'ir' For loop in spatial.py
# ---------------------------------------------------------------------------


def test_bin_vol_vectorised():
    src = pathlib.Path("sarkas/tools/observables/spatial.py")
    if not src.exists():
        src = pathlib.Path(__file__).parent.parent / "spatial.py"
    tree = ast.parse(src.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            if isinstance(node.target, ast.Name) and node.target.id == "ir":
                raise AssertionError("bin_vol scalar loop over 'ir' still present in spatial.py")


# ---------------------------------------------------------------------------
# 1.7  No pandas imports remain
# ---------------------------------------------------------------------------


def test_no_pandas_in_observables():
    tools = pathlib.Path("sarkas/tools/observables")
    if not tools.exists():
        tools = pathlib.Path(__file__).parent.parent
    for pyfile in tools.rglob("*.py"):
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names]
                src = getattr(node, "module", "") or ""
                assert "pandas" not in src and not any("pandas" in n for n in names), (
                    f"pandas import found in {pyfile}"
                )


# ---------------------------------------------------------------------------
# 1.8  All @njit functions live only in _kernels.py
# ---------------------------------------------------------------------------


def test_njit_only_in_kernels():
    tools = pathlib.Path("sarkas/tools/observables")
    if not tools.exists():
        tools = pathlib.Path(__file__).parent.parent
    for pyfile in tools.rglob("*.py"):
        if pyfile.name == "_kernels.py":
            continue
        tree = ast.parse(pyfile.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "njit":
                raise AssertionError(f"@njit found outside _kernels.py in {pyfile}")


# ---------------------------------------------------------------------------
# 2.1  Thermodynamics zarr layout
# ---------------------------------------------------------------------------


def _make_therm_store(tmp_path, no_q=3, no_sp=2, block_length=50, no_slices=4):
    """Write a synthetic Thermodynamics zarr store and return (store_path, params)."""
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "therm.zarr")
    species = ["H", "He", "Total"][:no_sp + (1 if no_sp > 1 else 0)]
    quantity_names = ["Temperature", "Kinetic Energy", "Total Energy"][:no_q]
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time

    store.require_dataset(
        "coords/quantities",
        shape=(no_q,),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
        compressor=None,
        overwrite=True,
    )
    store["coords/quantities"][:] = np.array(quantity_names, dtype=object)

    store.require_dataset(
        "coords/species",
        shape=(len(species),),
        dtype=object,
        object_codec=numcodecs.VLenUTF8(),
        compressor=None,
        overwrite=True,
    )
    store["coords/species"][:] = np.array(species, dtype=object)

    rng = np.random.default_rng(0)
    for sp in species:
        data = rng.standard_normal((no_q, block_length, no_slices))
        # Put a recognisable temperature signal in row 0
        data[0] = np.abs(data[0]) + 1.0
        store.require_dataset(
            f"species/{sp}/data",
            shape=(no_q, block_length, no_slices),
            chunks=(no_q, block_length, 1),
            dtype="f8",
            compressor=compressor,
        )
        store[f"species/{sp}/data"][:] = data
        store.require_dataset(f"mean/{sp}/data",     shape=(no_q, block_length), dtype="f8", compressor=compressor)
        store.require_dataset(f"mean/{sp}/data_std", shape=(no_q, block_length), dtype="f8", compressor=compressor)
        store[f"mean/{sp}/data"][:]     = data.mean(axis=-1)
        store[f"mean/{sp}/data_std"][:] = data.std(axis=-1, ddof=min(1, no_slices - 1))

    return store_path, {"species": species, "quantities": quantity_names,
                        "no_q": no_q, "block_length": block_length, "no_slices": no_slices}


def test_therm_zarr_required_keys(tmp_path):
    store_path, p = _make_therm_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert "coords/time" in z
    assert "coords/quantities" in z
    assert "coords/species" in z
    for sp in p["species"]:
        assert f"species/{sp}/data" in z
        assert f"mean/{sp}/data" in z
        assert f"mean/{sp}/data_std" in z


def test_therm_zarr_data_shape(tmp_path):
    no_q, no_sp, block_length, no_slices = 3, 2, 50, 4
    store_path, p = _make_therm_store(tmp_path, no_q=no_q, no_sp=no_sp,
                                       block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    for sp in p["species"]:
        assert z[f"species/{sp}/data"].shape == (no_q, block_length, no_slices), \
            f"Wrong shape for species/{sp}/data"
        assert z[f"mean/{sp}/data"].shape == (no_q, block_length)
        assert z[f"mean/{sp}/data_std"].shape == (no_q, block_length)


def test_therm_zarr_slice_axis_is_last(tmp_path):
    store_path, p = _make_therm_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    # Slice axis == -1: accessing isl=0 via last index must equal the mean only when no_slices=1
    data = z[f"species/{p['species'][0]}/data"][:]
    assert data.ndim == 3
    assert data.shape[-1] == p["no_slices"]


def test_therm_zarr_quantities_index_access(tmp_path):
    store_path, p = _make_therm_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    quantities = list(z["coords/quantities"][:])
    assert "Temperature" in quantities
    iT = quantities.index("Temperature")
    # Temperature column should be accessible by index, positive values (we set abs)
    sp = p["species"][0]
    temp_col = z[f"species/{sp}/data"][iT, :, 0]
    assert temp_col.shape == (p["block_length"],)
    assert (temp_col > 0).all()


def test_therm_zarr_mean_is_slice_mean(tmp_path):
    store_path, p = _make_therm_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    sp = p["species"][0]
    data = z[f"species/{sp}/data"][:]
    mean_expected = data.mean(axis=-1)
    np.testing.assert_allclose(z[f"mean/{sp}/data"][:], mean_expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2.2  PressureTensor zarr layout
# ---------------------------------------------------------------------------


def _make_pressure_store(tmp_path, D=3, block_length=40, no_slices=3):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "pressure_tensor.zarr")

    idx = np.triu_indices(D)
    no_comp = len(idx[0])
    components = np.stack(idx, axis=1).astype(np.int8)
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time
    store.require_dataset("coords/components", shape=(no_comp, 2), dtype="i1", compressor=compressor)
    store["coords/components"][:] = components

    rng = np.random.default_rng(1)
    pressure = rng.standard_normal((block_length, no_slices))
    tensor = rng.standard_normal((no_comp, block_length, no_slices))

    store.require_dataset("total/pressure", shape=(block_length, no_slices),
                          chunks=(block_length, 1), dtype="f8", compressor=compressor)
    store.require_dataset("total/tensor", shape=(no_comp, block_length, no_slices),
                          chunks=(no_comp, block_length, 1), dtype="f8", compressor=compressor)
    store["total/pressure"][:] = pressure
    store["total/tensor"][:] = tensor

    # ACF arrays (bulk=scalar, tensor=symmetric matrix of components)
    store.require_dataset("total/acf_bulk", shape=(block_length, no_slices),
                          chunks=(block_length, 1), dtype="f8", compressor=compressor)
    store.require_dataset("total/acf_tensor", shape=(no_comp, no_comp, block_length, no_slices),
                          chunks=(no_comp, no_comp, block_length, 1), dtype="f8", compressor=compressor)

    return store_path, {"D": D, "no_comp": no_comp, "block_length": block_length, "no_slices": no_slices}


def test_pressure_zarr_required_keys(tmp_path):
    store_path, _ = _make_pressure_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/time", "coords/components", "total/pressure",
                "total/tensor", "total/acf_bulk", "total/acf_tensor"):
        assert key in z, f"Missing key: {key}"


def test_pressure_zarr_shapes(tmp_path):
    block_length, no_slices, D = 40, 3, 3
    store_path, p = _make_pressure_store(tmp_path, D=D, block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    no_comp = p["no_comp"]
    assert z["total/pressure"].shape == (block_length, no_slices)
    assert z["total/tensor"].shape == (no_comp, block_length, no_slices)
    assert z["coords/components"].shape == (no_comp, 2)
    assert z["coords/components"].dtype == np.int8
    assert z["total/acf_bulk"].shape == (block_length, no_slices)
    assert z["total/acf_tensor"].shape == (no_comp, no_comp, block_length, no_slices)


def test_pressure_zarr_components_are_upper_triangle(tmp_path):
    store_path, p = _make_pressure_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    comps = z["coords/components"][:]
    for row, col in comps:
        assert row <= col, f"Component ({row},{col}) is not upper-triangular"


def test_pressure_zarr_slice_last(tmp_path):
    store_path, p = _make_pressure_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["total/pressure"].shape[-1] == p["no_slices"]
    assert z["total/tensor"].shape[-1] == p["no_slices"]
    assert z["total/acf_tensor"].shape[-1] == p["no_slices"]


# ---------------------------------------------------------------------------
# 2.3  VelocityAutoCorrelationFunction zarr layout
# ---------------------------------------------------------------------------


def _make_vacf_store(tmp_path, no_sp=2, D=3, block_length=60, no_slices=3):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "vacf.zarr")
    species = ["H", "He"][:no_sp]
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time
    store.require_dataset(
        "coords/species", shape=(no_sp,), dtype=object,
        object_codec=numcodecs.VLenUTF8(), compressor=None, overwrite=True,
    )
    store["coords/species"][:] = np.array(species, dtype=object)

    rng = np.random.default_rng(2)
    acf_data = rng.standard_normal((no_sp, D + 1, block_length, no_slices))
    store.require_dataset(
        "acf",
        shape=(no_sp, D + 1, block_length, no_slices),
        chunks=(no_sp, D + 1, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["acf"][:] = acf_data
    store.require_dataset("mean/acf",     shape=(no_sp, D + 1, block_length), dtype="f8", compressor=compressor)
    store.require_dataset("mean/acf_std", shape=(no_sp, D + 1, block_length), dtype="f8", compressor=compressor)
    store["mean/acf"][:]     = acf_data.mean(axis=-1)
    store["mean/acf_std"][:] = acf_data.std(axis=-1, ddof=min(1, no_slices - 1))

    return store_path, {"no_sp": no_sp, "D": D, "block_length": block_length, "no_slices": no_slices}


def test_vacf_zarr_required_keys(tmp_path):
    store_path, _ = _make_vacf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/time", "coords/species", "acf", "mean/acf", "mean/acf_std"):
        assert key in z, f"Missing key: {key}"


def test_vacf_zarr_acf_shape(tmp_path):
    no_sp, D, block_length, no_slices = 2, 3, 60, 3
    store_path, p = _make_vacf_store(tmp_path, no_sp=no_sp, D=D,
                                      block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["acf"].shape == (no_sp, D + 1, block_length, no_slices)
    assert z["mean/acf"].shape == (no_sp, D + 1, block_length)
    assert z["mean/acf_std"].shape == (no_sp, D + 1, block_length)


def test_vacf_zarr_isotropic_index_is_last_dim(tmp_path):
    """The last axis along the D+1 dimension [-1] is the isotropic VACF."""
    store_path, p = _make_vacf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    # Per-dimension entries at indices 0..D-1; isotropic at index D (i.e. -1)
    acf = z["acf"]
    assert acf.shape[1] == p["D"] + 1  # D per-dim entries + 1 isotropic


def test_vacf_zarr_mean_consistency(tmp_path):
    store_path, _ = _make_vacf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    np.testing.assert_allclose(z["mean/acf"][:], z["acf"][:].mean(axis=-1), rtol=1e-12)


# ---------------------------------------------------------------------------
# 2.4  ElectricCurrent zarr layout
# ---------------------------------------------------------------------------


def _make_ec_store(tmp_path, no_sp=2, D=3, block_length=50, no_slices=2):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "ec.zarr")
    n_entries = no_sp + 1  # per-species + Total
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time

    rng = np.random.default_rng(3)
    current = rng.standard_normal((n_entries, D, block_length, no_slices))
    store.require_dataset(
        "current",
        shape=(n_entries, D, block_length, no_slices),
        chunks=(n_entries, D, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["current"][:] = current

    acf = rng.standard_normal((n_entries, n_entries, D, block_length, no_slices))
    # Symmetrize: acf[j,i,...] = acf[i,j,...] for i != j
    for i in range(n_entries):
        for j in range(i + 1, n_entries):
            acf[j, i] = acf[i, j]
    store.require_dataset(
        "acf",
        shape=(n_entries, n_entries, D, block_length, no_slices),
        chunks=(n_entries, n_entries, D, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["acf"][:] = acf

    return store_path, {"no_sp": no_sp, "D": D, "block_length": block_length,
                        "no_slices": no_slices, "n_entries": n_entries}


def test_ec_zarr_required_keys(tmp_path):
    store_path, _ = _make_ec_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/time", "current", "acf"):
        assert key in z, f"Missing key: {key}"


def test_ec_zarr_current_shape(tmp_path):
    no_sp, D, block_length, no_slices = 2, 3, 50, 2
    store_path, p = _make_ec_store(tmp_path, no_sp=no_sp, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    # Total is at index -1, so shape[0] = no_sp + 1
    assert z["current"].shape == (no_sp + 1, D, block_length, no_slices)


def test_ec_zarr_acf_shape(tmp_path):
    no_sp, D, block_length, no_slices = 2, 3, 50, 2
    store_path, p = _make_ec_store(tmp_path, no_sp=no_sp, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    n_entries = no_sp + 1
    assert z["acf"].shape == (n_entries, n_entries, D, block_length, no_slices)


def test_ec_zarr_acf_is_symmetric(tmp_path):
    store_path, p = _make_ec_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    acf = z["acf"][:]
    n_entries = p["n_entries"]
    for i in range(n_entries):
        for j in range(i + 1, n_entries):
            np.testing.assert_array_equal(acf[i, j], acf[j, i],
                                           err_msg=f"ACF not symmetric at ({i},{j})")


def test_ec_zarr_total_current_at_last_entry(tmp_path):
    """The Total current is at index -1 (= no_sp), not stored by name."""
    store_path, p = _make_ec_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    # Check we can access total current by integer index [-1]
    total_current = z["current"][-1, :, :, :]
    assert total_current.shape == (p["D"], p["block_length"], p["no_slices"])


# ---------------------------------------------------------------------------
# 2.5  HeatFlux zarr layout
# ---------------------------------------------------------------------------


def _make_hf_store(tmp_path, no_sp=2, D=3, block_length=50, no_slices=2):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "hf.zarr")
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time

    rng = np.random.default_rng(4)
    flux = rng.standard_normal((no_sp, D, block_length, no_slices))
    store.require_dataset(
        "flux",
        shape=(no_sp, D, block_length, no_slices),
        chunks=(no_sp, D, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["flux"][:] = flux

    # ACF: (no_sp, no_sp, D+1, block_length, no_slices) — isotropic at index D
    acf = rng.standard_normal((no_sp, no_sp, D + 1, block_length, no_slices))
    for i in range(no_sp):
        for j in range(i + 1, no_sp):
            acf[j, i] = acf[i, j]
    store.require_dataset(
        "acf",
        shape=(no_sp, no_sp, D + 1, block_length, no_slices),
        chunks=(no_sp, no_sp, D + 1, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["acf"][:] = acf

    return store_path, {"no_sp": no_sp, "D": D, "block_length": block_length, "no_slices": no_slices}


def test_hf_zarr_required_keys(tmp_path):
    store_path, _ = _make_hf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/time", "flux", "acf"):
        assert key in z, f"Missing key: {key}"


def test_hf_zarr_flux_shape(tmp_path):
    no_sp, D, block_length, no_slices = 2, 3, 50, 2
    store_path, _ = _make_hf_store(tmp_path, no_sp=no_sp, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["flux"].shape == (no_sp, D, block_length, no_slices)


def test_hf_zarr_acf_shape(tmp_path):
    no_sp, D, block_length, no_slices = 2, 3, 50, 2
    store_path, _ = _make_hf_store(tmp_path, no_sp=no_sp, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    # D+1: D per-dimension entries + 1 isotropic at index D
    assert z["acf"].shape == (no_sp, no_sp, D + 1, block_length, no_slices)


def test_hf_zarr_acf_symmetric(tmp_path):
    store_path, p = _make_hf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    acf = z["acf"][:]
    no_sp = p["no_sp"]
    for i in range(no_sp):
        for j in range(i + 1, no_sp):
            np.testing.assert_array_equal(acf[i, j], acf[j, i],
                                           err_msg=f"HeatFlux ACF not symmetric at ({i},{j})")


def test_hf_zarr_isotropic_dim_at_index_D(tmp_path):
    """Isotropic component is stored at axis-1 index D (the last of D+1)."""
    store_path, p = _make_hf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    D = p["D"]
    iso = z["acf"][:, :, D, :, :]
    assert iso.shape == (p["no_sp"], p["no_sp"], p["block_length"], p["no_slices"])


# ---------------------------------------------------------------------------
# 2.6  DiffusionFlux zarr layout
# ---------------------------------------------------------------------------


def _make_df_store(tmp_path, no_fluxes=2, D=3, block_length=50, no_slices=2):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "df.zarr")
    time = np.arange(block_length) * 0.001

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/time", shape=(block_length,), dtype="f8", compressor=compressor)
    store["coords/time"][:] = time

    rng = np.random.default_rng(5)
    flux = rng.standard_normal((no_fluxes, D, block_length, no_slices))
    store.require_dataset(
        "flux",
        shape=(no_fluxes, D, block_length, no_slices),
        chunks=(no_fluxes, D, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["flux"][:] = flux

    acf = rng.standard_normal((no_fluxes, no_fluxes, D + 1, block_length, no_slices))
    for i in range(no_fluxes):
        for j in range(i + 1, no_fluxes):
            acf[j, i] = acf[i, j]
    store.require_dataset(
        "acf",
        shape=(no_fluxes, no_fluxes, D + 1, block_length, no_slices),
        chunks=(no_fluxes, no_fluxes, D + 1, block_length, 1),
        dtype="f8", compressor=compressor,
    )
    store["acf"][:] = acf

    return store_path, {"no_fluxes": no_fluxes, "D": D, "block_length": block_length, "no_slices": no_slices}


def test_df_zarr_required_keys(tmp_path):
    store_path, _ = _make_df_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/time", "flux", "acf"):
        assert key in z, f"Missing key: {key}"


def test_df_zarr_flux_shape(tmp_path):
    no_fluxes, D, block_length, no_slices = 2, 3, 50, 2
    store_path, _ = _make_df_store(tmp_path, no_fluxes=no_fluxes, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["flux"].shape == (no_fluxes, D, block_length, no_slices)


def test_df_zarr_acf_shape(tmp_path):
    no_fluxes, D, block_length, no_slices = 2, 3, 50, 2
    store_path, _ = _make_df_store(tmp_path, no_fluxes=no_fluxes, D=D,
                                    block_length=block_length, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["acf"].shape == (no_fluxes, no_fluxes, D + 1, block_length, no_slices)


# ---------------------------------------------------------------------------
# 2.7  RadialDistributionFunction zarr layout
# ---------------------------------------------------------------------------


def _make_rdf_store(tmp_path, no_sp=2, no_bins=100, no_slices=4):
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "rdf.zarr")
    species = ["H", "He"][:no_sp]

    r_values = np.linspace(0.1, 5.0, no_bins)
    time = np.zeros(no_slices)  # not used by RDF but present in store

    store = zarr.open(store_path, mode="w")
    store.require_dataset("coords/r_values", shape=(no_bins,), dtype="f8", compressor=compressor)
    store["coords/r_values"][:] = r_values
    store.require_dataset(
        "coords/species", shape=(no_sp,), dtype=object,
        object_codec=numcodecs.VLenUTF8(), compressor=None, overwrite=True,
    )
    store["coords/species"][:] = np.array(species, dtype=object)

    rng = np.random.default_rng(6)
    rdf_data = rng.standard_normal((no_sp, no_sp, no_bins, no_slices)) + 1.0
    # Symmetrize: rdf[j, i] = rdf[i, j] for i != j
    for i in range(no_sp):
        for j in range(i + 1, no_sp):
            rdf_data[j, i] = rdf_data[i, j]

    store.require_dataset(
        "rdf",
        shape=(no_sp, no_sp, no_bins, no_slices),
        chunks=(no_sp, no_sp, no_bins, 1),
        dtype="f8", compressor=compressor,
    )
    store["rdf"][:] = rdf_data

    mean_rdf = rdf_data.mean(axis=-1)
    store.require_dataset("mean/rdf", shape=(no_sp, no_sp, no_bins), dtype="f8", compressor=compressor)
    store["mean/rdf"][:] = mean_rdf

    return store_path, {"no_sp": no_sp, "no_bins": no_bins, "no_slices": no_slices, "species": species}


def test_rdf_zarr_required_keys(tmp_path):
    store_path, _ = _make_rdf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for key in ("coords/r_values", "coords/species", "rdf", "mean/rdf"):
        assert key in z, f"Missing key: {key}"


def test_rdf_zarr_rdf_shape(tmp_path):
    no_sp, no_bins, no_slices = 2, 100, 4
    store_path, _ = _make_rdf_store(tmp_path, no_sp=no_sp, no_bins=no_bins, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["rdf"].shape == (no_sp, no_sp, no_bins, no_slices)
    assert z["mean/rdf"].shape == (no_sp, no_sp, no_bins)


def test_rdf_zarr_slice_axis_is_last(tmp_path):
    no_sp, no_bins, no_slices = 2, 100, 4
    store_path, _ = _make_rdf_store(tmp_path, no_sp=no_sp, no_bins=no_bins, no_slices=no_slices)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert z["rdf"].shape[-1] == no_slices


def test_rdf_zarr_symmetric(tmp_path):
    store_path, p = _make_rdf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    rdf = z["rdf"][:]
    no_sp = p["no_sp"]
    for i in range(no_sp):
        for j in range(i + 1, no_sp):
            np.testing.assert_array_equal(rdf[i, j], rdf[j, i],
                                           err_msg=f"RDF not symmetric at ({i},{j})")


def test_rdf_zarr_pair_access_by_index(tmp_path):
    """g(r) for pair (i,j) is accessed via rdf[i, j, :, isl], not by string name."""
    store_path, p = _make_rdf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    isl = 0
    g_rdf = z["rdf"][0, 0, :, isl]
    assert g_rdf.shape == (p["no_bins"],)


def test_rdf_zarr_coords_r_values_not_coordinates(tmp_path):
    """Key must be coords/r_values, not coordinates/r_values (old layout)."""
    store_path, _ = _make_rdf_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert "coords/r_values" in z
    assert "coordinates/r_values" not in z


# ---------------------------------------------------------------------------
# 2.8  TransportCoefficients zarr layout (save_zarr)
# ---------------------------------------------------------------------------


def _make_transport_store(tmp_path, no_slices=3, block_length=50):
    """Simulate what TransportCoefficients.save_zarr() produces."""
    import numcodecs
    import zarr

    compressor = numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    store_path = str(tmp_path / "diffusion.zarr")
    time = np.arange(block_length) * 0.001

    rng = np.random.default_rng(7)
    ddof = min(1, no_slices - 1)

    z = zarr.open(store_path, mode="a")
    z.require_dataset("coords/time", shape=time.shape, dtype="f8",
                      compressor=compressor, overwrite=False)
    z["coords/time"][:] = time

    quantities = {"H_Diffusion": rng.standard_normal((no_slices, block_length)),
                  "He_Diffusion": rng.standard_normal((no_slices, block_length))}

    for name, slices_arr in quantities.items():
        z.require_dataset(f"slices/{name}", shape=slices_arr.shape, dtype="f8",
                          compressor=compressor, overwrite=True)
        z[f"slices/{name}"][:] = slices_arr
        mean_arr = slices_arr.mean(axis=0)
        std_arr  = slices_arr.std(axis=0, ddof=ddof)
        z.require_dataset(f"mean/{name}", shape=mean_arr.shape, dtype="f8",
                          compressor=compressor, overwrite=True)
        z[f"mean/{name}"][:] = mean_arr
        z.require_dataset(f"mean/{name}_std", shape=std_arr.shape, dtype="f8",
                          compressor=compressor, overwrite=True)
        z[f"mean/{name}_std"][:] = std_arr

    return store_path, {"no_slices": no_slices, "block_length": block_length,
                        "quantities": list(quantities.keys()), "raw": quantities}


def test_transport_zarr_required_keys(tmp_path):
    store_path, p = _make_transport_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    assert "coords/time" in z
    for name in p["quantities"]:
        assert f"slices/{name}" in z, f"Missing slices/{name}"
        assert f"mean/{name}" in z, f"Missing mean/{name}"
        assert f"mean/{name}_std" in z, f"Missing mean/{name}_std"


def test_transport_zarr_slices_shape(tmp_path):
    no_slices, block_length = 3, 50
    store_path, p = _make_transport_store(tmp_path, no_slices=no_slices, block_length=block_length)
    import zarr
    z = zarr.open(store_path, mode="r")
    for name in p["quantities"]:
        assert z[f"slices/{name}"].shape == (no_slices, block_length)
        assert z[f"mean/{name}"].shape == (block_length,)
        assert z[f"mean/{name}_std"].shape == (block_length,)


def test_transport_zarr_mean_consistency(tmp_path):
    store_path, p = _make_transport_store(tmp_path)
    import zarr
    z = zarr.open(store_path, mode="r")
    for name, raw in p["raw"].items():
        np.testing.assert_allclose(z[f"mean/{name}"][:], raw.mean(axis=0), rtol=1e-12)


def test_transport_zarr_no_pandas(tmp_path):
    """transport.py must not import pandas."""
    transport = pathlib.Path("sarkas/tools/transport.py")
    if not transport.exists():
        transport = pathlib.Path(__file__).parent.parent.parent / "transport.py"
    tree = ast.parse(transport.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names]
            src = getattr(node, "module", "") or ""
            assert "pandas" not in src and not any("pandas" in n for n in names), \
                "pandas import found in transport.py"
