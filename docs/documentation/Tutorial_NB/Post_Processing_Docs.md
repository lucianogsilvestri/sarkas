# Data Analysis

Once the simulation is complete we enter the **Post Processing** stage where we _measure_ physical quantities and calculate transport coefficients.

Sarkas was developed to make this process as simple and straightforward as possible.

The YAML input file can be found at [input_file](https://raw.githubusercontent.com/murillo-group/sarkas/master/docs/documentation/Tutorial_NB/input_files/yukawa_mks_p3m.yaml) and this notebook at [notebook](https://raw.githubusercontent.com/murillo-group/sarkas/master/docs/documentation/Tutorial_NB/Post_Processing_Docs.ipynb)

Let's import the needed packages.

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import zarr

plt.style.use('MSUstyle')

from sarkas.processes import PostProcess
from sarkas.tools.observables import (
    Thermodynamics,
    RadialDistributionFunction,
    StaticStructureFactor,
    DynamicStructureFactor,
)

input_file_name = os.path.join('input_files', 'yukawa_mks_p3m.yaml')
```

Similar to the **Pre Processing** and **Simulation** stages, three lines are enough.

```python
postproc = PostProcess(input_file_name)
postproc.setup(read_yaml=True)
```

Each observable is an object stored as an attribute of the `PostProcess`. The class names are:

| Attribute | Class |
|---|---|
| `therm` | `Thermodynamics` |
| `rdf` | `RadialDistributionFunction` |
| `ssf` | `StaticStructureFactor` |
| `dsf` | `DynamicStructureFactor` |
| `ccf` | `CurrentCorrelationFunction` |
| `ec` | `ElectricCurrent` |
| `vd` | `VelocityDistribution` |
| `vacf` | `VelocityAutoCorrelationFunction` |
| `diff_flux` | `DiffusionFlux` *(mixtures only)* |
| `p_tensor` | `PressureTensor` |

> **Note on data storage.** In this version of Sarkas all observable data is stored on disk as [zarr](https://zarr.readthedocs.io) arrays instead of pandas DataFrames. The path to the store is always available as `obs.zarr_store_path`. To load any result, open the store and read the arrays you need:
>
> ```python
> import zarr
> z = zarr.open(obs.zarr_store_path, mode="r")
> z.tree()   # inspect the full layout
> ```

---

## Thermodynamics

```python
therm = Thermodynamics()
therm.setup(postproc.parameters)
therm.compute()
```

After `compute()` the zarr store has the following layout (for a single-species simulation with species name `"H"`):

```
coordinates/
    time               (block_length,)        – time axis of one slice
species/
    H/
        Temperature    (no_slices, block_length)
        Kinetic Energy (no_slices, block_length)
        Potential Energy (no_slices, block_length)
        Total Energy   (no_slices, block_length)
        …
mean/
    H/
        Temperature    (block_length,)         – mean over slices
        Temperature_std (block_length,)        – std over slices
        …                                        (NaN-free; single-slice → std = 0)
```

To read the slice-averaged temperature of species `"H"`:

```python
z = zarr.open(therm.zarr_store_path, mode="r")
time = z["coordinates/time"][:]          # shape (block_length,)
T_mean = z["mean/H/Temperature"][:]      # shape (block_length,)
T_std  = z["mean/H/Temperature_std"][:]  # shape (block_length,)
```

The diagnostic temperature + energy plot works exactly as before:

```python
fig, T_axes, E_axes = therm.temp_energy_plot(
    postproc,          # pass the Process object for the info panel
    phase='production'
)
```

On the left the info panel reproduces key simulation parameters. The main panels show Temperature (left) and Total Energy (right) vs time. The smaller panels above show percentage deviation from the desired temperature / initial energy. The panels to the right are histograms compared to theoretical Gaussian distributions.

---

## Radial Distribution Function

```python
rdf = RadialDistributionFunction()
rdf.setup(postproc.parameters)
rdf.compute()
```

The zarr store layout is:

```
coordinates/
    r_values           (no_bins,)    – bin centres in simulation length units
    ra_values          (no_bins,)    – bin centres normalised by a_ws  (r/a_ws)
slices/
    rdf                (no_slices, no_pairs, no_bins)
mean/
    rdf                (no_pairs, no_bins)   – mean over slices
std/
    rdf                (no_pairs, no_bins)   – std over slices  (zero for single slice)
metadata/
    species_pairs      (no_pairs,)   – pair labels, e.g. ["H-H"]
```

To load and plot:

```python
z = zarr.open(rdf.zarr_store_path, mode="r")

ra      = z["coordinates/ra_values"][:]   # r/a_ws
pairs   = list(z["metadata/species_pairs"][:])   # e.g. ["H-H"]
rdf_mean = z["mean/rdf"][:]    # (no_pairs, no_bins)
rdf_std  = z["std/rdf"][:]     # (no_pairs, no_bins)

fig, ax = plt.subplots()
for i, pair in enumerate(pairs):
    ax.plot(ra, rdf_mean[i], label=pair)
    ax.fill_between(ra,
                    rdf_mean[i] - rdf_std[i],
                    rdf_mean[i] + rdf_std[i],
                    alpha=0.2)
ax.set(xlabel=r'$r/a_{\rm ws}$', ylabel=r'$g(r)$')
ax.legend()
```

> The `Std` is zero (not `NaN`) when only one slice was computed — so the `fill_between` call is always safe.

---

## Velocity Auto-Correlation Function

```python
from sarkas.tools.observables import VelocityAutoCorrelationFunction

vacf = VelocityAutoCorrelationFunction()
vacf.setup(postproc.parameters)
vacf.compute()   # computes ACF by default
```

The zarr store layout (3D simulation, species `"H"`):

```
coordinates/
    time               (block_length,)
species/
    H/
        acf            (no_slices, D+1, block_length)
            # axis-1 index: 0=x, 1=y, 2=z, 3=total (isotropic average)
mean/
    H/
        acf            (D+1, block_length)
        acf_std        (D+1, block_length)
```

To load and plot, normalising by the plasma period:

```python
z = zarr.open(vacf.zarr_store_path, mode="r")

time = z["coordinates/time"][:]         # shape (block_length,)
acf  = z["mean/H/acf"][:]               # shape (D+1, block_length)
std  = z["mean/H/acf_std"][:]

# Normalise: divide by the zero-lag value so VACF(0) = 1
acf_norm = acf / acf[:, 0:1]

t_pp = time / vacf.plasma_period        # time in plasma periods

dim_labels = ["X", "Y", "Z", "Total"]
fig, ax = plt.subplots()
for d, lbl in enumerate(dim_labels):
    ax.plot(t_pp, acf_norm[d], label=lbl)
ax.set(xlabel="Plasma Periods", ylabel="VACF (normalised)")
ax.legend()
```

To compute the self-diffusion coefficient from the VACF:

```python
from sarkas.tools.transport import Diffusion

diff = Diffusion()
diff.setup(postproc.parameters, vacf, therm)
diff.compute(vacf)
```

---

## Static Structure Factor

The SSF is the time-averaged auto-correlation of density fluctuations:

$$S(\mathbf{k}) = \langle n(-\mathbf{k})\, n(\mathbf{k}) \rangle$$

For each snapshot Sarkas computes $n(\mathbf{k}, t) = \sum_j e^{-i\mathbf{k}\cdot\mathbf{r}_j(t)}$ then averages $|n(\mathbf{k},t)|^2$ over the slice.

The allowed wave-vectors are:

$$\mathbf{k}(n_x, n_y, n_z) = \frac{2\pi}{L}(n_x, n_y, n_z), \quad n_{x,y,z} = 0, 1, 2, \dots$$

Sarkas offers three angle-averaging strategies selectable in the YAML:

| `angle_averaging` | Description |
|---|---|
| `'principal_axis'` | Only $\mathbf{k}(n,0,0)$ and permutations *(default, fast)* |
| `'custom'` | Full averaging up to `max_aa_ka_value`, principal axis beyond |
| `'full'` | Every integer triplet *(slowest, most accurate at large $k$)* |

```python
postproc.ssf.compute()
```

The zarr store layout (pair `"H-H"`):

```
coordinates/
    ka_values          (no_ka_values,)   – unique |k|·a_ws values
H-H/
    sk_raw             (no_ka_values, no_dumps)  – raw S(k,t) for all dumps
    mean               (no_ka_values,)
    std                (no_ka_values,)
```

To load and plot with error shading:

```python
z = zarr.open(postproc.ssf.zarr_store_path, mode="r")

ka      = z["coordinates/ka_values"][:]
sk_mean = z["H-H/mean"][:]
sk_std  = z["H-H/std"][:]

fig, ax = plt.subplots()
ax.plot(ka, sk_mean, label="H-H SSF")
ax.fill_between(ka, sk_mean - sk_std, sk_mean + sk_std, alpha=0.2)
ax.set(xlabel=r"$ka$", ylabel=r"$S(k)$", ylim=(0, 3))
ax.legend()
```

The standard deviation is zero (not `NaN`) for a single slice, so the `fill_between` is always safe.

To switch to full angle averaging and recompute:

```python
postproc.ssf.angle_averaging = 'full'
postproc.ssf.setup(postproc.parameters)
postproc.ssf.pretty_print()   # inspect the new k-grid (will be large!)
postproc.ssf.compute()
```

---

### Summary of zarr store layouts

| Observable | Key paths after `compute()` |
|---|---|
| `Thermodynamics` | `coordinates/time`, `species/{sp}/{Key}`, `mean/{sp}/{Key}`, `mean/{sp}/{Key}_std` |
| `RadialDistributionFunction` | `coordinates/r_values`, `coordinates/ra_values`, `mean/rdf`, `std/rdf`, `metadata/species_pairs` |
| `VelocityAutoCorrelationFunction` | `coordinates/time`, `species/{sp}/acf` `(no_slices, D+1, T)`, `mean/{sp}/acf`, `mean/{sp}/acf_std` |
| `StaticStructureFactor` | `coordinates/ka_values`, `{sp1}-{sp2}/mean`, `{sp1}-{sp2}/std` |
| `DynamicStructureFactor` | `coordinates/frequencies`, `coordinates/ka_values`, `{sp1}-{sp2}/mean` `(no_k, no_freqs)`, `{sp1}-{sp2}/std` |
| `PressureTensor` | `coordinates/time`, `slices/Total_Pressure`, `mean/Total_Pressure`, `mean/Total_Pressure_Tensor_{ij}` |