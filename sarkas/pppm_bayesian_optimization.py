"""
Bayesian Optimization for PPPM Parameter Selection in Sarkas.

This module implements a Bayesian Optimization (BO) approach to find
optimal PPPM parameters (r_c, alpha, M, CAO, fftw_threads) that minimise
simulation time subject to a force error constraint.

Architecture
------------
The optimization treats the problem as constrained BO with two surrogate models:
  - Objective GP  : models tot_acc_time (to minimise)
  - Constraint GP : models force_error  (must be <= target_error)

A multi-fidelity warm-start uses the analytical force error approximation
(force_error_approx, already computed in Sarkas) to cheaply seed the surrogates
before any expensive MD timing calls are made.

Parameters optimised
--------------------
  r_c          (continuous) : real-space cutoff radius
  alpha        (continuous) : Ewald splitting parameter
  M            (integer)    : mesh size per dimension, restricted to FFT-friendly values
  CAO          (integer)    : charge assignment order (B-spline order), 1-7
  fftw_threads (integer)    : number of threads for FFTW in the PM step

Dependencies
------------
  botorch  >= 0.9
  gpytorch >= 1.11
  torch    >= 2.0
  scipy    >= 1.10
  numpy
  pandas

Usage
-----
Add the mixin class BayesianPPPMOptimizer to PreProcess (or call its methods
directly after instantiation). The main entry point mirrors the existing API:

    results = preprocess.timing_study_calculation(
        target_error=1e-5,
        method="bayesian"
    )

The method returns the same dict as the brute-force approach so that
downstream code is unaffected.
"""

from __future__ import annotations

import warnings
from os.path import join
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from numpy import (
    argmin,
    array,
    asarray,
    clip,
    exp,
    full,
    isnan,
    linspace,
    log,
    log10,
    logspace,
    ndarray,
    pi,
    sqrt,
)
from numpy import random as np_random

# BoTorch / GPyTorch imports — guarded so the rest of Sarkas still loads
# even when BO dependencies are not installed.
try:
    from botorch.acquisition.analytic import LogConstrainedExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MixedSingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.optim import optimize_acqf_mixed
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False
    warnings.warn(
        "BoTorch / GPyTorch not found. Install with:\n"
        "  pip install botorch gpytorch\n"
        "Bayesian optimization methods will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# FFT-friendly mesh sizes (products of small primes 2, 3, 5)
# ---------------------------------------------------------------------------
FFT_FRIENDLY_MESHES: List[int] = [
    8, 12, 16, 18, 24, 32, 36, 48, 54, 64, 72, 96,
    108, 128, 144, 192, 216, 256, 288, 384, 432, 512,
]

# CAO (charge assignment order / B-spline order) options
CAO_OPTIONS: List[int] = [1, 2, 3, 4, 5, 6, 7]

# Default FFTW thread counts to explore.  Powers of two up to physical core
# count are typical sweet spots; the caller may pass a narrower list.
FFTW_THREAD_OPTIONS: List[int] = [1, 2, 4, 8, 16]


# ---------------------------------------------------------------------------
# Utility: physical parameter bounds
# ---------------------------------------------------------------------------


def compute_physical_bounds(
    a_ws: float,
    box_length: float,
    mesh_options: List[int],
    cao_options: Optional[List[int]] = None,
    fftw_thread_options: Optional[List[int]] = None,
    rc_min_override: Optional[float] = None,
    rc_max_override: Optional[float] = None,
) -> Dict:
    """
    Compute physically motivated bounds for PPPM parameters.

    Parameters
    ----------
    a_ws : float
        Wigner-Seitz radius (length unit).
    box_length : float
        Minimum box side length.
    mesh_options : list of int
        Candidate mesh sizes (e.g. pm_meshes from the user).
    cao_options : list of int, optional
        Candidate charge assignment orders. Defaults to CAO_OPTIONS [1..7].
    fftw_thread_options : list of int, optional
        Candidate thread counts for the PM FFT step.
        Defaults to FFTW_THREAD_OPTIONS.
    rc_min_override : float, optional
        Override the physical lower bound on r_c.  If None, defaults to
        2 * a_ws (hard-core exclusion).
    rc_max_override : float, optional
        Override the physical upper bound on r_c.  If None, defaults to
        0.49 * box_length (minimum image convention).

    Returns
    -------
    dict with keys:
        rc_min, rc_max       : float bounds for r_c
        alpha_min, alpha_max : float bounds for alpha
        M_options            : list of int, feasible mesh sizes
        cao_options          : list of int
        fftw_thread_options  : list of int
    """
    rc_min = rc_min_override if rc_min_override is not None else 2.0 * a_ws
    rc_max = rc_max_override if rc_max_override is not None else 0.49 * box_length

    # Safety: rc_min must be strictly less than rc_max.
    if rc_min >= rc_max:
        rc_min = 2.0 * a_ws
        rc_max = 0.49 * box_length

    largest_mesh = max(mesh_options)
    smallest_mesh = min(mesh_options)
    alpha_min = 0.15 * smallest_mesh / box_length
    alpha_max = 0.60 * largest_mesh / box_length

    cao_list = cao_options if cao_options is not None else CAO_OPTIONS
    thread_list = fftw_thread_options if fftw_thread_options is not None else FFTW_THREAD_OPTIONS

    # Cap thread options at the available core count so we never schedule more
    # threads than the machine can run concurrently.
    n_cores = _retrieve_available_cores()
    thread_list = [t for t in thread_list if t <= n_cores]
    if not thread_list:
        thread_list = [1]

    return {
        "rc_min": rc_min,
        "rc_max": rc_max,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "M_options": mesh_options,
        "cao_options": cao_list,
        "fftw_thread_options": thread_list,
    }


def _retrieve_available_cores() -> int:
    """
    Retrieve the number of available CPU cores.

    Returns
    -------
    int
        Number of available CPU cores, capped at 64.
    """
    import multiprocessing

    return min(multiprocessing.cpu_count(), 64)


# ---------------------------------------------------------------------------
# Utility: encode / decode mixed parameter vectors
# ---------------------------------------------------------------------------


class PPPMParameterCodec:
    """
    Handles encoding/decoding between the mixed (continuous + categorical)
    PPPM parameter space and the normalised tensor that BoTorch expects.

    Continuous dimensions (normalised to [0, 1]):
        0 : r_c
        1 : alpha

    Categorical dimensions (integer indices into option lists):
        2 : index into M_options
        3 : index into cao_options
        4 : index into fftw_thread_options

    The MixedSingleTaskGP in BoTorch takes a raw tensor where categorical
    columns hold the integer *index* (not the actual value).  This codec
    provides the translation layer.

    Note on the fixed_features_list
    --------------------------------
    optimize_acqf_mixed enumerates every combination of categorical values and
    optimises the continuous part for each combination.  With three categorical
    parameters the list has len(M) * len(CAO) * len(threads) entries.  For
    large option sets this can be expensive; consider restricting the option
    lists to the most promising candidates before calling the BO loop.
    """

    def __init__(self, bounds: Dict):
        self.rc_min = bounds["rc_min"]
        self.rc_max = bounds["rc_max"]
        self.alpha_min = bounds["alpha_min"]
        self.alpha_max = bounds["alpha_max"]
        self.M_options = bounds["M_options"]
        self.cao_options = bounds["cao_options"]
        self.fftw_thread_options = bounds["fftw_thread_options"]

        # Indices of categorical columns for BoTorch
        self.cat_dims = [2, 3, 4]

    @property
    def n_dims(self) -> int:
        return 5  # rc, alpha, M_idx, cao_idx, thread_idx

    @property
    def botorch_bounds(self) -> torch.Tensor:
        """
        Bounds tensor of shape (2, n_dims) for optimize_acqf_mixed.
        Continuous dims normalised to [0, 1]; categorical dims [0, n-1].
        """
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.double)
        upper = torch.tensor(
            [
                1.0,
                1.0,
                float(len(self.M_options) - 1),
                float(len(self.cao_options) - 1),
                float(len(self.fftw_thread_options) - 1),
            ],
            dtype=torch.double,
        )
        return torch.stack([lower, upper])

    def encode(
        self,
        rc: float,
        alpha: float,
        M: int,
        cao: int,
        fftw_threads: int,
    ) -> torch.Tensor:
        """Physical params -> normalised tensor of shape (1, 5)."""
        rc_norm = (rc - self.rc_min) / (self.rc_max - self.rc_min)
        alpha_norm = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)

        M_idx = min(range(len(self.M_options)), key=lambda i: abs(self.M_options[i] - M))
        cao_idx = min(range(len(self.cao_options)), key=lambda i: abs(self.cao_options[i] - cao))
        thread_idx = min(
            range(len(self.fftw_thread_options)),
            key=lambda i: abs(self.fftw_thread_options[i] - fftw_threads),
        )

        return torch.tensor(
            [[rc_norm, alpha_norm, float(M_idx), float(cao_idx), float(thread_idx)]],
            dtype=torch.double,
        )

    def decode(self, x: torch.Tensor) -> Tuple[float, float, int, int, int]:
        """Normalised tensor (1, 5) or (5,) -> physical params."""
        x = x.squeeze()
        rc = float(x[0]) * (self.rc_max - self.rc_min) + self.rc_min
        alpha = float(x[1]) * (self.alpha_max - self.alpha_min) + self.alpha_min

        M_idx = max(0, min(int(round(float(x[2]))), len(self.M_options) - 1))
        cao_idx = max(0, min(int(round(float(x[3]))), len(self.cao_options) - 1))
        thread_idx = max(0, min(int(round(float(x[4]))), len(self.fftw_thread_options) - 1))

        return (
            rc,
            alpha,
            self.M_options[M_idx],
            self.cao_options[cao_idx],
            self.fftw_thread_options[thread_idx],
        )

    def fixed_features_list(self) -> List[Dict[int, float]]:
        """
        Returns the list of fixed-feature dicts required by
        optimize_acqf_mixed — one dict per combination of categorical values.
        """
        fixed = []
        for m_idx in range(len(self.M_options)):
            for c_idx in range(len(self.cao_options)):
                for t_idx in range(len(self.fftw_thread_options)):
                    fixed.append({2: float(m_idx), 3: float(c_idx), 4: float(t_idx)})
        return fixed


# ---------------------------------------------------------------------------
# High-fidelity initial design
# ---------------------------------------------------------------------------


def _find_analytical_best_alpha(
    rc: float,
    M: int,
    cao: int,
    codec: PPPMParameterCodec,
    analytical_error_fn,
    n_grid: int = 100,
) -> float:
    """
    Find the alpha that minimises the analytical force error at fixed (rc, M, cao).

    Used to place initial design points in the most informative part of the
    alpha axis — the valley where PP and PM errors are balanced — so that
    initial HF evaluations are more likely to be feasible and the GP starts
    with useful signal rather than pure noise.

    Parameters
    ----------
    rc, M, cao          : fixed parameter values
    codec               : PPPMParameterCodec (provides alpha bounds)
    analytical_error_fn : callable(rc, alpha, M, cao) -> float
    n_grid              : number of alpha values on the search grid

    Returns
    -------
    float : alpha that minimises the analytical force error
    """
    alpha_grid = logspace(log10(codec.alpha_min), log10(codec.alpha_max), n_grid)
    errors = array([analytical_error_fn(rc, a, M, cao) for a in alpha_grid])
    return float(alpha_grid[int(argmin(errors))])


def hf_initial_design(
    codec: PPPMParameterCodec,
    evaluate_fn,
    analytical_error_fn,
    target_error: float,
    n_top_cats: int = 6,
    verbose: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Structured high-fidelity initial design for the BO warm-start.

    Replaces the analytical warm-start grid with real MD evaluations chosen
    to give the GP surrogates the most informative possible starting data.
    The design has three phases:

    Phase A — Thread sweep
    ----------------------
    Evaluates every thread count option at a single fixed (M, cao, rc, alpha)
    that is analytically predicted to be feasible.  This gives the GP its
    only source of truth about thread scaling before BO begins.  Without
    these points the surrogate has a completely flat prior over the thread
    dimension and the acquisition function cannot distinguish between thread
    counts.

    The anchor (M, cao) is chosen as the analytically cheapest (M, cao) pair
    that is predicted to satisfy the error constraint.  If no pair is
    analytically predicted to be feasible, the pair with the smallest
    analytical error is used instead.  alpha is set to the analytical optimum
    for that pair; rc is set to the midpoint of its range.

    Cost: len(fftw_thread_options) evaluations.

    Phase B — Categorical coverage
    --------------------------------
    Evaluates the n_top_cats (M, cao) combinations predicted to have the
    smallest analytical force error at their optimal alpha, using the fastest
    thread count identified in Phase A and rc at the midpoint.

    This ensures the GP has at least one real data point per categorical cell
    it is likely to explore, so the kernel length-scales over M and CAO can be
    estimated from data rather than defaulting to the prior.

    n_top_cats is capped at len(M_options) * len(cao_options) to avoid
    redundant evaluations when the option lists are small.

    Cost: min(n_top_cats, |M| * |CAO|) evaluations.

    Phase C — Continuous LHS refinement
    -------------------------------------
    For the single best (M, cao, threads) combination identified so far,
    evaluates two additional (rc, alpha) points drawn from a 2-D Latin
    Hypercube Sample.  This gives the GP enough continuous variation to
    estimate the length-scales in rc and alpha within the best categorical
    cell, preventing the acquisition function from over-exploiting a single
    continuous point.

    Cost: 2 evaluations.

    Total cost: len(fftw_thread_options) + min(n_top_cats, |M|*|CAO|) + 2

    Parameters
    ----------
    codec               : PPPMParameterCodec instance
    evaluate_fn         : callable(rc, alpha, M, cao, fftw_threads) -> (time, error)
    analytical_error_fn : callable(rc, alpha, M, cao) -> float
    target_error        : force error constraint threshold
    n_top_cats          : number of (M, cao) pairs to evaluate in Phase B
    verbose             : print progress

    Returns
    -------
    X_init   : torch.Tensor, shape (n_init, 5)  — encoded parameter vectors
    Y_time   : torch.Tensor, shape (n_init, 1)  — negated log10 times
    Y_error  : torch.Tensor, shape (n_init, 1)  — log10 force errors
    records  : list of dict, one entry per evaluation
    """
    X_list:  List[torch.Tensor] = []
    T_list:  List[float]        = []
    E_list:  List[float]        = []
    records: List[Dict]         = []

    rc_mid = 0.5 * (codec.rc_min + codec.rc_max)

    def _record_and_store(rc, alpha, M, cao, fftw_threads, label):
        """Run one HF evaluation and accumulate tensors + records."""
        time_val, error_val, pp_acc_time, pm_acc_time = evaluate_fn(rc, alpha, M, cao, fftw_threads)
        feasible = bool(error_val <= target_error)

        if verbose:
            status = "✓" if feasible else "✗"
            print(
                f"  [{label}] rc={rc:.6e}  alpha={alpha:.6e}  M={M:3d}  "
                f"CAO={cao}  thr={fftw_threads:2d}  "
                f"PP acc: {pp_acc_time:.4e}s  PM acc: {pm_acc_time:.4e}s  "
                f"err={error_val:.2e}  t={time_val:.3e}s  {status}"
            )

        x = codec.encode(rc, alpha, M, cao, fftw_threads)
        X_list.append(x)
        T_list.append(-log10(max(time_val, 1e-30)))
        E_list.append( log10(max(error_val, 1e-30)))
        records.append({
            "rc": rc, "alpha": alpha, "M": M, "cao": cao,
            "fftw_threads": fftw_threads,
            "pp_acc_time": pp_acc_time, "pm_acc_time": pm_acc_time,
            "time": time_val, "force_error": error_val,
            "fidelity": "initial", "feasible": feasible,
        })
        return time_val, error_val, feasible

    # ------------------------------------------------------------------
    # Pre-compute analytical optima for every (M, cao) pair.
    # Sort by analytical error so Phase A and B can pick the best cells.
    # ------------------------------------------------------------------
    cat_cells = []  # list of (analytical_error, M, cao, best_alpha)
    # grab only 4 random M values to keep the initial design cost manageable; the BO loop will explore the rest.
    M_vals = np_random.choice(codec.M_options, size=min(4, len(codec.M_options)), replace=False)
    for M in M_vals:
        for cao in codec.cao_options:
            best_alpha = _find_analytical_best_alpha(rc_mid, M, cao, codec, analytical_error_fn)
            an_err = analytical_error_fn(rc_mid, best_alpha, M, cao)
            cat_cells.append((an_err, M, cao, best_alpha))
    cat_cells.sort(key=lambda t: t[0])  # ascending by analytical error

    # Anchor cell for Phase A: analytically cheapest feasible cell, or
    # the cell with the smallest error if none are predicted feasible.
    feasible_cells = [(err, M, cao, alpha) for err, M, cao, alpha in cat_cells
                      if err <= target_error]
    anchor_err, anchor_M, anchor_cao, anchor_alpha = (
        feasible_cells[0] if feasible_cells else cat_cells[0]
    )

    # ------------------------------------------------------------------
    # Phase A: Thread sweep at the anchor cell
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Phase A: Thread sweep ".center(60, "-"))
        print(
            f"  Anchor: M={anchor_M}  CAO={anchor_cao}  "
            f"alpha={anchor_alpha:.3e}  rc={rc_mid:.3e}"
        )

    phase_a_times: Dict[int, float] = {}
    for thr in codec.fftw_thread_options:
        t, e, _ = _record_and_store(rc_mid, anchor_alpha, anchor_M, anchor_cao, thr, f"A thr={thr}")
        phase_a_times[thr] = t

    # Best thread count = fastest observed in Phase A.
    best_threads = min(phase_a_times, key=phase_a_times.get)
    if verbose:
        print(f"  → Fastest thread count: {best_threads}")

    # ------------------------------------------------------------------
    # Phase B: Categorical coverage — top n_top_cats (M, cao) cells
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Phase B: Categorical coverage ".center(60, "-"))

    n_cats = min(n_top_cats, len(cat_cells))
    # Skip the anchor cell if it already appears among the top n_cats
    # (it was evaluated at all thread counts in Phase A; use best_threads).
    evaluated_cats = {(anchor_M, anchor_cao)}

    for an_err, M, cao, best_alpha in cat_cells[:n_cats]:
        if (M, cao) in evaluated_cats:
            continue
        evaluated_cats.add((M, cao))
        _record_and_store(rc_mid, best_alpha, M, cao, best_threads, f"B M={M} cao={cao}")

    # ------------------------------------------------------------------
    # Phase C: LHS refinement on (rc, alpha) at the best categorical cell
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Phase C: LHS refinement on (rc, alpha) ".center(60, "-"))

    # Best categorical cell = lowest observed error among initial points so far.
    best_rec = min(records, key=lambda r: r["force_error"])
    lhs_M, lhs_cao, lhs_threads = best_rec["M"], best_rec["cao"], best_threads

    # 2-point maximin LHS in [0,1]^2 mapped to (rc, alpha).
    # A 2-point LHS in 2 dimensions places one point in each half of each
    # axis: {(U[0,0.5], U[0.5,1]), (U[0.5,1], U[0,0.5])} — maximally spread.
    rng = np_random.default_rng(seed=0)
    lhs_u = array([[rng.uniform(0.0, 0.5), rng.uniform(0.5, 1.0)],
                   [rng.uniform(0.5, 1.0), rng.uniform(0.0, 0.5)]])

    for i, (u_rc, u_alpha) in enumerate(lhs_u):
        lhs_rc    = codec.rc_min    + u_rc    * (codec.rc_max    - codec.rc_min)
        lhs_alpha = codec.alpha_min + u_alpha * (codec.alpha_max - codec.alpha_min)
        _record_and_store(lhs_rc, lhs_alpha, lhs_M, lhs_cao, lhs_threads, f"C lhs={i}")

    # ------------------------------------------------------------------
    # Assemble tensors
    # ------------------------------------------------------------------
    X_init  = torch.cat(X_list, dim=0)
    Y_time  = torch.tensor(T_list, dtype=torch.double).unsqueeze(-1)
    Y_error = torch.tensor(E_list, dtype=torch.double).unsqueeze(-1)

    if verbose:
        n_feas = sum(1 for r in records if r["feasible"])
        print(
            f"\n  Initial design complete: {len(records)} HF evaluations, "
            f"{n_feas} feasible."
        )

    return X_init, Y_time, Y_error, records


# ---------------------------------------------------------------------------
# Surrogate model builder
# ---------------------------------------------------------------------------


def build_surrogate(
    X: torch.Tensor,
    Y: torch.Tensor,
    cat_dims: List[int],
) -> MixedSingleTaskGP:
    """
    Build and fit a MixedSingleTaskGP surrogate.

    Parameters
    ----------
    X        : (n, d) training inputs
    Y        : (n, 1) training targets
    cat_dims : list of column indices that are categorical

    Returns
    -------
    Fitted MixedSingleTaskGP
    """
    model = MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=cat_dims,
        outcome_transform=Standardize(m=1),
    )
    model = model.double()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


# ---------------------------------------------------------------------------
# Acquisition function: Constrained Log Expected Improvement
# ---------------------------------------------------------------------------


def build_constrained_acqf(
    objective_model: MixedSingleTaskGP,
    constraint_model: MixedSingleTaskGP,
    best_f: float,
    log_target_error: float,
) -> LogConstrainedExpectedImprovement:
    """
    Build a Constrained Log Expected Improvement acquisition function.

    We minimise time (negate for EI), subject to force_error <= target_error.
    Both the objective and constraint GPs operate in log10 space.

    Parameters
    ----------
    objective_model   : GP for -log10(tot_acc_time) (negated, so we maximise)
    constraint_model  : GP for log10(force_error)
    best_f            : current best negated log10 time among feasible points
    log_target_error  : log10(target_error), the constraint upper bound

    Returns
    -------
    LogConstrainedExpectedImprovement
    """
    from botorch.models import ModelListGP

    model_list = ModelListGP(objective_model, constraint_model)
    # Constraint: log10(force_error) <= log10(target_error)
    constraints = {1: (None, log_target_error)}

    return LogConstrainedExpectedImprovement(
        model=model_list,
        best_f=best_f,
        objective_index=0,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Previous-run loader
# ---------------------------------------------------------------------------


def load_previous_run(
    path_or_df,
    codec: PPPMParameterCodec,
    target_error: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
    """
    Load results from a previous BO run and convert them into the tensors
    expected by the surrogate models.

    Only rows with ``fidelity == "high"`` are used as seed data.  Analytical
    rows are discarded; the analytical model can regenerate them cheaply, and
    mixing fidelity levels in a single GP requires a multi-fidelity model.

    Parameters
    ----------
    path_or_df   : str, Path, or pandas.DataFrame
    codec        : PPPMParameterCodec for the *current* run
    target_error : float, feasibility threshold

    Returns
    -------
    X_prev  : torch.Tensor, shape (n, 5)
    Y_time  : torch.Tensor, shape (n, 1)  — negated log10 times
    Y_error : torch.Tensor, shape (n, 1)  — log10 force errors
    records : list of dict, ready to prepend to the run log

    Raises
    ------
    ValueError  if required columns are missing.
    RuntimeError if no high-fidelity rows are found.
    """
    df = path_or_df.copy() if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)

    required_cols = {"rc", "alpha", "M", "cao", "time", "force_error"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Previous run CSV is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "fidelity" in df.columns:
        hf_df = df[df["fidelity"] == "high"].copy()
    else:
        hf_df = df.copy()

    if hf_df.empty:
        raise RuntimeError(
            "No high-fidelity rows found in the previous run data.\n"
            "The previous run may not have completed any MD evaluations."
        )

    # Default thread count for old CSVs that pre-date the fftw_threads column.
    default_threads = codec.fftw_thread_options[0]

    X_list, T_list, E_list, records = [], [], [], []

    for _, row in hf_df.iterrows():
        rc = float(row["rc"])
        alpha = float(row["alpha"])
        M = int(row["M"])
        cao = int(row["cao"])
        fftw_threads = int(row.get("fftw_threads", default_threads))
        pp_acc_time = float(row.get("pp_acc_time", 0.0))
        pm_acc_time = float(row.get("pm_acc_time", 0.0))
        time = float(row["time"])
        error = float(row["force_error"])

        x = codec.encode(rc, alpha, M, cao, fftw_threads)
        X_list.append(x)
        T_list.append(-log10(max(time, 1e-30)))
        E_list.append(log10(max(error, 1e-30)))

        records.append(
            {
                "rc": rc,
                "alpha": alpha,
                "M": M,
                "cao": cao,
                "fftw_threads": fftw_threads,
                "pp_acc_time": pp_acc_time,
                "pm_acc_time": pm_acc_time,
                "time": time,
                "force_error": error,
                "fidelity": "previous",
                "feasible": error <= target_error,
            }
        )

    X_prev = torch.cat(X_list, dim=0)
    Y_time = torch.tensor(T_list, dtype=torch.double).unsqueeze(-1)
    Y_error = torch.tensor(E_list, dtype=torch.double).unsqueeze(-1)

    return X_prev, Y_time, Y_error, records


# ---------------------------------------------------------------------------
# Core BO loop
# ---------------------------------------------------------------------------


def run_bo_loop(
    codec: PPPMParameterCodec,
    evaluate_fn,
    analytical_error_fn,
    target_error: float,
    n_top_cats: int = 6,
    n_bo_iterations: int = 30,
    n_restarts: int = 10,
    raw_samples: int = 256,
    verbose: bool = True,
    warm_start_from=None,
) -> pd.DataFrame:
    """
    Bayesian Optimization loop for PPPM parameters.

    Phase 1 — Structured HF initial design (or previous-run seed)
        Runs a small number of real MD evaluations chosen to give the GP
        surrogates maximally informative starting data:
          A. Thread sweep at the analytically best (M, cao) cell.
          B. Categorical coverage of the top n_top_cats (M, cao) pairs.
          C. LHS refinement on (rc, alpha) within the best observed cell.
        If ``warm_start_from`` is provided, HF measurements from a prior
        run replace phases A-C entirely.

    Phase 2 — High-fidelity BO
        Iteratively proposes candidates via Constrained LogEI, evaluates
        them with actual MD timing calls, and updates the surrogates.

    Parameters
    ----------
    codec               : PPPMParameterCodec instance
    evaluate_fn         : (rc, alpha, M, cao, fftw_threads) -> (time, error)
    analytical_error_fn : (rc, alpha, M, cao) -> float
                          Used only to rank (M, cao) cells and locate the
                          optimal alpha for initial design points.
    target_error        : force error constraint threshold
    n_top_cats          : number of (M, cao) pairs covered in Phase B of the
                          initial design (capped at |M| * |CAO|)
    n_bo_iterations     : number of high-fidelity BO evaluations in Phase 2
    n_restarts          : restarts for acquisition optimisation
    raw_samples         : random samples for acquisition initialisation
    verbose             : print progress
    warm_start_from     : path/DataFrame for previous-run seeding, or None

    Returns
    -------
    pandas.DataFrame with columns:
        rc, alpha, M, cao, fftw_threads, time, force_error, fidelity, feasible
    """
    if not _BOTORCH_AVAILABLE:
        raise ImportError(
            "BoTorch is required for Bayesian optimization. "
            "Install with: pip install botorch gpytorch"
        )

    records: List[Dict] = []

    # ------------------------------------------------------------------
    # Phase 1: Initial design
    # ------------------------------------------------------------------
    if warm_start_from is not None:
        if verbose:
            src_label = (
                str(warm_start_from)
                if not isinstance(warm_start_from, pd.DataFrame)
                else f"DataFrame ({len(warm_start_from)} rows)"
            )
            print("\n" + " Phase 1: Loading previous run ".center(60, "="))
            print(f"  Source: {src_label}")

        X_init, Y_time_init, Y_error_init, init_records = load_previous_run(
            warm_start_from, codec, target_error
        )
        records.extend(init_records)

        if verbose:
            n_feasible_prev = sum(1 for r in init_records if r["feasible"])
            print(f"  Loaded {len(init_records)} previous HF points ({n_feasible_prev} feasible)")
            best_prev = min(
                (r["time"] for r in init_records if r["feasible"]),
                default=None,
            )
            if best_prev is not None:
                print(f"  Best feasible time from previous run: {best_prev:.4e} s")

        # load_previous_run already returns negated log10 times.
        Y_time_neg_init = Y_time_init

    else:
        if verbose:
            print("\n" + " Phase 1: Structured HF initial design ".center(60, "="))

        X_init, Y_time_neg_init, Y_error_init, init_records = hf_initial_design(
            codec=codec,
            evaluate_fn=evaluate_fn,
            analytical_error_fn=analytical_error_fn,
            target_error=target_error,
            n_top_cats=n_top_cats,
            verbose=verbose,
        )
        records.extend(init_records)

    # ------------------------------------------------------------------
    # Phase 2: High-fidelity BO
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Phase 2: High-fidelity Bayesian Optimization ".center(60, "="))
        print(f"  Budget: {n_bo_iterations} MD evaluations")

    # Build initial surrogates from the initial design data.
    # Objective GP: maximises -log10(time)  (= minimises time)
    # Constraint GP: models log10(force_error)
    obj_model = build_surrogate(X_init, Y_time_neg_init, codec.cat_dims)
    con_model = build_surrogate(X_init, Y_error_init, codec.cat_dims)

    # Accumulate HF observations for incremental surrogate updates.
    X_hf = torch.empty((0, codec.n_dims), dtype=torch.double)
    Y_time_hf = torch.empty((0, 1), dtype=torch.double)   # negated log10 time
    Y_error_hf = torch.empty((0, 1), dtype=torch.double)  # log10 error

    log_target_error = log10(target_error)

    best_feasible_time = float("inf")
    best_feasible_point: Optional[Dict] = None

    bounds = codec.botorch_bounds
    fixed_features = codec.fixed_features_list()

    for iteration in range(n_bo_iterations):
        # ---- Determine best_f from GP posterior mean --------------------
        # We approximate feasibility by the constraint GP's posterior mean
        # rather than the raw observations, so the acquisition function can
        # explore regions the GP predicts to be feasible but that have not
        # yet been sampled.
        with torch.no_grad():
            X_all = torch.cat([X_init, X_hf], dim=0) if X_hf.shape[0] > 0 else X_init
            con_mean = con_model.posterior(X_all).mean.squeeze(-1)   # (n,)
            obj_mean = obj_model.posterior(X_all).mean.squeeze(-1)   # (n,)

            feasible_mask = con_mean <= log_target_error
            if feasible_mask.any():
                best_f = float(obj_mean[feasible_mask].max())
            else:
                # No feasible point known yet; use the global optimistic value
                # to encourage exploration toward the constraint boundary.
                best_f = float(obj_mean.max())

        acqf = build_constrained_acqf(obj_model, con_model, best_f, log_target_error)

        # ---- Optimise acquisition over mixed space ----------------------
        try:
            candidate, _ = optimize_acqf_mixed(
                acq_function=acqf,
                bounds=bounds,
                fixed_features_list=fixed_features,
                q=1,
                num_restarts=n_restarts,
                raw_samples=raw_samples,
            )
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration + 1}] Acquisition optimisation failed: {e}")
                print("  Falling back to random candidate.")
            rng = np_random.default_rng()
            candidate = torch.as_tensor(
                rng.uniform(
                    bounds[0].numpy(), bounds[1].numpy(), size=(1, codec.n_dims)
                ),
                dtype=torch.double,
            )
            # Snap categorical dims to valid integer indices
            for cat_dim, n_opts in zip(
                codec.cat_dims,
                [len(codec.M_options), len(codec.cao_options), len(codec.fftw_thread_options)],
            ):
                candidate[0, cat_dim] = float(rng.integers(0, n_opts))

        # ---- Decode and evaluate ----------------------------------------
        rc, alpha, M, cao, fftw_threads = codec.decode(candidate)
        rc = float(clip(rc, codec.rc_min, codec.rc_max))
        alpha = float(clip(alpha, codec.alpha_min, codec.alpha_max))

        try:
            time_val, error_val, pp_acc_time, pm_acc_time = evaluate_fn(rc, alpha, M, cao, fftw_threads)
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration + 1}] MD evaluation failed: {e}. Skipping.")
            continue

        feasible = bool(error_val <= target_error)

        if verbose:
            status = "✓ FEASIBLE" if feasible else "✗ infeasible"
            print(
                f"  [{iteration + 1:3d}/{n_bo_iterations}] "
                f"rc={rc:.6e}  alpha={alpha:.6e}  M={M:3d}  CAO={cao}  "
                f"threads={fftw_threads:2d}  "
                f"pp_acc={pp_acc_time:.3e}s  pm_acc={pm_acc_time:.3e}s  "
                f"err={error_val:.2e}  t={time_val:.4e}s  {status}"
            )

        if feasible and time_val < best_feasible_time:
            best_feasible_time = time_val
            best_feasible_point = {
                "rc": rc,
                "alpha": alpha,
                "M": M,
                "cao": cao,
                "fftw_threads": fftw_threads,
                "pp_acc_time": pp_acc_time,
                "pm_acc_time": pm_acc_time,
                "time": time_val,
                "force_error": error_val,
            }
            if verbose:
                print(f"        *** New best: time={time_val:.4e}s ***")

        records.append(
            {
                "rc": rc,
                "alpha": alpha,
                "M": M,
                "cao": cao,
                "fftw_threads": fftw_threads,
                "time": time_val,
                "pp_acc_time": pp_acc_time,
                "pm_acc_time": pm_acc_time,
                "force_error": error_val,
                "fidelity": "high",
                "feasible": feasible,
            }
        )

        # ---- Accumulate HF data and rebuild surrogates ------------------
        x_new = codec.encode(rc, alpha, M, cao, fftw_threads)
        y_time_new = torch.tensor([[-log10(max(time_val, 1e-30))]], dtype=torch.double)
        y_error_new = torch.tensor([[log10(max(error_val, 1e-30))]], dtype=torch.double)

        X_hf = torch.cat([X_hf, x_new], dim=0)
        Y_time_hf = torch.cat([Y_time_hf, y_time_new], dim=0)
        Y_error_hf = torch.cat([Y_error_hf, y_error_new], dim=0)

        X_all = torch.cat([X_init, X_hf], dim=0)
        Y_time_all = torch.cat([Y_time_neg_init, Y_time_hf], dim=0)
        Y_err_all = torch.cat([Y_error_init, Y_error_hf], dim=0)

        try:
            obj_model = build_surrogate(X_all, Y_time_all, codec.cat_dims)
            con_model = build_surrogate(X_all, Y_err_all, codec.cat_dims)
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration + 1}] Surrogate refit failed: {e}. Keeping old model.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Optimization Complete ".center(60, "="))
        if best_feasible_point:
            p = best_feasible_point
            print(
                f"  Best feasible configuration:\n"
                f"    rc={p['rc']:.6e}  alpha={p['alpha']:.6e}  M={p['M']}\n"
                f"    CAO={p['cao']}  threads={p['fftw_threads']}\n"
                f"    pp_acc_time={p['pp_acc_time']:.3e}s  pm_acc_time={p['pm_acc_time']:.3e}s\n"
                f"    force_error={p['force_error']:.6e}  time={p['time']:.3e}s"
            )
        else:
            print("  No feasible configuration found within budget.")
            print("  Consider increasing target_error or n_bo_iterations.")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Analytical model helpers (wired to Sarkas internals)
# ---------------------------------------------------------------------------


def make_analytical_error_fn(
    a_ws: float,
    screening_length: float,
    rescaling_constant: float,
    box_length: float,
):
    """
    Returns a closure computing the analytical PPPM force error approximation.

    Internally uses force_error_analytic_pp and force_error_approx_pm from
    sarkas.utilities.maths when available, with a fallback approximation.

    Parameters
    ----------
    a_ws               : Wigner-Seitz radius (same units as rc, box_length).
    screening_length   : Screening length lambda.
    rescaling_constant : Pre-factor for the error expressions.
    box_length         : Minimum box side length.

    Returns
    -------
    callable(rc, alpha, M, cao) -> float
    """
    try:
        from .utilities.maths import force_error_analytic_pp, force_error_approx_pm
        _sarkas_available = True
    except ImportError:
        _sarkas_available = False
        warnings.warn(
            "sarkas.utilities.maths not found. "
            "Falling back to built-in force error approximations.",
            ImportWarning,
            stacklevel=2,
        )

    kappa = a_ws / screening_length

    def _fn(rc: float, alpha: float, M: int, cao: int) -> float:
        rc_adim = rc / a_ws
        alpha_adim = alpha * a_ws
        h_adim = (box_length / M) / a_ws

        if _sarkas_available:
            pp_err = force_error_analytic_pp(rc_adim, kappa, alpha_adim, rescaling_constant)
            pm_err = force_error_approx_pm(kappa, cao, h_adim, alpha_adim, rescaling_constant)
        else:
            pp_err = (
                2.0
                * exp(-(((0.5 * kappa) / (alpha_adim + 1e-30)) ** 2))
                * exp(-((alpha_adim * rc_adim) ** 2))
                / sqrt(rc_adim + 1e-30)
                * rescaling_constant
            )
            pm_err = rescaling_constant * (alpha_adim * h_adim) ** (2 * cao)

        return float(sqrt(pp_err**2 + pm_err**2))

    return _fn


def make_analytical_time_fn(N: int, box_length: float):
    """
    Returns a closure estimating PPPM cost per step analytically.

    The thread count is intentionally excluded: thread scaling is machine-
    dependent and cannot be captured analytically.  The BO will learn
    thread performance from high-fidelity measurements.

    Cost model (relative units):
        T_PP  ~ N * (4π/3) * (rc/L)^3          (particle pairs in cutoff sphere)
        T_PM  ~ M^3 * log(M^3)                  (FFT on the mesh)
        T_CAO ~ cao^3 * N                        (charge assignment)

    Parameters
    ----------
    N          : int, number of particles
    box_length : float

    Returns
    -------
    callable(rc, alpha, M, cao) -> float  (relative cost, arbitrary units)
    """
    def _fn(rc: float, alpha: float, M: int, cao: int) -> float:
        occupancy = (4.0 / 3.0) * pi * (rc / box_length) ** 3
        t_pp = N * occupancy
        t_pm = M**3 * log(M**3 + 1)
        t_cao = cao**3 * N
        return float(t_pp + t_pm + t_cao)

    return _fn


# ---------------------------------------------------------------------------
# Sarkas integration: the mixin class
# ---------------------------------------------------------------------------


class BayesianPPPMOptimizer:
    """
    Mixin class to add Bayesian PPPM optimization to Sarkas' PreProcess.

    Add to PreProcess via multiple inheritance:

        class PreProcess(BayesianPPPMOptimizer, ...):
            ...

    The mixin expects the following attributes on ``self``, all present on a
    standard Sarkas PreProcess instance:

        potential.rc, pppm_mesh, pppm_alpha_ewald, pppm_cao, pppm_aliases,
        pppm_h_array, pppm_pp_err, pppm_pm_err, force_error,
        force_error_approx, box_lengths, type, screening_length,
        total_num_ptcls, a_ws, pbox_volume, pppm_fftw_threads,
        pot_update_params, update_pm, update_linked_list,
        calculate_force_error, pppm_setup, setup

        parameters.verbose
        particles
        timer
        io
        species
    """

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _bayesian_parameter_selection(
        self,
        target_error: float = 1e-5,
        pp_cells: Optional[ndarray] = None,
        pm_meshes: Optional[List[int]] = None,
        pm_caos: Optional[List[int]] = None,
        fftw_thread_options: Optional[List[int]] = None,
        n_top_cats: int = 6,
        n_bo_iterations: int = 30,
        n_restarts: int = 10,
        raw_samples: int = 256,
        save_csv: bool = True,
        warm_start_from=None,
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """
        Run Bayesian Optimization to find optimal PPPM parameters.

        Parameters
        ----------
        target_error : float
            Force error constraint (Delta_F <= target_error).
        pp_cells : array-like of int, optional
            Candidate numbers of PP cells; rc = box_length / pp_cells.
        pm_meshes : list of int, optional
            Candidate mesh sizes M. Defaults to FFT_FRIENDLY_MESHES.
        pm_caos : list of int, optional
            Candidate charge assignment orders. Defaults to [1..7].
        fftw_thread_options : list of int, optional
            Candidate FFTW thread counts for the PM FFT.  Defaults to
            [1, 2, 4, 8, 16] capped at the available core count.
        n_top_cats : int
            Number of (M, cao) pairs to evaluate in Phase B of the initial
            design (categorical coverage).  Each pair costs one MD timing
            call.  Capped automatically at |M_options| * |cao_options|.
            Default 6 gives reasonable coverage without excessive cost.
        n_bo_iterations : int
            Number of high-fidelity MD evaluations in the BO phase (Phase 2).
        n_restarts : int
            Restarts for acquisition function optimisation.
        raw_samples : int
            Random samples for acquisition initialisation.
        save_csv : bool
            Whether to save results to CSV.
        warm_start_from : str, Path, or DataFrame, optional
            Seed from a previous BO run instead of the structured initial
            design.  When provided, phases A-C are skipped entirely.

        Returns
        -------
        (results_df, best_point)
            results_df : DataFrame with all evaluated configurations
            best_point : dict with optimal parameters, or None.
        """
        if not _BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required. Install with: pip install botorch gpytorch")

        msg = "\n\n{:=^70} \n".format(" PPPM Bayesian Optimization ")
        self.io.write_to_logger(msg)

        self._save_original_pppm_params()

        box_length = self.potential.box_lengths.min()
        a_ws = self.potential.a_ws
        N = self.potential.total_num_ptcls
        rescaling_constant = sqrt(3.0 / (4.0 * pi))

        mesh_options = list(pm_meshes) if pm_meshes is not None else FFT_FRIENDLY_MESHES
        cao_list = list(pm_caos) if pm_caos is not None else CAO_OPTIONS

        # Derive rc bounds from pp_cells if provided.
        rc_min_override = rc_max_override = None
        if pp_cells is not None and len(pp_cells) > 0:
            pp_cells_arr = asarray(pp_cells, dtype=float)
            rc_min_override = box_length / float(pp_cells_arr.max())
            rc_max_override = box_length / float(pp_cells_arr.min())

        bounds = compute_physical_bounds(
            a_ws,
            box_length,
            mesh_options,
            cao_options=cao_list,
            fftw_thread_options=fftw_thread_options,
            rc_min_override=rc_min_override,
            rc_max_override=rc_max_override,
        )

        if self.parameters.verbose:
            print("\nPhysical bounds:")
            print(f"  r_c          : [{bounds['rc_min']:.4e}, {bounds['rc_max']:.4e}]")
            print(f"  alpha        : [{bounds['alpha_min']:.4e}, {bounds['alpha_max']:.4e}]")
            print(f"  M            : {bounds['M_options']}")
            print(f"  CAO          : {bounds['cao_options']}")
            print(f"  FFTW threads : {bounds['fftw_thread_options']}")

        codec = PPPMParameterCodec(bounds)
        self._bo_codec = codec

        analytical_error_fn = make_analytical_error_fn(
            a_ws,
            self.potential.screening_length,
            rescaling_constant,
            box_length,
        )
        evaluate_fn = self._make_hf_evaluator(rescaling_constant)

        results_df = run_bo_loop(
            codec=codec,
            evaluate_fn=evaluate_fn,
            analytical_error_fn=analytical_error_fn,
            target_error=target_error,
            n_top_cats=n_top_cats,
            n_bo_iterations=n_bo_iterations,
            n_restarts=n_restarts,
            raw_samples=raw_samples,
            verbose=self.parameters.verbose,
            warm_start_from=warm_start_from,
        )

        best_point = self._extract_best_point(results_df, target_error)

        # ------------------------------------------------------------------
        # Alpha refinement at fixed (rc, M, cao, fftw_threads)
        # ------------------------------------------------------------------
        # The BO finds the *first* feasible alpha, not necessarily the optimal
        # one.  Since alpha does not affect PP/PM/CAO costs (see _refine_alpha
        # docstring), we can minimise the analytical error along alpha at zero
        # additional MD cost.
        if best_point is not None:
            best_point = self._refine_alpha(
                best_point,
                analytical_error_fn,
                evaluate_fn,
                target_error,
            )
            results_df = pd.concat(
                [results_df, pd.DataFrame([best_point])], ignore_index=True
            )

            self._set_best_point_as_current_params(best_point)

        if save_csv:
            csv_path = join(
                self.io.directory_tree["preprocessing"]["path"],
                f"BayesianPPPM_data_{self.io.job_id}.csv",
            )
            results_df.to_csv(csv_path, index=False)
            msg = f"\nBO results saved to: {csv_path}"
            self.io.write_to_logger(msg)
            if self.parameters.verbose:
                print(msg)

        if best_point:
            msg = (
                f"\nOPTIMAL PPPM CONFIGURATION (BAYESIAN):\n"
                f"  Mesh: {best_point['M']} | CAO: {best_point['cao']} "
                f"| rc: {best_point['rc']:.6e}\n"
                f"  Ewald alpha: {best_point['alpha']:.6e} "
                f"| FFTW threads: {best_point['fftw_threads']}\n"
                f"  PP acc time: {best_point['pp_acc_time']:.3e}s"
                f"|  PM acc time: {best_point['pm_acc_time']:.3e}s "
                f"  Force Error: {best_point['force_error']:.6e} "
                f"| Total Time: {best_point['time']:.6e} s"
            )
        else:
            msg = "\nNo feasible configuration found.\nConsider increasing target_error or n_bo_iterations."
        self.io.write_to_logger(msg)
        if self.parameters.verbose:
            print(msg)

        # self._restore_original_pppm_params()

        return results_df, best_point

    # ------------------------------------------------------------------
    # Alpha refinement
    # ------------------------------------------------------------------

    def _refine_alpha(
        self,
        best_point: Dict,
        analytical_error_fn,
        evaluate_fn,
        target_error: float,
        n_grid: int = 200,
    ) -> Dict:
        """
        Given a feasible (rc, M, cao, fftw_threads) from the BO, find the
        alpha that minimises the analytical force error subject to the error
        staying below target_error, then verify with one HF call.

        Why this is valid
        -----------------
        PP cost  ~ N * (4π/3)(rc/L)³  — depends only on rc.
        PM cost  ~ M³ log M³          — depends only on M.
        CAO cost ~ cao³ N             — depends only on cao.
        Thread speedup                — depends only on fftw_threads.

        None of these costs depend on alpha, so total step time is constant
        as alpha varies.  The force error has a single valley (PP and PM
        errors balanced) that we locate with a dense 1-D grid on the cheap
        analytical model, then refine with scipy's bounded scalar minimiser.

        Parameters
        ----------
        best_point         : dict with keys rc, alpha, M, cao, fftw_threads,
                             time, force_error
        analytical_error_fn: cheap analytical error closure
        evaluate_fn        : HF evaluator closure
        target_error       : feasibility threshold
        n_grid             : number of alpha values for the initial grid

        Returns
        -------
        Updated best_point dict with refined alpha and force_error.
        """
        from scipy.optimize import minimize_scalar

        rc = best_point["rc"]
        M = best_point["M"]
        cao = best_point["cao"]
        fftw_threads = best_point["fftw_threads"]

        alpha_lo = self._bo_codec.alpha_min
        alpha_hi = self._bo_codec.alpha_max

        alpha_grid = linspace(alpha_lo, alpha_hi, n_grid)
        errors = array([analytical_error_fn(rc, a, M, cao) for a in alpha_grid])

        best_idx = int(argmin(errors))
        best_alpha = float(alpha_grid[best_idx])
        best_err = float(errors[best_idx])

        bracket_lo = alpha_grid[max(0, best_idx - 5)]
        bracket_hi = alpha_grid[min(n_grid - 1, best_idx + 5)]

        try:
            result = minimize_scalar(
                lambda a: analytical_error_fn(rc, float(a), M, cao),
                bounds=(bracket_lo, bracket_hi),
                method="bounded",
                options={"xatol": 1e-6},
            )
            if result.fun < best_err:
                best_alpha = float(result.x)
                best_err = float(result.fun)
        except Exception:
            pass  # fall back to grid result

        if best_err < best_point["force_error"]:
            time_val, error_val, pp_acc_time, pm_acc_time = evaluate_fn(rc, best_alpha, M, cao, fftw_threads)

            if self.parameters.verbose:
                print(
                    f"\n  Alpha refinement: {best_point['alpha']:.6e} → {best_alpha:.6e}"
                    f"  (analytical error {best_point['force_error']:.2e} → {best_err:.2e})"
                )

            updated = dict(best_point)
            updated["alpha"] = best_alpha
            updated["time"] = time_val
            updated["force_error"] = error_val
            updated["fidelity"] = "refined_alpha"
            updated["feasible"] = error_val <= target_error
            updated["pp_acc_time"] = pp_acc_time
            updated["pm_acc_time"] = pm_acc_time
            return updated

        return best_point

    # ------------------------------------------------------------------
    # High-fidelity evaluator
    # ------------------------------------------------------------------

    def _make_hf_evaluator(self, rescaling_constant: float):
        """
        Returns a closure that sets PPPM parameters in the Sarkas potential,
        measures actual PP and PM timing, and returns (total_time, force_error).

        Reinitialisation policy
        -----------------------
        M changed
            Full PM setup: reallocates the mesh (M³ elements), recomputes
            the influence function, and rebuilds all PM data structures.
            fftw_threads is always applied before pppm_setup.

        cao changed (M unchanged)
            Green's function must be recomputed; mesh array can be reused.
            pppm_setup is called again.

        alpha changed (M, cao unchanged)
            Only the influence function needs updating; mesh and assignment
            kernel are unchanged.  pppm_setup is called again.

        fftw_threads changed (M, cao, alpha unchanged)
            The thread count is passed to the FFT library before pppm_setup.
            pppm_setup is called to reinitialise the FFTW plan with the new
            thread count.

        rc changed only
            No PM reinitialisation needed.  The PP linked-list is rebuilt
            every timing call anyway.

        Parameters
        ----------
        rescaling_constant : float

        Returns
        -------
        callable(rc, alpha, M, cao, fftw_threads) -> (float, float)
        """
        _last: Dict = {"M": -1, "cao": -1, "alpha": float("nan"), "threads": -1}
        _alpha_tol: float = 1e-6  # relative tolerance for alpha change detection

        def _evaluate(
            rc: float, alpha: float, M: int, cao: int, fftw_threads: int
        ) -> Tuple[float, float]:
            M_changed = M != _last["M"]
            cao_changed = cao != _last["cao"]
            threads_changed = fftw_threads != _last["threads"]
            alpha_rel_diff = abs(alpha - _last["alpha"]) / (abs(_last["alpha"]) + 1e-30)
            alpha_changed = isnan(_last["alpha"]) or alpha_rel_diff > _alpha_tol

            pm_needs_setup = M_changed or cao_changed or alpha_changed or threads_changed

            # Always update the continuous parameters.
            self.potential.rc = rc
            self.potential.pppm_alpha_ewald = alpha

            if pm_needs_setup:
                self.potential.pppm_mesh = full(3, M, dtype=int)
                self.potential.pppm_cao = full(3, cao, dtype=int)
                self.potential.pppm_alpha_ewald = alpha
                # Set thread count BEFORE pppm_setup so the FFTW plan is
                # created with the correct number of threads.
                self.potential.pppm_fftw_threads = fftw_threads
                self.potential.pppm_setup()
                self.potential.pot_update_params(self.potential, self.species)

                _last["M"] = M
                _last["cao"] = cao
                _last["alpha"] = alpha
                _last["threads"] = fftw_threads

            # ---- Measure PM time ----------------------------------------
            n_trials = 1
            pm_acc_time = 0.0
            for _ in range(n_trials):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_acc_time += self.timer.stop()
            pm_acc_time = (pm_acc_time / n_trials) * 1.0e-9  # ns -> s

            # ---- Measure PP time ----------------------------------------
            pp_acc_time = 0.0
            for _ in range(n_trials):
                self.timer.start()
                self.potential.update_linked_list(self.particles)
                pp_acc_time += self.timer.stop()
            pp_acc_time = (pp_acc_time / n_trials) * 1.0e-9

            total_time = pp_acc_time + pm_acc_time

            # ---- Compute force error ------------------------------------
            self.potential.calculate_force_error()
            force_error = float(self.potential.force_error)

            return total_time, force_error, pp_acc_time, pm_acc_time

        return _evaluate

    # ------------------------------------------------------------------
    # Parameter save / restore
    # ------------------------------------------------------------------

    def _save_original_pppm_params(self):
        """Save current PPPM parameters so they can be restored later."""
        self._bo_saved_rc = float(self.potential.rc)
        self._bo_saved_mesh = self.potential.pppm_mesh.copy()
        self._bo_saved_alpha = float(self.potential.pppm_alpha_ewald)
        self._bo_saved_cao = self.potential.pppm_cao.copy()
        self._bo_saved_aliases = self.potential.pppm_aliases.copy()
        self._bo_saved_fftw_threads = int(self.potential.pppm_fftw_threads)

    def _restore_original_pppm_params(self):
        """Restore PPPM parameters saved by _save_original_pppm_params."""
        self.potential.rc = self._bo_saved_rc
        self.potential.pppm_mesh = self._bo_saved_mesh.copy()
        self.potential.pppm_alpha_ewald = self._bo_saved_alpha
        self.potential.pppm_cao = self._bo_saved_cao.copy()
        self.potential.pppm_aliases = self._bo_saved_aliases.copy()
        self.potential.pppm_fftw_threads = self._bo_saved_fftw_threads
        self.potential.estimate_parameters = False
        self.potential.setup(self.parameters, self.species)

    def _set_best_point_as_current_params(self, best_point: Dict):
        """Set the PPPM parameters in self.potential to the given best_point."""
        self.potential.rc = best_point["rc"]
        self.potential.pppm_alpha_ewald = best_point["alpha"]
        self.potential.pppm_mesh = full(3, best_point["M"], dtype=int)
        self.potential.pppm_cao = full(3, best_point["cao"], dtype=int)
        self.potential.pppm_fftw_threads = best_point.get("fftw_threads", 1)
        self.potential.estimate_parameters = False
        self.potential.setup(self.parameters, self.species)
    # ------------------------------------------------------------------
    # Result extraction and Pareto analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_best_point(
        results_df: pd.DataFrame,
        target_error: float,
    ) -> Optional[Dict]:
        """
        Return the high-fidelity feasible row with the smallest total time.

        Falls back to the analytical feasible row with smallest time if no
        high-fidelity feasible point exists.
        """
        for fid in ("high", "initial", "previous", "analytical"):
            subset = results_df[
                (results_df["fidelity"] == fid) & results_df["feasible"]
            ]
            if not subset.empty:
                row = subset.loc[subset["time"].idxmin()]
                result = {
                    "rc": float(row["rc"]),
                    "alpha": float(row["alpha"]),
                    "M": int(row["M"]),
                    "cao": int(row["cao"]),
                    "pp_acc_time": float(row["pp_acc_time"]),
                    "pm_acc_time": float(row["pm_acc_time"]),
                    "time": float(row["time"]),
                    "force_error": float(row["force_error"]),
                }
                # fftw_threads may be absent in DataFrames built from old code.
                if "fftw_threads" in row.index:
                    result["fftw_threads"] = int(row["fftw_threads"])
                return result

        return None

    @staticmethod
    def compute_pareto_front(
        results_df: pd.DataFrame,
        fidelity: str = "high",
    ) -> pd.DataFrame:
        """
        Compute the Pareto front in (force_error, time) space.

        A point is Pareto-optimal if no other point has both smaller time
        AND smaller force_error.

        Parameters
        ----------
        results_df : DataFrame from _bayesian_parameter_selection
        fidelity   : 'high', 'analytical', or 'all'

        Returns
        -------
        DataFrame of Pareto-optimal rows, sorted by force_error ascending.
        """
        df = results_df.copy() if fidelity == "all" else results_df[
            results_df["fidelity"] == fidelity
        ].copy()

        if df.empty:
            return df

        times = df["time"].values
        errors = df["force_error"].values
        n = len(df)

        # Vectorised Pareto dominance check using broadcasting.
        # dominated[i] = True if any j strictly dominates i.
        t_col = times.reshape(1, n)   # (1, n)
        e_col = errors.reshape(1, n)  # (1, n)
        t_row = times.reshape(n, 1)   # (n, 1)
        e_row = errors.reshape(n, 1)  # (n, 1)

        # j dominates i when t_col[j] <= t_row[i] and e_col[j] <= e_row[i]
        # with at least one strict inequality.
        strict_t = t_col < t_row   # (n, n): j strictly better in time
        strict_e = e_col < e_row   # (n, n): j strictly better in error
        weak_t = t_col <= t_row
        weak_e = e_col <= e_row

        # dominated[i] is True if any j != i strictly dominates i.
        # Self-dominance is impossible by construction: the strict inequality
        # in at least one objective is never satisfied when i == j, so the
        # diagonal of the (n, n) matrix is always False before the reduction.
        # No post-reduction correction is needed.
        dominated = ((weak_t & strict_e) | (strict_t & weak_e)).any(axis=1)

        pareto_df = df[~dominated].sort_values("force_error").reset_index(drop=True)
        return pareto_df

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def bo_convergence_summary(results_df: pd.DataFrame, target_error: float) -> Dict:
        """
        Compute convergence statistics from a BO run.

        Both 'initial' (structured design) and 'high' (BO phase) rows are
        real MD evaluations and count toward the improvement curve.
        'previous' rows from a warm-start chain are included too.
        Analytical rows are excluded.

        Returns
        -------
        dict with keys:
            n_initial           : evaluations in the structured initial design
            n_bo                : evaluations in the BO phase
            n_total_hf          : total real MD evaluations (initial + BO)
            n_feasible          : feasible real MD evaluations
            first_feasible_iter : index (across all real evals) of the first
                                  feasible point
            best_time           : time at best feasible point
            best_error          : error at best feasible point
            improvement_curve   : list of (eval_index, best_time_so_far)
        """
        real_fidelities = {"initial", "high", "previous", "refined_alpha"}
        hf = results_df[results_df["fidelity"].isin(real_fidelities)].reset_index(drop=True)

        n_initial  = int((results_df["fidelity"] == "initial").sum())
        n_bo       = int((results_df["fidelity"] == "high").sum())
        n_feasible = int(hf["feasible"].sum())

        first_idx        = None
        best_time_so_far = float("inf")
        best_error       = None
        improvement_curve = []

        for i, row in hf.iterrows():
            if row["feasible"] and row["time"] < best_time_so_far:
                best_time_so_far = row["time"]
                best_error = row["force_error"]
                if first_idx is None:
                    first_idx = i
            improvement_curve.append((i, best_time_so_far))

        return {
            "n_initial":           n_initial,
            "n_bo":                n_bo,
            "n_total_hf":          len(hf),
            "n_feasible":          n_feasible,
            "first_feasible_iter": first_idx,
            "best_time":           best_time_so_far if best_time_so_far < float("inf") else None,
            "best_error":          best_error,
            "improvement_curve":   improvement_curve,
        }

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_bo_results(
        self,
        results_df: pd.DataFrame,
        target_error: float,
        save_dir: Optional[str] = None,
        show_plots: bool = False,
    ):
        """
        Generate diagnostic plots for the BO run.

        Figures produced:
            1. Pareto front (force_error vs time)
            2. BO convergence curve (best feasible time vs iteration)
            3. Parameter scatter — rc vs alpha, M vs CAO, and thread count histogram
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plots.")
            return

        if save_dir is None:
            save_dir = getattr(self, "pppm_plots_dir", ".")

        an   = results_df[results_df["fidelity"] == "analytical"]
        init = results_df[results_df["fidelity"] == "initial"]
        hf   = results_df[results_df["fidelity"] == "high"]
        hf_feas   = hf[hf["feasible"]]
        hf_infeas = hf[~hf["feasible"]]

        # ---- Figure 1: Pareto front ------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))
        if not an.empty:
            ax.scatter(an["force_error"], an["time"] * 1e3,
                       c="lightblue", alpha=0.3, s=10, label="Analytical warm-start")
        if not init.empty:
            init_feas   = init[init["feasible"]]
            init_infeas = init[~init["feasible"]]
            ax.scatter(init_infeas["force_error"], init_infeas["time"] * 1e3,
                       c="lightsalmon", alpha=0.6, s=30, marker="^", label="Initial design infeasible")
            ax.scatter(init_feas["force_error"], init_feas["time"] * 1e3,
                       c="limegreen", alpha=0.8, s=50, marker="^", label="Initial design feasible")
        ax.scatter(hf_infeas["force_error"], hf_infeas["time"] * 1e3,
                   c="salmon", alpha=0.7, s=40, marker="x", label="BO infeasible")
        ax.scatter(hf_feas["force_error"], hf_feas["time"] * 1e3,
                   c="green", alpha=0.9, s=60, marker="o", label="BO feasible")

        pareto = self.compute_pareto_front(results_df, fidelity="all")
        if not pareto.empty:
            ax.plot(pareto["force_error"], pareto["time"] * 1e3,
                    "k--", linewidth=1.5, label="Pareto front (all HF)")

        ax.axvline(target_error, color="red", linestyle=":", linewidth=1.5,
                   label=f"Target error = {target_error:.0e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Force error", fontsize=12)
        ax.set_ylabel("Time per step (ms)", fontsize=12)
        ax.set_title("PPPM Pareto Front — BO Results", fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(join(save_dir, "BO_pareto_front.png"), dpi=150)
        plt.show() if show_plots else plt.close(fig)

        # ---- Figure 2: Convergence curve --------------------------------
        summary = self.bo_convergence_summary(results_df, target_error)
        if summary["improvement_curve"]:
            iters, best_times = zip(*summary["improvement_curve"])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.step(iters, [t * 1e3 for t in best_times], where="post",
                    color="steelblue", linewidth=2)
            ax.set_xlabel(
                f"MD evaluation index  "
                f"(initial design: 0 - {summary['n_initial'] - 1}, "
                f"BO: {summary['n_initial']}+)",
                fontsize=11,
            )
            ax.set_ylabel("Best feasible time (ms)", fontsize=12)
            ax.set_title("BO Convergence", fontsize=13)
            # Shade the initial design region
            if summary["n_initial"] > 0:
                ax.axvspan(0, summary["n_initial"] - 0.5,
                           alpha=0.08, color="orange", label="Initial design")
            if summary["first_feasible_iter"] is not None:
                ax.axvline(summary["first_feasible_iter"], color="green", linestyle="--",
                           label=f"First feasible (eval {summary['first_feasible_iter']})")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(join(save_dir, "BO_convergence.png"), dpi=150)
            plt.show() if show_plots else plt.close(fig)

        # ---- Figure 3: Parameter scatter --------------------------------
        if not hf.empty:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            edge_colors = ["green" if f else "red" for f in hf["feasible"]]
            sc = axes[0].scatter(
                hf["rc"], hf["alpha"],
                c=hf["M"], cmap="viridis", s=60, alpha=0.8,
                edgecolors=edge_colors, linewidths=1.5,
            )
            plt.colorbar(sc, ax=axes[0], label="Mesh size M")
            axes[0].set_xlabel(r"$r_c$", fontsize=12)
            axes[0].set_ylabel(r"$\alpha$", fontsize=12)
            axes[0].set_title("HF evaluations (colour = M)", fontsize=12)

            sc2 = axes[1].scatter(
                hf["M"], hf["cao"],
                c=hf["time"] * 1e3, cmap="plasma", s=80, alpha=0.8,
            )
            plt.colorbar(sc2, ax=axes[1], label="Time (ms)")
            axes[1].set_xlabel("Mesh size M", fontsize=12)
            axes[1].set_ylabel("CAO (B-spline order)", fontsize=12)
            axes[1].set_title("HF evaluations (colour = time)", fontsize=12)

            if "fftw_threads" in hf.columns:
                thread_counts = sorted(hf["fftw_threads"].unique())
                thread_times = [
                    hf.loc[hf["fftw_threads"] == t, "time"].mean() * 1e3
                    for t in thread_counts
                ]
                axes[2].bar([str(t) for t in thread_counts], thread_times,
                            color="steelblue", alpha=0.8)
                axes[2].set_xlabel("FFTW threads", fontsize=12)
                axes[2].set_ylabel("Mean time (ms)", fontsize=12)
                axes[2].set_title("Mean step time by thread count", fontsize=12)

            fig.tight_layout()
            fig.savefig(join(save_dir, "BO_parameter_scatter.png"), dpi=150)
            plt.show() if show_plots else plt.close(fig)

        if self.parameters.verbose:
            print(f"\nPlots saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Integration patch for PreProcess.timing_study_calculation
# ---------------------------------------------------------------------------

TIMING_STUDY_PATCH = '''
def timing_study_calculation(
        self, target_error=1e-5, pp_cells=None, pm_meshes=None,
        pm_caos=None, fftw_thread_options=None, method="brute_force", **kwargs
    ):
    """... (existing docstring) ..."""
    # ... (existing setup code) ...

    if method.lower() == "automated":
        return self._automated_parameter_selection(
            target_error, rescaling_constant, max_cells
        )
    elif method.lower() == "bayesian":
        return self._bayesian_parameter_selection(
            target_error=target_error,
            fftw_thread_options=fftw_thread_options,
            **kwargs
        )
    else:
        return self._brute_force_parameter_selection(
            pp_cells, pm_meshes, pm_caos, target_error, max_cells
        )
'''


# ---------------------------------------------------------------------------
# Standalone usage example (no Sarkas required)
# ---------------------------------------------------------------------------


def _demo_standalone():
    """
    Demonstrate the BO loop with synthetic models only.
    No Sarkas or actual MD is required.

    Run with:  python pppm_bayesian_optimization.py
    """
    print("=" * 60)
    print(" PPPM Bayesian Optimization — Standalone Demo ")
    print("=" * 60)

    N = 4096
    a_ws = 1.0
    box_length = (4.0 / 3.0 * pi * N) ** (1.0 / 3.0) * a_ws
    rescaling = sqrt(N) * a_ws**2 / sqrt(box_length**3)
    screening = 0.5 * box_length
    target_error = 1e-4

    bounds = compute_physical_bounds(
        a_ws,
        box_length,
        FFT_FRIENDLY_MESHES[:8],
        fftw_thread_options=[1, 2, 4],
    )
    codec = PPPMParameterCodec(bounds)

    print(f"\nSystem: N={N}, L={box_length:.2e}, a_ws={a_ws}")
    print(f"Target force error: {target_error:.0e}")
    print(f"r_c   bounds: [{bounds['rc_min']:.3e}, {bounds['rc_max']:.3e}]")
    print(f"alpha bounds: [{bounds['alpha_min']:.4e}, {bounds['alpha_max']:.4e}]")
    print(f"M options: {bounds['M_options']}")
    print(f"FFTW threads: {bounds['fftw_thread_options']}")

    analytical_error_fn = make_analytical_error_fn(a_ws, screening, rescaling, box_length)
    analytical_time_fn = make_analytical_time_fn(N, box_length)  # used only for synthetic_hf

    rng = np_random.default_rng(42)

    # Simulate thread scaling: more threads help the PM step up to a point.
    def _thread_speedup(fftw_threads: int) -> float:
        """Amdahl's law with 80 % parallel fraction."""
        parallel_fraction = 0.80
        return 1.0 / ((1 - parallel_fraction) + parallel_fraction / fftw_threads)

    def synthetic_hf(rc: float, alpha: float, M: int, cao: int, fftw_threads: int):
        base_time = analytical_time_fn(rc, alpha, M, cao)
        # PM portion benefits from threads; PP does not.
        pm_fraction = 0.6
        pp_time = base_time * (1 - pm_fraction)
        pm_time = base_time * pm_fraction / _thread_speedup(fftw_threads)
        time = (pp_time + pm_time) * (1.0 + 0.10 * rng.standard_normal())
        err = analytical_error_fn(rc, alpha, M, cao) * (1.0 + 0.05 * abs(rng.standard_normal()))
        return abs(time), abs(err)

    results_df = run_bo_loop(
        codec=codec,
        evaluate_fn=synthetic_hf,
        analytical_error_fn=analytical_error_fn,
        target_error=target_error,
        n_top_cats=4,
        n_bo_iterations=20,
        n_restarts=5,
        raw_samples=64,
        verbose=True,
    )

    best = BayesianPPPMOptimizer._extract_best_point(results_df, target_error)
    summary = BayesianPPPMOptimizer.bo_convergence_summary(results_df, target_error)

    print("\n" + "=" * 60)
    print("Convergence summary:")
    print(f"  HF evaluations : {summary['n_hf_total']}")
    print(f"  Feasible HF    : {summary['n_hf_feasible']}")
    print(f"  First feasible : iteration {summary['first_feasible_iter']}")
    if best:
        print("\nBest configuration:")
        for k, v in best.items():
            print(f"  {k}: {v}")

    results_df.to_csv("bo_demo_results.csv", index=False)
    print("\nResults saved to bo_demo_results.csv")


if __name__ == "__main__":
    _demo_standalone()