"""
Bayesian Optimization for PPPM Parameter Selection in Sarkas.

This module implements a Bayesian Optimization (BO) approach to find
optimal PPPM parameters (r_c, alpha, M, CAO) that minimize simulation
time subject to a force error constraint.

Architecture
------------
The optimization treats the problem as constrained BO with two surrogate models:
  - Objective GP  : models tot_acc_time (to minimize)
  - Constraint GP : models force_error  (must be <= target_error)

A multi-fidelity warm-start uses the analytical force error approximation
(force_error_approx, already computed in Sarkas) to cheaply seed the surrogates
before any expensive MD timing calls are made.

Parameters optimized
--------------------
  r_c   (continuous) : real-space cutoff radius
  alpha (continuous) : Ewald splitting parameter
  M     (integer)    : mesh size per dimension, restricted to FFT-friendly values
  CAO   (integer)    : charge assignment order (B-spline order), 1-7

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

import pandas as pd
import torch
import warnings
from numpy import (
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
    random,
    sqrt,
)
from os.path import join
from typing import Dict, List, Optional, Tuple

# BoTorch / GPyTorch imports — all guarded so the rest of Sarkas still
# loads even if BO dependencies are not installed.
try:
    # import gpytorch
    from botorch.acquisition.analytic import (  # LogExpectedImprovement,
        LogConstrainedExpectedImprovement,
    )
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import MixedSingleTaskGP

    # from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize

    # from botorch.acquisition import (
    #     qExpectedImprovement,
    #     qNoisyExpectedImprovement,
    # )
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

# from .utilities.maths import force_error_analytic_pp, force_error_approx_pm

# ---------------------------------------------------------------------------
# FFT-friendly mesh sizes (products of small primes 2, 3, 5)
# ---------------------------------------------------------------------------
FFT_FRIENDLY_MESHES: List[int] = [
    8,
    12,
    16,
    18,
    24,
    32,
    36,
    48,
    54,
    64,
    72,
    96,
    108,
    128,
    144,
    192,
    216,
    256,
    288,
    384,
    432,
    512,
]

# CAO (charge assignment order / B-spline order) options
CAO_OPTIONS: List[int] = [1, 2, 3, 4, 5, 6, 7]


# ---------------------------------------------------------------------------
# Utility: physical parameter bounds
# ---------------------------------------------------------------------------


def compute_physical_bounds(
    a_ws: float,
    box_length: float,
    mesh_options: List[int],
    cao_options: Optional[List[int]] = None,
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
        Candidate charge assignment orders (e.g. pm_caos from the user).
        Defaults to the global CAO_OPTIONS [1..7] if not provided.
    rc_min_override : float, optional
        Override the physical lower bound on r_c.  If None, defaults to
        2 * a_ws (hard-core exclusion).  Derived from pp_cells via
        rc = box_length / pp_cells_max when pp_cells are provided.
    rc_max_override : float, optional
        Override the physical upper bound on r_c.  If None, defaults to
        0.49 * box_length (minimum image convention).  Derived from
        pp_cells via rc = box_length / pp_cells_min when pp_cells provided.

    Returns
    -------
    dict with keys:
        rc_min, rc_max       : float bounds for r_c
        alpha_min, alpha_max : float bounds for alpha
        M_options            : list of int, feasible mesh sizes
        cao_options          : list of int
    """
    # r_c bounds — use overrides from pp_cells if provided, else physical defaults.
    rc_min = rc_min_override if rc_min_override is not None else 2.0 * a_ws
    rc_max = rc_max_override if rc_max_override is not None else 0.49 * box_length

    # Safety: rc_min must be strictly less than rc_max.
    if rc_min >= rc_max:
        rc_min = 2.0 * a_ws
        rc_max = 0.49 * box_length

    # alpha bounds derived from mesh options.
    largest_mesh = max(mesh_options)
    smallest_mesh = min(mesh_options)
    alpha_min = 0.15 * smallest_mesh / box_length
    alpha_max = 0.60 * largest_mesh / box_length

    # Discard mesh sizes whose spacing h = L/M would be larger than rc_min
    # (the mesh must be fine enough to resolve the real-space cutoff sphere).
    feasible_meshes = [m for m in mesh_options if m <= int(box_length / rc_min) * 4]
    if not feasible_meshes:
        feasible_meshes = mesh_options[:4]

    # CAO options — use caller-supplied list or fall back to global default.
    cao_list = cao_options if cao_options is not None else CAO_OPTIONS

    return {
        "rc_min": rc_min,
        "rc_max": rc_max,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "M_options": feasible_meshes,
        "cao_options": cao_list,
    }


# ---------------------------------------------------------------------------
# Utility: encode / decode mixed parameter vectors
# ---------------------------------------------------------------------------


class PPPMParameterCodec:
    """
    Handles encoding/decoding between the mixed (continuous + categorical)
    PPPM parameter space and the normalised [0,1]^d continuous tensor
    that BoTorch expects.

    Continuous dimensions (normalised to [0,1]):
        0 : r_c
        1 : alpha

    Categorical dimensions (integer indices into option lists):
        2 : index into M_options
        3 : index into cao_options

    The MixedSingleTaskGP in BoTorch takes a raw tensor where categorical
    columns hold the integer *index* (not the actual value).  This codec
    provides the translation layer.
    """

    def __init__(self, bounds: Dict):
        self.rc_min = bounds["rc_min"]
        self.rc_max = bounds["rc_max"]
        self.alpha_min = bounds["alpha_min"]
        self.alpha_max = bounds["alpha_max"]
        self.M_options = bounds["M_options"]
        self.cao_options = bounds["cao_options"]

        # Indices of categorical columns for BoTorch
        self.cat_dims = [2, 3]

    @property
    def n_dims(self) -> int:
        return 4  # rc, alpha, M_idx, cao_idx

    @property
    def botorch_bounds(self) -> torch.Tensor:
        """
        Bounds tensor of shape (2, n_dims) for optimize_acqf_mixed.
        Continuous dims: [0,1] (we normalise manually).
        Categorical dims: [0, len(options)-1].
        """
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.double)
        upper = torch.tensor(
            [
                1.0,
                1.0,
                float(len(self.M_options) - 1),
                float(len(self.cao_options) - 1),
            ],
            dtype=torch.double,
        )
        return torch.stack([lower, upper])

    def encode(self, rc: float, alpha: float, M: int, cao: int) -> torch.Tensor:
        """Physical params -> normalised tensor of shape (1, 4)."""
        rc_norm = (rc - self.rc_min) / (self.rc_max - self.rc_min)
        alpha_norm = (alpha - self.alpha_min) / (self.alpha_max - self.alpha_min)

        # Find closest index in option lists
        M_idx = min(range(len(self.M_options)), key=lambda i: abs(self.M_options[i] - M))
        cao_idx = min(range(len(self.cao_options)), key=lambda i: abs(self.cao_options[i] - cao))

        return torch.tensor(
            [[rc_norm, alpha_norm, float(M_idx), float(cao_idx)]],
            dtype=torch.double,
        )

    def decode(self, x: torch.Tensor) -> Tuple[float, float, int, int]:
        """Normalised tensor (1, 4) -> physical params."""
        x = x.squeeze()
        rc = float(x[0]) * (self.rc_max - self.rc_min) + self.rc_min
        alpha = float(x[1]) * (self.alpha_max - self.alpha_min) + self.alpha_min
        M_idx = int(round(float(x[2])))
        cao_idx = int(round(float(x[3])))

        M_idx = max(0, min(M_idx, len(self.M_options) - 1))
        cao_idx = max(0, min(cao_idx, len(self.cao_options) - 1))

        M = self.M_options[M_idx]
        cao = self.cao_options[cao_idx]
        return rc, alpha, M, cao

    def fixed_features_list(self) -> List[Dict[int, float]]:
        """
        Returns the list of fixed-feature dicts required by
        optimize_acqf_mixed — one dict per combination of categorical values.
        """
        fixed = []
        for m_idx in range(len(self.M_options)):
            for c_idx in range(len(self.cao_options)):
                fixed.append({2: float(m_idx), 3: float(c_idx)})
        return fixed


# ---------------------------------------------------------------------------
# Low-fidelity (analytical) warm-start grid
# ---------------------------------------------------------------------------


def analytical_warm_start(
    codec: PPPMParameterCodec,
    analytical_error_fn,
    analytical_time_fn,
    n_rc: int = 8,
    n_alpha: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate the analytical force error and timing model on a coarse grid
    to warm-start the BO surrogates.  No actual MD is run here.

    Grid design
    -----------
    rc : uniform in [rc_min, rc_max].
        The PP cost scales as rc^3 so uniform spacing gives reasonable
        coverage of the time objective.

    alpha : log-uniform in [alpha_min, alpha_max].
        The force error varies over many orders of magnitude as alpha
        changes — a linear grid concentrates almost all points in the
        high-error region and leaves the low-error valley (where the
        constraint becomes active) nearly unsampled.  Log spacing ensures
        points are spread across every decade of the error landscape.

    Both targets (time and error) are stored in log10 scale.
        Gaussian processes assume smooth, approximately Gaussian-distributed
        targets.  Raw force errors span ~5 orders of magnitude; fitting a
        GP to those raw values means the kernel length-scales are dominated
        by the large values and the surrogate is essentially flat everywhere
        except near the maximum.  Log-transforming makes the landscape
        smooth and homoscedastic, so the GP learns the gradient of the
        error surface correctly.  The constraint threshold is also
        log-transformed consistently everywhere it is used.

    Parameters
    ----------
    codec : PPPMParameterCodec
    analytical_error_fn : callable(rc, alpha, M, cao) -> float
    analytical_time_fn  : callable(rc, alpha, M, cao) -> float
    n_rc, n_alpha : int
        Number of grid points along each continuous dimension.

    Returns
    -------
    X_warm   : torch.Tensor, shape (n_pts, 4)
    Y_time   : torch.Tensor, shape (n_pts, 1)  — log10(analytical time)
    Y_error  : torch.Tensor, shape (n_pts, 1)  — log10(analytical error)
    """
    rc_vals = linspace(codec.rc_min, codec.rc_max, n_rc)
    # Log-uniform alpha: evenly spaced in log space so the low-error valley
    # (small alpha * a_ws) is sampled as densely as the high-error region.
    alpha_vals = logspace(
        log10(codec.alpha_min),
        log10(codec.alpha_max),
        n_alpha,
    )

    X_list, T_list, E_list = [], [], []

    for M_idx, M in enumerate(codec.M_options):
        for cao_idx, cao in enumerate(codec.cao_options):
            for rc in rc_vals:
                for alpha in alpha_vals:
                    err = analytical_error_fn(rc, alpha, M, cao)
                    time = analytical_time_fn(rc, alpha, M, cao)

                    # Clamp before log to avoid log(0) or log(negative)
                    err = max(err, 1e-30)
                    time = max(time, 1e-30)

                    x = codec.encode(rc, alpha, M, cao)
                    X_list.append(x)
                    T_list.append([[log10(time)]])
                    E_list.append([[log10(err)]])

    X_warm = torch.cat(X_list, dim=0)
    Y_time = torch.tensor(T_list, dtype=torch.double).squeeze(-1)
    Y_error = torch.tensor(E_list, dtype=torch.double).squeeze(-1)

    return X_warm, Y_time, Y_error


# ---------------------------------------------------------------------------
# Surrogate model builders
# ---------------------------------------------------------------------------


def build_surrogate(
    X: torch.Tensor,
    Y: torch.Tensor,
    cat_dims: List[int],
    noise_level: float = 1e-4,
) -> MixedSingleTaskGP:
    """
    Build and fit a MixedSingleTaskGP surrogate.

    Parameters
    ----------
    X          : (n, d) training inputs
    Y          : (n, 1) training targets
    cat_dims   : list of column indices that are categorical
    noise_level: initial noise variance

    Returns
    -------
    Fitted MixedSingleTaskGP
    """
    # Standardise outputs
    model = MixedSingleTaskGP(
        train_X=X,
        train_Y=Y,
        cat_dims=cat_dims,
        outcome_transform=Standardize(m=1),
    )
    model = model.double()
    # Create the Marginal Log Likelihood and fit the model hyperparameters.
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # Fit the model hyperparameters by maximizing the marginal log likelihood.
    fit_gpytorch_mll(mll)
    return model


def update_surrogate(
    model: MixedSingleTaskGP,
    X_new: torch.Tensor,
    Y_new: torch.Tensor,
) -> MixedSingleTaskGP:
    """
    Add new observations to an existing surrogate and refit hyperparameters.

    BoTorch does not support incremental updates on MixedSingleTaskGP, so
    we rebuild with the full dataset. For the typical sizes here (< 200 pts)
    this is fast.
    """
    X_all = torch.cat([model.train_inputs[0], X_new], dim=0)
    Y_all = torch.cat([model.train_targets.unsqueeze(-1), Y_new], dim=0)
    return build_surrogate(X_all, Y_all, cat_dims=[2, 3])


# ---------------------------------------------------------------------------
# Acquisition function: Constrained Log Expected Improvement
# ---------------------------------------------------------------------------


def build_constrained_acqf(
    objective_model: MixedSingleTaskGP,
    constraint_model: MixedSingleTaskGP,
    best_f: float,
    target_error: float,
) -> LogConstrainedExpectedImprovement:
    """
    Build a Constrained Log Expected Improvement acquisition function.

    We minimise time (negate for EI), subject to force_error <= target_error.
    The constraint is expressed as:
        c(x) <= 0   where  c(x) = force_error(x) - target_error

    BoTorch's LogConstrainedExpectedImprovement expects the constraint
    as a dict {model_index: (lower, upper)} on the *standardised* output.
    We pass the constraint GP as a separate model in a ModelList.

    Parameters
    ----------
    objective_model  : GP for -tot_acc_time (negated so we maximise EI)
    constraint_model : GP for force_error
    best_f           : current best (negated) time among feasible points
    target_error     : force error threshold

    Returns
    -------
    LogConstrainedExpectedImprovement acquisition function
    """
    from botorch.models import ModelListGP

    # Wrap both models into a ModelList so BoTorch can index them
    model_list = ModelListGP(objective_model, constraint_model)

    # Constraint: constraint model output <= target_error
    # Because the constraint GP is standardised internally, we pass the
    # raw target and let BoTorch handle the transformation.
    constraints = {1: (None, target_error)}  # model index 1 = constraint GP

    acqf = LogConstrainedExpectedImprovement(
        model=model_list,
        best_f=best_f,
        objective_index=0,
        constraints=constraints,
    )
    return acqf


# ---------------------------------------------------------------------------
# Previous-run loader
# ---------------------------------------------------------------------------


def load_previous_run(
    path_or_df,
    codec: "PPPMParameterCodec",
    target_error: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Load results from a previous BO run and convert them into the tensors
    expected by the surrogate models, ready to replace the analytical
    warm-start.

    The CSV written by ``_bayesian_parameter_selection`` contains at minimum
    the columns: ``rc``, ``alpha``, ``M``, ``cao``, ``time``,
    ``force_error``, ``fidelity``.

    Only rows with ``fidelity == "high"`` are used as seed data.  Analytical
    rows from the previous run are discarded because the analytical model can
    regenerate them cheaply, and mixing two fidelity levels in a single GP
    requires a multi-fidelity model.  Using only real MD measurements gives
    the surrogate the most accurate prior possible before new calls are made.

    Parameters
    ----------
    path_or_df : str, Path, or pandas.DataFrame
        Path to the CSV saved by a previous run, or a DataFrame already
        loaded by the caller.
    codec : PPPMParameterCodec
        Codec for the *current* run.  Parameter values from the previous run
        are re-encoded using this codec so that any change in search-space
        bounds is handled correctly (out-of-range values are clipped to the
        nearest boundary).
    target_error : float
        Feasibility threshold used to populate the ``feasible`` field in the
        returned records list.

    Returns
    -------
    X_prev  : torch.Tensor, shape (n, 4)  — encoded parameter vectors
    Y_time  : torch.Tensor, shape (n, 1)  — negated times (for objective GP)
    Y_error : torch.Tensor, shape (n, 1)  — force errors (for constraint GP)
    records : list of dict                — ready to prepend to the run log

    Raises
    ------
    ValueError
        If the DataFrame is missing required columns.
    RuntimeError
        If no high-fidelity rows are found in the previous run data.
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    required_cols = {"rc", "alpha", "M", "cao", "time", "force_error"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Previous run CSV is missing required columns: {missing}\n" f"Available columns: {list(df.columns)}"
        )

    # Keep only high-fidelity rows if the fidelity column exists.
    # If it doesn't exist (e.g. a hand-crafted CSV), treat all rows as HF.
    if "fidelity" in df.columns:
        hf_df = df[df["fidelity"] == "high"].copy()
    else:
        hf_df = df.copy()

    if hf_df.empty:
        raise RuntimeError(
            "No high-fidelity rows found in the previous run data.\n"
            "The previous run may not have completed any MD evaluations."
        )

    X_list, T_list, E_list, records = [], [], [], []

    for _, row in hf_df.iterrows():
        rc = float(row["rc"])
        alpha = float(row["alpha"])
        M = int(row["M"])
        cao = int(row["cao"])
        time = float(row["time"])
        error = float(row["force_error"])

        # Re-encode using the current codec — handles search-space changes.
        x = codec.encode(rc, alpha, M, cao)

        X_list.append(x)
        T_list.append(-np.log10(max(time, 1e-30)))  # negated log10 for objective GP
        E_list.append(np.log10(max(error, 1e-30)))  # log10 for constraint GP

        records.append(
            {
                "rc": rc,
                "alpha": alpha,
                "M": M,
                "cao": cao,
                "time": time,
                "force_error": error,
                "fidelity": "previous",  # distinct tag so results stay traceable
                "feasible": error <= target_error,
            }
        )

    X_prev = torch.cat(X_list, dim=0)  # (n, 4)
    Y_time = torch.tensor(T_list, dtype=torch.double).unsqueeze(-1)  # (n, 1)  negated log10
    Y_error = torch.tensor(E_list, dtype=torch.double).unsqueeze(-1)  # (n, 1)  log10

    return X_prev, Y_time, Y_error, records


# ---------------------------------------------------------------------------
# Core BO loop
# ---------------------------------------------------------------------------
def run_bo_loop(
    codec: PPPMParameterCodec,
    evaluate_fn,  # callable: (rc, alpha, M, cao) -> (time, error)
    analytical_error_fn,  # callable: (rc, alpha, M, cao) -> float
    analytical_time_fn,  # callable: (rc, alpha, M, cao) -> float
    target_error: float,
    n_warm_rc: int = 6,
    n_warm_alpha: int = 6,
    n_bo_iterations: int = 30,
    n_restarts: int = 10,
    raw_samples: int = 256,
    verbose: bool = True,
    warm_start_from=None,
) -> pd.DataFrame:
    """
    Full multi-fidelity Bayesian Optimization loop for PPPM parameters.

    Phase 1 — Analytical warm-start
        Evaluate cheap analytical models on a grid.  These seed the GP
        surrogates with rough landscape information at zero MD cost.

    Phase 2 — High-fidelity BO
        Iteratively propose candidates via Constrained LogEI, evaluate them
        with actual MD timing calls, and update the surrogates.

    Parameters
    ----------
    codec               : PPPMParameterCodec instance
    evaluate_fn         : high-fidelity evaluator (rc, alpha, M, cao) -> (time, error)
                          This should call the actual Sarkas timing routines.
    analytical_error_fn : cheap force error model
    analytical_time_fn  : cheap timing model
    target_error        : force error constraint threshold
    n_warm_rc           : grid points along r_c for warm-start
    n_warm_alpha        : grid points along alpha for warm-start
    n_bo_iterations     : number of high-fidelity BO evaluations
    n_restarts          : restarts for acquisition optimisation
    raw_samples         : random samples for acquisition initialisation
    verbose             : print progress

    Returns
    -------
    pandas.DataFrame with all evaluated points and their results, including
    a column 'fidelity' ('analytical' or 'high') and 'feasible' (bool).
    """
    if not _BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for Bayesian optimization. " "Install with: pip install botorch gpytorch")

    records = []  # accumulates all evaluated points

    # ------------------------------------------------------------------
    # Phase 1: Warm-start — analytical grid or previous run
    # ------------------------------------------------------------------
    if warm_start_from is not None:
        # --- Previous-run mode ----------------------------------------
        # Load real HF measurements from a prior run.  These seed the GP
        # surrogates with actual data, which is strictly more informative
        # than the analytical approximation.
        if verbose:
            src_label = (
                str(warm_start_from)
                if not isinstance(warm_start_from, pd.DataFrame)
                else f"DataFrame ({len(warm_start_from)} rows)"
            )
            print("\n" + " Phase 1: Loading previous run ".center(60, "="))
            print(f"  Source: {src_label}")

        X_warm, Y_time_warm, Y_error_warm, prev_records = load_previous_run(warm_start_from, codec, target_error)
        records.extend(prev_records)

        if verbose:
            n_feasible_prev = sum(1 for r in prev_records if r["feasible"])
            print(f"  Loaded {len(prev_records)} previous HF points " f"({n_feasible_prev} feasible)")
            best_prev = min(
                (r["time"] for r in prev_records if r["feasible"]),
                default=None,
            )
            if best_prev is not None:
                print(f"  Best feasible time from previous run: {best_prev:.4e} s")

    else:
        # --- Analytical warm-start mode --------------------------------
        if verbose:
            print("\n" + " Phase 1: Analytical warm-start ".center(60, "="))
            print(f"  Grid: {n_warm_rc} x {n_warm_alpha} x " f"{len(codec.M_options)} x {len(codec.cao_options)}")
            n_warm = n_warm_rc * n_warm_alpha * len(codec.M_options) * len(codec.cao_options)
            print(f"  Total analytical evaluations: {n_warm}")

        X_warm, Y_time_warm, Y_error_warm = analytical_warm_start(
            codec, analytical_error_fn, analytical_time_fn, n_warm_rc, n_warm_alpha
        )

        for i in range(X_warm.shape[0]):
            rc, alpha, M, cao = codec.decode(X_warm[i])
            records.append(
                {
                    "rc": rc,
                    "alpha": alpha,
                    "M": M,
                    "cao": cao,
                    "time": float(Y_time_warm[i]),
                    "force_error": float(Y_error_warm[i]),
                    "fidelity": "analytical",
                    "feasible": float(Y_error_warm[i]) <= target_error,
                }
            )

        if verbose:
            n_feasible_warm = sum(1 for r in records if r["feasible"])
            print(f"  Feasible analytical points: {n_feasible_warm} / {len(records)}")

    # ------------------------------------------------------------------
    # Phase 2: High-fidelity BO
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Phase 2: High-fidelity Bayesian Optimization ".center(60, "="))
        print(f"  Budget: {n_bo_iterations} MD evaluations")

    # Initialise surrogates from analytical data.
    # Negate time so the objective GP is maximised (BoTorch maximises EI).
    Y_time_neg = -Y_time_warm  # shape (n, 1)

    obj_model = build_surrogate(X_warm, Y_time_neg, codec.cat_dims)
    con_model = build_surrogate(X_warm, Y_error_warm, codec.cat_dims)

    # Track high-fidelity observations separately for surrogate updates
    X_hf = torch.empty((0, codec.n_dims), dtype=torch.double)
    Y_time_hf = torch.empty((0, 1), dtype=torch.double)
    Y_error_hf = torch.empty((0, 1), dtype=torch.double)

    # Surrogate is trained on log10(error) and log10(time), so the
    # feasibility threshold must also be in log10 space.
    log_target_error = log10(target_error)

    best_feasible_time = float("inf")
    best_feasible_point = None

    bounds = codec.botorch_bounds
    fixed_features = codec.fixed_features_list()

    for iteration in range(n_bo_iterations):
        # ---- Build acquisition function --------------------------------
        # best_f: best negated time among points the GP believes are feasible
        # We approximate feasibility using the constraint GP's posterior mean.
        with torch.no_grad():
            # Evaluate constraint GP mean over all warm + HF inputs
            X_all_so_far = torch.cat([X_warm, X_hf], dim=0) if X_hf.shape[0] > 0 else X_warm
            con_post = con_model.posterior(X_all_so_far)
            con_mean = con_post.mean.squeeze()  # (n,)

            feasible_mask = con_mean <= log_target_error
            if feasible_mask.any():
                obj_mean = obj_model.posterior(X_all_so_far).mean.squeeze()
                best_f = float(obj_mean[feasible_mask].max())
            else:
                best_f = float(obj_model.posterior(X_all_so_far).mean.max())

        acqf = build_constrained_acqf(obj_model, con_model, best_f, log_target_error)

        # ---- Optimise acquisition over mixed space ----------------------
        try:
            candidate, acq_value = optimize_acqf_mixed(
                acq_function=acqf,
                bounds=bounds,
                fixed_features_list=fixed_features,
                q=1,
                num_restarts=n_restarts,
                raw_samples=raw_samples,
            )
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration+1}] Acquisition optimisation failed: {e}")
                print("  Falling back to random candidate.")
            candidate = torch.rand(1, codec.n_dims, dtype=torch.double)
            candidate[0, 2] = float(random.randint(0, len(codec.M_options)))
            candidate[0, 3] = float(random.randint(0, len(codec.cao_options)))
            candidate = candidate * (bounds[1] - bounds[0]) + bounds[0]

        # ---- Decode and evaluate ---------------------------------------
        rc, alpha, M, cao = codec.decode(candidate)
        rc = clip(rc, codec.rc_min, codec.rc_max)
        alpha = clip(alpha, codec.alpha_min, codec.alpha_max)

        try:
            time_val, error_val = evaluate_fn(rc, alpha, M, cao)
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration+1}] MD evaluation failed: {e}. Skipping.")
            continue

        feasible = error_val <= target_error

        if verbose:
            status = "✓ FEASIBLE" if feasible else "✗ infeasible"
            print(
                f"  [{iteration+1:3d}/{n_bo_iterations}] "
                f"rc={rc:.3e}  α={alpha:.4e}  M={M:3d}  cao={cao}  "
                f"err={error_val:.2e}  t={time_val:.4e}s  {status}"
            )

        if feasible and time_val < best_feasible_time:
            best_feasible_time = time_val
            best_feasible_point = {
                "rc": rc,
                "alpha": alpha,
                "M": M,
                "cao": cao,
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
                "time": time_val,
                "force_error": error_val,
                "fidelity": "high",
                "feasible": feasible,
            }
        )

        # ---- Update surrogates — store HF observations in log space ----
        x_new = codec.encode(rc, alpha, M, cao)
        log_time_new = log10(max(time_val, 1e-30))
        log_err_new = log10(max(error_val, 1e-30))
        y_time_new = torch.tensor([[-log_time_new]], dtype=torch.double)  # negated log
        y_error_new = torch.tensor([[log_err_new]], dtype=torch.double)

        X_hf = torch.cat([X_hf, x_new], dim=0)
        Y_time_hf = torch.cat([Y_time_hf, y_time_new], dim=0)
        Y_error_hf = torch.cat([Y_error_hf, y_error_new], dim=0)

        # Rebuild surrogates using ALL data (warm + HF)
        X_all = torch.cat([X_warm, X_hf], dim=0)
        Y_time_all = torch.cat([Y_time_neg, Y_time_hf], dim=0)
        Y_err_all = torch.cat([Y_error_warm, Y_error_hf], dim=0)

        try:
            obj_model = build_surrogate(X_all, Y_time_all, codec.cat_dims)
            con_model = build_surrogate(X_all, Y_err_all, codec.cat_dims)
        except Exception as e:
            if verbose:
                print(f"  [iter {iteration+1}] Surrogate refit failed: {e}. Keeping old model.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if verbose:
        print("\n" + " Optimization Complete ".center(60, "="))
        if best_feasible_point:
            p = best_feasible_point
            print(
                f"  Best feasible configuration:\n"
                f"    rc={p['rc']:.4e}  alpha={p['alpha']:.4e}  "
                f"M={p['M']}  cao={p['cao']}\n"
                f"    force_error={p['force_error']:.4e}  time={p['time']:.4e}s"
            )
        else:
            print("  No feasible configuration found within budget.")
            print("  Consider increasing target_error or n_bo_iterations.")

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# Analytical model helpers (to be wired to Sarkas internals)
# ---------------------------------------------------------------------------


def make_analytical_error_fn(
    a_ws,
    screening_length,
    rescaling_constant,
    box_length,
):
    """
    Returns a closure that computes the analytical PPPM force error
    approximation using the Sarkas functions force_error_analytic_pp
    and force_error_approx_pm from sarkas.utilities.maths.

    Both functions expect dimensionless inputs normalised by a_ws:

        rc_adim    = rc    / a_ws
        alpha_adim = alpha * a_ws
        h_adim     = (box_length / M) / a_ws
        kappa      = a_ws / screening_length

    This matches the convention in force_error_approx_pppm:
        alpha = potential.pppm_alpha_ewald * potential.a_ws
        ha    = potential.pppm_h_array[0]  / potential.a_ws
        rc    = potential.rc               / potential.a_ws
        kappa = potential.a_ws             / potential.screening_length

    The total force error is combined in quadrature:
        Delta_F = sqrt(Delta_F_PP^2 + Delta_F_PM^2)

    Parameters
    ----------
    potential_type     : str
        Sarkas potential type, e.g. "yukawa", "coulomb".
    a_ws               : float
        Wigner-Seitz radius (same physical units as rc and box_length).
    screening_length   : float
        Screening length lambda (same physical units as a_ws).
    rescaling_constant : float
        QFactor / (N * e^2/(4 pi eps0)) * sqrt(3/(4 pi)) as computed
        inside force_error_approx_pppm.
    box_length         : float
        Minimum box side length (same physical units as a_ws).

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
            "sarkas.utilities.maths not found. " "Falling back to built-in force error approximations.",
            ImportWarning,
            stacklevel=2,
        )

    # kappa = a_ws / lambda (dimensionless inverse screening length).
    # For Coulomb / QSP there is no screening so kappa = 0.
    kappa = a_ws / screening_length

    def _fn(rc, alpha, M, cao):
        # Dimensionless conversions to match Sarkas internal convention.
        rc_adim = rc / a_ws
        alpha_adim = alpha * a_ws
        h_adim = (box_length / M) / a_ws

        if _sarkas_available:
            # PP short-range error.
            # Signature: (potential_type, cutoff_length, screening_parameter,
            #              alpha_ewald, rescaling_const)
            pp_err = force_error_analytic_pp(
                rc_adim,
                kappa,
                alpha_adim,
                rescaling_constant,
            )

            # PM long-range error.
            # Signature: (kappa, p, h, alpha, rescaling_const)
            pm_err = force_error_approx_pm(
                kappa,
                cao,
                h_adim,
                alpha_adim,
                rescaling_constant,
            )
        else:
            # Fallback approximations (no Sarkas dependency).
            pp_err = (
                2.0
                * exp(-((0.5 * kappa / (alpha_adim + 1e-30)) ** 2))
                * exp(-((alpha_adim * rc_adim) ** 2))
                / sqrt(rc_adim + 1e-30)
                * rescaling_constant
            )
            pm_err = rescaling_constant * (alpha_adim * h_adim) ** (2 * cao)

        return float(sqrt(pp_err**2 + pm_err**2))

    return _fn


def make_analytical_time_fn(
    N: int,
    box_length: float,
):
    """
    Returns a closure that estimates PPPM cost per step analytically.

    The asymptotic complexity of PPPM is:
        T_PP  ~ C_pp  * N * (r_c / L)^3
        T_PM  ~ C_pm  * M^3 * log(M^3)  +  C_cao * cao^3 * N

    where C_pp, C_pm, C_cao are machine-dependent constants that we
    set to 1 (the returned value is in *relative* units, sufficient
    for BO ordering purposes).

    For the warm-start, absolute accuracy is not required — we only
    need the correct ordering of configurations.

    Parameters
    ----------
    N          : int, number of particles
    box_length : float

    Returns
    -------
    callable(rc, alpha, M, cao) -> float  (relative cost, arbitrary units)
    """

    def _fn(rc: float, alpha: float, M: int, cao: int) -> float:
        # Real-space (PP) cost: particles within a sphere of radius r_c
        occupancy = (4.0 / 3.0) * pi * (rc / box_length) ** 3
        t_pp = N * occupancy

        # Mesh (PM) cost: FFT on M^3 grid + charge assignment
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

    Or call the methods directly on a PreProcess instance by passing
    the instance as the first argument (duck typing).

    The mixin expects the following attributes on `self`, all of which
    are already present on a standard Sarkas PreProcess instance:

        self.potential.rc
        self.potential.pppm_mesh          (ndarray, shape (3,))
        self.potential.pppm_alpha_ewald
        self.potential.pppm_cao           (ndarray, shape (3,))
        self.potential.pppm_aliases
        self.potential.pppm_h_array
        self.potential.pppm_pp_err
        self.potential.pppm_pm_err
        self.potential.force_error
        self.potential.force_error_approx
        self.potential.box_lengths        (ndarray, shape (3,))
        self.potential.type               (str)
        self.potential.screening_length   (float)
        self.potential.total_num_ptcls    (int)
        self.potential.a_ws               (float)
        self.potential.pbox_volume        (float)
        self.potential.pot_update_params  (callable)
        self.potential.update_pm          (callable)
        self.potential.update_linked_list (callable)
        self.potential.calculate_force_error (callable)
        self.parameters.verbose           (bool)
        self.particles                    (Particles instance)
        self.timer                        (Timer instance)
        self.io                           (IO instance)
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
        mesh_options: Optional[List[int]] = None,
        n_warm_rc: int = 10,
        n_warm_alpha: int = 10,
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
            Candidate numbers of PP cells (rc = box_length / cell).
            The BO search range for r_c is derived as:
                rc_min = box_length / max(pp_cells)
                rc_max = box_length / min(pp_cells)
            If None, physical defaults are used (rc in [2*a_ws, 0.49*L]).
        pm_meshes : list of int, optional
            Candidate mesh sizes M for the PM step.  Passed directly as
            the M_options for the codec.  Defaults to FFT_FRIENDLY_MESHES.
        pm_caos : list of int, optional
            Candidate charge assignment orders (B-spline orders).
            Defaults to [1, 2, 3, 4, 5, 6, 7].
        n_warm_rc : int
            Grid resolution for r_c in the analytical warm-start.
        n_warm_alpha : int
            Grid resolution for alpha in the analytical warm-start.
        n_bo_iterations : int
            Number of high-fidelity (MD timing) BO evaluations.
        n_restarts : int
            Number of restarts for acquisition function optimisation.
        raw_samples : int
            Number of raw random samples for acquisition initialisation.
        save_csv : bool
            Whether to save results to CSV.

        Returns
        -------
        (results_df, best_point)
            results_df : DataFrame with all evaluated configurations
            best_point : dict with optimal parameters, or None if no
                         feasible point was found within budget.
        """
        if not _BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required. Install with: pip install botorch gpytorch")

        msg = "\n\n{:=^70} \n".format(" PPPM Bayesian Optimization ")
        self.io.write_to_logger(msg)

        # Save original parameters so we can restore them after the study
        self._save_original_pppm_params()

        box_length = self.potential.box_lengths.min()
        a_ws = self.potential.a_ws
        N = self.potential.total_num_ptcls

        QFactor = self.potential.QFactor / (self.potential.matrix[0, 0, 0] * self.potential.total_num_ptcls)
        rescaling_constant = sqrt(3.0 / (4.0 * pi)) * QFactor

        # --- Resolve mesh and CAO option lists --------------------------
        mesh_options = list(pm_meshes) if pm_meshes is not None else FFT_FRIENDLY_MESHES
        cao_list = list(pm_caos) if pm_caos is not None else CAO_OPTIONS

        # --- Derive rc bounds from pp_cells if provided -----------------
        # pp_cells is the number of cells along one dimension, so
        # rc = box_length / pp_cells.  Larger cell count => smaller rc.
        rc_min_override = None
        rc_max_override = None
        if pp_cells is not None and len(pp_cells) > 0:
            pp_cells_arr = asarray(pp_cells, dtype=float)
            # box_length here is already dimensionless (L / a_ws)
            rc_min_override = box_length / float(pp_cells_arr.max())
            rc_max_override = box_length / float(pp_cells_arr.min())

        # compute_physical_bounds receives dimensionless a_ws=1 and
        # dimensionless box_length = L/a_ws, giving rc bounds in units of a_ws.
        bounds = compute_physical_bounds(
            a_ws,
            box_length,
            mesh_options,
            cao_options=cao_list,
            rc_min_override=rc_min_override,
            rc_max_override=rc_max_override,
        )
        if self.parameters.verbose:
            print(f"\nPhysical bounds:")
            print(f"  r_c   : [{bounds['rc_min']:.4e}, {bounds['rc_max']:.4e}]")
            print(f"  alpha : [{bounds['alpha_min']:.4e}, {bounds['alpha_max']:.4e}]")
            print(f"  M     : {bounds['M_options']}")
            print(f"  CAO   : {bounds['cao_options']}")

        codec = PPPMParameterCodec(bounds)

        # Build analytical model closures.
        # Note: a_ws is passed as the second argument so that the function
        # can convert rc and alpha to the dimensionless units expected by
        # the Sarkas force error routines (rc/a_ws, alpha*a_ws, etc.).
        analytical_error_fn = make_analytical_error_fn(
            a_ws,
            self.potential.screening_length,
            rescaling_constant,
            box_length,
        )
        analytical_time_fn = make_analytical_time_fn(N, box_length)

        # Build high-fidelity evaluator (actual timing calls)
        evaluate_fn = self._make_hf_evaluator(rescaling_constant)

        # Run the BO loop
        results_df = run_bo_loop(
            codec=codec,
            evaluate_fn=evaluate_fn,
            analytical_error_fn=analytical_error_fn,
            analytical_time_fn=analytical_time_fn,
            target_error=target_error,
            n_warm_rc=n_warm_rc,
            n_warm_alpha=n_warm_alpha,
            n_bo_iterations=n_bo_iterations,
            n_restarts=n_restarts,
            raw_samples=raw_samples,
            verbose=self.parameters.verbose,
            warm_start_from=warm_start_from,
        )

        # Find the best feasible high-fidelity point
        best_point = self._extract_best_point(results_df, target_error)

        # Save results
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

        # Log optimal configuration
        if best_point:
            msg = (
                f"\nOPTIMAL PPPM CONFIGURATION (BAYESIAN):\n"
                f"  Mesh: {best_point['M']} | CAO: {best_point['cao']} "
                f"| rc: {best_point['rc']:.4e}\n"
                f"  Ewald alpha: {best_point['alpha']:.4e} "
                f"| Force Error: {best_point['force_error']:.4e}\n"
                f"  Total Time: {best_point['time']:.4e} s"
            )
        else:
            msg = "\nNo feasible configuration found.\n" "Consider increasing target_error or n_bo_iterations."
        self.io.write_to_logger(msg)
        if self.parameters.verbose:
            print(msg)

        # Restore original parameters
        self._restore_original_pppm_params()

        return results_df, best_point

    # ------------------------------------------------------------------
    # High-fidelity evaluator
    # ------------------------------------------------------------------

    def _make_hf_evaluator(self, rescaling_constant: float):
        """
        Returns a closure that sets PPPM parameters in the Sarkas potential,
        measures actual PP and PM timing via linked-list and FFT routines,
        and returns (total_time, force_error).

        The closure captures `self` (the PreProcess instance) so it has
        direct access to the potential and particle objects.

        Parameters
        ----------
        rescaling_constant : float

        Returns
        -------
        callable(rc, alpha, M, cao) -> (float, float)
        """
        # a_ws in SI — needed to convert dimensionless BO parameters back
        # to the SI units Sarkas stores in potential attributes.
        a_ws = self.potential.a_ws

        # Track the last (M, cao, alpha_adim) that was fully set up so we can
        # skip redundant reinitialisation when only rc changes between calls.
        _last_M: int = -1
        _last_cao: int = -1
        _last_alpha: float = float("nan")
        _alpha_tol: float = 1e-6  # relative tolerance for alpha comparison

        def _evaluate(rc: float, alpha: float, M: int, cao: int) -> Tuple[float, float]:
            """
            Set PPPM parameters (converting from dimensionless BO units to
            SI), run Sarkas timing routines, return (total_time_s, force_error).

            Parameter changes require different levels of reinitialisation:

            M changed
                The PM charge mesh is a completely different array size
                (M^3 elements).  A full mesh setup must be done — this
                allocates a new mesh, recomputes the Green's function /
                influence function, and rebuilds all PM data structures.

            cao changed (M unchanged)
                The B-spline order changes the charge assignment kernel but
                the mesh size stays the same.  Green's function must be
                recomputed; mesh array can be reused.

            alpha changed (M, cao unchanged)
                Only the Green's function / influence function needs updating;
                the mesh and assignment kernel are unchanged.

            rc changed only
                No PM reinitialisation needed.  The PP linked-list is rebuilt
                every timing call anyway.
            """
            nonlocal _last_M, _last_cao, _last_alpha

            # --- Determine what changed since the last call --------------
            M_changed = M != _last_M
            cao_changed = cao != _last_cao
            alpha_rel_diff = abs(alpha - _last_alpha) / (abs(_last_alpha) + 1e-30)
            alpha_changed = isnan(_last_alpha) or alpha_rel_diff > _alpha_tol

            # --- Write new parameter values to the Sarkas potential -----
            self.potential.rc = rc
            self.potential.pppm_alpha_ewald = alpha

            if M_changed or cao_changed or alpha_changed:
                # Mesh size or assignment order changed.  Update the discrete
                # attributes and then run the full PM setup so that:
                #   1. The mesh array is reallocated at the new size (if M changed)
                #   2. The influence function / optimised Green's function is
                #      recomputed for the new (M, cao, alpha) triple
                #   3. The charge assignment weights are rebuilt for the new cao
                self.potential.pppm_mesh = full(3, M, dtype=int)
                self.potential.pppm_cao = full(3, cao, dtype=int)
                self.potential.pppm_alpha_ewald = alpha

                # Full PM setup: reallocates mesh, recalculates influence fn
                self.potential.pppm_setup()

                # Update potential parameters since alpha has changed
                self.potential.pot_update_params(self.potential, self.species)

            # else: only rc changed — no PM reinitialisation needed.

            # --- Measure PM time ----------------------------------------
            pm_acc_time = 0.0
            n_trials = 3
            for _ in range(n_trials):
                self.timer.start()
                self.potential.update_pm(self.particles)
                pm_acc_time += self.timer.stop() / n_trials
            pm_acc_time *= 1.0e-9  # ns -> s

            # --- Measure PP time ----------------------------------------
            pp_acc_time = 0.0
            for _ in range(n_trials):
                self.timer.start()
                self.potential.update_linked_list(self.particles)
                pp_acc_time += self.timer.stop() / n_trials
            pp_acc_time *= 1.0e-9

            total_time = pp_acc_time + pm_acc_time

            # --- Compute force error ------------------------------------
            self.potential.calculate_force_error()
            force_error = float(self.potential.force_error)

            return total_time, force_error

        return _evaluate

    # ------------------------------------------------------------------
    # Parameter save / restore helpers
    # ------------------------------------------------------------------

    def _save_original_pppm_params(self):
        """Save current PPPM parameters so they can be restored later."""
        self._bo_saved_rc = float(self.potential.rc)
        self._bo_saved_mesh = self.potential.pppm_mesh.copy()
        self._bo_saved_alpha = float(self.potential.pppm_alpha_ewald)
        self._bo_saved_cao = self.potential.pppm_cao.copy()
        self._bo_saved_aliases = self.potential.pppm_aliases.copy()

    def _restore_original_pppm_params(self):
        """Restore PPPM parameters saved by _save_original_pppm_params."""
        self.potential.rc = self._bo_saved_rc
        self.potential.pppm_mesh = self._bo_saved_mesh.copy()
        self.potential.pppm_alpha_ewald = self._bo_saved_alpha
        self.potential.pppm_cao = self._bo_saved_cao.copy()
        self.potential.pppm_aliases = self._bo_saved_aliases.copy()
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
        From the results DataFrame, return the high-fidelity feasible point
        with the smallest total time.

        Falls back to the analytical feasible point with smallest time if no
        high-fidelity feasible point exists.
        """
        # Prefer high-fidelity feasible points
        hf_feasible = results_df[(results_df["fidelity"] == "high") & (results_df["feasible"])]
        if not hf_feasible.empty:
            idx = hf_feasible["time"].idxmin()
            row = hf_feasible.loc[idx]
            return {
                "rc": float(row["rc"]),
                "alpha": float(row["alpha"]),
                "M": int(row["M"]),
                "cao": int(row["cao"]),
                "time": float(row["time"]),
                "force_error": float(row["force_error"]),
            }

        # Fallback: analytical feasible
        an_feasible = results_df[(results_df["fidelity"] == "analytical") & (results_df["feasible"])]
        if not an_feasible.empty:
            idx = an_feasible["time"].idxmin()
            row = an_feasible.loc[idx]
            return {
                "rc": float(row["rc"]),
                "alpha": float(row["alpha"]),
                "M": int(row["M"]),
                "cao": int(row["cao"]),
                "time": float(row["time"]),
                "force_error": float(row["force_error"]),
            }

        return None

    @staticmethod
    def compute_pareto_front(
        results_df: pd.DataFrame,
        fidelity: str = "high",
    ) -> pd.DataFrame:
        """
        Compute the Pareto front in (force_error, time) space from BO results.

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
        if fidelity != "all":
            df = results_df[results_df["fidelity"] == fidelity].copy()
        else:
            df = results_df.copy()

        if df.empty:
            return df

        pareto_mask = full(len(df), True, dtype=bool)
        times = df["time"].values
        errors = df["force_error"].values

        for i in range(len(df)):
            for j in range(len(df)):
                if i == j:
                    continue
                # i is dominated by j if j is strictly better in one metric
                # and at least as good in the other
                if times[j] <= times[i] and errors[j] < errors[i]:
                    pareto_mask[i] = False
                    break
                if times[j] < times[i] and errors[j] <= errors[i]:
                    pareto_mask[i] = False
                    break

        pareto_df = df[pareto_mask].sort_values("force_error").reset_index(drop=True)
        return pareto_df

    # ------------------------------------------------------------------
    # Convergence diagnostics
    # ------------------------------------------------------------------

    @staticmethod
    def bo_convergence_summary(results_df: pd.DataFrame, target_error: float) -> Dict:
        """
        Compute convergence statistics from a BO run.

        Returns a dict with:
            n_hf_total        : total high-fidelity evaluations
            n_hf_feasible     : feasible HF evaluations
            first_feasible_iter : iteration at which first feasible HF point was found
            best_time         : time at optimal feasible point
            best_error        : error at optimal feasible point
            improvement_curve : list of (iteration, best_time_so_far)
        """
        hf = results_df[results_df["fidelity"] == "high"].reset_index(drop=True)

        n_total = len(hf)
        n_feasible = int(hf["feasible"].sum())

        first_idx = None
        best_time_so_far = float("inf")
        best_error = None
        improvement_curve = []

        for i, row in hf.iterrows():
            if row["feasible"] and row["time"] < best_time_so_far:
                best_time_so_far = row["time"]
                best_error = row["force_error"]
                if first_idx is None:
                    first_idx = i
            improvement_curve.append((i, best_time_so_far))

        return {
            "n_hf_total": n_total,
            "n_hf_feasible": n_feasible,
            "first_feasible_iter": first_idx,
            "best_time": best_time_so_far if best_time_so_far < float("inf") else None,
            "best_error": best_error,
            "improvement_curve": improvement_curve,
        }

    # ------------------------------------------------------------------
    # Plotting helpers
    # ------------------------------------------------------------------

    def plot_bo_results(
        self,
        results_df: pd.DataFrame,
        target_error: float,
        save_dir: Optional[str] = None,
    ):
        """
        Generate diagnostic plots for the BO run:

        1. Pareto front (force_error vs time) comparing analytical warm-start
           and high-fidelity BO evaluations.
        2. BO convergence curve (best feasible time vs iteration).
        3. Parameter distributions of high-fidelity evaluations, coloured
           by feasibility.

        Parameters
        ----------
        results_df   : DataFrame from _bayesian_parameter_selection
        target_error : float
        save_dir     : directory to save figures. If None, uses
                       self.pppm_plots_dir if available.
        """
        try:
            import matplotlib.cm as cm
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available — skipping plots.")
            return

        if save_dir is None:
            save_dir = getattr(self, "pppm_plots_dir", ".")

        # ---- Figure 1: Pareto front ------------------------------------
        fig, ax = plt.subplots(figsize=(7, 5))

        an = results_df[results_df["fidelity"] == "analytical"]
        hf = results_df[results_df["fidelity"] == "high"]

        ax.scatter(an["force_error"], an["time"] * 1e3, c="lightblue", alpha=0.3, s=10, label="Analytical warm-start")

        hf_feas = hf[hf["feasible"]]
        hf_infeas = hf[~hf["feasible"]]
        ax.scatter(
            hf_infeas["force_error"],
            hf_infeas["time"] * 1e3,
            c="salmon",
            alpha=0.7,
            s=40,
            marker="x",
            label="HF infeasible",
        )
        ax.scatter(
            hf_feas["force_error"], hf_feas["time"] * 1e3, c="green", alpha=0.9, s=60, marker="o", label="HF feasible"
        )

        # Pareto front
        pareto = self.compute_pareto_front(results_df, fidelity="high")
        if not pareto.empty:
            ax.plot(pareto["force_error"], pareto["time"] * 1e3, "k--", linewidth=1.5, label="Pareto front (HF)")

        ax.axvline(target_error, color="red", linestyle=":", linewidth=1.5, label=f"Target error = {target_error:.0e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Force error", fontsize=12)
        ax.set_ylabel("Time per step (ms)", fontsize=12)
        ax.set_title("PPPM Pareto Front — BO Results", fontsize=13)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(join(save_dir, "BO_pareto_front.png"), dpi=150)
        plt.close(fig)

        # ---- Figure 2: Convergence curve ------------------------------
        summary = self.bo_convergence_summary(results_df, target_error)
        iters, best_times = zip(*summary["improvement_curve"]) if summary["improvement_curve"] else ([], [])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.step(iters, [t * 1e3 for t in best_times], where="post", color="steelblue", linewidth=2)
        ax.set_xlabel("BO iteration (high-fidelity)", fontsize=12)
        ax.set_ylabel("Best feasible time (ms)", fontsize=12)
        ax.set_title("BO Convergence", fontsize=13)
        if summary["first_feasible_iter"] is not None:
            ax.axvline(
                summary["first_feasible_iter"],
                color="green",
                linestyle="--",
                label=f"First feasible (iter {summary['first_feasible_iter']})",
            )
            ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(join(save_dir, "BO_convergence.png"), dpi=150)
        plt.close(fig)

        # ---- Figure 3: Parameter scatter (rc vs alpha, coloured by M) -
        if not hf.empty:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            sc = axes[0].scatter(
                hf["rc"],
                hf["alpha"],
                c=hf["M"],
                cmap="viridis",
                s=60,
                alpha=0.8,
                edgecolors=["green" if f else "red" for f in hf["feasible"]],
                linewidths=1.5,
            )
            plt.colorbar(sc, ax=axes[0], label="Mesh size M")
            axes[0].set_xlabel("r_c", fontsize=12)
            axes[0].set_ylabel("alpha (Ewald)", fontsize=12)
            axes[0].set_title("HF evaluations (colour = M)", fontsize=12)

            sc2 = axes[1].scatter(
                hf["M"],
                hf["cao"],
                c=hf["time"] * 1e3,
                cmap="plasma",
                s=80,
                alpha=0.8,
            )
            plt.colorbar(sc2, ax=axes[1], label="Time (ms)")
            axes[1].set_xlabel("Mesh size M", fontsize=12)
            axes[1].set_ylabel("CAO (B-spline order)", fontsize=12)
            axes[1].set_title("HF evaluations (colour = time)", fontsize=12)

            fig.tight_layout()
            fig.savefig(join(save_dir, "BO_parameter_scatter.png"), dpi=150)
            plt.close(fig)

        if self.parameters.verbose:
            print(f"\nPlots saved to: {save_dir}")


# ---------------------------------------------------------------------------
# Integration patch for PreProcess.timing_study_calculation
# ---------------------------------------------------------------------------
# Add this to the body of timing_study_calculation in PreProcess:
#
#   elif method.lower() == "bayesian":
#       return self._bayesian_parameter_selection(
#           target_error=target_error,
#           **kwargs
#       )
#
# The snippet below shows the full modified timing_study_calculation
# signature for reference.

TIMING_STUDY_PATCH = '''
def timing_study_calculation(
        self, target_error=1e-5, pp_cells=None, pm_meshes=None,
        pm_caos=None, method="brute_force", **kwargs
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
    Demonstrate the BO loop with synthetic analytical models only.
    No Sarkas or actual MD is required.

    Run with:  python pppm_bayesian_optimization.py
    """

    print("=" * 60)
    print(" PPPM Bayesian Optimization — Standalone Demo ")
    print("=" * 60)

    # Synthetic system parameters
    N = 4096
    a_ws = 1.0
    box_length = (4.0 / 3.0 * pi * N) ** (1.0 / 3.0) * a_ws
    rescaling = sqrt(N) * a_ws**2 / sqrt(box_length**3)
    screening = 0.5 * box_length  # arbitrary screening length

    target_error = 1e-4

    # Bounds and codec
    bounds = compute_physical_bounds(a_ws, box_length, FFT_FRIENDLY_MESHES[:10])
    codec = PPPMParameterCodec(bounds)

    print(f"\nSystem: N={N}, L={box_length:.2e}, a_ws={a_ws}")
    print(f"Target force error: {target_error:.0e}")
    print(f"r_c   bounds: [{bounds['rc_min']:.3e}, {bounds['rc_max']:.3e}]")
    print(f"alpha bounds: [{bounds['alpha_min']:.4e}, {bounds['alpha_max']:.4e}]")
    print(f"M     options: {bounds['M_options']}")

    # Analytical models
    analytical_error_fn = make_analytical_error_fn("yukawa", screening, rescaling, box_length)
    analytical_time_fn = make_analytical_time_fn(N, box_length)

    # Synthetic high-fidelity evaluator (adds noise to analytical)
    rng = random.default_rng(42)

    def synthetic_hf(rc, alpha, M, cao):
        err = analytical_error_fn(rc, alpha, M, cao)
        time = analytical_time_fn(rc, alpha, M, cao)
        # Add ~10% noise to simulate real timing variance
        time *= 1.0 + 0.10 * rng.standard_normal()
        err *= 1.0 + 0.05 * abs(rng.standard_normal())
        return abs(time), abs(err)

    # Run BO
    results_df = run_bo_loop(
        codec=codec,
        evaluate_fn=synthetic_hf,
        analytical_error_fn=analytical_error_fn,
        analytical_time_fn=analytical_time_fn,
        target_error=target_error,
        n_warm_rc=4,
        n_warm_alpha=4,
        n_bo_iterations=20,
        n_restarts=5,
        raw_samples=64,
        verbose=True,
    )

    # Summary
    best = BayesianPPPMOptimizer._extract_best_point(results_df, target_error)
    summary = BayesianPPPMOptimizer.bo_convergence_summary(results_df, target_error)

    print("\n" + "=" * 60)
    print("Convergence summary:")
    print(f"  HF evaluations: {summary['n_hf_total']}")
    print(f"  Feasible HF   : {summary['n_hf_feasible']}")
    print(f"  First feasible: iteration {summary['first_feasible_iter']}")
    if best:
        print(f"\nBest configuration:")
        for k, v in best.items():
            print(f"  {k}: {v}")

    results_df.to_csv("bo_demo_results.csv", index=False)
    print("\nResults saved to bo_demo_results.csv")


if __name__ == "__main__":
    _demo_standalone()
