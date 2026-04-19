
from __future__ import annotations

"""
Experiment III active-sensing planner (paper experiment version).

Main updates relative to the previous version:
1) Earlier triggering for hard image-plane boundary approach.
2) Adaptive total delta-v budget instead of a fixed 0.08 m/s for all cases.
3) Explicit late / unobservable / budget-limited hard-case labeling when no
   feasible schedule is found, so these cases are not mixed with ordinary planner failures.
"""

import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Stable imports regardless of how the script is launched
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name in {"pipeline", "sim"} else THIS_FILE.parent
for extra in [PROJECT_ROOT, PROJECT_ROOT / "pipeline", PROJECT_ROOT / "sim"]:
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

from scipy.linalg import expm
from imm_ukf_estimator import imm_run, nearest_spd, safe_inv_spd  # noqa: E402

# -----------------------------------------------------------------------------
# User configuration
# -----------------------------------------------------------------------------
EXPERIMENT_MODE = "exp3"
INPUT_CSV = Path(r"D:\robot_view\ore\renders_float\safe\sample_00150_float\merged_with_pseudo_los.csv")
OUT_DIR = Path(r"D:\robot_view\ore\mc_consistency_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Base / adaptive total DV budget
DV_MAG_BASE = 0.08
DV_MAG_HARD = 0.12
DV_MAG_VERY_HARD = 0.16
DV_MAG_EXTREME = 0.24
DV_MAG_ULTRA = 0.28
DV_MAG_SUPER = 0.32
DV_MAG_MEGA = 0.40
DV_MAG_HYPER = 0.48
BUDGET_RETRY_SEQUENCE = [0.20, 0.24, 0.28, 0.32, 0.36, 0.40, 0.48]

PLAN_DT = 5.0
PLAN_HORIZON_SEC = 120.0
ROLLOUT_TIMING_SEMANTICS = "step_begin_post_impulse_sample"
SAFE_RATIO = 0.80
IMG_W = 1024
IMG_H = 1024
FOV_X_DEG = 15.0
MU = 3.986004418e14
CODE_VERSION = "planner_v10_nominal_hardening_20260409"

# Earlier trigger than the previous version
MIN_TRIGGER_SEC = 20.0
MIN_TRIGGER_FRAME = 4
PREBOUNDARY_BAND_PX = 240.0
TRACE_POS_THR = 2.0e2
SIGMA_T_THR = 3.0

# Hard-case labeling
HARD_MARGIN_THR_PX = -120.0
VERY_HARD_MARGIN_THR_PX = -300.0
EXTREME_MARGIN_THR_PX = -600.0

LATE_TRIGGER_SEC = 600.0
LATE_TRIGGER_FRAME = 120

SIGMA_ENVELOPE_SCALE = 2.0
OUTER_SEARCH_PASSES = 5
SEED_EARLY_STEPS = 10
SEARCH_ACTIVE_STEPS = 20
MAG_FRACTIONS = np.array([0.0, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00], dtype=float)
FD_EPS_DIR = 1.0e-3

# Trigger preemption: allow the planner to enter earlier when a short-horizon
# boundary approach is already visible, instead of waiting for a hard full-horizon breach.
SHORT_TRIGGER_HORIZON_SEC = 60.0
SHORT_PREBOUNDARY_BAND_PX = 360.0
SHORT_URGENT_MARGIN_PX = 180.0
SHORT_BREACH_URGENT_SEC = 40.0
SHORT_TRIGGER_PERSIST_FRAMES = 2
SHORT_TRIGGER_FORCE_RANGE_M = 1800.0
SHORT_TRIGGER_MAX_RANGE_M = 3000.0
SHORT_TRIGGER_MIN_VISIBLE_COUNT = 8
SHORT_TRIGGER_CUR_SAFE_MAX_PX = 220.0
SHORT_TRIGGER_EARLY_CUR_SAFE_MAX_PX = 460.0
TRIGGER_SAFE_DIST_RATE_THR_PXPS = -4.0
TRIGGER_SAFE_DIST_FAST_RATE_THR_PXPS = -7.0
SHORT_TRIGGER_EARLY_TIME_TO_BREACH_SEC = 30.0
TRIGGER_TASK_CONSISTENCY_ENABLE = 1
TRIGGER_LOW_RISK_PC_MAX = 1.0e-6
TRIGGER_LOW_RISK_MIN_FUTURE_DISTANCE_M = 1200.0
TRIGGER_LOW_RISK_REQUIRED_WORST_RISK_RANK = 1
TRIGGER_LOW_RISK_REQUIRED_SAFE_DIST_RATE_PXPS = -1.0
TRIGGER_LOW_RISK_EXTRA_PERSIST_FRAMES = 1

# Unified trigger score: one score model, four states.
TRIGGER_SCORE_SAFE_MARGIN_SCALE_PX = 360.0
TRIGGER_SCORE_SAFE_MARGIN_RATE_SCALE_PXPS = 6.0
TRIGGER_SCORE_WARNING_TIME_SCALE_SEC = 80.0
TRIGGER_SCORE_BREACH_TIME_SCALE_SEC = 60.0
TRIGGER_SCORE_EST_RANGE_SCALE_M = 3000.0
TRIGGER_SCORE_FUTURE_DISTANCE_SCALE_M = 1400.0
TRIGGER_SCORE_RISK_RANK_SCALE = 2.0
TRIGGER_SCORE_W_SAFE_MARGIN = 0.24
TRIGGER_SCORE_W_SAFE_MARGIN_RATE = 0.16
TRIGGER_SCORE_W_WARNING_TIME = 0.16
TRIGGER_SCORE_W_BREACH_TIME = 0.16
TRIGGER_SCORE_W_EST_RANGE = 0.10
TRIGGER_SCORE_W_FUTURE_DISTANCE = 0.12
TRIGGER_SCORE_W_RISK_RANK = 0.06
TRIGGER_STATE_TRIGGER_SCORE_THR = 0.58
TRIGGER_STATE_WATCH_SCORE_THR = 0.32
TRIGGER_STATE_SKIP_LOW_RISK_SCORE_MAX = 0.22

TRIGGER_DIAGNOSTIC_KEYS = [
    "safe_dist_px",
    "safe_dist_rate_pxps",
    "short_time_to_warning_sec",
    "short_time_to_breach_sec",
    "trigger_score",
    "trigger_state",
    "risk_score",
    "warning_score",
    "risk_integral",
    "warning_integral",
    "short_pred_min_safe_margin_px",
    "pred_min_safe_margin_px",
    "est_range_m",
]

LATE_TRIGGER_SKIP_ENABLE = 1
UNOBSERVABLE_TRIGGER_SKIP_ENABLE = 1

# Search strengthening: multi-step seed bank + multi-start local refinement.
MULTISTEP_SEED_STEPS = 6
MULTISTEP_SEED_TOPK = 12
RANDOM_MULTISTART_COUNT = 36
RANDOM_NONZERO_STEPS_MAX = 4
DOMINANT_SPLIT_PASSES = 5
DOMINANT_SPLIT_MAX_LOOKAHEAD = 6
RANDOM_SEED = 7

# Parallel acceleration controls.
PARALLEL_STRATEGY_SOLVE_ENABLE = 1
PARALLEL_STRATEGY_MAX_WORKERS = 2
PARALLEL_SEED_SOLVE_ENABLE = 1
PARALLEL_SEED_MAX_WORKERS = 4
PARALLEL_SEED_MIN_TASKS = 4
ALLOW_NESTED_PROCESS_PARALLEL = 1

# Best-effort execution control: do not let the planner repeatedly choose
# a single dominant impulse if a more distributed nonzero schedule exists.
BEST_EFFORT_MIN_NONZERO_STEPS = 1
BEST_EFFORT_MAX_DOMINANT_RATIO = 0.8
BEST_EFFORT_MIN_SPAN_STEPS = 0
BEST_EFFORT_REPAIR_PASSES = 6

# Unified acceptance policy: all candidates are evaluated by one score model
# with a very small hard-veto set.
ACCEPTANCE_CONTROL_MODE = str(os.environ.get("ORE_ACCEPTANCE_CONTROL_MODE", "unified")).strip().lower()
if ACCEPTANCE_CONTROL_MODE not in {"legacy", "unified"}:
    ACCEPTANCE_CONTROL_MODE = "unified"
ACCEPTANCE_DUAL_TRACK_LOGGING = 1

ACCEPTANCE_MAX_TRACE_RATIO_SAFE = 1.05
ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_SAFE = 1.05
ACCEPTANCE_MAX_TRACE_RATIO_CAUTION = 1.10
ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_CAUTION = 1.08
ACCEPTANCE_MAX_TRACE_RATIO_DANGER = 1.15
ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_DANGER = 1.12
ACCEPTANCE_MIN_INFO_DELTA = -1.0e-9

ACCEPTANCE_GEOM_MIN_MARGIN_GAIN_VS_ZERO_PX = -2.0
ACCEPTANCE_GEOM_MIN_TERMINAL_MARGIN_GAIN_VS_ZERO_PX = -2.0
ACCEPTANCE_GEOM_MAX_NEG_MARGIN_SUM_DELTA_VS_ZERO = 5.0

ACCEPTANCE_HARD_VETO_MAX_VIOLATION_DELTA_VS_ZERO = 0
ACCEPTANCE_HARD_VETO_MIN_MARGIN_PX = 0.0
ACCEPTANCE_HARD_VETO_REQUIRE_STRUCTURALLY_SANE = 1
ACCEPTANCE_HARD_VETO_FUTURE_DIST_NON_REGRESSION_TOL_M = 0.0
ACCEPTANCE_HARD_VETO_FUTURE_DIST_GATE_FOR_CAUTION_DANGER_ONLY = 1

ACCEPTANCE_SCORE_ACCEPTED_THR = 0.15
ACCEPTANCE_SCORE_BORDERLINE_THR = -0.05
ACCEPTANCE_SCORE_MARGIN_SCALE_PX = 60.0
ACCEPTANCE_SCORE_TERMINAL_MARGIN_SCALE_PX = 60.0
ACCEPTANCE_SCORE_INFO_DELTA_SCALE = 8.0
ACCEPTANCE_SCORE_FUTURE_DIST_GAIN_SCALE_M = 300.0
ACCEPTANCE_SCORE_VIOLATION_DELTA_SCALE = 2.0
ACCEPTANCE_SCORE_NEG_MARGIN_SUM_DELTA_SCALE_PX = 20.0
ACCEPTANCE_SCORE_TRACE_PENALTY_SCALE = 0.20
ACCEPTANCE_SCORE_TERMINAL_TRACE_PENALTY_SCALE = 0.20
ACCEPTANCE_SCORE_WEIGHT_MARGIN = 0.24
ACCEPTANCE_SCORE_WEIGHT_TERMINAL_MARGIN = 0.16
ACCEPTANCE_SCORE_WEIGHT_INFO_DELTA = 0.13
ACCEPTANCE_SCORE_WEIGHT_FUTURE_DISTANCE = 0.14
ACCEPTANCE_SCORE_WEIGHT_STRUCTURAL = 0.08
ACCEPTANCE_SCORE_WEIGHT_VIOLATION_DELTA = 0.08
ACCEPTANCE_SCORE_WEIGHT_NEG_MARGIN_DELTA = 0.06
ACCEPTANCE_SCORE_WEIGHT_TRACE_PENALTY = 0.06
ACCEPTANCE_SCORE_WEIGHT_TERMINAL_TRACE_PENALTY = 0.05

# Search mode now switches profile sets directly.
FAST_SEARCH_MODE = 0

# -----------------------------------------------------------------------------
# Information-theoretic active sensing
# -----------------------------------------------------------------------------
INFO_OBJECTIVE = "hybrid"
INFO_TRACE_WEIGHT = 12.0
INFO_TERMINAL_TRACE_WEIGHT = 6.0
INFO_ENTROPY_GAIN_WEIGHT = 5.0
INFO_FIM_WEIGHT = 2.0
INFO_EPS = 1.0e-9

# Geometry-aware objective extensions (switchable by weight).
GEOM_FUTURE_DIST_WEIGHT = 8.0
GEOM_RISK_RANK_WEIGHT = 24.0
FUTURE_DIST_DANGER_THR_M = 200.0
FUTURE_DIST_CAUTION_THR_M = 1000.0

PLANNER_PROC_SIGMA_POS_M = 0.25
PLANNER_PROC_SIGMA_VEL_MPS = 0.01
PLANNER_MEAS_SIGMA_AZ_DEG = 0.08
PLANNER_MEAS_SIGMA_EL_DEG = 0.08
PLANNER_MEAS_EDGE_INFLATION = 3.0
PLANNER_MEAS_OUTSIDE_FOV_INFLATION = 8.0

@dataclass
class PlanResult:
    plan_df: pd.DataFrame
    trigger_row: pd.Series
    trigger_reason: str
    x_est: np.ndarray
    p_diag: np.ndarray
    schedule_nom: np.ndarray
    schedule_rob: np.ndarray
    plan_meta: dict


@dataclass(frozen=True)
class SearchProfile:
    plan_horizon_sec: float
    search_active_steps: int
    multistep_seed_topk: int
    random_multistart_count: int
    random_nonzero_steps_max: int


@dataclass(frozen=True)
class TriggerProfile:
    short_horizon_sec: float
    short_preboundary_band_px: float
    score_safe_margin_scale_px: float
    score_safe_margin_rate_scale_pxps: float
    score_warning_time_scale_sec: float
    score_breach_time_scale_sec: float
    score_est_range_scale_m: float
    score_future_distance_scale_m: float
    score_risk_rank_scale: float
    w_safe_margin: float
    w_safe_margin_rate: float
    w_warning_time: float
    w_breach_time: float
    w_est_range: float
    w_future_distance: float
    w_risk_rank: float
    trigger_score_thr: float
    watch_score_thr: float
    skip_low_risk_score_max: float
    task_consistency_enable: bool
    low_risk_pc_max: float
    low_risk_min_future_distance_m: float
    low_risk_required_worst_risk_rank: int
    low_risk_required_safe_dist_rate_pxps: float


@dataclass(frozen=True)
class AcceptanceProfile:
    control_mode: str
    max_trace_ratio_safe: float
    max_terminal_trace_ratio_safe: float
    max_trace_ratio_caution: float
    max_terminal_trace_ratio_caution: float
    max_trace_ratio_danger: float
    max_terminal_trace_ratio_danger: float
    min_info_delta: float
    geom_min_margin_gain_vs_zero_px: float
    geom_min_terminal_margin_gain_vs_zero_px: float
    geom_max_neg_margin_sum_delta_vs_zero_px: float
    hard_veto_max_violation_delta_vs_zero: int
    hard_veto_min_margin_px: float
    hard_veto_require_structurally_sane: bool
    hard_veto_future_dist_non_regression_tol_m: float
    hard_veto_future_dist_gate_for_caution_danger_only: bool
    accepted_score_thr: float
    borderline_score_thr: float
    score_margin_scale_px: float
    score_terminal_margin_scale_px: float
    score_info_delta_scale: float
    score_future_dist_gain_scale_m: float
    score_violation_delta_scale: float
    score_neg_margin_sum_delta_scale_px: float
    score_trace_penalty_scale: float
    score_terminal_trace_penalty_scale: float
    w_margin: float
    w_terminal_margin: float
    w_info_delta: float
    w_future_distance: float
    w_structural: float
    w_violation_delta: float
    w_neg_margin_delta: float
    w_trace_penalty: float
    w_terminal_trace_penalty: float


@dataclass(frozen=True)
class BudgetProfile:
    retry_enable: bool
    retry_sequence: tuple[float, ...]
    efficiency_diagnostics_enable: bool
    efficiency_min_budget_mps: float
    efficiency_min_margin_gain_per_dv: float
    efficiency_min_future_gain_per_dv: float
    efficiency_max_trace_penalty_per_dv: float
    efficiency_max_terminal_trace_penalty_per_dv: float
    high_gain_override_enable: bool
    high_gain_override_min_margin_gain_px: float
    high_gain_override_min_abs_margin_px: float
    high_gain_override_min_violation_reduction: int
    high_gain_override_max_trace_ratio: float
    high_gain_override_max_terminal_trace_ratio: float


@dataclass(frozen=True)
class NominalDiagnosticsProfile:
    min_margin_advantage_px: float
    max_min_relative_distance_loss_m: float
    max_terminal_relative_distance_loss_m: float
    max_trace_ratio_penalty: float
    max_terminal_trace_ratio_penalty: float
    min_robust_support_margin_px: float
    max_robust_support_violations: int
    require_robust_strict_acceptance: bool


TRIGGER_PROFILE = TriggerProfile(
    short_horizon_sec=float(SHORT_TRIGGER_HORIZON_SEC),
    short_preboundary_band_px=float(SHORT_PREBOUNDARY_BAND_PX),
    score_safe_margin_scale_px=float(TRIGGER_SCORE_SAFE_MARGIN_SCALE_PX),
    score_safe_margin_rate_scale_pxps=float(TRIGGER_SCORE_SAFE_MARGIN_RATE_SCALE_PXPS),
    score_warning_time_scale_sec=float(TRIGGER_SCORE_WARNING_TIME_SCALE_SEC),
    score_breach_time_scale_sec=float(TRIGGER_SCORE_BREACH_TIME_SCALE_SEC),
    score_est_range_scale_m=float(TRIGGER_SCORE_EST_RANGE_SCALE_M),
    score_future_distance_scale_m=float(TRIGGER_SCORE_FUTURE_DISTANCE_SCALE_M),
    score_risk_rank_scale=float(TRIGGER_SCORE_RISK_RANK_SCALE),
    w_safe_margin=float(TRIGGER_SCORE_W_SAFE_MARGIN),
    w_safe_margin_rate=float(TRIGGER_SCORE_W_SAFE_MARGIN_RATE),
    w_warning_time=float(TRIGGER_SCORE_W_WARNING_TIME),
    w_breach_time=float(TRIGGER_SCORE_W_BREACH_TIME),
    w_est_range=float(TRIGGER_SCORE_W_EST_RANGE),
    w_future_distance=float(TRIGGER_SCORE_W_FUTURE_DISTANCE),
    w_risk_rank=float(TRIGGER_SCORE_W_RISK_RANK),
    trigger_score_thr=float(TRIGGER_STATE_TRIGGER_SCORE_THR),
    watch_score_thr=float(TRIGGER_STATE_WATCH_SCORE_THR),
    skip_low_risk_score_max=float(TRIGGER_STATE_SKIP_LOW_RISK_SCORE_MAX),
    task_consistency_enable=bool(TRIGGER_TASK_CONSISTENCY_ENABLE),
    low_risk_pc_max=float(TRIGGER_LOW_RISK_PC_MAX),
    low_risk_min_future_distance_m=float(TRIGGER_LOW_RISK_MIN_FUTURE_DISTANCE_M),
    low_risk_required_worst_risk_rank=int(TRIGGER_LOW_RISK_REQUIRED_WORST_RISK_RANK),
    low_risk_required_safe_dist_rate_pxps=float(TRIGGER_LOW_RISK_REQUIRED_SAFE_DIST_RATE_PXPS),
)


ACCEPTANCE_PROFILE = AcceptanceProfile(
    control_mode=str(ACCEPTANCE_CONTROL_MODE),
    max_trace_ratio_safe=float(ACCEPTANCE_MAX_TRACE_RATIO_SAFE),
    max_terminal_trace_ratio_safe=float(ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_SAFE),
    max_trace_ratio_caution=float(ACCEPTANCE_MAX_TRACE_RATIO_CAUTION),
    max_terminal_trace_ratio_caution=float(ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_CAUTION),
    max_trace_ratio_danger=float(ACCEPTANCE_MAX_TRACE_RATIO_DANGER),
    max_terminal_trace_ratio_danger=float(ACCEPTANCE_MAX_TERMINAL_TRACE_RATIO_DANGER),
    min_info_delta=float(ACCEPTANCE_MIN_INFO_DELTA),
    geom_min_margin_gain_vs_zero_px=float(ACCEPTANCE_GEOM_MIN_MARGIN_GAIN_VS_ZERO_PX),
    geom_min_terminal_margin_gain_vs_zero_px=float(ACCEPTANCE_GEOM_MIN_TERMINAL_MARGIN_GAIN_VS_ZERO_PX),
    geom_max_neg_margin_sum_delta_vs_zero_px=float(ACCEPTANCE_GEOM_MAX_NEG_MARGIN_SUM_DELTA_VS_ZERO),
    hard_veto_max_violation_delta_vs_zero=int(ACCEPTANCE_HARD_VETO_MAX_VIOLATION_DELTA_VS_ZERO),
    hard_veto_min_margin_px=float(ACCEPTANCE_HARD_VETO_MIN_MARGIN_PX),
    hard_veto_require_structurally_sane=bool(ACCEPTANCE_HARD_VETO_REQUIRE_STRUCTURALLY_SANE),
    hard_veto_future_dist_non_regression_tol_m=float(ACCEPTANCE_HARD_VETO_FUTURE_DIST_NON_REGRESSION_TOL_M),
    hard_veto_future_dist_gate_for_caution_danger_only=bool(ACCEPTANCE_HARD_VETO_FUTURE_DIST_GATE_FOR_CAUTION_DANGER_ONLY),
    accepted_score_thr=float(ACCEPTANCE_SCORE_ACCEPTED_THR),
    borderline_score_thr=float(ACCEPTANCE_SCORE_BORDERLINE_THR),
    score_margin_scale_px=float(ACCEPTANCE_SCORE_MARGIN_SCALE_PX),
    score_terminal_margin_scale_px=float(ACCEPTANCE_SCORE_TERMINAL_MARGIN_SCALE_PX),
    score_info_delta_scale=float(ACCEPTANCE_SCORE_INFO_DELTA_SCALE),
    score_future_dist_gain_scale_m=float(ACCEPTANCE_SCORE_FUTURE_DIST_GAIN_SCALE_M),
    score_violation_delta_scale=float(ACCEPTANCE_SCORE_VIOLATION_DELTA_SCALE),
    score_neg_margin_sum_delta_scale_px=float(ACCEPTANCE_SCORE_NEG_MARGIN_SUM_DELTA_SCALE_PX),
    score_trace_penalty_scale=float(ACCEPTANCE_SCORE_TRACE_PENALTY_SCALE),
    score_terminal_trace_penalty_scale=float(ACCEPTANCE_SCORE_TERMINAL_TRACE_PENALTY_SCALE),
    w_margin=float(ACCEPTANCE_SCORE_WEIGHT_MARGIN),
    w_terminal_margin=float(ACCEPTANCE_SCORE_WEIGHT_TERMINAL_MARGIN),
    w_info_delta=float(ACCEPTANCE_SCORE_WEIGHT_INFO_DELTA),
    w_future_distance=float(ACCEPTANCE_SCORE_WEIGHT_FUTURE_DISTANCE),
    w_structural=float(ACCEPTANCE_SCORE_WEIGHT_STRUCTURAL),
    w_violation_delta=float(ACCEPTANCE_SCORE_WEIGHT_VIOLATION_DELTA),
    w_neg_margin_delta=float(ACCEPTANCE_SCORE_WEIGHT_NEG_MARGIN_DELTA),
    w_trace_penalty=float(ACCEPTANCE_SCORE_WEIGHT_TRACE_PENALTY),
    w_terminal_trace_penalty=float(ACCEPTANCE_SCORE_WEIGHT_TERMINAL_TRACE_PENALTY),
)


BUDGET_PROFILE = BudgetProfile(
    retry_enable=False,
    retry_sequence=tuple(float(v) for v in BUDGET_RETRY_SEQUENCE),
    efficiency_diagnostics_enable=True,
    efficiency_min_budget_mps=0.24,
    efficiency_min_margin_gain_per_dv=10.0,
    efficiency_min_future_gain_per_dv=15.0,
    efficiency_max_trace_penalty_per_dv=0.45,
    efficiency_max_terminal_trace_penalty_per_dv=0.45,
    high_gain_override_enable=True,
    high_gain_override_min_margin_gain_px=120.0,
    high_gain_override_min_abs_margin_px=24.0,
    high_gain_override_min_violation_reduction=4,
    high_gain_override_max_trace_ratio=1.08,
    high_gain_override_max_terminal_trace_ratio=1.08,
)


NOMINAL_DIAGNOSTICS_PROFILE = NominalDiagnosticsProfile(
    min_margin_advantage_px=15.0,
    max_min_relative_distance_loss_m=2.0,
    max_terminal_relative_distance_loss_m=2.0,
    max_trace_ratio_penalty=0.01,
    max_terminal_trace_ratio_penalty=0.01,
    min_robust_support_margin_px=5.0,
    max_robust_support_violations=0,
    require_robust_strict_acceptance=True,
)


class EvalResult(NamedTuple):
    metric: float
    min_safe_margin_px: float
    terminal_margin_px: float
    violation_count: int
    negative_margin_sum_px: float
    total_dv_mag_mps: float
    margins_by_anchor: dict
    info_metric: float
    trace_pos_sum: float
    terminal_trace_pos: float
    entropy_reduction: float
    fim_trace_sum: float
    visible_update_steps: int
    min_relative_distance_m: float
    terminal_relative_distance_m: float
    min_future_min_distance_m: float
    terminal_future_min_distance_m: float
    worst_risk_rank_pred: int


EVAL_CACHE_ROUND_DECIMALS = 10
_SEQ_KEY_ROUND_DECIMALS = 10
_EVAL_CACHE: dict[tuple, EvalResult] = {}


def _round_key_array(arr: np.ndarray | Iterable[float], decimals: int = EVAL_CACHE_ROUND_DECIMALS) -> tuple[float, ...]:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return ()
    return tuple(np.round(a.reshape(-1), decimals=decimals).tolist())


def _anchors_cache_key(anchors: list[tuple[str, np.ndarray]]) -> tuple:
    return tuple((str(name), _round_key_array(x0)) for name, x0 in anchors)


def _eval_cache_key(dv_seq: np.ndarray, anchors: list[tuple[str, np.ndarray]], dt: float, n_orbit: float, p_diag: np.ndarray | None) -> tuple:
    return (
        _round_key_array(dv_seq),
        _anchors_cache_key(anchors),
        round(float(dt), EVAL_CACHE_ROUND_DECIMALS),
        round(float(n_orbit), EVAL_CACHE_ROUND_DECIMALS),
        None if p_diag is None else _round_key_array(p_diag),
    )


def _clear_eval_cache() -> None:
    _EVAL_CACHE.clear()


def _process_pool_kwargs(max_workers: int) -> dict:
    kwargs: dict = {"max_workers": int(max_workers)}
    try:
        kwargs["mp_context"] = mp.get_context("spawn")
    except Exception:
        pass
    return kwargs


STANDARD_SEARCH_PROFILES: dict[str, SearchProfile] = {
    "normal": SearchProfile(
        plan_horizon_sec=120.0,
        search_active_steps=20,
        multistep_seed_topk=12,
        random_multistart_count=36,
        random_nonzero_steps_max=4,
    ),
    "hard": SearchProfile(
        plan_horizon_sec=260.0,
        search_active_steps=40,
        multistep_seed_topk=36,
        random_multistart_count=128,
        random_nonzero_steps_max=5,
    ),
    "extreme": SearchProfile(
        plan_horizon_sec=320.0,
        search_active_steps=48,
        multistep_seed_topk=48,
        random_multistart_count=160,
        random_nonzero_steps_max=6,
    ),
}

FAST_SEARCH_PROFILES: dict[str, SearchProfile] = {
    "normal": SearchProfile(
        plan_horizon_sec=120.0,
        search_active_steps=20,
        multistep_seed_topk=12,
        random_multistart_count=36,
        random_nonzero_steps_max=4,
    ),
    "hard": SearchProfile(
        plan_horizon_sec=190.0,
        search_active_steps=28,
        multistep_seed_topk=18,
        random_multistart_count=52,
        random_nonzero_steps_max=6,
    ),
    "extreme": SearchProfile(
        plan_horizon_sec=210.0,
        search_active_steps=32,
        multistep_seed_topk=20,
        random_multistart_count=64,
        random_nonzero_steps_max=8,
    ),
}


def _search_profile_to_dict(profile: SearchProfile) -> dict[str, float | int]:
    return {
        "plan_horizon_sec": float(profile.plan_horizon_sec),
        "search_active_steps": int(profile.search_active_steps),
        "multistep_seed_topk": int(profile.multistep_seed_topk),
        "random_multistart_count": int(profile.random_multistart_count),
        "random_nonzero_steps_max": int(profile.random_nonzero_steps_max),
    }


def _classify_search_hardness(case_label: str, total_dv_budget: float) -> str:
    label = str(case_label).strip().lower()
    budget = float(total_dv_budget)
    if label == "extreme_hard_case" or budget >= DV_MAG_SUPER - 1.0e-9:
        return "extreme"
    if label in {"hard_case", "very_hard_case"} or budget >= DV_MAG_EXTREME - 1.0e-9:
        return "hard"
    return "normal"


def _current_search_profile_snapshot() -> dict[str, float | int]:
    return {
        "plan_horizon_sec": float(PLAN_HORIZON_SEC),
        "search_active_steps": int(SEARCH_ACTIVE_STEPS),
        "multistep_seed_topk": int(MULTISTEP_SEED_TOPK),
        "random_multistart_count": int(RANDOM_MULTISTART_COUNT),
        "random_nonzero_steps_max": int(RANDOM_NONZERO_STEPS_MAX),
    }


def _seq_key(seq: np.ndarray) -> tuple[float, ...]:
    return _round_key_array(np.asarray(seq, dtype=float), decimals=_SEQ_KEY_ROUND_DECIMALS)

def _safe_int(value, default: int = -1) -> int:
    try:
        if value is None:
            return int(default)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return int(value) if np.isfinite(value) else int(default)
        text = str(value).strip()
        if text == "" or text.lower() in {"nan", "none", "null"}:
            return int(default)
        return int(float(text))
    except Exception:
        return int(default)


# -----------------------------------------------------------------------------
# Geometry / dynamics
# -----------------------------------------------------------------------------
def focal_length_px() -> float:
    return IMG_W / (2.0 * math.tan(math.radians(FOV_X_DEG) / 2.0))


def state_to_pixel(R: float, T: float, N: float) -> tuple[float, float]:
    T_safe = T if abs(T) > 1e-6 else (1e-6 if T >= 0.0 else -1e-6)
    f = focal_length_px()
    u = IMG_W / 2.0 + f * (R / T_safe)
    v = IMG_H / 2.0 - f * (N / T_safe)
    return float(u), float(v)


def safe_box_bounds(safe_ratio: float = SAFE_RATIO) -> tuple[float, float, float, float]:
    margin = IMG_W * (1.0 - safe_ratio) / 2.0
    return float(margin), float(IMG_W - margin), float(margin), float(IMG_H - margin)


def signed_distance_to_safe_box(u: float, v: float, safe_ratio: float = SAFE_RATIO) -> float:
    left, right, top, bottom = safe_box_bounds(safe_ratio)
    return float(min(u - left, right - u, v - top, bottom - v))


def estimate_mean_motion_from_truth(df: pd.DataFrame) -> float:
    r_norm = np.linalg.norm([
        float(df.loc[0, "rc_x_m"]),
        float(df.loc[0, "rc_y_m"]),
        float(df.loc[0, "rc_z_m"]),
    ])
    v_norm = np.linalg.norm([
        float(df.loc[0, "vc_x_mps"]),
        float(df.loc[0, "vc_y_mps"]),
        float(df.loc[0, "vc_z_mps"]),
    ])
    a = 1.0 / (2.0 / r_norm - v_norm ** 2 / MU)
    return math.sqrt(MU / a ** 3)


def hcw_deriv(x: np.ndarray, n: float) -> np.ndarray:
    R, T, N, dR, dT, dN = x
    ddR = 2.0 * n * dT + 3.0 * n**2 * R
    ddT = -2.0 * n * dR
    ddN = -n**2 * N
    return np.array([dR, dT, dN, ddR, ddT, ddN], dtype=float)


def rk4_step(x: np.ndarray, dt: float, n: float) -> np.ndarray:
    k1 = hcw_deriv(x, n)
    k2 = hcw_deriv(x + 0.5 * dt * k1, n)
    k3 = hcw_deriv(x + 0.5 * dt * k2, n)
    k4 = hcw_deriv(x + dt * k3, n)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def propagate_with_schedule(x0: np.ndarray, dv_seq: np.ndarray, dt: float, n: float) -> list[np.ndarray]:
    """
    Apply dv_k at the beginning of each step and record the post-impulse state
    at that sampling instant.

    This keeps planner rollout semantics consistent with
    sim/apply_active_micro_dv_to_truth.py and preview geometry checks: all are
    step-begin, post-impulse sampled trajectories.

    Relative-state convention here is target relative to observer, so an observer
    impulse +dv induces relative velocity change -dv.
    """
    x = np.array(x0, dtype=float).copy()
    states = []
    for dv in dv_seq:
        x = x.copy()
        x[3:6] -= np.asarray(dv, dtype=float)
        states.append(x.copy())
        x = rk4_step(x, dt, n)
    return states


def rollout_no_control(x0: np.ndarray, dt: float, n: float, horizon_sec: float) -> list[np.ndarray]:
    K = int(round(horizon_sec / dt))
    seq = np.zeros((K, 3), dtype=float)
    return propagate_with_schedule(x0, seq, dt, n)


def _future_distance_profile(ranges: np.ndarray) -> np.ndarray:
    arr = np.asarray(ranges, dtype=float).reshape(-1)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    return np.minimum.accumulate(arr[::-1])[::-1]


def _risk_rank_from_distance_m(distance_m: float) -> int:
    if not np.isfinite(float(distance_m)):
        return -999
    if float(distance_m) <= FUTURE_DIST_DANGER_THR_M:
        return 2
    if float(distance_m) <= FUTURE_DIST_CAUTION_THR_M:
        return 1
    return 0


def _risk_rank_from_label(label: str) -> int:
    s = str(label).strip().lower()
    if s == "danger":
        return 2
    if s == "caution":
        return 1
    if s == "safe":
        return 0
    return -999


def _strict_accept_trace_limits(
    zero_worst_risk_rank_pred: int,
    *,
    profile: AcceptanceProfile = ACCEPTANCE_PROFILE,
) -> tuple[float, float]:
    rank = _safe_int(zero_worst_risk_rank_pred, -999)
    if rank >= 2:
        return float(profile.max_trace_ratio_danger), float(profile.max_terminal_trace_ratio_danger)
    if rank >= 1:
        return float(profile.max_trace_ratio_caution), float(profile.max_terminal_trace_ratio_caution)
    return float(profile.max_trace_ratio_safe), float(profile.max_terminal_trace_ratio_safe)


def _internal_preview_metrics_from_state(x0: np.ndarray, dv_seq: np.ndarray, dt: float, n_orbit: float) -> dict[str, float | int]:
    states = propagate_with_schedule(np.asarray(x0, dtype=float), np.asarray(dv_seq, dtype=float), dt, n_orbit)
    if len(states) == 0:
        return {
            "internal_preview_min_relative_distance_m": float("nan"),
            "internal_preview_min_future_distance_m": float("nan"),
            "internal_preview_terminal_future_distance_m": float("nan"),
            "internal_preview_worst_risk_rank": -999,
        }
    st = np.asarray(states, dtype=float)
    ranges = np.linalg.norm(st[:, :3], axis=1)
    future_profile = _future_distance_profile(ranges)
    min_rel = float(np.min(ranges))
    min_future = float(np.min(future_profile))
    terminal_future = float(future_profile[-1])
    return {
        "internal_preview_min_relative_distance_m": min_rel,
        "internal_preview_min_future_distance_m": min_future,
        "internal_preview_terminal_future_distance_m": terminal_future,
        "internal_preview_worst_risk_rank": int(_risk_rank_from_distance_m(min_future)),
    }


def pixel_safe_margins(states: Iterable[np.ndarray], safe_ratio: float = SAFE_RATIO) -> np.ndarray:
    margins = []
    for x in states:
        u, v = state_to_pixel(float(x[0]), float(x[1]), float(x[2]))
        margins.append(signed_distance_to_safe_box(u, v, safe_ratio=safe_ratio))
    return np.asarray(margins, dtype=float)


def _build_static_candidate_directions() -> tuple[np.ndarray, ...]:
    raw_dirs = [
        np.array([1.0, 0.0, 0.0], dtype=float), np.array([-1.0, 0.0, 0.0], dtype=float),
        np.array([0.0, 1.0, 0.0], dtype=float), np.array([0.0, -1.0, 0.0], dtype=float),
        np.array([0.0, 0.0, 1.0], dtype=float), np.array([0.0, 0.0, -1.0], dtype=float),
    ]
    for a in (-1.0, 1.0):
        for b in (-1.0, 1.0):
            raw_dirs.append(np.array([a, b, 0.0], dtype=float))
            raw_dirs.append(np.array([a, 0.0, b], dtype=float))
            raw_dirs.append(np.array([0.0, a, b], dtype=float))
            raw_dirs.append(np.array([a, b, b], dtype=float))

    out: list[np.ndarray] = []
    seen = set()
    for d in raw_dirs:
        nrm = float(np.linalg.norm(d))
        if nrm < 1.0e-12:
            continue
        u = d / nrm
        key = tuple(np.round(u, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return tuple(out)


STATIC_CANDIDATE_DIRECTIONS = _build_static_candidate_directions()


# -----------------------------------------------------------------------------
# Trigger selection
# -----------------------------------------------------------------------------

def _extract_trigger_diagnostics(trig: pd.Series) -> dict[str, float | int]:
    out: dict[str, float | int] = {}
    for key in TRIGGER_DIAGNOSTIC_KEYS:
        if key == "trigger_state":
            out[key] = str(trig.get(key, ""))
            continue
        if key in {"risk_score", "warning_score"}:
            try:
                out[key] = _safe_int(trig.get(key, -1))
            except Exception:
                out[key] = -1
            continue
        try:
            val = float(trig.get(key, float("nan")))
            out[key] = val if np.isfinite(val) else float("nan")
        except Exception:
            out[key] = float("nan")
    return out


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


def _evaluate_trigger_score(
    *,
    safe_dist_px: float,
    safe_dist_rate_pxps: float,
    short_time_to_warning_sec: float,
    short_time_to_breach_sec: float,
    est_range_m: float,
    future_min_distance_m: float,
    risk_rank: int,
    profile: TriggerProfile = TRIGGER_PROFILE,
) -> float:
    margin_urgency = _clip01(
        (float(profile.short_preboundary_band_px) - safe_dist_px)
        / max(float(profile.score_safe_margin_scale_px), 1.0e-9)
    )
    rate_urgency = _clip01((-safe_dist_rate_pxps) / max(float(profile.score_safe_margin_rate_scale_pxps), 1.0e-9))
    warning_urgency = _clip01(
        (float(profile.score_warning_time_scale_sec) - short_time_to_warning_sec)
        / max(float(profile.score_warning_time_scale_sec), 1.0e-9)
    )
    breach_urgency = _clip01(
        (float(profile.score_breach_time_scale_sec) - short_time_to_breach_sec)
        / max(float(profile.score_breach_time_scale_sec), 1.0e-9)
    )
    range_urgency = _clip01(
        (float(profile.score_est_range_scale_m) - est_range_m)
        / max(float(profile.score_est_range_scale_m), 1.0e-9)
    )
    future_distance_urgency = _clip01(
        (float(profile.score_future_distance_scale_m) - future_min_distance_m)
        / max(float(profile.score_future_distance_scale_m), 1.0e-9)
    )
    risk_rank_urgency = _clip01(float(risk_rank) / max(float(profile.score_risk_rank_scale), 1.0e-9))

    score = (
        float(profile.w_safe_margin) * margin_urgency
        + float(profile.w_safe_margin_rate) * rate_urgency
        + float(profile.w_warning_time) * warning_urgency
        + float(profile.w_breach_time) * breach_urgency
        + float(profile.w_est_range) * range_urgency
        + float(profile.w_future_distance) * future_distance_urgency
        + float(profile.w_risk_rank) * risk_rank_urgency
    )
    return float(np.clip(score, 0.0, 1.0))


def evaluate_trigger_state(
    row: pd.Series,
    *,
    low_risk_context: bool,
    profile: TriggerProfile = TRIGGER_PROFILE,
) -> tuple[str, float]:
    safe_dist_px = float(row.get("safe_dist_px", float("nan")))
    safe_dist_rate_pxps = float(row.get("safe_dist_rate_pxps", float("nan")))
    short_warning = float(row.get("short_time_to_warning_sec", float("nan")))
    short_breach = float(row.get("short_time_to_breach_sec", float("nan")))
    est_range_m = float(row.get("est_range_m", float("nan")))
    future_min_distance_m = float(row.get("pred_min_future_min_distance_m", float("nan")))
    risk_rank = _safe_int(row.get("pred_worst_risk_rank", -999), -999)

    required = [safe_dist_px, safe_dist_rate_pxps, short_warning, short_breach, est_range_m, future_min_distance_m]
    if not all(np.isfinite(v) for v in required):
        return "skip_unobservable", 0.0

    score = _evaluate_trigger_score(
        safe_dist_px=safe_dist_px,
        safe_dist_rate_pxps=safe_dist_rate_pxps,
        short_time_to_warning_sec=short_warning,
        short_time_to_breach_sec=short_breach,
        est_range_m=est_range_m,
        future_min_distance_m=future_min_distance_m,
        risk_rank=max(risk_rank, 0),
        profile=profile,
    )

    if score >= float(profile.trigger_score_thr):
        return "trigger", score

    if low_risk_context:
        low_risk_stable = (
            future_min_distance_m >= float(profile.low_risk_min_future_distance_m)
            and max(risk_rank, 0) < int(profile.low_risk_required_worst_risk_rank)
            and safe_dist_rate_pxps > float(profile.low_risk_required_safe_dist_rate_pxps)
        )
        if low_risk_stable and score <= float(profile.skip_low_risk_score_max):
            return "skip_low_risk", score

    if score >= float(profile.watch_score_thr):
        return "watch", score

    return ("skip_low_risk", score) if low_risk_context else ("watch", score)

def choose_trigger_row_exp3(
    imm_df: pd.DataFrame,
    n_orbit: float,
    *,
    orig_last_risk_level: str = "",
    orig_max_pc_bplane: float = float("nan"),
    orig_min_future_min_distance_m: float = float("nan"),
    profile: TriggerProfile = TRIGGER_PROFILE,
) -> tuple[pd.Series, str]:
    use_df = imm_df.copy()
    if "visible" in use_df.columns:
        use_df = use_df[use_df["visible"] == 1].copy()
    if len(use_df) == 0:
        raise RuntimeError("No visible rows are available for Experiment III trigger selection.")

    cur_safe = []
    pred_min = []
    pred_breach = []
    short_pred_min = []
    short_pred_breach = []
    short_time_to_breach_sec = []
    short_time_to_warning_sec = []
    est_range_m = []
    pred_min_future_min_distance_m = []
    pred_worst_risk_rank = []

    short_steps = max(1, int(round(float(profile.short_horizon_sec) / PLAN_DT)))
    for _, row in use_df.iterrows():
        x_est = np.array([
            float(row["R_est_m"]), float(row["T_est_m"]), float(row["N_est_m"]),
            float(row["dR_est_mps"]), float(row["dT_est_mps"]), float(row["dN_est_mps"]),
        ], dtype=float)
        est_range_m.append(float(np.linalg.norm(x_est[:3])))
        u, v = state_to_pixel(x_est[0], x_est[1], x_est[2])
        cur_safe.append(signed_distance_to_safe_box(u, v, SAFE_RATIO))

        future = rollout_no_control(x_est, PLAN_DT, n_orbit, PLAN_HORIZON_SEC)
        margins = pixel_safe_margins(future, SAFE_RATIO)
        pred_min.append(float(np.min(margins)))
        pred_breach.append(int(np.any(margins <= 0.0)))

        future_arr = np.asarray(future, dtype=float)
        if future_arr.ndim == 2 and future_arr.shape[1] >= 3:
            future_ranges = np.linalg.norm(future_arr[:, :3], axis=1)
            future_profile = _future_distance_profile(future_ranges)
            min_future_dist = float(np.min(future_profile)) if future_profile.size > 0 else float("nan")
        else:
            min_future_dist = float("nan")
        pred_min_future_min_distance_m.append(min_future_dist)
        pred_worst_risk_rank.append(int(_risk_rank_from_distance_m(min_future_dist)))

        short_margins = margins[:short_steps]
        short_pred_min.append(float(np.min(short_margins)))
        short_pred_breach.append(int(np.any(short_margins <= 0.0)))

        breach_idx = np.where(short_margins <= 0.0)[0]
        warn_idx = np.where(short_margins <= float(profile.short_preboundary_band_px))[0]
        short_time_to_breach_sec.append(float((breach_idx[0] + 1) * PLAN_DT) if breach_idx.size > 0 else float("inf"))
        short_time_to_warning_sec.append(float((warn_idx[0] + 1) * PLAN_DT) if warn_idx.size > 0 else float("inf"))

    use_df = use_df.copy()
    use_df["safe_dist_px"] = cur_safe
    use_df["pred_min_safe_margin_px"] = pred_min
    use_df["predicted_breach"] = pred_breach
    use_df["short_pred_min_safe_margin_px"] = short_pred_min
    use_df["short_predicted_breach"] = short_pred_breach
    use_df["short_time_to_breach_sec"] = short_time_to_breach_sec
    use_df["short_time_to_warning_sec"] = short_time_to_warning_sec
    use_df["est_range_m"] = est_range_m
    use_df["pred_min_future_min_distance_m"] = pred_min_future_min_distance_m
    use_df["pred_worst_risk_rank"] = pred_worst_risk_rank
    use_df["visible_count"] = np.arange(1, len(use_df) + 1, dtype=int)

    safe_rate = np.full(len(use_df), 0.0, dtype=float)
    t_arr = use_df["t_sec"].to_numpy(dtype=float)
    safe_arr = use_df["safe_dist_px"].to_numpy(dtype=float)
    for i in range(1, len(use_df)):
        dt = max(float(t_arr[i] - t_arr[i - 1]), 1.0e-6)
        safe_rate[i] = (safe_arr[i] - safe_arr[i - 1]) / dt
    if len(use_df) > 1:
        safe_rate[0] = safe_rate[1]
    use_df["safe_dist_rate_pxps"] = pd.Series(safe_rate).rolling(2, min_periods=1).mean().to_numpy()

    eligible = use_df[(use_df["t_sec"] >= MIN_TRIGGER_SEC) & (use_df["frame"] >= MIN_TRIGGER_FRAME)].copy()
    if len(eligible) == 0:
        return use_df.iloc[0], "fallback_first_visible"

    try:
        orig_pc = float(orig_max_pc_bplane)
    except Exception:
        orig_pc = float("nan")
    try:
        orig_min_future = float(orig_min_future_min_distance_m)
    except Exception:
        orig_min_future = float("nan")

    has_task_context = (
        bool(str(orig_last_risk_level).strip())
        or np.isfinite(orig_pc)
        or np.isfinite(orig_min_future)
    )
    context_rank = _risk_rank_from_label(str(orig_last_risk_level))
    if np.isfinite(orig_min_future):
        context_rank = max(context_rank, _risk_rank_from_distance_m(orig_min_future))
    if np.isfinite(orig_pc) and orig_pc > float(profile.low_risk_pc_max):
        context_rank = max(context_rank, 1)

    low_risk_context = bool(
        bool(profile.task_consistency_enable)
        and has_task_context
        and (context_rank <= 0)
        and ((not np.isfinite(orig_pc)) or (orig_pc <= float(profile.low_risk_pc_max)))
        and (
            (not np.isfinite(orig_min_future))
            or (orig_min_future >= float(profile.low_risk_min_future_distance_m))
        )
    )

    trigger_states: list[str] = []
    trigger_scores: list[float] = []
    for _, row in eligible.iterrows():
        state, score = evaluate_trigger_state(
            row,
            low_risk_context=low_risk_context,
            profile=profile,
        )
        trigger_states.append(str(state))
        trigger_scores.append(float(score))

    eligible["trigger_state"] = trigger_states
    eligible["trigger_score"] = np.asarray(trigger_scores, dtype=float)
    eligible["warning_score"] = np.clip(np.round(eligible["trigger_score"] * 4.0), 0, 4).astype(int)
    eligible["risk_score"] = np.clip(np.round(eligible["trigger_score"] * 10.0), 0, 10).astype(int)
    eligible["risk_integral"] = pd.Series(eligible["risk_score"], dtype=float).rolling(5, min_periods=1).sum().to_numpy()
    eligible["warning_integral"] = pd.Series(eligible["warning_score"], dtype=float).rolling(5, min_periods=1).sum().to_numpy()

    trigger_rows = eligible[eligible["trigger_state"] == "trigger"]
    if len(trigger_rows) > 0:
        return trigger_rows.iloc[0], "trigger"

    skip_unobservable_rows = eligible[eligible["trigger_state"] == "skip_unobservable"]
    if len(skip_unobservable_rows) == len(eligible):
        return skip_unobservable_rows.iloc[0], "skip_unobservable"

    if low_risk_context:
        skip_low_risk_rows = eligible[eligible["trigger_state"] == "skip_low_risk"]
        if len(skip_low_risk_rows) > 0:
            return skip_low_risk_rows.iloc[0], "skip_low_risk"

    watch_rows = eligible[eligible["trigger_state"] == "watch"]
    if len(watch_rows) > 0:
        watch_rows = watch_rows.sort_values(["trigger_score", "t_sec"], ascending=[False, True])
        return watch_rows.iloc[0], "watch"

    if len(skip_unobservable_rows) > 0:
        return skip_unobservable_rows.iloc[0], "skip_unobservable"

    return eligible.iloc[0], "watch"

# -----------------------------------------------------------------------------
# Anchors and exact schedule evaluation
# -----------------------------------------------------------------------------
def extract_planner_cov_diag_from_trigger_row(trig: pd.Series) -> np.ndarray:
    return np.array([
        float(trig.get("P00", 50.0**2)),
        float(trig.get("P11", 80.0**2)),
        float(trig.get("P22", 50.0**2)),
        float(trig.get("P33", 0.1**2)),
        float(trig.get("P44", 0.1**2)),
        float(trig.get("P55", 0.1**2)),
    ], dtype=float)


def trigger_truth_state(trig: pd.Series) -> np.ndarray | None:
    keys = ["rel_R_m", "rel_T_m", "rel_N_m", "rel_dR_mps", "rel_dT_mps", "rel_dN_mps"]
    if not all(k in trig.index for k in keys):
        return None
    arr = np.array([float(trig[k]) for k in keys], dtype=float)
    if np.all(np.isfinite(arr)):
        return arr
    return None


def build_anchor_states(trig: pd.Series, x_est: np.ndarray, p_diag: np.ndarray, robust: bool = True) -> list[tuple[str, np.ndarray]]:
    anchors: list[tuple[str, np.ndarray]] = [("estimate", x_est.copy())]
    x_truth = trigger_truth_state(trig)
    if robust and x_truth is not None:
        anchors.append(("truth", x_truth.copy()))

    if robust:
        sig = np.sqrt(np.maximum(p_diag, 1.0e-12))
        for sr in (-1.0, 1.0):
            x = x_est.copy()
            x[0] += sr * SIGMA_ENVELOPE_SCALE * sig[0]
            anchors.append((f"sigma_R_{sr:+.0f}", x))
        for st in (-1.0, 1.0):
            x = x_est.copy()
            x[1] += st * 0.8 * SIGMA_ENVELOPE_SCALE * sig[1]
            anchors.append((f"sigma_T_{st:+.0f}", x))
        for sn in (-1.0, 1.0):
            x = x_est.copy()
            x[2] += sn * SIGMA_ENVELOPE_SCALE * sig[2]
            anchors.append((f"sigma_N_{sn:+.0f}", x))
    return anchors


def _inside_image(u: float, v: float) -> bool:
    return (0.0 <= u <= IMG_W) and (0.0 <= v <= IMG_H)


def _continuous_hcw_matrix(n: float) -> np.ndarray:
    return np.array([
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [3.0 * n ** 2, 0.0, 0.0, 0.0, 2.0 * n, 0.0],
        [0.0, 0.0, 0.0, -2.0 * n, 0.0, 0.0],
        [0.0, 0.0, -n ** 2, 0.0, 0.0, 0.0],
    ], dtype=float)


def _planner_process_noise(dt: float) -> np.ndarray:
    q = np.diag([
        PLANNER_PROC_SIGMA_POS_M ** 2,
        PLANNER_PROC_SIGMA_POS_M ** 2,
        PLANNER_PROC_SIGMA_POS_M ** 2,
        PLANNER_PROC_SIGMA_VEL_MPS ** 2,
        PLANNER_PROC_SIGMA_VEL_MPS ** 2,
        PLANNER_PROC_SIGMA_VEL_MPS ** 2,
    ])
    return nearest_spd(q * max(float(dt), 1.0))


def _bearing_jacobian(x: np.ndarray) -> np.ndarray:
    R, T, N = float(x[0]), float(x[1]), float(x[2])
    den_az = max(R * R + T * T, 1.0e-12)
    den_el = max(N * N + T * T, 1.0e-12)
    H = np.zeros((2, 6), dtype=float)
    H[0, 0] = T / den_az
    H[0, 1] = -R / den_az
    H[1, 1] = -N / den_el
    H[1, 2] = T / den_el
    return H


def _planner_measurement_cov(x: np.ndarray) -> np.ndarray:
    u, v = state_to_pixel(float(x[0]), float(x[1]), float(x[2]))
    base = np.diag(np.deg2rad([PLANNER_MEAS_SIGMA_AZ_DEG, PLANNER_MEAS_SIGMA_EL_DEG]) ** 2)

    edge_margin = min(u, IMG_W - u, v, IMG_H - v)
    edge_ref = 0.5 * min(IMG_W, IMG_H)
    edge_ratio = np.clip(edge_margin / max(edge_ref, 1.0), 0.0, 1.0)
    inflation = 1.0 + PLANNER_MEAS_EDGE_INFLATION * (1.0 - edge_ratio) ** 2
    if not _inside_image(u, v):
        inflation *= PLANNER_MEAS_OUTSIDE_FOV_INFLATION
    return nearest_spd(base * inflation)


def _safe_slogdet_pos(P: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(nearest_spd(P[:3, :3], eps=1.0e-9))
    if sign <= 0:
        return math.log(max(np.linalg.det(nearest_spd(P[:3, :3], eps=1.0e-9)), 1.0e-20))
    return float(logdet)


def _rollout_information_metrics(states: list[np.ndarray], p_diag: np.ndarray, dt: float, n_orbit: float) -> dict:
    if len(states) == 0:
        return {
            "trace_pos_sum": float("nan"),
            "terminal_trace_pos": float("nan"),
            "entropy_reduction": 0.0,
            "fim_trace_sum": 0.0,
            "visible_update_steps": 0,
        }

    p_diag = np.asarray(p_diag, dtype=float).reshape(-1)
    if p_diag.size < 6:
        p0 = np.diag([50.0 ** 2, 80.0 ** 2, 50.0 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2])
    else:
        p0 = np.diag(np.maximum(p_diag[:6], 1.0e-12))

    F = expm(_continuous_hcw_matrix(n_orbit) * float(dt))
    Q = _planner_process_noise(dt)

    P = nearest_spd(p0)
    trace_pos_sum = 0.0
    entropy_reduction = 0.0
    fim_trace_sum = 0.0
    visible_update_steps = 0

    for x in states:
        P_pred = nearest_spd(F @ P @ F.T + Q)
        H = _bearing_jacobian(x)
        Rk = _planner_measurement_cov(x)
        H_pos = H[:, :3]
        fim_pos = nearest_spd(H_pos.T @ safe_inv_spd(Rk, eps=1.0e-9) @ H_pos + INFO_EPS * np.eye(3))

        u, v = state_to_pixel(float(x[0]), float(x[1]), float(x[2]))
        if _inside_image(u, v):
            S = nearest_spd(H @ P_pred @ H.T + Rk)
            K = P_pred @ H.T @ safe_inv_spd(S, eps=1.0e-9)
            I = np.eye(6, dtype=float)
            P_upd = nearest_spd((I - K @ H) @ P_pred @ (I - K @ H).T + K @ Rk @ K.T)
            entropy_reduction += max(0.0, _safe_slogdet_pos(P_pred) - _safe_slogdet_pos(P_upd))
            fim_trace_sum += float(np.trace(fim_pos))
            P = P_upd
            visible_update_steps += 1
        else:
            P = P_pred

        trace_pos_sum += float(np.trace(P[:3, :3]))

    terminal_trace_pos = float(np.trace(P[:3, :3]))
    return {
        "trace_pos_sum": float(trace_pos_sum),
        "terminal_trace_pos": terminal_trace_pos,
        "entropy_reduction": float(entropy_reduction),
        "fim_trace_sum": float(fim_trace_sum),
        "visible_update_steps": int(visible_update_steps),
    }


def _compute_information_metric(info_summary: dict, p_diag: np.ndarray) -> float:
    p_diag = np.asarray(p_diag, dtype=float).reshape(-1)
    base_trace = float(np.sum(np.maximum(p_diag[:3], 1.0e-12))) if p_diag.size >= 3 else float(50.0 ** 2 + 80.0 ** 2 + 50.0 ** 2)
    horizon_steps = max(int(round(PLAN_HORIZON_SEC / PLAN_DT)), 1)

    trace_avg_ratio = float(info_summary["trace_pos_sum"]) / max(base_trace * horizon_steps, INFO_EPS)
    terminal_trace_ratio = float(info_summary["terminal_trace_pos"]) / max(base_trace, INFO_EPS)
    entropy_reduction = float(info_summary["entropy_reduction"])
    fim_trace_avg = float(info_summary["fim_trace_sum"]) / max(horizon_steps, 1)

    if INFO_OBJECTIVE == "trace":
        return -INFO_TRACE_WEIGHT * trace_avg_ratio
    if INFO_OBJECTIVE == "fim":
        return INFO_FIM_WEIGHT * fim_trace_avg
    if INFO_OBJECTIVE == "entropy":
        return INFO_ENTROPY_GAIN_WEIGHT * entropy_reduction

    return (
        -INFO_TRACE_WEIGHT * trace_avg_ratio
        -INFO_TERMINAL_TRACE_WEIGHT * terminal_trace_ratio
        +INFO_ENTROPY_GAIN_WEIGHT * entropy_reduction
        +INFO_FIM_WEIGHT * fim_trace_avg
    )


def evaluate_schedule_exact(dv_seq: np.ndarray, anchors: list[tuple[str, np.ndarray]], dt: float, n_orbit: float, p_diag: np.ndarray | None = None) -> EvalResult:
    cache_key = _eval_cache_key(dv_seq, anchors, dt, n_orbit, p_diag)
    cached = _EVAL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    all_margins = []
    margins_by_anchor: dict[str, list[float]] = {}
    states_by_anchor: dict[str, list[np.ndarray]] = {}
    all_ranges = []
    terminal_ranges = []
    all_future_ranges = []
    terminal_future_ranges = []
    worst_risk_rank_pred = -999
    for name, x0 in anchors:
        states = propagate_with_schedule(x0, dv_seq, dt, n_orbit)
        states_by_anchor[name] = states
        margins = pixel_safe_margins(states, SAFE_RATIO)
        margins_by_anchor[name] = [float(v) for v in margins]
        all_margins.extend(margins.tolist())
        if len(states) > 0:
            states_arr = np.asarray(states, dtype=float)
            ranges = np.linalg.norm(states_arr[:, :3], axis=1)
            future_ranges = _future_distance_profile(ranges)
            all_ranges.extend(ranges.tolist())
            all_future_ranges.extend(future_ranges.tolist())
            terminal_ranges.append(float(ranges[-1]))
            terminal_future_ranges.append(float(future_ranges[-1]))
            worst_risk_rank_pred = max(worst_risk_rank_pred, _risk_rank_from_distance_m(float(np.min(future_ranges))))

    arr = np.asarray(all_margins, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Empty margin array in schedule evaluation.")

    neg = arr[arr < 0.0]
    violation_count = int(neg.size)
    negative_margin_sum = float(-np.sum(neg)) if neg.size > 0 else 0.0
    min_margin = float(np.min(arr))
    terminal_margin = float(min(m[-1] for m in margins_by_anchor.values()))
    total_dv = float(np.sum(np.linalg.norm(dv_seq, axis=1)))
    min_relative_distance = float(np.min(np.asarray(all_ranges, dtype=float))) if len(all_ranges) > 0 else float("nan")
    terminal_relative_distance = float(np.min(np.asarray(terminal_ranges, dtype=float))) if len(terminal_ranges) > 0 else float("nan")
    min_future_min_distance = float(np.min(np.asarray(all_future_ranges, dtype=float))) if len(all_future_ranges) > 0 else float("nan")
    terminal_future_min_distance = float(np.min(np.asarray(terminal_future_ranges, dtype=float))) if len(terminal_future_ranges) > 0 else float("nan")

    info_metric = 0.0
    trace_pos_sum = float("nan")
    terminal_trace_pos = float("nan")
    entropy_reduction = 0.0
    fim_trace_sum = 0.0
    visible_update_steps = 0

    if p_diag is not None:
        estimate_states = states_by_anchor.get("estimate")
        if estimate_states is None and len(states_by_anchor) > 0:
            estimate_states = next(iter(states_by_anchor.values()))
        if estimate_states is not None:
            info_summary = _rollout_information_metrics(estimate_states, p_diag, dt, n_orbit)
            info_metric = _compute_information_metric(info_summary, p_diag)
            trace_pos_sum = float(info_summary["trace_pos_sum"])
            terminal_trace_pos = float(info_summary["terminal_trace_pos"])
            entropy_reduction = float(info_summary["entropy_reduction"])
            fim_trace_sum = float(info_summary["fim_trace_sum"])
            visible_update_steps = int(info_summary["visible_update_steps"])

    geom_future_reward = (min_future_min_distance / 1000.0) if np.isfinite(min_future_min_distance) else 0.0
    geom_risk_penalty = float(max(int(worst_risk_rank_pred), 0))

    metric = (
        -200.0 * violation_count
        - 1.5 * negative_margin_sum
        + 1.0 * min_margin
        + 0.35 * terminal_margin
        + info_metric
        + GEOM_FUTURE_DIST_WEIGHT * geom_future_reward
        - GEOM_RISK_RANK_WEIGHT * geom_risk_penalty
        - 5.0 * total_dv
    )
    result = EvalResult(
        metric=metric,
        min_safe_margin_px=min_margin,
        terminal_margin_px=terminal_margin,
        violation_count=violation_count,
        negative_margin_sum_px=negative_margin_sum,
        total_dv_mag_mps=total_dv,
        margins_by_anchor=margins_by_anchor,
        info_metric=float(info_metric),
        trace_pos_sum=float(trace_pos_sum),
        terminal_trace_pos=float(terminal_trace_pos),
        entropy_reduction=float(entropy_reduction),
        fim_trace_sum=float(fim_trace_sum),
        visible_update_steps=int(visible_update_steps),
        min_relative_distance_m=float(min_relative_distance),
        terminal_relative_distance_m=float(terminal_relative_distance),
        min_future_min_distance_m=float(min_future_min_distance),
        terminal_future_min_distance_m=float(terminal_future_min_distance),
        worst_risk_rank_pred=int(worst_risk_rank_pred),
    )
    _EVAL_CACHE[cache_key] = result
    return result

def build_candidate_directions(
    seq: np.ndarray | None = None,
    k: int | None = None,
    anchors: list[tuple[str, np.ndarray]] | None = None,
    dt: float | None = None,
    n_orbit: float | None = None,
    p_diag: np.ndarray | None = None,
) -> list[np.ndarray]:
    dirs = [d.copy() for d in STATIC_CANDIDATE_DIRECTIONS]

    if seq is not None and k is not None and anchors is not None and dt is not None and n_orbit is not None:
        g = np.zeros(3, dtype=float)
        for i in range(3):
            seq_p = np.asarray(seq, dtype=float).copy()
            seq_m = np.asarray(seq, dtype=float).copy()
            seq_p[k, i] += FD_EPS_DIR
            seq_m[k, i] -= FD_EPS_DIR
            m_p = evaluate_schedule_exact(seq_p, anchors, dt, n_orbit, p_diag=p_diag).metric
            m_m = evaluate_schedule_exact(seq_m, anchors, dt, n_orbit, p_diag=p_diag).metric
            g[i] = (m_p - m_m) / (2.0 * FD_EPS_DIR)
        ng = float(np.linalg.norm(g))
        if ng > 1.0e-10:
            dirs.append(g / ng)
            dirs.append(-g / ng)

    out: list[np.ndarray] = []
    seen = set()
    for d in dirs:
        nrm = float(np.linalg.norm(d))
        if nrm < 1.0e-12:
            continue
        u = np.asarray(d, dtype=float) / nrm
        key = tuple(np.round(u, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def _rank_steps_from_eval(ev: EvalResult, active_steps: int) -> list[int]:
    if not ev.margins_by_anchor:
        return list(range(active_steps))
    per_step = []
    for k in range(active_steps):
        vals = []
        for margins in ev.margins_by_anchor.values():
            if k < len(margins):
                vals.append(float(margins[k]))
        score = min(vals) if vals else float("inf")
        per_step.append((score, k))
    per_step.sort(key=lambda item: item[0])
    return [k for _, k in per_step]


def _seed_direction_bank(anchors: list[tuple[str, np.ndarray]], total_dv_budget: float, dt: float, n_orbit: float, K: int, p_diag: np.ndarray | None = None) -> list[np.ndarray]:
    dirs = build_candidate_directions()
    early_steps = min(K, MULTISTEP_SEED_STEPS)
    scored = []
    for k in range(early_steps):
        for frac in (1.0, 0.75, 0.50, 0.25):
            mag = total_dv_budget * frac
            for d in dirs:
                cand = np.zeros((K, 3), dtype=float)
                cand[k, :] = d * mag
                ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
                scored.append((float(ev.metric), d.copy()))
    scored.sort(key=lambda item: item[0], reverse=True)

    bank: list[np.ndarray] = []
    seen = set()
    for _, d in scored:
        u = np.asarray(d, dtype=float) / max(float(np.linalg.norm(d)), 1.0e-12)
        key = tuple(np.round(u, 6))
        if key in seen:
            continue
        seen.add(key)
        bank.append(u)
        if len(bank) >= MULTISTEP_SEED_TOPK:
            break
    if not bank:
        bank = dirs[:min(len(dirs), MULTISTEP_SEED_TOPK)]
    return bank


def _generate_seed_bank(anchors: list[tuple[str, np.ndarray]], total_dv_budget: float, dt: float, n_orbit: float, K: int, p_diag: np.ndarray | None = None) -> list[np.ndarray]:
    seq0 = np.zeros((K, 3), dtype=float)
    seeds: list[np.ndarray] = [seq0]
    best = evaluate_schedule_exact(seq0, anchors, dt, n_orbit, p_diag=p_diag)
    best_seq = seq0.copy()
    early_steps = min(K, SEED_EARLY_STEPS)
    seed_steps = min(K, MULTISTEP_SEED_STEPS)
    dir_bank = _seed_direction_bank(anchors, total_dv_budget, dt, n_orbit, K, p_diag=p_diag)

    # One-step seeds (kept for compatibility, but now only one part of the seed bank)
    for k in range(early_steps):
        for frac in (1.0, 0.75, 0.50, 0.25):
            mag = total_dv_budget * frac
            for d in dir_bank:
                cand = np.zeros((K, 3), dtype=float)
                cand[k, :] = d * mag
                ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
                seeds.append(cand)
                if ev.metric > best.metric:
                    best = ev
                    best_seq = cand.copy()

    # Knife 2: add true multi-step seeds so the planner does not collapse into
    # "all budget on step 0" as the only meaningful initial guess.
    split_patterns = [
        (0.60, 0.40),
        (0.50, 0.50),
        (0.70, 0.30),
        (0.40, 0.35, 0.25),
    ]
    candidate_steps = list(range(seed_steps))
    for i in range(len(candidate_steps)):
        for j in range(i + 1, len(candidate_steps)):
            k1, k2 = candidate_steps[i], candidate_steps[j]
            for d1 in dir_bank[:max(3, min(5, len(dir_bank)))]:
                for d2 in dir_bank[:max(3, min(5, len(dir_bank)))]:
                    for split in split_patterns[:3]:
                        cand = np.zeros((K, 3), dtype=float)
                        cand[k1, :] = d1 * (total_dv_budget * split[0])
                        cand[k2, :] = d2 * (total_dv_budget * split[1])
                        seeds.append(cand)
    if seed_steps >= 3:
        k1, k2, k3 = 0, 1, 2
        for d1 in dir_bank[:4]:
            for d2 in dir_bank[:4]:
                for d3 in dir_bank[:4]:
                    cand = np.zeros((K, 3), dtype=float)
                    cand[k1, :] = d1 * (total_dv_budget * split_patterns[3][0])
                    cand[k2, :] = d2 * (total_dv_budget * split_patterns[3][1])
                    cand[k3, :] = d3 * (total_dv_budget * split_patterns[3][2])
                    seeds.append(cand)

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(RANDOM_MULTISTART_COUNT):
        cand = np.zeros((K, 3), dtype=float)
        nnz = int(rng.integers(2, RANDOM_NONZERO_STEPS_MAX + 1))
        steps = rng.choice(np.arange(seed_steps), size=nnz, replace=False)
        weights = rng.dirichlet(np.ones(nnz))
        for step, w in zip(steps, weights):
            d = dir_bank[int(rng.integers(0, len(dir_bank)))]
            cand[int(step), :] = d * (total_dv_budget * float(w))
        seeds.append(cand)

    # De-duplicate and rank by metric.
    uniq = []
    seen = set()
    for cand in seeds:
        key = tuple(np.round(cand.reshape(-1), 6))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(cand)

    scored = []
    for cand in uniq:
        ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
        scored.append((float(ev.metric), cand.copy(), ev))
    scored.sort(key=lambda item: item[0], reverse=True)
    top = [cand for _, cand, _ in scored[:max(6, MULTISTEP_SEED_TOPK)]]
    if not any(np.allclose(best_seq, cand) for cand in top):
        top.insert(0, best_seq)
    return top


def initial_schedule_guess(anchors: list[tuple[str, np.ndarray]], total_dv_budget: float, dt: float, n_orbit: float, K: int, p_diag: np.ndarray | None = None) -> np.ndarray:
    return _generate_seed_bank(anchors, total_dv_budget, dt, n_orbit, K, p_diag=p_diag)[0].copy()


def total_dv_used(seq: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(seq, axis=1)))


def _seq_total_and_norms(seq: np.ndarray) -> tuple[np.ndarray, float]:
    norms = np.linalg.norm(np.asarray(seq, dtype=float), axis=1)
    return norms, float(np.sum(norms))


def _split_dominant_impulse(seq: np.ndarray, best: EvalResult, anchors: list[tuple[str, np.ndarray]], total_dv_budget: float, dt: float, n_orbit: float, active_steps: int, p_diag: np.ndarray | None = None) -> tuple[np.ndarray, EvalResult, bool]:
    norms, seq_total = _seq_total_and_norms(seq)
    if norms.size == 0:
        return seq, best, False
    k0 = int(np.argmax(norms[:active_steps]))
    dominant_mag = float(norms[k0])
    if dominant_mag <= max(0.35 * total_dv_budget, 1.0e-6):
        return seq, best, False

    dir0 = seq[k0] / max(np.linalg.norm(seq[k0]), 1.0e-12)
    ranked_steps = _rank_steps_from_eval(best, active_steps)
    target_steps = []
    for kk in ranked_steps:
        if kk != k0 and kk < active_steps:
            target_steps.append(kk)
    for look in range(1, DOMINANT_SPLIT_MAX_LOOKAHEAD + 1):
        kk = k0 + look
        if kk < active_steps and kk not in target_steps:
            target_steps.append(kk)

    improved = False
    current_seq = seq.copy()
    current_best = best
    current_norms = norms.copy()
    current_total = seq_total
    for j in target_steps[:6]:
        dirs_j = build_candidate_directions(seq=current_seq, k=j, anchors=anchors, dt=dt, n_orbit=n_orbit, p_diag=p_diag)[:8]
        for split in (0.80, 0.70, 0.60, 0.50):
            mag0 = dominant_mag * split
            magj = dominant_mag * (1.0 - split)
            new_total = current_total - current_norms[k0] - current_norms[j] + mag0 + magj
            if new_total > total_dv_budget + 1.0e-9:
                continue
            for dj in dirs_j:
                cand = current_seq.copy()
                cand[k0, :] = dir0 * mag0
                cand[j, :] = dj * magj
                ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
                if ev.metric > current_best.metric + 1.0e-9:
                    current_seq = cand
                    current_best = ev
                    current_norms, current_total = _seq_total_and_norms(current_seq)
                    dominant_mag = float(current_norms[k0])
                    dir0 = current_seq[k0] / max(np.linalg.norm(current_seq[k0]), 1.0e-12)
                    improved = True
    return current_seq, current_best, improved


def _refine_from_seed(seq0: np.ndarray, anchors: list[tuple[str, np.ndarray]], total_dv_budget: float, dt: float, n_orbit: float, p_diag: np.ndarray | None = None) -> tuple[np.ndarray, EvalResult, list[dict]]:
    K = int(round(PLAN_HORIZON_SEC / dt))
    active_steps = min(K, SEARCH_ACTIVE_STEPS)
    seq = seq0.copy()
    best = evaluate_schedule_exact(seq, anchors, dt, n_orbit, p_diag=p_diag)
    history = []

    for outer in range(OUTER_SEARCH_PASSES):
        improved_any = False
        step_order = _rank_steps_from_eval(best, active_steps)
        current_norms, current_total = _seq_total_and_norms(seq)
        nonzero_steps = [k for k in range(active_steps) if current_norms[k] > 1.0e-10]
        for k in nonzero_steps:
            if k not in step_order:
                step_order.insert(0, k)

        for k in step_order:
            current_norms, current_total = _seq_total_and_norms(seq)
            other_budget = current_total - float(current_norms[k])
            max_here = max(total_dv_budget - other_budget, 0.0)
            mags = np.unique(np.concatenate([MAG_FRACTIONS * max_here, [float(current_norms[k])]]))
            dirs = build_candidate_directions(seq=seq, k=k, anchors=anchors, dt=dt, n_orbit=n_orbit, p_diag=p_diag)
            local_best_seq = seq.copy()
            local_best = best

            for mag in mags:
                if mag <= 1.0e-12:
                    cand = seq.copy()
                    cand[k, :] = 0.0
                    ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
                    if ev.metric > local_best.metric + 1.0e-9:
                        local_best = ev
                        local_best_seq = cand
                    continue

                new_total = other_budget + float(mag)
                if new_total > total_dv_budget + 1.0e-9:
                    continue
                for d in dirs:
                    cand = seq.copy()
                    cand[k, :] = d * mag
                    ev = evaluate_schedule_exact(cand, anchors, dt, n_orbit, p_diag=p_diag)
                    if ev.metric > local_best.metric + 1.0e-9:
                        local_best = ev
                        local_best_seq = cand

            if not np.allclose(local_best_seq, seq):
                seq = local_best_seq
                best = local_best
                improved_any = True

        for _ in range(DOMINANT_SPLIT_PASSES):
            seq2, best2, improved2 = _split_dominant_impulse(seq, best, anchors, total_dv_budget, dt, n_orbit, active_steps, p_diag=p_diag)
            if improved2:
                seq = seq2
                best = best2
                improved_any = True
            else:
                break

        history.append({
            "outer_iter": outer + 1,
            "metric": float(best.metric),
            "min_safe_margin_px": float(best.min_safe_margin_px),
            "terminal_margin_px": float(best.terminal_margin_px),
            "violation_count": int(best.violation_count),
            "negative_margin_sum_px": float(best.negative_margin_sum_px),
            "total_dv_mag_mps": float(best.total_dv_mag_mps),
        })
        if not improved_any:
            break

    return seq, best, history


def _refine_from_seed_worker(
    seed: np.ndarray,
    anchors: list[tuple[str, np.ndarray]],
    total_dv_budget: float,
    dt: float,
    n_orbit: float,
    p_diag: np.ndarray | None,
    search_profile: dict[str, float | int],
) -> tuple[np.ndarray, EvalResult, list[dict]]:
    with _temporary_search_profile(search_profile):
        return _refine_from_seed(np.asarray(seed, dtype=float), anchors, total_dv_budget, dt, n_orbit, p_diag=p_diag)


def optimize_schedule_direct_shooting(
    anchors: list[tuple[str, np.ndarray]],
    total_dv_budget: float,
    dt: float,
    n_orbit: float,
    p_diag: np.ndarray | None = None,
    *,
    enable_seed_parallel: bool = True,
) -> tuple[np.ndarray, dict]:
    K = int(round(PLAN_HORIZON_SEC / dt))
    seed_bank = _generate_seed_bank(anchors, total_dv_budget, dt, n_orbit, K, p_diag=p_diag)
    best_seq = None
    best_ev = None
    best_history: list[dict] = []

    use_seed_parallel = (
        bool(PARALLEL_SEED_SOLVE_ENABLE)
        and bool(enable_seed_parallel)
        and (len(seed_bank) >= int(PARALLEL_SEED_MIN_TASKS))
    )
    if use_seed_parallel:
        profile_snapshot = _current_search_profile_snapshot()
        max_workers = max(1, min(int(PARALLEL_SEED_MAX_WORKERS), len(seed_bank)))
        try:
            with ProcessPoolExecutor(**_process_pool_kwargs(max_workers)) as executor:
                result_iter = executor.map(
                    _refine_from_seed_worker,
                    seed_bank,
                    repeat(anchors),
                    repeat(float(total_dv_budget)),
                    repeat(float(dt)),
                    repeat(float(n_orbit)),
                    repeat(p_diag),
                    repeat(profile_snapshot),
                )
                for seq, ev, history in result_iter:
                    if best_ev is None or ev.metric > best_ev.metric + 1.0e-9:
                        best_seq = seq.copy()
                        best_ev = ev
                        best_history = history
        except Exception as exc:
            print(f"[WARN] seed-level process parallel failed ({type(exc).__name__}: {exc}); fallback to sequential")
            use_seed_parallel = False

    if not use_seed_parallel:
        for seed in seed_bank:
            seq, ev, history = _refine_from_seed(seed, anchors, total_dv_budget, dt, n_orbit, p_diag=p_diag)
            if best_ev is None or ev.metric > best_ev.metric + 1.0e-9:
                best_seq = seq.copy()
                best_ev = ev
                best_history = history

    assert best_seq is not None and best_ev is not None
    info = {
        "planner": "exact_direct_shooting",
        "outer_passes": len(best_history),
        "num_multistarts": len(seed_bank),
        "history": best_history,
        "min_safe_margin_px": float(best_ev.min_safe_margin_px),
        "terminal_margin_px": float(best_ev.terminal_margin_px),
        "violation_count_pred": int(best_ev.violation_count),
        "negative_margin_sum_px_pred": float(best_ev.negative_margin_sum_px),
        "total_dv_mag_mps": float(best_ev.total_dv_mag_mps),
        "margins_by_anchor": best_ev.margins_by_anchor,
        "info_metric": float(best_ev.info_metric),
        "trace_pos_sum": float(best_ev.trace_pos_sum),
        "terminal_trace_pos": float(best_ev.terminal_trace_pos),
        "entropy_reduction": float(best_ev.entropy_reduction),
        "fim_trace_sum": float(best_ev.fim_trace_sum),
        "visible_update_steps": int(best_ev.visible_update_steps),
        "min_relative_distance_m": float(best_ev.min_relative_distance_m),
        "terminal_relative_distance_m": float(best_ev.terminal_relative_distance_m),
        "min_future_min_distance_m": float(best_ev.min_future_min_distance_m),
        "terminal_future_min_distance_m": float(best_ev.terminal_future_min_distance_m),
        "worst_risk_rank_pred": int(best_ev.worst_risk_rank_pred),
        "info_objective": INFO_OBJECTIVE,
    }
    return best_seq, info


def solve_nominal_schedule(
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    total_dv_budget: float,
    *,
    enable_seed_parallel: bool = True,
) -> tuple[np.ndarray, dict]:
    anchors = [("estimate", x_est.copy())]
    return optimize_schedule_direct_shooting(
        anchors,
        total_dv_budget,
        PLAN_DT,
        n_orbit,
        p_diag=p_diag,
        enable_seed_parallel=enable_seed_parallel,
    )


def solve_robust_schedule_exp3(
    trig: pd.Series,
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    total_dv_budget: float,
    *,
    enable_seed_parallel: bool = True,
) -> tuple[np.ndarray, dict]:
    anchors = build_anchor_states(trig, x_est, p_diag, robust=True)
    return optimize_schedule_direct_shooting(
        anchors,
        total_dv_budget,
        PLAN_DT,
        n_orbit,
        p_diag=p_diag,
        enable_seed_parallel=enable_seed_parallel,
    )


def _solve_strategy_worker(
    strategy: str,
    trig_payload: dict,
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    total_dv_budget: float,
    search_profile: dict[str, float | int],
    enable_seed_parallel: bool,
) -> tuple[str, np.ndarray, dict]:
    trig = pd.Series(trig_payload)
    with _temporary_search_profile(search_profile):
        if str(strategy) == "nominal":
            schedule, info = solve_nominal_schedule(
                x_est,
                p_diag,
                n_orbit,
                total_dv_budget,
                enable_seed_parallel=enable_seed_parallel,
            )
            return "nominal", schedule, info
        schedule, info = solve_robust_schedule_exp3(
            trig,
            x_est,
            p_diag,
            n_orbit,
            total_dv_budget,
            enable_seed_parallel=enable_seed_parallel,
        )
        return "robust", schedule, info


def _evaluate_zero_baseline(trig: pd.Series, x_est: np.ndarray, p_diag: np.ndarray, n_orbit: float, robust: bool) -> tuple[np.ndarray, dict]:
    K = int(round(PLAN_HORIZON_SEC / PLAN_DT))
    seq = np.zeros((K, 3), dtype=float)
    anchors = build_anchor_states(trig, x_est, p_diag, robust=robust)
    ev = evaluate_schedule_exact(seq, anchors, PLAN_DT, n_orbit, p_diag=p_diag)
    info = {
        "planner": "zero_baseline",
        "outer_passes": 0,
        "history": [],
        "min_safe_margin_px": float(ev.min_safe_margin_px),
        "terminal_margin_px": float(ev.terminal_margin_px),
        "violation_count_pred": int(ev.violation_count),
        "negative_margin_sum_px_pred": float(ev.negative_margin_sum_px),
        "total_dv_mag_mps": float(ev.total_dv_mag_mps),
        "margins_by_anchor": ev.margins_by_anchor,
        "info_metric": float(ev.info_metric),
        "trace_pos_sum": float(ev.trace_pos_sum),
        "terminal_trace_pos": float(ev.terminal_trace_pos),
        "entropy_reduction": float(ev.entropy_reduction),
        "fim_trace_sum": float(ev.fim_trace_sum),
        "visible_update_steps": int(ev.visible_update_steps),
        "min_relative_distance_m": float(ev.min_relative_distance_m),
        "terminal_relative_distance_m": float(ev.terminal_relative_distance_m),
        "min_future_min_distance_m": float(ev.min_future_min_distance_m),
        "terminal_future_min_distance_m": float(ev.terminal_future_min_distance_m),
        "worst_risk_rank_pred": int(ev.worst_risk_rank_pred),
        "info_objective": INFO_OBJECTIVE,
    }
    return seq, info


def _is_plan_feasible(info: dict) -> bool:
    min_margin = float(info.get("min_safe_margin_px", float("nan")))
    viol = _safe_int(info.get("violation_count_pred", -1))
    return np.isfinite(min_margin) and (min_margin > 0.0) and (viol == 0)


def _clip_signed(value: float, scale: float) -> float:
    if (not np.isfinite(value)) or (scale <= 1.0e-12):
        return -1.0
    return float(np.clip(value / scale, -1.0, 1.0))


def _clip_penalty(value: float, scale: float) -> float:
    if (not np.isfinite(value)) or (scale <= 1.0e-12):
        return 1.0
    return float(np.clip(max(value, 0.0) / scale, 0.0, 1.0))


def _acceptance_profile_to_dict(profile: AcceptanceProfile = ACCEPTANCE_PROFILE) -> dict[str, float | int]:
    return {
        "min_info_delta": float(profile.min_info_delta),
        "geom_min_margin_gain_vs_zero_px": float(profile.geom_min_margin_gain_vs_zero_px),
        "geom_min_terminal_margin_gain_vs_zero_px": float(profile.geom_min_terminal_margin_gain_vs_zero_px),
        "geom_max_neg_margin_sum_delta_vs_zero_px": float(profile.geom_max_neg_margin_sum_delta_vs_zero_px),
        "hard_veto_max_violation_delta_vs_zero": int(profile.hard_veto_max_violation_delta_vs_zero),
        "hard_veto_min_margin_px": float(profile.hard_veto_min_margin_px),
        "hard_veto_require_structurally_sane": int(bool(profile.hard_veto_require_structurally_sane)),
        "hard_veto_future_dist_non_regression_tol_m": float(profile.hard_veto_future_dist_non_regression_tol_m),
        "hard_veto_future_dist_gate_for_caution_danger_only": int(
            bool(profile.hard_veto_future_dist_gate_for_caution_danger_only)
        ),
        "accepted_score_thr": float(profile.accepted_score_thr),
        "borderline_score_thr": float(profile.borderline_score_thr),
        "score_margin_scale_px": float(profile.score_margin_scale_px),
        "score_terminal_margin_scale_px": float(profile.score_terminal_margin_scale_px),
        "score_info_delta_scale": float(profile.score_info_delta_scale),
        "score_future_dist_gain_scale_m": float(profile.score_future_dist_gain_scale_m),
        "score_violation_delta_scale": float(profile.score_violation_delta_scale),
        "score_neg_margin_sum_delta_scale_px": float(profile.score_neg_margin_sum_delta_scale_px),
        "score_trace_penalty_scale": float(profile.score_trace_penalty_scale),
        "score_terminal_trace_penalty_scale": float(profile.score_terminal_trace_penalty_scale),
        "w_margin": float(profile.w_margin),
        "w_terminal_margin": float(profile.w_terminal_margin),
        "w_info_delta": float(profile.w_info_delta),
        "w_future_distance": float(profile.w_future_distance),
        "w_structural": float(profile.w_structural),
        "w_violation_delta": float(profile.w_violation_delta),
        "w_neg_margin_delta": float(profile.w_neg_margin_delta),
        "w_trace_penalty": float(profile.w_trace_penalty),
        "w_terminal_trace_penalty": float(profile.w_terminal_trace_penalty),
    }


def _acceptance_core_features(name: str, seq: np.ndarray, info: dict, zero_info: dict) -> dict[str, float | int | bool | str]:
    min_margin = float(info.get("min_safe_margin_px", float("nan")))
    terminal_margin = float(info.get("terminal_margin_px", float("nan")))
    violation_count = _safe_int(info.get("violation_count_pred", 10**9))
    negative_margin_sum = float(info.get("negative_margin_sum_px_pred", float("inf")))

    zero_min_margin = float(zero_info.get("min_safe_margin_px", float("nan")))
    zero_terminal_margin = float(zero_info.get("terminal_margin_px", float("nan")))
    zero_violation_count = _safe_int(zero_info.get("violation_count_pred", 10**9))
    zero_negative_margin_sum = float(zero_info.get("negative_margin_sum_px_pred", float("inf")))

    trace_val = float(info.get("trace_pos_sum", float("nan")))
    trace_zero = float(zero_info.get("trace_pos_sum", float("nan")))
    term_val = float(info.get("terminal_trace_pos", float("nan")))
    term_zero = float(zero_info.get("terminal_trace_pos", float("nan")))
    info_val = float(info.get("info_metric", float("nan")))
    info_zero = float(zero_info.get("info_metric", float("nan")))

    trace_ratio = (
        float(trace_val / trace_zero)
        if np.isfinite(trace_val) and np.isfinite(trace_zero) and abs(trace_zero) > 1.0e-12
        else float("inf")
    )
    terminal_trace_ratio = (
        float(term_val / term_zero)
        if np.isfinite(term_val) and np.isfinite(term_zero) and abs(term_zero) > 1.0e-12
        else float("inf")
    )
    info_delta = float(info_val - info_zero) if np.isfinite(info_val) and np.isfinite(info_zero) else float("-inf")

    margin_gain_vs_zero = (
        float(min_margin - zero_min_margin)
        if np.isfinite(min_margin) and np.isfinite(zero_min_margin)
        else float("nan")
    )
    terminal_margin_gain_vs_zero = (
        float(terminal_margin - zero_terminal_margin)
        if np.isfinite(terminal_margin) and np.isfinite(zero_terminal_margin)
        else float("nan")
    )
    violation_delta_vs_zero = (
        int(violation_count - zero_violation_count)
        if np.isfinite(float(violation_count)) and np.isfinite(float(zero_violation_count))
        else int(10**9)
    )
    negative_margin_sum_delta_vs_zero = (
        float(negative_margin_sum - zero_negative_margin_sum)
        if np.isfinite(negative_margin_sum) and np.isfinite(zero_negative_margin_sum)
        else float("inf")
    )

    min_future_min_distance = float(info.get("min_future_min_distance_m", float("nan")))
    zero_min_future_min_distance = float(zero_info.get("min_future_min_distance_m", float("nan")))
    future_distance_gain_vs_zero = (
        float(min_future_min_distance - zero_min_future_min_distance)
        if np.isfinite(min_future_min_distance) and np.isfinite(zero_min_future_min_distance)
        else float("nan")
    )
    zero_worst_risk_rank_pred = _safe_int(zero_info.get("worst_risk_rank_pred", -999))
    worst_risk_rank_pred = _safe_int(info.get("worst_risk_rank_pred", -999))

    structurally_sane = bool(_sequence_is_structurally_sane(np.asarray(seq, dtype=float))) and (str(name) != "zero")
    safe_ok = bool(_is_plan_feasible(info))

    return {
        "name": str(name),
        "safe_ok": bool(safe_ok),
        "structurally_sane": bool(structurally_sane),
        "min_margin": float(min_margin),
        "terminal_margin": float(terminal_margin),
        "violation_count": int(violation_count),
        "negative_margin_sum": float(negative_margin_sum),
        "margin_gain_vs_zero": float(margin_gain_vs_zero),
        "terminal_margin_gain_vs_zero": float(terminal_margin_gain_vs_zero),
        "violation_delta_vs_zero": int(violation_delta_vs_zero),
        "neg_margin_sum_delta_vs_zero": float(negative_margin_sum_delta_vs_zero),
        "trace_ratio_vs_zero": float(trace_ratio),
        "terminal_trace_ratio_vs_zero": float(terminal_trace_ratio),
        "info_delta_vs_zero": float(info_delta),
        "future_distance_gain_vs_zero": float(future_distance_gain_vs_zero),
        "min_future_min_distance": float(min_future_min_distance),
        "zero_min_future_min_distance": float(zero_min_future_min_distance),
        "worst_risk_rank_pred": int(worst_risk_rank_pred),
        "zero_worst_risk_rank_pred": int(zero_worst_risk_rank_pred),
    }


def legacy_acceptance(candidate: dict, zero_eval: dict, profile: dict) -> dict[str, float | int | bool | str]:
    del zero_eval
    zero_worst_risk_rank_pred = _safe_int(candidate.get("zero_worst_risk_rank_pred", -999))
    max_trace_ratio, max_terminal_trace_ratio = _strict_accept_trace_limits(zero_worst_risk_rank_pred)

    information_ok = (
        np.isfinite(float(candidate.get("trace_ratio_vs_zero", float("inf"))))
        and np.isfinite(float(candidate.get("terminal_trace_ratio_vs_zero", float("inf"))))
        and np.isfinite(float(candidate.get("info_delta_vs_zero", float("-inf"))))
        and (float(candidate.get("trace_ratio_vs_zero", float("inf"))) <= max_trace_ratio)
        and (float(candidate.get("terminal_trace_ratio_vs_zero", float("inf"))) <= max_terminal_trace_ratio)
        and (float(candidate.get("info_delta_vs_zero", float("-inf"))) >= float(profile["min_info_delta"]))
    )

    geometry_non_degenerate_ok = (
        np.isfinite(float(candidate.get("margin_gain_vs_zero", float("nan"))))
        and np.isfinite(float(candidate.get("terminal_margin_gain_vs_zero", float("nan"))))
        and np.isfinite(float(candidate.get("neg_margin_sum_delta_vs_zero", float("inf"))))
        and (int(candidate.get("violation_delta_vs_zero", 10**9)) <= int(profile["hard_veto_max_violation_delta_vs_zero"]))
        and (float(candidate.get("margin_gain_vs_zero", float("nan"))) >= float(profile["geom_min_margin_gain_vs_zero_px"]))
        and (float(candidate.get("terminal_margin_gain_vs_zero", float("nan"))) >= float(profile["geom_min_terminal_margin_gain_vs_zero_px"]))
        and (float(candidate.get("neg_margin_sum_delta_vs_zero", float("inf"))) <= float(profile["geom_max_neg_margin_sum_delta_vs_zero_px"]))
    )

    future_distance_gate_required = int(
        (not bool(profile["hard_veto_future_dist_gate_for_caution_danger_only"]))
        or (zero_worst_risk_rank_pred >= 1)
    )
    future_distance_non_regression = bool(
        np.isfinite(float(candidate.get("min_future_min_distance", float("nan"))))
        and np.isfinite(float(candidate.get("zero_min_future_min_distance", float("nan"))))
        and (
            float(candidate.get("min_future_min_distance", float("nan")))
            >= float(candidate.get("zero_min_future_min_distance", float("nan")))
            - float(profile["hard_veto_future_dist_non_regression_tol_m"])
        )
    )
    if future_distance_gate_required == 0:
        future_distance_non_regression = True

    status = "rejected"
    if (
        bool(candidate.get("safe_ok", False))
        and bool(candidate.get("structurally_sane", False))
        and information_ok
        and geometry_non_degenerate_ok
        and future_distance_non_regression
    ):
        status = "accepted"

    return {
        "status": status,
        "information_ok": int(information_ok),
        "geometry_non_degenerate_ok": int(geometry_non_degenerate_ok),
        "future_distance_gate_required": int(future_distance_gate_required),
        "future_distance_non_regression": int(future_distance_non_regression),
    }


def acceptance_score(candidate: dict, zero_eval: dict, profile: dict) -> dict[str, float | int | bool | str]:
    del zero_eval
    hard_veto_reasons: list[str] = []

    safe_ok = bool(candidate.get("safe_ok", False))
    structurally_sane = bool(candidate.get("structurally_sane", False))
    min_margin = float(candidate.get("min_margin", float("nan")))
    violation_count = int(candidate.get("violation_count", 10**9))
    violation_delta = int(candidate.get("violation_delta_vs_zero", 10**9))
    future_distance_gain = float(candidate.get("future_distance_gain_vs_zero", float("nan")))
    zero_worst_risk_rank_pred = _safe_int(candidate.get("zero_worst_risk_rank_pred", -999))

    future_distance_gate_required = int(
        (not bool(profile["hard_veto_future_dist_gate_for_caution_danger_only"]))
        or (zero_worst_risk_rank_pred >= 1)
    )
    future_distance_non_regression = bool(
        np.isfinite(future_distance_gain)
        and (future_distance_gain >= -float(profile["hard_veto_future_dist_non_regression_tol_m"]))
    )
    if future_distance_gate_required == 0:
        future_distance_non_regression = True

    if (not safe_ok) or (violation_count > 0) or (violation_delta > int(profile["hard_veto_max_violation_delta_vs_zero"])):
        hard_veto_reasons.append("violation")
    if (not np.isfinite(min_margin)) or (min_margin <= float(profile["hard_veto_min_margin_px"])):
        hard_veto_reasons.append("margin_non_positive")
    if future_distance_gate_required == 1 and (not future_distance_non_regression):
        hard_veto_reasons.append("future_distance_regression")
    if bool(profile["hard_veto_require_structurally_sane"]) and (not structurally_sane):
        hard_veto_reasons.append("structural_not_sane")

    margin_term = _clip_signed(float(candidate.get("margin_gain_vs_zero", float("nan"))), float(profile["score_margin_scale_px"]))
    terminal_margin_term = _clip_signed(
        float(candidate.get("terminal_margin_gain_vs_zero", float("nan"))),
        float(profile["score_terminal_margin_scale_px"]),
    )
    info_term = _clip_signed(float(candidate.get("info_delta_vs_zero", float("nan"))), float(profile["score_info_delta_scale"]))
    future_term = _clip_signed(future_distance_gain, float(profile["score_future_dist_gain_scale_m"]))
    structural_term = 1.0 if structurally_sane else -1.0

    violation_penalty = _clip_penalty(float(violation_delta), float(profile["score_violation_delta_scale"]))
    neg_margin_penalty = _clip_penalty(
        float(candidate.get("neg_margin_sum_delta_vs_zero", float("nan"))),
        float(profile["score_neg_margin_sum_delta_scale_px"]),
    )
    trace_penalty = _clip_penalty(
        float(candidate.get("trace_ratio_vs_zero", float("inf"))) - 1.0,
        float(profile["score_trace_penalty_scale"]),
    )
    terminal_trace_penalty = _clip_penalty(
        float(candidate.get("terminal_trace_ratio_vs_zero", float("inf"))) - 1.0,
        float(profile["score_terminal_trace_penalty_scale"]),
    )

    score = (
        float(profile["w_margin"]) * margin_term
        + float(profile["w_terminal_margin"]) * terminal_margin_term
        + float(profile["w_info_delta"]) * info_term
        + float(profile["w_future_distance"]) * future_term
        + float(profile["w_structural"]) * structural_term
        - float(profile["w_violation_delta"]) * violation_penalty
        - float(profile["w_neg_margin_delta"]) * neg_margin_penalty
        - float(profile["w_trace_penalty"]) * trace_penalty
        - float(profile["w_terminal_trace_penalty"]) * terminal_trace_penalty
    )

    if hard_veto_reasons:
        status = "rejected"
    elif score >= float(profile["accepted_score_thr"]):
        status = "accepted"
    elif score >= float(profile["borderline_score_thr"]):
        status = "borderline_review"
    else:
        status = "rejected"

    return {
        "status": status,
        "score": float(score),
        "hard_veto": int(len(hard_veto_reasons) > 0),
        "hard_veto_reason": "|".join(hard_veto_reasons),
        "future_distance_gate_required": int(future_distance_gate_required),
        "future_distance_non_regression": int(future_distance_non_regression),
    }


def _acceptance_gate_diagnostics(name: str, seq: np.ndarray, info: dict, zero_info: dict) -> dict:
    profile = _acceptance_profile_to_dict(ACCEPTANCE_PROFILE)

    core = _acceptance_core_features(name, seq, info, zero_info)
    legacy = legacy_acceptance(core, zero_info, profile)
    unified = acceptance_score(core, zero_info, profile)

    mode = str(ACCEPTANCE_PROFILE.control_mode).strip().lower()
    if mode not in {"legacy", "unified"}:
        mode = "unified"
    effective = unified if mode == "unified" else legacy
    effective_status = str(effective.get("status", "rejected"))

    return {
        "acceptance_gate_safe": int(bool(core["safe_ok"])),
        "acceptance_gate_structural": int(bool(core["structurally_sane"])),
        "acceptance_gate_information": int(legacy["information_ok"]),
        "acceptance_gate_geometry_non_degenerate": int(legacy["geometry_non_degenerate_ok"]),
        "acceptance_gate_future_distance": int(legacy["future_distance_non_regression"]),
        "acceptance_gate_strong_safety_recovery": 0,
        "acceptance_single_step_recovery_override": 0,
        "acceptance_trace_ratio_vs_zero": float(core["trace_ratio_vs_zero"]),
        "acceptance_terminal_trace_ratio_vs_zero": float(core["terminal_trace_ratio_vs_zero"]),
        "acceptance_info_delta_vs_zero": float(core["info_delta_vs_zero"]),
        "acceptance_margin_gain_vs_zero_px": float(core["margin_gain_vs_zero"]),
        "acceptance_terminal_margin_gain_vs_zero_px": float(core["terminal_margin_gain_vs_zero"]),
        "acceptance_violation_delta_vs_zero": int(core["violation_delta_vs_zero"]),
        "acceptance_negative_margin_sum_delta_vs_zero_px": float(core["neg_margin_sum_delta_vs_zero"]),
        "acceptance_future_distance_gain_vs_zero_m": float(core["future_distance_gain_vs_zero"]),
        "acceptance_future_distance_non_regression": int(unified["future_distance_non_regression"]),
        "acceptance_future_distance_gate_required": int(unified["future_distance_gate_required"]),
        "acceptance_unified_score": float(unified["score"]),
        "acceptance_unified_status": str(unified["status"]),
        "acceptance_unified_hard_veto": int(unified["hard_veto"]),
        "acceptance_unified_hard_veto_reason": str(unified["hard_veto_reason"]),
        "acceptance_legacy_status": str(legacy["status"]),
        "acceptance_control_mode": mode,
        "acceptance_status": effective_status,
        "acceptance_borderline_review": int(effective_status == "borderline_review"),
        "pred_min_relative_distance_m": float(info.get("min_relative_distance_m", float("nan"))),
        "pred_terminal_relative_distance_m": float(info.get("terminal_relative_distance_m", float("nan"))),
        "zero_pred_min_relative_distance_m": float(zero_info.get("min_relative_distance_m", float("nan"))),
        "zero_pred_terminal_relative_distance_m": float(zero_info.get("terminal_relative_distance_m", float("nan"))),
        "pred_min_future_distance_m": float(core["min_future_min_distance"]),
        "pred_terminal_future_distance_m": float(info.get("terminal_future_min_distance_m", info.get("terminal_future_distance_m", float("nan")))),
        "zero_pred_min_future_distance_m": float(core["zero_min_future_min_distance"]),
        "zero_pred_terminal_future_distance_m": float(
            zero_info.get("terminal_future_min_distance_m", zero_info.get("terminal_future_distance_m", float("nan")))
        ),
        "pred_worst_risk_rank": int(core["worst_risk_rank_pred"]),
        "zero_pred_worst_risk_rank": int(core["zero_worst_risk_rank_pred"]),
        "nominal_hard_gate_passed": int(legacy["geometry_non_degenerate_ok"]),
        "strict_acceptance_passed": int(effective_status == "accepted"),
    }


def _attach_acceptance_gate_diagnostics(name: str, seq: np.ndarray, info: dict, zero_info: dict) -> dict:
    out = dict(info)
    out.update(_acceptance_gate_diagnostics(name, seq, out, zero_info))
    return out


def _passes_strict_acceptance(name: str, seq: np.ndarray, info: dict, zero_info: dict) -> bool:
    diag = _acceptance_gate_diagnostics(name, seq, info, zero_info)
    return str(diag.get("acceptance_status", "rejected")) == "accepted"


def _passes_borderline_rescue(name: str, seq: np.ndarray, info: dict, zero_info: dict) -> bool:
    diag = _acceptance_gate_diagnostics(name, seq, info, zero_info)
    return str(diag.get("acceptance_status", "rejected")) == "borderline_review"

def compare_nominal_vs_robust(
    nominal_info: dict,
    robust_info: dict,
    *,
    profile: NominalDiagnosticsProfile = NOMINAL_DIAGNOSTICS_PROFILE,
) -> dict[str, int | float | str]:
    nom_margin = float(nominal_info.get("min_safe_margin_px", float("nan")))
    rob_margin = float(robust_info.get("min_safe_margin_px", float("nan")))
    nom_viol = _safe_int(nominal_info.get("violation_count_pred", 10**9))
    rob_viol = _safe_int(robust_info.get("violation_count_pred", 10**9))

    nom_min_rel = float(nominal_info.get("min_relative_distance_m", float("nan")))
    rob_min_rel = float(robust_info.get("min_relative_distance_m", float("nan")))
    nom_terminal_rel = float(nominal_info.get("terminal_relative_distance_m", float("nan")))
    rob_terminal_rel = float(robust_info.get("terminal_relative_distance_m", float("nan")))

    nom_trace_ratio = float(nominal_info.get("acceptance_trace_ratio_vs_zero", float("inf")))
    rob_trace_ratio = float(robust_info.get("acceptance_trace_ratio_vs_zero", float("inf")))
    nom_terminal_trace_ratio = float(nominal_info.get("acceptance_terminal_trace_ratio_vs_zero", float("inf")))
    rob_terminal_trace_ratio = float(robust_info.get("acceptance_terminal_trace_ratio_vs_zero", float("inf")))

    robust_strict = _safe_int(robust_info.get("strict_acceptance_passed", 0))
    robust_support_ok = int(
        np.isfinite(rob_margin)
        and (rob_margin >= float(profile.min_robust_support_margin_px))
        and (rob_viol <= int(profile.max_robust_support_violations))
        and ((not bool(profile.require_robust_strict_acceptance)) or (robust_strict == 1))
    )

    geometric_advantage_ok = int(
        (nom_viol <= rob_viol)
        and np.isfinite(nom_margin)
        and np.isfinite(rob_margin)
        and ((nom_margin - rob_margin) >= float(profile.min_margin_advantage_px))
    )

    relative_distance_ok = int(
        (
            (not np.isfinite(rob_min_rel))
            or (not np.isfinite(nom_min_rel))
            or (nom_min_rel >= rob_min_rel - float(profile.max_min_relative_distance_loss_m))
        )
        and (
            (not np.isfinite(rob_terminal_rel))
            or (not np.isfinite(nom_terminal_rel))
            or (
                nom_terminal_rel
                >= rob_terminal_rel - float(profile.max_terminal_relative_distance_loss_m)
            )
        )
    )

    trace_penalty_ok = int(
        (
            (not np.isfinite(rob_trace_ratio))
            or (not np.isfinite(nom_trace_ratio))
            or (nom_trace_ratio <= rob_trace_ratio + float(profile.max_trace_ratio_penalty))
        )
        and (
            (not np.isfinite(rob_terminal_trace_ratio))
            or (not np.isfinite(nom_terminal_trace_ratio))
            or (
                nom_terminal_trace_ratio
                <= rob_terminal_trace_ratio + float(profile.max_terminal_trace_ratio_penalty)
            )
        )
    )

    nominal_allowed = int(
        bool(robust_support_ok)
        and bool(geometric_advantage_ok)
        and bool(relative_distance_ok)
        and bool(trace_penalty_ok)
    )

    if nominal_allowed:
        reason = "nominal_geometric_advantage_with_stable_penalties"
    elif not robust_support_ok:
        reason = "robust_support_missing"
    elif not geometric_advantage_ok:
        reason = "nominal_geometric_advantage_insufficient"
    elif not relative_distance_ok:
        reason = "nominal_relative_distance_degraded"
    else:
        reason = "nominal_trace_penalty_too_high"

    return {
        "selected_strategy": "nominal" if nominal_allowed else "robust",
        "reason": str(reason),
        "robust_support_ok": int(robust_support_ok),
        "geometric_advantage_ok": int(geometric_advantage_ok),
        "relative_distance_ok": int(relative_distance_ok),
        "trace_penalty_ok": int(trace_penalty_ok),
        "nominal_allowed": int(nominal_allowed),
        "robust_margin_px": float(rob_margin),
        "robust_violation_count": int(rob_viol),
        "robust_strict_acceptance_passed": int(robust_strict),
    }

def _nominal_support_diagnostics(nominal_info: dict, robust_info: dict) -> dict[str, int | float]:
    comp = compare_nominal_vs_robust(nominal_info, robust_info)
    return {
        "nominal_support_robust_margin_px": float(comp["robust_margin_px"]),
        "nominal_support_robust_violation_count": int(comp["robust_violation_count"]),
        "nominal_support_robust_strict_acceptance_passed": int(comp["robust_strict_acceptance_passed"]),
        "nominal_support_ok": int(comp["robust_support_ok"]),
        "nominal_advantage_vs_robust_ok": int(comp["nominal_allowed"]),
        "nominal_vs_robust_geometric_advantage_ok": int(comp["geometric_advantage_ok"]),
        "nominal_vs_robust_relative_distance_ok": int(comp["relative_distance_ok"]),
        "nominal_vs_robust_trace_penalty_ok": int(comp["trace_penalty_ok"]),
        "nominal_vs_robust_selected_strategy": str(comp["selected_strategy"]),
        "nominal_vs_robust_reason": str(comp["reason"]),
    }


def _attach_candidate_internal_preview(seq: np.ndarray, info: dict, x_est: np.ndarray, n_orbit: float) -> dict:
    out = dict(info)
    out.update(_internal_preview_metrics_from_state(x_est, np.asarray(seq, dtype=float), PLAN_DT, n_orbit))
    return out


def choose_budget_ladder(pred_min_margin_px: float) -> tuple[list[float], str]:
    m = float(pred_min_margin_px)
    if not np.isfinite(m):
        return [DV_MAG_BASE], "unknown_margin_case"
    if m <= -800.0:
        return [DV_MAG_EXTREME, DV_MAG_ULTRA, DV_MAG_SUPER, DV_MAG_MEGA], "extreme_hard_case"
    if m <= -300.0:
        return [DV_MAG_VERY_HARD, DV_MAG_EXTREME, DV_MAG_ULTRA, DV_MAG_SUPER, DV_MAG_MEGA], "very_hard_case"
    if m <= -120.0:
        return [DV_MAG_HARD, DV_MAG_VERY_HARD, 0.20, DV_MAG_EXTREME, DV_MAG_ULTRA, DV_MAG_SUPER, DV_MAG_MEGA], "hard_case"
    if m <= -40.0:
        return [DV_MAG_BASE, DV_MAG_HARD, DV_MAG_VERY_HARD, 0.20, DV_MAG_EXTREME], "normal_case"
    return [DV_MAG_BASE, DV_MAG_HARD, DV_MAG_VERY_HARD, 0.20], "normal_case"


def _search_profile_for_case(
    case_label: str,
    total_dv_budget: float,
    *,
    fast_mode: bool | None = None,
) -> dict[str, float | int]:
    use_fast_mode = bool(FAST_SEARCH_MODE) if fast_mode is None else bool(fast_mode)
    profile_set = FAST_SEARCH_PROFILES if use_fast_mode else STANDARD_SEARCH_PROFILES
    hardness = _classify_search_hardness(case_label, total_dv_budget)
    selected = profile_set.get(hardness, profile_set["normal"])
    return _search_profile_to_dict(selected)


@contextmanager
def _temporary_search_profile(profile: dict[str, float | int]):
    global PLAN_HORIZON_SEC
    global SEARCH_ACTIVE_STEPS
    global MULTISTEP_SEED_TOPK
    global RANDOM_MULTISTART_COUNT
    global RANDOM_NONZERO_STEPS_MAX

    old_values = {
        "PLAN_HORIZON_SEC": PLAN_HORIZON_SEC,
        "SEARCH_ACTIVE_STEPS": SEARCH_ACTIVE_STEPS,
        "MULTISTEP_SEED_TOPK": MULTISTEP_SEED_TOPK,
        "RANDOM_MULTISTART_COUNT": RANDOM_MULTISTART_COUNT,
        "RANDOM_NONZERO_STEPS_MAX": RANDOM_NONZERO_STEPS_MAX,
    }
    try:
        PLAN_HORIZON_SEC = float(profile.get("plan_horizon_sec", PLAN_HORIZON_SEC))
        SEARCH_ACTIVE_STEPS = _safe_int(profile.get("search_active_steps", SEARCH_ACTIVE_STEPS))
        MULTISTEP_SEED_TOPK = _safe_int(profile.get("multistep_seed_topk", MULTISTEP_SEED_TOPK))
        RANDOM_MULTISTART_COUNT = _safe_int(profile.get("random_multistart_count", RANDOM_MULTISTART_COUNT))
        RANDOM_NONZERO_STEPS_MAX = _safe_int(profile.get("random_nonzero_steps_max", RANDOM_NONZERO_STEPS_MAX))
        yield
    finally:
        PLAN_HORIZON_SEC = float(old_values["PLAN_HORIZON_SEC"])
        SEARCH_ACTIVE_STEPS = int(old_values["SEARCH_ACTIVE_STEPS"])
        MULTISTEP_SEED_TOPK = int(old_values["MULTISTEP_SEED_TOPK"])
        RANDOM_MULTISTART_COUNT = int(old_values["RANDOM_MULTISTART_COUNT"])
        RANDOM_NONZERO_STEPS_MAX = int(old_values["RANDOM_NONZERO_STEPS_MAX"])

def classify_failed_case(
    trig: pd.Series,
    pred_min_margin_px: float,
    used_budget: float,
    *,
    selected_info: dict | None = None,
    nominal_info: dict | None = None,
    robust_info: dict | None = None,
) -> str:
    t_sec = float(trig.get("t_sec", -1.0))
    frame = _safe_int(trig.get("frame", -1))

    def _primary_gate_reason(info: dict | None) -> str:
        if info is None:
            return ""
        src = dict(info)
        margin = float(src.get("min_safe_margin_px", float("nan")))
        viol = _safe_int(src.get("violation_count_pred", 10**9))
        safe_gate = _safe_int(src.get("acceptance_gate_safe", -1))
        structural_gate = _safe_int(src.get("acceptance_gate_structural", -1))
        geometry_gate = _safe_int(src.get("acceptance_gate_geometry_non_degenerate", -1))
        information_gate = _safe_int(src.get("acceptance_gate_information", -1))
        future_gate = _safe_int(src.get("acceptance_gate_future_distance", -1))
        strict_gate = _safe_int(src.get("strict_acceptance_passed", -1))

        if (viol > 0 and np.isfinite(margin) and margin <= 0.0) or (safe_gate == 0):
            return "safety_gate_veto"
        if structural_gate == 0 or geometry_gate == 0:
            return "structural_gate_veto"
        if information_gate == 0:
            return "information_gate_veto"
        if future_gate == 0:
            return "future_distance_gate_veto"
        if strict_gate == 0:
            return "strict_acceptance_veto"
        return ""

    if (t_sec >= LATE_TRIGGER_SEC) or (frame >= LATE_TRIGGER_FRAME):
        return "late_hard_case"
    if not np.isfinite(pred_min_margin_px):
        return "unobservable_hard_case"

    if selected_info is not None:
        selected_info = dict(selected_info)
        selected_strategy = str(selected_info.get("selected_strategy", ""))
        selection_reason = str(selected_info.get("execution_selection_reason", ""))
        selected_reason = str(selected_info.get("plan_acceptance_reason", ""))
        selected_margin = float(selected_info.get("min_safe_margin_px", float("nan")))
        selected_viol = _safe_int(selected_info.get("violation_count_pred", 10**9))
        selected_strict = _safe_int(selected_info.get("strict_acceptance_passed", -1))
        selected_safe_gate = _safe_int(selected_info.get("acceptance_gate_safe", -1))
        selected_structural_gate = _safe_int(selected_info.get("acceptance_gate_structural", -1))
        selected_information_gate = _safe_int(selected_info.get("acceptance_gate_information", -1))
        selected_geometry_gate = _safe_int(selected_info.get("acceptance_gate_geometry_non_degenerate", -1))
        selected_future_gate = _safe_int(selected_info.get("acceptance_gate_future_distance", -1))

        selected_gate_reason = _primary_gate_reason(selected_info)
        if selected_gate_reason:
            return selected_gate_reason

        if selection_reason.startswith("best_effort"):
            if (
                selected_strategy == "robust"
                and nominal_info is not None
                and robust_info is not None
            ):
                nom_margin = float(dict(nominal_info).get("min_safe_margin_px", float("nan")))
                rob_margin = float(dict(robust_info).get("min_safe_margin_px", float("nan")))
                nom_viol = _safe_int(dict(nominal_info).get("violation_count_pred", 10**9))
                rob_viol = _safe_int(dict(robust_info).get("violation_count_pred", 10**9))
                nominal_clearly_better = (
                    (nom_viol < rob_viol)
                    or (
                        nom_viol == rob_viol
                        and np.isfinite(nom_margin)
                        and np.isfinite(rob_margin)
                        and (nom_margin >= rob_margin + 5.0)
                    )
                )
                if nominal_clearly_better:
                    return "best_effort_policy_veto"
            return "best_effort_policy_veto"

        if selected_strict == 0:
            return "strict_acceptance_veto"

    gate_reasons = [
        _primary_gate_reason(nominal_info),
        _primary_gate_reason(robust_info),
    ]
    gate_reasons = [it for it in gate_reasons if it]
    if gate_reasons:
        if all(it == "safety_gate_veto" for it in gate_reasons):
            return "safety_gate_veto"
        for label in (
            "structural_gate_veto",
            "information_gate_veto",
            "future_distance_gate_veto",
            "strict_acceptance_veto",
        ):
            if label in gate_reasons:
                return label

    if pred_min_margin_px <= EXTREME_MARGIN_THR_PX and used_budget >= DV_MAG_SUPER - 1e-9:
        return "budget_limited_extreme_case"
    if pred_min_margin_px <= HARD_MARGIN_THR_PX:
        return "budget_limited_hard_case"
    return "strict_acceptance_veto"
def should_try_budget_retry(
    plan_accepted: bool,
    trigger_case_label: str,
    used_budget: float | None,
    final_pred_margin_px: float | None = None,
    final_violation_count: int | None = None,
) -> bool:
    del trigger_case_label
    del used_budget
    del final_pred_margin_px
    del final_violation_count
    if plan_accepted:
        return False
    # Legacy retry loop is intentionally retired from final decision logic.
    if not bool(BUDGET_PROFILE.retry_enable):
        return False
    return False


def _budget_efficiency_diagnostics(
    info: dict,
    *,
    profile: BudgetProfile = BUDGET_PROFILE,
) -> dict[str, float | int]:
    selected_total_dv = float(info.get("selected_total_dv_mag_mps", info.get("total_dv_mag_mps", float("nan"))))
    budget_dv = float(info.get("total_dv_budget_mps", float("nan")))
    dv_ref = selected_total_dv if np.isfinite(selected_total_dv) and selected_total_dv > 1.0e-9 else budget_dv
    if (not np.isfinite(dv_ref)) or dv_ref <= 1.0e-9:
        dv_ref = 1.0e-9

    selected_margin = float(info.get("selected_pred_min_safe_margin_px", info.get("min_safe_margin_px", float("nan"))))
    zero_margin = float(info.get("zero_pred_min_safe_margin_px", float("nan")))
    selected_viol = _safe_int(info.get("selected_pred_violation_count", info.get("violation_count_pred", -1)))
    zero_viol = _safe_int(info.get("zero_pred_violation_count", -1))

    margin_gain = float(info.get("acceptance_margin_gain_vs_zero_px", float("nan")))
    if not np.isfinite(margin_gain):
        margin_gain = float(selected_margin - zero_margin) if np.isfinite(selected_margin) and np.isfinite(zero_margin) else float("nan")

    future_gain = float(info.get("acceptance_future_distance_gain_vs_zero_m", float("nan")))
    if not np.isfinite(future_gain):
        selected_future = float(info.get("selected_pred_min_future_min_distance_m", info.get("min_future_min_distance_m", float("nan"))))
        zero_future = float(info.get("zero_pred_min_future_distance_m", float("nan")))
        future_gain = float(selected_future - zero_future) if np.isfinite(selected_future) and np.isfinite(zero_future) else float("nan")

    trace_ratio = float(info.get("acceptance_trace_ratio_vs_zero", float("nan")))
    terminal_trace_ratio = float(info.get("acceptance_terminal_trace_ratio_vs_zero", float("nan")))
    trace_penalty = max(0.0, trace_ratio - 1.0) if np.isfinite(trace_ratio) else float("inf")
    terminal_trace_penalty = max(0.0, terminal_trace_ratio - 1.0) if np.isfinite(terminal_trace_ratio) else float("inf")

    margin_gain_per_dv = float(margin_gain / dv_ref) if np.isfinite(margin_gain) else float("nan")
    future_distance_gain_per_dv = float(future_gain / dv_ref) if np.isfinite(future_gain) else float("nan")
    trace_penalty_per_dv = float(trace_penalty / dv_ref) if np.isfinite(trace_penalty) else float("inf")
    terminal_trace_penalty_per_dv = float(terminal_trace_penalty / dv_ref) if np.isfinite(terminal_trace_penalty) else float("inf")

    efficiency_diagnostics_required = int(
        bool(profile.efficiency_diagnostics_enable)
        and np.isfinite(budget_dv)
        and (budget_dv >= float(profile.efficiency_min_budget_mps))
    )
    normal_rule_passed = int(
        (
            (
                np.isfinite(margin_gain_per_dv)
                and margin_gain_per_dv >= float(profile.efficiency_min_margin_gain_per_dv)
            )
            or (
                np.isfinite(future_distance_gain_per_dv)
                and future_distance_gain_per_dv >= float(profile.efficiency_min_future_gain_per_dv)
            )
        )
        and np.isfinite(trace_penalty_per_dv)
        and np.isfinite(terminal_trace_penalty_per_dv)
        and (trace_penalty_per_dv <= float(profile.efficiency_max_trace_penalty_per_dv))
        and (
            terminal_trace_penalty_per_dv
            <= float(profile.efficiency_max_terminal_trace_penalty_per_dv)
        )
    )
    high_gain_override_passed = int(
        bool(profile.high_gain_override_enable)
        and np.isfinite(selected_margin)
        and (selected_margin >= float(profile.high_gain_override_min_abs_margin_px))
        and np.isfinite(margin_gain)
        and (margin_gain >= float(profile.high_gain_override_min_margin_gain_px))
        and (selected_viol == 0)
        and (zero_viol - selected_viol >= int(profile.high_gain_override_min_violation_reduction))
        and np.isfinite(trace_ratio)
        and (trace_ratio <= float(profile.high_gain_override_max_trace_ratio))
        and np.isfinite(terminal_trace_ratio)
        and (terminal_trace_ratio <= float(profile.high_gain_override_max_terminal_trace_ratio))
    )
    efficiency_flagged = int(
        (efficiency_diagnostics_required == 1)
        and (not bool(normal_rule_passed))
        and (not bool(high_gain_override_passed))
    )

    return {
        "margin_gain_per_dv": float(margin_gain_per_dv),
        "future_distance_gain_per_dv": float(future_distance_gain_per_dv),
        "trace_penalty_per_dv": float(trace_penalty_per_dv),
        "terminal_trace_penalty_per_dv": float(terminal_trace_penalty_per_dv),
        "budget_efficiency_diagnostics_required": int(efficiency_diagnostics_required),
        "budget_efficiency_normal_rule_passed": int(normal_rule_passed),
        "budget_efficiency_high_gain_override_passed": int(high_gain_override_passed),
        "budget_efficiency_flagged": int(efficiency_flagged),
        # Compatibility aliases: diagnostics-only and never used as policy gates.
        "budget_worthiness_required": 0,
        "budget_worthiness_diagnostics_required": int(efficiency_diagnostics_required),
        "budget_worthiness_normal_rule_passed": int(normal_rule_passed),
        "budget_force_worthy_passed": int(high_gain_override_passed),
        "budget_worthiness_flagged": int(efficiency_flagged),
        "budget_worthiness_passed": 1,
    }



def _finalize_schedule(
    chosen_name: str,
    chosen_seq: np.ndarray,
    chosen_info: dict,
    zero_info: dict,
    accepted: bool,
    reason: str,
    *,
    total_dv_budget: float,
    trigger_case_label: str | None = None,
) -> tuple[np.ndarray, dict]:
    finalized = dict(chosen_info)
    acceptance_status = str(finalized.get("acceptance_status", "")).strip().lower()
    if acceptance_status not in {"accepted", "borderline_review", "rejected"}:
        if bool(accepted):
            acceptance_status = "accepted"
        elif str(reason).strip().lower() == "borderline_review":
            acceptance_status = "borderline_review"
        else:
            acceptance_status = "rejected"

    finalized["plan_accepted"] = bool(accepted)
    finalized["plan_acceptance_reason"] = str(reason)
    finalized.setdefault("execution_selection_reason", str(reason))
    finalized["strictly_feasible_plan"] = int(bool(accepted))
    finalized["acceptance_status"] = acceptance_status
    finalized["selected_strategy"] = str(chosen_name)
    finalized["zero_pred_min_safe_margin_px"] = float(zero_info.get("min_safe_margin_px", float("nan")))
    finalized["zero_pred_violation_count"] = _safe_int(zero_info.get("violation_count_pred", -1))
    finalized["zero_pred_terminal_margin_px"] = float(zero_info.get("terminal_margin_px", float("nan")))
    finalized["zero_pred_negative_margin_sum_px"] = float(zero_info.get("negative_margin_sum_px_pred", float("nan")))
    finalized["zero_pred_min_relative_distance_m"] = float(
        zero_info.get("min_relative_distance_m", float("nan"))
    )
    finalized["zero_pred_terminal_relative_distance_m"] = float(
        zero_info.get("terminal_relative_distance_m", float("nan"))
    )
    finalized["zero_pred_min_future_min_distance_m"] = float(
        zero_info.get("min_future_min_distance_m", zero_info.get("min_future_distance_m", float("nan")))
    )
    finalized["zero_pred_terminal_future_min_distance_m"] = float(
        zero_info.get("terminal_future_min_distance_m", zero_info.get("terminal_future_distance_m", float("nan")))
    )
    finalized["zero_pred_min_future_distance_m"] = float(
        zero_info.get("min_future_distance_m", finalized.get("zero_pred_min_future_min_distance_m", float("nan")))
    )
    finalized["zero_pred_terminal_future_distance_m"] = float(
        zero_info.get("terminal_future_distance_m", finalized.get("zero_pred_terminal_future_min_distance_m", float("nan")))
    )
    finalized["total_dv_budget_mps"] = float(total_dv_budget)
    if trigger_case_label is not None:
        case_label = str(trigger_case_label)
    elif acceptance_status == "accepted":
        case_label = "accepted_case"
    elif acceptance_status == "borderline_review":
        case_label = "borderline_review"
    else:
        case_label = "rejected"
    finalized["case_label"] = case_label
    nonzero_steps, total_dv, dominant_ratio = _sequence_structure_stats(chosen_seq)
    finalized["selected_nonzero_steps"] = int(nonzero_steps)
    finalized["selected_total_dv_mag_mps"] = float(total_dv)
    finalized["selected_dominant_ratio"] = float(dominant_ratio)
    finalized["selected_span_steps"] = int(_sequence_step_span(chosen_seq))
    finalized["selected_structurally_sane"] = int(_sequence_is_structurally_sane(chosen_seq))
    finalized["selected_pred_min_safe_margin_px"] = float(finalized.get("min_safe_margin_px", float("nan")))
    finalized["selected_pred_terminal_margin_px"] = float(finalized.get("terminal_margin_px", float("nan")))
    finalized["selected_pred_violation_count"] = _safe_int(finalized.get("violation_count_pred", -1))
    finalized["selected_pred_negative_margin_sum_px"] = float(finalized.get("negative_margin_sum_px_pred", float("nan")))
    finalized["selected_pred_min_relative_distance_m"] = float(finalized.get("min_relative_distance_m", float("nan")))
    finalized["selected_pred_terminal_relative_distance_m"] = float(finalized.get("terminal_relative_distance_m", float("nan")))
    finalized["selected_pred_min_future_min_distance_m"] = float(finalized.get("min_future_min_distance_m", float("nan")))
    finalized["selected_pred_terminal_future_min_distance_m"] = float(finalized.get("terminal_future_min_distance_m", float("nan")))
    finalized["selected_pred_worst_risk_rank"] = _safe_int(finalized.get("worst_risk_rank_pred", -999))
    for gate_key in [
        "acceptance_gate_safe",
        "acceptance_gate_structural",
        "acceptance_gate_information",
        "acceptance_gate_geometry_non_degenerate",
        "acceptance_gate_future_distance",
    ]:
        finalized[gate_key] = _safe_int(finalized.get(gate_key, -1))
        finalized[f"selected_{gate_key}"] = int(finalized[gate_key])
    finalized["selected_acceptance_trace_ratio_vs_zero"] = float(
        finalized.get("acceptance_trace_ratio_vs_zero", float("nan"))
    )
    finalized["selected_acceptance_terminal_trace_ratio_vs_zero"] = float(
        finalized.get("acceptance_terminal_trace_ratio_vs_zero", float("nan"))
    )
    finalized["selected_acceptance_info_delta_vs_zero"] = float(
        finalized.get("acceptance_info_delta_vs_zero", float("nan"))
    )
    finalized["selected_acceptance_margin_gain_vs_zero_px"] = float(
        finalized.get("acceptance_margin_gain_vs_zero_px", float("nan"))
    )
    finalized["selected_acceptance_terminal_margin_gain_vs_zero_px"] = float(
        finalized.get("acceptance_terminal_margin_gain_vs_zero_px", float("nan"))
    )
    finalized["selected_acceptance_violation_delta_vs_zero"] = int(
        finalized.get("acceptance_violation_delta_vs_zero", -1)
    )
    finalized["selected_acceptance_negative_margin_sum_delta_vs_zero_px"] = float(
        finalized.get("acceptance_negative_margin_sum_delta_vs_zero_px", float("nan"))
    )
    finalized["selected_acceptance_future_distance_gain_vs_zero_m"] = float(
        finalized.get("acceptance_future_distance_gain_vs_zero_m", float("nan"))
    )
    finalized["selected_future_distance_gain_vs_zero_m"] = float(
        finalized.get(
            "selected_acceptance_future_distance_gain_vs_zero_m",
            finalized.get("acceptance_future_distance_gain_vs_zero_m", float("nan")),
        )
    )
    finalized["selected_acceptance_future_distance_non_regression"] = int(
        finalized.get("acceptance_future_distance_non_regression", -1)
    )
    finalized["selected_acceptance_future_distance_gate_required"] = int(
        finalized.get("acceptance_future_distance_gate_required", -1)
    )
    finalized["selection_verdict"] = {
        "selected_strategy": str(chosen_name),
        "selection_reason": str(finalized.get("execution_selection_reason", reason)),
    }

    planner_acceptance_status = acceptance_status

    finalized["planner_candidate_status"] = f"candidate_{str(chosen_name)}"
    finalized["planner_acceptance_status"] = planner_acceptance_status
    finalized_initial_selection_status = str(
        finalized.get("planner_initial_selection_status", f"selected_{str(chosen_name)}")
    )
    finalized["planner_initial_selection_status"] = finalized_initial_selection_status
    finalized["final_selected_strategy"] = str(chosen_name)
    finalized["selection_correction_applied"] = int(
        finalized_initial_selection_status.strip().lower() != f"selected_{str(chosen_name)}"
    )
    finalized["planner_selection_status"] = f"selected_{str(chosen_name)}"
    finalized.pop("execution_final_status", None)
    planner_stage_status = "planner_accepted_pending_execution" if bool(accepted) else "planner_rejected"
    finalized["planner_execution_stage_status"] = planner_stage_status
    finalized["pre_execution_status"] = planner_stage_status

    finalized["acceptance_verdict"] = {
        "plan_accepted": int(bool(accepted)),
        "acceptance_status": acceptance_status,
        "acceptance_reason": str(reason),
        "strict_acceptance_passed": _safe_int(finalized.get("strict_acceptance_passed", -1)),
        "acceptance_gate_safe": _safe_int(finalized.get("acceptance_gate_safe", -1)),
        "acceptance_gate_structural": _safe_int(finalized.get("acceptance_gate_structural", -1)),
        "acceptance_gate_information": _safe_int(finalized.get("acceptance_gate_information", -1)),
        "acceptance_gate_geometry_non_degenerate": _safe_int(finalized.get("acceptance_gate_geometry_non_degenerate", -1)),
        "acceptance_gate_future_distance": _safe_int(finalized.get("acceptance_gate_future_distance", -1)),
    }
    finalized["execution_verdict"] = {
        "execution_requested_strategy": str(chosen_name),
        "execution_should_apply": int(bool(accepted)),
        "execution_reason": str(finalized.get("execution_selection_reason", reason)),
    }
    finalized["execution_request_strategy"] = str(chosen_name)
    finalized["execution_request_source_reason"] = str(
        finalized.get("execution_selection_reason", reason)
    )
    return chosen_seq, finalized


def _sequence_structure_stats(seq: np.ndarray) -> tuple[int, float, float]:
    norms = np.linalg.norm(np.asarray(seq, dtype=float), axis=1)
    total = float(np.sum(norms))
    nonzero = int(np.sum(norms > 1.0e-10))
    dominant_ratio = float(np.max(norms) / total) if total > 1.0e-10 else 1.0
    return nonzero, total, dominant_ratio


def _sequence_step_span(seq: np.ndarray) -> int:
    norms = np.linalg.norm(np.asarray(seq, dtype=float), axis=1)
    idx = np.where(norms > 1.0e-10)[0]
    if idx.size <= 1:
        return 0
    return int(idx[-1] - idx[0])


def _sequence_is_structurally_sane(seq: np.ndarray) -> bool:
    nonzero_steps, total_dv, dominant_ratio = _sequence_structure_stats(seq)
    if total_dv <= 1.0e-10:
        return False
    if nonzero_steps < BEST_EFFORT_MIN_NONZERO_STEPS:
        return False
    if dominant_ratio > BEST_EFFORT_MAX_DOMINANT_RATIO:
        return False
    if _sequence_step_span(seq) < BEST_EFFORT_MIN_SPAN_STEPS:
        return False
    return True


def _update_info_from_eval(info: dict, ev: EvalResult) -> dict:
    out = dict(info)
    out.update({
        'min_safe_margin_px': float(ev.min_safe_margin_px),
        'terminal_margin_px': float(ev.terminal_margin_px),
        'violation_count_pred': int(ev.violation_count),
        'negative_margin_sum_px_pred': float(ev.negative_margin_sum_px),
        'total_dv_mag_mps': float(ev.total_dv_mag_mps),
        'margins_by_anchor': ev.margins_by_anchor,
        'info_metric': float(ev.info_metric),
        'trace_pos_sum': float(ev.trace_pos_sum),
        'terminal_trace_pos': float(ev.terminal_trace_pos),
        'entropy_reduction': float(ev.entropy_reduction),
        'fim_trace_sum': float(ev.fim_trace_sum),
        'visible_update_steps': int(ev.visible_update_steps),
        'min_relative_distance_m': float(ev.min_relative_distance_m),
        'terminal_relative_distance_m': float(ev.terminal_relative_distance_m),
        'min_future_min_distance_m': float(ev.min_future_min_distance_m),
        'terminal_future_min_distance_m': float(ev.terminal_future_min_distance_m),
        'worst_risk_rank_pred': int(ev.worst_risk_rank_pred),
        'info_objective': INFO_OBJECTIVE,
    })
    return out


def _repair_best_effort_structure(
    trig: pd.Series,
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    seq: np.ndarray,
    info: dict,
    *,
    total_dv_budget: float,
) -> tuple[np.ndarray, dict]:
    if _sequence_is_structurally_sane(seq):
        return seq, info

    anchors = build_anchor_states(trig, x_est, p_diag, robust=True)
    best_seq = np.asarray(seq, dtype=float).copy()
    best_ev = evaluate_schedule_exact(best_seq, anchors, PLAN_DT, n_orbit, p_diag=p_diag)
    best_info = _update_info_from_eval(info, best_ev)

    active_steps = min(best_seq.shape[0], SEARCH_ACTIVE_STEPS)
    for _ in range(BEST_EFFORT_REPAIR_PASSES):
        repaired_seq, repaired_ev, improved = _split_dominant_impulse(
            best_seq, best_ev, anchors, total_dv_budget, PLAN_DT, n_orbit, active_steps
        )
        if improved:
            best_seq = repaired_seq
            best_ev = repaired_ev
            best_info = _update_info_from_eval(best_info, best_ev)
        if _sequence_is_structurally_sane(best_seq):
            return best_seq, best_info

    norms = np.linalg.norm(best_seq, axis=1)
    if norms.size == 0 or float(np.max(norms)) <= 1.0e-10:
        return best_seq, best_info

    k0 = int(np.argmax(norms[:active_steps]))
    dominant_mag = float(norms[k0])
    dir0 = best_seq[k0] / max(dominant_mag, 1.0e-12)
    ranked_steps = _rank_steps_from_eval(best_ev, active_steps)
    candidate_steps = [k for k in ranked_steps if k != k0]
    for look in range(1, min(DOMINANT_SPLIT_MAX_LOOKAHEAD, active_steps - k0 - 1) + 1):
        kk = k0 + look
        if kk not in candidate_steps:
            candidate_steps.append(kk)

    candidate_dirs = build_candidate_directions(seq=best_seq, k=min(k0 + 1, active_steps - 1), anchors=anchors, dt=PLAN_DT, n_orbit=n_orbit, p_diag=p_diag)
    if not candidate_dirs:
        candidate_dirs = [dir0]

    for j in candidate_steps[:8]:
        for split in (0.60, 0.55, 0.50):
            mag0 = dominant_mag * split
            magj = dominant_mag * (1.0 - split)
            for dj in candidate_dirs[:8]:
                cand = best_seq.copy()
                cand[k0, :] = dir0 * mag0
                cand[j, :] = dj * magj
                if total_dv_used(cand) > total_dv_budget + 1.0e-9:
                    continue
                ev = evaluate_schedule_exact(cand, anchors, PLAN_DT, n_orbit, p_diag=p_diag)
                cand_sane = _sequence_is_structurally_sane(cand)
                best_sane = _sequence_is_structurally_sane(best_seq)
                better = False
                if cand_sane and (not best_sane):
                    better = True
                elif cand_sane == best_sane and ev.metric > best_ev.metric + 1.0e-9:
                    better = True
                if better:
                    best_seq = cand
                    best_ev = ev
                    best_info = _update_info_from_eval(best_info, best_ev)
        if _sequence_is_structurally_sane(best_seq):
            break

    return best_seq, best_info


def _candidate_failure_rank(name: str, seq: np.ndarray, info: dict) -> tuple:
    min_margin = float(info.get('min_safe_margin_px', -1.0e18))
    violation_count = _safe_int(info.get('violation_count_pred', 10**9))
    negative_margin_sum = float(info.get('negative_margin_sum_px_pred', 1.0e18))
    terminal_margin = float(info.get('terminal_margin_px', -1.0e18))
    info_metric = float(info.get('info_metric', -1.0e18))
    trace_pos_sum = float(info.get('trace_pos_sum', 1.0e18))
    terminal_trace_pos = float(info.get('terminal_trace_pos', 1.0e18))
    nonzero_steps, total_dv, dominant_ratio = _sequence_structure_stats(seq)
    span_steps = _sequence_step_span(seq)

    is_zero = 0 if total_dv > 1.0e-10 else 1
    lacks_two_pulses = 1 if nonzero_steps < BEST_EFFORT_MIN_NONZERO_STEPS else 0
    dominant_penalty = 1 if dominant_ratio > BEST_EFFORT_MAX_DOMINANT_RATIO else 0
    span_penalty = 1 if span_steps < BEST_EFFORT_MIN_SPAN_STEPS else 0
    name_priority = {'robust': 0, 'nominal': 1, 'zero': 2}.get(str(name), 9)

    return (
        violation_count,
        negative_margin_sum,
        -min_margin,
        -terminal_margin,
        is_zero,
        lacks_two_pulses,
        dominant_penalty,
        span_penalty,
        trace_pos_sum,
        terminal_trace_pos,
        -info_metric,
        total_dv,
        name_priority,
    )


def _best_effort_primary_tuple(info: dict) -> tuple[int, float, float]:
    violation_count = _safe_int(info.get("violation_count_pred", 10**9))
    min_margin = float(info.get("min_safe_margin_px", float("nan")))
    terminal_margin = float(info.get("terminal_margin_px", float("nan")))

    if not np.isfinite(min_margin):
        min_margin = -1.0e18
    if not np.isfinite(terminal_margin):
        terminal_margin = -1.0e18

    return (violation_count, -min_margin, -terminal_margin)

def _best_effort_rank_key(name: str, seq: np.ndarray, info: dict) -> tuple:
    return _best_effort_primary_tuple(info) + _candidate_failure_rank(name, seq, info)


def _select_execution_plan(
    trig: pd.Series,
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    schedule_nom: np.ndarray,
    info_nom: dict,
    schedule_rob: np.ndarray,
    info_rob: dict,
    *,
    total_dv_budget: float,
) -> tuple[np.ndarray, dict]:
    zero_seq, zero_info = _evaluate_zero_baseline(trig, x_est, p_diag, n_orbit, robust=True)
    zero_info = _attach_acceptance_gate_diagnostics('zero', zero_seq, zero_info, zero_info)

    candidates: list[tuple[str, np.ndarray, dict]] = [
        ('robust', schedule_rob, dict(info_rob)),
        ('nominal', schedule_nom, dict(info_nom)),
        ('zero', zero_seq, dict(zero_info)),
    ]

    def _internal_preview_sort_key(info: dict) -> float:
        val = float(info.get("internal_preview_min_future_distance_m", float("nan")))
        return -val if np.isfinite(val) else 1.0e18

    diagnosed_candidates: list[tuple[str, np.ndarray, dict]] = []
    for name, seq, info in candidates:
        info_diag = _attach_acceptance_gate_diagnostics(name, seq, info, zero_info)
        info_diag = _attach_candidate_internal_preview(np.asarray(seq, dtype=float), info_diag, x_est, n_orbit)
        diagnosed_candidates.append((name, np.asarray(seq, dtype=float).copy(), info_diag))

    strict_candidates = [
        (name, seq, info)
        for name, seq, info in diagnosed_candidates
        if _passes_strict_acceptance(name, seq, info, zero_info)
    ]
    if strict_candidates:
        strict_candidates.sort(
            key=lambda item: (
                -float(item[2].get('min_safe_margin_px', -1e18)),
                _safe_int(item[2].get('violation_count_pred', 10**9)),
                _internal_preview_sort_key(item[2]),
                float(item[2].get('acceptance_trace_ratio_vs_zero', float('inf'))),
                float(item[2].get('acceptance_terminal_trace_ratio_vs_zero', float('inf'))),
                -float(item[2].get('acceptance_info_delta_vs_zero', -1e18)),
                _candidate_failure_rank(item[0], item[1], item[2]),
            )
        )
        strict_map = {str(name): (seq, info) for name, seq, info in strict_candidates}
        chosen_name, chosen_seq, chosen_info = strict_candidates[0]
        chosen_info = dict(chosen_info)
        if str(chosen_name) == 'nominal':
            robust_entry = strict_map.get('robust')
            if robust_entry is not None:
                chosen_info.update(_nominal_support_diagnostics(dict(chosen_info), dict(robust_entry[1])))
        chosen_info['execution_selection_reason'] = f'strict_acceptance_{chosen_name}'
        return _finalize_schedule(
            chosen_name=chosen_name,
            chosen_seq=chosen_seq,
            chosen_info=chosen_info,
            zero_info=zero_info,
            accepted=True,
            reason='accepted',
            total_dv_budget=total_dv_budget,
        )

    repaired_candidates: list[tuple[str, np.ndarray, dict]] = []
    for name, seq, info in diagnosed_candidates:
        if str(name) == 'zero':
            repaired_candidates.append((name, np.asarray(seq, dtype=float).copy(), dict(info)))
            continue
        seq_rep, info_rep = _repair_best_effort_structure(
            trig, x_est, p_diag, n_orbit, np.asarray(seq, dtype=float).copy(), dict(info), total_dv_budget=total_dv_budget
        )
        info_rep = _attach_acceptance_gate_diagnostics(name, seq_rep, info_rep, zero_info)
        info_rep = _attach_candidate_internal_preview(np.asarray(seq_rep, dtype=float), info_rep, x_est, n_orbit)
        repaired_candidates.append((name, seq_rep, info_rep))

    repaired_strict_candidates = [
        (name, seq, info)
        for name, seq, info in repaired_candidates
        if _passes_strict_acceptance(name, seq, info, zero_info)
    ]
    if repaired_strict_candidates:
        repaired_strict_candidates.sort(
            key=lambda item: (
                -float(item[2].get('min_safe_margin_px', -1e18)),
                _safe_int(item[2].get('violation_count_pred', 10**9)),
                _internal_preview_sort_key(item[2]),
                float(item[2].get('acceptance_trace_ratio_vs_zero', float('inf'))),
                float(item[2].get('acceptance_terminal_trace_ratio_vs_zero', float('inf'))),
                -float(item[2].get('acceptance_info_delta_vs_zero', -1e18)),
                _candidate_failure_rank(item[0], item[1], item[2]),
            )
        )
        repaired_strict_map = {str(name): (seq, info) for name, seq, info in repaired_strict_candidates}
        chosen_name, chosen_seq, chosen_info = repaired_strict_candidates[0]
        chosen_info = dict(chosen_info)
        if str(chosen_name) == 'nominal':
            robust_entry = repaired_strict_map.get('robust')
            if robust_entry is not None:
                chosen_info.update(_nominal_support_diagnostics(dict(chosen_info), dict(robust_entry[1])))
        chosen_info['execution_selection_reason'] = f'repaired_strict_acceptance_{chosen_name}'
        return _finalize_schedule(
            chosen_name=chosen_name,
            chosen_seq=chosen_seq,
            chosen_info=chosen_info,
            zero_info=zero_info,
            accepted=True,
            reason='accepted',
            total_dv_budget=total_dv_budget,
        )

    rescue_candidates = []
    for name, seq, info in repaired_candidates:
        if not _passes_borderline_rescue(name, seq, info, zero_info):
            continue
        info = dict(info)
        rescue_candidates.append((name, seq, info))
    if rescue_candidates:
        rescue_candidates.sort(
            key=lambda item: (
                -float(item[2].get('min_safe_margin_px', -1e18)),
                _safe_int(item[2].get('violation_count_pred', 10**9)),
                _internal_preview_sort_key(item[2]),
                float(item[2].get('acceptance_trace_ratio_vs_zero', float('inf'))),
                float(item[2].get('acceptance_terminal_trace_ratio_vs_zero', float('inf'))),
                -float(item[2].get('acceptance_info_delta_vs_zero', -1e18)),
                {'robust': 0, 'nominal': 1, 'zero': 2}.get(str(item[0]), 9),
            )
        )
        chosen_name, chosen_seq, chosen_info = rescue_candidates[0]
        chosen_info = dict(chosen_info)
        chosen_info['execution_selection_reason'] = f'borderline_review_{chosen_name}'
        return _finalize_schedule(
            chosen_name=chosen_name,
            chosen_seq=chosen_seq,
            chosen_info=chosen_info,
            zero_info=zero_info,
            accepted=False,
            reason='borderline_review',
            total_dv_budget=total_dv_budget,
            trigger_case_label='borderline_review',
        )

    sane_nonzero = [item for item in repaired_candidates if item[0] != 'zero' and _sequence_is_structurally_sane(item[1])]
    ranked_source = sane_nonzero if sane_nonzero else repaired_candidates
    ranked = sorted(
        ranked_source,
        key=lambda item: (_best_effort_rank_key(item[0], item[1], item[2]), _internal_preview_sort_key(item[2])),
    )

    robust_item = next((item for item in ranked if str(item[0]) == "robust"), None)

    chosen_name, chosen_seq, chosen_info = ranked[0]
    chosen_info = dict(chosen_info)
    chosen_info = _attach_acceptance_gate_diagnostics(str(chosen_name), np.asarray(chosen_seq, dtype=float), chosen_info, zero_info)
    chosen_info = _attach_candidate_internal_preview(np.asarray(chosen_seq, dtype=float), chosen_info, x_est, n_orbit)
    if str(chosen_name) == "nominal" and robust_item is not None:
        chosen_info.update(_nominal_support_diagnostics(dict(chosen_info), dict(robust_item[2])))
    chosen_info['execution_selection_reason'] = f'best_effort_{chosen_name}'
    chosen_info['best_effort_preemptive_correction'] = 0
    return _finalize_schedule(
        chosen_name=chosen_name,
        chosen_seq=chosen_seq,
        chosen_info=chosen_info,
        zero_info=zero_info,
        accepted=False,
        reason='rejected',
        total_dv_budget=total_dv_budget,
        trigger_case_label=None,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_plan_table(
    trig: pd.Series,
    trigger_reason: str,
    schedule_nom: np.ndarray,
    schedule_rob: np.ndarray,
    info_nom: dict,
    info_rob: dict,
    dt: float,
    total_dv_budget: float,
) -> pd.DataFrame:
    rows = []
    trigger_t = float(trig.get("t_sec", 0.0))
    trigger_frame = _safe_int(trig.get("frame", -1))
    trigger_diag = _extract_trigger_diagnostics(trig)
    for strategy, seq, meta in [
        ("nominal", schedule_nom, info_nom),
        ("robust", schedule_rob, info_rob),
    ]:
        cum = 0.0
        for k, dv in enumerate(seq):
            mag = float(np.linalg.norm(dv))
            cum += mag
            rows.append({
                "strategy": strategy,
                "step_index": k,
                "apply_t_sec": trigger_t + k * dt,
                "trigger_t_sec": trigger_t,
                "trigger_frame": trigger_frame,
                "trigger_reason": trigger_reason,
                "trigger_safe_dist_px": float(trigger_diag.get("safe_dist_px", np.nan)),
                "trigger_safe_dist_rate_pxps": float(trigger_diag.get("safe_dist_rate_pxps", np.nan)),
                "trigger_short_time_to_warning_sec": float(trigger_diag.get("short_time_to_warning_sec", np.nan)),
                "trigger_short_time_to_breach_sec": float(trigger_diag.get("short_time_to_breach_sec", np.nan)),
                "trigger_risk_score": _safe_int(trigger_diag.get("risk_score", -1)),
                "trigger_warning_score": _safe_int(trigger_diag.get("warning_score", -1)),
                "trigger_risk_integral": float(trigger_diag.get("risk_integral", np.nan)),
                "trigger_warning_integral": float(trigger_diag.get("warning_integral", np.nan)),
                "trigger_short_pred_min_safe_margin_px": float(trigger_diag.get("short_pred_min_safe_margin_px", np.nan)),
                "trigger_pred_min_safe_margin_px": float(trigger_diag.get("pred_min_safe_margin_px", np.nan)),
                "trigger_est_range_m": float(trigger_diag.get("est_range_m", np.nan)),
                "dv_R_mps": float(dv[0]),
                "dv_T_mps": float(dv[1]),
                "dv_N_mps": float(dv[2]),
                "step_dv_mag_mps": mag,
                "cumulative_dv_mag_mps": cum,
                "total_dv_budget_mps": float(total_dv_budget),
                "plan_dt_sec": dt,
                "plan_horizon_sec": len(seq) * dt,
                "rollout_timing_semantics": ROLLOUT_TIMING_SEMANTICS,
                "planner_safe_ratio": SAFE_RATIO,
                "eval_safe_ratio": SAFE_RATIO,
                "pred_min_safe_margin_px": float(meta.get("min_safe_margin_px", np.nan)),
                "pred_terminal_margin_px": float(meta.get("terminal_margin_px", np.nan)),
                "pred_violation_count": _safe_int(meta.get("violation_count_pred", -1)),
                "pred_negative_margin_sum_px": float(meta.get("negative_margin_sum_px_pred", np.nan)),
                "pred_min_relative_distance_m": float(meta.get("min_relative_distance_m", np.nan)),
                "pred_terminal_relative_distance_m": float(meta.get("terminal_relative_distance_m", np.nan)),
                "pred_min_future_min_distance_m": float(meta.get("min_future_min_distance_m", np.nan)),
                "pred_terminal_future_min_distance_m": float(meta.get("terminal_future_min_distance_m", np.nan)),
                "pred_worst_risk_rank": _safe_int(meta.get("worst_risk_rank_pred", -999)),
                "plan_acceptance_reason": str(meta.get("plan_acceptance_reason", "")),
                "execution_selection_reason": str(meta.get("execution_selection_reason", meta.get("plan_acceptance_reason", ""))),
                "strictly_feasible_plan": _safe_int(meta.get("strictly_feasible_plan", _safe_int(bool(meta.get("plan_accepted", False))))),
                "strict_acceptance_passed": _safe_int(meta.get("strict_acceptance_passed", -1)),
                "acceptance_gate_safe": _safe_int(meta.get("acceptance_gate_safe", -1)),
                "acceptance_gate_structural": _safe_int(meta.get("acceptance_gate_structural", -1)),
                "acceptance_gate_information": _safe_int(meta.get("acceptance_gate_information", -1)),
                "acceptance_gate_geometry_non_degenerate": _safe_int(meta.get("acceptance_gate_geometry_non_degenerate", -1)),
                "acceptance_gate_future_distance": _safe_int(meta.get("acceptance_gate_future_distance", -1)),
                "acceptance_trace_ratio_vs_zero": float(meta.get("acceptance_trace_ratio_vs_zero", np.nan)),
                "acceptance_terminal_trace_ratio_vs_zero": float(meta.get("acceptance_terminal_trace_ratio_vs_zero", np.nan)),
                "acceptance_info_delta_vs_zero": float(meta.get("acceptance_info_delta_vs_zero", np.nan)),
                "acceptance_margin_gain_vs_zero_px": float(meta.get("acceptance_margin_gain_vs_zero_px", np.nan)),
                "acceptance_terminal_margin_gain_vs_zero_px": float(meta.get("acceptance_terminal_margin_gain_vs_zero_px", np.nan)),
                "acceptance_violation_delta_vs_zero": _safe_int(meta.get("acceptance_violation_delta_vs_zero", -1)),
                "acceptance_negative_margin_sum_delta_vs_zero_px": float(meta.get("acceptance_negative_margin_sum_delta_vs_zero_px", np.nan)),
                "acceptance_future_distance_gain_vs_zero_m": float(meta.get("acceptance_future_distance_gain_vs_zero_m", np.nan)),
                "acceptance_future_distance_non_regression": _safe_int(meta.get("acceptance_future_distance_non_regression", -1)),
                "acceptance_future_distance_gate_required": _safe_int(meta.get("acceptance_future_distance_gate_required", -1)),
                "planner_candidate_status": str(meta.get("planner_candidate_status", "")),
                "planner_acceptance_status": str(meta.get("planner_acceptance_status", "")),
                "planner_selection_status": str(meta.get("planner_selection_status", "")),
                "planner_initial_selection_status": str(meta.get("planner_initial_selection_status", "")),
                "final_selected_strategy": str(meta.get("final_selected_strategy", meta.get("selected_strategy", strategy))),
                "selection_correction_applied": _safe_int(meta.get("selection_correction_applied", 0)),
                "planner_execution_stage_status": str(meta.get("planner_execution_stage_status", meta.get("pre_execution_status", ""))),
                "pre_execution_status": str(meta.get("pre_execution_status", meta.get("planner_execution_stage_status", ""))),
                "execution_final_status": str(meta.get("execution_final_status", "")),
                "selected_strategy": str(meta.get("selected_strategy", strategy)),
                "case_label": str(meta.get("case_label", "normal_case")),
                "selected_nonzero_steps": _safe_int(meta.get("selected_nonzero_steps", -1)),
                "selected_total_dv_mag_mps": float(meta.get("selected_total_dv_mag_mps", np.nan)),
                "selected_dominant_ratio": float(meta.get("selected_dominant_ratio", np.nan)),
                "selected_span_steps": _safe_int(meta.get("selected_span_steps", -1)),
                "selected_structurally_sane": _safe_int(meta.get("selected_structurally_sane", -1)),
                "selected_future_distance_gain_vs_zero_m": float(meta.get("selected_future_distance_gain_vs_zero_m", np.nan)),
                "budget_worthiness_passed": _safe_int(meta.get("budget_worthiness_passed", -1)),
                "margin_gain_per_dv": float(meta.get("margin_gain_per_dv", np.nan)),
                "future_distance_gain_per_dv": float(meta.get("future_distance_gain_per_dv", np.nan)),
                "trace_penalty_per_dv": float(meta.get("trace_penalty_per_dv", np.nan)),
                "info_objective": str(meta.get("info_objective", INFO_OBJECTIVE)),
                "info_metric": float(meta.get("info_metric", np.nan)),
                "trace_pos_sum": float(meta.get("trace_pos_sum", np.nan)),
                "terminal_trace_pos": float(meta.get("terminal_trace_pos", np.nan)),
                "entropy_reduction": float(meta.get("entropy_reduction", np.nan)),
                "fim_trace_sum": float(meta.get("fim_trace_sum", np.nan)),
                "visible_update_steps": _safe_int(meta.get("visible_update_steps", -1)),
            })
    return pd.DataFrame(rows)


def _extract_strategy_plan_snapshot(plan_df: pd.DataFrame, strategy: str) -> dict[str, float | int | str]:
    use = plan_df.copy()
    if "strategy" in use.columns:
        rows = use[use["strategy"].astype(str).str.lower() == str(strategy).lower()].copy()
        if len(rows) > 0:
            use = rows
    if len(use) == 0:
        return {}

    first = use.iloc[0]
    mags = use["step_dv_mag_mps"].to_numpy(dtype=float) if "step_dv_mag_mps" in use.columns else np.zeros(len(use), dtype=float)
    nonzero_steps = int(np.count_nonzero(mags > 1.0e-12))
    total_dv_mag = float(np.sum(mags))
    dominant_ratio = float(np.max(mags) / max(total_dv_mag, 1.0e-12)) if nonzero_steps > 0 else 0.0
    nz_idx = np.where(mags > 1.0e-12)[0]
    span_steps = int(nz_idx[-1] - nz_idx[0]) if nz_idx.size >= 2 else 0

    def _f(key: str, default: float = float("nan")) -> float:
        try:
            val = float(first.get(key, default))
            return val if np.isfinite(val) else default
        except Exception:
            return default

    def _i(key: str, default: int = -1) -> int:
        try:
            return _safe_int(first.get(key, default))
        except Exception:
            return default

    def _s(key: str, default: str = "") -> str:
        try:
            return str(first.get(key, default))
        except Exception:
            return default

    return {
        "strategy": str(strategy),
        "pred_min_safe_margin_px": _f("pred_min_safe_margin_px"),
        "pred_terminal_margin_px": _f("pred_terminal_margin_px"),
        "pred_violation_count": _i("pred_violation_count"),
        "pred_negative_margin_sum_px": _f("pred_negative_margin_sum_px"),
        "pred_min_relative_distance_m": _f("pred_min_relative_distance_m"),
        "pred_terminal_relative_distance_m": _f("pred_terminal_relative_distance_m"),
        "pred_min_future_min_distance_m": _f("pred_min_future_min_distance_m"),
        "pred_terminal_future_min_distance_m": _f("pred_terminal_future_min_distance_m"),
        "pred_worst_risk_rank": _i("pred_worst_risk_rank", default=-999),
        "strict_acceptance_passed": _i("strict_acceptance_passed"),
        "acceptance_gate_safe": _i("acceptance_gate_safe"),
        "acceptance_gate_structural": _i("acceptance_gate_structural"),
        "acceptance_gate_information": _i("acceptance_gate_information"),
        "acceptance_gate_geometry_non_degenerate": _i("acceptance_gate_geometry_non_degenerate"),
        "acceptance_gate_future_distance": _i("acceptance_gate_future_distance"),
        "acceptance_trace_ratio_vs_zero": _f("acceptance_trace_ratio_vs_zero"),
        "acceptance_terminal_trace_ratio_vs_zero": _f("acceptance_terminal_trace_ratio_vs_zero"),
        "acceptance_info_delta_vs_zero": _f("acceptance_info_delta_vs_zero"),
        "acceptance_margin_gain_vs_zero_px": _f("acceptance_margin_gain_vs_zero_px"),
        "acceptance_terminal_margin_gain_vs_zero_px": _f("acceptance_terminal_margin_gain_vs_zero_px"),
        "acceptance_violation_delta_vs_zero": _i("acceptance_violation_delta_vs_zero"),
        "acceptance_negative_margin_sum_delta_vs_zero_px": _f("acceptance_negative_margin_sum_delta_vs_zero_px"),
        "acceptance_future_distance_gain_vs_zero_m": _f("acceptance_future_distance_gain_vs_zero_m"),
        "acceptance_future_distance_non_regression": _i("acceptance_future_distance_non_regression"),
        "acceptance_future_distance_gate_required": _i("acceptance_future_distance_gate_required"),
        "planner_candidate_status": _s("planner_candidate_status"),
        "planner_acceptance_status": _s("planner_acceptance_status"),
        "planner_selection_status": _s("planner_selection_status"),
        "planner_initial_selection_status": _s("planner_initial_selection_status"),
        "final_selected_strategy": _s("final_selected_strategy"),
        "selection_correction_applied": _i("selection_correction_applied", default=0),
        "planner_execution_stage_status": _s("planner_execution_stage_status"),
        "pre_execution_status": _s("pre_execution_status"),
        "plan_acceptance_reason": _s("plan_acceptance_reason"),
        "execution_selection_reason": _s("execution_selection_reason"),
        "case_label": _s("case_label"),
        "selected_future_distance_gain_vs_zero_m": _f("selected_future_distance_gain_vs_zero_m"),
        "budget_worthiness_passed": _i("budget_worthiness_passed"),
        "margin_gain_per_dv": _f("margin_gain_per_dv"),
        "future_distance_gain_per_dv": _f("future_distance_gain_per_dv"),
        "trace_penalty_per_dv": _f("trace_penalty_per_dv"),
        "info_metric": _f("info_metric"),
        "trace_pos_sum": _f("trace_pos_sum"),
        "terminal_trace_pos": _f("terminal_trace_pos"),
        "entropy_reduction": _f("entropy_reduction"),
        "fim_trace_sum": _f("fim_trace_sum"),
        "visible_update_steps": _i("visible_update_steps"),
        "nonzero_steps": nonzero_steps,
        "total_dv_mag_mps": total_dv_mag,
        "dominant_ratio": dominant_ratio,
        "span_steps": span_steps,
        "structurally_sane": int((nonzero_steps >= BEST_EFFORT_MIN_NONZERO_STEPS) and (dominant_ratio <= BEST_EFFORT_MAX_DOMINANT_RATIO) and (span_steps >= BEST_EFFORT_MIN_SPAN_STEPS or nonzero_steps <= 1)),
    }


def _load_or_run_imm(obs_df: pd.DataFrame, imm_csv: Path) -> pd.DataFrame:
    if imm_csv.exists():
        try:
            df = pd.read_csv(imm_csv)
            if len(df) > 0:
                return df
        except Exception:
            pass
    return imm_run(obs_df, imm_csv)


def _build_immediate_reject_plan(
    trig: pd.Series,
    x_est: np.ndarray,
    p_diag: np.ndarray,
    n_orbit: float,
    *,
    total_dv_budget: float,
    reason: str,
    case_label: str,
) -> tuple[np.ndarray, dict, np.ndarray, dict]:
    steps = max(int(round(PLAN_HORIZON_SEC / PLAN_DT)), 1)
    zero_seq = np.zeros((steps, 3), dtype=float)

    ev_nom = evaluate_schedule_exact(
        zero_seq,
        build_anchor_states(trig, x_est, p_diag, robust=False),
        PLAN_DT,
        n_orbit,
        p_diag=p_diag,
    )
    ev_rob = evaluate_schedule_exact(
        zero_seq,
        build_anchor_states(trig, x_est, p_diag, robust=True),
        PLAN_DT,
        n_orbit,
        p_diag=p_diag,
    )

    common = {
        "plan_accepted": False,
        "plan_acceptance_reason": str(reason),
        "execution_selection_reason": str(reason),
        "strictly_feasible_plan": 0,
        "case_label": str(case_label),
        "total_dv_budget_mps": float(total_dv_budget),
        "selected_nonzero_steps": 0,
        "selected_total_dv_mag_mps": 0.0,
        "selected_dominant_ratio": 0.0,
        "selected_span_steps": 0,
        "selected_structurally_sane": 0,
        "acceptance_gate_safe": 0,
        "acceptance_gate_structural": 0,
        "acceptance_gate_information": 0,
        "acceptance_gate_geometry_non_degenerate": 0,
        "acceptance_gate_future_distance": 0,
        "acceptance_trace_ratio_vs_zero": 1.0,
        "acceptance_terminal_trace_ratio_vs_zero": 1.0,
        "acceptance_info_delta_vs_zero": 0.0,
        "strict_acceptance_passed": 0,
        "borderline_margin_gain_px": 0.0,
        "borderline_material_margin_gain": 0,
        "borderline_material_abs_margin": 0,
        "borderline_info_not_too_bad": 0,
        "nominal_hard_gate_passed": 0,
        "borderline_worth_execution": 0,
        "budget_retry_used": 0,
        "retry_from_case_label": "",
        "max_retry_budget_reached": 0,
        "info_objective": INFO_OBJECTIVE,
    }

    info_nom = _update_info_from_eval({**common, "selected_strategy": "nominal"}, ev_nom)
    info_rob = _update_info_from_eval({**common, "selected_strategy": "robust"}, ev_rob)
    return zero_seq.copy(), info_nom, zero_seq.copy(), info_rob


def plan_active_sensing(
    input_csv: Path | str = INPUT_CSV,
    out_dir: Path | str = OUT_DIR,
    experiment_mode: str = EXPERIMENT_MODE,
    imm_csv: Path | str | None = None,
    orig_last_risk_level: str = "",
    orig_max_pc_bplane: float = float("nan"),
    orig_min_future_min_distance_m: float = float("nan"),
) -> PlanResult:
    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if imm_csv is None:
        imm_csv = input_csv.with_name("imm_ukf_estimation_result.csv")
    imm_csv = Path(imm_csv)

    obs_df = pd.read_csv(input_csv)
    imm_df = _load_or_run_imm(obs_df, imm_csv)
    n_orbit = estimate_mean_motion_from_truth(obs_df)

    if experiment_mode != "exp3":
        raise NotImplementedError("This planner file now focuses on Experiment III only.")

    trig, trigger_reason = choose_trigger_row_exp3(
        imm_df,
        n_orbit,
        orig_last_risk_level=orig_last_risk_level,
        orig_max_pc_bplane=orig_max_pc_bplane,
        orig_min_future_min_distance_m=orig_min_future_min_distance_m,
    )
    trigger_diag = _extract_trigger_diagnostics(trig)
    x_est = np.array([
        float(trig["R_est_m"]), float(trig["T_est_m"]), float(trig["N_est_m"]),
        float(trig["dR_est_mps"]), float(trig["dT_est_mps"]), float(trig["dN_est_mps"]),
    ], dtype=float)
    p_diag = extract_planner_cov_diag_from_trigger_row(trig)

    pred_min_margin_px = float(trig.get("pred_min_safe_margin_px", np.nan))
    budget_ladder, trigger_case_label = choose_budget_ladder(pred_min_margin_px)

    chosen_schedule_nom = None
    chosen_info_nom = None
    chosen_schedule_rob = None
    chosen_info_rob = None
    chosen_exec_schedule = None
    chosen_exec_info = None
    used_budget = None
    retry_used = False

    late_trigger = (float(trig.get("t_sec", -1.0)) >= LATE_TRIGGER_SEC) or (_safe_int(trig.get("frame", -1)) >= LATE_TRIGGER_FRAME)
    unobservable_trigger = not np.isfinite(pred_min_margin_px)

    # Keep cache valid across budget ladder attempts within one case.
    _clear_eval_cache()
    trig_payload = {
        str(k): (v.item() if hasattr(v, "item") else v)
        for k, v in trig.to_dict().items()
    }
    strategy_executor: ProcessPoolExecutor | None = None
    if bool(PARALLEL_STRATEGY_SOLVE_ENABLE):
        try:
            strategy_executor = ProcessPoolExecutor(
                **_process_pool_kwargs(max(1, int(PARALLEL_STRATEGY_MAX_WORKERS)))
            )
        except Exception as exc:
            print(f"[WARN] strategy-level process parallel unavailable ({type(exc).__name__}: {exc}); fallback to sequential")
            strategy_executor = None

    def _run_one_budget(total_dv_budget: float, label_for_case: str):
        nonlocal strategy_executor
        full_search_profile = _search_profile_for_case(str(label_for_case), float(total_dv_budget), fast_mode=False)
        if bool(FAST_SEARCH_MODE):
            nominal_search_profile = dict(full_search_profile)
            robust_search_profile = _search_profile_for_case(str(label_for_case), float(total_dv_budget), fast_mode=True)
        else:
            nominal_search_profile = dict(full_search_profile)
            robust_search_profile = dict(full_search_profile)

        print(
            "search profile nominal:",
            f"horizon={float(nominal_search_profile['plan_horizon_sec']):.1f}s,",
            f"active_steps={int(nominal_search_profile['search_active_steps'])},",
            f"seed_topk={int(nominal_search_profile['multistep_seed_topk'])},",
            f"multistart={int(nominal_search_profile['random_multistart_count'])}"
        )
        if bool(FAST_SEARCH_MODE):
            print(
                "search profile robust(light):",
                f"horizon={float(robust_search_profile['plan_horizon_sec']):.1f}s,",
                f"active_steps={int(robust_search_profile['search_active_steps'])},",
                f"seed_topk={int(robust_search_profile['multistep_seed_topk'])},",
                f"multistart={int(robust_search_profile['random_multistart_count'])}"
            )
        print(f"尝试总预算 total_dv_budget = {total_dv_budget:.3f} m/s")

        def _apply_preemptive_nominal_correction(
            exec_schedule_in: np.ndarray,
            exec_info_in: dict,
            info_nom_in: dict,
            info_rob_in: dict,
        ) -> tuple[np.ndarray, dict]:
            del info_nom_in
            del info_rob_in
            return np.asarray(exec_schedule_in, dtype=float).copy(), dict(exec_info_in)

        schedule_nom_local = None
        info_nom_local = None
        schedule_rob_local = None
        info_rob_local = None

        if strategy_executor is not None:
            worker_seed_parallel = bool(PARALLEL_SEED_SOLVE_ENABLE) and bool(ALLOW_NESTED_PROCESS_PARALLEL)
            try:
                print("求解标称/鲁棒计划（进程并行）...")
                fut_nom = strategy_executor.submit(
                    _solve_strategy_worker,
                    "nominal",
                    trig_payload,
                    x_est,
                    p_diag,
                    float(n_orbit),
                    float(total_dv_budget),
                    nominal_search_profile,
                    worker_seed_parallel,
                )
                fut_rob = strategy_executor.submit(
                    _solve_strategy_worker,
                    "robust",
                    trig_payload,
                    x_est,
                    p_diag,
                    float(n_orbit),
                    float(total_dv_budget),
                    robust_search_profile,
                    worker_seed_parallel,
                )
                _, schedule_nom_local, info_nom_local = fut_nom.result()
                _, schedule_rob_local, info_rob_local = fut_rob.result()
            except Exception as exc:
                print(f"[WARN] strategy-level parallel failed ({type(exc).__name__}: {exc}); fallback to sequential")
                try:
                    strategy_executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                strategy_executor = None

        if schedule_nom_local is None or info_nom_local is None or schedule_rob_local is None or info_rob_local is None:
            with _temporary_search_profile(nominal_search_profile):
                print("求解标称计划（exact direct shooting）...")
                schedule_nom_local, info_nom_local = solve_nominal_schedule(
                    x_est,
                    p_diag,
                    n_orbit,
                    total_dv_budget,
                    enable_seed_parallel=bool(PARALLEL_SEED_SOLVE_ENABLE),
                )

            with _temporary_search_profile(robust_search_profile):
                print("求解鲁棒计划（多锚点 / 全时域 exact image-plane search）...")
                schedule_rob_local, info_rob_local = solve_robust_schedule_exp3(
                    trig,
                    x_est,
                    p_diag,
                    n_orbit,
                    total_dv_budget,
                    enable_seed_parallel=bool(PARALLEL_SEED_SOLVE_ENABLE),
                )

        exec_schedule_local, exec_info_local = _select_execution_plan(
            trig, x_est, p_diag, n_orbit,
            schedule_nom_local, info_nom_local,
            schedule_rob_local, info_rob_local,
            total_dv_budget=total_dv_budget,
        )
        exec_schedule_local, exec_info_local = _apply_preemptive_nominal_correction(
            exec_schedule_local,
            exec_info_local,
            info_nom_local,
            info_rob_local,
        )

        if bool(FAST_SEARCH_MODE):
            need_robust_upgrade = str(exec_info_local.get("selected_strategy", "")).lower() == "robust"
            if need_robust_upgrade and (robust_search_profile != full_search_profile):
                print("fast search mode: robust second-stage full-profile refinement")
                upgraded_schedule_rob = None
                upgraded_info_rob = None
                if strategy_executor is not None:
                    worker_seed_parallel = bool(PARALLEL_SEED_SOLVE_ENABLE) and bool(ALLOW_NESTED_PROCESS_PARALLEL)
                    try:
                        fut_rob_full = strategy_executor.submit(
                            _solve_strategy_worker,
                            "robust",
                            trig_payload,
                            x_est,
                            p_diag,
                            float(n_orbit),
                            float(total_dv_budget),
                            full_search_profile,
                            worker_seed_parallel,
                        )
                        _, upgraded_schedule_rob, upgraded_info_rob = fut_rob_full.result()
                    except Exception as exc:
                        print(f"[WARN] fast robust second-stage parallel failed ({type(exc).__name__}: {exc}); fallback to sequential")
                        try:
                            strategy_executor.shutdown(wait=False, cancel_futures=True)
                        except Exception:
                            pass
                        strategy_executor = None

                if upgraded_schedule_rob is None or upgraded_info_rob is None:
                    with _temporary_search_profile(full_search_profile):
                        upgraded_schedule_rob, upgraded_info_rob = solve_robust_schedule_exp3(
                            trig,
                            x_est,
                            p_diag,
                            n_orbit,
                            total_dv_budget,
                            enable_seed_parallel=bool(PARALLEL_SEED_SOLVE_ENABLE),
                        )

                schedule_rob_local = np.asarray(upgraded_schedule_rob, dtype=float).copy()
                info_rob_local = dict(upgraded_info_rob)
                exec_schedule_local, exec_info_local = _select_execution_plan(
                    trig, x_est, p_diag, n_orbit,
                    schedule_nom_local, info_nom_local,
                    schedule_rob_local, info_rob_local,
                    total_dv_budget=total_dv_budget,
                )
                exec_schedule_local, exec_info_local = _apply_preemptive_nominal_correction(
                    exec_schedule_local,
                    exec_info_local,
                    info_nom_local,
                    info_rob_local,
                )
        exec_info_local["total_dv_budget_mps"] = float(total_dv_budget)
        exec_info_local.setdefault("case_label", str(label_for_case))
        exec_info_local.update(_budget_efficiency_diagnostics(exec_info_local))
        return (
            schedule_nom_local,
            info_nom_local,
            schedule_rob_local,
            info_rob_local,
            exec_schedule_local,
            exec_info_local,
        )

    if LATE_TRIGGER_SKIP_ENABLE and late_trigger:
        immediate_reason = "late_hard_case"
        chosen_schedule_nom, chosen_info_nom, chosen_schedule_rob, chosen_info_rob = _build_immediate_reject_plan(
            trig, x_est, p_diag, n_orbit,
            total_dv_budget=float(budget_ladder[0]),
            reason=immediate_reason,
            case_label=immediate_reason,
        )
        chosen_exec_schedule = np.asarray(chosen_schedule_rob, dtype=float).copy()
        chosen_exec_info = dict(chosen_info_rob)
        used_budget = float(budget_ladder[0])
    elif UNOBSERVABLE_TRIGGER_SKIP_ENABLE and unobservable_trigger:
        immediate_reason = "unobservable_hard_case"
        chosen_schedule_nom, chosen_info_nom, chosen_schedule_rob, chosen_info_rob = _build_immediate_reject_plan(
            trig, x_est, p_diag, n_orbit,
            total_dv_budget=float(budget_ladder[0]),
            reason=immediate_reason,
            case_label=immediate_reason,
        )
        chosen_exec_schedule = np.asarray(chosen_schedule_rob, dtype=float).copy()
        chosen_exec_info = dict(chosen_info_rob)
        used_budget = float(budget_ladder[0])

    if chosen_exec_info is None:
        for total_dv_budget in budget_ladder:
            schedule_nom, info_nom, schedule_rob, info_rob, exec_schedule, exec_info = _run_one_budget(total_dv_budget, trigger_case_label)
            chosen_schedule_nom = schedule_nom
            chosen_info_nom = info_nom
            chosen_schedule_rob = schedule_rob
            chosen_info_rob = info_rob
            chosen_exec_schedule = exec_schedule
            chosen_exec_info = exec_info
            used_budget = float(total_dv_budget)
            if bool(exec_info.get("plan_accepted", False)):
                break

    if chosen_exec_info is None:
        if strategy_executor is not None:
            try:
                strategy_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            strategy_executor = None
        raise RuntimeError("Planner failed before generating any candidate schedule.")

    if should_try_budget_retry(
        bool(chosen_exec_info.get("plan_accepted", False)),
        trigger_case_label,
        used_budget,
        final_pred_margin_px=float(chosen_exec_info.get("min_safe_margin_px", np.nan)),
        final_violation_count=_safe_int(chosen_exec_info.get("violation_count_pred", -1)),
    ):
        for total_dv_budget in BUDGET_PROFILE.retry_sequence:
            if used_budget is not None and total_dv_budget <= float(used_budget) + 1e-9:
                continue
            print(f"budget-limited retry: total_dv_budget = {total_dv_budget:.3f} m/s")
            schedule_nom, info_nom, schedule_rob, info_rob, exec_schedule, exec_info = _run_one_budget(total_dv_budget, "budget_retry_case")
            exec_info["retry_from_case_label"] = str(trigger_case_label)
            exec_info["budget_retry_used"] = 1
            chosen_schedule_nom = schedule_nom
            chosen_info_nom = info_nom
            chosen_schedule_rob = schedule_rob
            chosen_info_rob = info_rob
            chosen_exec_schedule = exec_schedule
            chosen_exec_info = exec_info
            used_budget = float(total_dv_budget)
            retry_used = True
            if bool(exec_info.get("plan_accepted", False)):
                break

    if not bool(chosen_exec_info.get("plan_accepted", False)):
        existing_reason = str(chosen_exec_info.get("plan_acceptance_reason", ""))
        if existing_reason == "borderline_review":
            chosen_exec_info["case_label"] = existing_reason
        else:
            fail_label = classify_failed_case(
                trig,
                pred_min_margin_px,
                used_budget if used_budget is not None else DV_MAG_BASE,
                selected_info=chosen_exec_info,
                nominal_info=chosen_info_nom,
                robust_info=chosen_info_rob,
            )
            chosen_exec_info["plan_acceptance_reason"] = "rejected"
            chosen_exec_info["acceptance_diagnostics_reason"] = fail_label
            chosen_exec_info["case_label"] = fail_label

        acceptance_status = "borderline_review" if str(chosen_exec_info.get("plan_acceptance_reason", "")) == "borderline_review" else "rejected"
        chosen_exec_info["acceptance_status"] = acceptance_status
        chosen_exec_info["planner_acceptance_status"] = acceptance_status
        chosen_exec_info["strict_acceptance_passed"] = 0
        chosen_exec_info["strictly_feasible_plan"] = 0
        if retry_used:
            chosen_exec_info["retry_from_case_label"] = str(trigger_case_label)
            chosen_exec_info["budget_retry_used"] = 1

    chosen_exec_info["max_retry_budget_reached"] = int(
        (used_budget is not None) and (used_budget >= max(BUDGET_PROFILE.retry_sequence) - 1e-9) and (
            not bool(chosen_exec_info.get("plan_accepted", False)))
    )

    if (
        chosen_schedule_nom is None
        or chosen_info_nom is None
        or chosen_schedule_rob is None
        or chosen_info_rob is None
        or chosen_exec_schedule is None
        or chosen_exec_info is None
    ):
        raise RuntimeError("Planner failed to retain full nominal/robust/selected candidate snapshots.")

    schedule_nom = np.asarray(chosen_schedule_nom, dtype=float).copy()
    schedule_rob = np.asarray(chosen_schedule_rob, dtype=float).copy()
    chosen_exec_schedule = np.asarray(chosen_exec_schedule, dtype=float).copy()

    total_dv_budget = float(chosen_exec_info.get("total_dv_budget_mps", used_budget if used_budget is not None else DV_MAG_BASE))
    zero_seq_support, zero_info_support = _evaluate_zero_baseline(trig, x_est, p_diag, n_orbit, robust=True)
    zero_info_support = _attach_acceptance_gate_diagnostics("zero", zero_seq_support, zero_info_support, zero_info_support)

    info_nom = _attach_acceptance_gate_diagnostics("nominal", schedule_nom, dict(chosen_info_nom), zero_info_support)
    info_rob = _attach_acceptance_gate_diagnostics("robust", schedule_rob, dict(chosen_info_rob), zero_info_support)

    selected_name = str(chosen_exec_info.get("selected_strategy", "robust"))
    chosen_meta = _attach_acceptance_gate_diagnostics(selected_name, chosen_exec_schedule, dict(chosen_exec_info), zero_info_support)
    chosen_meta["total_dv_budget_mps"] = float(total_dv_budget)
    chosen_meta.setdefault("execution_selection_reason", str(chosen_meta.get("plan_acceptance_reason", "")))
    initial_selection_status = str(
        chosen_meta.get(
            "planner_initial_selection_status",
            chosen_meta.get("planner_selection_status", f"selected_{str(selected_name).lower()}"),
        )
    )
    final_selected_strategy = str(
        chosen_meta.get("final_selected_strategy", chosen_meta.get("selected_strategy", selected_name))
    ).lower()
    if final_selected_strategy not in {"robust", "nominal", "zero"}:
        fallback_selected = str(selected_name).lower()
        final_selected_strategy = fallback_selected if fallback_selected in {"robust", "nominal", "zero"} else "robust"
    chosen_meta["planner_initial_selection_status"] = initial_selection_status
    chosen_meta["final_selected_strategy"] = final_selected_strategy
    chosen_meta["selected_strategy"] = final_selected_strategy
    chosen_meta["planner_selection_status"] = f"selected_{final_selected_strategy}"
    chosen_meta["selection_correction_applied"] = int(
        initial_selection_status.strip().lower() != f"selected_{final_selected_strategy}"
    )
    chosen_meta["execution_request_strategy"] = final_selected_strategy
    chosen_meta["execution_request_source_reason"] = str(
        chosen_meta.get("execution_selection_reason", "")
    )
    chosen_meta["rollout_timing_semantics"] = ROLLOUT_TIMING_SEMANTICS
    try:
        orig_pc_meta = float(orig_max_pc_bplane)
    except Exception:
        orig_pc_meta = float("nan")
    try:
        orig_future_meta = float(orig_min_future_min_distance_m)
    except Exception:
        orig_future_meta = float("nan")
    chosen_meta["orig_last_risk_level"] = str(orig_last_risk_level)
    chosen_meta["orig_max_pc_bplane"] = float(orig_pc_meta) if np.isfinite(orig_pc_meta) else float("nan")
    chosen_meta["orig_min_future_min_distance_m"] = float(orig_future_meta) if np.isfinite(orig_future_meta) else float("nan")
    for key, value in trigger_diag.items():
        chosen_meta[f"trigger_{key}"] = value

    selected_strategy = str(chosen_meta.get("selected_strategy", "robust"))
    for info in (info_nom, info_rob):
        info["selected_strategy"] = selected_strategy
        info.setdefault("plan_acceptance_reason", "")
        info.setdefault("execution_selection_reason", "")
        info.setdefault("case_label", str(chosen_meta.get("case_label", "normal_case")))
        info.setdefault("strictly_feasible_plan", _safe_int(bool(chosen_meta.get("plan_accepted", False))))
        info.setdefault("info_objective", INFO_OBJECTIVE)
        info["total_dv_budget_mps"] = float(total_dv_budget)
        if "budget_retry_used" in chosen_meta:
            info.setdefault("budget_retry_used", _safe_int(chosen_meta.get("budget_retry_used", 0)))
            info.setdefault("retry_from_case_label", str(chosen_meta.get("retry_from_case_label", "")))

    def _attach_candidate_snapshot(prefix: str, seq: np.ndarray, info: dict) -> None:
        nonzero_steps, total_dv_mag, dominant_ratio = _sequence_structure_stats(np.asarray(seq, dtype=float))
        span_steps = _sequence_step_span(np.asarray(seq, dtype=float))
        chosen_meta[f"{prefix}_pred_min_safe_margin_px"] = float(info.get("min_safe_margin_px", np.nan))
        chosen_meta[f"{prefix}_pred_terminal_margin_px"] = float(info.get("terminal_margin_px", np.nan))
        chosen_meta[f"{prefix}_pred_violation_count"] = _safe_int(info.get("violation_count_pred", -1))
        chosen_meta[f"{prefix}_pred_negative_margin_sum_px"] = float(info.get("negative_margin_sum_px_pred", np.nan))
        chosen_meta[f"{prefix}_pred_min_relative_distance_m"] = float(info.get("min_relative_distance_m", np.nan))
        chosen_meta[f"{prefix}_pred_terminal_relative_distance_m"] = float(info.get("terminal_relative_distance_m", np.nan))
        chosen_meta[f"{prefix}_pred_min_future_min_distance_m"] = float(info.get("min_future_min_distance_m", np.nan))
        chosen_meta[f"{prefix}_pred_terminal_future_min_distance_m"] = float(info.get("terminal_future_min_distance_m", np.nan))
        chosen_meta[f"{prefix}_pred_worst_risk_rank"] = _safe_int(info.get("worst_risk_rank_pred", -999))
        chosen_meta[f"{prefix}_strict_acceptance_passed"] = _safe_int(info.get("strict_acceptance_passed", -1))
        chosen_meta[f"{prefix}_acceptance_gate_safe"] = _safe_int(info.get("acceptance_gate_safe", -1))
        chosen_meta[f"{prefix}_acceptance_gate_structural"] = _safe_int(info.get("acceptance_gate_structural", -1))
        chosen_meta[f"{prefix}_acceptance_gate_information"] = _safe_int(info.get("acceptance_gate_information", -1))
        chosen_meta[f"{prefix}_acceptance_gate_geometry_non_degenerate"] = _safe_int(info.get("acceptance_gate_geometry_non_degenerate", -1))
        chosen_meta[f"{prefix}_acceptance_gate_future_distance"] = _safe_int(info.get("acceptance_gate_future_distance", -1))
        chosen_meta[f"{prefix}_trace_ratio_vs_zero"] = float(info.get("acceptance_trace_ratio_vs_zero", np.nan))
        chosen_meta[f"{prefix}_terminal_trace_ratio_vs_zero"] = float(info.get("acceptance_terminal_trace_ratio_vs_zero", np.nan))
        chosen_meta[f"{prefix}_info_delta_vs_zero"] = float(info.get("acceptance_info_delta_vs_zero", np.nan))
        chosen_meta[f"{prefix}_future_distance_gain_vs_zero_m"] = float(info.get("acceptance_future_distance_gain_vs_zero_m", np.nan))
        chosen_meta[f"{prefix}_future_distance_non_regression"] = _safe_int(info.get("acceptance_future_distance_non_regression", -1))
        chosen_meta[f"{prefix}_internal_preview_min_future_distance_m"] = float(info.get("internal_preview_min_future_distance_m", np.nan))
        chosen_meta[f"{prefix}_internal_preview_worst_risk_rank"] = _safe_int(info.get("internal_preview_worst_risk_rank", -999))
        chosen_meta[f"{prefix}_nonzero_steps"] = int(nonzero_steps)
        chosen_meta[f"{prefix}_total_dv_mag_mps"] = float(total_dv_mag)
        chosen_meta[f"{prefix}_dominant_ratio"] = float(dominant_ratio)
        chosen_meta[f"{prefix}_span_steps"] = int(span_steps)
        chosen_meta[f"{prefix}_structurally_sane"] = int(_sequence_is_structurally_sane(np.asarray(seq, dtype=float)))
        if str(prefix) != "selected":
            chosen_meta[f"{prefix}_budget_worthiness_passed"] = _safe_int(info.get("budget_worthiness_passed", -1))
            chosen_meta[f"{prefix}_margin_gain_per_dv"] = float(info.get("margin_gain_per_dv", np.nan))
            chosen_meta[f"{prefix}_future_distance_gain_per_dv"] = float(info.get("future_distance_gain_per_dv", np.nan))
            chosen_meta[f"{prefix}_trace_penalty_per_dv"] = float(info.get("trace_penalty_per_dv", np.nan))
        chosen_meta[f"{prefix}_plan_acceptance_reason"] = str(info.get("plan_acceptance_reason", ""))
        chosen_meta[f"{prefix}_execution_selection_reason"] = str(info.get("execution_selection_reason", ""))

    _attach_candidate_snapshot("nominal", schedule_nom, info_nom)
    _attach_candidate_snapshot("robust", schedule_rob, info_rob)

    selected_seq_map = {
        "nominal": np.asarray(schedule_nom, dtype=float).copy(),
        "robust": np.asarray(schedule_rob, dtype=float).copy(),
        "zero": np.asarray(zero_seq_support, dtype=float).copy(),
    }
    selected_info_map = {
        "nominal": dict(info_nom),
        "robust": dict(info_rob),
        "zero": dict(zero_info_support),
    }
    selected_snapshot_strategy = str(chosen_meta.get("final_selected_strategy", chosen_meta.get("selected_strategy", selected_name))).lower()
    if selected_snapshot_strategy not in selected_seq_map:
        selected_snapshot_strategy = str(selected_name).lower()
    if selected_snapshot_strategy not in selected_seq_map:
        selected_snapshot_strategy = "robust"
    selected_snapshot_seq = np.asarray(selected_seq_map.get(selected_snapshot_strategy, chosen_exec_schedule), dtype=float).copy()
    selected_snapshot_info = dict(selected_info_map.get(selected_snapshot_strategy, chosen_meta))
    selected_snapshot_info = _attach_acceptance_gate_diagnostics(
        selected_snapshot_strategy, selected_snapshot_seq, selected_snapshot_info, zero_info_support
    )
    selected_snapshot_info.update({
        "plan_accepted": bool(chosen_meta.get("plan_accepted", False)),
        "plan_acceptance_reason": str(chosen_meta.get("plan_acceptance_reason", "")),
        "execution_selection_reason": str(chosen_meta.get("execution_selection_reason", chosen_meta.get("plan_acceptance_reason", ""))),
        "planner_candidate_status": str(chosen_meta.get("planner_candidate_status", f"candidate_{selected_snapshot_strategy}")),
        "planner_acceptance_status": str(chosen_meta.get("planner_acceptance_status", "")),
        "planner_initial_selection_status": str(chosen_meta.get("planner_initial_selection_status", f"selected_{selected_snapshot_strategy}")),
        "planner_selection_status": str(chosen_meta.get("planner_selection_status", f"selected_{selected_snapshot_strategy}")),
        "final_selected_strategy": str(chosen_meta.get("final_selected_strategy", selected_snapshot_strategy)),
        "selected_strategy": str(chosen_meta.get("selected_strategy", selected_snapshot_strategy)),
        "selection_correction_applied": _safe_int(chosen_meta.get("selection_correction_applied", 0)),
        "case_label": str(chosen_meta.get("case_label", "")),
        "total_dv_budget_mps": float(chosen_meta.get("total_dv_budget_mps", total_dv_budget)),
    })
    _attach_candidate_snapshot("selected", selected_snapshot_seq, selected_snapshot_info)

    chosen_meta["selected_snapshot_bound_strategy"] = selected_snapshot_strategy
    chosen_meta["selected_pred_min_safe_margin_px"] = float(selected_snapshot_info.get("min_safe_margin_px", np.nan))
    chosen_meta["selected_pred_terminal_margin_px"] = float(selected_snapshot_info.get("terminal_margin_px", np.nan))
    chosen_meta["selected_pred_violation_count"] = _safe_int(selected_snapshot_info.get("violation_count_pred", -1))
    chosen_meta["selected_pred_negative_margin_sum_px"] = float(selected_snapshot_info.get("negative_margin_sum_px_pred", np.nan))
    chosen_meta["selected_pred_min_future_min_distance_m"] = float(selected_snapshot_info.get("min_future_min_distance_m", np.nan))
    chosen_meta["selected_pred_terminal_future_min_distance_m"] = float(selected_snapshot_info.get("terminal_future_min_distance_m", np.nan))
    selected_nonzero_steps, selected_total_dv_mag, selected_dominant_ratio = _sequence_structure_stats(selected_snapshot_seq)
    chosen_meta["selected_nonzero_steps"] = int(selected_nonzero_steps)
    chosen_meta["selected_total_dv_mag_mps"] = float(selected_total_dv_mag)
    chosen_meta["selected_dominant_ratio"] = float(selected_dominant_ratio)
    chosen_meta["selected_span_steps"] = int(_sequence_step_span(selected_snapshot_seq))
    chosen_meta["selected_structurally_sane"] = int(_sequence_is_structurally_sane(selected_snapshot_seq))
    for gate_key in [
        "acceptance_gate_safe",
        "acceptance_gate_structural",
        "acceptance_gate_information",
        "acceptance_gate_geometry_non_degenerate",
        "acceptance_gate_future_distance",
    ]:
        chosen_meta[f"selected_{gate_key}"] = _safe_int(selected_snapshot_info.get(gate_key, -1))
    for src_key, dst_key in [
        ("acceptance_trace_ratio_vs_zero", "selected_acceptance_trace_ratio_vs_zero"),
        ("acceptance_terminal_trace_ratio_vs_zero", "selected_acceptance_terminal_trace_ratio_vs_zero"),
        ("acceptance_info_delta_vs_zero", "selected_acceptance_info_delta_vs_zero"),
        ("acceptance_margin_gain_vs_zero_px", "selected_acceptance_margin_gain_vs_zero_px"),
        ("acceptance_terminal_margin_gain_vs_zero_px", "selected_acceptance_terminal_margin_gain_vs_zero_px"),
        ("acceptance_violation_delta_vs_zero", "selected_acceptance_violation_delta_vs_zero"),
        ("acceptance_negative_margin_sum_delta_vs_zero_px", "selected_acceptance_negative_margin_sum_delta_vs_zero_px"),
        ("acceptance_future_distance_gain_vs_zero_m", "selected_acceptance_future_distance_gain_vs_zero_m"),
        ("acceptance_future_distance_non_regression", "selected_acceptance_future_distance_non_regression"),
        ("acceptance_future_distance_gate_required", "selected_acceptance_future_distance_gate_required"),
    ]:
        chosen_meta[dst_key] = selected_snapshot_info.get(src_key, np.nan)

    selected_future = float(chosen_meta.get("selected_pred_min_future_min_distance_m", np.nan))
    zero_future = float(chosen_meta.get("zero_pred_min_future_distance_m", np.nan))
    if np.isfinite(selected_future) and np.isfinite(zero_future):
        selected_future_gain = float(selected_future - zero_future)
    else:
        selected_future_gain = float(
            chosen_meta.get(
                "selected_future_distance_gain_vs_zero_m",
                chosen_meta.get("selected_acceptance_future_distance_gain_vs_zero_m", np.nan),
            )
        )
    chosen_meta["selected_future_distance_gain_vs_zero_m"] = float(selected_future_gain)
    chosen_meta["selected_acceptance_future_distance_gain_vs_zero_m"] = float(selected_future_gain)

    for gate_key in [
        "acceptance_gate_safe",
        "acceptance_gate_structural",
        "acceptance_gate_information",
        "acceptance_gate_geometry_non_degenerate",
        "acceptance_gate_future_distance",
    ]:
        chosen_meta[f"selected_{gate_key}"] = int(
            chosen_meta.get(f"selected_{gate_key}", chosen_meta.get(gate_key, -1))
        )

    chosen_meta["nominal_support_min_safe_margin_px"] = float(chosen_meta.get("nominal_pred_min_safe_margin_px", np.nan))
    chosen_meta["nominal_support_violation_count_pred"] = _safe_int(chosen_meta.get("nominal_pred_violation_count", -1))
    chosen_meta["nominal_support_strict_acceptance_passed"] = _safe_int(chosen_meta.get("nominal_strict_acceptance_passed", -1))
    chosen_meta["nominal_support_acceptance_gate_safe"] = _safe_int(chosen_meta.get("nominal_acceptance_gate_safe", -1))
    chosen_meta["nominal_support_acceptance_gate_structural"] = _safe_int(chosen_meta.get("nominal_acceptance_gate_structural", -1))
    chosen_meta["nominal_support_acceptance_gate_information"] = _safe_int(chosen_meta.get("nominal_acceptance_gate_information", -1))
    chosen_meta["nominal_support_acceptance_gate_geometry_non_degenerate"] = _safe_int(chosen_meta.get("nominal_acceptance_gate_geometry_non_degenerate", -1))
    chosen_meta["nominal_support_acceptance_gate_future_distance"] = _safe_int(chosen_meta.get("nominal_acceptance_gate_future_distance", -1))

    chosen_meta["robust_support_min_safe_margin_px"] = float(chosen_meta.get("robust_pred_min_safe_margin_px", np.nan))
    chosen_meta["robust_support_violation_count_pred"] = _safe_int(chosen_meta.get("robust_pred_violation_count", -1))
    chosen_meta["robust_support_strict_acceptance_passed"] = _safe_int(chosen_meta.get("robust_strict_acceptance_passed", -1))
    chosen_meta["robust_support_acceptance_gate_safe"] = _safe_int(chosen_meta.get("robust_acceptance_gate_safe", -1))
    chosen_meta["robust_support_acceptance_gate_structural"] = _safe_int(chosen_meta.get("robust_acceptance_gate_structural", -1))
    chosen_meta["robust_support_acceptance_gate_information"] = _safe_int(chosen_meta.get("robust_acceptance_gate_information", -1))
    chosen_meta["robust_support_acceptance_gate_geometry_non_degenerate"] = _safe_int(chosen_meta.get("robust_acceptance_gate_geometry_non_degenerate", -1))
    chosen_meta["robust_support_acceptance_gate_future_distance"] = _safe_int(chosen_meta.get("robust_acceptance_gate_future_distance", -1))

    plan_df = build_plan_table(
        trig=trig,
        trigger_reason=trigger_reason,
        schedule_nom=schedule_nom,
        schedule_rob=schedule_rob,
        info_nom=info_nom,
        info_rob=info_rob,
        dt=PLAN_DT,
        total_dv_budget=total_dv_budget,
    )

    plan_csv = out_dir / "micro_dv_plan.csv"
    plan_df.to_csv(plan_csv, index=False, encoding="utf-8-sig")

    selected_strategy_for_snapshot = str(chosen_meta.get("selected_strategy", "robust") or "robust").lower()
    selected_table_metrics = _extract_strategy_plan_snapshot(plan_df, selected_strategy_for_snapshot)
    if selected_table_metrics:
        chosen_meta["selected_snapshot_strategy"] = selected_strategy_for_snapshot

        def _float_mismatch(meta_key: str, snap_key: str) -> int:
            mv = float(chosen_meta.get(meta_key, np.nan))
            sv = float(selected_table_metrics.get(snap_key, np.nan))
            return int(np.isfinite(mv) and np.isfinite(sv) and abs(mv - sv) > 1.0e-6)

        def _int_mismatch(meta_key: str, snap_key: str) -> int:
            try:
                mv = _safe_int(chosen_meta.get(meta_key, -1))
                sv = _safe_int(selected_table_metrics.get(snap_key, -1))
                return int((mv >= 0) and (sv >= 0) and (mv != sv))
            except Exception:
                return 0

        chosen_meta["selected_snapshot_margin_mismatch"] = _float_mismatch(
            "selected_pred_min_safe_margin_px", "pred_min_safe_margin_px"
        )
        chosen_meta["selected_snapshot_violation_mismatch"] = _int_mismatch(
            "selected_pred_violation_count", "pred_violation_count"
        )
        chosen_meta["selected_snapshot_future_distance_mismatch"] = _float_mismatch(
            "selected_pred_min_future_min_distance_m", "pred_min_future_min_distance_m"
        )
        chosen_meta["selected_snapshot_future_distance_gain_mismatch"] = _float_mismatch(
            "selected_future_distance_gain_vs_zero_m", "selected_future_distance_gain_vs_zero_m"
        )
        chosen_meta["selected_snapshot_strict_mismatch"] = _int_mismatch(
            "strict_acceptance_passed", "strict_acceptance_passed"
        )
        _planner_selection_status_left = str(chosen_meta.get("planner_selection_status", "")).strip()
        _planner_selection_status_right = str(selected_table_metrics.get("planner_selection_status", "")).strip()
        chosen_meta["selected_snapshot_planner_selection_status_mismatch"] = int(
            bool(_planner_selection_status_left)
            and bool(_planner_selection_status_right)
            and (_planner_selection_status_left != _planner_selection_status_right)
        )
        for gate_key in [
            "acceptance_gate_safe",
            "acceptance_gate_structural",
            "acceptance_gate_information",
            "acceptance_gate_geometry_non_degenerate",
            "acceptance_gate_future_distance",
        ]:
            chosen_meta[f"selected_snapshot_{gate_key}_mismatch"] = _int_mismatch(
                f"selected_{gate_key}", gate_key
            )

    def _candidate_meta_prefix(prefix: str, seq: np.ndarray, info: dict) -> dict[str, float | int | str]:
        nonzero_steps, total_dv_mag, dominant_ratio = _sequence_structure_stats(np.asarray(seq, dtype=float))
        span_steps = _sequence_step_span(np.asarray(seq, dtype=float))
        out: dict[str, float | int | str] = {
            f"{prefix}_pred_min_safe_margin_px": float(info.get("min_safe_margin_px", np.nan)),
            f"{prefix}_pred_terminal_margin_px": float(info.get("terminal_margin_px", np.nan)),
            f"{prefix}_pred_violation_count": _safe_int(info.get("violation_count_pred", -1)),
            f"{prefix}_pred_negative_margin_sum_px": float(info.get("negative_margin_sum_px_pred", np.nan)),
            f"{prefix}_pred_min_relative_distance_m": float(info.get("min_relative_distance_m", np.nan)),
            f"{prefix}_pred_terminal_relative_distance_m": float(info.get("terminal_relative_distance_m", np.nan)),
            f"{prefix}_pred_min_future_min_distance_m": float(info.get("min_future_min_distance_m", np.nan)),
            f"{prefix}_pred_terminal_future_min_distance_m": float(info.get("terminal_future_min_distance_m", np.nan)),
            f"{prefix}_pred_worst_risk_rank": _safe_int(info.get("worst_risk_rank_pred", -999)),
            f"{prefix}_strict_acceptance_passed": _safe_int(info.get("strict_acceptance_passed", -1)),
            f"{prefix}_acceptance_gate_safe": _safe_int(info.get("acceptance_gate_safe", -1)),
            f"{prefix}_acceptance_gate_structural": _safe_int(info.get("acceptance_gate_structural", -1)),
            f"{prefix}_acceptance_gate_information": _safe_int(info.get("acceptance_gate_information", -1)),
            f"{prefix}_acceptance_gate_geometry_non_degenerate": _safe_int(info.get("acceptance_gate_geometry_non_degenerate", -1)),
            f"{prefix}_acceptance_gate_future_distance": _safe_int(info.get("acceptance_gate_future_distance", -1)),
            f"{prefix}_trace_ratio_vs_zero": float(info.get("acceptance_trace_ratio_vs_zero", np.nan)),
            f"{prefix}_terminal_trace_ratio_vs_zero": float(info.get("acceptance_terminal_trace_ratio_vs_zero", np.nan)),
            f"{prefix}_info_delta_vs_zero": float(info.get("acceptance_info_delta_vs_zero", np.nan)),
            f"{prefix}_margin_gain_vs_zero_px": float(info.get("acceptance_margin_gain_vs_zero_px", np.nan)),
            f"{prefix}_terminal_margin_gain_vs_zero_px": float(info.get("acceptance_terminal_margin_gain_vs_zero_px", np.nan)),
            f"{prefix}_violation_delta_vs_zero": _safe_int(info.get("acceptance_violation_delta_vs_zero", -1)),
            f"{prefix}_negative_margin_sum_delta_vs_zero_px": float(info.get("acceptance_negative_margin_sum_delta_vs_zero_px", np.nan)),
            f"{prefix}_future_distance_gain_vs_zero_m": float(info.get("acceptance_future_distance_gain_vs_zero_m", np.nan)),
            f"{prefix}_future_distance_non_regression": _safe_int(info.get("acceptance_future_distance_non_regression", -1)),
            f"{prefix}_info_metric": float(info.get("info_metric", np.nan)),
            f"{prefix}_trace_pos_sum": float(info.get("trace_pos_sum", np.nan)),
            f"{prefix}_terminal_trace_pos": float(info.get("terminal_trace_pos", np.nan)),
            f"{prefix}_entropy_reduction": float(info.get("entropy_reduction", np.nan)),
            f"{prefix}_fim_trace_sum": float(info.get("fim_trace_sum", np.nan)),
            f"{prefix}_visible_update_steps": _safe_int(info.get("visible_update_steps", -1)),
            f"{prefix}_internal_preview_min_future_distance_m": float(info.get("internal_preview_min_future_distance_m", np.nan)),
            f"{prefix}_internal_preview_worst_risk_rank": _safe_int(info.get("internal_preview_worst_risk_rank", -999)),
            f"{prefix}_nonzero_steps": int(nonzero_steps),
            f"{prefix}_total_dv_mag_mps": float(total_dv_mag),
            f"{prefix}_dominant_ratio": float(dominant_ratio),
            f"{prefix}_span_steps": int(span_steps),
            f"{prefix}_structurally_sane": int(_sequence_is_structurally_sane(np.asarray(seq, dtype=float))),
            f"{prefix}_plan_acceptance_reason": str(info.get("plan_acceptance_reason", "")),
            f"{prefix}_execution_selection_reason": str(info.get("execution_selection_reason", "")),
        }
        if str(prefix) != "selected":
            out[f"{prefix}_budget_worthiness_passed"] = _safe_int(info.get("budget_worthiness_passed", -1))
            out[f"{prefix}_margin_gain_per_dv"] = float(info.get("margin_gain_per_dv", np.nan))
            out[f"{prefix}_future_distance_gain_per_dv"] = float(info.get("future_distance_gain_per_dv", np.nan))
            out[f"{prefix}_trace_penalty_per_dv"] = float(info.get("trace_penalty_per_dv", np.nan))
        return out

    meta_csv = out_dir / "planner_meta.csv"
    planner_meta_row = {
        "trigger_t_sec": float(trig.get("t_sec", 0.0)),
        "trigger_frame": _safe_int(trig.get("frame", -1)),
        "trigger_reason": trigger_reason,
        "total_dv_budget_mps": float(total_dv_budget),
        **{k: v for k, v in chosen_meta.items() if not isinstance(v, (dict, list, np.ndarray))},
        **_candidate_meta_prefix("nominal", schedule_nom, info_nom),
        **_candidate_meta_prefix("robust", schedule_rob, info_rob),
        **_candidate_meta_prefix("selected", chosen_exec_schedule, chosen_meta),
    }
    pd.DataFrame([planner_meta_row]).to_csv(meta_csv, index=False, encoding="utf-8-sig")

    if strategy_executor is not None:
        try:
            strategy_executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        strategy_executor = None

    return PlanResult(
        plan_df=plan_df,
        trigger_row=trig,
        trigger_reason=trigger_reason,
        x_est=x_est,
        p_diag=p_diag,
        schedule_nom=schedule_nom,
        schedule_rob=schedule_rob,
        plan_meta=chosen_meta,
    )

def main() -> None:
    res = plan_active_sensing()
    print("saved:", OUT_DIR / "micro_dv_plan.csv")
    print("trigger_t_sec:", float(res.trigger_row.get("t_sec", 0.0)))
    print("trigger_reason:", res.trigger_reason)
    print("selected_strategy:", res.plan_meta.get("selected_strategy", "unknown"))
    print("plan_accepted:", res.plan_meta.get("plan_accepted", False))
    print("plan_acceptance_reason:", res.plan_meta.get("plan_acceptance_reason", "unknown"))
    print("used total dv budget [m/s]          =", float(res.plan_meta.get("total_dv_budget_mps", np.nan)))


if __name__ == "__main__":
    main()
