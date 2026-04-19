from __future__ import annotations

from typing import Any

import numpy as np

PHYSICAL_SAFE_STRONG_DISTANCE_M = 1500.0
PHYSICAL_SAFE_STRONG_PC_MAX = 1.0e-6
PHYSICAL_SAFE_MODERATE_DISTANCE_M = 900.0
PHYSICAL_SAFE_MODERATE_PC_MAX = 1.0e-8
PHYSICAL_SAFE_ULTRA_DISTANCE_M = 3000.0
PHYSICAL_SAFE_ULTRA_PC_MAX = 1.0e-8
PHYSICAL_SAFE_VETO_MAX_TRACE_P_POS = 1.2e4
PHYSICAL_SAFE_VETO_MAX_RMSE_POS_M = 500.0
PHYSICAL_SAFE_ULTRA_VETO_MAX_TRACE_P_POS = 2.5e5
PHYSICAL_SAFE_ULTRA_VETO_MAX_RMSE_POS_M = 5.0e3

ESTIMATOR_DIVERGENCE_MEAN_TRACE_P_POS_THR = 1.0e6
ESTIMATOR_DIVERGENCE_RMSE_POS_THR = 5.0e4
ESTIMATOR_DIVERGENCE_MEAN_SIGMA_T_THR = 1.0e4
ESTIMATOR_DIVERGENCE_MEAN_ABS_E_T_THR = 5.0e4
ESTIMATOR_DIVERGENCE_SUPER_TRACE_P_POS_THR = 1.0e8
ESTIMATOR_DIVERGENCE_SUPER_RMSE_POS_THR = 1.0e6

EXEC_GEOMETRY_MARGIN_DEGRADE_TOL_PX = 2.0
EXEC_GEOMETRY_NEG_MARGIN_SUM_DEGRADE_TOL_PX = 5.0
EXEC_MIN_RELATIVE_DISTANCE_DEGRADE_TOL_M = 5.0
EXEC_TERMINAL_RELATIVE_DISTANCE_DEGRADE_TOL_M = 5.0


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _append_execution_warning(plan_meta: dict[str, Any], warning: str) -> None:
    msg = str(warning).strip()
    if not msg:
        return
    raw = str(plan_meta.get("execution_warning_flags", "")).strip()
    items = [it for it in raw.split("|") if it]
    if msg not in items:
        items.append(msg)
    plan_meta["execution_warning_flags"] = "|".join(items)
    plan_meta["execution_warning_count"] = int(len(items))


def detect_estimator_divergence(orig_summary: dict[str, Any], orig_err: dict[str, Any] | None = None) -> tuple[bool, str]:
    mean_trace_pos = _as_float(orig_summary.get("mean_trace_P_pos", float("nan")))
    rmse_pos = _as_float((orig_err or {}).get("rmse_pos_m", float("nan")))
    mean_sigma_t = _as_float(orig_summary.get("mean_sigma_T_m", float("nan")))
    mean_abs_e_t = _as_float((orig_err or {}).get("mean_abs_e_T", float("nan")))

    if (
        (np.isfinite(mean_trace_pos) and mean_trace_pos >= ESTIMATOR_DIVERGENCE_SUPER_TRACE_P_POS_THR)
        or (np.isfinite(rmse_pos) and rmse_pos >= ESTIMATOR_DIVERGENCE_SUPER_RMSE_POS_THR)
    ):
        return True, "estimator_divergence_veto"

    moderate_hits = 0
    if np.isfinite(mean_trace_pos) and mean_trace_pos >= ESTIMATOR_DIVERGENCE_MEAN_TRACE_P_POS_THR:
        moderate_hits += 1
    if np.isfinite(rmse_pos) and rmse_pos >= ESTIMATOR_DIVERGENCE_RMSE_POS_THR:
        moderate_hits += 1
    if np.isfinite(mean_sigma_t) and mean_sigma_t >= ESTIMATOR_DIVERGENCE_MEAN_SIGMA_T_THR:
        moderate_hits += 1
    if np.isfinite(mean_abs_e_t) and mean_abs_e_t >= ESTIMATOR_DIVERGENCE_MEAN_ABS_E_T_THR:
        moderate_hits += 1

    if moderate_hits >= 2:
        return True, "estimator_divergence_veto"
    return False, ""


def decide_physical_safe_veto(
    orig_summary: dict[str, Any],
    orig_err: dict[str, Any] | None = None,
    *,
    dataset_subset_label: str = "",
) -> tuple[bool, str]:
    subset = str(dataset_subset_label).strip().lower()
    if subset in {"caution", "danger"}:
        return False, ""

    orig_last_risk = str(orig_summary.get("last_risk_level", "")).lower()
    orig_min_future = _as_float(orig_summary.get("min_future_min_distance_m", float("nan")))
    orig_max_pc = _as_float(orig_summary.get("max_pc_bplane", float("nan")), default=0.0)
    orig_trace_pos = _as_float(orig_summary.get("mean_trace_P_pos", float("nan")))
    orig_rmse_pos = _as_float((orig_err or {}).get("rmse_pos_m", float("nan")))

    strong_physical_safe = (
        orig_last_risk == "safe"
        and np.isfinite(orig_min_future)
        and orig_min_future >= PHYSICAL_SAFE_STRONG_DISTANCE_M
        and (not np.isfinite(orig_max_pc) or orig_max_pc <= PHYSICAL_SAFE_STRONG_PC_MAX)
    )
    moderate_physical_safe = (
        orig_last_risk == "safe"
        and np.isfinite(orig_min_future)
        and orig_min_future >= PHYSICAL_SAFE_MODERATE_DISTANCE_M
        and (not np.isfinite(orig_max_pc) or orig_max_pc <= PHYSICAL_SAFE_MODERATE_PC_MAX)
    )
    ultra_physical_safe = (
        orig_last_risk == "safe"
        and np.isfinite(orig_min_future)
        and orig_min_future >= PHYSICAL_SAFE_ULTRA_DISTANCE_M
        and (not np.isfinite(orig_max_pc) or orig_max_pc <= PHYSICAL_SAFE_ULTRA_PC_MAX)
    )

    estimator_health_ok = (
        (not np.isfinite(orig_trace_pos) or orig_trace_pos <= PHYSICAL_SAFE_VETO_MAX_TRACE_P_POS)
        and (not np.isfinite(orig_rmse_pos) or orig_rmse_pos <= PHYSICAL_SAFE_VETO_MAX_RMSE_POS_M)
    )
    ultra_safe_health_ok = (
        (not np.isfinite(orig_trace_pos) or orig_trace_pos <= PHYSICAL_SAFE_ULTRA_VETO_MAX_TRACE_P_POS)
        and (not np.isfinite(orig_rmse_pos) or orig_rmse_pos <= PHYSICAL_SAFE_ULTRA_VETO_MAX_RMSE_POS_M)
    )

    if ultra_physical_safe and ultra_safe_health_ok:
        return True, "physical_safe_veto"
    if estimator_health_ok and (strong_physical_safe or moderate_physical_safe):
        return True, "physical_safe_veto"
    return False, ""


def resolve_execution_strategy(plan_meta: dict[str, Any]) -> tuple[str, str]:
    selected_strategy = str(
        plan_meta.get("selected_strategy", plan_meta.get("final_selected_strategy", "zero"))
        or "zero"
    ).lower()
    if selected_strategy not in {"robust", "nominal", "zero"}:
        selected_strategy = "zero"
    return selected_strategy, "planner_selected_strategy"


def decide_execution_veto(
    orig_summary: dict[str, Any],
    orig_err: dict[str, Any],
    plan_meta: dict[str, Any],
    *,
    dataset_subset_label: str,
    plan_ok: bool,
    pred_margin: float,
    pred_viol: int,
) -> tuple[bool, str]:
    estimator_diverged, divergence_reason = detect_estimator_divergence(orig_summary, orig_err)
    if estimator_diverged:
        return True, divergence_reason

    physical_safe_vetoed, physical_safe_reason = decide_physical_safe_veto(
        orig_summary,
        orig_err,
        dataset_subset_label=dataset_subset_label,
    )
    if physical_safe_vetoed:
        return True, physical_safe_reason

    zero_pred_margin = _as_float(plan_meta.get("zero_pred_min_safe_margin_px", float("nan")))
    zero_pred_viol = int(plan_meta.get("zero_pred_violation_count", -1))
    selected_neg_margin_sum = _as_float(
        plan_meta.get("selected_pred_negative_margin_sum_px", plan_meta.get("negative_margin_sum_px_pred", float("nan")))
    )
    zero_neg_margin_sum = _as_float(plan_meta.get("zero_pred_negative_margin_sum_px", float("nan")))
    selected_min_rel_dist = _as_float(plan_meta.get("selected_pred_min_relative_distance_m", float("nan")))
    zero_min_rel_dist = _as_float(plan_meta.get("zero_pred_min_relative_distance_m", float("nan")))
    selected_terminal_rel_dist = _as_float(plan_meta.get("selected_pred_terminal_relative_distance_m", float("nan")))
    zero_terminal_rel_dist = _as_float(plan_meta.get("zero_pred_terminal_relative_distance_m", float("nan")))
    geometry_preview_status = str(plan_meta.get("geometry_preview_status", "")).lower()

    if (
        plan_ok
        and np.isfinite(pred_margin)
        and np.isfinite(zero_pred_margin)
        and (pred_margin < zero_pred_margin - EXEC_GEOMETRY_MARGIN_DEGRADE_TOL_PX)
    ):
        return True, "predicted_geometry_regression_veto"

    if (
        plan_ok
        and np.isfinite(selected_min_rel_dist)
        and np.isfinite(zero_min_rel_dist)
        and (selected_min_rel_dist < zero_min_rel_dist - EXEC_MIN_RELATIVE_DISTANCE_DEGRADE_TOL_M)
    ):
        return True, "predicted_relative_distance_regression_veto"

    if (
        plan_ok
        and np.isfinite(selected_terminal_rel_dist)
        and np.isfinite(zero_terminal_rel_dist)
        and (selected_terminal_rel_dist < zero_terminal_rel_dist - EXEC_TERMINAL_RELATIVE_DISTANCE_DEGRADE_TOL_M)
    ):
        return True, "predicted_terminal_relative_distance_regression_veto"

    if plan_ok and (zero_pred_viol >= 0) and (pred_viol > zero_pred_viol):
        return True, "predicted_violation_regression_veto"

    if (
        plan_ok
        and np.isfinite(selected_neg_margin_sum)
        and np.isfinite(zero_neg_margin_sum)
        and (selected_neg_margin_sum > zero_neg_margin_sum + EXEC_GEOMETRY_NEG_MARGIN_SUM_DEGRADE_TOL_PX)
    ):
        return True, "predicted_negative_margin_regression_veto"

    if geometry_preview_status == "failed":
        return True, "execution_geometry_preview_failed"

    return False, ""
