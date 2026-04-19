from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_str(value: Any, default: str = "") -> str:
    try:
        s = str(value)
        return s
    except Exception:
        return default


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _clean_optional_status(value: Any) -> str:
    s = _to_str(value, "").strip()
    if s.lower() in {"", "nan", "none"}:
        return ""
    return s


ACCEPTANCE_STATES = {"accepted", "borderline_review", "rejected"}


def _normalize_acceptance_reason(
    plan_acceptance_reason: Any,
    *,
    plan_accepted: int,
    planner_acceptance_status: str,
) -> tuple[str, str]:
    raw_reason = _clean_optional_status(plan_acceptance_reason)
    raw_reason_lower = raw_reason.lower()
    status_lower = _clean_optional_status(planner_acceptance_status).lower()

    if status_lower in ACCEPTANCE_STATES:
        normalized = status_lower
    elif raw_reason_lower in ACCEPTANCE_STATES:
        normalized = raw_reason_lower
    elif plan_accepted == 1:
        normalized = "accepted"
    elif "borderline" in raw_reason_lower:
        normalized = "borderline_review"
    else:
        normalized = "rejected"

    diagnostics_reason = ""
    if raw_reason and raw_reason_lower not in ACCEPTANCE_STATES:
        diagnostics_reason = raw_reason
    return normalized, diagnostics_reason


def _apply_five_layer_contract(out: dict[str, Any]) -> None:
    # 1) baseline layer (zero strategy)
    out["zero_pred_min_safe_margin_px"] = _to_float(out.get("zero_pred_min_safe_margin_px", float("nan")))
    out["zero_pred_terminal_margin_px"] = _to_float(out.get("zero_pred_terminal_margin_px", float("nan")))
    out["zero_pred_violation_count"] = _to_int(out.get("zero_pred_violation_count", -1), default=-1)
    out["zero_pred_negative_margin_sum_px"] = _to_float(out.get("zero_pred_negative_margin_sum_px", float("nan")))
    out["zero_pred_min_relative_distance_m"] = _to_float(out.get("zero_pred_min_relative_distance_m", float("nan")))
    out["zero_pred_terminal_relative_distance_m"] = _to_float(out.get("zero_pred_terminal_relative_distance_m", float("nan")))
    out["zero_pred_min_future_min_distance_m"] = _to_float(out.get("zero_pred_min_future_min_distance_m", float("nan")))
    out["baseline_strategy"] = "zero"
    out["baseline_pred_min_safe_margin_px"] = float(out["zero_pred_min_safe_margin_px"])
    out["baseline_pred_terminal_margin_px"] = float(out["zero_pred_terminal_margin_px"])
    out["baseline_pred_violation_count"] = int(out["zero_pred_violation_count"])
    out["baseline_pred_negative_margin_sum_px"] = float(out["zero_pred_negative_margin_sum_px"])
    out["baseline_pred_min_relative_distance_m"] = float(out["zero_pred_min_relative_distance_m"])
    out["baseline_pred_terminal_relative_distance_m"] = float(out["zero_pred_terminal_relative_distance_m"])
    out["baseline_pred_min_future_min_distance_m"] = float(out["zero_pred_min_future_min_distance_m"])

    # 2) selected layer
    selected_strategy = _to_str(out.get("selected_strategy", "")).strip().lower()
    if selected_strategy not in {"robust", "nominal", "zero"}:
        selected_strategy = _to_str(out.get("final_selected_strategy", "")).strip().lower()
    out["selected_strategy"] = selected_strategy if selected_strategy in {"robust", "nominal", "zero"} else ""
    out["selected_pred_min_safe_margin_px"] = _to_float(out.get("selected_pred_min_safe_margin_px", float("nan")))
    out["selected_pred_terminal_margin_px"] = _to_float(out.get("selected_pred_terminal_margin_px", float("nan")))
    out["selected_pred_violation_count"] = _to_int(out.get("selected_pred_violation_count", -1), default=-1)
    out["selected_pred_negative_margin_sum_px"] = _to_float(out.get("selected_pred_negative_margin_sum_px", float("nan")))
    out["selected_pred_min_relative_distance_m"] = _to_float(out.get("selected_pred_min_relative_distance_m", float("nan")))
    out["selected_pred_terminal_relative_distance_m"] = _to_float(out.get("selected_pred_terminal_relative_distance_m", float("nan")))
    out["selected_nonzero_steps"] = _to_int(out.get("selected_nonzero_steps", -1), default=-1)
    out["selected_total_dv_mag_mps"] = _to_float(out.get("selected_total_dv_mag_mps", float("nan")))
    out["selected_span_steps"] = _to_int(out.get("selected_span_steps", -1), default=-1)
    out["selected_structurally_sane"] = _to_int(out.get("selected_structurally_sane", -1), default=-1)
    out["selected_dominant_ratio"] = _to_float(out.get("selected_dominant_ratio", float("nan")))

    # 3) requested layer (execution-request snapshot)
    requested_strategy = _to_str(
        out.get("execution_strategy_requested", out.get("execution_request_strategy", out.get("selected_strategy", "")))
    ).strip().lower()
    if requested_strategy not in {"robust", "nominal", "zero"}:
        requested_strategy = ""
    out["execution_strategy_requested"] = requested_strategy
    out["execution_requested_pred_min_safe_margin_px"] = _to_float(
        out.get("execution_requested_pred_min_safe_margin_px", float("nan"))
    )
    out["execution_requested_pred_terminal_margin_px"] = _to_float(
        out.get("execution_requested_pred_terminal_margin_px", float("nan"))
    )
    out["execution_requested_pred_violation_count"] = _to_int(
        out.get("execution_requested_pred_violation_count", -1), default=-1
    )
    out["execution_requested_pred_negative_margin_sum_px"] = _to_float(
        out.get("execution_requested_pred_negative_margin_sum_px", float("nan"))
    )
    out["execution_requested_pred_min_relative_distance_m"] = _to_float(
        out.get("execution_requested_pred_min_relative_distance_m", float("nan"))
    )
    out["execution_requested_pred_terminal_relative_distance_m"] = _to_float(
        out.get("execution_requested_pred_terminal_relative_distance_m", float("nan"))
    )
    out["execution_requested_nonzero_steps"] = _to_int(out.get("execution_requested_nonzero_steps", -1), default=-1)
    out["execution_requested_total_dv_mag_mps"] = _to_float(
        out.get("execution_requested_total_dv_mag_mps", float("nan"))
    )
    out["execution_requested_span_steps"] = _to_int(out.get("execution_requested_span_steps", -1), default=-1)
    out["execution_requested_structurally_sane"] = _to_int(
        out.get("execution_requested_structurally_sane", -1), default=-1
    )
    out["execution_requested_dominant_ratio"] = _to_float(
        out.get("execution_requested_dominant_ratio", float("nan"))
    )
    out["requested_strategy"] = requested_strategy
    out["requested_pred_min_safe_margin_px"] = float(out["execution_requested_pred_min_safe_margin_px"])
    out["requested_pred_terminal_margin_px"] = float(out["execution_requested_pred_terminal_margin_px"])
    out["requested_pred_violation_count"] = int(out["execution_requested_pred_violation_count"])
    out["requested_pred_negative_margin_sum_px"] = float(out["execution_requested_pred_negative_margin_sum_px"])
    out["requested_pred_min_relative_distance_m"] = float(out["execution_requested_pred_min_relative_distance_m"])
    out["requested_pred_terminal_relative_distance_m"] = float(out["execution_requested_pred_terminal_relative_distance_m"])
    out["requested_nonzero_steps"] = int(out["execution_requested_nonzero_steps"])
    out["requested_total_dv_mag_mps"] = float(out["execution_requested_total_dv_mag_mps"])
    out["requested_span_steps"] = int(out["execution_requested_span_steps"])
    out["requested_structurally_sane"] = int(out["execution_requested_structurally_sane"])
    out["requested_dominant_ratio"] = float(out["execution_requested_dominant_ratio"])
    out["requested_source_reason"] = _to_str(
        out.get("execution_request_source_reason", out.get("execution_strategy_reason", ""))
    ).strip()

    # 4) executed layer
    executed_strategy = _to_str(
        out.get("executed_strategy", out.get("execution_strategy_applied", out.get("requested_strategy", "")))
    ).strip().lower()
    if executed_strategy not in {"robust", "nominal", "zero"}:
        executed_strategy = ""
    out["executed_strategy"] = executed_strategy
    out["execution_strategy_applied"] = executed_strategy
    out["executed_pred_min_safe_margin_px"] = _to_float(out.get("executed_pred_min_safe_margin_px", float("nan")))
    out["executed_pred_terminal_margin_px"] = _to_float(out.get("executed_pred_terminal_margin_px", float("nan")))
    out["executed_pred_violation_count"] = _to_int(out.get("executed_pred_violation_count", -1), default=-1)
    out["executed_pred_negative_margin_sum_px"] = _to_float(out.get("executed_pred_negative_margin_sum_px", float("nan")))
    out["executed_pred_min_relative_distance_m"] = _to_float(out.get("executed_pred_min_relative_distance_m", float("nan")))
    out["executed_pred_terminal_relative_distance_m"] = _to_float(out.get("executed_pred_terminal_relative_distance_m", float("nan")))
    out["executed_nonzero_steps"] = _to_int(out.get("executed_nonzero_steps", -1), default=-1)
    out["executed_total_dv_mag_mps"] = _to_float(out.get("executed_total_dv_mag_mps", float("nan")))
    out["executed_span_steps"] = _to_int(out.get("executed_span_steps", -1), default=-1)
    out["executed_structurally_sane"] = _to_int(out.get("executed_structurally_sane", -1), default=-1)
    out["executed_dominant_ratio"] = _to_float(out.get("executed_dominant_ratio", float("nan")))

    # 5) derived layer marker
    out.setdefault("derived_planner_candidate_status", "")
    out.setdefault("derived_planner_acceptance_status", "")
    out.setdefault("derived_planner_selection_status", "")
    out.setdefault("derived_execution_final_status", "")
    out.setdefault("derived_failure_family", "")
    out.setdefault("derived_planner_preview_consistency_flag", -1)

    out["decision_row_schema_version"] = "v2_five_layer"
    out["decision_row_layer_contract"] = "baseline|selected|execution_requested|executed|derived"


def enrich_execution_decision_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)

    plan_accepted = int(bool(out.get("plan_accepted", 0)))
    selected_strategy = _to_str(out.get("selected_strategy", "")).strip().lower()
    strict_passed = _to_int(out.get("strict_acceptance_passed", out.get("strictly_feasible_plan", -1)), default=-1)
    raw_candidate_status = _clean_optional_status(out.get("planner_candidate_status", ""))
    raw_acceptance_status = _clean_optional_status(out.get("planner_acceptance_status", ""))
    raw_selection_status = _clean_optional_status(out.get("planner_selection_status", ""))
    raw_planner_execution_stage_status = _clean_optional_status(
        out.get("planner_execution_stage_status", out.get("pre_execution_status", ""))
    )
    raw_execution_final_status = _clean_optional_status(out.get("execution_final_status", ""))
    if (not raw_planner_execution_stage_status) and raw_execution_final_status.startswith("planner_"):
        raw_planner_execution_stage_status = raw_execution_final_status

    out["raw_planner_candidate_status"] = raw_candidate_status
    out["raw_planner_acceptance_status"] = raw_acceptance_status
    out["raw_planner_selection_status"] = raw_selection_status
    out["raw_planner_execution_stage_status"] = raw_planner_execution_stage_status
    out["raw_execution_final_status"] = raw_execution_final_status
    out["planner_execution_stage_status"] = raw_planner_execution_stage_status
    out["pre_execution_status"] = raw_planner_execution_stage_status

    normalized_acceptance_reason, acceptance_diagnostics_reason = _normalize_acceptance_reason(
        out.get("plan_acceptance_reason", ""),
        plan_accepted=plan_accepted,
        planner_acceptance_status=raw_acceptance_status,
    )
    out["plan_acceptance_reason"] = normalized_acceptance_reason
    if acceptance_diagnostics_reason and _clean_optional_status(out.get("acceptance_diagnostics_reason", "")) == "":
        out["acceptance_diagnostics_reason"] = acceptance_diagnostics_reason
    accept_reason = str(out.get("plan_acceptance_reason", ""))

    if raw_candidate_status:
        out["derived_planner_candidate_status"] = raw_candidate_status
    elif selected_strategy in {"robust", "nominal", "zero"}:
        out["derived_planner_candidate_status"] = f"candidate_{selected_strategy}"
    else:
        out["derived_planner_candidate_status"] = "candidate_unknown"

    if raw_selection_status:
        out["derived_planner_selection_status"] = raw_selection_status
    elif selected_strategy in {"robust", "nominal", "zero"}:
        out["derived_planner_selection_status"] = f"selected_{selected_strategy}"
    else:
        out["derived_planner_selection_status"] = "selected_unknown"

    if raw_acceptance_status in ACCEPTANCE_STATES:
        out["derived_planner_acceptance_status"] = raw_acceptance_status
    elif accept_reason in ACCEPTANCE_STATES:
        out["derived_planner_acceptance_status"] = accept_reason
    elif plan_accepted == 1 and strict_passed == 1:
        out["derived_planner_acceptance_status"] = "accepted"
    elif "borderline" in accept_reason:
        out["derived_planner_acceptance_status"] = "borderline_review"
    else:
        out["derived_planner_acceptance_status"] = "rejected"
    if not raw_acceptance_status:
        out["planner_acceptance_status"] = str(out.get("derived_planner_acceptance_status", ""))

    execution_applied = int(bool(out.get("execution_applied_flag", 0)))
    execution_vetoed = int(bool(out.get("execution_vetoed", 0)))
    execution_veto_reason = _to_str(out.get("execution_veto_reason", "")).strip()
    execution_decision_reason = _to_str(out.get("execution_decision_reason", "")).strip()
    action = _to_str(out.get("action", "")).strip().lower()

    # Derive final execution state from post-planner facts first.
    if execution_applied == 1:
        out["derived_execution_final_status"] = "execution_applied"
    elif execution_vetoed == 1 and execution_veto_reason:
        out["derived_execution_final_status"] = f"execution_veto_{execution_veto_reason}"
    elif execution_vetoed == 1:
        out["derived_execution_final_status"] = "execution_vetoed"
    elif action == "track_only" and plan_accepted == 0:
        out["derived_execution_final_status"] = "planner_rejected_not_executed"
    elif action == "track_only":
        out["derived_execution_final_status"] = "execution_skipped_track_only"
    elif raw_planner_execution_stage_status:
        out["derived_execution_final_status"] = raw_planner_execution_stage_status
    else:
        out["derived_execution_final_status"] = "execution_pending"

    if execution_applied == 1 and plan_accepted == 0:
        derived_failure_family = "planner_rejected_but_forced_executed"
        derived_preview_flag = -1
    elif execution_applied == 1:
        derived_failure_family = ""
        derived_preview_flag = 1
    elif plan_accepted == 1 and execution_vetoed == 1 and execution_veto_reason.startswith("preview_"):
        derived_failure_family = "planner_accepted_but_preview_vetoed"
        derived_preview_flag = 0
    elif plan_accepted == 1 and execution_vetoed == 1:
        derived_failure_family = "planner_accepted_but_execution_vetoed"
        derived_preview_flag = 1
    elif plan_accepted == 0 and execution_applied == 0:
        derived_failure_family = "planner_rejected_not_executed"
        derived_preview_flag = -1
    else:
        derived_failure_family = ""
        derived_preview_flag = -1

    out["derived_failure_family"] = derived_failure_family
    out["derived_planner_preview_consistency_flag"] = int(derived_preview_flag)

    if _clean_optional_status(out.get("failure_family", "")) == "":
        out["failure_family"] = derived_failure_family
    if _to_int(out.get("planner_preview_consistency_flag", -1), default=-1) == -1:
        out["planner_preview_consistency_flag"] = int(derived_preview_flag)

    warning_flags = _clean_optional_status(out.get("execution_warning_flags", ""))
    out["execution_warning_flags"] = warning_flags
    if warning_flags:
        out["execution_warning_count"] = int(len([it for it in warning_flags.split("|") if it]))
    else:
        out["execution_warning_count"] = int(_to_int(out.get("execution_warning_count", 0), default=0))

    out.setdefault("failure_family", "")
    out.setdefault("planner_preview_consistency_flag", -1)

    _apply_five_layer_contract(out)

    return out


def build_execution_decision_row(template_row: dict[str, Any], updates: dict[str, Any] | None = None) -> dict[str, Any]:
    row = dict(template_row)
    if updates:
        row.update(dict(updates))
    return enrich_execution_decision_row(row)


def classify_rejected_plan_veto(plan_ok: bool, pred_margin: float, pred_viol: int, row: dict[str, Any]) -> str:
    accept_reason = _clean_optional_status(row.get("plan_acceptance_reason", "")).lower()
    accept_diagnostics_reason = _clean_optional_status(
        row.get("acceptance_diagnostics_reason", row.get("execution_selection_reason", ""))
    ).lower()
    effective_reason = accept_diagnostics_reason or accept_reason

    gate_safe = _to_int(row.get("selected_acceptance_gate_safe", row.get("acceptance_gate_safe", -1)), default=-1)
    gate_struct = _to_int(row.get("selected_acceptance_gate_structural", row.get("acceptance_gate_structural", -1)), default=-1)
    gate_info = _to_int(row.get("selected_acceptance_gate_information", row.get("acceptance_gate_information", -1)), default=-1)
    gate_geom = _to_int(
        row.get(
            "selected_acceptance_gate_geometry_non_degenerate",
            row.get("acceptance_gate_geometry_non_degenerate", -1),
        ),
        default=-1,
    )
    gate_future = _to_int(
        row.get("selected_acceptance_gate_future_distance", row.get("acceptance_gate_future_distance", -1)),
        default=-1,
    )

    margin_val = _to_float(pred_margin, float("nan"))
    pred_viol_i = _to_int(pred_viol, default=-1)

    # Minimal families: policy details stay in diagnostics fields.
    if (
        effective_reason in {"late_hard_case", "unobservable_hard_case"}
        or effective_reason.startswith("budget_limited_")
        or ("best_effort" in effective_reason)
    ):
        return "planner_policy_veto"

    # Then geometry/safety gates.
    if gate_safe == 0 or pred_viol_i > 0 or (np.isfinite(margin_val) and margin_val <= 0.0):
        return "planner_safety_veto"

    if gate_geom == 0:
        return "planner_safety_veto"

    if gate_future == 0:
        return "planner_safety_veto"

    # Structural / information gates for non-accepted planner outcomes.
    if not bool(plan_ok):
        if gate_struct == 0:
            return "planner_acceptance_veto"
        if gate_info == 0:
            return "planner_acceptance_veto"
        return "planner_acceptance_veto"

    return "accepted_plan_execute"


def run_execution_preview_and_gate(
    *,
    truth_csv: Path,
    plan_csv: Path,
    preview_truth_csv: Path,
    strategy: str,
    plan_meta_for_execution: dict[str, Any],
    decision_row: dict[str, Any],
    preview_func: Callable[..., tuple[dict[str, Any], str]],
    veto_func: Callable[..., tuple[bool, str]],
    orig_summary: dict[str, Any],
    orig_err: dict[str, Any],
    dataset_subset_label: str,
    plan_ok: bool,
    pred_margin: float,
    pred_viol: int,
) -> tuple[dict[str, Any], dict[str, Any], bool, str]:
    plan_meta = dict(plan_meta_for_execution)
    row = dict(decision_row)

    # Only accepted plans should go through preview/veto chain.
    planner_unsafe = (not bool(plan_ok)) or (int(pred_viol) > 0) or (float(pred_margin) <= 0.0)
    if planner_unsafe:
        plan_meta["geometry_preview_status"] = "skipped_planner_rejected"
        plan_meta["geometry_preview_error"] = ""
        row.update({
            "geometry_preview_truth_csv": str(preview_truth_csv),
            "geometry_preview_status": "skipped_planner_rejected",
            "geometry_preview_error": "",
            "execution_warning_flags": str(plan_meta.get("execution_warning_flags", "")),
            "execution_warning_count": int(plan_meta.get("execution_warning_count", 0)),
        })
        row = enrich_execution_decision_row(row)
        return plan_meta, row, False, ""

    preview_summary, geometry_preview_error = preview_func(
        truth_csv,
        plan_csv,
        preview_truth_csv,
        strategy=strategy,
    )
    if geometry_preview_error:
        plan_meta["geometry_preview_status"] = "failed"
        plan_meta["geometry_preview_error"] = geometry_preview_error
    else:
        plan_meta["geometry_preview_status"] = "ok"
        plan_meta["preview_pred_min_relative_distance_m"] = float(preview_summary.get("min_relative_distance_m", np.nan))
        plan_meta["preview_pred_min_future_min_distance_m"] = float(preview_summary.get("min_future_min_distance_m", np.nan))
        plan_meta["preview_pred_last_risk_level"] = str(preview_summary.get("last_risk_level", ""))
        plan_meta["preview_pred_worst_risk_rank"] = int(preview_summary.get("worst_risk_rank", -999))
        plan_meta["preview_pred_num_safe_steps"] = int(preview_summary.get("num_safe_steps", -1))
        plan_meta["preview_pred_num_caution_steps"] = int(preview_summary.get("num_caution_steps", -1))
        plan_meta["preview_pred_num_danger_steps"] = int(preview_summary.get("num_danger_steps", -1))

    row.update({
        "geometry_preview_truth_csv": str(preview_truth_csv),
        "geometry_preview_status": str(plan_meta.get("geometry_preview_status", "")),
        "geometry_preview_error": str(plan_meta.get("geometry_preview_error", geometry_preview_error)),
        "execution_warning_flags": str(plan_meta.get("execution_warning_flags", "")),
        "execution_warning_count": int(plan_meta.get("execution_warning_count", 0)),
        "preview_pred_min_relative_distance_m": float(plan_meta.get("preview_pred_min_relative_distance_m", np.nan)),
        "preview_pred_min_future_min_distance_m": float(plan_meta.get("preview_pred_min_future_min_distance_m", np.nan)),
        "preview_pred_last_risk_level": str(plan_meta.get("preview_pred_last_risk_level", "")),
        "preview_pred_worst_risk_rank": int(plan_meta.get("preview_pred_worst_risk_rank", -999)),
        "preview_pred_num_safe_steps": int(plan_meta.get("preview_pred_num_safe_steps", -1)),
        "preview_pred_num_caution_steps": int(plan_meta.get("preview_pred_num_caution_steps", -1)),
        "preview_pred_num_danger_steps": int(plan_meta.get("preview_pred_num_danger_steps", -1)),
    })

    execution_vetoed, execution_veto_reason = veto_func(
        orig_summary,
        orig_err,
        plan_meta,
        dataset_subset_label=dataset_subset_label,
        plan_ok=plan_ok,
        pred_margin=pred_margin,
        pred_viol=pred_viol,
    )

    row["execution_warning_flags"] = str(plan_meta.get("execution_warning_flags", ""))
    row["execution_warning_count"] = int(plan_meta.get("execution_warning_count", 0))

    row = enrich_execution_decision_row(row)
    return plan_meta, row, execution_vetoed, execution_veto_reason
