from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


ACCEPTANCE_GATE_KEYS = [
    "acceptance_gate_safe",
    "acceptance_gate_structural",
    "acceptance_gate_information",
    "acceptance_gate_geometry_non_degenerate",
    "acceptance_gate_future_distance",
]

TRIGGER_DIAGNOSTIC_KEYS = [
    "trigger_safe_dist_px",
    "trigger_safe_dist_rate_pxps",
    "trigger_short_time_to_warning_sec",
    "trigger_short_time_to_breach_sec",
    "trigger_risk_score",
    "trigger_warning_score",
    "trigger_risk_integral",
    "trigger_warning_integral",
    "trigger_short_pred_min_safe_margin_px",
    "trigger_pred_min_safe_margin_px",
    "trigger_est_range_m",
]


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except Exception:
        return default


def _as_int(value: Any, default: int = -1) -> int:
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


def _first_finite_float(*values: Any, default: float = float("nan")) -> float:
    for value in values:
        out = _as_float(value, float("nan"))
        if np.isfinite(out):
            return float(out)
    return float(default)


ALLOWED_STRATEGIES = {"robust", "nominal", "zero"}


def _normalized_strategy(value: Any, default: str = "robust") -> str:
    strategy = str(value or "").strip().lower()
    return strategy if strategy in ALLOWED_STRATEGIES else default


def _prefixed_value(row: pd.Series, strategy: str, stem: str, default: Any = None) -> Any:
    strategy = _normalized_strategy(strategy, default="robust")
    if strategy == "zero":
        zero_keys = [
            f"zero_{stem}",
            f"zero_pred_{stem}",
        ]
        for key in zero_keys:
            if key in row:
                return row.get(key, default)
        return default

    keys = [
        f"{strategy}_{stem}",
        f"{strategy}_pred_{stem}",
    ]
    for key in keys:
        if key in row:
            return row.get(key, default)
    return default


def extract_strategy_plan_metrics(plan_df: pd.DataFrame, strategy: str) -> dict[str, Any]:
    strategy = _normalized_strategy(strategy, default="robust")
    if len(plan_df) == 0:
        out: dict[str, Any] = {
            "strategy": str(strategy),
            "pred_min_safe_margin_px": float("nan"),
            "pred_terminal_margin_px": float("nan"),
            "pred_violation_count": -1,
            "pred_negative_margin_sum_px": float("nan"),
            "pred_min_relative_distance_m": float("nan"),
            "pred_terminal_relative_distance_m": float("nan"),
            "pred_min_future_min_distance_m": float("nan"),
            "pred_terminal_future_min_distance_m": float("nan"),
            "pred_worst_risk_rank": -999,
            "strict_acceptance_passed": -1,
            "acceptance_trace_ratio_vs_zero": float("nan"),
            "acceptance_terminal_trace_ratio_vs_zero": float("nan"),
            "acceptance_info_delta_vs_zero": float("nan"),
            "planner_candidate_status": "",
            "planner_acceptance_status": "",
            "planner_selection_status": "",
            "planner_initial_selection_status": "",
            "final_selected_strategy": "",
            "selection_correction_applied": 0,
            "planner_execution_stage_status": "",
            "pre_execution_status": "",
            "nonzero_steps": -1,
            "span_steps": -1,
            "dominant_ratio": float("nan"),
            "structurally_sane": -1,
            "total_dv_mag_mps": float("nan"),
            "selected_future_distance_gain_vs_zero_m": float("nan"),
        }
        for gate in ACCEPTANCE_GATE_KEYS:
            out[gate] = -1
        return out

    full_first = plan_df.iloc[0]
    use = plan_df.copy()
    if strategy != "zero" and "strategy" in use.columns:
        rows = use[use["strategy"].astype(str).str.lower() == strategy].copy()
        if len(rows) == 0:
            out: dict[str, Any] = {
                "strategy": str(strategy),
                "pred_min_safe_margin_px": float("nan"),
                "pred_terminal_margin_px": float("nan"),
                "pred_violation_count": -1,
                "pred_negative_margin_sum_px": float("nan"),
                "pred_min_relative_distance_m": float("nan"),
                "pred_terminal_relative_distance_m": float("nan"),
                "pred_min_future_min_distance_m": float("nan"),
                "pred_terminal_future_min_distance_m": float("nan"),
                "pred_worst_risk_rank": -999,
                "strict_acceptance_passed": -1,
                "acceptance_trace_ratio_vs_zero": float("nan"),
                "acceptance_terminal_trace_ratio_vs_zero": float("nan"),
                "acceptance_info_delta_vs_zero": float("nan"),
                "planner_candidate_status": "",
                "planner_acceptance_status": "",
                "planner_selection_status": "",
                "planner_initial_selection_status": "",
                "final_selected_strategy": "",
                "selection_correction_applied": 0,
                "planner_execution_stage_status": "",
                "pre_execution_status": "",
                "nonzero_steps": -1,
                "span_steps": -1,
                "dominant_ratio": float("nan"),
                "structurally_sane": -1,
                "total_dv_mag_mps": float("nan"),
                "selected_future_distance_gain_vs_zero_m": float("nan"),
            }
            for gate in ACCEPTANCE_GATE_KEYS:
                out[gate] = -1
            return out
        use = rows

    first = use.iloc[0] if len(use) > 0 else pd.Series(dtype=object)

    if strategy == "zero":
        pred_min = _as_float(_prefixed_value(full_first, strategy, "min_safe_margin_px", float("nan")))
        pred_terminal = _as_float(_prefixed_value(full_first, strategy, "terminal_margin_px", float("nan")))
        pred_viol = _as_int(_prefixed_value(full_first, strategy, "violation_count", -1), -1)
        pred_neg_margin_sum = _as_float(_prefixed_value(full_first, strategy, "negative_margin_sum_px", float("nan")))
        pred_min_rel_dist = _as_float(_prefixed_value(full_first, strategy, "min_relative_distance_m", float("nan")))
        pred_terminal_rel_dist = _as_float(_prefixed_value(full_first, strategy, "terminal_relative_distance_m", float("nan")))
        pred_min_future = _as_float(
            _prefixed_value(full_first, strategy, "min_future_min_distance_m", _prefixed_value(full_first, strategy, "min_future_distance_m", float("nan")))
        )
        pred_terminal_future = _as_float(
            _prefixed_value(full_first, strategy, "terminal_future_min_distance_m", _prefixed_value(full_first, strategy, "terminal_future_distance_m", float("nan")))
        )
        pred_worst_rank = _as_int(_prefixed_value(full_first, strategy, "worst_risk_rank", _prefixed_value(full_first, strategy, "worst_risk_rank_pred", -999)), -999)
        strict_acceptance = -1
        acceptance_trace_ratio_vs_zero = float("nan")
        acceptance_terminal_trace_ratio_vs_zero = float("nan")
        acceptance_info_delta_vs_zero = float("nan")
        planner_candidate_status = "candidate_zero"
        planner_acceptance_status = "zero_baseline"
        planner_selection_status = str(full_first.get("planner_selection_status", ""))
        planner_initial_selection_status = str(full_first.get("planner_initial_selection_status", ""))
        final_selected_strategy = str(full_first.get("final_selected_strategy", ""))
        try:
            selection_correction_applied = _as_int(full_first.get("selection_correction_applied", 0), 0)
        except Exception:
            selection_correction_applied = 0
        planner_execution_stage_status = str(full_first.get("planner_execution_stage_status", full_first.get("pre_execution_status", "")))
        pre_execution_status = str(full_first.get("pre_execution_status", full_first.get("planner_execution_stage_status", "")))
        selected_future_distance_gain_vs_zero_m = 0.0
        total_dv = 0.0
        nonzero_steps = 0
        dominant_ratio = 0.0
        span_steps = 0
        structurally_sane = 1
        gate_defaults = {
            "acceptance_gate_safe": int(pred_viol == 0 and np.isfinite(pred_min) and pred_min > 0.0),
            "acceptance_gate_structural": 1,
            "acceptance_gate_information": 1,
            "acceptance_gate_geometry_non_degenerate": 1,
            "acceptance_gate_future_distance": 1,
        }
    else:
        pred_min = _as_float(first.get("pred_min_safe_margin_px", float("nan")))
        pred_terminal = _as_float(first.get("pred_terminal_margin_px", float("nan")))
        pred_viol = _as_int(first.get("pred_violation_count", -1), -1) if "pred_violation_count" in first else -1
        pred_neg_margin_sum = _as_float(first.get("pred_negative_margin_sum_px", float("nan")))
        pred_min_rel_dist = _as_float(first.get("pred_min_relative_distance_m", float("nan")))
        pred_terminal_rel_dist = _as_float(first.get("pred_terminal_relative_distance_m", float("nan")))
        pred_min_future = _as_float(first.get("pred_min_future_min_distance_m", float("nan")))
        pred_terminal_future = _as_float(first.get("pred_terminal_future_min_distance_m", float("nan")))
        pred_worst_rank = _as_int(first.get("pred_worst_risk_rank", -999), -999) if "pred_worst_risk_rank" in first else -999
        strict_acceptance = _as_int(first.get("strict_acceptance_passed", -1), -1) if "strict_acceptance_passed" in first else -1
        acceptance_trace_ratio_vs_zero = _as_float(first.get("acceptance_trace_ratio_vs_zero", float("nan")))
        acceptance_terminal_trace_ratio_vs_zero = _as_float(first.get("acceptance_terminal_trace_ratio_vs_zero", float("nan")))
        acceptance_info_delta_vs_zero = _as_float(first.get("acceptance_info_delta_vs_zero", float("nan")))
        planner_candidate_status = str(first.get("planner_candidate_status", ""))
        planner_acceptance_status = str(first.get("planner_acceptance_status", ""))
        planner_selection_status = str(first.get("planner_selection_status", ""))
        planner_initial_selection_status = str(first.get("planner_initial_selection_status", ""))
        final_selected_strategy = str(first.get("final_selected_strategy", ""))
        try:
            selection_correction_applied = _as_int(first.get("selection_correction_applied", 0), 0) if "selection_correction_applied" in first else 0
        except Exception:
            selection_correction_applied = 0
        planner_execution_stage_status = str(first.get("planner_execution_stage_status", first.get("pre_execution_status", "")))
        pre_execution_status = str(first.get("pre_execution_status", first.get("planner_execution_stage_status", "")))
        selected_future_distance_gain_vs_zero_m = _as_float(first.get("selected_future_distance_gain_vs_zero_m", float("nan")))

        if "step_dv_mag_mps" in use.columns:
            dv = use["step_dv_mag_mps"].to_numpy(dtype=float)
            total_dv = float(np.sum(dv))
            nonzero_steps = int(np.sum(dv > 1.0e-10))
            dominant_ratio = float(np.max(dv) / max(total_dv, 1.0e-12)) if nonzero_steps > 0 else 0.0
            nz_idx = np.where(dv > 1.0e-10)[0]
            span_steps = int(nz_idx[-1] - nz_idx[0]) if nz_idx.size >= 2 else 0
        else:
            total_dv = _as_float(_prefixed_value(full_first, strategy, "total_dv_mag_mps", float("nan")))
            nonzero_steps = _as_int(_prefixed_value(full_first, strategy, "nonzero_steps", -1), -1)
            dominant_ratio = _as_float(_prefixed_value(full_first, strategy, "dominant_ratio", float("nan")))
            span_steps = _as_int(_prefixed_value(full_first, strategy, "span_steps", -1), -1)

        structural_pref = _prefixed_value(full_first, strategy, "structurally_sane", None)
        if structural_pref is not None:
            try:
                structurally_sane = _as_int(structural_pref, -1)
            except Exception:
                structurally_sane = -1
        else:
            structurally_sane = int(
                (nonzero_steps >= 2)
                and np.isfinite(dominant_ratio)
                and (dominant_ratio <= 0.95)
                and ((span_steps >= 1) or (nonzero_steps <= 1))
            ) if nonzero_steps >= 0 else -1
        gate_defaults = {gate: _as_int(first.get(gate, -1), -1) if gate in first else -1 for gate in ACCEPTANCE_GATE_KEYS}

    out = {
        "strategy": str(strategy),
        "pred_min_safe_margin_px": float(pred_min),
        "pred_terminal_margin_px": float(pred_terminal),
        "pred_violation_count": int(pred_viol),
        "pred_negative_margin_sum_px": float(pred_neg_margin_sum),
        "pred_min_relative_distance_m": float(pred_min_rel_dist),
        "pred_terminal_relative_distance_m": float(pred_terminal_rel_dist),
        "pred_min_future_min_distance_m": float(pred_min_future),
        "pred_terminal_future_min_distance_m": float(pred_terminal_future),
        "pred_worst_risk_rank": int(pred_worst_rank),
        "strict_acceptance_passed": int(strict_acceptance),
        "acceptance_trace_ratio_vs_zero": float(acceptance_trace_ratio_vs_zero),
        "acceptance_terminal_trace_ratio_vs_zero": float(acceptance_terminal_trace_ratio_vs_zero),
        "acceptance_info_delta_vs_zero": float(acceptance_info_delta_vs_zero),
        "planner_candidate_status": planner_candidate_status,
        "planner_acceptance_status": planner_acceptance_status,
        "planner_selection_status": planner_selection_status,
        "planner_initial_selection_status": planner_initial_selection_status,
        "final_selected_strategy": final_selected_strategy,
        "selection_correction_applied": int(selection_correction_applied),
        "planner_execution_stage_status": planner_execution_stage_status,
        "pre_execution_status": pre_execution_status,
        "nonzero_steps": int(nonzero_steps),
        "span_steps": int(span_steps),
        "dominant_ratio": float(dominant_ratio),
        "structurally_sane": int(structurally_sane),
        "total_dv_mag_mps": float(total_dv),
        "selected_future_distance_gain_vs_zero_m": float(selected_future_distance_gain_vs_zero_m),
    }
    for gate in ACCEPTANCE_GATE_KEYS:
        out[gate] = _as_int(gate_defaults.get(gate, -1), -1)
    return out


def reconcile_selected_plan_meta(plan_meta: dict[str, Any], selected_metrics: dict[str, Any]) -> dict[str, Any]:
    reconciled = dict(plan_meta)
    for key in [
        "pred_min_safe_margin_px",
        "pred_terminal_margin_px",
        "pred_negative_margin_sum_px",
        "pred_min_relative_distance_m",
        "pred_terminal_relative_distance_m",
        "pred_min_future_min_distance_m",
        "pred_terminal_future_min_distance_m",
    ]:
        metric_val = _as_float(selected_metrics.get(key, float("nan")))
        if np.isfinite(metric_val):
            reconciled[f"selected_{key}"] = float(metric_val)

    metric_rank = _as_int(selected_metrics.get("pred_worst_risk_rank", -999), -999)
    if metric_rank >= -998:
        reconciled["selected_pred_worst_risk_rank"] = int(metric_rank)

    metric_viol = selected_metrics.get("pred_violation_count", -1)
    try:
        metric_viol = _as_int(metric_viol, -1)
    except Exception:
        metric_viol = -1
    if metric_viol >= 0:
        reconciled["selected_pred_violation_count"] = int(metric_viol)

    metric_nonzero = selected_metrics.get("nonzero_steps", -1)
    try:
        metric_nonzero = _as_int(metric_nonzero, -1)
    except Exception:
        metric_nonzero = -1
    if metric_nonzero >= 0:
        reconciled["selected_nonzero_steps"] = int(metric_nonzero)

    metric_total_dv = _as_float(selected_metrics.get("total_dv_mag_mps", float("nan")))
    if np.isfinite(metric_total_dv):
        reconciled["selected_total_dv_mag_mps"] = float(metric_total_dv)

    metric_span = selected_metrics.get("span_steps", -1)
    try:
        metric_span = _as_int(metric_span, -1)
    except Exception:
        metric_span = -1
    if metric_span >= 0:
        reconciled["selected_span_steps"] = int(metric_span)

    metric_dominant_ratio = _as_float(selected_metrics.get("dominant_ratio", float("nan")))
    if np.isfinite(metric_dominant_ratio):
        reconciled["selected_dominant_ratio"] = float(metric_dominant_ratio)

    metric_structurally_sane = selected_metrics.get("structurally_sane", -1)
    try:
        metric_structurally_sane = _as_int(metric_structurally_sane, -1)
    except Exception:
        metric_structurally_sane = -1
    if metric_structurally_sane >= 0:
        reconciled["selected_structurally_sane"] = int(metric_structurally_sane)

    metric_future_gain = _as_float(
        selected_metrics.get("selected_future_distance_gain_vs_zero_m", float("nan"))
    )
    if np.isfinite(metric_future_gain):
        reconciled["selected_future_distance_gain_vs_zero_m"] = float(metric_future_gain)

    planner_initial_selection_status = str(
        selected_metrics.get("planner_initial_selection_status", "")
    ).strip()
    if planner_initial_selection_status:
        reconciled["planner_initial_selection_status"] = planner_initial_selection_status

    final_selected_strategy = str(selected_metrics.get("final_selected_strategy", "")).strip().lower()
    if final_selected_strategy in {"robust", "nominal", "zero"}:
        reconciled["final_selected_strategy"] = final_selected_strategy

    selection_correction_applied = selected_metrics.get("selection_correction_applied", None)
    if selection_correction_applied is not None:
        try:
            reconciled["selection_correction_applied"] = _as_int(selection_correction_applied, 0)
        except Exception:
            pass

    strict_accept = _as_int(selected_metrics.get("strict_acceptance_passed", -1), -1)
    if strict_accept >= 0:
        reconciled["selected_strict_acceptance_passed"] = int(strict_accept)

    for gate in ACCEPTANCE_GATE_KEYS:
        gate_val = _as_int(selected_metrics.get(gate, -1), -1)
        if gate_val >= 0:
            reconciled[f"selected_{gate}"] = int(gate_val)
    return reconciled


def build_plan_summary_context(
    plan_df: pd.DataFrame,
    plan_meta: dict[str, Any],
    orig_truth_summary: dict[str, Any] | None,
    resolve_execution_strategy_fn: Callable[[dict[str, Any]], tuple[str, str]],
) -> dict[str, Any]:
    orig_truth_summary = dict(orig_truth_summary or {})

    selected_strategy = _normalized_strategy(
        plan_meta.get("final_selected_strategy", plan_meta.get("selected_strategy", "robust")),
        default="robust",
    )

    robust_metrics = extract_strategy_plan_metrics(plan_df, "robust")
    nominal_metrics = extract_strategy_plan_metrics(plan_df, "nominal")
    zero_metrics = extract_strategy_plan_metrics(plan_df, "zero")
    selected_metrics = extract_strategy_plan_metrics(plan_df, selected_strategy)

    selected_metrics_final_strategy = _normalized_strategy(
        selected_metrics.get("final_selected_strategy", ""),
        default=selected_strategy,
    )
    if selected_metrics_final_strategy != selected_strategy:
        selected_strategy = selected_metrics_final_strategy
        selected_metrics = extract_strategy_plan_metrics(plan_df, selected_strategy)

    planner_initial_selection_status = str(
        selected_metrics.get(
            "planner_initial_selection_status",
            plan_meta.get("planner_initial_selection_status", ""),
        )
    )
    planner_selection_status = f"selected_{selected_strategy}"
    if planner_initial_selection_status.strip() == "":
        planner_initial_selection_status = planner_selection_status
    selection_correction_applied = int(
        planner_initial_selection_status.strip().lower() != planner_selection_status
    )

    authoritative_plan_meta = dict(plan_meta)
    authoritative_plan_meta["strict_acceptance_passed"] = _as_int(selected_metrics.get("strict_acceptance_passed", -1), -1)
    authoritative_plan_meta["acceptance_trace_ratio_vs_zero"] = float(selected_metrics.get("acceptance_trace_ratio_vs_zero", np.nan))
    authoritative_plan_meta["acceptance_terminal_trace_ratio_vs_zero"] = float(
        selected_metrics.get("acceptance_terminal_trace_ratio_vs_zero", np.nan)
    )
    authoritative_plan_meta["acceptance_info_delta_vs_zero"] = float(
        selected_metrics.get("acceptance_info_delta_vs_zero", np.nan)
    )
    for gate in ACCEPTANCE_GATE_KEYS:
        gate_val = _as_int(selected_metrics.get(gate, -1), -1)
        authoritative_plan_meta[gate] = int(gate_val)
        authoritative_plan_meta[f"selected_{gate}"] = int(gate_val)

    for prefix, metrics in (("nominal", nominal_metrics), ("robust", robust_metrics)):
        authoritative_plan_meta[f"{prefix}_pred_min_safe_margin_px"] = float(metrics["pred_min_safe_margin_px"])
        authoritative_plan_meta[f"{prefix}_pred_terminal_margin_px"] = float(metrics["pred_terminal_margin_px"])
        authoritative_plan_meta[f"{prefix}_pred_violation_count"] = _as_int(metrics["pred_violation_count"], -1)
        authoritative_plan_meta[f"{prefix}_pred_negative_margin_sum_px"] = float(metrics["pred_negative_margin_sum_px"])
        authoritative_plan_meta[f"{prefix}_pred_min_relative_distance_m"] = float(metrics["pred_min_relative_distance_m"])
        authoritative_plan_meta[f"{prefix}_pred_terminal_relative_distance_m"] = float(metrics["pred_terminal_relative_distance_m"])
        authoritative_plan_meta[f"{prefix}_pred_min_future_min_distance_m"] = float(metrics["pred_min_future_min_distance_m"])
        authoritative_plan_meta[f"{prefix}_pred_terminal_future_min_distance_m"] = float(metrics["pred_terminal_future_min_distance_m"])
        authoritative_plan_meta[f"{prefix}_pred_worst_risk_rank"] = _as_int(metrics["pred_worst_risk_rank"], -999)
        authoritative_plan_meta[f"{prefix}_strict_acceptance_passed"] = _as_int(metrics["strict_acceptance_passed"], -1)
        authoritative_plan_meta[f"{prefix}_trace_ratio_vs_zero"] = float(
            metrics.get("acceptance_trace_ratio_vs_zero", np.nan)
        )
        authoritative_plan_meta[f"{prefix}_terminal_trace_ratio_vs_zero"] = float(
            metrics.get("acceptance_terminal_trace_ratio_vs_zero", np.nan)
        )
        authoritative_plan_meta[f"{prefix}_support_min_safe_margin_px"] = float(metrics["pred_min_safe_margin_px"])
        authoritative_plan_meta[f"{prefix}_support_violation_count_pred"] = _as_int(metrics["pred_violation_count"], -1)
        authoritative_plan_meta[f"{prefix}_support_strict_acceptance_passed"] = _as_int(metrics["strict_acceptance_passed"], -1)
        for gate in ACCEPTANCE_GATE_KEYS:
            gate_val = _as_int(metrics.get(gate, -1), -1)
            authoritative_plan_meta[f"{prefix}_{gate}"] = int(gate_val)
            authoritative_plan_meta[f"{prefix}_support_{gate}"] = int(gate_val)

    authoritative_plan_meta["planner_candidate_status"] = str(selected_metrics.get("planner_candidate_status", ""))
    authoritative_plan_meta["planner_acceptance_status"] = str(selected_metrics.get("planner_acceptance_status", ""))
    authoritative_plan_meta["planner_selection_status"] = planner_selection_status
    authoritative_plan_meta["planner_initial_selection_status"] = planner_initial_selection_status
    authoritative_plan_meta["final_selected_strategy"] = selected_strategy
    authoritative_plan_meta["selection_correction_applied"] = int(selection_correction_applied)
    authoritative_plan_meta["selected_strategy"] = selected_strategy
    authoritative_plan_meta["planner_execution_stage_status"] = str(
        selected_metrics.get("planner_execution_stage_status", selected_metrics.get("pre_execution_status", ""))
    )
    authoritative_plan_meta["pre_execution_status"] = str(
        selected_metrics.get("pre_execution_status", selected_metrics.get("planner_execution_stage_status", ""))
    )

    execution_strategy, execution_strategy_reason = resolve_execution_strategy_fn(authoritative_plan_meta)
    execution_strategy = _normalized_strategy(execution_strategy, default=selected_strategy)
    execution_metrics = extract_strategy_plan_metrics(plan_df, execution_strategy)

    plan_meta_for_execution = reconcile_selected_plan_meta(authoritative_plan_meta, execution_metrics)

    # Keep zero baseline summary values self-consistent: prefer the extracted
    # zero strategy metrics from plan_df, and use planner metadata only as a
    # fallback when those values are unavailable.
    plan_meta_for_execution["zero_pred_min_safe_margin_px"] = _first_finite_float(
        zero_metrics.get("pred_min_safe_margin_px", np.nan),
        plan_meta_for_execution.get("zero_pred_min_safe_margin_px", np.nan),
        authoritative_plan_meta.get("zero_pred_min_safe_margin_px", np.nan),
    )
    plan_meta_for_execution["zero_pred_terminal_margin_px"] = _first_finite_float(
        zero_metrics.get("pred_terminal_margin_px", np.nan),
        plan_meta_for_execution.get("zero_pred_terminal_margin_px", np.nan),
        authoritative_plan_meta.get("zero_pred_terminal_margin_px", np.nan),
    )
    plan_meta_for_execution["zero_pred_negative_margin_sum_px"] = _first_finite_float(
        zero_metrics.get("pred_negative_margin_sum_px", np.nan),
        plan_meta_for_execution.get("zero_pred_negative_margin_sum_px", np.nan),
        authoritative_plan_meta.get("zero_pred_negative_margin_sum_px", np.nan),
    )
    plan_meta_for_execution["zero_pred_min_relative_distance_m"] = _first_finite_float(
        zero_metrics.get("pred_min_relative_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_min_relative_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_min_relative_distance_m", np.nan),
    )
    plan_meta_for_execution["zero_pred_terminal_relative_distance_m"] = _first_finite_float(
        zero_metrics.get("pred_terminal_relative_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_terminal_relative_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_terminal_relative_distance_m", np.nan),
    )
    plan_meta_for_execution["zero_pred_min_future_min_distance_m"] = _first_finite_float(
        zero_metrics.get("pred_min_future_min_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_min_future_min_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_min_future_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_min_future_min_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_min_future_distance_m", np.nan),
    )
    plan_meta_for_execution["zero_pred_terminal_future_min_distance_m"] = _first_finite_float(
        zero_metrics.get("pred_terminal_future_min_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_terminal_future_min_distance_m", np.nan),
        plan_meta_for_execution.get("zero_pred_terminal_future_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_terminal_future_min_distance_m", np.nan),
        authoritative_plan_meta.get("zero_pred_terminal_future_distance_m", np.nan),
    )
    plan_meta_for_execution["zero_pred_violation_count"] = _as_int(
        zero_metrics.get("pred_violation_count", plan_meta_for_execution.get("zero_pred_violation_count", -1)),
        _as_int(plan_meta_for_execution.get("zero_pred_violation_count", -1), -1),
    )
    plan_meta_for_execution["zero_pred_worst_risk_rank"] = _as_int(
        zero_metrics.get("pred_worst_risk_rank", plan_meta_for_execution.get("zero_pred_worst_risk_rank", -999)),
        _as_int(plan_meta_for_execution.get("zero_pred_worst_risk_rank", -999), -999),
    )

    plan_meta_for_execution["requested_selected_strategy"] = selected_strategy
    plan_meta_for_execution["selected_strategy"] = selected_strategy
    plan_meta_for_execution["execution_strategy"] = execution_strategy
    plan_meta_for_execution["execution_strategy_reason"] = execution_strategy_reason
    plan_meta_for_execution["orig_full_min_relative_distance_m"] = float(orig_truth_summary.get("min_relative_distance_m", np.nan))
    plan_meta_for_execution["orig_full_min_future_min_distance_m"] = float(orig_truth_summary.get("min_future_min_distance_m", np.nan))
    plan_meta_for_execution["orig_full_worst_risk_rank"] = _as_int(orig_truth_summary.get("worst_risk_rank", -999), -999)
    plan_meta_for_execution["robust_support_min_safe_margin_px"] = float(robust_metrics["pred_min_safe_margin_px"])
    plan_meta_for_execution["robust_support_violation_count_pred"] = _as_int(robust_metrics["pred_violation_count"], -1)
    plan_meta_for_execution["robust_support_strict_acceptance_passed"] = _as_int(robust_metrics["strict_acceptance_passed"], -1)
    plan_meta_for_execution["nominal_support_min_safe_margin_px"] = float(nominal_metrics["pred_min_safe_margin_px"])
    plan_meta_for_execution["nominal_support_violation_count_pred"] = _as_int(nominal_metrics["pred_violation_count"], -1)
    plan_meta_for_execution["nominal_support_strict_acceptance_passed"] = _as_int(nominal_metrics["strict_acceptance_passed"], -1)
    for gate in ACCEPTANCE_GATE_KEYS:
        robust_gate = _as_int(robust_metrics.get(gate, -1), -1)
        nominal_gate = _as_int(nominal_metrics.get(gate, -1), -1)
        plan_meta_for_execution[f"robust_support_{gate}"] = _as_int(robust_gate, -1)
        plan_meta_for_execution[f"nominal_support_{gate}"] = _as_int(nominal_gate, -1)

    execution_prefix = str(execution_strategy).lower()
    execution_span_steps = _as_int(execution_metrics.get("span_steps", -1), -1)
    execution_structurally_sane = _as_int(execution_metrics.get("structurally_sane", -1), -1)
    execution_dominant_ratio = _as_float(execution_metrics.get("dominant_ratio", np.nan))
    if execution_prefix in {"robust", "nominal"}:
        execution_span_steps = _as_int(
            plan_meta_for_execution.get(f"{execution_prefix}_span_steps", execution_span_steps), -1
        )
        execution_structurally_sane = _as_int(
            plan_meta_for_execution.get(
                f"{execution_prefix}_structurally_sane", execution_structurally_sane
            ), -1
        )
        execution_dominant_ratio = _as_float(
            plan_meta_for_execution.get(f"{execution_prefix}_dominant_ratio", execution_dominant_ratio)
        )

    execution_request: dict[str, Any] = {
        "strategy": str(execution_strategy),
        "source_reason": str(execution_strategy_reason),
        "requested_selected_strategy": str(selected_strategy),
        "pred_min_safe_margin_px": float(execution_metrics["pred_min_safe_margin_px"]),
        "pred_terminal_margin_px": float(execution_metrics["pred_terminal_margin_px"]),
        "pred_violation_count": _as_int(execution_metrics["pred_violation_count"], -1),
        "pred_negative_margin_sum_px": float(execution_metrics["pred_negative_margin_sum_px"]),
        "pred_min_relative_distance_m": float(execution_metrics["pred_min_relative_distance_m"]),
        "pred_terminal_relative_distance_m": float(execution_metrics["pred_terminal_relative_distance_m"]),
        "nonzero_steps": _as_int(execution_metrics["nonzero_steps"], -1),
        "total_dv_mag_mps": float(execution_metrics["total_dv_mag_mps"]),
        "span_steps": int(execution_span_steps),
        "structurally_sane": int(execution_structurally_sane),
        "dominant_ratio": float(execution_dominant_ratio),
    }
    plan_meta_for_execution["execution_request_strategy"] = str(execution_request["strategy"])
    plan_meta_for_execution["execution_request_source_reason"] = str(execution_request["source_reason"])
    plan_meta_for_execution["execution_request_requested_selected_strategy"] = str(
        execution_request["requested_selected_strategy"]
    )
    plan_meta_for_execution["execution_request_pred_min_safe_margin_px"] = float(
        execution_request["pred_min_safe_margin_px"]
    )
    plan_meta_for_execution["execution_request_pred_terminal_margin_px"] = float(
        execution_request["pred_terminal_margin_px"]
    )
    plan_meta_for_execution["execution_request_pred_violation_count"] = _as_int(
        execution_request["pred_violation_count"], -1
    )
    plan_meta_for_execution["execution_request_pred_negative_margin_sum_px"] = float(
        execution_request["pred_negative_margin_sum_px"]
    )
    plan_meta_for_execution["execution_request_pred_min_relative_distance_m"] = float(
        execution_request["pred_min_relative_distance_m"]
    )
    plan_meta_for_execution["execution_request_pred_terminal_relative_distance_m"] = float(
        execution_request["pred_terminal_relative_distance_m"]
    )
    plan_meta_for_execution["execution_request_nonzero_steps"] = _as_int(execution_request["nonzero_steps"], -1)
    plan_meta_for_execution["execution_request_total_dv_mag_mps"] = float(
        execution_request["total_dv_mag_mps"]
    )
    plan_meta_for_execution["execution_request_span_steps"] = _as_int(execution_request["span_steps"], -1)
    plan_meta_for_execution["execution_request_structurally_sane"] = _as_int(
        execution_request["structurally_sane"], -1
    )
    plan_meta_for_execution["execution_request_dominant_ratio"] = float(
        execution_request["dominant_ratio"]
    )

    plan_ok = bool(plan_meta.get("plan_accepted", True))
    pred_margin = float(execution_metrics["pred_min_safe_margin_px"])
    pred_viol = _as_int(execution_metrics["pred_violation_count"], -1)
    accept_reason = str(plan_meta.get("plan_acceptance_reason", "unknown"))

    selected_meta_margin = _as_float(plan_meta.get("selected_pred_min_safe_margin_px", float("nan")))
    selected_meta_viol = _as_int(plan_meta.get("selected_pred_violation_count", -1), -1)
    selected_margin_mismatch = int(
        np.isfinite(selected_meta_margin)
        and np.isfinite(selected_metrics["pred_min_safe_margin_px"])
        and abs(selected_meta_margin - float(selected_metrics["pred_min_safe_margin_px"])) > 1.0e-6
    )
    selected_viol_mismatch = int(
        selected_meta_viol >= 0 and selected_meta_viol != _as_int(selected_metrics["pred_violation_count"], -1)
    )

    selected_nonzero_steps = _as_int(
        authoritative_plan_meta.get("selected_nonzero_steps", selected_metrics.get("nonzero_steps", -1))
    )
    selected_total_dv_mag_mps = float(
        _as_float(
            authoritative_plan_meta.get(
                "selected_total_dv_mag_mps", selected_metrics.get("total_dv_mag_mps", float("nan"))
            )
        )
    )
    selected_span_steps = _as_int(
        authoritative_plan_meta.get("selected_span_steps", selected_metrics.get("span_steps", -1))
    )
    selected_structurally_sane = _as_int(
        authoritative_plan_meta.get(
            "selected_structurally_sane", selected_metrics.get("structurally_sane", -1)
        )
    )
    selected_dominant_ratio = _as_float(
        authoritative_plan_meta.get("selected_dominant_ratio", selected_metrics.get("dominant_ratio", np.nan))
    )
    selected_future_distance_gain_vs_zero_m = _as_float(
        authoritative_plan_meta.get(
            "selected_future_distance_gain_vs_zero_m",
            selected_metrics.get("selected_future_distance_gain_vs_zero_m", float("nan")),
        )
    )
    if not np.isfinite(selected_future_distance_gain_vs_zero_m):
        selected_future = _as_float(selected_metrics.get("pred_min_future_min_distance_m", float("nan")))
        zero_future = _as_float(
            plan_meta_for_execution.get(
                "zero_pred_min_future_min_distance_m",
                plan_meta_for_execution.get("zero_pred_min_future_distance_m", float("nan")),
            )
        )
        if np.isfinite(selected_future) and np.isfinite(zero_future):
            selected_future_distance_gain_vs_zero_m = float(selected_future - zero_future)

    summary_fields: dict[str, Any] = {
        "nominal_pred_min_safe_margin_px": float(nominal_metrics["pred_min_safe_margin_px"]),
        "nominal_pred_terminal_margin_px": float(nominal_metrics["pred_terminal_margin_px"]),
        "nominal_pred_violation_count": _as_int(nominal_metrics["pred_violation_count"], -1),
        "nominal_pred_negative_margin_sum_px": float(nominal_metrics["pred_negative_margin_sum_px"]),
        "nominal_pred_min_relative_distance_m": float(nominal_metrics["pred_min_relative_distance_m"]),
        "nominal_pred_terminal_relative_distance_m": float(nominal_metrics["pred_terminal_relative_distance_m"]),
        "robust_pred_min_safe_margin_px": float(robust_metrics["pred_min_safe_margin_px"]),
        "robust_pred_terminal_margin_px": float(robust_metrics["pred_terminal_margin_px"]),
        "robust_pred_violation_count": _as_int(robust_metrics["pred_violation_count"], -1),
        "robust_pred_negative_margin_sum_px": float(robust_metrics["pred_negative_margin_sum_px"]),
        "robust_pred_min_relative_distance_m": float(robust_metrics["pred_min_relative_distance_m"]),
        "robust_pred_terminal_relative_distance_m": float(robust_metrics["pred_terminal_relative_distance_m"]),
        "selected_pred_min_safe_margin_px": float(selected_metrics["pred_min_safe_margin_px"]),
        "selected_pred_terminal_margin_px": float(selected_metrics["pred_terminal_margin_px"]),
        "selected_pred_violation_count": _as_int(selected_metrics["pred_violation_count"], -1),
        "selected_pred_negative_margin_sum_px": float(selected_metrics["pred_negative_margin_sum_px"]),
        "selected_pred_min_relative_distance_m": float(selected_metrics["pred_min_relative_distance_m"]),
        "selected_pred_terminal_relative_distance_m": float(selected_metrics["pred_terminal_relative_distance_m"]),
        "execution_strategy_requested": str(execution_request["strategy"]),
        "execution_request_source_reason": str(execution_request["source_reason"]),
        "execution_requested_pred_min_safe_margin_px": float(execution_request["pred_min_safe_margin_px"]),
        "execution_requested_pred_terminal_margin_px": float(execution_request["pred_terminal_margin_px"]),
        "execution_requested_pred_violation_count": _as_int(execution_request["pred_violation_count"], -1),
        "execution_requested_pred_negative_margin_sum_px": float(execution_request["pred_negative_margin_sum_px"]),
        "execution_requested_pred_min_relative_distance_m": float(execution_request["pred_min_relative_distance_m"]),
        "execution_requested_pred_terminal_relative_distance_m": float(execution_request["pred_terminal_relative_distance_m"]),
        "execution_requested_nonzero_steps": _as_int(execution_request["nonzero_steps"], -1),
        "execution_requested_total_dv_mag_mps": float(execution_request["total_dv_mag_mps"]),
        "execution_requested_span_steps": _as_int(execution_request["span_steps"], -1),
        "execution_requested_structurally_sane": _as_int(execution_request["structurally_sane"], -1),
        "execution_requested_dominant_ratio": float(execution_request["dominant_ratio"]),
        "requested_strategy": str(execution_request["strategy"]),
        "requested_source_reason": str(execution_request["source_reason"]),
        "requested_pred_min_safe_margin_px": float(execution_request["pred_min_safe_margin_px"]),
        "requested_pred_terminal_margin_px": float(execution_request["pred_terminal_margin_px"]),
        "requested_pred_violation_count": _as_int(execution_request["pred_violation_count"], -1),
        "requested_pred_negative_margin_sum_px": float(execution_request["pred_negative_margin_sum_px"]),
        "requested_pred_min_relative_distance_m": float(execution_request["pred_min_relative_distance_m"]),
        "requested_pred_terminal_relative_distance_m": float(execution_request["pred_terminal_relative_distance_m"]),
        "requested_nonzero_steps": _as_int(execution_request["nonzero_steps"], -1),
        "requested_total_dv_mag_mps": float(execution_request["total_dv_mag_mps"]),
        "requested_span_steps": _as_int(execution_request["span_steps"], -1),
        "requested_structurally_sane": _as_int(execution_request["structurally_sane"], -1),
        "requested_dominant_ratio": float(execution_request["dominant_ratio"]),
        "plan_accepted": int(bool(authoritative_plan_meta.get("plan_accepted", False))),
        "plan_acceptance_reason": str(authoritative_plan_meta.get("plan_acceptance_reason", "")),
        "planner_candidate_status": str(selected_metrics.get("planner_candidate_status", "")),
        "planner_acceptance_status": str(selected_metrics.get("planner_acceptance_status", "")),
        "planner_selection_status": str(planner_selection_status),
        "planner_initial_selection_status": str(planner_initial_selection_status),
        "final_selected_strategy": str(selected_strategy),
        "selection_correction_applied": int(selection_correction_applied),
        "planner_execution_stage_status": str(
            selected_metrics.get("planner_execution_stage_status", selected_metrics.get("pre_execution_status", ""))
        ),
        "pre_execution_status": str(
            selected_metrics.get("pre_execution_status", selected_metrics.get("planner_execution_stage_status", ""))
        ),
        "strict_acceptance_passed": _as_int(selected_metrics.get("strict_acceptance_passed", -1), -1),
        "acceptance_trace_ratio_vs_zero": float(selected_metrics.get("acceptance_trace_ratio_vs_zero", np.nan)),
        "acceptance_info_delta_vs_zero": float(selected_metrics.get("acceptance_info_delta_vs_zero", np.nan)),
        "selected_strategy": selected_strategy,
        "case_label": str(authoritative_plan_meta.get("case_label", "")),
        "total_dv_budget_mps": float(authoritative_plan_meta.get("total_dv_budget_mps", np.nan)),
        "budget_retry_used": _as_int(authoritative_plan_meta.get("budget_retry_used", 0), 0),
        "retry_from_case_label": str(authoritative_plan_meta.get("retry_from_case_label", "")),
        "max_retry_budget_reached": _as_int(authoritative_plan_meta.get("max_retry_budget_reached", 0), 0),
        "zero_pred_min_safe_margin_px": float(plan_meta_for_execution.get("zero_pred_min_safe_margin_px", np.nan)),
        "zero_pred_terminal_margin_px": float(plan_meta_for_execution.get("zero_pred_terminal_margin_px", np.nan)),
        "zero_pred_violation_count": _as_int(plan_meta_for_execution.get("zero_pred_violation_count", -1), -1),
        "zero_pred_negative_margin_sum_px": float(plan_meta_for_execution.get("zero_pred_negative_margin_sum_px", np.nan)),
        "zero_pred_min_relative_distance_m": float(plan_meta_for_execution.get("zero_pred_min_relative_distance_m", np.nan)),
        "zero_pred_terminal_relative_distance_m": float(plan_meta_for_execution.get("zero_pred_terminal_relative_distance_m", np.nan)),
        "zero_pred_min_future_min_distance_m": float(
            plan_meta_for_execution.get(
                "zero_pred_min_future_min_distance_m",
                plan_meta_for_execution.get("zero_pred_min_future_distance_m", np.nan),
            )
        ),
        "baseline_strategy": "zero",
        "baseline_pred_min_safe_margin_px": float(plan_meta_for_execution.get("zero_pred_min_safe_margin_px", np.nan)),
        "baseline_pred_terminal_margin_px": float(plan_meta_for_execution.get("zero_pred_terminal_margin_px", np.nan)),
        "baseline_pred_violation_count": _as_int(plan_meta_for_execution.get("zero_pred_violation_count", -1), -1),
        "baseline_pred_negative_margin_sum_px": float(plan_meta_for_execution.get("zero_pred_negative_margin_sum_px", np.nan)),
        "baseline_pred_min_relative_distance_m": float(plan_meta_for_execution.get("zero_pred_min_relative_distance_m", np.nan)),
        "baseline_pred_terminal_relative_distance_m": float(plan_meta_for_execution.get("zero_pred_terminal_relative_distance_m", np.nan)),
        "baseline_pred_min_future_min_distance_m": float(
            plan_meta_for_execution.get(
                "zero_pred_min_future_min_distance_m",
                plan_meta_for_execution.get("zero_pred_min_future_distance_m", np.nan),
            )
        ),
        "robust_support_min_safe_margin_px": float(robust_metrics["pred_min_safe_margin_px"]),
        "robust_support_violation_count_pred": _as_int(robust_metrics["pred_violation_count"], -1),
        "robust_support_strict_acceptance_passed": _as_int(robust_metrics["strict_acceptance_passed"], -1),
        "robust_support_acceptance_gate_safe": _as_int(robust_metrics["acceptance_gate_safe"], -1),
        "robust_support_acceptance_gate_structural": _as_int(robust_metrics["acceptance_gate_structural"], -1),
        "robust_support_acceptance_gate_information": _as_int(robust_metrics["acceptance_gate_information"], -1),
        "robust_support_acceptance_gate_geometry_non_degenerate": _as_int(robust_metrics["acceptance_gate_geometry_non_degenerate"], -1),
        "robust_support_acceptance_gate_future_distance": _as_int(robust_metrics["acceptance_gate_future_distance"], -1),
        "nominal_support_min_safe_margin_px": float(nominal_metrics["pred_min_safe_margin_px"]),
        "nominal_support_violation_count_pred": _as_int(nominal_metrics["pred_violation_count"], -1),
        "nominal_support_strict_acceptance_passed": _as_int(nominal_metrics["strict_acceptance_passed"], -1),
        "nominal_support_acceptance_gate_safe": _as_int(nominal_metrics["acceptance_gate_safe"], -1),
        "nominal_support_acceptance_gate_structural": _as_int(nominal_metrics["acceptance_gate_structural"], -1),
        "nominal_support_acceptance_gate_information": _as_int(nominal_metrics["acceptance_gate_information"], -1),
        "nominal_support_acceptance_gate_geometry_non_degenerate": _as_int(nominal_metrics["acceptance_gate_geometry_non_degenerate"], -1),
        "nominal_support_acceptance_gate_future_distance": _as_int(nominal_metrics["acceptance_gate_future_distance"], -1),
        "acceptance_gate_safe": _as_int(selected_metrics["acceptance_gate_safe"], -1),
        "acceptance_gate_structural": _as_int(selected_metrics["acceptance_gate_structural"], -1),
        "acceptance_gate_information": _as_int(selected_metrics["acceptance_gate_information"], -1),
        "acceptance_gate_geometry_non_degenerate": _as_int(selected_metrics["acceptance_gate_geometry_non_degenerate"], -1),
        "acceptance_gate_future_distance": _as_int(selected_metrics["acceptance_gate_future_distance"], -1),
        "selected_acceptance_gate_safe": _as_int(selected_metrics["acceptance_gate_safe"], -1),
        "selected_acceptance_gate_structural": _as_int(selected_metrics["acceptance_gate_structural"], -1),
        "selected_acceptance_gate_information": _as_int(selected_metrics["acceptance_gate_information"], -1),
        "selected_acceptance_gate_geometry_non_degenerate": _as_int(selected_metrics["acceptance_gate_geometry_non_degenerate"], -1),
        "selected_acceptance_gate_future_distance": _as_int(selected_metrics["acceptance_gate_future_distance"], -1),
        "borderline_margin_gain_px": float(authoritative_plan_meta.get("borderline_margin_gain_px", np.nan)),
        "selected_nonzero_steps": _as_int(selected_nonzero_steps, -1),
        "selected_total_dv_mag_mps": float(selected_total_dv_mag_mps),
        "selected_span_steps": _as_int(selected_span_steps, -1),
        "selected_structurally_sane": _as_int(selected_structurally_sane, -1),
        "selected_dominant_ratio": float(selected_dominant_ratio),
        "selected_future_distance_gain_vs_zero_m": float(selected_future_distance_gain_vs_zero_m),
        "selected_pred_source": "plan_meta_frozen_selected_schedule",
        "selected_plan_meta_margin_mismatch": int(selected_margin_mismatch),
        "selected_plan_meta_violation_mismatch": int(selected_viol_mismatch),
        "execution_warning_flags": str(authoritative_plan_meta.get("execution_warning_flags", "")),
        "execution_warning_count": _as_int(authoritative_plan_meta.get("execution_warning_count", 0), 0),
    }

    for key in TRIGGER_DIAGNOSTIC_KEYS:
        if key.endswith("_score"):
            summary_fields[key] = _as_int(authoritative_plan_meta.get(key, -1), -1)
        else:
            summary_fields[key] = float(authoritative_plan_meta.get(key, np.nan))

    return {
        "selected_strategy": selected_strategy,
        "robust_metrics": robust_metrics,
        "nominal_metrics": nominal_metrics,
        "selected_metrics": selected_metrics,
        "execution_strategy": str(execution_strategy),
        "execution_strategy_reason": str(execution_strategy_reason),
        "execution_metrics": execution_metrics,
        "execution_request": execution_request,
        "plan_meta_for_execution": plan_meta_for_execution,
        "plan_ok": bool(plan_ok),
        "pred_margin": float(pred_margin),
        "pred_viol": _as_int(pred_viol, -1),
        "accept_reason": accept_reason,
        "selected_meta_margin_mismatch": _as_int(selected_margin_mismatch, 0),
        "selected_meta_violation_mismatch": _as_int(selected_viol_mismatch, 0),
        "decision_summary_fields": summary_fields,
    }
