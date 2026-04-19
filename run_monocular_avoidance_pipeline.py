from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Stable imports regardless of where the script is launched from
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent
for extra in [PROJECT_ROOT, PROJECT_ROOT / "pipeline", PROJECT_ROOT / "sim"]:
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import config  # noqa: E402
from build_pseudo_los import build_pseudo_los_df  # noqa: E402
from imm_ukf_estimator import imm_run  # noqa: E402
from risk_prediction_from_ukf import run_risk_prediction  # noqa: E402
from active_sensing_plan_one_sample import plan_active_sensing  # noqa: E402
from apply_active_micro_dv_to_truth import apply_plan_to_truth  # noqa: E402
from pipeline.execution_policy import (  # noqa: E402
    decide_execution_veto as shared_decide_execution_veto,
    detect_estimator_divergence as shared_detect_estimator_divergence,
    resolve_execution_strategy as shared_resolve_execution_strategy,
)
from pipeline.decision_flow import (  # noqa: E402
    build_execution_decision_row as shared_build_execution_decision_row,
    classify_rejected_plan_veto as shared_classify_rejected_plan_veto,
    enrich_execution_decision_row as shared_enrich_execution_decision_row,
    run_execution_preview_and_gate as shared_run_execution_preview_and_gate,
)
from pipeline.plan_summary import (  # noqa: E402
    build_plan_summary_context as shared_build_plan_summary_context,
    extract_strategy_plan_metrics as shared_extract_strategy_plan_metrics,
    reconcile_selected_plan_meta as shared_reconcile_selected_plan_meta,
)


@dataclass
class PipelineResult:
    run_root: Path
    original_render_dir: Path
    original_risk_csv: Path
    decision_csv: Path
    action: str
    active_truth_csv: Path | None = None
    active_render_dir: Path | None = None
    active_risk_csv: Path | None = None
    plan_csv: Path | None = None
    executed_plan_csv: Path | None = None


USE_PARQUET_IO = False
USE_BLENDER_SERVER = False
BLENDER_SERVER_URL = str(getattr(config, "BLENDER_SERVER_URL", "http://127.0.0.1:8765/render"))
BLENDER_SERVER_TIMEOUT_SEC = float(getattr(config, "BLENDER_SERVER_TIMEOUT_SEC", 120.0))


def _resolve_use_parquet(use_parquet: bool | None = None) -> bool:
    if use_parquet is None:
        return bool(USE_PARQUET_IO)
    return bool(use_parquet)


def _parquet_path(path: Path) -> Path:
    if path.suffix.lower() == ".parquet":
        return path
    return path.with_suffix(".parquet")


def read_table(path: Path, *, use_parquet: bool | None = None) -> pd.DataFrame:
    use_pq = _resolve_use_parquet(use_parquet)
    if use_pq:
        pq_path = _parquet_path(path)
        if pq_path.exists():
            try:
                return pd.read_parquet(pq_path)
            except Exception as exc:
                print(f"[WARN] failed to read parquet {pq_path}: {exc}; fallback to CSV")
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: Path, *, use_parquet: bool | None = None) -> None:
    use_pq = _resolve_use_parquet(use_parquet)
    path.parent.mkdir(parents=True, exist_ok=True)
    if use_pq:
        pq_path = _parquet_path(path)
        try:
            df.to_parquet(pq_path, index=False, engine="pyarrow")
        except Exception as exc:
            print(f"[WARN] failed to write parquet {pq_path}: {exc}; CSV output is kept")
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_parquet_sidecar(df: pd.DataFrame, path: Path, *, use_parquet: bool | None = None) -> None:
    use_pq = _resolve_use_parquet(use_parquet)
    if not use_pq:
        return
    pq_path = _parquet_path(path)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(pq_path, index=False, engine="pyarrow")
    except Exception as exc:
        print(f"[WARN] failed to write parquet {pq_path}: {exc}")


def _render_via_blender_server(truth_csv: Path, out_dir: Path) -> None:
    payload = {
        "csv": str(truth_csv),
        "obj": str(config.TARGET_MODEL),
        "out": str(out_dir),
        "resx": int(config.IMG_W),
        "resy": int(config.IMG_H),
        "fps": int(config.RENDER_FPS),
        "fov_deg": float(config.RENDER_FOV_DEG),
        "scale": float(config.RENDER_SCALE),
        "emission": float(config.RENDER_EMISSION),
    }
    req = urlrequest.Request(
        BLENDER_SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=float(BLENDER_SERVER_TIMEOUT_SEC)) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    try:
        parsed = json.loads(body) if body else {}
    except Exception as exc:
        raise RuntimeError(f"invalid blender server response: {body[:200]}") from exc
    if str(parsed.get("status", "")).lower() != "ok":
        raise RuntimeError(str(parsed.get("error", "unknown blender server error")))


def render_truth(truth_csv: Path, out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if USE_BLENDER_SERVER:
        try:
            _render_via_blender_server(truth_csv, out_dir)
            print("Rendered via Blender server:", out_dir)
            return
        except (urlerror.URLError, TimeoutError, RuntimeError) as exc:
            print(f"[WARN] blender server unavailable or failed ({exc}); fallback to subprocess")

    cmd = [
        config.BLENDER_EXE,
        "-b",
        "-P", str(config.BLENDER_SCRIPT),
        "--",
        "--csv", str(truth_csv),
        "--obj", str(config.TARGET_MODEL),
        "--out", str(out_dir),
        "--resx", str(config.IMG_W),
        "--resy", str(config.IMG_H),
        "--fps", str(config.RENDER_FPS),
        "--fov_deg", str(config.RENDER_FOV_DEG),
        "--scale", str(config.RENDER_SCALE),
        "--emission", str(config.RENDER_EMISSION),
    ]
    print("Running Blender:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def merge_truth_and_labels(truth_csv: Path, render_dir: Path) -> pd.DataFrame:
    label_csv = render_dir / "labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing labels.csv: {label_csv}")

    truth_df = read_table(truth_csv)
    label_df = pd.read_csv(label_csv)

    label_df["truth_idx"] = label_df["frame"] - 1
    truth_df = truth_df.reset_index().rename(columns={"index": "truth_idx"})
    merged = pd.merge(label_df, truth_df, on="truth_idx", how="inner")
    merged["cx_norm"] = merged["cx"] / config.IMG_W
    merged["cy_norm"] = merged["cy"] / config.IMG_H
    merged["w_norm"] = merged["w"] / config.IMG_W
    merged["h_norm"] = merged["h"] / config.IMG_H
    merged["bbox_area"] = merged["w"] * merged["h"]
    return merged


def run_estimation_chain(truth_csv: Path, render_dir: Path, tag: str) -> pd.DataFrame:
    merged_csv = render_dir / "merged_observation_truth.csv"
    los_csv = render_dir / "merged_with_pseudo_los.csv"
    imm_csv = render_dir / "imm_ukf_estimation_result.csv"
    risk_csv = render_dir / "risk_prediction_result.csv"

    merged_df = merge_truth_and_labels(truth_csv, render_dir)
    write_table(merged_df, merged_csv)

    los_df = build_pseudo_los_df(
        merged_df,
        fov_x_deg=config.FOV_X_DEG,
        img_w=config.IMG_W,
        img_h=config.IMG_H,
    )
    write_table(los_df, los_csv)

    imm_df = imm_run(los_df, imm_csv)
    write_parquet_sidecar(imm_df, imm_csv)
    risk_df = run_risk_prediction(imm_df, risk_csv)
    risk_df["tag"] = tag
    write_table(risk_df, risk_csv)
    return risk_df


def infer_dataset_subset_label(truth_csv: Path) -> str:
    label = str(truth_csv.parent.name).strip().lower()
    if label in {"safe", "caution", "danger"}:
        return label
    return ""


def extract_strategy_plan_metrics(plan_df: pd.DataFrame, strategy: str) -> dict[str, Any]:
    return shared_extract_strategy_plan_metrics(plan_df, strategy)


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default




def reconcile_selected_plan_meta(plan_meta: dict[str, Any], selected_metrics: dict[str, Any]) -> dict[str, Any]:
    return shared_reconcile_selected_plan_meta(plan_meta, selected_metrics)
def build_empty_decision_row(truth_csv: Path, dataset_subset_label: str) -> dict[str, Any]:
    return {
        "truth_csv": str(truth_csv),
        "dataset_subset_label": str(dataset_subset_label),
        "last_risk_level": "",
        "max_pc_bplane": float("nan"),
        "min_pred_distance_m": float("nan"),
        "should_avoid": 0,
        "decision_reason": "",
        "risk_decision_reason": "",
        "force_avoid_requested": 0,
        "execute_avoidance_branch": 0,
        "execution_decision_reason": "",
        "apply_even_if_plan_rejected_requested": 0,
        "forced_apply_rejected_plan": 0,
        "trigger_frame": -1,
        "trigger_t_sec": float("nan"),
        "trigger_reason": "",
        "trigger_safe_dist_px": float("nan"),
        "trigger_safe_dist_rate_pxps": float("nan"),
        "trigger_short_time_to_warning_sec": float("nan"),
        "trigger_short_time_to_breach_sec": float("nan"),
        "trigger_risk_score": -1,
        "trigger_warning_score": -1,
        "trigger_risk_integral": float("nan"),
        "trigger_warning_integral": float("nan"),
        "trigger_short_pred_min_safe_margin_px": float("nan"),
        "trigger_pred_min_safe_margin_px": float("nan"),
        "trigger_est_range_m": float("nan"),
        "nominal_pred_min_safe_margin_px": float("nan"),
        "nominal_pred_terminal_margin_px": float("nan"),
        "nominal_pred_violation_count": -1,
        "nominal_pred_negative_margin_sum_px": float("nan"),
        "nominal_pred_min_relative_distance_m": float("nan"),
        "nominal_pred_terminal_relative_distance_m": float("nan"),
        "robust_pred_min_safe_margin_px": float("nan"),
        "robust_pred_terminal_margin_px": float("nan"),
        "robust_pred_violation_count": -1,
        "robust_pred_negative_margin_sum_px": float("nan"),
        "robust_pred_min_relative_distance_m": float("nan"),
        "robust_pred_terminal_relative_distance_m": float("nan"),
        "selected_pred_min_safe_margin_px": float("nan"),
        "selected_pred_terminal_margin_px": float("nan"),
        "selected_pred_violation_count": -1,
        "selected_pred_negative_margin_sum_px": float("nan"),
        "selected_pred_min_relative_distance_m": float("nan"),
        "selected_pred_terminal_relative_distance_m": float("nan"),
        "acceptance_gate_safe": -1,
        "acceptance_gate_structural": -1,
        "acceptance_gate_information": -1,
        "acceptance_gate_geometry_non_degenerate": -1,
        "acceptance_gate_future_distance": -1,
        "selected_acceptance_gate_safe": -1,
        "selected_acceptance_gate_structural": -1,
        "selected_acceptance_gate_information": -1,
        "selected_acceptance_gate_geometry_non_degenerate": -1,
        "selected_acceptance_gate_future_distance": -1,
        "acceptance_trace_ratio_vs_zero": float("nan"),
        "acceptance_info_delta_vs_zero": float("nan"),
        "zero_pred_min_safe_margin_px": float("nan"),
        "zero_pred_terminal_margin_px": float("nan"),
        "zero_pred_negative_margin_sum_px": float("nan"),
        "zero_pred_min_relative_distance_m": float("nan"),
        "zero_pred_terminal_relative_distance_m": float("nan"),
        "zero_pred_min_future_min_distance_m": float("nan"),
        "robust_support_min_safe_margin_px": float("nan"),
        "robust_support_violation_count_pred": -1,
        "robust_support_strict_acceptance_passed": -1,
        "robust_support_acceptance_gate_safe": -1,
        "robust_support_acceptance_gate_structural": -1,
        "robust_support_acceptance_gate_information": -1,
        "robust_support_acceptance_gate_geometry_non_degenerate": -1,
        "robust_support_acceptance_gate_future_distance": -1,
        "nominal_support_min_safe_margin_px": float("nan"),
        "nominal_support_violation_count_pred": -1,
        "nominal_support_strict_acceptance_passed": -1,
        "nominal_support_acceptance_gate_safe": -1,
        "nominal_support_acceptance_gate_structural": -1,
        "nominal_support_acceptance_gate_information": -1,
        "nominal_support_acceptance_gate_geometry_non_degenerate": -1,
        "nominal_support_acceptance_gate_future_distance": -1,
        "borderline_margin_gain_px": float("nan"),
        "total_dv_budget_mps": float("nan"),
        "case_label": "",
        "plan_accepted": 0,
        "plan_acceptance_reason": "",
        "strictly_feasible_plan": 0,
        "selected_strategy": "",
        "selected_nonzero_steps": -1,
        "selected_total_dv_mag_mps": float("nan"),
        "selected_span_steps": -1,
        "selected_structurally_sane": -1,
        "selected_future_distance_gain_vs_zero_m": float("nan"),
        "execution_selection_reason": "",
        "requested_plan_row_strategy": "",
        "requested_plan_selected_strategy": "",
        "requested_total_dv_mag_mps": float("nan"),
        "requested_nonzero_steps": -1,
        "execution_vetoed": 0,
        "execution_veto_reason": "",
        "execution_warning_flags": "",
        "execution_warning_count": 0,
        "execution_strategy_requested": "",
        "execution_request_source_reason": "",
        "execution_strategy_applied": "",
        "execution_applied_flag": 0,
        "executed_strategy": "",
        "executed_pred_min_safe_margin_px": float("nan"),
        "executed_pred_terminal_margin_px": float("nan"),
        "executed_pred_violation_count": -1,
        "executed_pred_negative_margin_sum_px": float("nan"),
        "executed_pred_min_relative_distance_m": float("nan"),
        "executed_pred_terminal_relative_distance_m": float("nan"),
        "executed_nonzero_steps": -1,
        "executed_total_dv_mag_mps": float("nan"),
        "execution_requested_pred_min_safe_margin_px": float("nan"),
        "execution_requested_pred_terminal_margin_px": float("nan"),
        "execution_requested_pred_violation_count": -1,
        "execution_requested_pred_negative_margin_sum_px": float("nan"),
        "execution_requested_pred_min_relative_distance_m": float("nan"),
        "execution_requested_pred_terminal_relative_distance_m": float("nan"),
        "execution_requested_nonzero_steps": -1,
        "execution_requested_total_dv_mag_mps": float("nan"),
        "execution_requested_span_steps": -1,
        "execution_requested_structurally_sane": -1,
        "execution_requested_dominant_ratio": float("nan"),
        "geometry_preview_truth_csv": "",
        "geometry_preview_status": "",
        "geometry_preview_error": "",
        "preview_pred_min_relative_distance_m": float("nan"),
        "preview_pred_min_future_min_distance_m": float("nan"),
        "preview_pred_last_risk_level": "",
        "preview_pred_worst_risk_rank": -999,
        "preview_pred_num_safe_steps": -1,
        "preview_pred_num_caution_steps": -1,
        "preview_pred_num_danger_steps": -1,
        "planner_candidate_status": "",
        "planner_acceptance_status": "",
        "planner_selection_status": "",
        "planner_initial_selection_status": "",
        "final_selected_strategy": "",
        "selection_correction_applied": 0,
        "planner_execution_stage_status": "",
        "pre_execution_status": "",
        "execution_final_status": "",
        "raw_planner_candidate_status": "",
        "raw_planner_acceptance_status": "",
        "raw_planner_selection_status": "",
        "raw_planner_execution_stage_status": "",
        "raw_execution_final_status": "",
        "derived_planner_candidate_status": "",
        "derived_planner_acceptance_status": "",
        "derived_planner_selection_status": "",
        "derived_execution_final_status": "",
        "derived_failure_family": "",
        "derived_planner_preview_consistency_flag": -1,
        "failure_family": "",
        "planner_preview_consistency_flag": -1,
    }


def decide_avoidance(risk_df: pd.DataFrame) -> dict[str, Any]:
    if len(risk_df) == 0:
        raise RuntimeError("risk_df is empty.")

    risk_level_col = "risk_level" if "risk_level" in risk_df.columns else None
    pc_col = "pc_bplane" if "pc_bplane" in risk_df.columns else None
    dist_col_candidates = [
        "future_min_distance_m",
        "future_min_distance_300s_m",
        "tca_min_distance_m",
        "pred_min_distance_m",
    ]
    dist_col = next((c for c in dist_col_candidates if c in risk_df.columns), None)

    last = risk_df.iloc[-1]
    max_pc = float(risk_df[pc_col].max()) if pc_col else float("nan")
    min_dist = float(risk_df[dist_col].min()) if dist_col else float("nan")
    last_level = str(last[risk_level_col]) if risk_level_col else "unknown"

    should_avoid = False
    reason = "continue_track"

    if risk_level_col and (risk_df[risk_level_col] == "danger").any():
        should_avoid = True
        reason = "danger_level_detected"
    elif pc_col and max_pc >= config.PC_DANGER:
        should_avoid = True
        reason = "pc_exceeds_danger_threshold"
    elif pc_col and dist_col and max_pc >= config.PC_CAUTION and min_dist <= config.DIST_CAUTION_M:
        should_avoid = True
        reason = "caution_pc_and_distance"

    return {
        "last_risk_level": last_level,
        "max_pc_bplane": max_pc,
        "min_pred_distance_m": min_dist,
        "should_avoid": int(should_avoid),
        "decision_reason": reason,
    }


def summarize_chain(risk_df: pd.DataFrame, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tag": tag,
        "summary_scope": "visible_window_estimation_chain",
        "num_visible": int(len(risk_df)),
    }
    for col in ["sigma_R_m", "sigma_T_m", "sigma_N_m", "trace_P_pos", "pc_bplane"]:
        if col in risk_df.columns:
            out[f"mean_{col}"] = float(risk_df[col].mean())
            out[f"max_{col}"] = float(risk_df[col].max())
    if "risk_level" in risk_df.columns and len(risk_df) > 0:
        out["last_risk_level"] = str(risk_df["risk_level"].iloc[-1])
    if "future_min_distance_m" in risk_df.columns:
        out["min_future_min_distance_m"] = float(risk_df["future_min_distance_m"].min())
    elif "pred_future_min_distance_m" in risk_df.columns:
        out["min_future_min_distance_m"] = float(risk_df["pred_future_min_distance_m"].min())
    return out


def summarize_truth_horizon(truth_df: pd.DataFrame, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tag": tag,
        "summary_scope": "full_horizon_truth",
        "num_truth_steps": int(len(truth_df)),
    }
    if len(truth_df) == 0:
        return out
    if "relative_distance_m" in truth_df.columns:
        idx_rel = int(truth_df["relative_distance_m"].astype(float).idxmin())
        out["min_relative_distance_m"] = float(truth_df.loc[idx_rel, "relative_distance_m"])
        if "t_sec" in truth_df.columns:
            out["min_relative_distance_t_sec"] = float(truth_df.loc[idx_rel, "t_sec"])
    if "future_min_distance_m" in truth_df.columns:
        idx_fut = int(truth_df["future_min_distance_m"].astype(float).idxmin())
        out["min_future_min_distance_m"] = float(truth_df.loc[idx_fut, "future_min_distance_m"])
        if "t_sec" in truth_df.columns:
            out["min_future_min_distance_t_sec"] = float(truth_df.loc[idx_fut, "t_sec"])
    if "risk_level" in truth_df.columns and len(truth_df) > 0:
        risk = truth_df["risk_level"].astype(str)
        out["last_risk_level"] = str(risk.iloc[-1])
        out["num_safe_steps"] = int((risk.str.lower() == "safe").sum())
        out["num_caution_steps"] = int((risk.str.lower() == "caution").sum())
        out["num_danger_steps"] = int((risk.str.lower() == "danger").sum())
    return out


def preview_execution_geometry(
    truth_csv: Path,
    plan_csv: Path,
    preview_truth_csv: Path,
    *,
    strategy: str,
) -> tuple[dict[str, Any], str]:
    try:
        preview_df = apply_plan_to_truth(
            truth_csv,
            plan_csv,
            preview_truth_csv,
            strategy=strategy,
            use_parquet=_resolve_use_parquet(),
        )
        preview_summary = summarize_truth_horizon(preview_df, "preview")
        return preview_summary, ""
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def compute_truth_error_metrics(df_est: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}

    axis_specs = [
        ("R", "R_est_m", "rel_R_m", "sigma_R_m"),
        ("T", "T_est_m", "rel_T_m", "sigma_T_m"),
        ("N", "N_est_m", "rel_N_m", "sigma_N_m"),
    ]

    for axis, est_col, truth_col, sigma_col in axis_specs:
        if est_col in df_est.columns and truth_col in df_est.columns:
            e = df_est[est_col].to_numpy(dtype=float) - df_est[truth_col].to_numpy(dtype=float)
            out[f"mean_abs_e_{axis}"] = float(abs(e).mean())
            out[f"rmse_{axis}"] = float((e ** 2).mean() ** 0.5)
            if sigma_col in df_est.columns:
                sigma = df_est[sigma_col].to_numpy(dtype=float)
                out[f"cover_3sigma_{axis}"] = float((abs(e) <= 3.0 * sigma).mean())

    needed = {"R_est_m", "T_est_m", "N_est_m", "rel_R_m", "rel_T_m", "rel_N_m"}
    if needed.issubset(set(df_est.columns)):
        epos = pd.DataFrame({
            "R": df_est["R_est_m"].to_numpy(dtype=float) - df_est["rel_R_m"].to_numpy(dtype=float),
            "T": df_est["T_est_m"].to_numpy(dtype=float) - df_est["rel_T_m"].to_numpy(dtype=float),
            "N": df_est["N_est_m"].to_numpy(dtype=float) - df_est["rel_N_m"].to_numpy(dtype=float),
        }).to_numpy(dtype=float)
        pos_norm = (epos ** 2).sum(axis=1) ** 0.5
        out["mean_abs_pos_err_m"] = float(pos_norm.mean())
        out["rmse_pos_m"] = float((pos_norm ** 2).mean() ** 0.5)

    return out


def detect_estimator_divergence(orig_summary: dict[str, Any], orig_err: dict[str, Any]) -> tuple[bool, str]:
    return shared_detect_estimator_divergence(orig_summary, orig_err)


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
    return shared_decide_execution_veto(
        orig_summary,
        orig_err,
        plan_meta,
        dataset_subset_label=dataset_subset_label,
        plan_ok=plan_ok,
        pred_margin=pred_margin,
        pred_viol=pred_viol,
    )



def resolve_execution_strategy(plan_meta: dict[str, Any]) -> tuple[str, str]:
    return shared_resolve_execution_strategy(plan_meta)



def extract_executed_plan_snapshot(plan_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(plan_df) == 0:
        return plan_df.copy(), {
            "applied_plan_row_strategy": "unknown",
            "applied_plan_selected_strategy": "unknown",
            "applied_total_dv_mag_mps": 0.0,
            "applied_nonzero_steps": 0,
        }

    use = plan_df.copy()
    selected_strategy = str(use["selected_strategy"].iloc[0]) if "selected_strategy" in use.columns and len(use) > 0 else "robust"
    applied_row_strategy = selected_strategy

    if "strategy" in use.columns:
        selected_rows = use[use["strategy"].astype(str).str.lower() == selected_strategy.lower()].copy()
        if len(selected_rows) == 0:
            selected_rows = use[use["strategy"].astype(str).str.lower() == "robust"].copy()
            if len(selected_rows) > 0:
                applied_row_strategy = "robust"
        if len(selected_rows) > 0:
            use = selected_rows

    applied_total_dv = float(use["step_dv_mag_mps"].sum()) if "step_dv_mag_mps" in use.columns else float("nan")
    applied_nonzero_steps = int((use["step_dv_mag_mps"] > 1.0e-10).sum()) if "step_dv_mag_mps" in use.columns else -1

    return use, {
        "applied_plan_row_strategy": applied_row_strategy,
        "applied_plan_selected_strategy": selected_strategy,
        "applied_total_dv_mag_mps": applied_total_dv,
        "applied_nonzero_steps": applied_nonzero_steps,
    }

def run_pipeline(
    truth_csv: Path,
    run_name: str,
    *,
    render_original: bool = True,
    render_active: bool = True,
    force_avoid: bool = False,
    apply_even_if_plan_rejected: bool = False,
) -> PipelineResult:
    run_root = config.VALIDATION_RUNS_DIR / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    original_render_dir = run_root / "original_render"
    active_render_dir = run_root / "active_render"
    active_truth_csv = run_root / f"{truth_csv.stem}_active.csv"
    plan_csv = run_root / "micro_dv_plan.csv"
    decision_csv = run_root / "decision_summary.csv"
    compare_csv = run_root / "before_after_summary.csv"
    full_horizon_csv = run_root / "full_horizon_truth_summary.csv"
    executed_plan_csv = run_root / "executed_plan_snapshot.csv"

    def persist_decision_row(row: dict[str, Any]) -> dict[str, Any]:
        enriched = shared_enrich_execution_decision_row(dict(row))
        write_table(pd.DataFrame([enriched]), decision_csv)
        return enriched

    print("=" * 72)
    print("[1/6] original render")
    if render_original or not (original_render_dir / "labels.csv").exists():
        render_truth(truth_csv, original_render_dir)

    print("[2/6] original estimation chain")
    orig_risk_df = run_estimation_chain(truth_csv, original_render_dir, "original")
    orig_truth_df = read_table(truth_csv)
    orig_truth_summary = summarize_truth_horizon(orig_truth_df, "original")
    dataset_subset_label = infer_dataset_subset_label(truth_csv)
    decision = decide_avoidance(orig_risk_df)
    if (
        dataset_subset_label in {"", "safe"}
        and
        str(decision.get("last_risk_level", "")).lower() == "safe"
        and pd.notna(decision.get("max_pc_bplane", float("nan")))
        and float(decision.get("max_pc_bplane", float("inf"))) <= float(config.PC_CAUTION)
    ):
        decision["should_avoid"] = 0
        decision["decision_reason"] = "safe_veto"

    should_avoid_by_risk = int(decision.get("should_avoid", 0))
    risk_decision_reason = str(decision.get("decision_reason", "unknown"))
    execute_avoidance_branch = int(bool(force_avoid) or bool(should_avoid_by_risk))
    execution_decision_reason = "force_avoid_override" if force_avoid else (
        "risk_triggered_avoidance" if should_avoid_by_risk else "risk_veto_track_only"
    )

    orig_summary = summarize_chain(orig_risk_df, "original")
    orig_err = compute_truth_error_metrics(orig_risk_df)
    estimator_diverged, divergence_reason = detect_estimator_divergence(orig_summary, orig_err)
    if estimator_diverged and not force_avoid:
        decision["should_avoid"] = 0
        decision["decision_reason"] = divergence_reason
        should_avoid_by_risk = 0
        risk_decision_reason = divergence_reason
        execute_avoidance_branch = 0
        execution_decision_reason = divergence_reason

    decision_row = shared_build_execution_decision_row(
        build_empty_decision_row(truth_csv, dataset_subset_label),
        {
            **decision,
            "risk_decision_reason": risk_decision_reason,
            "should_avoid": should_avoid_by_risk,
            "force_avoid_requested": int(bool(force_avoid)),
            "execute_avoidance_branch": execute_avoidance_branch,
            "execution_decision_reason": execution_decision_reason,
            "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
            **orig_summary,
            **{f"orig_{k}": v for k, v in orig_err.items()},
        },
    )

    action = "avoid" if execute_avoidance_branch == 1 else "track_only"

    if action == "track_only":
        decision_row = persist_decision_row(decision_row)
        print("[3/6] decision = track_only")
        return PipelineResult(
            run_root=run_root,
            original_render_dir=original_render_dir,
            original_risk_csv=original_render_dir / "risk_prediction_result.csv",
            decision_csv=decision_csv,
            action=action,
        )

    print("[3/6] planning avoidance schedule")
    plan_res = plan_active_sensing(
        input_csv=original_render_dir / "merged_with_pseudo_los.csv",
        out_dir=run_root,
        experiment_mode="exp3",
        imm_csv=original_render_dir / "imm_ukf_estimation_result.csv",
        orig_last_risk_level=str(orig_summary.get("last_risk_level", "")),
        orig_max_pc_bplane=float(orig_summary.get("max_pc_bplane", np.nan)),
        orig_min_future_min_distance_m=float(orig_summary.get("min_future_min_distance_m", np.nan)),
    )
    write_table(plan_res.plan_df, plan_csv)
    executed_plan_df, executed_plan_meta = extract_executed_plan_snapshot(plan_res.plan_df)
    write_table(executed_plan_df, executed_plan_csv)

    summary_ctx = shared_build_plan_summary_context(
        plan_res.plan_df,
        plan_res.plan_meta,
        orig_truth_summary,
        resolve_execution_strategy,
    )
    selected_strategy = str(summary_ctx["selected_strategy"])
    selected_metrics = dict(summary_ctx["selected_metrics"])
    execution_strategy = str(summary_ctx["execution_strategy"])
    execution_strategy_reason = str(summary_ctx["execution_strategy_reason"])
    execution_metrics = dict(summary_ctx["execution_metrics"])
    execution_request = dict(summary_ctx.get("execution_request", {}))
    plan_meta_for_execution = dict(summary_ctx["plan_meta_for_execution"])
    plan_ok = bool(summary_ctx["plan_ok"])
    pred_margin = float(summary_ctx["pred_margin"])
    pred_viol = int(summary_ctx["pred_viol"])

    decision_row.update({
        "trigger_frame": int(plan_res.trigger_row.get("frame", -1)),
        "trigger_t_sec": float(plan_res.trigger_row.get("t_sec", -1.0)),
        "trigger_reason": plan_res.trigger_reason,
        **dict(summary_ctx["decision_summary_fields"]),
        "execution_strategy_requested": str(execution_request.get("strategy", execution_strategy)),
        "execution_request_source_reason": str(execution_request.get("source_reason", execution_strategy_reason)),
        "execution_strategy_reason": str(execution_strategy_reason),
        "execution_selection_reason": str(plan_res.plan_meta.get("execution_selection_reason", "")),
        "strictly_feasible_plan": int(bool(plan_res.plan_meta.get("strictly_feasible_plan", plan_ok))),
        "requested_plan_row_strategy": str(executed_plan_meta.get("applied_plan_row_strategy", "")),
        "requested_plan_selected_strategy": str(executed_plan_meta.get("applied_plan_selected_strategy", "")),
        "requested_total_dv_mag_mps": float(executed_plan_meta.get("applied_total_dv_mag_mps", float("nan"))),
        "requested_nonzero_steps": int(executed_plan_meta.get("applied_nonzero_steps", -1)),
        "execution_vetoed": 0,
        "execution_veto_reason": "",
        "execution_applied_flag": 0,
        "executed_strategy": "",
        "executed_pred_min_safe_margin_px": float("nan"),
        "executed_pred_terminal_margin_px": float("nan"),
        "executed_pred_violation_count": -1,
        "executed_pred_negative_margin_sum_px": float("nan"),
        "executed_pred_min_relative_distance_m": float("nan"),
        "executed_pred_terminal_relative_distance_m": float("nan"),
        "executed_nonzero_steps": -1,
        "executed_total_dv_mag_mps": float("nan"),
        "executed_span_steps": -1,
        "executed_structurally_sane": -1,
        "executed_dominant_ratio": float("nan"),
    })
    decision_row = shared_enrich_execution_decision_row(decision_row)

    preview_truth_csv = run_root / f"preview_{execution_strategy}_truth.csv"
    planner_rejected = (not plan_ok) or (pred_margin <= 0.0) or (pred_viol > 0)
    if planner_rejected:
        rejection_reason = shared_classify_rejected_plan_veto(plan_ok, pred_margin, pred_viol, decision_row)
        decision_row["geometry_preview_truth_csv"] = str(preview_truth_csv)
        decision_row["geometry_preview_status"] = "skipped_planner_rejected"
        decision_row["geometry_preview_error"] = ""
        decision_row["execution_vetoed"] = 0
        decision_row["execution_veto_reason"] = ""
        decision_row["execution_decision_reason"] = rejection_reason

        if not apply_even_if_plan_rejected:
            action = "track_only"
            decision_row["action"] = action
            decision_row = persist_decision_row(decision_row)
            print(f"[4/6] planner rejected ({rejection_reason}) -> track_only")
            return PipelineResult(
                run_root=run_root,
                original_render_dir=original_render_dir,
                original_risk_csv=original_render_dir / "risk_prediction_result.csv",
                decision_csv=decision_csv,
                action=action,
                plan_csv=plan_csv,
                executed_plan_csv=executed_plan_csv,
            )

        action = "audit_apply_rejected_plan"
        decision_row["action"] = action
        decision_row["forced_apply_rejected_plan"] = 1
        print("[4/6] applying best-effort plan despite planner rejection")
    else:
        plan_meta_for_execution, decision_row, execution_vetoed, execution_veto_reason = shared_run_execution_preview_and_gate(
            truth_csv=truth_csv,
            plan_csv=plan_csv,
            preview_truth_csv=preview_truth_csv,
            strategy=execution_strategy,
            plan_meta_for_execution=plan_meta_for_execution,
            decision_row=decision_row,
            preview_func=preview_execution_geometry,
            veto_func=decide_execution_veto,
            orig_summary=orig_summary,
            orig_err=orig_err,
            dataset_subset_label=dataset_subset_label,
            plan_ok=plan_ok,
            pred_margin=pred_margin,
            pred_viol=pred_viol,
        )
        if execution_vetoed and not apply_even_if_plan_rejected:
            action = "track_only"
            decision_row["action"] = action
            decision_row["execution_vetoed"] = 1
            decision_row["execution_veto_reason"] = execution_veto_reason
            decision_row["execution_decision_reason"] = execution_veto_reason
            decision_row = persist_decision_row(decision_row)
            print(f"[4/6] {execution_veto_reason} -> track_only")
            return PipelineResult(
                run_root=run_root,
                original_render_dir=original_render_dir,
                original_risk_csv=original_render_dir / "risk_prediction_result.csv",
                decision_csv=decision_csv,
                action=action,
                plan_csv=plan_csv,
                executed_plan_csv=executed_plan_csv,
            )

        decision_row["execution_decision_reason"] = str(execution_strategy_reason or "accepted_plan_execute")

    applied_strategy = str(execution_strategy or selected_strategy or "robust")
    if applied_strategy not in {"robust", "nominal"}:
        applied_strategy = "robust"
    applied_metrics = extract_strategy_plan_metrics(plan_res.plan_df, applied_strategy)
    decision_row["execution_strategy_applied"] = applied_strategy
    decision_row["execution_applied_flag"] = 1
    decision_row["executed_strategy"] = applied_strategy
    decision_row["executed_pred_min_safe_margin_px"] = float(applied_metrics["pred_min_safe_margin_px"])
    decision_row["executed_pred_terminal_margin_px"] = float(applied_metrics["pred_terminal_margin_px"])
    decision_row["executed_pred_violation_count"] = int(applied_metrics["pred_violation_count"])
    decision_row["executed_pred_negative_margin_sum_px"] = float(applied_metrics["pred_negative_margin_sum_px"])
    decision_row["executed_pred_min_relative_distance_m"] = float(applied_metrics["pred_min_relative_distance_m"])
    decision_row["executed_pred_terminal_relative_distance_m"] = float(applied_metrics["pred_terminal_relative_distance_m"])
    decision_row["executed_nonzero_steps"] = int(applied_metrics["nonzero_steps"])
    decision_row["executed_total_dv_mag_mps"] = float(applied_metrics["total_dv_mag_mps"])
    decision_row["executed_span_steps"] = int(applied_metrics.get("span_steps", -1))
    decision_row["executed_structurally_sane"] = int(applied_metrics.get("structurally_sane", -1))
    decision_row["executed_dominant_ratio"] = float(applied_metrics.get("dominant_ratio", float("nan")))

    print(f"[4/6] apply {applied_strategy} schedule to truth")
    apply_plan_to_truth(
        truth_csv,
        plan_csv,
        active_truth_csv,
        strategy=applied_strategy,
        use_parquet=_resolve_use_parquet(),
    )

    print("truth_csv       :", truth_csv)
    print("plan_csv        :", plan_csv)
    print("active_truth_csv:", active_truth_csv)
    print("active_render_dir:", active_render_dir)
    print("compare_csv     :", compare_csv)

    print("[5/6] active render")
    if render_active or not (active_render_dir / "labels.csv").exists():
        render_truth(active_truth_csv, active_render_dir)

    print("[6/6] active estimation chain")
    act_risk_df = run_estimation_chain(active_truth_csv, active_render_dir, "active")

    act_truth_df = read_table(active_truth_csv)
    print("active truth head:")
    print(act_truth_df.loc[:5, ["t_sec", "rc_x_m", "rel_R_m", "observer_maneuver_flag"]])
    print("render_active =", render_active)
    print("active labels exists =", (active_render_dir / "labels.csv").exists())

    summary_rows = [
        summarize_chain(orig_risk_df, "original"),
        summarize_chain(act_risk_df, "active"),
    ]
    write_table(pd.DataFrame(summary_rows), compare_csv)

    full_horizon_rows = [
        summarize_truth_horizon(orig_truth_df, "original"),
        summarize_truth_horizon(act_truth_df, "active"),
    ]
    write_table(pd.DataFrame(full_horizon_rows), full_horizon_csv)

    decision_row.update({
        "action": action,
    })
    decision_row = persist_decision_row(decision_row)

    return PipelineResult(
        run_root=run_root,
        original_render_dir=original_render_dir,
        original_risk_csv=original_render_dir / "risk_prediction_result.csv",
        decision_csv=decision_csv,
        action=action,
        active_truth_csv=active_truth_csv,
        active_render_dir=active_render_dir,
        active_risk_csv=active_render_dir / "risk_prediction_result.csv",
        plan_csv=plan_csv,
        executed_plan_csv=executed_plan_csv,
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monocular estimation, risk assessment, and avoidance pipeline.")
    parser.add_argument("--truth_csv", type=Path, required=True, help="Input truth CSV for one sample.")
    parser.add_argument("--run_name", type=str, default="sample_run", help="Output folder name under outputs/validation_runs.")
    parser.add_argument("--no_render_original", action="store_true", help="Reuse existing original labels if present.")
    parser.add_argument("--no_render_active", action="store_true", help="Reuse existing active labels if present.")
    parser.add_argument("--force_avoid", action="store_true", help="Force avoidance branch regardless of risk decision.")
    parser.add_argument("--apply_even_if_plan_rejected", action="store_true", help="Apply the selected robust/best-effort schedule even when planner marks it unsafe/rejected.")
    parser.add_argument("--use_parquet", action="store_true",
                        help="Enable parquet sidecar IO while preserving CSV outputs for backward compatibility.")
    parser.add_argument("--use_blender_server", action="store_true",
                        help="Try Blender server first for rendering, then fallback to subprocess if unavailable.")
    parser.add_argument("--blender_server_url", type=str, default=BLENDER_SERVER_URL,
                        help="Blender server render endpoint, e.g. http://127.0.0.1:8765/render")
    parser.add_argument("--blender_server_timeout_sec", type=float, default=BLENDER_SERVER_TIMEOUT_SEC,
                        help="HTTP timeout for Blender server rendering requests.")
    return parser


def main() -> None:
    global USE_PARQUET_IO
    global USE_BLENDER_SERVER
    global BLENDER_SERVER_URL
    global BLENDER_SERVER_TIMEOUT_SEC
    args = build_argparser().parse_args()
    USE_PARQUET_IO = bool(args.use_parquet)
    USE_BLENDER_SERVER = bool(args.use_blender_server)
    BLENDER_SERVER_URL = str(args.blender_server_url)
    BLENDER_SERVER_TIMEOUT_SEC = float(args.blender_server_timeout_sec)
    result = run_pipeline(
        truth_csv=args.truth_csv,
        run_name=args.run_name,
        render_original=not args.no_render_original,
        render_active=not args.no_render_active,
        force_avoid=args.force_avoid,
        apply_even_if_plan_rejected=args.apply_even_if_plan_rejected,
    )

    print("\nDone.")
    print("run_root:", result.run_root)
    print("decision_csv:", result.decision_csv)
    print("action:", result.action)
    print("use_parquet:", USE_PARQUET_IO)
    print("use_blender_server:", USE_BLENDER_SERVER)
    print("blender_server_url:", BLENDER_SERVER_URL)
    print("original_risk_csv:", result.original_risk_csv)
    if result.plan_csv is not None:
        print("plan_csv:", result.plan_csv)
    if result.executed_plan_csv is not None:
        print("executed_plan_csv:", result.executed_plan_csv)
    if result.active_risk_csv is not None:
        print("active_risk_csv:", result.active_risk_csv)


if __name__ == "__main__":
    main()
