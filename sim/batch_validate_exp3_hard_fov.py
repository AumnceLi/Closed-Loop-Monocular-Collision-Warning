
from __future__ import annotations

"""
Batch validation for Experiment III.

This version is designed to be run directly as a script, e.g.

    python sim/batch_validate_exp3_hard_fov.py --dataset_kind rtn --split val --subsets no_maneuver

It supports:
- dataset_rtn and dataset_risky
- automatic sample discovery from dataset folders
- batch summary exports suitable for paper tables
- failed-case logging
"""

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Stable imports regardless of launch path
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
for extra in [PROJECT_ROOT, PROJECT_ROOT / "pipeline", PROJECT_ROOT / "sim"]:
    s = str(extra)
    if s not in sys.path:
        sys.path.insert(0, s)

import config  # noqa: E402
from build_pseudo_los import build_pseudo_los_df  # noqa: E402
from imm_ukf_estimator import imm_run  # noqa: E402
from risk_prediction_from_ukf import run_risk_prediction  # noqa: E402
from active_sensing_plan_one_sample import plan_active_sensing  # noqa: E402
from pipeline.execution_policy import (  # noqa: E402
    decide_execution_veto as shared_decide_execution_veto,
    decide_physical_safe_veto as shared_decide_physical_safe_veto,
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


# -----------------------------------------------------------------------------
# Globals from config
# -----------------------------------------------------------------------------
BLENDER_EXE = config.BLENDER_EXE
BLENDER_SCRIPT = str(config.BLENDER_SCRIPT)
TARGET_MODEL = str(config.TARGET_MODEL)

FOV_X_DEG = config.FOV_X_DEG
IMG_W = config.IMG_W
IMG_H = config.IMG_H
CODE_VERSION = "batch_v6_nominal_hardening_20260409"


class InsufficientVisibleFramesError(RuntimeError):
    """Raised when rendered observations are insufficient for IMM-UKF."""


USE_PARQUET_IO = False
USE_BLENDER_SERVER = False
BLENDER_SERVER_URL = str(getattr(config, "BLENDER_SERVER_URL", "http://127.0.0.1:8765/render"))
BLENDER_SERVER_TIMEOUT_SEC = float(getattr(config, "BLENDER_SERVER_TIMEOUT_SEC", 120.0))


def _lazy_apply_plan_to_truth(*args, **kwargs):
    from apply_active_micro_dv_to_truth import apply_plan_to_truth  # noqa: E402
    return apply_plan_to_truth(*args, **kwargs)


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
        "obj": str(TARGET_MODEL),
        "out": str(out_dir),
        "resx": int(IMG_W),
        "resy": int(IMG_H),
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

# -----------------------------------------------------------------------------
# CLI / dataset discovery
# -----------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Batch validation for monocular avoidance experiments.")
    ap.add_argument("--dataset_kind", choices=["rtn", "risky"], default="rtn",
                    help="Select dataset root: dataset_rtn or dataset_risky.")
    ap.add_argument("--split", default="val", help="Dataset split, e.g. train/val/test.")
    ap.add_argument("--subsets", default="no_maneuver",
                    help="Comma-separated subset names under the split. "
                         "Examples: no_maneuver or safe,caution,danger")
    ap.add_argument("--max_samples", type=int, default=0,
                    help="Optional limit on number of samples. 0 means all.")
    ap.add_argument("--sample_ids", type=str, default="",
                    help="Optional comma-separated sample ids to run only, "
                         "e.g. 00167,00172,00174,00175 or sample_00167,sample_00172")
    ap.add_argument("--run_name", type=str, default="batch_exp3_validation",
                    help="Output folder name under outputs/validation_runs.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip a case if its before_after_summary.csv already exists.")
    ap.add_argument("--continue_on_error", action="store_true",
                    help="Continue batch run even if a case fails.")
    ap.add_argument("--min_visible_frames", type=int, default=10,
                    help="Minimum number of visible frames required before IMM-UKF is attempted.")
    ap.add_argument("--apply_even_if_plan_rejected", action="store_true",
                    help="Force apply plan to truth even if planner marks it unsafe/rejected.")
    ap.add_argument("--use_parquet", action="store_true",
                    help="Enable parquet sidecar IO while preserving CSV outputs for backward compatibility.")
    ap.add_argument("--use_blender_server", action="store_true",
                    help="Try Blender server first for rendering, then fallback to subprocess if unavailable.")
    ap.add_argument("--blender_server_url", type=str, default=BLENDER_SERVER_URL,
                    help="Blender server render endpoint, e.g. http://127.0.0.1:8765/render")
    ap.add_argument("--blender_server_timeout_sec", type=float, default=BLENDER_SERVER_TIMEOUT_SEC,
                    help="HTTP timeout for Blender server rendering requests.")
    return ap


def get_dataset_root(dataset_kind: str) -> Path:
    if dataset_kind == "rtn":
        return Path(config.DATASET_RTN_DIR)
    if dataset_kind == "risky":
        return Path(config.DATASET_RISKY_DIR)
    raise ValueError(f"Unknown dataset_kind: {dataset_kind}")


def discover_sample_csvs(dataset_kind: str, split: str, subsets: list[str], max_samples: int = 0) -> list[Path]:
    root = get_dataset_root(dataset_kind)
    sample_paths: list[Path] = []

    for subset in subsets:
        subset_dir = root / split / subset
        if not subset_dir.exists():
            print(f"[WARN] subset dir does not exist, skipping: {subset_dir}")
            continue
        sample_paths.extend(sorted(subset_dir.glob("sample_*.csv")))

    if not sample_paths:
        raise FileNotFoundError(
            f"No sample CSVs found under dataset_kind={dataset_kind}, split={split}, subsets={subsets}"
        )

    if max_samples > 0:
        sample_paths = sample_paths[:max_samples]
    return sample_paths


def _normalize_sample_id(sample_id: str) -> str:
    s = str(sample_id).strip()
    if not s:
        return ""
    if s.lower().endswith(".csv"):
        s = s[:-4]
    if not s.lower().startswith("sample_"):
        s = f"sample_{s}"
    return s


def filter_sample_csvs_by_ids(sample_paths: list[Path], sample_ids: list[str]) -> list[Path]:
    wanted = {_normalize_sample_id(s) for s in sample_ids if str(s).strip()}
    wanted.discard("")
    if not wanted:
        return sample_paths

    filtered = [p for p in sample_paths if p.stem in wanted]
    missing = sorted(wanted - {p.stem for p in filtered})
    if missing:
        print(f"[WARN] requested sample ids not found in discovered set: {missing}")
    if not filtered:
        raise FileNotFoundError(f"No discovered sample CSVs matched sample_ids={sorted(wanted)}")
    return filtered


# -----------------------------------------------------------------------------
# Rendering / chain
# -----------------------------------------------------------------------------
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
        BLENDER_EXE,
        "-b",
        "-P", BLENDER_SCRIPT,
        "--",
        "--csv", str(truth_csv),
        "--obj", str(TARGET_MODEL),
        "--out", str(out_dir),
        "--resx", str(IMG_W),
        "--resy", str(IMG_H),
        "--fps", str(config.RENDER_FPS),
        "--fov_deg", str(config.RENDER_FOV_DEG),
        "--scale", str(config.RENDER_SCALE),
        "--emission", str(config.RENDER_EMISSION),
    ]
    print("Running Blender:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def merge_truth_and_labels(truth_csv: Path, render_dir: Path) -> pd.DataFrame:
    truth_df = read_table(truth_csv)
    label_csv = render_dir / "labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing labels.csv: {label_csv}")
    label_df = pd.read_csv(label_csv)

    label_df["truth_idx"] = label_df["frame"] - 1
    truth_df = truth_df.reset_index().rename(columns={"index": "truth_idx"})
    merged = pd.merge(label_df, truth_df, on="truth_idx", how="inner")
    merged["cx_norm"] = merged["cx"] / IMG_W
    merged["cy_norm"] = merged["cy"] / IMG_H
    merged["w_norm"] = merged["w"] / IMG_W
    merged["h_norm"] = merged["h"] / IMG_H
    merged["bbox_area"] = merged["w"] * merged["h"]
    return merged


def run_chain(truth_csv: Path, render_dir: Path, tag: str, min_visible_frames: int = 10) -> pd.DataFrame:
    merged_csv = render_dir / "merged_observation_truth.csv"
    los_csv = render_dir / "merged_with_pseudo_los.csv"
    imm_csv = render_dir / "imm_ukf_estimation_result.csv"
    risk_csv = render_dir / "risk_prediction_result.csv"

    merged_df = merge_truth_and_labels(truth_csv, render_dir)
    write_table(merged_df, merged_csv)

    los_df = build_pseudo_los_df(merged_df, fov_x_deg=FOV_X_DEG, img_w=IMG_W, img_h=IMG_H)
    write_table(los_df, los_csv)

    visible_count = int(los_df["visible"].sum()) if "visible" in los_df.columns else int(len(los_df))
    if visible_count < int(min_visible_frames):
        raise InsufficientVisibleFramesError(
            f"visible frames = {visible_count} < min_visible_frames = {int(min_visible_frames)}"
        )

    imm_df = imm_run(los_df, imm_csv)
    write_parquet_sidecar(imm_df, imm_csv)
    risk_df = run_risk_prediction(imm_df, risk_csv)
    risk_df["tag"] = tag
    write_table(risk_df, risk_csv)
    return risk_df


# -----------------------------------------------------------------------------
# Metrics / summaries
# -----------------------------------------------------------------------------
def summarize_df(df: pd.DataFrame, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tag": tag,
        "summary_scope": "visible_window_estimation_chain",
        "num_visible": int(len(df)),
    }

    for col in ["sigma_R_m", "sigma_T_m", "sigma_N_m", "trace_P_pos", "pc_bplane"]:
        if col in df.columns:
            out[f"mean_{col}"] = float(df[col].mean())
            out[f"max_{col}"] = float(df[col].max())

    if "risk_level" in df.columns and len(df) > 0:
        out["last_risk_level"] = str(df["risk_level"].iloc[-1])

    if "future_min_distance_m" in df.columns:
        out["min_future_min_distance_m"] = float(df["future_min_distance_m"].min())
    elif "pred_future_min_distance_m" in df.columns:
        out["min_future_min_distance_m"] = float(df["pred_future_min_distance_m"].min())

    return out


def summarize_truth_horizon(df: pd.DataFrame, tag: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "tag": tag,
        "summary_scope": "full_horizon_truth",
        "num_truth_steps": int(len(df)),
    }

    if len(df) == 0:
        return out

    if "relative_distance_m" in df.columns:
        idx_rel = int(df["relative_distance_m"].astype(float).idxmin())
        out["min_relative_distance_m"] = float(df.loc[idx_rel, "relative_distance_m"])
        if "t_sec" in df.columns:
            out["min_relative_distance_t_sec"] = float(df.loc[idx_rel, "t_sec"])

    if "future_min_distance_m" in df.columns:
        idx_fut = int(df["future_min_distance_m"].astype(float).idxmin())
        out["min_future_min_distance_m"] = float(df.loc[idx_fut, "future_min_distance_m"])
        if "t_sec" in df.columns:
            out["min_future_min_distance_t_sec"] = float(df.loc[idx_fut, "t_sec"])

    if "risk_level" in df.columns:
        risk_series = df["risk_level"].astype(str)
        out["last_risk_level"] = str(risk_series.iloc[-1])
        out["num_safe_steps"] = int((risk_series.str.lower() == "safe").sum())
        out["num_caution_steps"] = int((risk_series.str.lower() == "caution").sum())
        out["num_danger_steps"] = int((risk_series.str.lower() == "danger").sum())
        out["worst_risk_rank"] = int(max((safe_rank(v) for v in risk_series.tolist()), default=-999))

    if "observer_maneuver_flag" in df.columns:
        out["num_maneuver_steps"] = int((df["observer_maneuver_flag"].astype(float) > 0.5).sum())

    return out


def preview_execution_geometry(
    truth_csv: Path,
    plan_csv: Path,
    preview_truth_csv: Path,
    *,
    strategy: str,
) -> tuple[dict[str, Any], str]:
    try:
        preview_df = _lazy_apply_plan_to_truth(
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
            out[f"mean_abs_e_{axis}"] = float(np.mean(np.abs(e)))
            out[f"rmse_{axis}"] = float(np.sqrt(np.mean(e ** 2)))
            if sigma_col in df_est.columns:
                sigma = df_est[sigma_col].to_numpy(dtype=float)
                out[f"cover_3sigma_{axis}"] = float(np.mean(np.abs(e) <= 3.0 * sigma))

    needed = {"R_est_m", "T_est_m", "N_est_m", "rel_R_m", "rel_T_m", "rel_N_m"}
    if needed.issubset(set(df_est.columns)):
        epos = np.stack([
            df_est["R_est_m"].to_numpy(dtype=float) - df_est["rel_R_m"].to_numpy(dtype=float),
            df_est["T_est_m"].to_numpy(dtype=float) - df_est["rel_T_m"].to_numpy(dtype=float),
            df_est["N_est_m"].to_numpy(dtype=float) - df_est["rel_N_m"].to_numpy(dtype=float),
        ], axis=1)
        pos_norm = np.linalg.norm(epos, axis=1)
        out["mean_abs_pos_err_m"] = float(np.mean(pos_norm))
        out["rmse_pos_m"] = float(np.sqrt(np.mean(pos_norm ** 2)))

    return out


def safe_rank(label: Any) -> int:
    s = str(label).lower()
    if s == "safe":
        return 0
    if s == "caution":
        return 1
    if s == "danger":
        return 2
    return -999


POST_ACTION_FAIL_TRACE_RATIO_THR = 1.10
POST_ACTION_FAIL_RMSE_RATIO_THR = 1.10


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


def infer_dataset_subset_label(truth_csv: Path) -> str:
    label = str(truth_csv.parent.name).strip().lower()
    if label in {"safe", "caution", "danger"}:
        return label
    return ""


def extract_strategy_plan_metrics(plan_df: pd.DataFrame, strategy: str) -> dict[str, Any]:
    return shared_extract_strategy_plan_metrics(plan_df, strategy)




def reconcile_selected_plan_meta(plan_meta: dict[str, Any], selected_metrics: dict[str, Any]) -> dict[str, Any]:
    reconciled = shared_reconcile_selected_plan_meta(plan_meta, selected_metrics)
    out = dict(reconciled)
    strategy = str(out.get("final_selected_strategy", out.get("selected_strategy", "robust"))).strip().lower()
    if strategy not in {"robust", "nominal", "zero"}:
        strategy = "robust"
    out["final_selected_strategy"] = strategy
    out["selected_strategy"] = strategy
    out["planner_selection_status"] = f"selected_{strategy}"
    # Bind the selected snapshot fields to the final strategy so downstream CSVs do not
    # mix corrected final selection with stale pre-correction acceptance values.
    for field in [
        "pred_min_safe_margin_px",
        "pred_terminal_margin_px",
        "pred_violation_count",
        "pred_negative_margin_sum_px",
        "pred_min_relative_distance_m",
        "pred_terminal_relative_distance_m",
        "pred_min_future_min_distance_m",
        "pred_terminal_future_min_distance_m",
        "pred_worst_risk_rank",
        "strict_acceptance_passed",
        "acceptance_gate_safe",
        "acceptance_gate_structural",
        "acceptance_gate_information",
        "acceptance_gate_geometry_non_degenerate",
        "acceptance_gate_future_distance",
        "trace_ratio_vs_zero",
        "terminal_trace_ratio_vs_zero",
        "info_delta_vs_zero",
        "margin_gain_vs_zero_px",
        "terminal_margin_gain_vs_zero_px",
        "violation_delta_vs_zero",
        "negative_margin_sum_delta_vs_zero_px",
        "future_distance_gain_vs_zero_m",
        "future_distance_non_regression",
        "nonzero_steps",
        "total_dv_mag_mps",
        "dominant_ratio",
        "span_steps",
        "structurally_sane",
        "plan_acceptance_reason",
        "execution_selection_reason",
    ]:
        key = f"{strategy}_{field}"
        if key in out:
            out[f"selected_{field}"] = out[key]
    out["selected_snapshot_bound_strategy"] = strategy
    return out


def build_case_decision_row_template(
    sample_stem: str,
    truth_csv: Path,
    dataset_subset_label: str,
    *,
    apply_even_if_plan_rejected: bool,
) -> dict[str, Any]:
    return {
        "sample": str(sample_stem),
        "truth_csv": str(truth_csv),
        "dataset_subset_label": str(dataset_subset_label),
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
        "plan_accepted": 0,
        "plan_acceptance_reason": "",
        "selected_strategy": "",
        "case_label": "",
        "total_dv_budget_mps": float("nan"),
        "budget_retry_used": 0,
        "retry_from_case_label": "",
        "max_retry_budget_reached": 0,
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
        "selected_nonzero_steps": -1,
        "selected_total_dv_mag_mps": float("nan"),
        "selected_span_steps": -1,
        "selected_structurally_sane": -1,
        "selected_future_distance_gain_vs_zero_m": float("nan"),
        "num_plan_steps": 0,
        "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
        "forced_apply_rejected_plan": 0,
        "execution_vetoed": 0,
        "execution_veto_reason": "",
        "execution_warning_flags": "",
        "execution_warning_count": 0,
        "execution_decision_reason": "",
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
        "executed_span_steps": -1,
        "executed_structurally_sane": -1,
        "executed_dominant_ratio": float("nan"),
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


def _is_finite_number(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _validate_decision_row_integrity(row: dict[str, Any]) -> None:
    sample = str(row.get("sample", ""))

    if _as_int(row.get("plan_accepted", 0), 0) == 1:
        missing_selected = [
            key
            for key in [
                "selected_pred_min_safe_margin_px",
                "selected_pred_terminal_margin_px",
                "selected_pred_negative_margin_sum_px",
                "selected_pred_min_relative_distance_m",
                "selected_pred_terminal_relative_distance_m",
            ]
            if not _is_finite_number(row.get(key, np.nan))
        ]
        if _as_int(row.get("selected_pred_violation_count", -1), -1) < 0:
            missing_selected.append("selected_pred_violation_count")
        if missing_selected:
            raise RuntimeError(
                f"Decision row integrity failed for accepted plan ({sample}): missing selected_pred_* fields: {missing_selected}"
            )

    if _as_int(row.get("execution_applied_flag", 0), 0) == 1:
        missing_executed = [
            key
            for key in [
                "executed_pred_min_safe_margin_px",
                "executed_pred_terminal_margin_px",
                "executed_pred_negative_margin_sum_px",
                "executed_pred_min_relative_distance_m",
                "executed_pred_terminal_relative_distance_m",
            ]
            if not _is_finite_number(row.get(key, np.nan))
        ]
        if _as_int(row.get("executed_pred_violation_count", -1), -1) < 0:
            missing_executed.append("executed_pred_violation_count")
        if missing_executed:
            raise RuntimeError(
                f"Decision row integrity failed for executed plan ({sample}): missing executed_pred_* fields: {missing_executed}"
            )

    planning_attempted = (
        _as_int(row.get("trigger_frame", -1), -1) >= 0
        and (not str(row.get("plan_acceptance_reason", "")).startswith("planning_skipped"))
    )
    if planning_attempted:
        missing_zero = [
            key
            for key in [
                "zero_pred_min_safe_margin_px",
                "zero_pred_terminal_margin_px",
                "zero_pred_negative_margin_sum_px",
                "zero_pred_min_relative_distance_m",
                "zero_pred_terminal_relative_distance_m",
                "zero_pred_min_future_min_distance_m",
            ]
            if not _is_finite_number(row.get(key, np.nan))
        ]
        if missing_zero:
            raise RuntimeError(
                f"Decision row integrity failed after planning ({sample}): missing zero_pred_* fields: {missing_zero}"
            )


def decide_physical_safe_veto(
    orig_summary: dict[str, Any],
    orig_err: dict[str, Any] | None = None,
    *,
    dataset_subset_label: str = "",
) -> tuple[bool, str]:
    return shared_decide_physical_safe_veto(
        orig_summary,
        orig_err,
        dataset_subset_label=dataset_subset_label,
    )


def detect_estimator_divergence(orig_summary: dict[str, Any], orig_err: dict[str, Any] | None = None) -> tuple[bool, str]:
    return shared_detect_estimator_divergence(orig_summary, orig_err)


def build_failed_case_row(row: dict[str, Any]) -> dict[str, Any] | None:
    action = str(row.get("action", ""))
    trigger_reason = str(row.get("trigger_reason", ""))
    accept_reason = str(row.get("plan_acceptance_reason", ""))
    exec_reason = str(row.get("execution_decision_reason", "") or row.get("execution_veto_reason", ""))
    failure_family = str(row.get("failure_family", "")).strip()
    try:
        planner_preview_consistency_flag = _as_int(row.get("planner_preview_consistency_flag", -1), -1)
    except Exception:
        planner_preview_consistency_flag = -1

    if trigger_reason == "physical_safe_skip_planning" or exec_reason == "physical_safe_veto":
        return None

    failure_reason = ""
    failure_stage = ""

    if failure_family == "planner_accepted_but_preview_vetoed":
        failure_reason = exec_reason or failure_family
        failure_stage = "execution_gate"
    elif failure_family == "planner_accepted_but_execution_vetoed":
        failure_reason = exec_reason or failure_family
        failure_stage = "execution_gate"
    elif failure_family == "planner_rejected_not_executed":
        # Use the refined rejected-case attribution from execution_decision_reason
        # instead of collapsing everything to the coarse family label.
        failure_reason = exec_reason or accept_reason or failure_family
        failure_stage = "planning"
    elif failure_family == "planner_rejected_but_forced_executed":
        failure_reason = exec_reason or accept_reason or failure_family
        failure_stage = "planning"

    if action in {"track_only", "audit_apply_rejected_plan"}:
        if not failure_reason:
            return None
    elif action == "avoid":
        orig_min_future = _as_float(
            row.get("orig_full_min_future_min_distance_m", row.get("orig_min_future_min_distance_m", float("nan")))
        )
        act_min_future = _as_float(
            row.get("act_full_min_future_min_distance_m", row.get("act_min_future_min_distance_m", float("nan")))
        )
        orig_trace = _as_float(row.get("orig_mean_trace_P_pos", float("nan")))
        act_trace = _as_float(row.get("act_mean_trace_P_pos", float("nan")))
        orig_rmse_pos = _as_float(row.get("orig_rmse_pos_m", float("nan")))
        act_rmse_pos = _as_float(row.get("act_rmse_pos_m", float("nan")))

        geometry_regression = (
            np.isfinite(orig_min_future) and np.isfinite(act_min_future) and act_min_future + 1.0e-9 < orig_min_future
        )
        estimation_regression = (
            np.isfinite(orig_trace) and np.isfinite(act_trace) and act_trace > max(orig_trace * POST_ACTION_FAIL_TRACE_RATIO_THR, orig_trace + 1.0)
            and np.isfinite(orig_rmse_pos) and np.isfinite(act_rmse_pos) and act_rmse_pos > max(orig_rmse_pos * POST_ACTION_FAIL_RMSE_RATIO_THR, orig_rmse_pos + 1.0)
        )

        if geometry_regression and estimation_regression:
            failure_reason = "post_action_geometry_and_estimation_regression"
            failure_stage = "post_action_validation"
        elif geometry_regression:
            failure_reason = "post_action_geometry_regression"
            failure_stage = "post_action_validation"
        elif estimation_regression:
            failure_reason = "post_action_estimation_regression"
            failure_stage = "post_action_validation"

    if not failure_reason:
        return None

    return {
        "sample": str(row.get("sample", "")),
        "truth_csv": str(row.get("truth_csv", "")),
        "dataset_subset_label": str(row.get("dataset_subset_label", "")),
        "action": action,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "failure_family": failure_family,
        "planner_preview_consistency_flag": _as_int(planner_preview_consistency_flag, -1),
        "trigger_reason": trigger_reason,
        "plan_acceptance_reason": accept_reason,
        "case_label": str(row.get("case_label", "")),
        "execution_veto_reason": str(row.get("execution_veto_reason", "")),
        "execution_decision_reason": str(row.get("execution_decision_reason", "")),
        "pred_min_safe_margin_px": _as_float(
            row.get("selected_pred_min_safe_margin_px", row.get("robust_pred_min_safe_margin_px", float("nan")))
        ),
        "pred_violation_count": _as_int(
            row.get("selected_pred_violation_count", row.get("robust_pred_violation_count", -1)), -1
        ),
        "orig_min_future_min_distance_m": _as_float(row.get("orig_min_future_min_distance_m", float("nan"))),
        "act_min_future_min_distance_m": _as_float(row.get("act_min_future_min_distance_m", float("nan"))),
        "orig_rmse_pos_m": _as_float(row.get("orig_rmse_pos_m", float("nan"))),
        "act_rmse_pos_m": _as_float(row.get("act_rmse_pos_m", float("nan"))),
        "orig_mean_trace_P_pos": _as_float(row.get("orig_mean_trace_P_pos", float("nan"))),
        "act_mean_trace_P_pos": _as_float(row.get("act_mean_trace_P_pos", float("nan"))),
    }


def persist_failed_case_summary(case_root: Path, failed_row: dict[str, Any] | None) -> None:
    failed_case_csv = case_root / "failed_case_summary.csv"
    if failed_row is None:
        if failed_case_csv.exists():
            failed_case_csv.unlink()
        return
    write_table(pd.DataFrame([failed_row]), failed_case_csv)


FAILED_CASE_COLUMNS = [
    "sample",
    "truth_csv",
    "dataset_subset_label",
    "action",
    "failure_stage",
    "failure_reason",
    "failure_family",
    "planner_preview_consistency_flag",
    "trigger_reason",
    "plan_acceptance_reason",
    "case_label",
    "execution_veto_reason",
    "execution_decision_reason",
    "pred_min_safe_margin_px",
    "pred_violation_count",
    "orig_min_future_min_distance_m",
    "act_min_future_min_distance_m",
    "orig_rmse_pos_m",
    "act_rmse_pos_m",
    "orig_mean_trace_P_pos",
    "act_mean_trace_P_pos",
    "error_type",
    "error_message",
    "skip_reason",
]


def _finalize_case_return(
    case_root: Path,
    row: dict[str, Any],
    trigger_row: dict[str, Any],
    truth_err_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = dict(row)
    trigger_row = dict(trigger_row)
    truth_err_row = dict(truth_err_row)
    row.setdefault("code_version_tag", CODE_VERSION)
    trigger_row.setdefault("code_version_tag", CODE_VERSION)
    truth_err_row.setdefault("code_version_tag", CODE_VERSION)
    persist_failed_case_summary(case_root, build_failed_case_row(row))
    return row, trigger_row, truth_err_row


def rebuild_failed_rows_from_case_outputs(out_root: Path) -> list[dict[str, Any]]:
    failed_rows: list[dict[str, Any]] = []

    for case_root in sorted(p for p in out_root.iterdir() if p.is_dir()):
        failed_case_csv = case_root / "failed_case_summary.csv"
        case_decision_csv = case_root / "case_decision_summary.csv"

        row_dict: dict[str, Any] | None = None

        if failed_case_csv.exists():
            try:
                df = read_table(failed_case_csv)
                if len(df) > 0:
                    row_dict = {
                        k: (v.item() if hasattr(v, "item") else v)
                        for k, v in df.iloc[0].to_dict().items()
                    }
            except Exception:
                row_dict = None

        if row_dict is None and case_decision_csv.exists():
            try:
                df = read_table(case_decision_csv)
                if len(df) > 0:
                    decision_row = {
                        k: (v.item() if hasattr(v, "item") else v)
                        for k, v in df.iloc[0].to_dict().items()
                    }
                    rebuilt = build_failed_case_row(decision_row)
                    if rebuilt is not None:
                        row_dict = rebuilt
            except Exception:
                row_dict = None

        if row_dict is not None:
            failed_rows.append(row_dict)

    return failed_rows


WARNING_CASE_COLUMNS = [
    "sample",
    "truth_csv",
    "dataset_subset_label",
    "action",
    "execution_warning_flags",
    "execution_warning_count",
    "execution_applied_flag",
    "execution_vetoed",
    "execution_veto_reason",
    "execution_decision_reason",
    "selected_strategy",
    "execution_strategy_requested",
    "execution_strategy_applied",
    "plan_accepted",
    "plan_acceptance_reason",
    "failure_family",
]


def build_warning_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []

    df = pd.DataFrame(rows)
    warning_count = pd.to_numeric(df.get("execution_warning_count", 0), errors="coerce").fillna(0.0)
    execution_applied_flag = pd.to_numeric(df.get("execution_applied_flag", 0), errors="coerce").fillna(0.0)
    plan_accepted = pd.to_numeric(df.get("plan_accepted", 0), errors="coerce").fillna(0.0)
    execution_vetoed = pd.to_numeric(df.get("execution_vetoed", 0), errors="coerce").fillna(0.0)
    warning_flags = (
        df["execution_warning_flags"].astype(str).str.strip()
        if "execution_warning_flags" in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=object)
    )

    warning_present_mask = (warning_count > 0.5) | (warning_flags != "")
    warning_case_mask = warning_present_mask & ((execution_applied_flag > 0.5) | (plan_accepted > 0.5))
    if not bool(warning_case_mask.any()):
        return []

    warning_df = df.loc[warning_case_mask].copy()
    for col in WARNING_CASE_COLUMNS:
        if col not in warning_df.columns:
            warning_df[col] = ""

    # Keep veto marker for diagnostics, but warning-case table focuses on accepted/applied rows.
    warning_df["execution_vetoed"] = execution_vetoed.loc[warning_df.index].astype(int)
    return warning_df[WARNING_CASE_COLUMNS].to_dict(orient="records")


# -----------------------------------------------------------------------------
# Per-case run
# -----------------------------------------------------------------------------
def run_one_case(
    truth_csv: Path,
    case_root: Path,
    *,
    apply_even_if_plan_rejected: bool = False,
    min_visible_frames: int = 10,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sample_stem = truth_csv.stem
    dataset_subset_label = infer_dataset_subset_label(truth_csv)
    orig_render_dir = case_root / "original_render"
    active_truth_csv = case_root / f"{sample_stem}_active.csv"
    active_render_dir = case_root / "active_render"
    plan_csv = case_root / "micro_dv_plan.csv"
    per_case_summary = case_root / "before_after_summary.csv"
    full_horizon_summary_csv = case_root / "full_horizon_truth_summary.csv"
    case_decision_csv = case_root / "case_decision_summary.csv"

    def persist_decision_row(row: dict[str, Any]) -> dict[str, Any]:
        enriched = shared_enrich_execution_decision_row(dict(row))
        _validate_decision_row_integrity(enriched)
        write_table(pd.DataFrame([enriched]), case_decision_csv)
        return enriched

    print("\n" + "=" * 72)
    print("Processing:", sample_stem)
    print("=" * 72)

    print("1) render original")
    render_truth(truth_csv, orig_render_dir)

    print("2) original chain")
    orig_df = run_chain(truth_csv, orig_render_dir, "original", min_visible_frames=min_visible_frames)
    orig_summary = summarize_df(orig_df, "original")
    orig_err = compute_truth_error_metrics(orig_df)
    orig_truth_df = read_table(truth_csv)
    orig_truth_summary = summarize_truth_horizon(orig_truth_df, "original")

    physical_safe_vetoed, physical_safe_reason = decide_physical_safe_veto(
        orig_summary,
        orig_err,
        dataset_subset_label=dataset_subset_label,
    )
    estimator_diverged, estimator_divergence_reason = detect_estimator_divergence(orig_summary, orig_err)

    if estimator_diverged and not apply_even_if_plan_rejected:
        decision_row = {
            "sample": sample_stem,
            "truth_csv": str(truth_csv),
            "dataset_subset_label": dataset_subset_label,
            "trigger_frame": -1,
            "trigger_t_sec": float("nan"),
            "trigger_reason": "estimator_divergence_skip_planning",
            "robust_pred_min_safe_margin_px": float("nan"),
            "robust_pred_violation_count": -1,
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_estimator_divergence",
            "selected_strategy": "",
            "case_label": "estimator_divergence_skip",
            "total_dv_budget_mps": float("nan"),
            "budget_retry_used": 0,
            "retry_from_case_label": "",
            "max_retry_budget_reached": 0,
            "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
            "forced_apply_rejected_plan": 0,
            "execution_vetoed": 1,
            "execution_veto_reason": estimator_divergence_reason,
            "execution_decision_reason": estimator_divergence_reason,
            "execution_strategy_requested": "",
            "execution_strategy_applied": "",
            "action": "track_only",
        }
        decision_row = {
            **decision_row,
        }
        decision_row = shared_build_execution_decision_row(
            build_case_decision_row_template(
                sample_stem,
                truth_csv,
                dataset_subset_label,
                apply_even_if_plan_rejected=apply_even_if_plan_rejected,
            ),
            decision_row,
        )
        decision_row = persist_decision_row(decision_row)
        trigger_row = {
            "sample": sample_stem,
            "first_trigger_frame": -1,
            "first_trigger_t_sec": float("nan"),
            "trigger_reason": "estimator_divergence_skip_planning",
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_estimator_divergence",
            "selected_strategy": "",
            "pred_min_safe_margin_px": float("nan"),
            "pred_violation_count": -1,
        }
        row = {
            "sample": sample_stem,
            "action": "track_only",
            **decision_row,
            **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        truth_err_row = {
            "sample": sample_stem,
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

    if physical_safe_vetoed and not apply_even_if_plan_rejected:
        decision_row = {
            "sample": sample_stem,
            "truth_csv": str(truth_csv),
            "dataset_subset_label": dataset_subset_label,
            "trigger_frame": -1,
            "trigger_t_sec": float("nan"),
            "trigger_reason": "physical_safe_skip_planning",
            "robust_pred_min_safe_margin_px": float("nan"),
            "robust_pred_violation_count": -1,
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_physical_safe_veto",
            "selected_strategy": "",
            "case_label": "physical_safe_skip",
            "total_dv_budget_mps": float("nan"),
            "budget_retry_used": 0,
            "retry_from_case_label": "",
            "max_retry_budget_reached": 0,
            "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
            "forced_apply_rejected_plan": 0,
            "execution_vetoed": 1,
            "execution_veto_reason": physical_safe_reason,
            "execution_decision_reason": physical_safe_reason,
            "execution_strategy_requested": "",
            "execution_strategy_applied": "",
            "action": "track_only",
        }
        decision_row = {
            **decision_row,
        }
        decision_row = shared_build_execution_decision_row(
            build_case_decision_row_template(
                sample_stem,
                truth_csv,
                dataset_subset_label,
                apply_even_if_plan_rejected=apply_even_if_plan_rejected,
            ),
            decision_row,
        )
        decision_row = persist_decision_row(decision_row)
        trigger_row = {
            "sample": sample_stem,
            "first_trigger_frame": -1,
            "first_trigger_t_sec": float("nan"),
            "trigger_reason": "physical_safe_skip_planning",
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_physical_safe_veto",
            "selected_strategy": "",
            "pred_min_safe_margin_px": float("nan"),
            "pred_violation_count": -1,
        }
        row = {
            "sample": sample_stem,
            "action": "track_only",
            **decision_row,
            **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        truth_err_row = {
            "sample": sample_stem,
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

    print("3) planning avoidance schedule")
    plan_res = plan_active_sensing(
        input_csv=orig_render_dir / "merged_with_pseudo_los.csv",
        out_dir=case_root,
        experiment_mode="exp3",
        imm_csv=orig_render_dir / "imm_ukf_estimation_result.csv",
        orig_last_risk_level=str(orig_summary.get("last_risk_level", "")),
        orig_max_pc_bplane=float(orig_summary.get("max_pc_bplane", np.nan)),
        orig_min_future_min_distance_m=float(orig_summary.get("min_future_min_distance_m", np.nan)),
    )
    write_table(plan_res.plan_df, plan_csv)

    preselected_strategy = str(plan_res.plan_meta.get("final_selected_strategy", plan_res.plan_meta.get("selected_strategy", "robust"))).lower()
    if preselected_strategy not in {"robust", "nominal", "zero"}:
        preselected_strategy = "robust"
    preselected_metrics = extract_strategy_plan_metrics(plan_res.plan_df, preselected_strategy)
    plan_res.plan_meta = reconcile_selected_plan_meta(dict(plan_res.plan_meta), preselected_metrics)

    summary_ctx = shared_build_plan_summary_context(
        plan_res.plan_df,
        plan_res.plan_meta,
        orig_truth_summary,
        shared_resolve_execution_strategy,
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
    accept_reason = str(plan_res.plan_meta.get("plan_acceptance_reason", "unknown"))

    decision_row: dict[str, Any] = {
        "sample": sample_stem,
        "truth_csv": str(truth_csv),
        "dataset_subset_label": dataset_subset_label,
        "trigger_frame": int(plan_res.trigger_row.get("frame", -1)),
        "trigger_t_sec": float(plan_res.trigger_row.get("t_sec", -1.0)),
        "trigger_reason": str(plan_res.trigger_reason),
        **dict(summary_ctx["decision_summary_fields"]),
        "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
        "forced_apply_rejected_plan": 0,
        "execution_vetoed": 0,
        "execution_veto_reason": "",
        "execution_decision_reason": "",
        "execution_strategy_requested": str(execution_request.get("strategy", execution_strategy)),
        "execution_request_source_reason": str(execution_request.get("source_reason", execution_strategy_reason)),
        "execution_strategy_reason": str(execution_strategy_reason),
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
        "executed_span_steps": -1,
        "executed_structurally_sane": -1,
        "executed_dominant_ratio": float("nan"),
    }
    decision_row = shared_build_execution_decision_row(
        build_case_decision_row_template(
            sample_stem,
            truth_csv,
            dataset_subset_label,
            apply_even_if_plan_rejected=apply_even_if_plan_rejected,
        ),
        decision_row,
    )
    if "strategy" in plan_res.plan_df.columns:
        selected_rows = plan_res.plan_df[plan_res.plan_df["strategy"].astype(str).str.lower() == selected_strategy]
        decision_row["num_plan_steps"] = int(len(selected_rows))
    else:
        decision_row["num_plan_steps"] = int(len(plan_res.plan_df))

    trigger_row = {
        "sample": sample_stem,
        "first_trigger_frame": int(plan_res.trigger_row.get("frame", -1)),
        "first_trigger_t_sec": float(plan_res.trigger_row.get("t_sec", -1.0)),
        "trigger_reason": str(plan_res.trigger_reason),
        "trigger_safe_dist_px": float(summary_ctx["decision_summary_fields"].get("trigger_safe_dist_px", np.nan)),
        "trigger_safe_dist_rate_pxps": float(summary_ctx["decision_summary_fields"].get("trigger_safe_dist_rate_pxps", np.nan)),
        "trigger_short_time_to_warning_sec": float(summary_ctx["decision_summary_fields"].get("trigger_short_time_to_warning_sec", np.nan)),
        "trigger_short_time_to_breach_sec": float(summary_ctx["decision_summary_fields"].get("trigger_short_time_to_breach_sec", np.nan)),
        "trigger_risk_score": int(summary_ctx["decision_summary_fields"].get("trigger_risk_score", -1)),
        "trigger_warning_score": int(summary_ctx["decision_summary_fields"].get("trigger_warning_score", -1)),
        "trigger_risk_integral": float(summary_ctx["decision_summary_fields"].get("trigger_risk_integral", np.nan)),
        "trigger_warning_integral": float(summary_ctx["decision_summary_fields"].get("trigger_warning_integral", np.nan)),
        "plan_accepted": int(plan_ok),
        "plan_acceptance_reason": accept_reason,
        "selected_strategy": selected_strategy,
        "pred_min_safe_margin_px": float(selected_metrics["pred_min_safe_margin_px"]),
        "pred_violation_count": int(selected_metrics["pred_violation_count"]),
    }

    physical_safe_vetoed, physical_safe_reason = decide_physical_safe_veto(
        orig_summary,
        orig_err,
        dataset_subset_label=dataset_subset_label,
    )
    estimator_diverged, estimator_divergence_reason = detect_estimator_divergence(orig_summary, orig_err)

    if estimator_diverged and not apply_even_if_plan_rejected:
        decision_row = {
            "sample": sample_stem,
            "truth_csv": str(truth_csv),
            "dataset_subset_label": dataset_subset_label,
            "trigger_frame": -1,
            "trigger_t_sec": float("nan"),
            "trigger_reason": "estimator_divergence_skip_planning",
            "robust_pred_min_safe_margin_px": float("nan"),
            "robust_pred_violation_count": -1,
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_estimator_divergence",
            "selected_strategy": "",
            "case_label": "estimator_divergence_skip",
            "total_dv_budget_mps": float("nan"),
            "budget_retry_used": 0,
            "retry_from_case_label": "",
            "max_retry_budget_reached": 0,
            "apply_even_if_plan_rejected_requested": int(bool(apply_even_if_plan_rejected)),
            "forced_apply_rejected_plan": 0,
            "execution_vetoed": 1,
            "execution_veto_reason": estimator_divergence_reason,
            "execution_decision_reason": estimator_divergence_reason,
            "execution_strategy_requested": "",
            "execution_strategy_applied": "",
            "action": "track_only",
        }
        decision_row = {
            **decision_row,
        }
        decision_row = shared_build_execution_decision_row(
            build_case_decision_row_template(
                sample_stem,
                truth_csv,
                dataset_subset_label,
                apply_even_if_plan_rejected=apply_even_if_plan_rejected,
            ),
            decision_row,
        )
        decision_row = persist_decision_row(decision_row)
        trigger_row = {
            "sample": sample_stem,
            "first_trigger_frame": -1,
            "first_trigger_t_sec": float("nan"),
            "trigger_reason": "estimator_divergence_skip_planning",
            "plan_accepted": 0,
            "plan_acceptance_reason": "planning_skipped_estimator_divergence",
            "selected_strategy": "",
            "pred_min_safe_margin_px": float("nan"),
            "pred_violation_count": -1,
        }
        row = {
            "sample": sample_stem,
            "action": "track_only",
            **decision_row,
            **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        truth_err_row = {
            "sample": sample_stem,
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

    if physical_safe_vetoed and not apply_even_if_plan_rejected:
        decision_row["action"] = "track_only"
        decision_row["execution_vetoed"] = 1
        decision_row["execution_veto_reason"] = physical_safe_reason
        decision_row["execution_decision_reason"] = physical_safe_reason
        decision_row["execution_strategy_applied"] = ""
        decision_row = persist_decision_row(decision_row)

        row = {
            "sample": sample_stem,
            "action": "track_only",
            **decision_row,
            **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        truth_err_row = {
            "sample": sample_stem,
            **{f"orig_{k}": v for k, v in orig_err.items()},
        }
        return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

    preview_truth_csv = case_root / f"{sample_stem}_preview_{execution_strategy}.csv"
    planner_rejected = (not plan_ok) or (pred_viol > 0) or (pred_margin <= 0.0)
    if planner_rejected:
        rejection_reason = shared_classify_rejected_plan_veto(plan_ok, pred_margin, pred_viol, decision_row)
        decision_row["geometry_preview_truth_csv"] = str(preview_truth_csv)
        decision_row["geometry_preview_status"] = "skipped_planner_rejected"
        decision_row["geometry_preview_error"] = ""
        decision_row["execution_vetoed"] = 0
        decision_row["execution_veto_reason"] = ""
        decision_row["execution_decision_reason"] = rejection_reason

        if not apply_even_if_plan_rejected:
            decision_row["action"] = "track_only"
            decision_row["execution_strategy_applied"] = ""
            decision_row = persist_decision_row(decision_row)

            row = {
                "sample": sample_stem,
                "action": "track_only",
                **decision_row,
                **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
                **{f"orig_{k}": v for k, v in orig_err.items()},
            }
            truth_err_row = {
                "sample": sample_stem,
                **{f"orig_{k}": v for k, v in orig_err.items()},
            }
            return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

        decision_row["action"] = "audit_apply_rejected_plan"
        decision_row["forced_apply_rejected_plan"] = 1
    else:
        plan_meta_for_execution, decision_row, execution_vetoed, execution_veto_reason = shared_run_execution_preview_and_gate(
            truth_csv=truth_csv,
            plan_csv=plan_csv,
            preview_truth_csv=preview_truth_csv,
            strategy=execution_strategy,
            plan_meta_for_execution=plan_meta_for_execution,
            decision_row=decision_row,
            preview_func=preview_execution_geometry,
            veto_func=shared_decide_execution_veto,
            orig_summary=orig_summary,
            orig_err=orig_err,
            dataset_subset_label=dataset_subset_label,
            plan_ok=plan_ok,
            pred_margin=pred_margin,
            pred_viol=pred_viol,
        )
        if execution_vetoed and not apply_even_if_plan_rejected:
            decision_row["action"] = "track_only"
            decision_row["execution_vetoed"] = 1
            decision_row["execution_veto_reason"] = execution_veto_reason
            decision_row["execution_decision_reason"] = execution_veto_reason
            decision_row["execution_strategy_applied"] = ""
            decision_row = persist_decision_row(decision_row)

            row = {
                "sample": sample_stem,
                "action": "track_only",
                **decision_row,
                **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
                **{f"orig_{k}": v for k, v in orig_err.items()},
            }
            truth_err_row = {
                "sample": sample_stem,
                **{f"orig_{k}": v for k, v in orig_err.items()},
            }
            return _finalize_case_return(case_root, row, trigger_row, truth_err_row)

        decision_row["execution_decision_reason"] = str(execution_strategy_reason or "accepted_plan_execute")

    applied_strategy = str(execution_strategy or decision_row["execution_strategy_requested"] or "robust").lower()
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

    print(f"4) apply {decision_row['execution_strategy_applied']} schedule to truth")
    _lazy_apply_plan_to_truth(
        truth_csv,
        plan_csv,
        active_truth_csv,
        strategy=decision_row["execution_strategy_applied"],
        use_parquet=_resolve_use_parquet(),
    )

    print("5) render active")
    render_truth(active_truth_csv, active_render_dir)

    print("6) active chain")
    act_df = run_chain(active_truth_csv, active_render_dir, "active", min_visible_frames=min_visible_frames)
    act_summary = summarize_df(act_df, "active")
    act_err = compute_truth_error_metrics(act_df)
    act_truth_df = read_table(active_truth_csv)
    act_truth_summary = summarize_truth_horizon(act_truth_df, "active")

    write_table(pd.DataFrame([orig_summary, act_summary]), per_case_summary)
    write_table(pd.DataFrame([orig_truth_summary, act_truth_summary]), full_horizon_summary_csv)

    decision_row["action"] = decision_row.get("action", "avoid")
    decision_row = persist_decision_row(decision_row)

    row = {
        "sample": sample_stem,
        "action": decision_row["action"],
        **decision_row,
        **{f"orig_{k}": v for k, v in orig_summary.items() if k != "tag"},
        **{f"act_{k}": v for k, v in act_summary.items() if k != "tag"},
        **{f"orig_full_{k}": v for k, v in orig_truth_summary.items() if k != "tag"},
        **{f"act_full_{k}": v for k, v in act_truth_summary.items() if k != "tag"},
        **{f"orig_{k}": v for k, v in orig_err.items()},
        **{f"act_{k}": v for k, v in act_err.items()},
    }

    # Derived paper-friendly deltas
    for base in ["mean_trace_P_pos", "max_trace_P_pos", "mean_sigma_T_m", "max_pc_bplane", "mean_pc_bplane"]:
        ok = f"orig_{base}" in row and f"act_{base}" in row
        if ok:
            row[f"delta_{base}"] = float(row[f"act_{base}"] - row[f"orig_{base}"])

    if "orig_last_risk_level" in row and "act_last_risk_level" in row:
        row["risk_rank_change"] = safe_rank(row["act_last_risk_level"]) - safe_rank(row["orig_last_risk_level"])

    truth_err_row = {
        "sample": sample_stem,
        **{f"orig_{k}": v for k, v in orig_err.items()},
        **{f"act_{k}": v for k, v in act_err.items()},
        **{f"orig_full_{k}": v for k, v in orig_truth_summary.items() if k != "tag"},
        **{f"act_full_{k}": v for k, v in act_truth_summary.items() if k != "tag"},
    }

    return _finalize_case_return(case_root, row, trigger_row, truth_err_row)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
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

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    sample_ids = [s.strip() for s in args.sample_ids.split(",") if s.strip()]
    samples = discover_sample_csvs(args.dataset_kind, args.split, subsets, max_samples=args.max_samples)
    samples = filter_sample_csvs_by_ids(samples, sample_ids)

    out_root = Path(config.VALIDATION_RUNS_DIR) / args.run_name
    out_root.mkdir(parents=True, exist_ok=True)

    print("Dataset root:", get_dataset_root(args.dataset_kind))
    print("Split:", args.split)
    print("Subsets:", subsets)
    print("Sample ids filter:", sample_ids if sample_ids else "ALL")
    print("Selected samples:", len(samples))
    print("Use parquet:", USE_PARQUET_IO)
    print("Use blender server:", USE_BLENDER_SERVER)
    print("Blender server url:", BLENDER_SERVER_URL)
    print("Output root:", out_root)

    all_rows: list[dict[str, Any]] = []
    truth_error_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

    for idx, truth_csv in enumerate(samples, start=1):
        sample_stem = truth_csv.stem
        case_root = out_root / sample_stem
        case_root.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and (case_root / "before_after_summary.csv").exists():
            print(f"[{idx}/{len(samples)}] skip existing: {sample_stem}")
            continue

        print(f"\n[{idx}/{len(samples)}] {sample_stem}")
        try:
            row, trigger_row, truth_err_row = run_one_case(
                truth_csv,
                case_root,
                apply_even_if_plan_rejected=args.apply_even_if_plan_rejected,
                min_visible_frames=args.min_visible_frames,
            )
            all_rows.append(row)
            trigger_rows.append(trigger_row)
            truth_error_rows.append(truth_err_row)
            failed_row = build_failed_case_row(row)
            if failed_row is not None:
                failed_rows.append(failed_row)
        except Exception as exc:
            failed_row = {
                "sample": sample_stem,
                "truth_csv": str(truth_csv),
                "dataset_subset_label": infer_dataset_subset_label(truth_csv),
                "action": "runtime_failure",
                "failure_stage": "runtime",
                "failure_reason": type(exc).__name__,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "skip_reason": "insufficient_visibility" if isinstance(exc, InsufficientVisibleFramesError) else "runtime_failure",
            }
            failed_rows.append(failed_row)
            persist_failed_case_summary(case_root, failed_row)
            print(f"[FAILED] {sample_stem}: {type(exc).__name__}: {exc}")
            print(traceback.format_exc())
            if not args.continue_on_error:
                raise

    rebuilt_failed_rows = rebuild_failed_rows_from_case_outputs(out_root)
    final_failed_rows = rebuilt_failed_rows if len(rebuilt_failed_rows) > 0 or len(failed_rows) == 0 else failed_rows
    warning_case_rows = build_warning_case_rows(all_rows)

    write_table(pd.DataFrame(all_rows), out_root / "batch_validation_summary.csv")
    write_table(pd.DataFrame(truth_error_rows), out_root / "truth_error_table.csv")
    write_table(pd.DataFrame(trigger_rows), out_root / "trigger_validation_table.csv")
    write_table(pd.DataFrame(final_failed_rows, columns=FAILED_CASE_COLUMNS), out_root / "failed_cases.csv")
    write_table(pd.DataFrame(warning_case_rows, columns=WARNING_CASE_COLUMNS), out_root / "warning_cases.csv")

    print(f"Failed rows (rebuilt): {len(final_failed_rows)}")
    print(f"Warning-case rows: {len(warning_case_rows)}")

    print("\nSaved:")
    print(out_root / "batch_validation_summary.csv")
    print(out_root / "truth_error_table.csv")
    print(out_root / "trigger_validation_table.csv")
    print(out_root / "failed_cases.csv")
    print(out_root / "warning_cases.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
