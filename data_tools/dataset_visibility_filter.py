from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class VisibilityCriteria:
    # Cheap truth-domain precheck to avoid expensive Blender calls on hopeless samples.
    min_est_visible_frames: int = 8
    min_est_longest_streak: int = 6
    est_margin_ratio: float = 0.90

    # Final image-domain acceptance gate.
    min_visible_frames: int = 12
    min_longest_visible_streak: int = 8
    max_first_visible_frame: int = 120
    min_mean_bbox_area_px: float = 36.0
    min_inside_safe_ratio_visible: float = 0.35
    max_edge_touch_ratio_visible: float = 0.70

    # Rendering parameters should stay aligned with the main pipeline.
    fps: int = 5
    model_scale: float = 50.0
    emission_strength: float = 8.0


@dataclass
class VisibilityScreenResult:
    accepted: bool
    stage: str
    reason: str
    metrics: dict[str, Any]


def _longest_streak(mask: pd.Series) -> int:
    best = 0
    cur = 0
    for v in mask.astype(bool).tolist():
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _safe_box_mask(cx: pd.Series, cy: pd.Series, img_w: int, img_h: int, safe_ratio: float) -> pd.Series:
    mx = img_w * (1.0 - safe_ratio) / 2.0
    my = img_h * (1.0 - safe_ratio) / 2.0
    return (cx >= mx) & (cx <= img_w - mx) & (cy >= my) & (cy <= img_h - my)


def estimate_visibility_from_truth(
    truth_df: pd.DataFrame,
    fov_x_deg: float,
    criteria: VisibilityCriteria,
) -> dict[str, Any]:
    if truth_df.empty:
        return {
            "est_visible_frames": 0,
            "est_longest_visible_streak": 0,
            "est_first_visible_frame": -1,
            "est_visible_ratio": 0.0,
        }

    t_forward = pd.to_numeric(truth_df["rel_T_m"], errors="coerce").fillna(-1.0)
    r = pd.to_numeric(truth_df["rel_R_m"], errors="coerce").fillna(0.0).abs()
    n = pd.to_numeric(truth_df["rel_N_m"], errors="coerce").fillna(0.0).abs()

    half_span = math.tan(math.radians(fov_x_deg) * 0.5) * t_forward * float(criteria.est_margin_ratio)
    visible_est = (t_forward > 1.0) & (r <= half_span) & (n <= half_span)

    visible_count = int(visible_est.sum())
    longest = _longest_streak(visible_est)
    first_idx = truth_df.index[visible_est].tolist()
    first_frame = int(first_idx[0] + 1) if first_idx else -1

    return {
        "est_visible_frames": visible_count,
        "est_longest_visible_streak": longest,
        "est_first_visible_frame": first_frame,
        "est_visible_ratio": float(visible_count / max(len(truth_df), 1)),
    }


def render_truth_labels(
    truth_csv: Path,
    render_dir: Path,
    *,
    blender_exe: str,
    blender_script: str,
    target_model: str,
    img_w: int,
    img_h: int,
    fov_x_deg: float,
    criteria: VisibilityCriteria,
) -> Path:
    render_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        blender_exe,
        "-b",
        "-P", blender_script,
        "--",
        "--csv", str(truth_csv),
        "--obj", str(target_model),
        "--out", str(render_dir),
        "--resx", str(img_w),
        "--resy", str(img_h),
        "--fps", str(criteria.fps),
        "--fov_deg", str(fov_x_deg),
        "--scale", str(criteria.model_scale),
        "--emission", str(criteria.emission_strength),
    ]
    subprocess.run(cmd, check=True)
    label_csv = render_dir / "labels.csv"
    if not label_csv.exists():
        raise FileNotFoundError(f"Blender finished but labels.csv is missing: {label_csv}")
    return label_csv


def summarize_labels_csv(
    labels_csv: Path,
    *,
    img_w: int,
    img_h: int,
    safe_ratio: float,
) -> dict[str, Any]:
    labels = pd.read_csv(labels_csv)
    if labels.empty:
        return {
            "visible_frames": 0,
            "longest_visible_streak": 0,
            "first_visible_frame": -1,
            "visible_ratio": 0.0,
            "mean_bbox_area_visible": 0.0,
            "median_bbox_area_visible": 0.0,
            "inside_safe_ratio_visible": 0.0,
            "edge_touch_ratio_visible": 1.0,
        }

    visible_mask = labels["visible"].fillna(0).astype(int) == 1
    longest = _longest_streak(visible_mask)
    visible = labels.loc[visible_mask].copy()
    visible_count = int(len(visible))
    first_frame = int(visible["frame"].iloc[0]) if visible_count > 0 else -1

    if visible_count == 0:
        return {
            "visible_frames": 0,
            "longest_visible_streak": longest,
            "first_visible_frame": -1,
            "visible_ratio": 0.0,
            "mean_bbox_area_visible": 0.0,
            "median_bbox_area_visible": 0.0,
            "inside_safe_ratio_visible": 0.0,
            "edge_touch_ratio_visible": 1.0,
        }

    visible["bbox_area"] = visible["w"].astype(float) * visible["h"].astype(float)
    inside_safe = _safe_box_mask(visible["cx"].astype(float), visible["cy"].astype(float), img_w, img_h, safe_ratio)
    edge_touch = (
        (visible["xmin"].astype(float) <= 0.0)
        | (visible["ymin"].astype(float) <= 0.0)
        | (visible["xmax"].astype(float) >= float(img_w))
        | (visible["ymax"].astype(float) >= float(img_h))
    )

    return {
        "visible_frames": visible_count,
        "longest_visible_streak": longest,
        "first_visible_frame": first_frame,
        "visible_ratio": float(visible_count / max(len(labels), 1)),
        "mean_bbox_area_visible": float(visible["bbox_area"].mean()),
        "median_bbox_area_visible": float(visible["bbox_area"].median()),
        "inside_safe_ratio_visible": float(inside_safe.mean()),
        "edge_touch_ratio_visible": float(edge_touch.mean()),
    }


def check_visibility_metrics(metrics: dict[str, Any], criteria: VisibilityCriteria) -> tuple[bool, str]:
    if int(metrics.get("visible_frames", 0)) < int(criteria.min_visible_frames):
        return False, "visible_frames_below_threshold"
    if int(metrics.get("longest_visible_streak", 0)) < int(criteria.min_longest_visible_streak):
        return False, "visible_streak_below_threshold"
    first_visible = int(metrics.get("first_visible_frame", -1))
    if first_visible < 0 or first_visible > int(criteria.max_first_visible_frame):
        return False, "first_visible_frame_too_late"
    if float(metrics.get("mean_bbox_area_visible", 0.0)) < float(criteria.min_mean_bbox_area_px):
        return False, "bbox_too_small"
    if float(metrics.get("inside_safe_ratio_visible", 0.0)) < float(criteria.min_inside_safe_ratio_visible):
        return False, "safe_box_ratio_too_low"
    if float(metrics.get("edge_touch_ratio_visible", 1.0)) > float(criteria.max_edge_touch_ratio_visible):
        return False, "edge_touch_ratio_too_high"
    return True, "accepted"


def screen_truth_dataframe_for_monocular_use(
    truth_df: pd.DataFrame,
    *,
    truth_csv: Path,
    config_module: Any,
    criteria: VisibilityCriteria,
) -> VisibilityScreenResult:
    est = estimate_visibility_from_truth(truth_df, fov_x_deg=float(config_module.FOV_X_DEG), criteria=criteria)
    if int(est["est_visible_frames"]) < int(criteria.min_est_visible_frames):
        return VisibilityScreenResult(False, "analytic_precheck", "est_visible_frames_below_threshold", est)
    if int(est["est_longest_visible_streak"]) < int(criteria.min_est_longest_streak):
        return VisibilityScreenResult(False, "analytic_precheck", "est_visible_streak_below_threshold", est)

    tmp_root = Path(getattr(config_module, "OUTPUTS_DIR", truth_csv.parent)) / "_dataset_screen_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"screen_{truth_csv.stem}_", dir=str(tmp_root)) as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        label_csv = render_truth_labels(
            truth_csv,
            tmp_dir_path,
            blender_exe=str(config_module.BLENDER_EXE),
            blender_script=str(config_module.BLENDER_SCRIPT),
            target_model=str(config_module.TARGET_MODEL),
            img_w=int(config_module.IMG_W),
            img_h=int(config_module.IMG_H),
            fov_x_deg=float(config_module.FOV_X_DEG),
            criteria=criteria,
        )
        metrics = summarize_labels_csv(
            label_csv,
            img_w=int(config_module.IMG_W),
            img_h=int(config_module.IMG_H),
            safe_ratio=float(getattr(config_module, "SAFE_RATIO", 0.80)),
        )
        metrics.update(est)
        accepted, reason = check_visibility_metrics(metrics, criteria)
        return VisibilityScreenResult(accepted, "render_screen", reason, metrics)


def cleanup_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except OSError:
            pass


def criteria_to_meta(criteria: VisibilityCriteria) -> dict[str, Any]:
    return {f"screen_{k}": v for k, v in asdict(criteria).items()}
