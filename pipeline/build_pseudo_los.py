# -*- coding: utf-8 -*-
"""
build_pseudo_los.py

增强版：
1) 对图像坐标进行安全裁剪
2) 输出与 safe-box / 图像边缘相关的可诊断列
3) 调用强化版 adaptive_measurement_noise 生成 R_k
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .. import config
except ImportError:
    import config

try:
    from .adaptive_measurement_noise import AdaptiveRConfig, build_adaptive_R_columns
except ImportError:
    try:
        from ore.pipeline.adaptive_measurement_noise import AdaptiveRConfig, build_adaptive_R_columns
    except ImportError:
        from adaptive_measurement_noise import AdaptiveRConfig, build_adaptive_R_columns


MERGED_CSV = config.RENDERS_DIR / "merged_observation_truth.csv"
OUT_CSV = MERGED_CSV.with_name("merged_with_pseudo_los.csv")
FOV_X_DEG = config.FOV_X_DEG
IMG_W = config.IMG_W
IMG_H = config.IMG_H
SAFE_RATIO = 0.80


def _safe_box_distance_px(u: float, v: float, img_w: int, img_h: int, safe_ratio: float = SAFE_RATIO) -> float:
    mx = img_w * (1.0 - safe_ratio) / 2.0
    my = img_h * (1.0 - safe_ratio) / 2.0
    return float(min(u - mx, (img_w - mx) - u, v - my, (img_h - my) - v))


def build_pseudo_los_df(df: pd.DataFrame,
                        fov_x_deg: float = 15.0,
                        img_w: int = 1024,
                        img_h: int = 1024,
                        adaptive_cfg: AdaptiveRConfig | None = None) -> pd.DataFrame:
    fov_x_rad = math.radians(fov_x_deg)
    fx = img_w / (2.0 * math.tan(fov_x_rad / 2.0))
    fy = fx
    cx0 = img_w / 2.0
    cy0 = img_h / 2.0

    los_x, los_y, los_z = [], [], []
    az_deg, el_deg = [], []
    u_used, v_used = [], []
    safe_dist_px = []
    border_touch = []

    use_float = ("cx_float" in df.columns) and ("cy_float" in df.columns)
    print("build_pseudo_los[v2]: use_float =", use_float)

    for _, row in df.iterrows():
        if int(row["visible"]) != 1:
            los_x.append(np.nan); los_y.append(np.nan); los_z.append(np.nan)
            az_deg.append(np.nan); el_deg.append(np.nan)
            u_used.append(np.nan); v_used.append(np.nan)
            safe_dist_px.append(np.nan); border_touch.append(np.nan)
            continue

        if use_float and float(row["cx_float"]) >= 0 and float(row["cy_float"]) >= 0:
            u = float(row["cx_float"])
            v = float(row["cy_float"])
        else:
            u = float(row["cx"])
            v = float(row["cy"])

        u = float(np.clip(u, 0.0, img_w - 1.0))
        v = float(np.clip(v, 0.0, img_h - 1.0))
        x = (u - cx0) / fx
        y = (cy0 - v) / fy

        vec = np.array([x, y, 1.0], dtype=float)
        vec = vec / max(np.linalg.norm(vec), 1.0e-12)

        los_x.append(vec[0])
        los_y.append(vec[1])
        los_z.append(vec[2])
        az_deg.append(math.degrees(math.atan2(vec[0], vec[2])))
        el_deg.append(math.degrees(math.atan2(vec[1], vec[2])))
        u_used.append(u)
        v_used.append(v)
        safe_dist_px.append(_safe_box_distance_px(u, v, img_w, img_h, SAFE_RATIO))

        w = float(row["w_float"]) if "w_float" in row.index and pd.notna(row["w_float"]) else float(row["w"])
        h = float(row["h_float"]) if "h_float" in row.index and pd.notna(row["h_float"]) else float(row["h"])
        touches = int((u - 0.5 * w) <= 0.0 or (u + 0.5 * w) >= img_w or (v - 0.5 * h) <= 0.0 or (v + 0.5 * h) >= img_h)
        border_touch.append(touches)

    out = df.copy()
    out["los_cam_x"] = los_x
    out["los_cam_y"] = los_y
    out["los_cam_z"] = los_z
    out["azimuth_deg"] = az_deg
    out["elevation_deg"] = el_deg
    out["u_used_px"] = u_used
    out["v_used_px"] = v_used
    out["safe_box_dist_px"] = safe_dist_px
    out["bbox_touches_image_border"] = border_touch
    out["fx_px"] = fx
    out["fy_px"] = fy

    out = build_adaptive_R_columns(
        out,
        img_w=img_w,
        img_h=img_h,
        fx=fx,
        fy=fy,
        cfg=adaptive_cfg or AdaptiveRConfig(),
    )
    return out


def main():
    df = pd.read_csv(MERGED_CSV)
    out = build_pseudo_los_df(df, fov_x_deg=FOV_X_DEG, img_w=IMG_W, img_h=IMG_H)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print("saved:", OUT_CSV)
    cols = [
        "frame", "visible", "azimuth_deg", "elevation_deg",
        "sigma_u_px", "sigma_v_px", "R_az_az", "R_el_el",
        "center_jump_px", "safe_box_dist_px", "bbox_touches_image_border"
    ]
    print(out[[c for c in cols if c in out.columns]].head(10))


if __name__ == "__main__":
    main()
