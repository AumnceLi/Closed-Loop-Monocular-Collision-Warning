# -*- coding: utf-8 -*-
"""
adaptive_measurement_noise.py

强化版：
1) 对 bbox 小、中心跳变、机动邻域、图像边界、safe-box 边界、截断框、早期可见暖启动联合增噪
2) 对长时间可见间隔后的 jump 做 dt 归一化，避免“缺帧后首帧”被误判为剧烈观测突变
3) 输出可解释的权重列，便于后续排查
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class AdaptiveRConfig:
    base_sigma_px: float = 0.35
    k1_area: float = 25.0
    k2_jump: float = 0.12
    min_sigma_px: float = 0.20
    max_sigma_px: float = 10.00
    anisotropic: bool = True
    edge_boost: float = 0.20
    edge_margin_ratio: float = 0.12

    jump_clip_px: float = 24.0
    jump_soft_floor_px: float = 2.0
    jump_ema_alpha: float = 0.45

    maneuver_window_sec: float = 12.0
    maneuver_boost_px: float = 3.50
    post_maneuver_only: bool = False

    safe_box_ratio: float = 0.80
    safe_box_margin_px: float = 160.0
    safe_box_boost: float = 0.35

    truncation_window_px: float = 18.0
    truncation_boost_px: float = 2.50

    warmup_visible_frames: int = 8
    warmup_boost_px: float = 2.00


def _median_positive_dt(t: np.ndarray) -> float:
    if t.size < 2:
        return 1.0
    diffs = np.diff(t.astype(float))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def add_center_jump_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    use_float = ("cx_float" in out.columns) and ("cy_float" in out.columns)
    cx_col = "cx_float" if use_float else "cx"
    cy_col = "cy_float" if use_float else "cy"

    cx = out[cx_col].astype(float).to_numpy(copy=True)
    cy = out[cy_col].astype(float).to_numpy(copy=True)
    vis = out["visible"].astype(int).to_numpy(copy=True)
    t = out["t_sec"].astype(float).to_numpy(copy=True) if "t_sec" in out.columns else np.arange(len(out), dtype=float)
    nominal_dt = _median_positive_dt(t)

    dc = np.full(len(out), np.nan, dtype=float)
    du = np.full(len(out), np.nan, dtype=float)
    dv = np.full(len(out), np.nan, dtype=float)
    jump_raw = np.full(len(out), np.nan, dtype=float)

    prev_idx = None
    for i in range(len(out)):
        if vis[i] != 1:
            continue
        if prev_idx is None:
            du[i] = dv[i] = dc[i] = jump_raw[i] = 0.0
        else:
            dt = max(float(t[i] - t[prev_idx]), nominal_dt)
            step_scale = max(dt / max(nominal_dt, 1.0e-9), 1.0)
            du_i = cx[i] - cx[prev_idx]
            dv_i = cy[i] - cy[prev_idx]
            dist = math.hypot(du_i, dv_i)
            du[i] = du_i
            dv[i] = dv_i
            jump_raw[i] = dist
            dc[i] = dist / step_scale
        prev_idx = i

    out["du_px"] = du
    out["dv_px"] = dv
    out["center_jump_raw_px"] = jump_raw
    out["center_jump_px"] = dc

    vis_idx = np.full(len(out), np.nan, dtype=float)
    streak = np.zeros(len(out), dtype=int)
    cnt = 0
    cur_streak = 0
    for i in range(len(out)):
        if vis[i] == 1:
            cnt += 1
            cur_streak += 1
            vis_idx[i] = cnt
            streak[i] = cur_streak
        else:
            cur_streak = 0
    out["visible_count"] = vis_idx
    out["visible_streak"] = streak

    alpha = 0.45
    ema = np.full(len(out), np.nan, dtype=float)
    prev = None
    for i in range(len(out)):
        if vis[i] != 1:
            continue
        val = float(dc[i]) if np.isfinite(dc[i]) else 0.0
        prev = val if prev is None else alpha * val + (1.0 - alpha) * prev
        ema[i] = prev
    out["center_jump_ema_px"] = ema

    warmup_weight = np.zeros(len(out), dtype=float)
    warm_n = 8
    for i in range(len(out)):
        if vis[i] != 1:
            continue
        s = int(streak[i])
        warmup_weight[i] = max(0.0, 1.0 - (s - 1) / max(1, warm_n))
    out["visible_warmup_weight"] = warmup_weight
    return out


def add_maneuver_proximity_columns(df: pd.DataFrame, cfg: AdaptiveRConfig) -> pd.DataFrame:
    out = df.copy()
    t = out["t_sec"].astype(float).to_numpy() if "t_sec" in out.columns else np.arange(len(out), dtype=float)

    if "observer_maneuver_flag" not in out.columns:
        out["maneuver_proximity_weight"] = 0.0
        out["maneuver_proximity_dt_sec"] = np.nan
        return out

    flag = out["observer_maneuver_flag"].fillna(0).astype(float).to_numpy()
    maneuver_times = t[flag > 0.5]
    if maneuver_times.size == 0:
        out["maneuver_proximity_weight"] = 0.0
        out["maneuver_proximity_dt_sec"] = np.nan
        return out

    dt_nearest = np.empty_like(t, dtype=float)
    weight = np.zeros_like(t, dtype=float)
    window = max(float(cfg.maneuver_window_sec), 1.0e-9)

    for i, ti in enumerate(t):
        if cfg.post_maneuver_only:
            deltas = ti - maneuver_times
            deltas = deltas[deltas >= 0.0]
            if deltas.size == 0:
                dt_nearest[i] = np.nan
                weight[i] = 0.0
                continue
            dt_min = float(np.min(deltas))
        else:
            dt_min = float(np.min(np.abs(maneuver_times - ti)))
        dt_nearest[i] = dt_min
        weight[i] = max(0.0, 1.0 - dt_min / window)

    out["maneuver_proximity_weight"] = weight
    out["maneuver_proximity_dt_sec"] = dt_nearest
    return out


def _edge_proximity_penalty(u: float, v: float, img_w: int, img_h: int, cfg: AdaptiveRConfig) -> float:
    mx = cfg.edge_margin_ratio * img_w
    my = cfg.edge_margin_ratio * img_h

    left = max(0.0, mx - u)
    right = max(0.0, u - (img_w - mx))
    top = max(0.0, my - v)
    bottom = max(0.0, v - (img_h - my))

    p = (left + right) / max(mx, 1.0) + (top + bottom) / max(my, 1.0)
    return cfg.edge_boost * p


def _safe_box_proximity_penalty(u: float, v: float, img_w: int, img_h: int, cfg: AdaptiveRConfig) -> float:
    margin_x = img_w * (1.0 - cfg.safe_box_ratio) / 2.0
    margin_y = img_h * (1.0 - cfg.safe_box_ratio) / 2.0
    left, right = margin_x, img_w - margin_x
    top, bottom = margin_y, img_h - margin_y
    dist = min(u - left, right - u, v - top, bottom - v)
    if dist >= cfg.safe_box_margin_px:
        return 0.0
    return cfg.safe_box_boost * max(0.0, 1.0 - dist / max(cfg.safe_box_margin_px, 1.0))


def _truncation_weight(row: pd.Series, img_w: int, img_h: int, cfg: AdaptiveRConfig, cx_col: str, cy_col: str, w_col: str, h_col: str) -> float:
    u = float(row[cx_col]); v = float(row[cy_col]); w = float(row[w_col]); h = float(row[h_col])
    left = u - 0.5 * w
    right = u + 0.5 * w
    top = v - 0.5 * h
    bottom = v + 0.5 * h
    d = min(left, img_w - right, top, img_h - bottom)
    if d >= cfg.truncation_window_px:
        return 0.0
    return max(0.0, 1.0 - d / max(cfg.truncation_window_px, 1.0))


def compute_adaptive_pixel_sigma(
    w: float,
    h: float,
    jump_px: float,
    u: float,
    v: float,
    img_w: int,
    img_h: int,
    cfg: AdaptiveRConfig,
    maneuver_weight: float = 0.0,
    safe_box_weight: float = 0.0,
    truncation_weight: float = 0.0,
    warmup_weight: float = 0.0,
) -> Tuple[float, float]:
    area = max(float(w) * float(h), 1.0)
    jump = max(float(jump_px), 0.0) if not np.isnan(jump_px) else 0.0
    jump = max(0.0, jump - float(cfg.jump_soft_floor_px))
    jump = min(jump, float(cfg.jump_clip_px))

    base_var = cfg.base_sigma_px ** 2
    area_term = cfg.k1_area / area
    jump_term = cfg.k2_jump * jump
    edge_term = _edge_proximity_penalty(float(u), float(v), img_w, img_h, cfg)
    maneuver_term = max(0.0, float(cfg.maneuver_boost_px) * float(maneuver_weight))
    safebox_term = max(0.0, float(safe_box_weight))
    trunc_term = max(0.0, float(cfg.truncation_boost_px) * float(truncation_weight))
    warmup_term = max(0.0, float(cfg.warmup_boost_px) * float(warmup_weight))

    sigma_u = math.sqrt(base_var + area_term + jump_term + edge_term + maneuver_term + safebox_term + trunc_term + warmup_term)
    if cfg.anisotropic:
        sigma_v = math.sqrt(base_var + area_term + 1.15 * jump_term + edge_term + 1.10 * maneuver_term + safebox_term + 1.05 * trunc_term + warmup_term)
    else:
        sigma_v = sigma_u

    sigma_u = float(np.clip(sigma_u, cfg.min_sigma_px, cfg.max_sigma_px))
    sigma_v = float(np.clip(sigma_v, cfg.min_sigma_px, cfg.max_sigma_px))
    return sigma_u, sigma_v


def pixel_sigma_to_angle_cov(
    sigma_u_px: float,
    sigma_v_px: float,
    fx: float,
    fy: float,
) -> np.ndarray:
    var_az = (sigma_u_px / max(fx, 1e-9)) ** 2
    var_el = (sigma_v_px / max(fy, 1e-9)) ** 2
    return np.array([[var_az, 0.0], [0.0, var_el]], dtype=float)


def build_adaptive_R_columns(
    df: pd.DataFrame,
    img_w: int,
    img_h: int,
    fx: float,
    fy: float,
    cfg: AdaptiveRConfig | None = None,
) -> pd.DataFrame:
    if cfg is None:
        cfg = AdaptiveRConfig()

    out = add_center_jump_columns(df)
    out = add_maneuver_proximity_columns(out, cfg)
    use_float = ("cx_float" in out.columns) and ("cy_float" in out.columns)
    cx_col = "cx_float" if use_float else "cx"
    cy_col = "cy_float" if use_float else "cy"
    w_col = "w_float" if "w_float" in out.columns else "w"
    h_col = "h_float" if "h_float" in out.columns else "h"

    sigma_u_list = []
    sigma_v_list = []
    r00_list = []
    r01_list = []
    r11_list = []
    safebox_weight_list = []
    trunc_weight_list = []

    for _, row in out.iterrows():
        if int(row["visible"]) != 1:
            sigma_u_list.append(np.nan)
            sigma_v_list.append(np.nan)
            r00_list.append(np.nan)
            r01_list.append(np.nan)
            r11_list.append(np.nan)
            safebox_weight_list.append(np.nan)
            trunc_weight_list.append(np.nan)
            continue

        safebox_w = _safe_box_proximity_penalty(float(row[cx_col]), float(row[cy_col]), img_w, img_h, cfg)
        trunc_w = _truncation_weight(row, img_w, img_h, cfg, cx_col, cy_col, w_col, h_col)
        sigma_u, sigma_v = compute_adaptive_pixel_sigma(
            w=float(row[w_col]),
            h=float(row[h_col]),
            jump_px=float(row.get("center_jump_ema_px", row.get("center_jump_px", 0.0)) or 0.0),
            u=float(row[cx_col]),
            v=float(row[cy_col]),
            img_w=img_w,
            img_h=img_h,
            cfg=cfg,
            maneuver_weight=float(row.get("maneuver_proximity_weight", 0.0) or 0.0),
            safe_box_weight=safebox_w,
            truncation_weight=trunc_w,
            warmup_weight=float(row.get("visible_warmup_weight", 0.0) or 0.0),
        )
        Rk = pixel_sigma_to_angle_cov(sigma_u, sigma_v, fx, fy)
        sigma_u_list.append(sigma_u)
        sigma_v_list.append(sigma_v)
        r00_list.append(float(Rk[0, 0]))
        r01_list.append(float(Rk[0, 1]))
        r11_list.append(float(Rk[1, 1]))
        safebox_weight_list.append(float(safebox_w))
        trunc_weight_list.append(float(trunc_w))

    out["sigma_u_px"] = sigma_u_list
    out["sigma_v_px"] = sigma_v_list
    out["R_az_az"] = r00_list
    out["R_az_el"] = r01_list
    out["R_el_el"] = r11_list
    out["safebox_noise_weight"] = safebox_weight_list
    out["truncation_noise_weight"] = trunc_weight_list
    return out


def measurement_cov_from_row(row: pd.Series, fallback_deg: float = 0.08) -> np.ndarray:
    if {"R_az_az", "R_az_el", "R_el_el"}.issubset(row.index) and not pd.isna(row["R_az_az"]):
        R = np.array(
            [[float(row["R_az_az"]), float(row.get("R_az_el", 0.0))],
             [float(row.get("R_az_el", 0.0)), float(row["R_el_el"])]],
            dtype=float,
        )
        return R
    sig = math.radians(fallback_deg)
    return np.diag([sig ** 2, sig ** 2])


if __name__ == "__main__":
    print("This module is intended to be imported by build_pseudo_los.py")
