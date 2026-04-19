# -*- coding: utf-8 -*-
"""
imm_ukf_estimator.py

顶刊严谨版：解决初始虚假能观性与末端发散的严格物理闭环滤波器
核心理论创新：
1. 柔性零空间协方差防坍塌 (Flexible Null-Space Bounding) [严密封杀 NEES 爆炸]
2. 视觉特征过度包络校正 (Overbounding Correction) [精确提振 NIS，拒绝保守]
3. 强跟踪定向零空间注入 (STF Directional Injection) [剿灭末端突变]

本版新增：已知 observer micro-dv 后的三阶段恢复机制
- control: 机动当帧，直接 predict-only
- relax: 机动后短窗，按高 R / 高 Q 严格保护，必要时继续 skip
- recovery: 渐进恢复更新，若残差仍大则退回 skip 或仅弱更新
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    from .. import config
except ImportError:
    import config

try:
    from .adaptive_measurement_noise import measurement_cov_from_row
except ImportError:
    try:
        from ore.pipeline.adaptive_measurement_noise import measurement_cov_from_row
    except ImportError:
        from adaptive_measurement_noise import measurement_cov_from_row

MU = 3.986004418e14

INPUT_CSV = config.RENDERS_DIR / "merged_with_pseudo_los.csv"
OUT_CSV = INPUT_CSV.with_name("imm_ukf_estimation_result.csv")

IMG_W = config.IMG_W
IMG_H = config.IMG_H
FOV_X_DEG = config.FOV_X_DEG

MAX_BBOX_AREA_FRAC = 0.30

# 将几何包围框映射为真实的 1-sigma 统计协方差。
# 数值层当前问题是过度信观测，因此这里不再压到 0.5。
R_NOMINAL_SCALE = 1.25
R_INFLATION_MAX = 3.0

# Residual / innovation protection for hard cases.
NIS_STRONG_TRACK_THRESHOLD = 8.0
NIS_STRONG_TRACK_MAX_LAMBDA = 4.0
NIS_WEAK_UPDATE_THRESHOLD = 10.0
NIS_SKIP_UPDATE_THRESHOLD = 20.0
INNOV_WEAK_UPDATE_AZ_DEG = 0.90
INNOV_WEAK_UPDATE_EL_DEG = 0.70
INNOV_SKIP_UPDATE_AZ_DEG = 1.60
INNOV_SKIP_UPDATE_EL_DEG = 1.20
WEAK_UPDATE_GAIN_SCALE = 0.25

PI = np.array([[0.992, 0.008], [0.020, 0.980]], dtype=float)
MU0 = np.array([0.50, 0.50], dtype=float)

# 三阶段 control-aware 配置
CONTROL_Q1_SCALE_NOW = 4.0
CONTROL_Q2_SCALE_NOW = 1.0
CONTROL_Q1_SCALE_RELAX = 1.10
CONTROL_Q2_SCALE_RELAX = 0.40
CONTROL_Q1_SCALE_RECOVERY = 0.90
CONTROL_Q2_SCALE_RECOVERY = 0.25
CONTROL_RELAX_STEPS = 4
CONTROL_RECOVERY_STEPS = 4

CONTROL_R_SCALE_NOW = 12.0
CONTROL_R_SCALE_RELAX = 5.0
CONTROL_R_SCALE_RECOVERY = 3.5

CONTROL_JUMP_SOFT_FLOOR_PX = 2.0
CONTROL_JUMP_CLIP_PX = 24.0
CONTROL_JUMP_R_GAIN = 0.10
CONTROL_MANEUVER_PROX_R_GAIN = 2.5
CONTROL_SOFT_SUPPRESS_R_SCALE = 24.0
CONTROL_SOFT_SUPPRESS_JUMP_PX = 16.0

CONTROL_SKIP_UPDATE_ON_CONTROL_FRAME = True
CONTROL_SKIP_UPDATE_RELAX_MIN_COUNTDOWN = 2
CONTROL_SKIP_UPDATE_R_SCALE_THR = 18.0
CONTROL_SKIP_UPDATE_MANEUVER_WEIGHT_THR = 0.55

# recovery: 残差仍大时，不能突然恢复正常 update
CONTROL_RECOVERY_SOFT_RESID_AZ_DEG = 0.60
CONTROL_RECOVERY_SOFT_RESID_EL_DEG = 0.45
CONTROL_RECOVERY_HARD_RESID_AZ_DEG = 1.20
CONTROL_RECOVERY_HARD_RESID_EL_DEG = 0.90
CONTROL_RECOVERY_GAIN_SCALE_SOFT = 0.20
CONTROL_RECOVERY_GAIN_SCALE_HARD = 0.08
CONTROL_RECOVERY_GAIN_SCALE_CLEAR = 0.45

CONTROL_POS_FLOOR_RATIO_DURING_CONTROL = 0.995
CONTROL_VEL_FLOOR_RATIO_DURING_CONTROL = 0.990
CONTROL_POS_FLOOR_RATIO_DURING_RECOVERY = 0.985
CONTROL_VEL_FLOOR_RATIO_DURING_RECOVERY = 0.975

CONTROL_MODEL1_MIN_PROB_CONTROL = 0.998
CONTROL_MODEL1_MIN_PROB_RELAX = 0.996
CONTROL_MODEL1_MIN_PROB_RECOVERY = 0.999

CONTROL_M2_ACC_VAR_CONTROL = (5.0e-5) ** 2
CONTROL_M2_ACC_VAR_RELAX = (7.5e-5) ** 2
CONTROL_M2_ACC_VAR_RECOVERY = (1.0e-4) ** 2
CONTROL_M2_POS_VAR_RATIO_CONTROL = 1.15
CONTROL_M2_POS_VAR_RATIO_RELAX = 1.18
CONTROL_M2_POS_VAR_RATIO_RECOVERY = 1.03
CONTROL_M2_VEL_VAR_RATIO_CONTROL = 1.25
CONTROL_M2_VEL_VAR_RATIO_RELAX = 1.35
CONTROL_M2_VEL_VAR_RATIO_RECOVERY = 1.08
CONTROL_M2_POS_TRACE_RATIO_CONTROL = 1.20
CONTROL_M2_POS_TRACE_RATIO_RELAX = 1.25
CONTROL_M2_POS_TRACE_RATIO_RECOVERY = 1.05
CONTROL_M2_VEL_TRACE_RATIO_CONTROL = 1.35
CONTROL_M2_VEL_TRACE_RATIO_RELAX = 1.45
CONTROL_M2_VEL_TRACE_RATIO_RECOVERY = 1.12
CONTROL_M2_CROSS_SCALE_CONTROL = 0.75
CONTROL_M2_CROSS_SCALE_RELAX = 0.55
CONTROL_M2_CROSS_SCALE_RECOVERY = 0.40

CONTROL_M1_POS_GROWTH_CONTROL = 1.05
CONTROL_M1_POS_GROWTH_RELAX = 1.04
CONTROL_M1_POS_GROWTH_RECOVERY = 1.03
CONTROL_M1_VEL_GROWTH_CONTROL = 1.08
CONTROL_M1_VEL_GROWTH_RELAX = 1.06
CONTROL_M1_VEL_GROWTH_RECOVERY = 1.04
CONTROL_M1_POS_TRACE_GROWTH_CONTROL = 1.05
CONTROL_M1_POS_TRACE_GROWTH_RELAX = 1.04
CONTROL_M1_POS_TRACE_GROWTH_RECOVERY = 1.03
CONTROL_M1_VEL_TRACE_GROWTH_CONTROL = 1.10
CONTROL_M1_VEL_TRACE_GROWTH_RELAX = 1.07
CONTROL_M1_VEL_TRACE_GROWTH_RECOVERY = 1.05
CONTROL_M1_CROSS_SCALE_CONTROL = 0.85
CONTROL_M1_CROSS_SCALE_RELAX = 0.75
CONTROL_M1_CROSS_SCALE_RECOVERY = 0.65


@dataclass(frozen=True)
class ControlPhaseProfile:
    name: str
    r_scale: float
    q1_scale: float
    q2_scale: float
    min_pos_var_ratio: float
    min_vel_var_ratio: float
    model1_floor: float
    m2_acc_var: float
    m2_pos_var_ratio: float
    m2_vel_var_ratio: float
    m2_pos_trace_ratio: float
    m2_vel_trace_ratio: float
    m2_cross_scale: float
    m1_pos_growth: float
    m1_vel_growth: float
    m1_pos_trace_growth: float
    m1_vel_trace_growth: float
    m1_cross_scale: float


@dataclass(frozen=True)
class ResidualProtectionProfile:
    nis_weak_update_threshold: float
    nis_skip_update_threshold: float
    innov_weak_update_az_deg: float
    innov_weak_update_el_deg: float
    innov_skip_update_az_deg: float
    innov_skip_update_el_deg: float
    weak_update_gain_scale: float
    recovery_soft_resid_az_deg: float
    recovery_soft_resid_el_deg: float
    recovery_hard_resid_az_deg: float
    recovery_hard_resid_el_deg: float
    recovery_gain_scale_soft: float
    recovery_gain_scale_hard: float
    recovery_gain_scale_clear: float
    skip_on_control_frame: bool
    skip_update_relax_min_countdown: int
    skip_update_r_scale_thr: float
    skip_update_maneuver_weight_thr: float
    hard_skip_severity_threshold: float


CONTROL_PHASE_PROFILES: dict[str, ControlPhaseProfile] = {
    "normal": ControlPhaseProfile(
        name="normal",
        r_scale=1.0,
        q1_scale=1.0,
        q2_scale=1.0,
        min_pos_var_ratio=0.0,
        min_vel_var_ratio=0.0,
        model1_floor=0.0,
        m2_acc_var=CONTROL_M2_ACC_VAR_RECOVERY,
        m2_pos_var_ratio=1.0,
        m2_vel_var_ratio=1.0,
        m2_pos_trace_ratio=1.0,
        m2_vel_trace_ratio=1.0,
        m2_cross_scale=1.0,
        m1_pos_growth=1.0,
        m1_vel_growth=1.0,
        m1_pos_trace_growth=1.0,
        m1_vel_trace_growth=1.0,
        m1_cross_scale=1.0,
    ),
    "control": ControlPhaseProfile(
        name="control",
        r_scale=CONTROL_R_SCALE_NOW,
        q1_scale=CONTROL_Q1_SCALE_NOW,
        q2_scale=CONTROL_Q2_SCALE_NOW,
        min_pos_var_ratio=CONTROL_POS_FLOOR_RATIO_DURING_CONTROL,
        min_vel_var_ratio=CONTROL_VEL_FLOOR_RATIO_DURING_CONTROL,
        model1_floor=CONTROL_MODEL1_MIN_PROB_CONTROL,
        m2_acc_var=CONTROL_M2_ACC_VAR_CONTROL,
        m2_pos_var_ratio=CONTROL_M2_POS_VAR_RATIO_CONTROL,
        m2_vel_var_ratio=CONTROL_M2_VEL_VAR_RATIO_CONTROL,
        m2_pos_trace_ratio=CONTROL_M2_POS_TRACE_RATIO_CONTROL,
        m2_vel_trace_ratio=CONTROL_M2_VEL_TRACE_RATIO_CONTROL,
        m2_cross_scale=CONTROL_M2_CROSS_SCALE_CONTROL,
        m1_pos_growth=CONTROL_M1_POS_GROWTH_CONTROL,
        m1_vel_growth=CONTROL_M1_VEL_GROWTH_CONTROL,
        m1_pos_trace_growth=CONTROL_M1_POS_TRACE_GROWTH_CONTROL,
        m1_vel_trace_growth=CONTROL_M1_VEL_TRACE_GROWTH_CONTROL,
        m1_cross_scale=CONTROL_M1_CROSS_SCALE_CONTROL,
    ),
    "relax": ControlPhaseProfile(
        name="relax",
        r_scale=CONTROL_R_SCALE_RELAX,
        q1_scale=CONTROL_Q1_SCALE_RELAX,
        q2_scale=CONTROL_Q2_SCALE_RELAX,
        min_pos_var_ratio=CONTROL_POS_FLOOR_RATIO_DURING_CONTROL,
        min_vel_var_ratio=CONTROL_VEL_FLOOR_RATIO_DURING_CONTROL,
        model1_floor=CONTROL_MODEL1_MIN_PROB_RELAX,
        m2_acc_var=CONTROL_M2_ACC_VAR_RELAX,
        m2_pos_var_ratio=CONTROL_M2_POS_VAR_RATIO_RELAX,
        m2_vel_var_ratio=CONTROL_M2_VEL_VAR_RATIO_RELAX,
        m2_pos_trace_ratio=CONTROL_M2_POS_TRACE_RATIO_RELAX,
        m2_vel_trace_ratio=CONTROL_M2_VEL_TRACE_RATIO_RELAX,
        m2_cross_scale=CONTROL_M2_CROSS_SCALE_RELAX,
        m1_pos_growth=CONTROL_M1_POS_GROWTH_RELAX,
        m1_vel_growth=CONTROL_M1_VEL_GROWTH_RELAX,
        m1_pos_trace_growth=CONTROL_M1_POS_TRACE_GROWTH_RELAX,
        m1_vel_trace_growth=CONTROL_M1_VEL_TRACE_GROWTH_RELAX,
        m1_cross_scale=CONTROL_M1_CROSS_SCALE_RELAX,
    ),
    "recovery": ControlPhaseProfile(
        name="recovery",
        r_scale=CONTROL_R_SCALE_RECOVERY,
        q1_scale=CONTROL_Q1_SCALE_RECOVERY,
        q2_scale=CONTROL_Q2_SCALE_RECOVERY,
        min_pos_var_ratio=CONTROL_POS_FLOOR_RATIO_DURING_RECOVERY,
        min_vel_var_ratio=CONTROL_VEL_FLOOR_RATIO_DURING_RECOVERY,
        model1_floor=CONTROL_MODEL1_MIN_PROB_RECOVERY,
        m2_acc_var=CONTROL_M2_ACC_VAR_RECOVERY,
        m2_pos_var_ratio=CONTROL_M2_POS_VAR_RATIO_RECOVERY,
        m2_vel_var_ratio=CONTROL_M2_VEL_VAR_RATIO_RECOVERY,
        m2_pos_trace_ratio=CONTROL_M2_POS_TRACE_RATIO_RECOVERY,
        m2_vel_trace_ratio=CONTROL_M2_VEL_TRACE_RATIO_RECOVERY,
        m2_cross_scale=CONTROL_M2_CROSS_SCALE_RECOVERY,
        m1_pos_growth=CONTROL_M1_POS_GROWTH_RECOVERY,
        m1_vel_growth=CONTROL_M1_VEL_GROWTH_RECOVERY,
        m1_pos_trace_growth=CONTROL_M1_POS_TRACE_GROWTH_RECOVERY,
        m1_vel_trace_growth=CONTROL_M1_VEL_TRACE_GROWTH_RECOVERY,
        m1_cross_scale=CONTROL_M1_CROSS_SCALE_RECOVERY,
    ),
}


def get_control_phase_profile(phase: str) -> ControlPhaseProfile:
    return CONTROL_PHASE_PROFILES.get(str(phase), CONTROL_PHASE_PROFILES["normal"])


RESIDUAL_PROTECTION_PROFILE = ResidualProtectionProfile(
    nis_weak_update_threshold=float(NIS_WEAK_UPDATE_THRESHOLD),
    nis_skip_update_threshold=float(NIS_SKIP_UPDATE_THRESHOLD),
    innov_weak_update_az_deg=float(INNOV_WEAK_UPDATE_AZ_DEG),
    innov_weak_update_el_deg=float(INNOV_WEAK_UPDATE_EL_DEG),
    innov_skip_update_az_deg=float(INNOV_SKIP_UPDATE_AZ_DEG),
    innov_skip_update_el_deg=float(INNOV_SKIP_UPDATE_EL_DEG),
    weak_update_gain_scale=float(WEAK_UPDATE_GAIN_SCALE),
    recovery_soft_resid_az_deg=float(CONTROL_RECOVERY_SOFT_RESID_AZ_DEG),
    recovery_soft_resid_el_deg=float(CONTROL_RECOVERY_SOFT_RESID_EL_DEG),
    recovery_hard_resid_az_deg=float(CONTROL_RECOVERY_HARD_RESID_AZ_DEG),
    recovery_hard_resid_el_deg=float(CONTROL_RECOVERY_HARD_RESID_EL_DEG),
    recovery_gain_scale_soft=float(CONTROL_RECOVERY_GAIN_SCALE_SOFT),
    recovery_gain_scale_hard=float(CONTROL_RECOVERY_GAIN_SCALE_HARD),
    recovery_gain_scale_clear=float(CONTROL_RECOVERY_GAIN_SCALE_CLEAR),
    skip_on_control_frame=bool(CONTROL_SKIP_UPDATE_ON_CONTROL_FRAME),
    skip_update_relax_min_countdown=int(CONTROL_SKIP_UPDATE_RELAX_MIN_COUNTDOWN),
    skip_update_r_scale_thr=float(CONTROL_SKIP_UPDATE_R_SCALE_THR),
    skip_update_maneuver_weight_thr=float(CONTROL_SKIP_UPDATE_MANEUVER_WEIGHT_THR),
    hard_skip_severity_threshold=0.98,
)

MAX_VISIBLE_GAP_FACTOR = 3.0
MIN_VISIBLE_SEGMENT_LEN = 10

DIVERGENCE_TRACE_P_POS_MAX = 5.0e8
DIVERGENCE_SIGMA_POS_MAX_M = 2.0e4
DIVERGENCE_NEES_POS_MAX = 1.0e6
DIVERGENCE_ABS_POS_NORM_MAX_M = 2.0e5
DIVERGENCE_ABS_VEL_NORM_MAX_MPS = 50.0
DIVERGENCE_STEP_POS_GROWTH_MAX = 8.0
DIVERGENCE_STEP_VEL_GROWTH_MAX = 8.0
DIVERGENCE_SOFT_RECOVERY_ENABLE = 1
DIVERGENCE_SOFT_RECOVERY_MAX_COUNT = 3
DIVERGENCE_SOFT_RECOVERY_TRACE_CLAMP = 1.0e8
DIVERGENCE_SOFT_RECOVERY_SIGMA_CLAMP_M = 5.0e3
DIVERGENCE_SOFT_RECOVERY_NEES_CLAMP = 2.0e5
DIVERGENCE_HARD_FAIL_TRACE_P_POS = 1.0e12
DIVERGENCE_PRECAP_ENABLE = 1
DIVERGENCE_PRECAP_POS_TRACE_GROWTH = 4.0
DIVERGENCE_PRECAP_VEL_TRACE_GROWTH = 6.0
DIVERGENCE_PRECAP_SIGMA_GROWTH = 2.0
DIVERGENCE_PRECAP_TRACE_POS_SOFT_CAP = 2.0e8
DIVERGENCE_PRECAP_CROSS_SCALE = 0.85
DIVERGENCE_FALLBACK_TO_PREV_ENABLE = 1
DIVERGENCE_FALLBACK_TO_PREV_MAX_COUNT = 4
DIVERGENCE_FALLBACK_POS_TRACE_GROWTH = 1.35
DIVERGENCE_FALLBACK_VEL_TRACE_GROWTH = 1.50
DIVERGENCE_FALLBACK_CROSS_SCALE = 0.65
DIVERGENCE_FALLBACK_FORCE_RELAX_STEPS = 4
DIVERGENCE_FALLBACK_FORCE_RECOVERY_STEPS = 8

PROTECTED_PHASE_USE_MODEL1_ONLY = 1


@dataclass(frozen=True)
class DivergenceProfile:
    trace_p_pos_max: float
    sigma_pos_max_m: float
    nees_pos_max: float
    abs_pos_norm_max_m: float
    abs_vel_norm_max_mps: float
    step_pos_growth_max: float
    step_vel_growth_max: float
    soft_recovery_enable: bool
    soft_recovery_max_count: int
    soft_recovery_trace_clamp: float
    soft_recovery_sigma_clamp_m: float
    soft_recovery_nees_clamp: float
    hard_fail_trace_p_pos: float
    precap_enable: bool
    precap_pos_trace_growth: float
    precap_vel_trace_growth: float
    precap_sigma_growth: float
    precap_trace_pos_soft_cap: float
    precap_cross_scale: float
    fallback_to_prev_enable: bool
    fallback_to_prev_max_count: int
    fallback_pos_trace_growth: float
    fallback_vel_trace_growth: float
    fallback_cross_scale: float
    fallback_force_relax_steps: int
    fallback_force_recovery_steps: int


DIVERGENCE_PROFILE = DivergenceProfile(
    trace_p_pos_max=float(DIVERGENCE_TRACE_P_POS_MAX),
    sigma_pos_max_m=float(DIVERGENCE_SIGMA_POS_MAX_M),
    nees_pos_max=float(DIVERGENCE_NEES_POS_MAX),
    abs_pos_norm_max_m=float(DIVERGENCE_ABS_POS_NORM_MAX_M),
    abs_vel_norm_max_mps=float(DIVERGENCE_ABS_VEL_NORM_MAX_MPS),
    step_pos_growth_max=float(DIVERGENCE_STEP_POS_GROWTH_MAX),
    step_vel_growth_max=float(DIVERGENCE_STEP_VEL_GROWTH_MAX),
    soft_recovery_enable=bool(DIVERGENCE_SOFT_RECOVERY_ENABLE),
    soft_recovery_max_count=int(DIVERGENCE_SOFT_RECOVERY_MAX_COUNT),
    soft_recovery_trace_clamp=float(DIVERGENCE_SOFT_RECOVERY_TRACE_CLAMP),
    soft_recovery_sigma_clamp_m=float(DIVERGENCE_SOFT_RECOVERY_SIGMA_CLAMP_M),
    soft_recovery_nees_clamp=float(DIVERGENCE_SOFT_RECOVERY_NEES_CLAMP),
    hard_fail_trace_p_pos=float(DIVERGENCE_HARD_FAIL_TRACE_P_POS),
    precap_enable=bool(DIVERGENCE_PRECAP_ENABLE),
    precap_pos_trace_growth=float(DIVERGENCE_PRECAP_POS_TRACE_GROWTH),
    precap_vel_trace_growth=float(DIVERGENCE_PRECAP_VEL_TRACE_GROWTH),
    precap_sigma_growth=float(DIVERGENCE_PRECAP_SIGMA_GROWTH),
    precap_trace_pos_soft_cap=float(DIVERGENCE_PRECAP_TRACE_POS_SOFT_CAP),
    precap_cross_scale=float(DIVERGENCE_PRECAP_CROSS_SCALE),
    fallback_to_prev_enable=bool(DIVERGENCE_FALLBACK_TO_PREV_ENABLE),
    fallback_to_prev_max_count=int(DIVERGENCE_FALLBACK_TO_PREV_MAX_COUNT),
    fallback_pos_trace_growth=float(DIVERGENCE_FALLBACK_POS_TRACE_GROWTH),
    fallback_vel_trace_growth=float(DIVERGENCE_FALLBACK_VEL_TRACE_GROWTH),
    fallback_cross_scale=float(DIVERGENCE_FALLBACK_CROSS_SCALE),
    fallback_force_relax_steps=int(DIVERGENCE_FALLBACK_FORCE_RELAX_STEPS),
    fallback_force_recovery_steps=int(DIVERGENCE_FALLBACK_FORCE_RECOVERY_STEPS),
)


def nearest_spd(P: np.ndarray, eps: float = 1e-8, diag_floor: float = 1e-10) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.nan_to_num(P, nan=0.0, posinf=1e12, neginf=-1e12)
    P = 0.5 * (P + P.T)

    eigvals, eigvecs = np.linalg.eigh(P)
    floor = max(eps, diag_floor)
    eigvals = np.maximum(eigvals, floor)
    P_spd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    P_spd = 0.5 * (P_spd + P_spd.T)

    d = np.diag(P_spd).copy()
    d[d < floor] = floor
    np.fill_diagonal(P_spd, d)
    return P_spd


def chol_psd(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    P = nearest_spd(P, eps=eps)
    I = np.eye(P.shape[0], dtype=float)
    jitter = eps
    for _ in range(8):
        try:
            return np.linalg.cholesky(P + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.cholesky(nearest_spd(P + jitter * I, eps=max(jitter, 1e-6)))


def safe_inv_spd(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    P = nearest_spd(P, eps=eps)
    I = np.eye(P.shape[0], dtype=float)
    jitter = eps
    for _ in range(8):
        try:
            return np.linalg.inv(P + jitter * I)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.pinv(P + jitter * I, rcond=1e-10)


def wrap_angle_rad(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def residual_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    y = a - b
    y[0] = wrap_angle_rad(y[0])
    y[1] = wrap_angle_rad(y[1])
    return y


def circular_weighted_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    s = np.sum(weights * np.sin(angles))
    c = np.sum(weights * np.cos(angles))
    return math.atan2(s, c)


def compute_visual_degradation_factor(row: pd.Series) -> float:
    area_frac = float(row.get("bbox_area", 0)) / (IMG_W * IMG_H)
    w_norm = float(row.get("w_norm", 0))
    h_norm = float(row.get("h_norm", 0))

    factor = 1.0
    if area_frac > MAX_BBOX_AREA_FRAC or w_norm > 0.5 or h_norm > 0.5:
        degradation = (area_frac / MAX_BBOX_AREA_FRAC) ** 2
        factor = min(1.0 + degradation, 2.0)
    return max(1.0, factor)


def update_tpm_dynamic(nis_model1: float, base_PI: np.ndarray, k_step: int) -> np.ndarray:
    if k_step <= 3:
        return base_PI.copy()

    p_value = 1.0 - stats.chi2.cdf(nis_model1, df=2)
    gamma = np.exp(-50.0 * p_value)
    dynamic_PI = base_PI.copy()
    p11_min = 0.50
    dynamic_PI[0, 0] = max(p11_min, base_PI[0, 0] - gamma * 0.4)
    dynamic_PI[0, 1] = 1.0 - dynamic_PI[0, 0]
    p22_max = 0.99
    dynamic_PI[1, 1] = min(p22_max, base_PI[1, 1] + gamma * 0.1)
    dynamic_PI[1, 0] = 1.0 - dynamic_PI[1, 1]
    return dynamic_PI


def compute_nees(x_est: np.ndarray, P_est: np.ndarray, x_true: np.ndarray) -> float:
    e = x_est[:3] - x_true[:3]
    P_pos = nearest_spd(P_est[:3, :3], eps=1e-8)
    P_inv = safe_inv_spd(P_pos, eps=1e-8)
    return float(e.T @ P_inv @ e)


def _jump_factor_from_row(row: pd.Series) -> float:
    jump_px = float(row.get("center_jump_px", 0.0) or 0.0)
    if not np.isfinite(jump_px):
        jump_px = 0.0
    jump_px = max(0.0, jump_px - CONTROL_JUMP_SOFT_FLOOR_PX)
    jump_px = min(jump_px, CONTROL_JUMP_CLIP_PX)
    return 1.0 + CONTROL_JUMP_R_GAIN * jump_px


def _maneuver_proximity_factor_from_row(row: pd.Series) -> float:
    w = float(row.get("maneuver_proximity_weight", 0.0) or 0.0)
    if not np.isfinite(w):
        w = 0.0
    w = float(np.clip(w, 0.0, 1.0))
    return 1.0 + CONTROL_MANEUVER_PROX_R_GAIN * w


def determine_control_phase(has_known_control: bool, relax_countdown: int, recovery_countdown: int) -> str:
    if has_known_control:
        return "control"
    if relax_countdown > 0:
        return "relax"
    if recovery_countdown > 0:
        return "recovery"
    return "normal"


def compute_control_aware_measurement_scale(
    row: pd.Series,
    has_known_control: bool,
    relax_countdown: int,
    recovery_countdown: int,
) -> tuple[float, float, float, int, str]:
    phase = determine_control_phase(has_known_control, relax_countdown, recovery_countdown)
    phase_profile = get_control_phase_profile(phase)
    control_factor = float(phase_profile.r_scale)

    jump_factor = _jump_factor_from_row(row)
    maneuver_factor = _maneuver_proximity_factor_from_row(row)
    visual_factor = compute_visual_degradation_factor(row)
    scale = R_NOMINAL_SCALE * control_factor * jump_factor * maneuver_factor * visual_factor

    jump_px = float(row.get("center_jump_px", 0.0) or 0.0)
    soft_suppress = int((phase in {"control", "relax"}) and np.isfinite(jump_px) and jump_px >= CONTROL_SOFT_SUPPRESS_JUMP_PX)
    if soft_suppress:
        scale *= CONTROL_SOFT_SUPPRESS_R_SCALE
    return float(min(scale, 500.0)), float(jump_factor), float(maneuver_factor), soft_suppress, phase


def decide_measurement_update_mode(
    *,
    phase: str,
    residual_severity: float,
    has_known_control: bool,
    pre_skip_measurement_update: bool,
    profile: ResidualProtectionProfile = RESIDUAL_PROTECTION_PROFILE,
) -> tuple[bool, float, str]:
    phase_name = str(phase).strip().lower()
    if phase_name not in {"normal", "control", "relax", "recovery"}:
        phase_name = "normal"

    if bool(pre_skip_measurement_update):
        return True, 0.0, "skip_on_known_control_frame"
    if phase_name == "control" and bool(has_known_control) and bool(profile.skip_on_control_frame):
        return True, 0.0, "skip_on_known_control_frame"

    sev = float(residual_severity) if np.isfinite(float(residual_severity)) else 0.0
    sev = float(np.clip(sev, 0.0, 1.0))
    if sev >= float(profile.hard_skip_severity_threshold):
        return True, 0.0, f"{phase_name}_severity_hard_skip"

    if phase_name == "recovery":
        min_gain = float(profile.recovery_gain_scale_hard)
        max_gain = float(profile.recovery_gain_scale_clear)
    else:
        min_gain = float(profile.weak_update_gain_scale)
        max_gain = 1.0

    gain_scale = max_gain - (max_gain - min_gain) * sev
    gain_scale = float(np.clip(gain_scale, min_gain, max_gain))
    if sev <= 1.0e-12:
        mode = "normal_update"
    else:
        mode = f"{phase_name}_severity_guarded_update"
    return False, gain_scale, mode


def _continuous_severity(value: float, soft_threshold: float, hard_threshold: float) -> float:
    if not np.isfinite(value):
        return 0.0
    v = float(value)
    lo = float(soft_threshold)
    hi = float(hard_threshold)
    if hi <= lo + 1.0e-9:
        return float(v >= hi)
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def compute_residual_severity(
    quad: float,
    innov_az_deg: float,
    innov_el_deg: float,
    *,
    phase: str,
    profile: ResidualProtectionProfile = RESIDUAL_PROTECTION_PROFILE,
) -> float:
    phase_name = str(phase).strip().lower()
    if phase_name == "recovery":
        az_soft, az_hard = float(profile.recovery_soft_resid_az_deg), float(profile.recovery_hard_resid_az_deg)
        el_soft, el_hard = float(profile.recovery_soft_resid_el_deg), float(profile.recovery_hard_resid_el_deg)
    else:
        az_soft, az_hard = float(profile.innov_weak_update_az_deg), float(profile.innov_skip_update_az_deg)
        el_soft, el_hard = float(profile.innov_weak_update_el_deg), float(profile.innov_skip_update_el_deg)

    nis_soft = float(profile.nis_weak_update_threshold)
    nis_hard = float(profile.nis_skip_update_threshold)

    s_nis = _continuous_severity(float(quad), nis_soft, nis_hard)
    s_az = _continuous_severity(float(innov_az_deg), az_soft, az_hard)
    s_el = _continuous_severity(float(innov_el_deg), el_soft, el_hard)
    return float(max(s_nis, s_az, s_el))


def get_observer_control_from_row(row: pd.Series) -> tuple[np.ndarray, bool]:
    flag = int(float(row.get("observer_maneuver_flag", 0) or 0)) == 1
    dv_rtn = np.array([
        float(row.get("observer_dv_R_mps", 0.0) or 0.0),
        float(row.get("observer_dv_T_mps", 0.0) or 0.0),
        float(row.get("observer_dv_N_mps", 0.0) or 0.0),
    ], dtype=float)
    if np.linalg.norm(dv_rtn) <= 0.0:
        flag = False
    return dv_rtn, flag


def apply_observer_control_to_state(x: np.ndarray, dv_rtn: np.ndarray, model_dim: int) -> np.ndarray:
    x2 = x.copy()
    x2[3:6] = x2[3:6] - dv_rtn
    return x2


def estimate_mean_motion_from_truth(df: pd.DataFrame) -> Tuple[float, float]:
    r = np.array([df.loc[0, "rc_x_m"], df.loc[0, "rc_y_m"], df.loc[0, "rc_z_m"]], dtype=float)
    v = np.array([df.loc[0, "vc_x_mps"], df.loc[0, "vc_y_mps"], df.loc[0, "vc_z_mps"]], dtype=float)
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
    a = 1.0 / (2.0 / rnorm - vnorm ** 2 / MU)
    n = math.sqrt(MU / a ** 3)
    return a, n


def infer_nominal_dt_sec(df: pd.DataFrame) -> float:
    if "t_sec" not in df.columns or len(df) < 2:
        return 5.0
    t_all = np.asarray(df["t_sec"], dtype=float)
    diffs = np.diff(t_all)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
    if diffs.size == 0:
        return 5.0
    return float(np.median(diffs))


def keep_primary_visible_segment(obs: pd.DataFrame, nominal_dt_sec: float) -> pd.DataFrame:
    if len(obs) == 0:
        return obs.copy()
    gap_thr = max(nominal_dt_sec * MAX_VISIBLE_GAP_FACTOR, nominal_dt_sec + 1.0e-9)
    t = obs["t_sec"].to_numpy(dtype=float)
    seg_break = np.zeros(len(obs), dtype=int)
    if len(obs) > 1:
        seg_break[1:] = (np.diff(t) > gap_thr).astype(int)
    seg_id = np.cumsum(seg_break)
    obs2 = obs.copy()
    obs2["visible_segment_id"] = seg_id

    seg_meta = []
    for sid, g in obs2.groupby("visible_segment_id", sort=True):
        seg_meta.append((int(sid), int(len(g)), float(g["t_sec"].iloc[0]), float(g["t_sec"].iloc[-1])))
    if not seg_meta:
        return obs2.drop(columns=["visible_segment_id"])

    seg_meta.sort(key=lambda x: (-x[1], x[2]))
    best_sid, best_len, _, _ = seg_meta[0]
    if best_len < MIN_VISIBLE_SEGMENT_LEN:
        return obs2.drop(columns=["visible_segment_id"])

    kept = obs2[obs2["visible_segment_id"] == best_sid].copy().reset_index(drop=True)
    return kept.drop(columns=["visible_segment_id"])



def regularize_nominal_model_for_phase(
    x1: np.ndarray,
    P1: np.ndarray,
    P1_prev: np.ndarray,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    if phase == "normal":
        return x1.copy(), nearest_spd(P1.copy())

    phase_profile = get_control_phase_profile(phase)
    x1r = x1.copy()
    P1r = nearest_spd(P1.copy())
    P1_prev = nearest_spd(P1_prev.copy())

    pos_growth = float(phase_profile.m1_pos_growth)
    vel_growth = float(phase_profile.m1_vel_growth)
    pos_trace_growth = float(phase_profile.m1_pos_trace_growth)
    vel_trace_growth = float(phase_profile.m1_vel_trace_growth)
    cross_scale = float(phase_profile.m1_cross_scale)

    prev_pos_diag = np.maximum(np.diag(P1_prev[:3, :3]), 1.0e-12)
    prev_vel_diag = np.maximum(np.diag(P1_prev[3:6, 3:6]), 1.0e-12)
    for i in range(3):
        P1r[i, i] = min(P1r[i, i], pos_growth * prev_pos_diag[i])
        P1r[3 + i, 3 + i] = min(P1r[3 + i, 3 + i], vel_growth * prev_vel_diag[i])

    prev_pos_trace = max(float(np.trace(P1_prev[:3, :3])), 1.0e-12)
    prev_vel_trace = max(float(np.trace(P1_prev[3:6, 3:6])), 1.0e-12)
    cur_pos_trace = float(np.trace(P1r[:3, :3]))
    cur_vel_trace = float(np.trace(P1r[3:6, 3:6]))

    pos_trace_cap = pos_trace_growth * prev_pos_trace
    vel_trace_cap = vel_trace_growth * prev_vel_trace

    if cur_pos_trace > pos_trace_cap:
        s = pos_trace_cap / cur_pos_trace
        P1r[:3, :3] *= s
        P1r[:3, 3:6] *= math.sqrt(s)
        P1r[3:6, :3] *= math.sqrt(s)

    if cur_vel_trace > vel_trace_cap:
        s = vel_trace_cap / cur_vel_trace
        P1r[3:6, 3:6] *= s
        P1r[:3, 3:6] *= math.sqrt(s)
        P1r[3:6, :3] *= math.sqrt(s)

    P1r[:3, 3:6] *= cross_scale
    P1r[3:6, :3] *= cross_scale

    return x1r, nearest_spd(P1r)

def compute_model1_floor_for_phase(phase: str) -> float:
    return float(get_control_phase_profile(phase).model1_floor)


def detect_state_divergence(x_comb: np.ndarray, prev_x_comb: np.ndarray | None = None) -> tuple[bool, str]:
    x_comb = np.asarray(x_comb, dtype=float)
    if not np.all(np.isfinite(x_comb[:6])):
        return True, "nonfinite_state"

    pos_norm = float(np.linalg.norm(x_comb[:3]))
    vel_norm = float(np.linalg.norm(x_comb[3:6]))
    if pos_norm >= float(DIVERGENCE_PROFILE.abs_pos_norm_max_m):
        return True, f"abs_pos_norm_exceeded:{pos_norm:.6e}"
    if vel_norm >= float(DIVERGENCE_PROFILE.abs_vel_norm_max_mps):
        return True, f"abs_vel_norm_exceeded:{vel_norm:.6e}"

    if prev_x_comb is not None:
        prev_x_comb = np.asarray(prev_x_comb, dtype=float)
        if np.all(np.isfinite(prev_x_comb[:6])):
            prev_pos_norm = float(np.linalg.norm(prev_x_comb[:3]))
            prev_vel_norm = float(np.linalg.norm(prev_x_comb[3:6]))
            if prev_pos_norm >= 1.0e3 and pos_norm > float(DIVERGENCE_PROFILE.step_pos_growth_max) * prev_pos_norm:
                return True, f"step_pos_growth_exceeded:{pos_norm:.6e}"
            if prev_vel_norm >= 1.0e-2 and vel_norm > float(DIVERGENCE_PROFILE.step_vel_growth_max) * prev_vel_norm:
                return True, f"step_vel_growth_exceeded:{vel_norm:.6e}"

    return False, ""


def detect_estimator_divergence(P_comb: np.ndarray, nees_pos: float) -> tuple[bool, str]:
    if not np.all(np.isfinite(P_comb)):
        return True, "nonfinite_covariance"
    trace_pos = float(np.trace(P_comb[:3, :3]))
    sigma_pos = np.sqrt(np.maximum(np.diag(P_comb[:3, :3]), 0.0))
    if (not np.isfinite(trace_pos)) or trace_pos >= float(DIVERGENCE_PROFILE.trace_p_pos_max):
        return True, f"trace_P_pos_exceeded:{trace_pos:.6e}"
    if np.any(~np.isfinite(sigma_pos)) or float(np.max(sigma_pos)) >= float(DIVERGENCE_PROFILE.sigma_pos_max_m):
        return True, f"sigma_pos_exceeded:{float(np.max(sigma_pos)):.6e}"
    if (not np.isfinite(nees_pos)) or nees_pos >= float(DIVERGENCE_PROFILE.nees_pos_max):
        return True, f"nees_exceeded:{nees_pos:.6e}"
    return False, ""


def _apply_predivergence_covariance_cap(P: np.ndarray, P_prev: np.ndarray | None) -> tuple[np.ndarray, int]:
    if (not bool(DIVERGENCE_PROFILE.precap_enable)) or P_prev is None:
        return nearest_spd(P, eps=1e-7), 0

    P_cap = nearest_spd(P.copy(), eps=1e-7)
    P_prev = nearest_spd(P_prev.copy(), eps=1e-7)
    used = 0

    prev_pos_trace = max(float(np.trace(P_prev[:3, :3])), 1.0e-12)
    prev_vel_trace = max(float(np.trace(P_prev[3:6, 3:6])), 1.0e-12)
    cur_pos_trace = float(np.trace(P_cap[:3, :3]))
    cur_vel_trace = float(np.trace(P_cap[3:6, 3:6]))

    pos_trace_cap = min(
        float(DIVERGENCE_PROFILE.precap_trace_pos_soft_cap),
        float(DIVERGENCE_PROFILE.precap_pos_trace_growth) * prev_pos_trace,
    )
    vel_trace_cap = float(DIVERGENCE_PROFILE.precap_vel_trace_growth) * prev_vel_trace

    if np.isfinite(cur_pos_trace) and cur_pos_trace > pos_trace_cap:
        s = max(pos_trace_cap / max(cur_pos_trace, 1.0e-12), 1.0e-12)
        P_cap[:3, :3] *= s
        P_cap[:3, 3:6] *= math.sqrt(s)
        P_cap[3:6, :3] *= math.sqrt(s)
        used = 1

    if np.isfinite(cur_vel_trace) and cur_vel_trace > vel_trace_cap:
        s = max(vel_trace_cap / max(cur_vel_trace, 1.0e-12), 1.0e-12)
        P_cap[3:6, 3:6] *= s
        P_cap[:3, 3:6] *= math.sqrt(s)
        P_cap[3:6, :3] *= math.sqrt(s)
        used = 1

    prev_sigma = np.sqrt(np.maximum(np.diag(P_prev[:3, :3]), 1.0e-12))
    cur_sigma = np.sqrt(np.maximum(np.diag(P_cap[:3, :3]), 1.0e-12))
    sigma_cap = float(DIVERGENCE_PROFILE.precap_sigma_growth) * np.max(prev_sigma)
    if np.isfinite(sigma_cap) and sigma_cap > 0.0 and np.max(cur_sigma) > sigma_cap:
        s = max((sigma_cap / max(np.max(cur_sigma), 1.0e-12)) ** 2, 1.0e-12)
        P_cap[:3, :3] *= s
        P_cap[:3, 3:6] *= math.sqrt(s)
        P_cap[3:6, :3] *= math.sqrt(s)
        used = 1

    if used:
        P_cap[:3, 3:6] *= float(DIVERGENCE_PROFILE.precap_cross_scale)
        P_cap[3:6, :3] *= float(DIVERGENCE_PROFILE.precap_cross_scale)
    return nearest_spd(P_cap, eps=1e-7), int(used)


def _can_attempt_divergence_soft_recovery(diverged_reason: str, soft_recovery_count: int) -> bool:
    if not bool(DIVERGENCE_PROFILE.soft_recovery_enable):
        return False
    if soft_recovery_count >= int(DIVERGENCE_PROFILE.soft_recovery_max_count):
        return False
    reason = str(diverged_reason or "")
    if reason.startswith("nonfinite_covariance"):
        return False
    if reason.startswith("trace_P_pos_exceeded:"):
        try:
            trace_val = float(reason.split(":", 1)[1])
        except Exception:
            return False
        return np.isfinite(trace_val) and trace_val < float(DIVERGENCE_PROFILE.hard_fail_trace_p_pos)
    if (
        reason.startswith("sigma_pos_exceeded:")
        or reason.startswith("nees_exceeded:")
        or reason.startswith("abs_pos_norm_exceeded:")
        or reason.startswith("abs_vel_norm_exceeded:")
        or reason.startswith("step_pos_growth_exceeded:")
        or reason.startswith("step_vel_growth_exceeded:")
    ):
        return True
    return False


def _clamp_combined_covariance_for_soft_recovery(P: np.ndarray) -> np.ndarray:
    P = nearest_spd(P.copy(), eps=1e-7)
    sigma_pos = np.sqrt(np.maximum(np.diag(P[:3, :3]), 0.0))
    max_sigma = float(np.max(sigma_pos)) if sigma_pos.size > 0 else 0.0
    if np.isfinite(max_sigma) and max_sigma > float(DIVERGENCE_PROFILE.soft_recovery_sigma_clamp_m):
        s = (float(DIVERGENCE_PROFILE.soft_recovery_sigma_clamp_m) / max(max_sigma, 1.0e-12)) ** 2
        P[:3, :3] *= s
        P[:3, 3:6] *= math.sqrt(s)
        P[3:6, :3] *= math.sqrt(s)
    trace_pos = float(np.trace(P[:3, :3]))
    if np.isfinite(trace_pos) and trace_pos > float(DIVERGENCE_PROFILE.soft_recovery_trace_clamp):
        s = float(DIVERGENCE_PROFILE.soft_recovery_trace_clamp) / max(trace_pos, 1.0e-12)
        P[:3, :3] *= s
        P[:3, 3:6] *= math.sqrt(s)
        P[3:6, :3] *= math.sqrt(s)
    return nearest_spd(P, eps=1e-7)


def _inflate_previous_posterior_for_fallback(x_prev: np.ndarray, P_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_fb = np.asarray(x_prev, dtype=float).copy()
    P_fb = nearest_spd(np.asarray(P_prev, dtype=float).copy(), eps=1e-7)

    P_fb[:3, :3] *= float(DIVERGENCE_PROFILE.fallback_pos_trace_growth)
    P_fb[3:6, 3:6] *= float(DIVERGENCE_PROFILE.fallback_vel_trace_growth)
    P_fb[:3, 3:6] *= float(DIVERGENCE_PROFILE.fallback_cross_scale)
    P_fb[3:6, :3] *= float(DIVERGENCE_PROFILE.fallback_cross_scale)
    P_fb = _clamp_combined_covariance_for_soft_recovery(P_fb)
    return x_fb, nearest_spd(P_fb, eps=1e-7)


def _can_attempt_prev_posterior_fallback(diverged_reason: str, fallback_count: int, P_prev: np.ndarray | None) -> bool:
    if not bool(DIVERGENCE_PROFILE.fallback_to_prev_enable):
        return False
    if P_prev is None:
        return False
    if fallback_count >= int(DIVERGENCE_PROFILE.fallback_to_prev_max_count):
        return False
    reason = str(diverged_reason or "")
    return (
        reason.startswith("trace_P_pos_exceeded:")
        or reason.startswith("sigma_pos_exceeded:")
        or reason.startswith("nees_exceeded:")
        or reason.startswith("abs_pos_norm_exceeded:")
        or reason.startswith("abs_vel_norm_exceeded:")
        or reason.startswith("step_pos_growth_exceeded:")
        or reason.startswith("step_vel_growth_exceeded:")
    )


def regularize_maneuver_model_for_phase(
    x2: np.ndarray,
    P2: np.ndarray,
    P1_ref: np.ndarray,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    if phase == "normal":
        return x2.copy(), nearest_spd(P2)

    phase_profile = get_control_phase_profile(phase)
    x2r = x2.copy()
    P2r = nearest_spd(P2.copy())
    P1r = nearest_spd(P1_ref.copy())

    acc_var = float(phase_profile.m2_acc_var)
    pos_var_ratio = float(phase_profile.m2_pos_var_ratio)
    vel_var_ratio = float(phase_profile.m2_vel_var_ratio)
    pos_trace_ratio = float(phase_profile.m2_pos_trace_ratio)
    vel_trace_ratio = float(phase_profile.m2_vel_trace_ratio)
    cross_scale = float(phase_profile.m2_cross_scale)

    x2r[6:9] = 0.0
    P2r[6:9, :] = 0.0
    P2r[:, 6:9] = 0.0
    for idx in range(6, 9):
        P2r[idx, idx] = acc_var

    ref_pos_diag = np.maximum(np.diag(P1r[:3, :3]), 1.0e-12)
    ref_vel_diag = np.maximum(np.diag(P1r[3:6, 3:6]), 1.0e-12)

    pos_caps = pos_var_ratio * ref_pos_diag
    vel_caps = vel_var_ratio * ref_vel_diag
    for i in range(3):
        P2r[i, i] = min(P2r[i, i], pos_caps[i])
        P2r[3 + i, 3 + i] = min(P2r[3 + i, 3 + i], vel_caps[i])

    ref_pos_trace = float(np.trace(P1r[:3, :3]))
    ref_vel_trace = float(np.trace(P1r[3:6, 3:6]))
    cap_pos_trace = max(pos_trace_ratio * ref_pos_trace, 1.0e-12)
    cap_vel_trace = max(vel_trace_ratio * ref_vel_trace, 1.0e-12)

    cur_pos_trace = float(np.trace(P2r[:3, :3]))
    if cur_pos_trace > cap_pos_trace:
        scale = cap_pos_trace / cur_pos_trace
        s = math.sqrt(scale)
        P2r[:3, :3] *= scale
        P2r[:3, 3:6] *= s
        P2r[3:6, :3] *= s

    cur_vel_trace = float(np.trace(P2r[3:6, 3:6]))
    if cur_vel_trace > cap_vel_trace:
        scale = cap_vel_trace / cur_vel_trace
        s = math.sqrt(scale)
        P2r[3:6, 3:6] *= scale
        P2r[:3, 3:6] *= s
        P2r[3:6, :3] *= s

    P2r[:3, 3:6] *= cross_scale
    P2r[3:6, :3] *= cross_scale

    return x2r, nearest_spd(P2r)


def cap_model2_mixed_state_for_phase(
    x1_ref: np.ndarray,
    P1_ref: np.ndarray,
    x2_mix: np.ndarray,
    P2_mix: np.ndarray,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    if phase == "normal":
        return x2_mix.copy(), nearest_spd(P2_mix)
    return regularize_maneuver_model_for_phase(x2_mix, P2_mix, P1_ref, phase)


def shadow_model2_from_model1_for_phase(
    x1_ref: np.ndarray,
    P1_ref: np.ndarray,
    phase: str,
) -> tuple[np.ndarray, np.ndarray]:
    x2_shadow, P2_shadow = embed_6_to_9(x1_ref, P1_ref)
    return regularize_maneuver_model_for_phase(x2_shadow, P2_shadow, P1_ref, phase)


def hcw6_deriv(x: np.ndarray, n: float) -> np.ndarray:
    R, T, N, dR, dT, dN = x
    ddR = 2.0 * n * dT + 3.0 * n ** 2 * R
    ddT = -2.0 * n * dR
    ddN = -n ** 2 * N
    return np.array([dR, dT, dN, ddR, ddT, ddN], dtype=float)


def accel9_deriv(x: np.ndarray, n: float) -> np.ndarray:
    R, T, N, dR, dT, dN, aR, aT, aN = x
    ddR = 2.0 * n * dT + 3.0 * n ** 2 * R + aR
    ddT = -2.0 * n * dR + aT
    ddN = -n ** 2 * N + aN
    return np.array([dR, dT, dN, ddR, ddT, ddN, 0.0, 0.0, 0.0], dtype=float)


def rk4_step(f, x: np.ndarray, dt: float, n: float) -> np.ndarray:
    k1 = f(x, n)
    k2 = f(x + 0.5 * dt * k1, n)
    k3 = f(x + 0.5 * dt * k2, n)
    k4 = f(x + dt * k3, n)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def hx6(x: np.ndarray) -> np.ndarray:
    R, T, N = x[0], x[1], x[2]
    T_safe = T if abs(T) > 1e-6 else (1e-6 if T >= 0.0 else -1e-6)
    az = math.atan2(R, T_safe)
    el = math.atan2(N, T_safe)
    return np.array([az, el], dtype=float)


def hx9(x: np.ndarray) -> np.ndarray:
    return hx6(x[:6])


def unscented_weights(n: int, alpha: float = 0.1, beta: float = 2.0, kappa: float = 0.0):
    lam = alpha ** 2 * (n + kappa) - n
    c = n + lam
    Wm = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wc = np.full(2 * n + 1, 1.0 / (2.0 * c))
    Wm[0] = lam / c
    Wc[0] = lam / c + (1.0 - alpha ** 2 + beta)
    return lam, c, Wm, Wc


def ukf_predict_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    Rm: np.ndarray,
    dt: float,
    n_orbit: float,
    model_dim: int,
    process_Q: np.ndarray,
    known_observer_dv_rtn: np.ndarray | None = None,
    skip_measurement_update: bool = False,
    min_pos_var_ratio_to_pred: float = 0.0,
    min_vel_var_ratio_to_pred: float = 0.0,
    control_phase: str = "normal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int, float, float, float, float]:
    phase_name = str(control_phase).strip().lower()
    if phase_name not in {"normal", "control", "relax", "recovery"}:
        phase_name = "normal"
    recovery_phase = phase_name == "recovery"

    f = hcw6_deriv if model_dim == 6 else accel9_deriv
    hx = hx6 if model_dim == 6 else hx9

    _, c, Wm, Wc = unscented_weights(model_dim)
    S_chol = chol_psd(P)
    sigmas = [x]
    for i in range(model_dim):
        sigmas.append(x + math.sqrt(c) * S_chol[:, i])
        sigmas.append(x - math.sqrt(c) * S_chol[:, i])
    sigmas = np.asarray(sigmas)

    sigmas_pred = np.array([rk4_step(f, s, dt, n_orbit) for s in sigmas])

    has_control = False
    if known_observer_dv_rtn is not None and np.linalg.norm(known_observer_dv_rtn) > 1e-6:
        has_control = True
        sigmas_pred = np.array([apply_observer_control_to_state(s, known_observer_dv_rtn, model_dim) for s in sigmas_pred])

    x_pred = np.sum(Wm[:, None] * sigmas_pred, axis=0)
    P_pred = process_Q.copy()
    for i in range(sigmas_pred.shape[0]):
        dx = sigmas_pred[i] - x_pred
        P_pred += Wc[i] * np.outer(dx, dx)
    P_pred = nearest_spd(P_pred)

    if has_control and model_dim == 9:
        x_pred[6:9] = 0.0
        P_pred[6:9, 6:9] = np.diag(np.maximum(np.diag(P_pred[6:9, 6:9]), 1e-10))
        P_pred = nearest_spd(P_pred, eps=1e-8)

    Zsig = np.array([hx(s) for s in sigmas_pred])
    azhat = circular_weighted_mean(Zsig[:, 0], Wm)
    elhat = circular_weighted_mean(Zsig[:, 1], Wm)
    zhat = np.array([azhat, elhat], dtype=float)

    Szz_prior = Rm.copy()
    for i in range(Zsig.shape[0]):
        dz = residual_z(Zsig[i], zhat)
        Szz_prior += Wc[i] * np.outer(dz, dz)
    Szz_prior = nearest_spd(Szz_prior)

    Pxz_prior = np.zeros((model_dim, 2), dtype=float)
    for i in range(sigmas_pred.shape[0]):
        dx = sigmas_pred[i] - x_pred
        dz = residual_z(Zsig[i], zhat)
        Pxz_prior += Wc[i] * np.outer(dx, dz)

    innov = residual_z(z, zhat)
    innov[0] = np.clip(innov[0], -math.radians(5.0), math.radians(5.0))
    innov[1] = np.clip(innov[1], -math.radians(5.0), math.radians(5.0))

    Sinv_temp = safe_inv_spd(Szz_prior, eps=1e-8)
    gamma = float(innov.T @ Sinv_temp @ innov)

    if gamma > NIS_STRONG_TRACK_THRESHOLD and not has_control and not skip_measurement_update and not recovery_phase:
        lambda_factor = min(gamma / 3.0, NIS_STRONG_TRACK_MAX_LAMBDA)
        P_pred[0, 0] *= min(lambda_factor, 3.0)
        P_pred[3, 3] *= min(lambda_factor, 3.0)
        P_pred = P_pred * lambda_factor
        Szz = nearest_spd((Szz_prior - Rm) * lambda_factor + Rm)
        Pxz = Pxz_prior * lambda_factor
    else:
        Szz = Szz_prior
        Pxz = Pxz_prior

    Sinv = safe_inv_spd(Szz, eps=1e-8)
    quad = float(innov.T @ Sinv @ innov)

    innov_az_deg = abs(math.degrees(float(innov[0])))
    innov_el_deg = abs(math.degrees(float(innov[1])))
    effective_skip_measurement_update = bool(skip_measurement_update)
    update_gain_scale = 1.0
    residual_severity = compute_residual_severity(
        quad,
        innov_az_deg,
        innov_el_deg,
        phase=phase_name,
        profile=RESIDUAL_PROTECTION_PROFILE,
    )
    effective_skip_measurement_update, update_gain_scale, _ = decide_measurement_update_mode(
        phase=phase_name,
        residual_severity=float(residual_severity),
        has_known_control=bool(has_control),
        pre_skip_measurement_update=bool(skip_measurement_update),
        profile=RESIDUAL_PROTECTION_PROFILE,
    )

    if effective_skip_measurement_update:
        x_upd = x_pred.copy()
        P_upd = nearest_spd(P_pred)
    else:
        K = Pxz @ Sinv
        if update_gain_scale < 1.0:
            K = update_gain_scale * K
        x_upd = x_pred + K @ innov
        P_upd = P_pred - K @ Szz @ K.T

    if min_pos_var_ratio_to_pred > 0.0:
        for idx in range(min(3, model_dim)):
            P_upd[idx, idx] = max(P_upd[idx, idx], float(min_pos_var_ratio_to_pred) * P_pred[idx, idx])
    if min_vel_var_ratio_to_pred > 0.0:
        for idx in range(3, min(6, model_dim)):
            P_upd[idx, idx] = max(P_upd[idx, idx], float(min_vel_var_ratio_to_pred) * P_pred[idx, idx])

    if not has_control and not effective_skip_measurement_update:
        P_upd[0, 0] = max(P_upd[0, 0], 0.95 * P_pred[0, 0])
        P_upd[3, 3] = max(P_upd[3, 3], 0.95 * P_pred[3, 3])

    P_upd = nearest_spd(P_upd)

    sign, logdet_szz = np.linalg.slogdet(Szz)
    if sign <= 0:
        logdet_szz = math.log(max(np.linalg.det(nearest_spd(Szz)), 1e-20))

    log_likelihood = -0.5 * (quad + logdet_szz + 2 * math.log(2.0 * math.pi))
    if effective_skip_measurement_update:
        log_likelihood -= 0.05 * quad
    elif update_gain_scale < 1.0:
        log_likelihood -= 0.01 * quad

    return (
        x_upd,
        P_upd,
        zhat,
        innov,
        log_likelihood,
        quad,
        int(effective_skip_measurement_update),
        float(update_gain_scale),
        float(innov_az_deg),
        float(innov_el_deg),
        float(residual_severity),
    )


def embed_6_to_9(x6: np.ndarray, P6: np.ndarray, acc_var: float = (1e-5) ** 2) -> Tuple[np.ndarray, np.ndarray]:
    x9 = np.zeros(9, dtype=float)
    x9[:6] = x6
    P9 = np.zeros((9, 9), dtype=float)
    P9[:6, :6] = P6
    P9[6, 6] = acc_var
    P9[7, 7] = acc_var
    P9[8, 8] = acc_var
    return x9, nearest_spd(P9)


def extract_6_from_9(x9: np.ndarray, P9: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return x9[:6].copy(), nearest_spd(P9[:6, :6].copy())


def imm_run(df: pd.DataFrame, out_csv: Path | None = None, x0_noise: np.ndarray | None = None) -> pd.DataFrame:
    obs_all = df[df["visible"] == 1].copy().reset_index(drop=True)
    nominal_dt_sec = infer_nominal_dt_sec(df)
    obs = keep_primary_visible_segment(obs_all, nominal_dt_sec)
    if len(obs) < 10:
        raise RuntimeError("可见帧太少，无法执行 IMM-UKF。")

    _, n_orbit = estimate_mean_motion_from_truth(df)
    t = obs["t_sec"].values.astype(float)
    dt_arr = np.diff(t, prepend=t[0])

    z_all = np.vstack([np.deg2rad(obs["azimuth_deg"].values), np.deg2rad(obs["elevation_deg"].values)]).T
    x_true6 = obs[["rel_R_m", "rel_T_m", "rel_N_m", "rel_dR_mps", "rel_dT_mps", "rel_dN_mps"]].values.astype(float)

    P0_pos = np.diag([50.0 ** 2, 80.0 ** 2, 50.0 ** 2])
    P0_vel = np.diag([0.1 ** 2, 0.1 ** 2, 0.1 ** 2])
    P1 = np.block([[P0_pos, np.zeros((3, 3))], [np.zeros((3, 3)), P0_vel]])

    x1 = x_true6[0].copy()
    if x0_noise is not None:
        x1 += x0_noise
    else:
        x1 += np.array([50.0, 80.0, 50.0, 0.1, 0.1, 0.1])

    Q1_base = np.diag([0.1 ** 2, 0.1 ** 2, 0.1 ** 2, 0.005 ** 2, 0.005 ** 2, 0.005 ** 2])
    x2 = np.zeros(9, dtype=float)
    x2[:6] = x1
    P2 = np.diag([50.0 ** 2, 80.0 ** 2, 50.0 ** 2, 0.1 ** 2, 0.1 ** 2, 0.1 ** 2, (2e-4) ** 2, (2e-4) ** 2, (2e-4) ** 2])
    Q2_base = np.diag([1.5 ** 2, 1.5 ** 2, 1.5 ** 2, 0.05 ** 2, 0.05 ** 2, 0.05 ** 2, (2e-4) ** 2, (2e-4) ** 2, (2e-4) ** 2])

    mu = MU0.copy()
    rows = []
    control_relax_countdown = 0
    control_recovery_countdown = 0
    current_PI = PI.copy()
    divergence_soft_recovery_count = 0
    divergence_prev_fallback_count = 0
    prev_x_comb = None
    prev_P_comb = None

    for k in range(len(obs)):
        dt = dt_arr[k] if k > 0 else (dt_arr[1] if len(dt_arr) > 1 else nominal_dt_sec)
        dt = float(np.clip(dt, nominal_dt_sec, nominal_dt_sec * MAX_VISIBLE_GAP_FACTOR))
        z = z_all[k]
        rowk = obs.iloc[k]

        cv_factor = compute_visual_degradation_factor(rowk)
        observer_dv_rtn, has_known_control = get_observer_control_from_row(rowk)
        if has_known_control:
            control_relax_countdown = CONTROL_RELAX_STEPS
            control_recovery_countdown = CONTROL_RECOVERY_STEPS

        control_relax_before = int(control_relax_countdown)
        control_recovery_before = int(control_recovery_countdown)
        R_scale, jump_factor, maneuver_factor, soft_suppress_update, control_phase = compute_control_aware_measurement_scale(
            rowk, has_known_control, control_relax_countdown, control_recovery_countdown
        )
        phase_profile = get_control_phase_profile(control_phase)
        pre_skip_measurement_update = bool(
            (control_phase == "control")
            and bool(has_known_control)
            and bool(RESIDUAL_PROTECTION_PROFILE.skip_on_control_frame)
        )
        skip_measurement_update = bool(pre_skip_measurement_update)
        update_mode = "skip_on_known_control_frame" if skip_measurement_update else f"{control_phase}_severity_pending"
        Rk = nearest_spd(measurement_cov_from_row(rowk, fallback_deg=0.05) * R_scale)

        Q_inflation_multiplier = cv_factor
        if control_phase in {"control", "relax", "recovery"}:
            Q_inflation_multiplier *= 1.0 + 0.20 * float(maneuver_factor - 1.0)
            Q_inflation_multiplier *= 1.0 + 0.05 * max(jump_factor - 1.0, 0.0)

        Q1_dyn = Q1_base * Q_inflation_multiplier
        Q2_dyn = Q2_base * Q_inflation_multiplier

        q1_scale = float(phase_profile.q1_scale)
        q2_scale = float(phase_profile.q2_scale)

        Q1 = Q1_dyn * q1_scale
        Q2 = Q2_dyn * q2_scale

        pos_floor_ratio = float(phase_profile.min_pos_var_ratio)
        vel_floor_ratio = float(phase_profile.min_vel_var_ratio)

        if control_phase in {"control", "relax", "recovery"}:
            x2, P2 = regularize_maneuver_model_for_phase(x2, P2, P1, control_phase)

        x1c, P1c = embed_6_to_9(x1, P1)
        x2c, P2c = x2.copy(), P2.copy()
        Xc = [x1c, x2c]
        Pc = [P1c, P2c]

        cbar = current_PI.T @ mu
        cbar = np.clip(cbar, 1e-300, None)
        mix_prob = np.zeros((2, 2), dtype=float)
        for j in range(2):
            denom = cbar[j]
            for i in range(2):
                mix_prob[i, j] = current_PI[i, j] * mu[i] / denom

        mixed_common_x = []
        mixed_common_P = []
        for j in range(2):
            xmix = np.zeros(9, dtype=float)
            for i in range(2):
                xmix += mix_prob[i, j] * Xc[i]
            Pmix = np.zeros((9, 9), dtype=float)
            for i in range(2):
                dx = Xc[i] - xmix
                Pmix += mix_prob[i, j] * (Pc[i] + np.outer(dx, dx))
            mixed_common_x.append(xmix)
            mixed_common_P.append(nearest_spd(Pmix, eps=1e-7))

        x1_mix, P1_mix = extract_6_from_9(mixed_common_x[0], mixed_common_P[0])
        x2_mix, P2_mix = mixed_common_x[1], mixed_common_P[1]
        if control_phase in {"control", "relax", "recovery"}:
            x2_mix, P2_mix = cap_model2_mixed_state_for_phase(x1_mix, P1_mix, x2_mix, P2_mix, control_phase)

        x1_upd, P1_upd, z1hat, innov1, like1, nis1, skip_used_1, gain_scale_1, innov_az_deg_1, innov_el_deg_1, residual_severity_1 = ukf_predict_update(
            x1_mix, P1_mix, z, Rk, dt, n_orbit, 6, Q1,
            known_observer_dv_rtn=observer_dv_rtn if has_known_control else None,
            skip_measurement_update=skip_measurement_update,
            min_pos_var_ratio_to_pred=pos_floor_ratio,
            min_vel_var_ratio_to_pred=vel_floor_ratio,
            control_phase=control_phase,
        )
        if control_phase in {"control", "relax", "recovery"}:
            x1_upd, P1_upd = regularize_nominal_model_for_phase(x1_upd, P1_upd, P1, control_phase)

        x2_upd, P2_upd, z2hat, innov2, like2, nis2, skip_used_2, gain_scale_2, innov_az_deg_2, innov_el_deg_2, residual_severity_2 = ukf_predict_update(
            x2_mix, P2_mix, z, Rk, dt, n_orbit, 9, Q2,
            known_observer_dv_rtn=observer_dv_rtn if has_known_control else None,
            skip_measurement_update=skip_measurement_update,
            min_pos_var_ratio_to_pred=pos_floor_ratio,
            min_vel_var_ratio_to_pred=vel_floor_ratio,
            control_phase=control_phase,
        )

        if control_phase in {"control", "relax", "recovery"}:
            x2_upd, P2_upd = regularize_maneuver_model_for_phase(x2_upd, P2_upd, P1_upd, control_phase)
            if PROTECTED_PHASE_USE_MODEL1_ONLY:
                x2_upd, P2_upd = shadow_model2_from_model1_for_phase(x1_upd, P1_upd, control_phase)

        skip_measurement_update_used = int(max(skip_used_1, skip_used_2))
        update_gain_scale_used = float(min(gain_scale_1, gain_scale_2))
        innov_az_deg_used = float(max(innov_az_deg_1, innov_az_deg_2))
        innov_el_deg_used = float(max(innov_el_deg_1, innov_el_deg_2))
        residual_severity_used = float(max(residual_severity_1, residual_severity_2))
        _, expected_gain_scale, update_mode = decide_measurement_update_mode(
            phase=control_phase,
            residual_severity=residual_severity_used,
            has_known_control=bool(has_known_control),
            pre_skip_measurement_update=bool(pre_skip_measurement_update),
            profile=RESIDUAL_PROTECTION_PROFILE,
        )
        if skip_measurement_update_used == 1 and str(update_mode).endswith("severity_guarded_update"):
            update_mode = str(update_mode).replace("severity_guarded_update", "severity_hard_skip")
        if update_gain_scale_used > 0.0 and np.isfinite(expected_gain_scale):
            update_gain_scale_used = float(min(update_gain_scale_used, expected_gain_scale))

        protected_phase = control_phase in {"control", "relax", "recovery"}

        if protected_phase and PROTECTED_PHASE_USE_MODEL1_ONLY:
            mu = np.array([1.0, 0.0], dtype=float)
        else:
            log_cbar = np.log(np.clip(cbar, 1e-300, None))
            log_weights = np.array([log_cbar[0] + like1, log_cbar[1] + like2], dtype=float)
            max_logw = np.max(log_weights)
            weights = np.exp(log_weights - max_logw)
            sumw = np.sum(weights)

            if sumw <= 0 or not np.isfinite(sumw):
                mu = cbar.copy()
                mu = mu / np.sum(mu)
            else:
                mu = weights / sumw

            mu = np.asarray(mu, dtype=float)
            mu = np.clip(mu, 1e-12, None)
            mu = mu / np.sum(mu)
            if protected_phase:
                mu0_floor = compute_model1_floor_for_phase(control_phase)
                mu[0] = max(float(mu[0]), mu0_floor)
                mu[1] = max(1.0 - mu[0], 1.0e-12)
                mu = mu / np.sum(mu)

        x1c_upd, P1c_upd = embed_6_to_9(x1_upd, P1_upd)
        x2c_upd, P2c_upd = x2_upd.copy(), P2_upd.copy()
        if protected_phase:
            x2c_upd, P2c_upd = cap_model2_mixed_state_for_phase(x1_upd, P1_upd, x2c_upd, P2c_upd, control_phase)
        Xc_upd = [x1c_upd, x2c_upd]
        Pc_upd = [P1c_upd, P2c_upd]

        if protected_phase and PROTECTED_PHASE_USE_MODEL1_ONLY:
            x_comb = x1c_upd.copy()
            P_comb = nearest_spd(P1c_upd.copy(), eps=1e-7)
            az_pred = float(z1hat[0])
            el_pred = float(z1hat[1])
            x2, P2 = shadow_model2_from_model1_for_phase(x1_upd, P1_upd, control_phase)
        else:
            x_comb = mu[0] * Xc_upd[0] + mu[1] * Xc_upd[1]
            P_comb = np.zeros((9, 9), dtype=float)
            for i in range(2):
                dx = Xc_upd[i] - x_comb
                P_comb += mu[i] * (Pc_upd[i] + np.outer(dx, dx))
            P_comb = nearest_spd(P_comb, eps=1e-7)
            az_pred = circular_weighted_mean(np.array([z1hat[0], z2hat[0]]), mu)
            el_pred = circular_weighted_mean(np.array([z1hat[1], z2hat[1]]), mu)
            x2, P2 = x2_upd, P2_upd

        x1, P1 = x1_upd, P1_upd
        P_comb, divergence_precap_used = _apply_predivergence_covariance_cap(P_comb, prev_P_comb)
        nees_pos = compute_nees(x_comb, P_comb, x_true6[k])
        state_diverged, state_diverged_reason = detect_state_divergence(x_comb, prev_x_comb)
        diverged, diverged_reason = detect_estimator_divergence(P_comb, nees_pos)
        if state_diverged:
            diverged = True
            diverged_reason = state_diverged_reason
        divergence_soft_recovery_used = 0
        divergence_prev_fallback_used = 0
        divergence_recovery_mode = "none"

        if diverged and _can_attempt_divergence_soft_recovery(diverged_reason, divergence_soft_recovery_count):
            x_comb = x1c_upd.copy()
            P_comb = _clamp_combined_covariance_for_soft_recovery(P1c_upd.copy())
            mu = np.array([0.999, 0.001], dtype=float)
            x1, P1 = extract_6_from_9(x_comb, P_comb)
            x2, P2 = shadow_model2_from_model1_for_phase(x1, P1, "recovery")
            control_relax_countdown = max(control_relax_countdown, int(DIVERGENCE_PROFILE.fallback_force_relax_steps))
            control_recovery_countdown = max(
                control_recovery_countdown,
                int(DIVERGENCE_PROFILE.fallback_force_recovery_steps),
            )
            nees_pos = min(
                compute_nees(x_comb, P_comb, x_true6[k]),
                float(DIVERGENCE_PROFILE.soft_recovery_nees_clamp),
            )
            diverged, diverged_reason = detect_estimator_divergence(P_comb, nees_pos)
            divergence_soft_recovery_used = 1
            divergence_soft_recovery_count += 1
            divergence_recovery_mode = "soft_recovery"

        if diverged and _can_attempt_prev_posterior_fallback(diverged_reason, divergence_prev_fallback_count, prev_P_comb):
            x_comb, P_comb = _inflate_previous_posterior_for_fallback(prev_x_comb, prev_P_comb)
            x1, P1 = extract_6_from_9(x_comb, P_comb)
            x2, P2 = shadow_model2_from_model1_for_phase(x1, P1, "recovery")
            mu = np.array([0.999, 0.001], dtype=float)
            control_relax_countdown = max(control_relax_countdown, int(DIVERGENCE_PROFILE.fallback_force_relax_steps))
            control_recovery_countdown = max(
                control_recovery_countdown,
                int(DIVERGENCE_PROFILE.fallback_force_recovery_steps),
            )
            nees_pos = min(
                compute_nees(x_comb, P_comb, x_true6[k]),
                float(DIVERGENCE_PROFILE.soft_recovery_nees_clamp),
            )
            diverged, diverged_reason = detect_estimator_divergence(P_comb, nees_pos)
            divergence_prev_fallback_used = 1
            divergence_prev_fallback_count += 1
            divergence_recovery_mode = "fallback_to_previous_posterior"

        if not diverged:
            if divergence_soft_recovery_used == 0:
                divergence_soft_recovery_count = 0
            if divergence_prev_fallback_used == 0:
                divergence_prev_fallback_count = 0
        else:
            sample_name = str(rowk.get("sample", rowk.get("id", f"k={k}")))
            raise RuntimeError(
                f"Estimator diverged at frame={int(rowk.get('frame', -1))}, "
                f"t={float(rowk.get('t_sec', -1.0)):.3f}, sample={sample_name}, reason={diverged_reason}"
            )

        innovation_residual_norm_deg = float(math.hypot(innov_az_deg_used, innov_el_deg_used))
        weak_update_used = int((skip_measurement_update_used == 0) and (update_gain_scale_used < 0.999999))
        control_phase_active = int(control_phase in {"control", "relax", "recovery"})

        row = rowk.copy()
        row["R_est_m"] = x_comb[0]
        row["T_est_m"] = x_comb[1]
        row["N_est_m"] = x_comb[2]
        row["dR_est_mps"] = x_comb[3]
        row["dT_est_mps"] = x_comb[4]
        row["dN_est_mps"] = x_comb[5]
        row["P00"] = P_comb[0, 0]
        row["P11"] = P_comb[1, 1]
        row["P22"] = P_comb[2, 2]
        row["sigma_R_m"] = math.sqrt(max(P_comb[0, 0], 0.0))
        row["sigma_T_m"] = math.sqrt(max(P_comb[1, 1], 0.0))
        row["sigma_N_m"] = math.sqrt(max(P_comb[2, 2], 0.0))
        row["trace_P_pos"] = float(P_comb[0, 0] + P_comb[1, 1] + P_comb[2, 2])
        row["model_prob_nomaneuver"] = float(mu[0])
        row["model_prob_maneuver"] = float(mu[1])
        row["az_pred_deg"] = math.degrees(az_pred)
        row["el_pred_deg"] = math.degrees(el_pred)
        row["NIS_M1"] = float(nis1)
        row["NIS_M2"] = float(nis2)
        row["NEES_pos"] = float(nees_pos)
        row["R_scale_applied"] = float(R_scale)
        row["R_jump_factor"] = float(jump_factor)
        row["R_maneuver_factor"] = float(maneuver_factor)
        row["measurement_update_soft_suppressed"] = int(soft_suppress_update)
        row["measurement_update_skipped"] = int(skip_measurement_update_used)
        row["measurement_update_mode"] = str(update_mode)
        row["measurement_update_gain_scale"] = float(update_gain_scale_used)
        row["innovation_az_deg_abs"] = float(innov_az_deg_used)
        row["innovation_el_deg_abs"] = float(innov_el_deg_used)
        row["innovation_residual_norm_deg"] = float(innovation_residual_norm_deg)
        row["innovation_residual_severity_m1"] = float(residual_severity_1)
        row["innovation_residual_severity_m2"] = float(residual_severity_2)
        row["innovation_residual_severity"] = float(residual_severity_used)
        row["known_observer_control"] = int(has_known_control)
        row["control_phase"] = str(control_phase)
        row["control_phase_profile"] = str(get_control_phase_profile(control_phase).name)
        row["control_phase_active"] = int(control_phase_active)
        row["control_diag_r_scale"] = float(R_scale)
        row["control_diag_skip_update"] = int(skip_measurement_update_used)
        row["control_diag_weak_update"] = int(weak_update_used)
        row["control_diag_residual_norm_deg"] = float(innovation_residual_norm_deg)
        row["control_diag_residual_severity"] = float(residual_severity_used)
        row["control_diag_mode_weight_nominal"] = float(mu[0])
        row["control_diag_mode_weight_maneuver"] = float(mu[1])
        row["protected_phase_model1_only"] = int(bool(protected_phase and PROTECTED_PHASE_USE_MODEL1_ONLY))
        row["divergence_precap_used"] = int(divergence_precap_used)
        row["divergence_soft_recovery_used"] = int(divergence_soft_recovery_used)
        row["divergence_soft_recovery_count"] = int(divergence_soft_recovery_count)
        row["divergence_prev_fallback_used"] = int(divergence_prev_fallback_used)
        row["divergence_prev_fallback_count"] = int(divergence_prev_fallback_count)
        row["divergence_recovery_mode"] = str(divergence_recovery_mode)
        row["model1_floor_applied"] = float(compute_model1_floor_for_phase(control_phase))
        row["control_relax_countdown_before"] = int(control_relax_before)
        row["control_recovery_countdown_before"] = int(control_recovery_before)
        rows.append(row)
        prev_x_comb = np.asarray(x_comb, dtype=float).copy()
        prev_P_comb = nearest_spd(np.asarray(P_comb, dtype=float).copy(), eps=1e-7)

        if control_relax_countdown > 0:
            control_relax_countdown -= 1
        elif control_recovery_countdown > 0:
            control_recovery_countdown -= 1

        current_PI = PI.copy() if (control_relax_countdown > 0 or control_recovery_countdown > 0) else update_tpm_dynamic(nis1, PI, k)
        rows[-1]["control_relax_countdown_after"] = int(control_relax_countdown)
        rows[-1]["control_recovery_countdown_after"] = int(control_recovery_countdown)

    out = pd.DataFrame(rows)
    if out_csv is not None:
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def main():
    df = pd.read_csv(INPUT_CSV)
    out = imm_run(df, OUT_CSV)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
