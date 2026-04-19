# -*- coding: utf-8 -*-
"""
risk_prediction_from_ukf.py

新版本风险评估：
1) 从 IMM-UKF 后验状态和协方差出发
2) 传播到未来窗口，找出均值轨迹上的 TCA
3) 在 TCA 处取相对位置均值和位置协方差
4) 投影到 B-plane 计算碰撞概率 Pc
5) 用 Pc 做风险分级
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from .. import config
except ImportError:
    import config

try:
    from .calculate_collision_probability import collision_probability_bplane
except ImportError:
    try:
        from ore.pipeline.calculate_collision_probability import collision_probability_bplane
    except ImportError:
        from calculate_collision_probability import collision_probability_bplane

MU = 3.986004418e14


# ===================== 用户配置 =====================
INPUT_CSV = config.RENDERS_DIR / "imm_ukf_estimation_result.csv"
OUT_CSV = INPUT_CSV.with_name("risk_prediction_result.csv")
HORIZON_SEC = 300.0
PROP_DT = 5.0
HARD_BODY_RADIUS_M = 10.0
PC_DANGER = config.PC_DANGER
PC_CAUTION = config.PC_CAUTION
# ===================================================


def nearest_spd(P: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    P = 0.5 * (P + P.T)
    eigvals, eigvecs = np.linalg.eigh(P)
    eigvals = np.maximum(eigvals, eps)
    return 0.5 * (eigvecs @ np.diag(eigvals) @ eigvecs.T + eigvecs @ np.diag(eigvals) @ eigvecs.T)


def estimate_mean_motion_from_truth(df: pd.DataFrame) -> Tuple[float, float]:
    r = np.array([df.loc[0, "rc_x_m"], df.loc[0, "rc_y_m"], df.loc[0, "rc_z_m"]], dtype=float)
    v = np.array([df.loc[0, "vc_x_mps"], df.loc[0, "vc_y_mps"], df.loc[0, "vc_z_mps"]], dtype=float)
    rnorm = np.linalg.norm(r)
    vnorm = np.linalg.norm(v)
    a = 1.0 / (2.0 / rnorm - vnorm**2 / MU)
    n = math.sqrt(MU / a**3)
    return a, n


def accel9_deriv(x: np.ndarray, n: float) -> np.ndarray:
    R, T, N, dR, dT, dN, aR, aT, aN = x
    ddR = 2.0 * n * dT + 3.0 * n**2 * R + aR
    ddT = -2.0 * n * dR + aT
    ddN = -n**2 * N + aN
    return np.array([dR, dT, dN, ddR, ddT, ddN, 0.0, 0.0, 0.0], dtype=float)


def rk4_step(x: np.ndarray, dt: float, n: float) -> np.ndarray:
    k1 = accel9_deriv(x, n)
    k2 = accel9_deriv(x + 0.5 * dt * k1, n)
    k3 = accel9_deriv(x + 0.5 * dt * k2, n)
    k4 = accel9_deriv(x + dt * k3, n)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def numerical_state_transition(x: np.ndarray, dt: float, n: float, eps: float = 1e-4) -> np.ndarray:
    nx = len(x)
    F = np.zeros((nx, nx), dtype=float)
    x0 = rk4_step(x, dt, n)
    for i in range(nx):
        xp = x.copy(); xp[i] += eps
        xm = x.copy(); xm[i] -= eps
        yp = rk4_step(xp, dt, n)
        ym = rk4_step(xm, dt, n)
        F[:, i] = (yp - ym) / (2.0 * eps)
    return F


def classify_risk_from_pc(pc: float) -> str:
    if pc >= PC_DANGER:
        return "danger"
    if pc >= PC_CAUTION:
        return "caution"
    return "safe"


def propagate_mean_cov_to_tca(x0: np.ndarray, P0: np.ndarray, n: float,
                              horizon_sec: float, dt: float) -> Dict[str, np.ndarray | float]:
    x = x0.copy()
    P = P0.copy()

    times = [0.0]
    xs = [x.copy()]
    Ps = [P.copy()]
    dists = [float(np.linalg.norm(x[:3]))]

    steps = int(horizon_sec / dt)
    # very small process noise only for propagation stage
    Qp = np.diag([0.2**2, 0.2**2, 0.2**2, 0.01**2, 0.01**2, 0.01**2, 1e-8, 1e-8, 1e-8])

    for k in range(steps):
        F = numerical_state_transition(x, dt, n)
        x = rk4_step(x, dt, n)
        P = F @ P @ F.T + Qp
        P = 0.5 * (P + P.T)
        times.append((k + 1) * dt)
        xs.append(x.copy())
        Ps.append(P.copy())
        dists.append(float(np.linalg.norm(x[:3])))

    idx = int(np.argmin(dists))
    return {
        "tca_rel_sec": float(times[idx]),
        "x_tca": xs[idx],
        "P_tca": Ps[idx],
        "min_dist_m": float(dists[idx]),
    }


def run_risk_prediction(df: pd.DataFrame, out_csv: Path | None = None) -> pd.DataFrame:
    _, n = estimate_mean_motion_from_truth(df)
    rows = []

    for _, row in df.iterrows():
        x0 = np.array([
            row["R_est_m"], row["T_est_m"], row["N_est_m"],
            row["dR_est_mps"], row["dT_est_mps"], row["dN_est_mps"],
            row.get("aR_est_mps2", 0.0), row.get("aT_est_mps2", 0.0), row.get("aN_est_mps2", 0.0),
        ], dtype=float)

        P0 = np.diag([
            float(row.get("P00", 50.0**2)),
            float(row.get("P11", 80.0**2)),
            float(row.get("P22", 50.0**2)),
            float(row.get("P33", 0.1**2)),
            float(row.get("P44", 0.1**2)),
            float(row.get("P55", 0.1**2)),
            float(row.get("P66", (2e-4)**2)),
            float(row.get("P77", (2e-4)**2)),
            float(row.get("P88", (2e-4)**2)),
        ])

        pred = propagate_mean_cov_to_tca(x0, P0, n, HORIZON_SEC, PROP_DT)
        x_tca = pred["x_tca"]
        P_tca = pred["P_tca"]

        mu_r = x_tca[:3]
        mu_v = x_tca[3:6]
        P_r = P_tca[:3, :3]

        pc_out = collision_probability_bplane(mu_r, mu_v, P_r, HARD_BODY_RADIUS_M)
        pc = float(pc_out["Pc"])
        risk_level = classify_risk_from_pc(pc)

        new_row = row.copy()
        new_row["min_dist_m"] = pred["min_dist_m"]
        new_row["pred_future_min_distance_m"] = pred["min_dist_m"]
        new_row["pred_future_tca_sec"] = float(row["t_sec"]) + pred["tca_rel_sec"]
        new_row["pc_bplane"] = pc
        new_row["risk_level"] = risk_level
        new_row["pred_risk_level"] = risk_level
        new_row["bplane_mu_x_m"] = float(pc_out["mu_bplane_2d"][0])
        new_row["bplane_mu_y_m"] = float(pc_out["mu_bplane_2d"][1])
        new_row["bplane_c00"] = float(pc_out["C_bplane_2d"][0, 0])
        new_row["bplane_c01"] = float(pc_out["C_bplane_2d"][0, 1])
        new_row["bplane_c11"] = float(pc_out["C_bplane_2d"][1, 1])
        rows.append(new_row)

    out = pd.DataFrame(rows)
    if out_csv is not None:
        out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return out


def main():
    df = pd.read_csv(INPUT_CSV)
    out = run_risk_prediction(df, OUT_CSV)
    print("saved:", OUT_CSV)
    print(out[["frame", "pred_future_min_distance_m", "pc_bplane", "pred_risk_level"]].head(10))


if __name__ == "__main__":
    main()
