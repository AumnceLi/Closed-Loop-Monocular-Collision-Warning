# -*- coding: utf-8 -*-
"""
calculate_collision_probability.py

基于 B-plane 的短时碰撞概率计算。
输入：TCA 时刻的相对位置均值 mu_r、相对速度 mu_v、相对位置协方差 P_r
输出：B-plane 2D 均值/协方差与碰撞概率 Pc
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def make_bplane_basis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    vnorm = np.linalg.norm(v)
    if vnorm < 1e-9:
        # 退化处理
        e3 = np.array([1.0, 0.0, 0.0])
    else:
        e3 = v / vnorm

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ref, e3)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0])

    e1 = np.cross(e3, ref)
    e1 = e1 / max(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(e3, e1)
    e2 = e2 / max(np.linalg.norm(e2), 1e-12)
    B = np.vstack([e1, e2, e3])
    return B


def project_to_bplane(mu_r: np.ndarray, mu_v: np.ndarray, P_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    B = make_bplane_basis(mu_v)
    mu_b = B @ np.asarray(mu_r, dtype=float)
    P_b = B @ np.asarray(P_r, dtype=float) @ B.T
    mu_2d = mu_b[:2]
    C_2d = 0.5 * (P_b[:2, :2] + P_b[:2, :2].T)
    # regularize
    eigvals, eigvecs = np.linalg.eigh(C_2d)
    eigvals = np.maximum(eigvals, 1e-10)
    C_2d = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return B, mu_2d, C_2d


def gaussian_pdf_2d(x: np.ndarray, mu: np.ndarray, C: np.ndarray) -> float:
    d = x - mu
    invC = np.linalg.inv(C)
    detC = max(np.linalg.det(C), 1e-20)
    return float(math.exp(-0.5 * d.T @ invC @ d) / (2.0 * math.pi * math.sqrt(detC)))


def integrate_probability_over_disk(mu_2d: np.ndarray, C_2d: np.ndarray, radius: float,
                                    nr: int = 60, nth: int = 180) -> float:
    """极坐标数值积分。"""
    rs = np.linspace(0.0, radius, nr)
    thetas = np.linspace(0.0, 2.0 * math.pi, nth, endpoint=False)
    dr = rs[1] - rs[0] if nr > 1 else radius
    dth = thetas[1] - thetas[0] if nth > 1 else 2.0 * math.pi

    total = 0.0
    for r in rs:
        for th in thetas:
            x = np.array([r * math.cos(th), r * math.sin(th)], dtype=float)
            total += gaussian_pdf_2d(x, mu_2d, C_2d) * r * dr * dth
    return float(min(max(total, 0.0), 1.0))


def collision_probability_bplane(mu_r: np.ndarray, mu_v: np.ndarray, P_r: np.ndarray,
                                 hard_body_radius: float,
                                 nr: int = 60, nth: int = 180) -> Dict[str, np.ndarray | float]:
    B, mu_2d, C_2d = project_to_bplane(mu_r, mu_v, P_r)
    pc = integrate_probability_over_disk(mu_2d, C_2d, hard_body_radius, nr=nr, nth=nth)
    return {
        "B": B,
        "mu_bplane_2d": mu_2d,
        "C_bplane_2d": C_2d,
        "Pc": pc,
    }


if __name__ == "__main__":
    mu_r = np.array([30.0, -10.0, 2.0])
    mu_v = np.array([0.5, 7.2, 0.1])
    P_r = np.diag([50.0**2, 80.0**2, 30.0**2])
    out = collision_probability_bplane(mu_r, mu_v, P_r, hard_body_radius=10.0)
    print("Pc:", out["Pc"])
    print("mu_bplane_2d:", out["mu_bplane_2d"])
