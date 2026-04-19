from __future__ import annotations

# -*- coding: utf-8 -*-
"""
Core Orekit simulation helpers.

This version avoids `setup_orekit_curdir()` and explicitly registers the
Orekit data provider using an absolute path. It is therefore stable when the
script is launched from different working directories.
"""

import math
import os
import sys
from pathlib import Path
from typing import List, Dict

import pandas as pd
import orekit_jpype as orekit

# -----------------------------------------------------------------------------
# Project path bootstrap
# -----------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SIM_DIR = THIS_FILE.parent
PROJECT_ROOT = SIM_DIR.parent
for _p in [PROJECT_ROOT, SIM_DIR, PROJECT_ROOT / "pipeline"]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import config  # type: ignore
except Exception:
    config = None  # type: ignore

# -----------------------------------------------------------------------------
# Orekit VM + data registration
# -----------------------------------------------------------------------------
orekit.initVM()

from java.io import File  # type: ignore
from org.orekit.data import DataContext, DirectoryCrawler, ZipJarCrawler  # type: ignore
from org.hipparchus.ode.nonstiff import ClassicalRungeKuttaIntegrator  # type: ignore
from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
from org.orekit.frames import FramesFactory  # type: ignore
from org.orekit.utils import Constants, PVCoordinates, IERSConventions  # type: ignore
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
from org.orekit.orbits import KeplerianOrbit, CartesianOrbit, OrbitType, PositionAngleType  # type: ignore
from org.orekit.propagation import SpacecraftState  # type: ignore
from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel  # type: ignore
from org.orekit.forces.gravity.potential import GravityFieldFactory  # type: ignore


def _candidate_orekit_paths() -> list[Path]:
    candidates: list[Path] = []
    env_p = os.environ.get("OREKIT_DATA")
    if env_p:
        candidates.append(Path(env_p))
    if config is not None and getattr(config, "OREKIT_DATA", None):
        candidates.append(Path(config.OREKIT_DATA))
    candidates.extend([
        PROJECT_ROOT / "assets" / "orekit-data.zip",
        PROJECT_ROOT / "orekit-data.zip",
        Path.cwd() / "assets" / "orekit-data.zip",
        Path.cwd() / "orekit-data.zip",
    ])
    # de-duplicate preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        rp = str(Path(p).expanduser().resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(Path(rp))
    return out


def _setup_orekit_data() -> Path:
    candidates = _candidate_orekit_paths()
    chosen: Path | None = None
    for p in candidates:
        if p.exists():
            chosen = p
            break
    if chosen is None:
        raise FileNotFoundError(
            "Orekit data not found. Checked: " + ", ".join(str(p) for p in candidates)
        )

    manager = DataContext.getDefault().getDataProvidersManager()
    manager.clearProviders()

    jf = File(str(chosen))
    if chosen.is_dir():
        manager.addProvider(DirectoryCrawler(jf))
    else:
        manager.addProvider(ZipJarCrawler(jf))
    return chosen


OREKIT_DATA_PATH = _setup_orekit_data()

UTC = TimeScalesFactory.getUTC()
EME2000 = FramesFactory.getEME2000()
ITRF = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

MU = Constants.WGS84_EARTH_MU
G0 = Constants.G0_STANDARD_GRAVITY


def build_numerical_propagator(
    initial_state: SpacecraftState,
    integ_step_s: float = 1.0,
    gravity_degree: int = 8,
    gravity_order: int = 8,
) -> NumericalPropagator:
    integrator = ClassicalRungeKuttaIntegrator(integ_step_s)
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)

    gravity_provider = GravityFieldFactory.getNormalizedProvider(gravity_degree, gravity_order)
    gravity_model = HolmesFeatherstoneAttractionModel(ITRF, gravity_provider)
    propagator.addForceModel(gravity_model)
    return propagator


def rtn_basis_from_state(state: SpacecraftState):
    pv = state.getPVCoordinates()
    r = pv.getPosition()
    v = pv.getVelocity()

    r_hat = r.normalize()
    n_hat = Vector3D.crossProduct(r, v).normalize()
    t_hat = Vector3D.crossProduct(n_hat, r_hat)
    return r_hat, t_hat, n_hat


def vector_from_rtn(r_hat, t_hat, n_hat, comps):
    return (
        r_hat.scalarMultiply(comps[0])
        .add(t_hat.scalarMultiply(comps[1]))
        .add(n_hat.scalarMultiply(comps[2]))
    )


def offset_state_in_rtn(
    reference_state: SpacecraftState,
    dr_rtn_m,
    dv_rtn_mps,
    mass_kg: float,
    mu: float = MU,
) -> SpacecraftState:
    r_hat, t_hat, n_hat = rtn_basis_from_state(reference_state)

    ref_pv = reference_state.getPVCoordinates()
    r_ref = ref_pv.getPosition()
    v_ref = ref_pv.getVelocity()

    dr_eci = vector_from_rtn(r_hat, t_hat, n_hat, dr_rtn_m)
    dv_eci = vector_from_rtn(r_hat, t_hat, n_hat, dv_rtn_mps)

    r_new = r_ref.add(dr_eci)
    v_new = v_ref.add(dv_eci)

    orbit_new = CartesianOrbit(PVCoordinates(r_new, v_new), EME2000, reference_state.getDate(), mu)
    return SpacecraftState(orbit_new, mass_kg)


def apply_impulse_rtn(
    state: SpacecraftState,
    dv_rtn_mps,
    isp_s: float = 220.0,
    mu: float = MU,
) -> SpacecraftState:
    r_hat, t_hat, n_hat = rtn_basis_from_state(state)

    pv = state.getPVCoordinates()
    r = pv.getPosition()
    v = pv.getVelocity()

    dv_eci = vector_from_rtn(r_hat, t_hat, n_hat, dv_rtn_mps)
    v_new = v.add(dv_eci)

    dv_norm = math.sqrt(dv_rtn_mps[0] ** 2 + dv_rtn_mps[1] ** 2 + dv_rtn_mps[2] ** 2)

    m0 = state.getMass()
    if dv_norm > 0.0 and isp_s is not None and isp_s > 0.0:
        mf = m0 / math.exp(dv_norm / (isp_s * G0))
    else:
        mf = m0

    orbit_new = CartesianOrbit(PVCoordinates(r, v_new), EME2000, state.getDate(), mu)
    return SpacecraftState(orbit_new, mf)


def to_rtn_relative(chaser_state: SpacecraftState, target_state: SpacecraftState):
    ch_pv = chaser_state.getPVCoordinates()
    tg_pv = target_state.getPVCoordinates()

    rc = ch_pv.getPosition()
    vc = ch_pv.getVelocity()
    rt = tg_pv.getPosition()
    vt = tg_pv.getVelocity()

    rel_r = rt.subtract(rc)
    rel_v = vt.subtract(vc)

    r_hat = rc.normalize()
    n_hat = Vector3D.crossProduct(rc, vc).normalize()
    t_hat = Vector3D.crossProduct(n_hat, r_hat)

    R = Vector3D.dotProduct(rel_r, r_hat)
    T = Vector3D.dotProduct(rel_r, t_hat)
    N = Vector3D.dotProduct(rel_r, n_hat)

    dR = Vector3D.dotProduct(rel_v, r_hat)
    dT = Vector3D.dotProduct(rel_v, t_hat)
    dN = Vector3D.dotProduct(rel_v, n_hat)

    return R, T, N, dR, dT, dN


def make_default_chaser_state(
    start_date: AbsoluteDate,
    a_m: float = 7000e3,
    e: float = 0.001,
    inc_deg: float = 97.5,
    argp_deg: float = 10.0,
    raan_deg: float = 25.0,
    ta_deg: float = 0.0,
    mass_kg: float = 250.0,
) -> SpacecraftState:
    orbit = KeplerianOrbit(
        a_m,
        e,
        math.radians(inc_deg),
        math.radians(argp_deg),
        math.radians(raan_deg),
        math.radians(ta_deg),
        PositionAngleType.TRUE,
        EME2000,
        start_date,
        MU,
    )
    return SpacecraftState(orbit, mass_kg)


def propagate_piecewise_with_impulses(
    initial_state: SpacecraftState,
    start_date: AbsoluteDate,
    duration_s: float,
    output_step_s: float,
    maneuvers: List[Dict] | None = None,
    integ_step_s: float = 1.0,
):
    maneuvers = maneuvers or []
    maneuvers = sorted(maneuvers, key=lambda x: x["t"])

    output_times = []
    t = 0.0
    while t <= duration_s + 1e-9:
        output_times.append(t)
        t += output_step_s

    results = {}
    current_state = initial_state
    next_output_idx = 0

    all_breaks = maneuvers + [{"t": duration_s, "dv_rtn": None, "isp": None, "label": "END"}]

    for event in all_breaks:
        segment_end_t = event["t"]
        segment_end_date = start_date.shiftedBy(segment_end_t)

        propagator = build_numerical_propagator(current_state, integ_step_s=integ_step_s)

        while next_output_idx < len(output_times) and output_times[next_output_idx] < segment_end_t - 1e-9:
            out_t = output_times[next_output_idx]
            out_date = start_date.shiftedBy(out_t)
            state = propagator.propagate(out_date)
            results[out_t] = state
            next_output_idx += 1

        state_at_break = propagator.propagate(segment_end_date)

        is_output_on_break = (
            next_output_idx < len(output_times)
            and abs(output_times[next_output_idx] - segment_end_t) < 1e-9
        )

        if event["dv_rtn"] is not None:
            current_state = apply_impulse_rtn(state_at_break, event["dv_rtn"], event.get("isp", 220.0))
        else:
            current_state = state_at_break

        if is_output_on_break:
            results[output_times[next_output_idx]] = current_state
            next_output_idx += 1

    ordered_times = sorted(results.keys())
    ordered_states = [results[t] for t in ordered_times]
    return ordered_times, ordered_states


def add_future_labels(
    df: pd.DataFrame,
    window_sec: float = 300.0,
    danger_thr_m: float = 200.0,
    caution_thr_m: float = 1000.0,
    *,
    future_window_s: float | None = None,
) -> pd.DataFrame:
    if future_window_s is not None:
        window_sec = float(future_window_s)

    if len(df) < 2:
        out = df.copy()
        out["future_min_distance_m"] = out.get("relative_distance_m", pd.Series([math.nan] * len(out)))
        out["future_tca_sec"] = out.get("t_sec", pd.Series([math.nan] * len(out)))
        out["risk_level"] = "unknown"
        return out

    out = df.copy()
    dt = float(out["t_sec"].iloc[1] - out["t_sec"].iloc[0])
    window_n = max(1, int(round(window_sec / max(dt, 1e-9))))

    dist = out["relative_distance_m"].to_numpy(dtype=float)
    time = out["t_sec"].to_numpy(dtype=float)

    future_min_dist: list[float] = []
    future_tca: list[float] = []
    risk_level: list[str] = []

    for i in range(len(out)):
        j_end = min(len(out), i + window_n + 1)
        sub_dist = dist[i:j_end]
        sub_time = time[i:j_end]

        idx = int(sub_dist.argmin())
        min_d = float(sub_dist[idx])
        tca = float(sub_time[idx])

        future_min_dist.append(min_d)
        future_tca.append(tca)

        if min_d < danger_thr_m:
            risk_level.append("danger")
        elif min_d < caution_thr_m:
            risk_level.append("caution")
        else:
            risk_level.append("safe")

    out["future_min_distance_m"] = future_min_dist
    out["future_tca_sec"] = future_tca
    out["risk_level"] = risk_level
    return out


def scenario_to_dataframe(
    start_date: AbsoluteDate,
    chaser_state0: SpacecraftState,
    target_state0: SpacecraftState,
    target_maneuvers: List[Dict],
    duration_s: float = 1800.0,
    output_step_s: float = 5.0,
    future_window_sec: float = 300.0,
) -> pd.DataFrame:
    t_ch, chaser_states = propagate_piecewise_with_impulses(
        chaser_state0, start_date, duration_s, output_step_s, maneuvers=[]
    )
    t_tg, target_states = propagate_piecewise_with_impulses(
        target_state0, start_date, duration_s, output_step_s, maneuvers=target_maneuvers
    )

    rows = []
    man_events = [(m["t"], m.get("label", "maneuver")) for m in target_maneuvers]

    for t, sc, st in zip(t_ch, chaser_states, target_states):
        ch_pv = sc.getPVCoordinates()
        tg_pv = st.getPVCoordinates()

        rc = ch_pv.getPosition()
        vc = ch_pv.getVelocity()
        rt = tg_pv.getPosition()
        vt = tg_pv.getVelocity()

        R, T, N, dR, dT, dN = to_rtn_relative(sc, st)
        dist = rt.subtract(rc).getNorm()

        maneuver_flag = 0
        maneuver_type = "none"
        for mt, mlab in man_events:
            if abs(t - mt) < output_step_s / 2.0 + 1e-9:
                maneuver_flag = 1
                maneuver_type = mlab
                break

        rows.append({
            "t_sec": t,
            "rc_x_m": rc.getX(), "rc_y_m": rc.getY(), "rc_z_m": rc.getZ(),
            "vc_x_mps": vc.getX(), "vc_y_mps": vc.getY(), "vc_z_mps": vc.getZ(),
            "rt_x_m": rt.getX(), "rt_y_m": rt.getY(), "rt_z_m": rt.getZ(),
            "vt_x_mps": vt.getX(), "vt_y_mps": vt.getY(), "vt_z_mps": vt.getZ(),
            "rel_R_m": R, "rel_T_m": T, "rel_N_m": N,
            "rel_dR_mps": dR, "rel_dT_mps": dT, "rel_dN_mps": dN,
            "relative_distance_m": dist,
            "target_mass_kg": st.getMass(),
            "maneuver_flag": maneuver_flag,
            "maneuver_type": maneuver_type,
        })

    df = pd.DataFrame(rows)
    return add_future_labels(df, window_sec=future_window_sec)
