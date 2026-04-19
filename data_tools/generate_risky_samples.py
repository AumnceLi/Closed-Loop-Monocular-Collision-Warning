# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import pandas as pd

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1] if THIS_FILE.parent.name == "data_tools" else THIS_FILE.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

for _p in [PROJECT_ROOT, PROJECT_ROOT / "sim", PROJECT_ROOT / "pipeline", WORKSPACE_ROOT]:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

try:
    import config  # type: ignore
except ImportError:
    from ore import config  # type: ignore

if getattr(config, "OREKIT_DATA", None):
    os.environ.setdefault("OREKIT_DATA", str(Path(config.OREKIT_DATA).resolve()))

try:
    from sim.core_orekit_sim import (  # type: ignore
        UTC,
        AbsoluteDate,
        make_default_chaser_state,
        offset_state_in_rtn,
        scenario_to_dataframe,
    )
except ImportError:
    try:
        from ore.sim.core_orekit_sim import (  # type: ignore
            UTC,
            AbsoluteDate,
            make_default_chaser_state,
            offset_state_in_rtn,
            scenario_to_dataframe,
        )
    except ImportError as exc:
        raise ImportError(
            "Failed to import core_orekit_sim. "
            "Check that orekit_jpype is installed, Java is available, and assets/orekit-data.zip exists."
        ) from exc

from dataset_visibility_filter import (  # noqa: E402
    VisibilityCriteria,
    cleanup_path,
    criteria_to_meta,
    screen_truth_dataframe_for_monocular_use,
)

SEED = 123
random.seed(SEED)

OUT_DIR = config.DATASET_RISKY_DIR
OUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_PER_RISK = 50
VAL_PER_RISK = 10
TEST_PER_RISK = 10

DURATION_S = 1800.0
OUTPUT_STEP_S = 5.0
FUTURE_WINDOW_S = 300.0
MAX_ATTEMPTS_PER_BUCKET = 5000
RISK_LEVELS = ["safe", "caution", "danger"]

SCREEN = VisibilityCriteria(
    min_est_visible_frames=8,
    min_est_longest_streak=6,
    min_visible_frames=12,
    min_longest_visible_streak=8,
    max_first_visible_frame=120,
    min_mean_bbox_area_px=36.0,
    min_inside_safe_ratio_visible=0.35,
    max_edge_touch_ratio_visible=0.70,
)


def _visible_half_span(t_forward_m: float, margin_ratio: float) -> float:
    return max(20.0, float(t_forward_m) * math.tan(math.radians(float(config.FOV_X_DEG)) * 0.5) * margin_ratio)


def trajectory_risk_from_df(df: pd.DataFrame) -> tuple[str, float]:
    dmin = float(df["future_min_distance_m"].min())
    if dmin < 200.0:
        return "danger", dmin
    if dmin < 1000.0:
        return "caution", dmin
    return "safe", dmin


def random_chaser_state(start_date):
    return make_default_chaser_state(
        start_date=start_date,
        a_m=7000e3 + random.uniform(-20e3, 20e3),
        e=random.uniform(0.0005, 0.0030),
        inc_deg=97.5 + random.uniform(-1.0, 1.0),
        argp_deg=random.uniform(0.0, 30.0),
        raan_deg=random.uniform(0.0, 60.0),
        ta_deg=random.uniform(0.0, 10.0),
        mass_kg=250.0,
    )


def build_target_init(chaser_state0, target_risk: str):
    if target_risk == "safe":
        t_forward = random.uniform(3000.0, 7000.0)
        r_lim = min(500.0, _visible_half_span(t_forward, margin_ratio=0.72))
        n_lim = min(400.0, _visible_half_span(t_forward, margin_ratio=0.72))
        dr_rtn = [random.uniform(-r_lim, r_lim), t_forward, random.uniform(-n_lim, n_lim)]
        dv_rtn = [random.uniform(-0.03, 0.03), random.uniform(-0.03, 0.03), random.uniform(-0.02, 0.02)]
    elif target_risk == "caution":
        t_forward = random.uniform(1200.0, 2500.0)
        r_lim = min(250.0, _visible_half_span(t_forward, margin_ratio=0.62))
        n_lim = min(150.0, _visible_half_span(t_forward, margin_ratio=0.62))
        dr_rtn = [random.uniform(-r_lim, r_lim), t_forward, random.uniform(-n_lim, n_lim)]
        dv_rtn = [random.uniform(-0.03, 0.03), random.uniform(-0.04, 0.01), random.uniform(-0.01, 0.01)]
    elif target_risk == "danger":
        t_forward = random.uniform(300.0, 1200.0)
        r_lim = min(100.0, _visible_half_span(t_forward, margin_ratio=0.52))
        n_lim = min(80.0, _visible_half_span(t_forward, margin_ratio=0.52))
        dr_rtn = [random.uniform(-r_lim, r_lim), t_forward, random.uniform(-n_lim, n_lim)]
        dv_rtn = [random.uniform(-0.02, 0.02), random.uniform(-0.06, 0.00), random.uniform(-0.01, 0.01)]
    else:
        raise ValueError(f"unknown target_risk: {target_risk}")

    target_state0 = offset_state_in_rtn(
        chaser_state0,
        dr_rtn_m=dr_rtn,
        dv_rtn_mps=dv_rtn,
        mass_kg=180.0,
    )
    return target_state0, dr_rtn, dv_rtn


def build_maneuvers_for_risk(target_risk: str):
    if target_risk == "safe":
        choice = random.choice(["no_maneuver", "departure", "weak_departure"])
        if choice == "no_maneuver":
            return [], "no_maneuver"
        if choice == "departure":
            t0 = random.uniform(300.0, 900.0)
            dv = [0.0, random.uniform(0.04, 0.12), 0.0]
            return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "departure"}], "departure"
        t0 = random.uniform(300.0, 900.0)
        dv = [0.0, random.uniform(0.02, 0.05), 0.0]
        return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "weak_departure"}], "weak_departure"

    if target_risk == "caution":
        choice = random.choice(["impulse_approach", "continuous_approach"])
        if choice == "impulse_approach":
            t0 = random.uniform(250.0, 800.0)
            dv = [0.0, random.uniform(-0.08, -0.03), 0.0]
            return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "impulse_approach"}], "impulse_approach"
        t_start = random.uniform(200.0, 700.0)
        duration = random.uniform(80.0, 220.0)
        dt = 10.0
        n = int(duration / dt)
        small_dv_t = random.uniform(-0.0045, -0.0015)
        mans = []
        for k in range(n):
            mans.append({
                "t": t_start + k * dt,
                "dv_rtn": [0.0, small_dv_t, 0.0],
                "isp": 220.0,
                "label": "continuous_approach",
            })
        return mans, "continuous_approach"

    if target_risk == "danger":
        choice = random.choice(["strong_impulse_approach", "strong_continuous_approach"])
        if choice == "strong_impulse_approach":
            t0 = random.uniform(150.0, 500.0)
            dv = [0.0, random.uniform(-0.18, -0.08), 0.0]
            return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "strong_impulse_approach"}], "strong_impulse_approach"
        t_start = random.uniform(120.0, 400.0)
        duration = random.uniform(150.0, 350.0)
        dt = 10.0
        n = int(duration / dt)
        small_dv_t = random.uniform(-0.008, -0.004)
        mans = []
        for k in range(n):
            mans.append({
                "t": t_start + k * dt,
                "dv_rtn": [0.0, small_dv_t, 0.0],
                "isp": 220.0,
                "label": "strong_continuous_approach",
            })
        return mans, "strong_continuous_approach"

    raise ValueError(f"unknown target_risk: {target_risk}")


def generate_one_candidate(target_risk: str):
    start_date = AbsoluteDate(2026, 3, 24, 0, 0, 0.0, UTC)

    chaser_state0 = random_chaser_state(start_date)
    target_state0, dr_rtn, dv_rtn = build_target_init(chaser_state0, target_risk)
    maneuvers, scenario_type = build_maneuvers_for_risk(target_risk)

    df = scenario_to_dataframe(
        start_date=start_date,
        chaser_state0=chaser_state0,
        target_state0=target_state0,
        target_maneuvers=maneuvers,
        duration_s=DURATION_S,
        output_step_s=OUTPUT_STEP_S,
        future_window_sec=FUTURE_WINDOW_S,
    )

    traj_risk, traj_min_future_dist = trajectory_risk_from_df(df)
    meta = {
        "scenario_type": scenario_type,
        "target_risk": target_risk,
        "generated_risk": traj_risk,
        "traj_min_future_dist_m": traj_min_future_dist,
        "init_dr_R_m": dr_rtn[0],
        "init_dr_T_m": dr_rtn[1],
        "init_dr_N_m": dr_rtn[2],
        "init_dv_R_mps": dv_rtn[0],
        "init_dv_T_mps": dv_rtn[1],
        "init_dv_N_mps": dv_rtn[2],
        "num_maneuvers": len(maneuvers),
        **criteria_to_meta(SCREEN),
    }
    return df, meta


def save_sample(df: pd.DataFrame, split: str, risk_level: str, sample_id: int) -> Path:
    out_dir = OUT_DIR / split / risk_level
    out_dir.mkdir(exist_ok=True, parents=True)
    csv_path = out_dir / f"sample_{sample_id:05d}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_bucket(split: str, risk_level: str, target_count: int, start_id: int) -> tuple[list[dict], int]:
    accepted: list[dict] = []
    sample_id = start_id
    attempts = 0

    while len(accepted) < target_count and attempts < MAX_ATTEMPTS_PER_BUCKET:
        attempts += 1
        df, meta = generate_one_candidate(risk_level)
        if meta["generated_risk"] != risk_level:
            continue

        tmp_csv = OUT_DIR / f"_screen_{risk_level}_{split}_{sample_id:05d}.csv"
        try:
            df.to_csv(tmp_csv, index=False)
            screen = screen_truth_dataframe_for_monocular_use(
                df,
                truth_csv=tmp_csv,
                config_module=config,
                criteria=SCREEN,
            )
        finally:
            cleanup_path(tmp_csv)

        if not screen.accepted:
            if attempts % 50 == 0:
                print(f"[{split}][{risk_level}] attempts={attempts}, accepted={len(accepted)}, last_reject={screen.reason}")
            continue

        csv_path = save_sample(df, split, risk_level, sample_id)
        meta["sample_id"] = sample_id
        meta["split"] = split
        meta["csv_path"] = str(csv_path)
        meta["screen_stage"] = screen.stage
        meta["screen_reason"] = screen.reason
        meta.update(screen.metrics)
        accepted.append(meta)

        print(
            f"[{split}][{risk_level}] accepted {len(accepted):03d}/{target_count} "
            f"(sample_id={sample_id}, min_future_dist={meta['traj_min_future_dist_m']:.2f} m, "
            f"visible_frames={meta['visible_frames']}, streak={meta['longest_visible_streak']}, "
            f"scenario={meta['scenario_type']})"
        )
        sample_id += 1

    if len(accepted) < target_count:
        print(f"WARNING: [{split}][{risk_level}] only got {len(accepted)}/{target_count} after {attempts} attempts")

    return accepted, sample_id


def main() -> None:
    all_meta: list[dict] = []
    sample_id = 0

    split_plan = {"train": TRAIN_PER_RISK, "val": VAL_PER_RISK, "test": TEST_PER_RISK}
    for split, n_each in split_plan.items():
        for risk_level in RISK_LEVELS:
            metas, sample_id = generate_bucket(split, risk_level, n_each, sample_id)
            all_meta.extend(metas)

    meta_df = pd.DataFrame(all_meta)
    index_csv = OUT_DIR / "index.csv"
    meta_df.to_csv(index_csv, index=False)
    print(f"\n已保存索引: {index_csv}")
    if len(meta_df) > 0:
        print(meta_df.groupby(["split", "generated_risk"]).size())


if __name__ == "__main__":
    main()
