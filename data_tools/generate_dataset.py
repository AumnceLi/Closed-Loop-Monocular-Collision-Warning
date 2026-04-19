# -*- coding: utf-8 -*-
from __future__ import annotations

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

SEED = 42
random.seed(SEED)

OUT_DIR = config.DATASET_RTN_DIR
OUT_DIR.mkdir(exist_ok=True, parents=True)

TRAIN_N = 100
VAL_N = 20
TEST_N = 20

DURATION_S = 1800.0
OUTPUT_STEP_S = 5.0
FUTURE_WINDOW_S = 300.0
MAX_ATTEMPTS_PER_SCENARIO = 3000

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


def _visible_half_span(t_forward_m: float, margin_ratio: float = 0.60) -> float:
    return max(20.0, float(t_forward_m) * float(__import__("math").tan(__import__("math").radians(float(config.FOV_X_DEG)) * 0.5)) * margin_ratio)


def random_initial_offset() -> tuple[list[float], list[float]]:
    t_forward = random.uniform(1500.0, 5000.0)
    r_lim = min(300.0, _visible_half_span(t_forward, margin_ratio=0.58))
    n_lim = min(200.0, _visible_half_span(t_forward, margin_ratio=0.58))

    dr_rtn = [
        random.uniform(-r_lim, r_lim),
        t_forward,
        random.uniform(-n_lim, n_lim),
    ]
    dv_rtn = [
        random.uniform(-0.05, 0.05),
        random.uniform(-0.05, 0.05),
        random.uniform(-0.02, 0.02),
    ]
    return dr_rtn, dv_rtn


def build_maneuvers(scenario_type: str) -> list[dict]:
    if scenario_type == "no_maneuver":
        return []

    if scenario_type == "impulse_approach":
        t0 = random.uniform(400.0, 900.0)
        dv = [0.0, random.uniform(-0.12, -0.03), 0.0]
        return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "impulse_approach"}]

    if scenario_type == "departure":
        t0 = random.uniform(400.0, 900.0)
        dv = [0.0, random.uniform(0.03, 0.12), 0.0]
        return [{"t": t0, "dv_rtn": dv, "isp": 220.0, "label": "departure"}]

    if scenario_type == "continuous_approach":
        t_start = random.uniform(300.0, 700.0)
        duration = random.uniform(120.0, 300.0)
        dt = 10.0
        n = int(duration / dt)
        small_dv_t = random.uniform(-0.006, -0.0015)

        mans = []
        for k in range(n):
            mans.append({
                "t": t_start + k * dt,
                "dv_rtn": [0.0, small_dv_t, 0.0],
                "isp": 220.0,
                "label": "continuous_approach",
            })
        return mans

    raise ValueError(f"unknown scenario_type: {scenario_type}")


def generate_one_candidate(split: str, scenario_type: str) -> tuple[pd.DataFrame, dict]:
    start_date = AbsoluteDate(2026, 3, 24, 0, 0, 0.0, UTC)

    chaser_state0 = make_default_chaser_state(
        start_date=start_date,
        a_m=7000e3 + random.uniform(-20e3, 20e3),
        e=random.uniform(0.0005, 0.0030),
        inc_deg=97.5 + random.uniform(-1.0, 1.0),
        argp_deg=random.uniform(0.0, 30.0),
        raan_deg=random.uniform(0.0, 60.0),
        ta_deg=random.uniform(0.0, 10.0),
        mass_kg=250.0,
    )

    dr_rtn, dv_rtn = random_initial_offset()
    target_state0 = offset_state_in_rtn(
        chaser_state0,
        dr_rtn_m=dr_rtn,
        dv_rtn_mps=dv_rtn,
        mass_kg=180.0,
    )
    maneuvers = build_maneuvers(scenario_type)

    df = scenario_to_dataframe(
        start_date=start_date,
        chaser_state0=chaser_state0,
        target_state0=target_state0,
        target_maneuvers=maneuvers,
        duration_s=DURATION_S,
        output_step_s=OUTPUT_STEP_S,
        future_window_sec=FUTURE_WINDOW_S,
    )

    meta = {
        "split": split,
        "scenario_type": scenario_type,
        "min_distance_m": float(df["relative_distance_m"].min()),
        "min_future_distance_m": float(df["future_min_distance_m"].min()),
        "risk_level_counts": dict(df["risk_level"].value_counts()),
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


def _save_final_sample(df: pd.DataFrame, split: str, scenario_type: str, sample_id: int) -> Path:
    out_path = OUT_DIR / split / scenario_type
    out_path.mkdir(exist_ok=True, parents=True)
    csv_path = out_path / f"sample_{sample_id:05d}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def generate_split(split: str, n_each: int, start_id: int) -> tuple[list[dict], int]:
    scenario_types = ["no_maneuver", "impulse_approach", "continuous_approach", "departure"]

    metas: list[dict] = []
    sid = start_id
    for stype in scenario_types:
        accepted = 0
        attempts = 0
        while accepted < n_each and attempts < MAX_ATTEMPTS_PER_SCENARIO:
            attempts += 1
            df, meta = generate_one_candidate(split, stype)

            tmp_csv = OUT_DIR / f"_screen_{stype}_{split}_{sid:05d}.csv"
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
                    print(f"[{split}][{stype}] attempts={attempts}, accepted={accepted}, last_reject={screen.reason}")
                continue

            df["sample_id"] = sid
            df["split"] = split
            df["scenario_type"] = stype
            csv_path = _save_final_sample(df, split, stype, sid)

            meta.update({
                "sample_id": sid,
                "csv_path": str(csv_path),
                "screen_stage": screen.stage,
                "screen_reason": screen.reason,
                **screen.metrics,
            })
            metas.append(meta)
            accepted += 1
            print(
                f"[{split}][{stype}] accepted {accepted:03d}/{n_each} "
                f"(sample_id={sid:05d}, visible_frames={meta['visible_frames']}, "
                f"streak={meta['longest_visible_streak']}, first={meta['first_visible_frame']})"
            )
            sid += 1

        if accepted < n_each:
            print(f"WARNING: [{split}][{stype}] only got {accepted}/{n_each} after {attempts} attempts")
    return metas, sid


def main() -> None:
    all_meta: list[dict] = []
    sid = 0

    metas, sid = generate_split("train", TRAIN_N, sid)
    all_meta.extend(metas)

    metas, sid = generate_split("val", VAL_N, sid)
    all_meta.extend(metas)

    metas, sid = generate_split("test", TEST_N, sid)
    all_meta.extend(metas)

    meta_df = pd.DataFrame(all_meta)
    out_csv = OUT_DIR / "index.csv"
    meta_df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
