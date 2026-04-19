# -*- coding: utf-8 -*-

# @Time : 2026/4/19 19:21

# @Author : Aumnce
# @Email : 1270888213@qq.com
# @File : fig4.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# =========================================================
# IEEE / journal-style plotting configuration
# =========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PALETTE = {
    "planning_skipped": "#dbe9f6",
    "planner_rejected": "#f8d7b5",
    "accepted_but_vetoed": "#d7efd7",
    "executed": "#4c78a8",
    "executed_dark": "#2f5f98",
    "arrow": "#5b6470",
    "grid": "#c7ced6",
    "box_fill": "#eef3f8",
    "median": "#2f5f98",
}

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(r"D:\robot_view\ore\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\safe-batch_validation_summary.csv")
CAUTION_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\caution-batch_validation_summary.csv")
DANGER_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\danger-batch_validation_summary.csv")

# =========================================================
# Metric setting
# =========================================================
# 你要换指标，改这里就行
METRIC_COL = "selected_pred_min_safe_margin_px"
Y_LABEL = "Selected Predicted Min Safe Margin (px)"
OUT_STEM = "decision_boxplot_final"


# =========================================================
# Helpers
# =========================================================
def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def normalize_subset_name(value) -> str:
    return norm_text(value)


def filter_subset_rows(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    out = df.copy()
    if "dataset_subset_label" in out.columns:
        want = normalize_subset_name(subset_name)
        got = out["dataset_subset_label"].map(normalize_subset_name)
        out = out[got == want].copy()
    return out.reset_index(drop=True)


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.32, color=PALETTE["grid"])
    ax.tick_params(direction="in", length=2.8, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def find_csv_with_columns(path_or_dir: Path, required_cols: Iterable[str]) -> Path:
    required_cols = list(required_cols)

    if path_or_dir.is_file():
        df = pd.read_csv(path_or_dir)
        cols = set(df.columns)
        missing = [c for c in required_cols if c not in cols]
        if missing:
            raise ValueError(f"{path_or_dir} is missing required columns: {missing}")
        if len(df) == 0:
            raise ValueError(f"{path_or_dir} has the required columns but contains 0 rows.")
        return path_or_dir

    if not path_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {path_or_dir}")

    csv_files = list(path_or_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {path_or_dir}")

    candidates = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            cols = set(df.columns)
            if all(col in cols for col in required_cols):
                row_count = len(df)
                depth = len(csv_path.relative_to(path_or_dir).parts)

                name = csv_path.name.lower()
                penalty = 0
                if "failed" in name:
                    penalty += 6
                if "warning" in name:
                    penalty += 5
                if "trigger" in name:
                    penalty += 4
                if "truth" in name:
                    penalty += 4
                if "sample_" in str(csv_path).lower():
                    penalty += 3
                if "audit" in str(csv_path).lower():
                    penalty += 2

                candidates.append({
                    "row_count": row_count,
                    "depth": depth,
                    "penalty": penalty,
                    "path": csv_path,
                })
        except Exception:
            continue

    if not candidates:
        raise ValueError(
            f"Could not find a CSV under {path_or_dir} containing all required columns: {required_cols}"
        )

    candidates.sort(
        key=lambda x: (-x["row_count"], x["depth"], x["penalty"], str(x["path"]).lower())
    )

    print("[INFO] Candidate CSVs ranked by suitability:")
    for cand in candidates[:10]:
        print(
            f"  rows={cand['row_count']:>3d}, "
            f"depth={cand['depth']}, "
            f"penalty={cand['penalty']}, "
            f"path={cand['path']}"
        )

    best = candidates[0]["path"]
    best_df = pd.read_csv(best)
    if len(best_df) == 0:
        raise ValueError(f"Best-matching CSV is empty: {best}")

    return best


def load_csv(
    path_or_dir: Path,
    required_cols: Iterable[str],
    numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    csv_path = find_csv_with_columns(path_or_dir, required_cols)
    print(f"[INFO] Using CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    print(f"[INFO] Loaded rows: {len(df)}")

    if len(df) == 0:
        raise ValueError(f"Loaded CSV has 0 rows: {csv_path}")

    if numeric_cols is not None:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# Stage classification
# =========================================================
def classify_stage(row: pd.Series) -> str:
    action = norm_text(row.get("action"))
    accept_reason = norm_text(row.get("plan_acceptance_reason"))
    exec_reason = norm_text(row.get("execution_decision_reason"))
    veto_reason = norm_text(row.get("execution_veto_reason"))

    exec_vetoed = pd.to_numeric(pd.Series([row.get("execution_vetoed")]), errors="coerce").iloc[0]
    exec_applied = pd.to_numeric(pd.Series([row.get("execution_applied_flag")]), errors="coerce").iloc[0]

    planning_skipped = accept_reason.startswith("planning_skipped")

    vetoed_after_accept = (
        (not planning_skipped)
        and (
            exec_reason in {"physical_safe_veto", "safety_gate_veto"}
            or (pd.notna(exec_vetoed) and int(exec_vetoed) == 1)
            or veto_reason != ""
        )
    )

    executed = (
        (pd.notna(exec_applied) and int(exec_applied) == 1)
        or (action == "avoid" and not planning_skipped and not vetoed_after_accept)
    )

    if planning_skipped:
        return "planning_skipped"
    if executed:
        return "executed"
    if vetoed_after_accept:
        return "accepted_but_vetoed"
    return "planner_rejected"


def prepare_df(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["subset"] = subset_name
    out["final_stage"] = out.apply(classify_stage, axis=1)
    return out


# =========================================================
# Boxplot figure
# =========================================================
def plot_subset_boxplot(df_all: pd.DataFrame, metric_col: str, y_label: str, out_stem: str) -> None:
    if metric_col not in df_all.columns:
        raise KeyError(
            f"Metric column '{metric_col}' not found in CSV. "
            f"Available columns include: {list(df_all.columns)[:30]} ..."
        )

    plot_df = df_all.copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col]).copy()

    if len(plot_df) == 0:
        raise RuntimeError(f"No valid numeric data found for metric: {metric_col}")

    subsets = ["Safe", "Caution", "Danger"]
    data = []
    labels = []
    positions = []

    pos_map = {"Safe": 1, "Caution": 2, "Danger": 3}

    for subset in subsets:
        vals = plot_df.loc[plot_df["subset"] == subset, metric_col].to_numpy(dtype=float)
        if len(vals) > 0:
            data.append(vals)
            labels.append(subset)
            positions.append(pos_map[subset])

    fig, ax = plt.subplots(figsize=(3.55, 2.85), constrained_layout=True)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=PALETTE["median"], linewidth=1.2),
        boxprops=dict(facecolor=PALETTE["box_fill"], edgecolor=PALETTE["arrow"], linewidth=1.0),
        whiskerprops=dict(color=PALETTE["arrow"], linewidth=1.0),
        capprops=dict(color=PALETTE["arrow"], linewidth=1.0),
    )

    # 叠加散点，按 final_stage 着色
    stage_order = [
        "planning_skipped",
        "planner_rejected",
        "accepted_but_vetoed",
        "executed",
    ]

    rng = np.random.default_rng(20260419)

    for subset in subsets:
        sub_df = plot_df.loc[plot_df["subset"] == subset].copy()
        x0 = pos_map[subset]

        for stage in stage_order:
            tmp = sub_df.loc[sub_df["final_stage"] == stage]
            if len(tmp) == 0:
                continue

            y = tmp[metric_col].to_numpy(dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=len(y))
            x = np.full(len(y), x0, dtype=float) + jitter

            ax.scatter(
                x,
                y,
                s=14,
                color=PALETTE[stage],
                edgecolors=PALETTE["arrow"],
                linewidths=0.35,
                alpha=0.95,
                zorder=3,
            )

    # 中位数标注
    for subset in subsets:
        vals = plot_df.loc[plot_df["subset"] == subset, metric_col].to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        med = float(np.median(vals))
        ax.text(
            pos_map[subset],
            med,
            f"{med:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=PALETTE["executed_dark"],
        )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(subsets)
    ax.set_xlabel("Subset")
    ax.set_ylabel(y_label)
    setup_axis(ax)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["planning_skipped"],
               markeredgecolor=PALETTE["arrow"], markersize=4.8, label="Planning Skipped"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["planner_rejected"],
               markeredgecolor=PALETTE["arrow"], markersize=4.8, label="Planner Rejected"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["accepted_but_vetoed"],
               markeredgecolor=PALETTE["arrow"], markersize=4.8, label="Vetoed / Skipped"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["executed"],
               markeredgecolor=PALETTE["arrow"], markersize=4.8, label="Executed"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best", ncol=1)

    pdf_path = OUT_DIR / f"{out_stem}.pdf"
    png_path = OUT_DIR / f"{out_stem}.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    required_cols = [
        "sample",
        "action",
        "plan_acceptance_reason",
        "execution_decision_reason",
        "execution_veto_reason",
        "execution_vetoed",
        "execution_applied_flag",
        "dataset_subset_label",
        METRIC_COL,
    ]
    numeric_cols = [
        "execution_vetoed",
        "execution_applied_flag",
        METRIC_COL,
    ]

    safe_df = load_csv(SAFE_BATCH, required_cols, numeric_cols=numeric_cols)
    caution_df = load_csv(CAUTION_BATCH, required_cols, numeric_cols=numeric_cols)
    danger_df = load_csv(DANGER_BATCH, required_cols, numeric_cols=numeric_cols)

    safe_df = filter_subset_rows(safe_df, "safe")
    caution_df = filter_subset_rows(caution_df, "caution")
    danger_df = filter_subset_rows(danger_df, "danger")

    print(f"[CHECK] safe rows after subset filtering    = {len(safe_df)}")
    print(f"[CHECK] caution rows after subset filtering = {len(caution_df)}")
    print(f"[CHECK] danger rows after subset filtering  = {len(danger_df)}")

    if len(safe_df) < 5 or len(caution_df) < 5 or len(danger_df) < 5:
        raise RuntimeError(
            "At least one subset did not load a valid case-level batch table after subset filtering. "
            "Please inspect the ranked candidate CSV list above and point the script to the correct batch summary CSV."
        )

    safe_df = prepare_df(safe_df, "Safe")
    caution_df = prepare_df(caution_df, "Caution")
    danger_df = prepare_df(danger_df, "Danger")

    df_all = pd.concat([safe_df, caution_df, danger_df], axis=0, ignore_index=True)

    print("[INFO] Cross-check by subset:")
    print(df_all.groupby(["subset", "final_stage"]).size().unstack(fill_value=0))

    print("[INFO] Metric summary by subset:")
    print(df_all.groupby("subset")[METRIC_COL].describe())

    plot_subset_boxplot(df_all, METRIC_COL, Y_LABEL, OUT_STEM)


if __name__ == "__main__":
    main()