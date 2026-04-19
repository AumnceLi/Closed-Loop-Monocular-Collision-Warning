from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch


# =========================================================
# IEEE / journal-style plotting configuration
# =========================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.6,
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
    "triggered": "#cbd5e1",
    "accepted": "#cfe8f3",
    "planning_skipped": "#dbe9f6",
    "planner_rejected": "#f8d7b5",
    "accepted_but_vetoed": "#d7efd7",
    "executed": "#4c78a8",
    "executed_dark": "#2f5f98",
    "arrow": "#5b6470",
    "grid": "#c7ced6",
}

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(r"D:\robot_view\ore\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 这三个路径可以保留目录形式，让脚本自动选最合适的 CSV
# 如果你后面已经确定了最终汇总文件名，也可以直接改成具体文件路径
SAFE_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\safe-batch_validation_summary.csv")
CAUTION_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\caution-batch_validation_summary.csv")
DANGER_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\danger-batch_validation_summary.csv")


# =========================================================
# Helpers
# =========================================================
def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


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

                # 优先更浅层的 batch summary，而不是深层 sample 级文件
                depth = len(csv_path.relative_to(path_or_dir).parts)

                # 文件名惩罚项：尽量避开 failed / warning / trigger / truth / sample 中间文件
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

    # 排序规则：
    # 1) 行数多优先
    # 2) 层级浅优先
    # 3) penalty 小优先
    # 4) 路径名稳定排序
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
# Drawing helpers
# =========================================================
def add_box(ax, xy, w, h, text, fc="white", ec="black", hatch=None, fontsize=7.0, text_color="black"):
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor=ec,
        facecolor=fc,
        hatch=hatch,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
    )
    return box


def add_arrow(ax, p0, p1, text=None, text_offset=(0, 0), mutation_scale=10):
    arr = FancyArrowPatch(
        p0, p1,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=1.0,
        color=PALETTE["arrow"],
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(arr)
    if text is not None:
        mx = (p0[0] + p1[0]) / 2 + text_offset[0]
        my = (p0[1] + p1[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=6.8)
    return arr


# =========================================================
# Figure
# =========================================================
def plot_decision_flow(df_all: pd.DataFrame) -> None:
    stage_order = [
        "planning_skipped",
        "planner_rejected",
        "accepted_but_vetoed",
        "executed",
    ]

    stage_labels = {
        "planning_skipped": "Planning\nSkipped",
        "planner_rejected": "Planner\nRejected",
        "accepted_but_vetoed": "Vetoed /\nSkipped",
        "executed": "Executed",
    }

    total_n = len(df_all)
    n_skip = int((df_all["final_stage"] == "planning_skipped").sum())
    n_rej = int((df_all["final_stage"] == "planner_rejected").sum())
    n_veto = int((df_all["final_stage"] == "accepted_but_vetoed").sum())
    n_exec = int((df_all["final_stage"] == "executed").sum())
    n_accept = n_veto + n_exec

    print("[INFO] Overall stage counts:")
    print(df_all["final_stage"].value_counts(dropna=False))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)

    # =====================================================
    # (a) Overall flow diagram
    # =====================================================
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w = 0.22
    box_h = 0.14

    add_box(
        ax, (0.05, 0.43), box_w, box_h,
        f"Triggered\nn = {total_n}",
        fc="white", ec="black",
    )
    add_box(
        ax, (0.40, 0.76), box_w, box_h,
        f"Planning Skipped\nn = {n_skip}",
        fc="white", ec="black", hatch="//",
    )
    add_box(
        ax, (0.40, 0.10), box_w, box_h,
        f"Planner Rejected\nn = {n_rej}",
        fc="0.88", ec="black",
    )
    add_box(
        ax, (0.40, 0.43), box_w, box_h,
        f"Accepted for Preview\nn = {n_accept}",
        fc="white", ec="black",
    )
    add_box(
        ax, (0.74, 0.66), box_w, box_h,
        f"Vetoed / Skipped\nn = {n_veto}",
        fc="white", ec="black", hatch="xx",
    )
    add_box(
        ax, (0.74, 0.20), box_w, box_h,
        f"Executed\nn = {n_exec}",
        fc="black", ec="black", text_color="white",
    )

    add_arrow(ax, (0.27, 0.50), (0.40, 0.83), text=f"{n_skip}", text_offset=(0.00, 0.05))
    add_arrow(ax, (0.27, 0.50), (0.40, 0.50), text=f"{n_accept}", text_offset=(0.00, 0.05))
    add_arrow(ax, (0.27, 0.50), (0.40, 0.17), text=f"{n_rej}", text_offset=(0.00, -0.05))
    add_arrow(ax, (0.62, 0.50), (0.74, 0.73), text=f"{n_veto}", text_offset=(0.00, 0.04))
    add_arrow(ax, (0.62, 0.50), (0.74, 0.27), text=f"{n_exec}", text_offset=(0.00, -0.04))

    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, ha="left", va="top")

    # =====================================================
    # (b) Subset-wise stacked bars
    # =====================================================
    ax = axes[1]

    subsets = ["Safe", "Caution", "Danger"]
    subset_df = (
        df_all.groupby(["subset", "final_stage"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=subsets, columns=stage_order, fill_value=0)
    )

    x = np.arange(len(subsets))
    width = 0.62
    bottom = np.zeros(len(subsets))

    bar_styles = {
        "planning_skipped": dict(facecolor=PALETTE["planning_skipped"], edgecolor=PALETTE["arrow"], hatch="//"),
        "planner_rejected": dict(facecolor=PALETTE["planner_rejected"], edgecolor=PALETTE["arrow"]),
        "accepted_but_vetoed": dict(facecolor=PALETTE["accepted_but_vetoed"], edgecolor=PALETTE["arrow"], hatch="xx"),
        "executed": dict(facecolor=PALETTE["executed"], edgecolor=PALETTE["executed_dark"]),
    }

    for stage in stage_order:
        vals = subset_df[stage].to_numpy(dtype=float)
        style = bar_styles[stage]
        ax.bar(
            x,
            vals,
            width=width,
            bottom=bottom,
            linewidth=1.0,
            **style,
            label=stage_labels[stage],
        )

        for i, v in enumerate(vals):
            if v > 0:
                text_color = "white" if stage == "executed" else "black"
                ax.text(
                    x[i],
                    bottom[i] + v / 2,
                    f"{int(v)}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color=text_color,
                )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_xlabel("Subset")
    ax.set_ylabel("Cases")
    ax.set_ylim(0, max(bottom) + 1)
    setup_axis(ax)
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, ha="left", va="top")

    legend_handles = [
        Patch(facecolor=PALETTE["planning_skipped"], edgecolor=PALETTE["arrow"], hatch="//", label="Planning Skipped"),
        Patch(facecolor=PALETTE["planner_rejected"], edgecolor=PALETTE["arrow"], label="Planner Rejected"),
        Patch(facecolor=PALETTE["accepted_but_vetoed"], edgecolor=PALETTE["arrow"], hatch="xx", label="Vetoed / Skipped"),
        Patch(facecolor=PALETTE["executed"], edgecolor=PALETTE["executed_dark"], label="Executed"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right")

    pdf_path = OUT_DIR / "decision_flow_final.pdf"
    png_path = OUT_DIR / "decision_flow_final.png"
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
    ]
    numeric_cols = [
        "execution_vetoed",
        "execution_applied_flag",
    ]

    safe_df = load_csv(SAFE_BATCH, required_cols, numeric_cols=numeric_cols)
    caution_df = load_csv(CAUTION_BATCH, required_cols, numeric_cols=numeric_cols)
    danger_df = load_csv(DANGER_BATCH, required_cols, numeric_cols=numeric_cols)

    print(f"[CHECK] safe rows    = {len(safe_df)}")
    print(f"[CHECK] caution rows = {len(caution_df)}")
    print(f"[CHECK] danger rows  = {len(danger_df)}")

    if len(safe_df) < 5 or len(caution_df) < 5 or len(danger_df) < 5:
        raise RuntimeError(
            "At least one subset did not load a valid case-level batch table. "
            "Please inspect the ranked candidate CSV list above and point the script to the correct batch summary CSV."
        )

    safe_df = prepare_df(safe_df, "Safe")
    caution_df = prepare_df(caution_df, "Caution")
    danger_df = prepare_df(danger_df, "Danger")

    df_all = pd.concat([safe_df, caution_df, danger_df], axis=0, ignore_index=True)

    print("[INFO] Cross-check by subset:")
    print(df_all.groupby(["subset", "final_stage"]).size().unstack(fill_value=0))

    print("[INFO] Overall stage counts:")
    print(df_all["final_stage"].value_counts(dropna=False))

    plot_decision_flow(df_all)


if __name__ == "__main__":
    main()