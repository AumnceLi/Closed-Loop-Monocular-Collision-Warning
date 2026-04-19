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
    "legend.fontsize": 6.8,
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
    "blue": "#1f77b4",
    "orange": "#ff7f0e",
    "green": "#2ca02c",
    "red": "#d62728",
    "purple": "#9467bd",
    "gray": "#7f7f7f",
    "grid": "#c7ced6",
}

SUBSET_COLORS = {
    "Safe": PALETTE["blue"],
    "Caution": PALETTE["orange"],
    "Danger": PALETTE["red"],
}

# =========================================================
# Paths
# =========================================================
OUT_DIR = Path(r"D:\robot_view\ore\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\safe-batch_validation_summary.csv")
CAUTION_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\caution-batch_validation_summary.csv")
DANGER_BATCH = Path(r"D:\robot_view\ore\outputs\validation_runs\danger-batch_validation_summary.csv")

PC_VIS_FLOOR = 1e-60


# =========================================================
# Helpers
# =========================================================
def find_csv_with_columns(path_or_dir: Path, required_cols: Iterable[str]) -> Path:
    required_cols = list(required_cols)

    if path_or_dir.is_file():
        df_head = pd.read_csv(path_or_dir, nrows=5)
        cols = set(df_head.columns)
        missing = [c for c in required_cols if c not in cols]
        if missing:
            raise ValueError(f"{path_or_dir} is missing required columns: {missing}")
        return path_or_dir

    if not path_or_dir.exists():
        raise FileNotFoundError(f"Path not found: {path_or_dir}")

    csv_files = list(path_or_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under: {path_or_dir}")

    best_match = None
    best_score = -1
    for csv_path in csv_files:
        try:
            df_head = pd.read_csv(csv_path, nrows=5)
            cols = set(df_head.columns)
            score = sum(col in cols for col in required_cols)
            if score > best_score:
                best_score = score
                best_match = csv_path
        except Exception:
            continue

    if best_match is None or best_score < len(required_cols):
        raise ValueError(
            f"Could not find a CSV under {path_or_dir} containing all required columns: {required_cols}"
        )
    return best_match


def load_csv(
    path_or_dir: Path,
    required_cols: Iterable[str],
    numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    csv_path = find_csv_with_columns(path_or_dir, required_cols)
    print(f"[INFO] Using CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()

    if numeric_cols is not None:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def clean_series(series: pd.Series) -> pd.Series:
    s = safe_numeric(series)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    return s


def clip_pc_for_vis(series: pd.Series, floor: float = PC_VIS_FLOOR) -> pd.Series:
    s = clean_series(series)
    s = s.where(s > floor, floor)
    return s


def norm_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.32, color=PALETTE["grid"])
    ax.tick_params(direction="in", length=2.8, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def jitter_values(n: int, scale: float = 0.04, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=n)


# =========================================================
# Outcome classification
# =========================================================
def classify_outcome(row: pd.Series) -> str:
    action = norm_text(row.get("action"))
    accept_reason = norm_text(row.get("plan_acceptance_reason"))
    exec_reason = norm_text(row.get("execution_decision_reason"))
    veto_reason = norm_text(row.get("execution_veto_reason"))

    exec_vetoed = pd.to_numeric(pd.Series([row.get("execution_vetoed")]), errors="coerce").iloc[0]
    exec_applied = pd.to_numeric(pd.Series([row.get("execution_applied_flag")]), errors="coerce").iloc[0]

    is_vetoed = (
        exec_reason in {"physical_safe_veto", "safety_gate_veto"}
        or (pd.notna(exec_vetoed) and int(exec_vetoed) == 1)
        or veto_reason != ""
        or accept_reason.startswith("planning_skipped")
    )

    is_executed = (
        (pd.notna(exec_applied) and int(exec_applied) == 1)
        or (action == "avoid" and not is_vetoed)
    )

    if is_executed:
        return "executed"
    if is_vetoed:
        return "vetoed"
    return "rejected"


# =========================================================
# Scatter preparation
# =========================================================
def prepare_scatter_df(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    out = df.copy()
    out["margin_gain_px"] = (
        safe_numeric(out["selected_pred_min_safe_margin_px"])
        - safe_numeric(out["zero_pred_min_safe_margin_px"])
    )
    out["trace_ratio"] = safe_numeric(out["acceptance_trace_ratio_vs_zero"])
    out["outcome"] = out.apply(classify_outcome, axis=1)
    out["subset"] = subset_name
    out = out[np.isfinite(out["margin_gain_px"]) & np.isfinite(out["trace_ratio"])].copy()
    return out


def nice_limits_linear(vmin: float, vmax: float, pad: float, round_to: float) -> tuple[float, float]:
    lo = np.floor((vmin - pad) / round_to) * round_to
    hi = np.ceil((vmax + pad) / round_to) * round_to
    return float(lo), float(hi)


def compute_scatter_limits(
    series: pd.Series,
    base_pad_ratio: float,
    min_pad: float,
    jitter_pad: float,
    round_to: float,
) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    vmin = float(s.min())
    vmax = float(s.max())
    span = max(vmax - vmin, 1e-9)
    pad = max(span * base_pad_ratio, min_pad) + jitter_pad
    return nice_limits_linear(vmin, vmax, pad, round_to)


def draw_subset_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    labels_to_show: set[str] | None = None,
    jitter_seed: int = 0,
) -> None:
    labels_to_show = labels_to_show or set()
    rng = np.random.default_rng(jitter_seed)

    style_map = {
        "executed": dict(marker="o", s=64, c=PALETTE["blue"], linewidths=1.0, zorder=4),
        "rejected": dict(marker="x", s=60, c=PALETTE["orange"], linewidths=1.25, zorder=2),
        "vetoed": dict(marker="s", s=66, facecolors="#cfe8cf", edgecolors=PALETTE["green"], linewidths=1.1, zorder=3),
    }

    for outcome in ["rejected", "vetoed", "executed"]:
        sub = df[df["outcome"] == outcome].copy()
        if len(sub) == 0:
            continue

        x = sub["margin_gain_px"].to_numpy(dtype=float)
        y = sub["trace_ratio"].to_numpy(dtype=float)

        # visualization-only jitter
        x = x + rng.normal(0.0, 4.0, size=len(x))
        y = y + rng.normal(0.0, 0.004, size=len(y))

        ax.scatter(x, y, **style_map[outcome])


    ax.axvline(0.0, linestyle=":", linewidth=0.9, color=PALETTE["gray"])
    ax.axhline(1.0, linestyle=":", linewidth=0.9, color=PALETTE["gray"])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, pad=2)
    ax.set_xlabel(r"$\Delta$ Min Margin (px)")
    setup_axis(ax)


# =========================================================
# Figure 4: coupled scatter (main-text figure)
# =========================================================
def plot_figure4_coupled_scatter(caution_df: pd.DataFrame, danger_df: pd.DataFrame) -> None:
    caution_plot = prepare_scatter_df(caution_df, "Caution")
    danger_plot = prepare_scatter_df(danger_df, "Danger")

    print("Caution outcome counts:\n", caution_plot["outcome"].value_counts(dropna=False))
    print("Danger outcome counts:\n", danger_plot["outcome"].value_counts(dropna=False))

    # Important fix:
    # reserve extra room for scatter jitter and marker size, otherwise
    # boundary points can appear clipped in the exported PDF.
    all_y = pd.concat([caution_plot["trace_ratio"], danger_plot["trace_ratio"]], axis=0)
    shared_ylim = compute_scatter_limits(
        all_y,
        base_pad_ratio=0.10,
        min_pad=0.08,
        jitter_pad=0.03,
        round_to=0.05,
    )

    caution_xlim = compute_scatter_limits(
        caution_plot["margin_gain_px"],
        base_pad_ratio=0.22,
        min_pad=35.0,
        jitter_pad=10.0,
        round_to=25.0,
    )

    danger_xlim = compute_scatter_limits(
        danger_plot["margin_gain_px"],
        base_pad_ratio=0.22,
        min_pad=30.0,
        jitter_pad=10.0,
        round_to=25.0,
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), constrained_layout=True)

    draw_subset_scatter(
        axes[0],
        caution_plot,
        "Caution Subset",
        xlim=caution_xlim,
        ylim=shared_ylim,
        labels_to_show=set(),   # 不显示 S00155
        jitter_seed=1,
    )
    axes[0].set_ylabel(r"Trace Ratio vs. Zero Baseline")
    axes[0].text(0.03, 0.90, "(a)", transform=axes[0].transAxes)

    draw_subset_scatter(
        axes[1],
        danger_plot,
        "Danger Subset",
        xlim=danger_xlim,
        ylim=shared_ylim,
        labels_to_show=set(),   # 不显示 S00167
        jitter_seed=2,
    )
    axes[1].text(0.03, 0.90, "(b)", transform=axes[1].transAxes)

    legend_elements = [
        Line2D([0], [0], marker='o', color=PALETTE['blue'], label='Executed', markersize=5, linestyle='None'),
        Line2D([0], [0], marker='x', color=PALETTE['orange'], label='Rejected', markersize=6, linestyle='None'),
        Line2D([0], [0], marker='s', color=PALETTE['green'], markerfacecolor='#cfe8cf',
               label='Vetoed / Skipped', markersize=6, linestyle='None'),
    ]

    axes[1].legend(
        handles=legend_elements,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
        borderaxespad=0.0,
    )

    pdf_path = OUT_DIR / "coupled_batch_scatter_final.pdf"
    png_path = OUT_DIR / "coupled_batch_scatter_final.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")

# =========================================================
# Figure 5: baseline hazard distributions (main-text figure)
# =========================================================
def plot_figure5_baseline_hazard(
    safe_df: pd.DataFrame,
    caution_df: pd.DataFrame,
    danger_df: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), constrained_layout=True)

    # -------------------------
    # (a) Original future minimum distance
    # -------------------------
    ax = axes[0]

    safe_dist = clean_series(safe_df["orig_min_future_min_distance_m"])
    caution_dist = clean_series(caution_df["orig_min_future_min_distance_m"])
    danger_dist = clean_series(danger_df["orig_min_future_min_distance_m"])

    bp = ax.boxplot(
        [safe_dist, caution_dist, danger_dist],
        positions=[1, 2, 3],
        widths=0.52,
        patch_artist=True,
        medianprops=dict(color=PALETTE["gray"], linewidth=1.0),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(color=PALETTE["gray"], linewidth=1.0),
        capprops=dict(color=PALETTE["gray"], linewidth=1.0),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [SUBSET_COLORS["Safe"], SUBSET_COLORS["Caution"], SUBSET_COLORS["Danger"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)

    for xpos, vals, seed, color in [
        (1, safe_dist, 1, SUBSET_COLORS["Safe"]),
        (2, caution_dist, 2, SUBSET_COLORS["Caution"]),
        (3, danger_dist, 3, SUBSET_COLORS["Danger"]),
    ]:
        if len(vals) > 0:
            ax.scatter(
                xpos + jitter_values(len(vals), scale=0.04, seed=seed),
                vals,
                s=18,
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                zorder=3,
            )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Safe", "Caution", "Danger"])
    ax.set_xlabel("Subset")
    ax.set_ylabel("Original Future Min Dist. (m)")
    ax.text(0.04, 0.90, "(a)", transform=ax.transAxes)
    setup_axis(ax)

    # -------------------------
    # (b) Original worst collision probability
    # -------------------------
    ax = axes[1]

    safe_pc = clip_pc_for_vis(safe_df["orig_max_pc_bplane"])
    caution_pc = clip_pc_for_vis(caution_df["orig_max_pc_bplane"])
    danger_pc = clip_pc_for_vis(danger_df["orig_max_pc_bplane"])

    bp = ax.boxplot(
        [safe_pc, caution_pc, danger_pc],
        positions=[1, 2, 3],
        widths=0.52,
        patch_artist=True,
        medianprops=dict(color=PALETTE["gray"], linewidth=1.0),
        boxprops=dict(linewidth=1.0),
        whiskerprops=dict(color=PALETTE["gray"], linewidth=1.0),
        capprops=dict(color=PALETTE["gray"], linewidth=1.0),
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [SUBSET_COLORS["Safe"], SUBSET_COLORS["Caution"], SUBSET_COLORS["Danger"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)

    for xpos, vals, seed, color in [
        (1, safe_pc, 11, SUBSET_COLORS["Safe"]),
        (2, caution_pc, 12, SUBSET_COLORS["Caution"]),
        (3, danger_pc, 13, SUBSET_COLORS["Danger"]),
    ]:
        if len(vals) > 0:
            ax.scatter(
                xpos + jitter_values(len(vals), scale=0.04, seed=seed),
                vals,
                s=18,
                facecolors="white",
                edgecolors=color,
                linewidths=0.85,
                zorder=3,
            )

    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Safe", "Caution", "Danger"])
    ax.set_xlabel("Subset")
    ax.set_ylabel(r"Original Worst $P_c$")
    ax.text(0.04, 0.90, "(b)", transform=ax.transAxes)
    setup_axis(ax)

    pdf_path = OUT_DIR / "batch_hazard_baseline_final.pdf"
    png_path = OUT_DIR / "batch_hazard_baseline_final.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")


# =========================================================
# Appendix figure: executed closed-loop paired change
# =========================================================
def plot_appendix_executed_paired_change(caution_df: pd.DataFrame) -> None:
    """
    Appendix figure:
    Paired Original vs. Active closed-loop change for EXECUTED caution cases only.

    Important fix:
    Pairing must be preserved row-wise. Do NOT clean original/active columns separately,
    otherwise the connecting lines may join different samples.
    """
    df = caution_df.copy()
    df["outcome"] = df.apply(classify_outcome, axis=1)
    exec_df = df[df["outcome"] == "executed"].copy()

    if exec_df.empty:
        print("[WARN] No executed cases found; appendix paired-change figure skipped.")
        return

    # ---------------------------------------------------------------------
    # Build row-wise paired data for distance
    # ---------------------------------------------------------------------
    dist_cols = ["sample", "orig_min_future_min_distance_m", "act_min_future_min_distance_m"]
    dist_df = exec_df[dist_cols].copy()
    dist_df["orig_min_future_min_distance_m"] = pd.to_numeric(
        dist_df["orig_min_future_min_distance_m"], errors="coerce"
    )
    dist_df["act_min_future_min_distance_m"] = pd.to_numeric(
        dist_df["act_min_future_min_distance_m"], errors="coerce"
    )
    dist_df = dist_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["orig_min_future_min_distance_m", "act_min_future_min_distance_m"]
    )

    # ---------------------------------------------------------------------
    # Build row-wise paired data for Pc
    # ---------------------------------------------------------------------
    pc_cols = ["sample", "orig_max_pc_bplane", "act_max_pc_bplane"]
    pc_df = exec_df[pc_cols].copy()
    pc_df["orig_max_pc_bplane"] = pd.to_numeric(pc_df["orig_max_pc_bplane"], errors="coerce")
    pc_df["act_max_pc_bplane"] = pd.to_numeric(pc_df["act_max_pc_bplane"], errors="coerce")
    pc_df = pc_df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["orig_max_pc_bplane", "act_max_pc_bplane"]
    )

    # clip only AFTER pairing is preserved
    if not pc_df.empty:
        pc_df["orig_max_pc_bplane"] = pc_df["orig_max_pc_bplane"].clip(lower=PC_VIS_FLOOR)
        pc_df["act_max_pc_bplane"] = pc_df["act_max_pc_bplane"].clip(lower=PC_VIS_FLOOR)

    if dist_df.empty and pc_df.empty:
        print("[WARN] No valid executed paired results found after row-wise filtering.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8), constrained_layout=True)

    # =========================================================
    # (a) Future minimum distance
    # =========================================================
    ax = axes[0]

    if not dist_df.empty:
        x1 = np.full(len(dist_df), 1.0)
        x2 = np.full(len(dist_df), 2.0)

        y_orig = dist_df["orig_min_future_min_distance_m"].to_numpy(dtype=float)
        y_act = dist_df["act_min_future_min_distance_m"].to_numpy(dtype=float)

        for i in range(len(dist_df)):
            ax.plot([1, 2], [y_orig[i], y_act[i]], color=PALETTE["gray"], linewidth=0.9, alpha=0.75, zorder=1)

        ax.scatter(
            x1, y_orig,
            s=28, facecolors="white", edgecolors=PALETTE["blue"], linewidths=0.9,
            zorder=3, label="Original"
        )
        ax.scatter(
            x2, y_act,
            s=30, color=PALETTE["orange"],
            zorder=4, label="Active"
        )

        # median markers
        med_orig = float(np.median(y_orig))
        med_act = float(np.median(y_act))
        ax.plot(1, med_orig, marker="_", markersize=12, color=PALETTE["red"], zorder=5)
        ax.plot(2, med_act, marker="_", markersize=12, color=PALETTE["red"], zorder=5)

        improved = int(np.sum(y_act > y_orig))
        worsened = int(np.sum(y_act < y_orig))
        unchanged = int(np.sum(np.isclose(y_act, y_orig)))

        ax.text(
            0.03, 0.05,
            f"n = {len(dist_df)}\nImproved: {improved}\nWorsened: {worsened}\nUnchanged: {unchanged}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Original", "Active"])
    ax.set_ylabel("Future Min Dist. (m)")
    ax.text(0.04, 0.90, "(a)", transform=ax.transAxes)
    setup_axis(ax)

    # =========================================================
    # (b) Worst Pc
    # =========================================================
    ax = axes[1]

    if not pc_df.empty:
        x1 = np.full(len(pc_df), 1.0)
        x2 = np.full(len(pc_df), 2.0)

        y_orig = pc_df["orig_max_pc_bplane"].to_numpy(dtype=float)
        y_act = pc_df["act_max_pc_bplane"].to_numpy(dtype=float)

        for i in range(len(pc_df)):
            ax.plot([1, 2], [y_orig[i], y_act[i]], color=PALETTE["gray"], linewidth=0.9, alpha=0.75, zorder=1)

        ax.scatter(
            x1, y_orig,
            s=28, facecolors="white", edgecolors=PALETTE["blue"], linewidths=0.9,
            zorder=3, label="Original"
        )
        ax.scatter(
            x2, y_act,
            s=30, color=PALETTE["orange"],
            zorder=4, label="Active"
        )

        # geometric median-like display on log-scale: use median of raw values
        med_orig = float(np.median(y_orig))
        med_act = float(np.median(y_act))
        ax.plot(1, med_orig, marker="_", markersize=12, color=PALETTE["red"], zorder=5)
        ax.plot(2, med_act, marker="_", markersize=12, color=PALETTE["red"], zorder=5)

        improved = int(np.sum(y_act < y_orig))   # lower Pc is better
        worsened = int(np.sum(y_act > y_orig))
        unchanged = int(np.sum(np.isclose(y_act, y_orig)))

        ax.text(
            0.03, 0.05,
            f"n = {len(pc_df)}\nImproved: {improved}\nWorsened: {worsened}\nUnchanged: {unchanged}",
            transform=ax.transAxes,
            fontsize=6.5,
            va="bottom",
            ha="left",
        )

    ax.set_yscale("log")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Original", "Active"])
    ax.set_ylabel(r"Worst $P_c$")
    ax.text(0.04, 0.90, "(b)", transform=ax.transAxes)
    setup_axis(ax)

    # one shared legend is enough
    handles = [
        Line2D([0], [0], marker='o', markerfacecolor='white', markeredgecolor=PALETTE['blue'],
               linestyle='None', markersize=5, label='Original'),
        Line2D([0], [0], marker='o', color=PALETTE['orange'],
               linestyle='None', markersize=5, label='Active'),
        Line2D([0], [0], marker='_', color=PALETTE['red'],
               linestyle='None', markersize=10, label='Median'),
    ]
    axes[1].legend(
        handles=handles,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
    )

    pdf_path = OUT_DIR / "appendix_executed_paired_change_fixed.pdf"
    png_path = OUT_DIR / "appendix_executed_paired_change_fixed.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")


# =========================================================
# Main
# =========================================================
def main() -> None:
    safe_required = [
        "sample",
        "action",
        "plan_acceptance_reason",
        "execution_decision_reason",
        "execution_veto_reason",
        "execution_vetoed",
        "orig_min_future_min_distance_m",
        "orig_max_pc_bplane",
    ]
    caution_required = [
        "sample",
        "action",
        "plan_acceptance_reason",
        "execution_decision_reason",
        "execution_veto_reason",
        "execution_vetoed",
        "execution_applied_flag",
        "selected_pred_min_safe_margin_px",
        "zero_pred_min_safe_margin_px",
        "acceptance_trace_ratio_vs_zero",
        "orig_min_future_min_distance_m",
        "act_min_future_min_distance_m",
        "orig_max_pc_bplane",
        "act_max_pc_bplane",
    ]
    danger_required = [
        "sample",
        "action",
        "plan_acceptance_reason",
        "execution_decision_reason",
        "execution_veto_reason",
        "execution_vetoed",
        "execution_applied_flag",
        "selected_pred_min_safe_margin_px",
        "zero_pred_min_safe_margin_px",
        "acceptance_trace_ratio_vs_zero",
        "orig_min_future_min_distance_m",
        "orig_max_pc_bplane",
    ]

    safe_numeric_cols = [
        "execution_vetoed",
        "orig_min_future_min_distance_m",
        "orig_max_pc_bplane",
    ]
    caution_numeric_cols = [
        "execution_vetoed",
        "execution_applied_flag",
        "selected_pred_min_safe_margin_px",
        "zero_pred_min_safe_margin_px",
        "acceptance_trace_ratio_vs_zero",
        "orig_min_future_min_distance_m",
        "act_min_future_min_distance_m",
        "orig_max_pc_bplane",
        "act_max_pc_bplane",
    ]
    danger_numeric_cols = [
        "execution_vetoed",
        "execution_applied_flag",
        "selected_pred_min_safe_margin_px",
        "zero_pred_min_safe_margin_px",
        "acceptance_trace_ratio_vs_zero",
        "orig_min_future_min_distance_m",
        "orig_max_pc_bplane",
    ]

    safe_df = load_csv(SAFE_BATCH, safe_required, numeric_cols=safe_numeric_cols)
    caution_df = load_csv(CAUTION_BATCH, caution_required, numeric_cols=caution_numeric_cols)
    danger_df = load_csv(DANGER_BATCH, danger_required, numeric_cols=danger_numeric_cols)

    plot_figure4_coupled_scatter(caution_df, danger_df)
    plot_figure5_baseline_hazard(safe_df, caution_df, danger_df)
    plot_appendix_executed_paired_change(caution_df)


if __name__ == "__main__":
    main()