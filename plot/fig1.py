from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Low-saturation, IEEE-friendly palette
PALETTE = {
    "orig": "#1f77b4",       # blue
    "cmp": "#ff7f0e",        # orange
    "trigger": "#9467bd",    # purple
    "maneuver": "#2ca02c",   # green
    "baseline": "#7f7f7f",   # gray
    "grid": "#c7ced6",
}

OUT_DIR = Path(r"D:\robot_view\ore\paper_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAFE_CASE_ROOT = Path(r"D:\robot_view\ore\outputs\validation_runs\final_batch_safe\sample_00143")
SAFE_ORIG = SAFE_CASE_ROOT / "original_render"

CAUTION_CASE_ROOT = Path(r"D:\robot_view\ore\outputs\validation_runs\final_smoke_00155\sample_00155")
CAUTION_ORIG = CAUTION_CASE_ROOT / "original_render"
CAUTION_ACT = CAUTION_CASE_ROOT / "active_render"

DANGER_CASE_ROOT = Path(r"D:\robot_view\ore\outputs\validation_runs\audit_00167\sample_00167")
DANGER_ORIG = DANGER_CASE_ROOT / "original_render"
DANGER_AUDIT = DANGER_CASE_ROOT / "active_render"


def find_csv_with_columns(path_or_dir: Path, required_cols: list[str]) -> Path:
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


def load_case_csv(path_or_dir: Path, required_cols: list[str]) -> pd.DataFrame:
    csv_path = find_csv_with_columns(path_or_dir, required_cols)
    print(f"[INFO] Using CSV: {csv_path}")

    df = pd.read_csv(csv_path).copy()
    df = df.sort_values("t_sec").reset_index(drop=True)

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def infer_trigger_time(case_root: Path) -> Optional[float]:
    candidates = list(case_root.rglob("case_decision_summary.csv"))
    if not candidates:
        candidates = list(case_root.rglob("*decision*summary*.csv"))

    for csv_path in candidates:
        try:
            df = pd.read_csv(csv_path)
            if "trigger_t_sec" in df.columns and len(df) > 0:
                val = pd.to_numeric(df.loc[0, "trigger_t_sec"], errors="coerce")
                if pd.notna(val):
                    return float(val)
        except Exception:
            continue
    return None


def setup_axis(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.32, color=PALETTE["grid"])
    ax.tick_params(direction="in", length=2.8, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def add_vertical_marker(
    ax: plt.Axes,
    x: Optional[float],
    linestyle: str,
    label: Optional[str] = None,
    color: str = PALETTE["baseline"],
) -> None:
    if x is None or not np.isfinite(x):
        return
    ax.axvline(x=x, linestyle=linestyle, linewidth=1.0, color=color, label=label, alpha=0.95)


def crop_df(df: pd.DataFrame, xlim: Optional[tuple[float, float]]) -> pd.DataFrame:
    if xlim is None:
        return df
    x0, x1 = xlim
    return df[(df["t_sec"] >= x0) & (df["t_sec"] <= x1)].copy()


def auto_ylim(series_list: list[pd.Series], pad_ratio: float = 0.06) -> tuple[float, float]:
    vals = []
    for s in series_list:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if len(s) > 0:
            vals.append(s.values)

    if not vals:
        return (0.0, 1.0)

    vals = np.concatenate(vals)
    vmin = np.min(vals)
    vmax = np.max(vals)

    if np.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0)
        return (vmin - pad, vmax + pad)

    pad = (vmax - vmin) * pad_ratio
    return (vmin - pad, vmax + pad)


def plot_series(ax: plt.Axes, x: pd.Series, y: pd.Series, label: Optional[str], kind: str) -> None:
    if kind == "orig":
        ax.plot(
            x,
            y,
            "-",
            color=PALETTE["orig"],
            label=label,
            marker="o",
            markersize=2.9,
            markerfacecolor="white",
            markeredgecolor=PALETTE["orig"],
            markeredgewidth=0.7,
            linewidth=1.25,
        )
    else:
        ax.plot(
            x,
            y,
            "--",
            color=PALETTE["cmp"],
            label=label,
            marker="s",
            markersize=2.7,
            markerfacecolor="white",
            markeredgecolor=PALETTE["cmp"],
            markeredgewidth=0.7,
            linewidth=1.15,
        )


def plot_case(
    out_name: str,
    case_root: Path,
    original: Path,
    comparison: Optional[Path],
    comparison_label: Optional[str],
    trigger_t_sec: Optional[float],
    maneuver_t_sec: Optional[float],
    xlim: Optional[tuple[float, float]],
    common_time_only: bool,
    panel_a_col: str,
    panel_a_label: str,
    panel_a_zero_line: bool,
    panel_b_col: str,
    panel_b_label: str,
    panel_b_log: bool,
    panel_c_col: str,
    panel_c_label: str,
    panel_c_log: bool,
) -> None:
    required_cols = ["t_sec", panel_a_col, panel_b_col, panel_c_col]

    df_orig = load_case_csv(original, required_cols)
    df_cmp = load_case_csv(comparison, required_cols) if comparison is not None else None

    auto_trigger = infer_trigger_time(case_root)
    if trigger_t_sec is None:
        trigger_t_sec = auto_trigger

    if common_time_only and df_cmp is not None:
        common_end = min(df_orig["t_sec"].max(), df_cmp["t_sec"].max())
        df_orig = df_orig[df_orig["t_sec"] <= common_end].copy()
        df_cmp = df_cmp[df_cmp["t_sec"] <= common_end].copy()

    df_orig = crop_df(df_orig, xlim)
    if df_cmp is not None:
        df_cmp = crop_df(df_cmp, xlim)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)

    ax = axes[0]
    plot_series(ax, df_orig["t_sec"], df_orig[panel_a_col], label="Original", kind="orig")
    if df_cmp is not None:
        plot_series(ax, df_cmp["t_sec"], df_cmp[panel_a_col], label=comparison_label, kind="cmp")
    if panel_a_zero_line:
        ax.axhline(0.0, linestyle=":", linewidth=0.9, color=PALETTE["baseline"], alpha=0.9)
    add_vertical_marker(ax, trigger_t_sec, "-.", "Trigger", color=PALETTE["trigger"])
    add_vertical_marker(ax, maneuver_t_sec, "--", "Maneuver", color=PALETTE["maneuver"])
    ax.set_ylabel(panel_a_label)
    ax.set_xlabel("Time (s)")
    ax.text(-0.2, 0.95, "(a)", transform=ax.transAxes)
    ax.set_ylim(*auto_ylim(
        [df_orig[panel_a_col]] + ([df_cmp[panel_a_col]] if df_cmp is not None else []),
        pad_ratio=0.08,
    ))
    setup_axis(ax)
    ax.legend(loc="best", frameon=False)

    ax = axes[1]
    plot_series(ax, df_orig["t_sec"], df_orig[panel_b_col], label=None, kind="orig")
    if df_cmp is not None:
        plot_series(ax, df_cmp["t_sec"], df_cmp[panel_b_col], label=None, kind="cmp")
    add_vertical_marker(ax, trigger_t_sec, "-.", color=PALETTE["trigger"])
    add_vertical_marker(ax, maneuver_t_sec, "--", color=PALETTE["maneuver"])
    if panel_b_log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(*auto_ylim(
            [df_orig[panel_b_col]] + ([df_cmp[panel_b_col]] if df_cmp is not None else []),
            pad_ratio=0.10,
        ))
    ax.set_ylabel(panel_b_label)
    ax.set_xlabel("Time (s)")
    ax.text(-0.2, 0.95, "(b)", transform=ax.transAxes)
    setup_axis(ax)

    ax = axes[2]
    plot_series(ax, df_orig["t_sec"], df_orig[panel_c_col], label=None, kind="orig")
    if df_cmp is not None:
        plot_series(ax, df_cmp["t_sec"], df_cmp[panel_c_col], label=None, kind="cmp")
    add_vertical_marker(ax, trigger_t_sec, "-.", color=PALETTE["trigger"])
    add_vertical_marker(ax, maneuver_t_sec, "--", color=PALETTE["maneuver"])
    if panel_c_log:
        ax.set_yscale("log")
    else:
        ax.set_ylim(*auto_ylim(
            [df_orig[panel_c_col]] + ([df_cmp[panel_c_col]] if df_cmp is not None else []),
            pad_ratio=0.10,
        ))
    ax.set_ylabel(panel_c_label)
    ax.set_xlabel("Time (s)")
    ax.text(-0.24, 0.95, "(c)", transform=ax.transAxes)
    setup_axis(ax)

    pdf_path = OUT_DIR / f"{out_name}.pdf"
    png_path = OUT_DIR / f"{out_name}.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"[DONE] Saved: {pdf_path}")
    print(f"[DONE] Saved: {png_path}")


def main() -> None:
    plot_case(
        out_name="safe_case_curves",
        case_root=SAFE_CASE_ROOT,
        original=SAFE_ORIG,
        comparison=None,
        comparison_label=None,
        trigger_t_sec=None,
        maneuver_t_sec=None,
        xlim=(0.0, 120.0),
        common_time_only=False,
        panel_a_col="safe_box_dist_px",
        panel_a_label="Safe-box\ndist. (px)",
        panel_a_zero_line=True,
        panel_b_col="trace_P_pos",
        panel_b_label="Trace($P_{pos}$)",
        panel_b_log=False,
        panel_c_col="pred_future_min_distance_m",
        panel_c_label="Pred. future\nmin dist. (m)",
        panel_c_log=False,
    )

    plot_case(
        out_name="caution_case_curves",
        case_root=CAUTION_CASE_ROOT,
        original=CAUTION_ORIG,
        comparison=CAUTION_ACT,
        comparison_label="Active",
        trigger_t_sec=None,
        maneuver_t_sec=None,
        xlim=(0.0, 180.0),
        common_time_only=True,
        panel_a_col="safe_box_dist_px",
        panel_a_label="Safe-box\ndist. (px)",
        panel_a_zero_line=True,
        panel_b_col="trace_P_pos",
        panel_b_label="Trace($P_{pos}$)",
        panel_b_log=False,
        panel_c_col="pred_future_min_distance_m",
        panel_c_label="Pred. future\nmin dist. (m)",
        panel_c_log=False,
    )

    plot_case(
        out_name="danger_audit_curves",
        case_root=DANGER_CASE_ROOT,
        original=DANGER_ORIG,
        comparison=DANGER_AUDIT,
        comparison_label="Audit",
        trigger_t_sec=None,
        maneuver_t_sec=None,
        xlim=(0.0, 210.0),
        common_time_only=True,
        panel_a_col="safe_box_dist_px",
        panel_a_label="Safe-box\ndist. (px)",
        panel_a_zero_line=True,
        panel_b_col="NEES_pos",
        panel_b_label="NEES$_{pos}$",
        panel_b_log=False,
        panel_c_col="pc_bplane",
        panel_c_label="$P_c$",
        panel_c_log=True,
    )


if __name__ == "__main__":
    main()
