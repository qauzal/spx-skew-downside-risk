#!/usr/bin/env python3
"""Create thesis-facing figures from final result tables without refitting models."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "thesis" / "figures"
BT_RESULTS = ROOT / "thesis" / "tables" / "bt_replication.csv"
OOS_RESULTS = ROOT / "thesis" / "tables" / "oos_results.csv"

BLUE = "#2F5D8A"
BLUE_LIGHT = "#AFC4D8"
CHARCOAL = "#252A2E"
MID_GREY = "#71777C"
LIGHT_GREY = "#D9DDE0"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": CHARCOAL,
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_existing_figure_1() -> None:
    """Figure 1 is supplied because its licensed source series are not distributed."""
    for suffix in ("png", "pdf"):
        path = FIGURES / f"figure1_standardized_risk_measures.{suffix}"
        if not path.exists():
            raise RuntimeError(f"Required supplied Figure 1 is missing: {path}")


def create_figure_2() -> pd.DataFrame:
    data = pd.read_csv(BT_RESULTS).sort_values("horizon").reset_index(drop=True)
    expected = pd.DataFrame(
        {
            "horizon": [1, 3, 6, 12],
            "coefficient": [
                -0.045120216404200275,
                -0.08331851726874705,
                -0.13466303739007401,
                -0.07101605909440925,
            ],
            "standard_error": [
                0.044632899804879785,
                0.050020028088943166,
                0.055351196364248706,
                0.05147771262844937,
            ],
            "raw_p_value": [
                0.312055521552854,
                0.09577255859003604,
                0.014979100836082856,
                0.16772535719342674,
            ],
            "N": [262, 260, 257, 251],
            "event_count": [28, 28, 28, 28],
        }
    )
    for column in expected.columns:
        if not np.allclose(data[column].astype(float), expected[column].astype(float)):
            raise RuntimeError(f"B&T table values differ from expected values: {column}")

    data["ci_low"] = data["coefficient"] - 1.96 * data["standard_error"]
    data["ci_high"] = data["coefficient"] + 1.96 * data["standard_error"]

    fig, ax = plt.subplots(figsize=(7.2, 4.35))
    yerr = np.vstack(
        [data["coefficient"] - data["ci_low"], data["ci_high"] - data["coefficient"]]
    )
    ax.errorbar(
        data["horizon"],
        data["coefficient"],
        yerr=yerr,
        fmt="o",
        markersize=6,
        markerfacecolor=BLUE,
        markeredgecolor=CHARCOAL,
        markeredgewidth=0.6,
        ecolor=BLUE,
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        zorder=3,
    )
    ax.axhline(0, color=CHARCOAL, linewidth=0.9, linestyle="--", zorder=1)
    for row in data.itertuples(index=False):
        ax.annotate(
            f"p={row.raw_p_value:.3f}",
            (row.horizon, row.ci_low),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.7,
            color=MID_GREY,
        )
    ax.set_title("Figure 2. SKEW− coefficients across downturn horizons", pad=10)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("SKEW− logit coefficient")
    ax.set_xticks(data["horizon"])
    ax.set_ylim(min(-0.285, data["ci_low"].min() - 0.035), 0.085)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2_bt_horizon_coefficients.png", dpi=300)
    fig.savefig(FIGURES / "figure2_bt_horizon_coefficients.pdf")
    plt.close(fig)
    return data


def create_figure_3() -> pd.DataFrame:
    data = pd.read_csv(OOS_RESULTS)
    expected = pd.DataFrame(
        {
            "period": ["2006-2017", "2018-2025"],
            "vix_brier": [0.11036913883727736, 0.169042507532908],
            "augmented_brier": [0.10975031010122344, 0.166424387853698],
            "vix_auc": [0.7526515151515152, 0.5452631578947368],
            "augmented_auc": [0.7477272727272728, 0.5908771929824561],
        }
    )
    for column in expected.columns:
        if column == "period":
            if not data[column].astype(str).equals(expected[column].astype(str)):
                raise RuntimeError("OOS periods differ from expected periods")
        elif not np.allclose(data[column].astype(float), expected[column].astype(float)):
            raise RuntimeError(f"OOS values differ from expected values: {column}")

    periods = [value.replace("-", "–") for value in data["period"]]
    x = np.arange(len(periods))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4))

    panels = [
        (axes[0], "Panel A. Brier score (lower is better)", "vix_brier", "augmented_brier", (0, 0.19), "%.4f"),
        (axes[1], "Panel B. ROC-AUC (higher is better)", "vix_auc", "augmented_auc", (0, 1.0), "%.3f"),
    ]
    for ax, title, vix_column, augmented_column, limits, label_format in panels:
        bars_vix = ax.bar(
            x - width / 2,
            data[vix_column],
            width,
            label="VIX only",
            color="white",
            edgecolor=MID_GREY,
            linewidth=1.0,
            hatch="///",
        )
        bars_augmented = ax.bar(
            x + width / 2,
            data[augmented_column],
            width,
            label="VIX + ΔSKEW−",
            color=BLUE_LIGHT,
            edgecolor=BLUE,
            linewidth=1.0,
        )
        ax.set_title(title, fontsize=9.5, pad=8)
        ax.set_xticks(x, periods)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.6)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
        for bars in (bars_vix, bars_augmented):
            ax.bar_label(bars, fmt=label_format, padding=3, fontsize=7.8, color=CHARCOAL)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.suptitle(
        "Figure 3. Recursive out-of-sample DD21-event forecasting performance",
        y=0.985,
        fontsize=11,
    )
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.09, right=0.98, wspace=0.27)
    fig.savefig(FIGURES / "figure3_oos_forecast_performance.png", dpi=300)
    fig.savefig(FIGURES / "figure3_oos_forecast_performance.pdf")
    plt.close(fig)
    return data


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    set_style()
    validate_existing_figure_1()
    bt = create_figure_2()
    oos = create_figure_3()
    print("Validated supplied Figure 1; licensed monthly source data are not distributed")
    print("Figure 2 plotted values:")
    print(bt[["horizon", "coefficient", "standard_error", "raw_p_value", "N", "event_count", "ci_low", "ci_high"]].to_string(index=False))
    print("Figure 3 plotted values:")
    print(oos[["period", "vix_brier", "augmented_brier", "vix_auc", "augmented_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
