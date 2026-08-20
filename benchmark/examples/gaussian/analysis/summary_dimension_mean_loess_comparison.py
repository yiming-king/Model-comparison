"""Compare Gaussian summary dimensions using source-mean LOWESS curves.

This module is plotting-only. It reuses the cached comparison data assembled by
``summary_dimension_comparison`` and never loads a network or reruns inference.
Raw datasets are not distinguished by source-model color. Instead, color denotes
summary dimension, diamonds denote source-model means, and LOWESS is fit only to
those mean diamonds.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ..config import ASSUMED_MODELS, RESULT_DIR
from .summary_dimension_comparison import (
    AXIS_SCALES,
    DEFAULT_LOWESS_FRAC,
    DEFAULT_X_SYMLOG_LINTHRESH,
    DEFAULT_YSCALES,
    DEFAULT_Y_SYMLOG_LINTHRESH,
    DIAGNOSTICS,
    DIAGNOSTIC_LABELS,
    ERROR_LABELS,
    ERROR_METRICS,
    POSTERIOR_MMD_Y_MIN,
    SUMMARY_COLORS,
    SUMMARY_SPECS,
    TYPICAL_SET_FILL,
    _lowess_curve,
    _padded_limits,
    _rho_aligned_limits,
    _robust_limits,
    _robust_symlog_limits,
    _set_x_symlog_scale,
    _set_y_scale,
    _validate_selection,
    load_comparison_data,
)


DEFAULT_OUTPUT_DIR = RESULT_DIR / "plots" / "summary_dimension_mean_loess_comparison"
DEFAULT_MODELS = tuple(ASSUMED_MODELS)
RAW_POINT_SIZE = 15
RAW_POINT_ALPHA = 0.13
MEAN_DIAMOND_SIZE = 58
WELL_SPECIFIED_MEAN_DIAMOND_SIZE = 105
MEAN_DIAMOND_ALPHA = 0.98
RHO_ALIGNMENT_FRACTION = 0.20

AxisScaleSpec = str | Mapping[str, str]
NumericSpec = float | Mapping[str, float]


def _resolve_spec(spec: str | float | Mapping, key: str):
    if isinstance(spec, Mapping):
        if key not in spec:
            raise ValueError(f"No plotting option supplied for {key!r}")
        return spec[key]
    return spec


def aggregate_source_means(
    data: pd.DataFrame,
    *,
    normalized: bool = False,
) -> pd.DataFrame:
    """Average rho and error within each source-model/summary/model panel."""
    y_column = "normalized_error_value" if normalized else "error_value"
    group_columns = [
        "error_metric",
        "diagnostic",
        "summary",
        "network_tag",
        "assumed_model",
        "source_model",
    ]
    missing = sorted(set(group_columns + ["rho", y_column]).difference(data.columns))
    if missing:
        raise ValueError(f"Cannot aggregate source means; missing columns: {missing}")

    finite = data.loc[
        np.isfinite(pd.to_numeric(data["rho"], errors="coerce"))
        & np.isfinite(pd.to_numeric(data[y_column], errors="coerce"))
    ].copy()
    means = (
        finite.groupby(group_columns, observed=True, sort=False)
        .agg(
            mean_rho=("rho", "mean"),
            mean_error=(y_column, "mean"),
            n_datasets=("source_model", "size"),
        )
        .reset_index()
    )
    means["well_specified"] = means["source_model"].eq(means["assumed_model"])
    return means.sort_values(
        ["error_metric", "diagnostic", "assumed_model", "summary", "source_model"]
    ).reset_index(drop=True)


def _display_thresholds(panel: pd.DataFrame, normalized: bool) -> pd.DataFrame:
    columns = (
        [
            "normalized_error_lower_threshold",
            "normalized_error_upper_threshold",
        ]
        if normalized
        else ["error_lower_threshold", "error_upper_threshold"]
    )
    output = panel.groupby("summary", observed=True)[columns].median()
    output.columns = ["low", "high"]
    return output


def _add_model_strip(ax: plt.Axes, model: str) -> None:
    strip = Rectangle(
        (0.0, 1.02),
        1.0,
        0.115,
        transform=ax.transAxes,
        facecolor="0.85",
        edgecolor="0.35",
        linewidth=0.8,
        clip_on=False,
        zorder=8,
    )
    ax.add_patch(strip)
    ax.text(
        0.5,
        1.077,
        rf"Assumed $M_{{{model[1:]}}}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=16,
        zorder=9,
    )


def _draw_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    panel_means: pd.DataFrame,
    *,
    error_metric: str,
    normalized: bool,
    lowess_frac: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float] | None,
    x_symlog_linthresh: float,
    yscale: str,
    y_symlog_linthresh: float,
) -> None:
    y_column = "normalized_error_value" if normalized else "error_value"
    thresholds = _display_thresholds(panel, normalized)

    rho_low = panel.groupby("summary", observed=True)["rho_low"].median().min()
    ax.axvspan(rho_low, 1.0, color=TYPICAL_SET_FILL, alpha=0.70, zorder=0)
    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)

    band_low = float(thresholds["low"].max())
    band_high = float(thresholds["high"].min())
    if band_low < band_high:
        ax.axhspan(
            band_low,
            band_high,
            color=TYPICAL_SET_FILL,
            alpha=0.48,
            zorder=0,
        )

    for summary, color in SUMMARY_COLORS.items():
        raw = panel.loc[panel["summary"].eq(summary)].dropna(
            subset=["rho", y_column]
        )
        means = panel_means.loc[panel_means["summary"].eq(summary)].dropna(
            subset=["mean_rho", "mean_error"]
        )
        if raw.empty or means.empty:
            continue

        # Source-model identity is intentionally not encoded by color. The
        # larger square only marks the well-specified raw datasets.
        for well_specified, points in raw.groupby("well_specified", sort=False):
            ax.scatter(
                points["rho"],
                points[y_column],
                color=color,
                marker="s" if well_specified else "o",
                s=48 if well_specified else RAW_POINT_SIZE,
                alpha=0.74 if well_specified else RAW_POINT_ALPHA,
                edgecolors="black" if well_specified else "none",
                linewidths=0.9 if well_specified else 0.0,
                zorder=3 if well_specified else 2,
            )
        for well_specified, points in means.groupby("well_specified", sort=False):
            ax.scatter(
                points["mean_rho"],
                points["mean_error"],
                color=color,
                marker="D",
                s=(
                    WELL_SPECIFIED_MEAN_DIAMOND_SIZE
                    if well_specified
                    else MEAN_DIAMOND_SIZE
                ),
                alpha=MEAN_DIAMOND_ALPHA,
                edgecolors="black",
                linewidths=0.95 if well_specified else 0.7,
                zorder=6 if well_specified else 5,
            )

        # Deliberately fit to the source means only, not to all raw datasets.
        curve_x, curve_y = _lowess_curve(
            means["mean_rho"], means["mean_error"], frac=lowess_frac
        )
        ax.plot(curve_x, curve_y, color=color, linewidth=2.25, zorder=4)

        summary_thresholds = thresholds.loc[summary]
        plotted = (
            (float(summary_thresholds["high"]),)
            if error_metric == "posterior_mmd"
            else (
                float(summary_thresholds["low"]),
                float(summary_thresholds["high"]),
            )
        )
        for threshold in plotted:
            ax.axhline(
                threshold,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.88,
                zorder=1,
            )

    if error_metric != "posterior_mmd":
        ax.axhline(0.0, color="0.45", linewidth=0.75, zorder=1)
    _set_x_symlog_scale(ax, x_symlog_linthresh)
    _set_y_scale(ax, yscale, y_symlog_linthresh)
    ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.grid(color="#D7DCE2", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FCFCFD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)


def _ylabel(error_metric: str, normalized: bool) -> str:
    comparison = "NPE, MCMC; normalized" if normalized else "NPE, MCMC"
    return f"{ERROR_LABELS[error_metric]}\n({comparison})"


def _legend_handles() -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        Line2D(
            [0],
            [0],
            marker="D",
            color=color,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.65,
            linewidth=2.7,
            markersize=8,
            label=summary,
        )
        for summary, color in SUMMARY_COLORS.items()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="0.55",
                markeredgecolor="none",
                alpha=0.35,
                markersize=6,
                label="individual dataset",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=9,
                label="source-dataset mean",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=10,
                label="well-specified dataset",
            ),
            Patch(
                facecolor=TYPICAL_SET_FILL,
                edgecolor="none",
                alpha=0.70,
                label="typical set / accepted error",
            ),
        ]
    )
    return handles


def plot_mean_loess_comparison(
    data: pd.DataFrame,
    source_means: pd.DataFrame,
    diagnostic: str,
    error_metric: str,
    path: str | Path,
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: str = "linear",
    y_symlog_linthresh: float = 1.0,
) -> Path:
    """Plot one diagnostic/error pair with mean-only summary LOWESS curves."""
    models = tuple(models)
    _validate_selection((diagnostic,), (error_metric,), models)
    selector = (
        data["diagnostic"].eq(diagnostic)
        & data["error_metric"].eq(error_metric)
        & data["assumed_model"].isin(models)
    )
    plot_data = data.loc[selector].copy()
    mean_selector = (
        source_means["diagnostic"].eq(diagnostic)
        & source_means["error_metric"].eq(error_metric)
        & source_means["assumed_model"].isin(models)
    )
    plot_means = source_means.loc[mean_selector].copy()
    if plot_data.empty or plot_means.empty:
        raise ValueError(f"No rows for {diagnostic}/{error_metric}/{models}")

    x_limits = _padded_limits(plot_data["rho"])
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4.65 * len(models) + 1.5, 6.15),
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes[0], models, strict=True):
        panel = plot_data.loc[plot_data["assumed_model"].eq(model)]
        panel_means = plot_means.loc[plot_means["assumed_model"].eq(model)]
        _draw_panel(
            ax,
            panel,
            panel_means,
            error_metric=error_metric,
            normalized=normalized,
            lowess_frac=lowess_frac,
            x_limits=x_limits,
            y_limits=None,
            x_symlog_linthresh=x_symlog_linthresh,
            yscale=yscale,
            y_symlog_linthresh=y_symlog_linthresh,
        )
        _add_model_strip(ax, model)
        ax.set_xlabel(f"Diagnostic: {DIAGNOSTIC_LABELS[diagnostic]}", fontsize=13)

    fig.supylabel(
        _ylabel(error_metric, normalized),
        x=0.018,
        fontsize=15,
        ha="center",
        va="center",
        multialignment="center",
    )
    fig.suptitle("Simulated datasets: LOWESS fitted to source means", fontsize=17, y=0.985)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=6,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.22,
        top=0.80,
        wspace=0.09,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mean_loess_grid(
    data: pd.DataFrame,
    source_means: pd.DataFrame,
    error_metric: str,
    path: str | Path,
    *,
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: str = "linear",
    y_symlog_linthresh: float = 1.0,
    rho_alignment_fraction: float = RHO_ALIGNMENT_FRACTION,
) -> Path:
    """Plot diagnostics by rows and assumed models by columns in one figure."""
    diagnostics = tuple(diagnostics)
    models = tuple(models)
    _validate_selection(diagnostics, (error_metric,), models)
    plot_data = data.loc[
        data["diagnostic"].isin(diagnostics)
        & data["error_metric"].eq(error_metric)
        & data["assumed_model"].isin(models)
    ].copy()
    plot_means = source_means.loc[
        source_means["diagnostic"].isin(diagnostics)
        & source_means["error_metric"].eq(error_metric)
        & source_means["assumed_model"].isin(models)
    ].copy()
    if plot_data.empty or plot_means.empty:
        raise ValueError(f"No rows for {diagnostics}/{error_metric}/{models}")

    fig, axes = plt.subplots(
        len(diagnostics),
        len(models),
        figsize=(5.25 * len(models) + 1.4, 3.15 * len(diagnostics) + 2.1),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row_index, diagnostic in enumerate(diagnostics):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            panel = plot_data.loc[
                plot_data["diagnostic"].eq(diagnostic)
                & plot_data["assumed_model"].eq(model)
            ]
            panel_means = plot_means.loc[
                plot_means["diagnostic"].eq(diagnostic)
                & plot_means["assumed_model"].eq(model)
            ]
            x_limits = _rho_aligned_limits(
                panel["rho"], anchor_fraction=rho_alignment_fraction
            )
            y_column = (
                "normalized_error_value" if normalized else "error_value"
            )
            threshold_columns = (
                [
                    "normalized_error_lower_threshold",
                    "normalized_error_upper_threshold",
                ]
                if normalized
                else ["error_lower_threshold", "error_upper_threshold"]
            )
            if yscale == "symlog":
                y_limits = _robust_symlog_limits(
                    panel[y_column],
                    panel_means["mean_error"],
                    panel[threshold_columns].to_numpy(),
                    linthresh=y_symlog_linthresh,
                    include_zero=True,
                )
            else:
                y_limits = _robust_limits(
                    panel[y_column],
                    panel_means["mean_error"],
                    panel[threshold_columns].to_numpy(),
                    include_zero=error_metric != "posterior_mmd",
                )
            if error_metric == "posterior_mmd":
                y_limits = (POSTERIOR_MMD_Y_MIN, y_limits[1])
            _draw_panel(
                ax,
                panel,
                panel_means,
                error_metric=error_metric,
                normalized=normalized,
                lowess_frac=lowess_frac,
                x_limits=x_limits,
                y_limits=y_limits,
                x_symlog_linthresh=x_symlog_linthresh,
                yscale=yscale,
                y_symlog_linthresh=y_symlog_linthresh,
            )
            ax.set_xlabel(
                f"Diagnostic: {DIAGNOSTIC_LABELS[diagnostic]}", fontsize=12.5
            )
            if row_index == 0:
                _add_model_strip(ax, model)

    fig.supylabel(
        _ylabel(error_metric, normalized),
        x=0.018,
        fontsize=16,
        ha="center",
        va="center",
        multialignment="center",
    )
    fig.suptitle(
        "Simulated datasets: LOWESS fitted to source means", fontsize=18, y=0.958
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=7,
        frameon=False,
        fontsize=10.5,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.14,
        top=0.90,
        wspace=0.24,
        hspace=0.52,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run_mean_loess_comparison(
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    error_metrics: tuple[str, ...] = ERROR_METRICS,
    models: tuple[str, ...] = DEFAULT_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: AxisScaleSpec = DEFAULT_YSCALES,
    y_symlog_linthresh: NumericSpec = DEFAULT_Y_SYMLOG_LINTHRESH,
) -> dict[str, object]:
    """Generate every requested diagnostic/error mean-only LOWESS figure."""
    diagnostics = tuple(diagnostics)
    error_metrics = tuple(error_metrics)
    models = tuple(models)
    _validate_selection(diagnostics, error_metrics, models)
    data = load_comparison_data(
        diagnostics=diagnostics,
        error_metrics=error_metrics,
        summary_specs=SUMMARY_SPECS,
    )
    source_means = aggregate_source_means(data, normalized=normalized)
    output_dir = Path(output_dir)
    suffix = "normalized" if normalized else "raw"
    paths: dict[str, Path] = {}
    for error_metric in error_metrics:
        metric_scale = str(_resolve_spec(yscale, error_metric))
        metric_linthresh = float(_resolve_spec(y_symlog_linthresh, error_metric))
        path = output_dir / (
            f"{error_metric}_all_diagnostics_summary_mean_loess_{suffix}.png"
        )
        paths[error_metric] = plot_mean_loess_grid(
            data,
            source_means,
            error_metric,
            path,
            diagnostics=diagnostics,
            models=models,
            normalized=normalized,
            lowess_frac=lowess_frac,
            x_symlog_linthresh=x_symlog_linthresh,
            yscale=metric_scale,
            y_symlog_linthresh=metric_linthresh,
        )
    return {"data": data, "source_means": source_means, "paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics", nargs="+", choices=DIAGNOSTICS, default=list(DIAGNOSTICS)
    )
    parser.add_argument(
        "--error-metrics",
        nargs="+",
        choices=ERROR_METRICS,
        default=list(ERROR_METRICS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ASSUMED_MODELS, default=list(DEFAULT_MODELS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--lowess-frac", type=float, default=DEFAULT_LOWESS_FRAC)
    parser.add_argument(
        "--x-symlog-linthresh", type=float, default=DEFAULT_X_SYMLOG_LINTHRESH
    )
    parser.add_argument("--posterior-yscale", choices=AXIS_SCALES, default="linear")
    parser.add_argument("--logml-yscale", choices=AXIS_SCALES, default="symlog")
    parser.add_argument("--pmp-yscale", choices=AXIS_SCALES, default="linear")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_mean_loess_comparison(
        diagnostics=tuple(args.diagnostics),
        error_metrics=tuple(args.error_metrics),
        models=tuple(args.models),
        output_dir=args.output_dir,
        normalized=args.normalized,
        lowess_frac=args.lowess_frac,
        x_symlog_linthresh=args.x_symlog_linthresh,
        yscale={
            "posterior_mmd": args.posterior_yscale,
            "log10_logml_error": args.logml_yscale,
            "pmp_error": args.pmp_yscale,
        },
    )
    for error_metric, path in result["paths"].items():
        print(f"{error_metric}: {path}")


if __name__ == "__main__":
    main()
