"""Create PPT-ready 4x4 posterior-MMD/log-ML diagnostic figures.

This script is intentionally separate from summary_dimension_comparison.ipynb.
It reads the existing result CSV files but does not modify any existing notebook.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from benchmark.examples.gaussian.analysis import summry_diagnostic as sd


GAUSSIAN_DIR = Path(__file__).resolve().parents[1]
RESULT_DIR = GAUSSIAN_DIR / "results"
OUTPUT_DIR = RESULT_DIR / "ppt_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PANEL_SPECS = {
    "l2": [
        ("S=D", RESULT_DIR / "ood_20d_10n", "20d_10n"),
        ("S=4D", RESULT_DIR / "ood_80d_10n", "80d_10n"),
    ],
    "linf": [
        ("S=D", RESULT_DIR / "ood_20d_10n_inf", "20d_10n_linf"),
        ("S=4D", RESULT_DIR / "ood_80d_10n_inf", "80d_10n_linf"),
    ],
}

SOURCE_COLOR = dict(zip(sd.SOURCE_MODELS, sd.SOURCE_COLORS, strict=False))


def _rho(data: pd.DataFrame) -> pd.Series:
    return data["d_M"] / data["dm_high"]


def _load_frames(metric: str) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
    frames = []
    for label, directory, tag in PANEL_SPECS[metric]:
        posterior = pd.read_csv(directory / f"posterior_distance_frame_{tag}.csv")
        logml = pd.read_csv(directory / f"logml_distance_frame_{tag}.csv")
        frames.append((label, posterior, logml))
    return frames


def _scatter_source_with_mean(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
) -> None:
    for source in sd.SOURCE_MODELS:
        group = data[data["source_model"].eq(source)]
        if group.empty:
            continue
        color = SOURCE_COLOR[source]
        x_values = x.loc[group.index]
        y_values = y.loc[group.index]
        ax.scatter(
            x_values,
            y_values,
            color=color,
            s=22,
            alpha=0.42,
            edgecolors="black",
            linewidths=0.25,
        )
        ax.scatter(
            [x_values.mean()],
            [y_values.mean()],
            color=color,
            marker="*",
            s=280,
            alpha=0.98,
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
        )


def _add_distance_regions(ax: plt.Axes, low: float, high: float) -> None:
    ax.axvspan(low, high, color=sd.TYPICAL_SET_FILL, alpha=0.70, zorder=0)
    ax.axvline(low, color="0.45", linestyle=":", linewidth=0.9, zorder=1)
    ax.axvline(high, color="0.25", linestyle="--", linewidth=0.9, zorder=1)


def _signed_power_formatter(value: float, pos=None) -> str:
    del pos
    if np.isclose(value, 0.0):
        return "0"
    sign = "-" if value < 0 else ""
    abs_value = abs(value)
    exponent = int(np.round(np.log10(abs_value)))
    if np.isclose(abs_value, 10**exponent):
        return rf"${sign}10^{{{exponent}}}$"
    return f"{value:g}"


def _symlog_ticks(y_min: float, y_max: float) -> list[float]:
    ticks = [0.0]
    if y_min < 0:
        min_exp = int(np.floor(np.log10(abs(y_min))))
        ticks.extend([-10.0**exp for exp in range(min_exp, -1, -2)])
        ticks.append(-1.0)
    if y_max > 0:
        max_exp = int(np.floor(np.log10(y_max)))
        ticks.extend([10.0**exp for exp in range(0, max_exp + 1, 2)])
    return sorted(set(tick for tick in ticks if y_min <= tick <= y_max))


def _add_facet_strips(
    fig: plt.Figure,
    axes: np.ndarray,
    column_labels: list[str],
    row_labels: list[str],
    top_height: float = 0.027,
    right_width: float = 0.050,
    column_fontsize: float = 18,
    row_fontsize: float = 16,
) -> None:
    strip_color = "0.85"
    edge_color = "0.35"
    pad = 0.004
    for ax, label in zip(axes[0], column_labels, strict=False):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x0, pos.y1 + pad, pos.width, top_height])
        strip.set_facecolor(strip_color)
        strip.text(
            0.5,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=column_fontsize,
        )
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_color(edge_color)

    for ax, label in zip(axes[:, -1], row_labels, strict=False):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x1 + pad, pos.y0, right_width, pos.height])
        strip.set_facecolor(strip_color)
        strip.text(
            0.5,
            0.5,
            label,
            rotation=-90,
            ha="center",
            va="center",
            fontsize=row_fontsize,
            linespacing=1.05,
        )
        strip.set_xticks([])
        strip.set_yticks([])
        for spine in strip.spines.values():
            spine.set_color(edge_color)


def _add_bottom_legend(fig: plt.Figure) -> None:
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SOURCE_COLOR[source],
            markeredgecolor="black",
            markersize=9,
            label=source.upper(),
        )
        for source in sd.SOURCE_MODELS
    ]
    mean_handle = Line2D(
        [0],
        [0],
        marker="*",
        color="none",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=20,
        label="source mean",
    )
    handles = model_handles + [mean_handle]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=len(handles),
        frameon=False,
        fontsize=17,
    )


def _shared_limits(axes: list[plt.Axes]) -> None:
    lower = min(ax.get_ylim()[0] for ax in axes)
    upper = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(lower, upper)


def make_figure(metric: str, upper_quantile: float = 0.99) -> tuple[Path, Path]:
    frames = _load_frames(metric)
    assumed_models = sd.ASSUMED_MODELS

    posterior_rho = np.concatenate([_rho(posterior).to_numpy(float) for _, posterior, _ in frames])
    posterior_cutoff = float(np.quantile(posterior_rho, upper_quantile))
    posterior_x_min = -0.5
    posterior_x_max = posterior_cutoff * 1.05

    logml_rho = np.concatenate([_rho(logml).to_numpy(float) for _, _, logml in frames])
    positive_logml_rho = logml_rho[logml_rho > 0]
    logml_cutoff = float(np.quantile(positive_logml_rho, upper_quantile))
    logml_x_min = float(positive_logml_rho.min() * 0.8)
    logml_x_max = logml_cutoff * 1.05

    fig, axes = plt.subplots(
        4,
        len(assumed_models),
        figsize=(20.8, 16.0),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.105,
        right=0.925,
        bottom=0.105,
        top=0.925,
        wspace=0.10,
        hspace=0.24,
    )

    posterior_axes = []
    logml_axes = []
    plotted_logml_y = []

    for summary_index, (summary_label, posterior, logml) in enumerate(frames):
        posterior_row = 2 * summary_index
        logml_row = posterior_row + 1

        for column, model in enumerate(assumed_models):
            posterior_ax = axes[posterior_row, column]
            posterior_sub = posterior[posterior["assumed_model"].eq(model)].copy()
            posterior_x = _rho(posterior_sub)
            posterior_keep = posterior_x <= posterior_cutoff
            posterior_sub = posterior_sub.loc[posterior_keep].copy()
            posterior_x = posterior_x.loc[posterior_keep]
            posterior_low = float(
                (posterior_sub["dm_low"] / posterior_sub["dm_high"]).iloc[0]
            )
            _add_distance_regions(posterior_ax, posterior_low, 1.0)
            _scatter_source_with_mean(
                posterior_ax,
                posterior_sub,
                posterior_x,
                posterior_sub["posterior_mmd"],
            )
            posterior_ax.set_xlim(posterior_x_min, posterior_x_max)
            posterior_axes.append(posterior_ax)

            logml_ax = axes[logml_row, column]
            logml_sub = logml[logml["assumed_model"].eq(model)].copy()
            logml_x = _rho(logml_sub)
            logml_keep = (logml_x > 0) & (logml_x <= logml_cutoff)
            logml_sub = logml_sub.loc[logml_keep].copy()
            logml_x = logml_x.loc[logml_keep]
            logml_y = logml_sub["signed_logml_error"]
            plotted_logml_y.append(logml_y.to_numpy(float))
            logml_low = float((logml_sub["dm_low"] / logml_sub["dm_high"]).iloc[0])
            _add_distance_regions(logml_ax, logml_low, 1.0)
            _scatter_source_with_mean(logml_ax, logml_sub, logml_x, logml_y)
            logml_ax.axhline(0, color="0.35", linewidth=0.8)
            logml_ax.set_xscale("log")
            logml_ax.set_yscale("symlog", linthresh=1.0)
            logml_ax.set_xlim(logml_x_min, logml_x_max)
            logml_axes.append(logml_ax)

    _shared_limits(posterior_axes)
    _shared_limits(logml_axes)

    logml_y_values = np.concatenate(plotted_logml_y)
    logml_ticks = _symlog_ticks(
        float(np.nanmin(logml_y_values)),
        float(np.nanmax(logml_y_values)),
    )
    for ax in logml_axes:
        ax.yaxis.set_major_locator(FixedLocator(logml_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(_signed_power_formatter))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.get_offset_text().set_visible(False)

    for ax in axes.ravel():
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=16)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)

    for ax in axes[-1]:
        ax.set_xlabel(sd.DIAGNOSTIC_XLABELS[metric])

    for row in (0, 2):
        axes[row, 0].set_ylabel("Posterior MMD", fontsize=18)
    for row in (1, 3):
        axes[row, 0].set_ylabel(
            r"$\log \widehat{p}(y\mid M_j)-\log p(y\mid M_j)$",
            fontsize=18,
        )

    column_labels = [rf"Assumed $M_{model[1:]}$" for model in assumed_models]
    row_labels = [
        "S=D\nPosterior MMD",
        "S=D\nLogML error",
        "S=4D\nPosterior MMD",
        "S=4D\nLogML error",
    ]
    _add_facet_strips(fig, axes, column_labels, row_labels)
    _add_bottom_legend(fig)

    png_path = OUTPUT_DIR / f"m2_{metric}_posterior_logml_4x4.png"
    pdf_path = OUTPUT_DIR / f"m2_{metric}_posterior_logml_4x4.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(
        f"{metric}: posterior rho <= {posterior_cutoff:.3f}; "
        f"logml rho in [{logml_x_min:.3g}, {logml_cutoff:.3f}]"
    )
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


if __name__ == "__main__":
    for diagnostic_metric in ("l2", "linf"):
        make_figure(diagnostic_metric)
