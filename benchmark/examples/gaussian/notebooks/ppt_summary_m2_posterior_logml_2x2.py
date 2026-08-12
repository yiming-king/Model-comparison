"""Create PPT-ready 2x3 posterior/log-ML/PMP figures for one assumed model."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

from ppt_summary_posterior_logml_4x4 import (
    OUTPUT_DIR,
    PANEL_SPECS,
    SOURCE_COLOR,
    _add_distance_regions,
    _add_facet_strips,
    _load_frames,
    _rho,
    _scatter_source_with_mean,
    _shared_limits,
    _signed_power_formatter,
    _symlog_ticks,
    sd,
)


ASSUMED_MODEL = "m2"
X_LINTHRESH = 0.1
POSTERIOR_Y_LINTHRESH = 0.01
LOGML_BASE_CHANGE = np.log(10.0)
LOGML_Y_LINTHRESH = 1.0 / LOGML_BASE_CHANGE
LOGML_ERROR_BOUND = 1.0
PMP_Y_LINTHRESH = 0.05
PMP_ERROR_BOUND = 0.1
REFERENCE_DISTANCE = 1.0
REFERENCE_X_POSITION = 0.5


def _add_compact_legend(fig: plt.Figure) -> None:
    class_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markersize=12,
            label=label,
        )
        for marker, label in (
            ("o", "all high surprise"),
            ("D", "at least one not high surprise"),
        )
    ]
    model_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=SOURCE_COLOR[source],
            markeredgecolor="black",
            markersize=12,
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
        markersize=26,
        label="source mean",
    )
    fig.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=2,
        frameon=False,
        fontsize=19,
    )
    fig.legend(
        handles=model_handles + [mean_handle],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=13,
        frameon=False,
        fontsize=18,
        columnspacing=1.15,
        handletextpad=0.4,
    )


def _add_shared_column_ylabels(
    fig: plt.Figure,
    axes: np.ndarray,
    labels: list[str],
) -> None:
    for column, label in enumerate(labels):
        top_position = axes[0, column].get_position()
        bottom_position = axes[-1, column].get_position()
        fig.text(
            top_position.x0 - 0.037,
            0.5 * (bottom_position.y0 + top_position.y1),
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=22,
        )


def _set_independent_aligned_xlim(
    ax: plt.Axes,
    data_min: float,
    data_max: float,
) -> tuple[float, float]:
    """Use a panel-specific interval while centering distance=1."""
    transform = ax.xaxis.get_transform()
    inverse = transform.inverted()
    transformed_min, transformed_reference, transformed_max = transform.transform(
        [data_min, REFERENCE_DISTANCE, data_max]
    )
    left_span = (
        (transformed_reference - transformed_min) / REFERENCE_X_POSITION
    )
    right_span = (
        (transformed_max - transformed_reference) / (1.0 - REFERENCE_X_POSITION)
    )
    total_span = max(left_span, right_span)
    aligned_min = float(
        inverse.transform(
            transformed_reference - REFERENCE_X_POSITION * total_span
        )
    )
    aligned_max = float(
        inverse.transform(
            transformed_reference + (1.0 - REFERENCE_X_POSITION) * total_span
        )
    )
    ax.set_xlim(aligned_min, aligned_max)
    return aligned_min, aligned_max


def _set_tight_symlog_ylim(
    ax: plt.Axes,
    values: pd.Series,
    pad_fraction: float,
) -> tuple[float, float]:
    """Pad in transformed space so signed log axes stay tight without clipping."""
    finite_values = values.to_numpy(float)
    finite_values = finite_values[np.isfinite(finite_values)]
    transform = ax.yaxis.get_transform()
    inverse = transform.inverted()
    transformed_min, transformed_max = transform.transform(
        [float(finite_values.min()), float(finite_values.max())]
    )
    transformed_span = transformed_max - transformed_min
    padded_min = float(inverse.transform(transformed_min - pad_fraction * transformed_span))
    padded_max = float(inverse.transform(transformed_max + pad_fraction * transformed_span))
    ax.set_ylim(padded_min, padded_max)
    return padded_min, padded_max


def _scatter_pmp_by_class(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
) -> None:
    styles = {
        False: {"marker": "o", "size": 32},
        True: {"marker": "D", "size": 48},
    }
    for flag, style in styles.items():
        group = data[data["at_least_one_not_extrapolative"].eq(flag)]
        ax.scatter(
            x.loc[group.index],
            y.loc[group.index],
            c=group["source_model"].map(SOURCE_COLOR),
            marker=style["marker"],
            s=style["size"],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.55,
        )


def _add_first_large_pmp_error(
    ax: plt.Axes,
    data: pd.DataFrame,
    x: pd.Series,
    y: pd.Series,
) -> None:
    ordered = (
        pd.DataFrame({"x": x.loc[data.index], "y": y.loc[data.index]})
        .dropna()
        .sort_values("x")
    )
    hit = ordered[np.abs(ordered["y"]) > PMP_ERROR_BOUND].head(1)
    if hit.empty:
        return
    x0 = float(hit["x"].iloc[0])
    ax.axvline(x0, color="0.2", linestyle=":", linewidth=1.2)
    ax.text(
        x0,
        0.96,
        f"{x0:.2g}",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        rotation=90,
        fontsize=18,
    )


def _add_max_pmp_error_within_typical_set(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
) -> None:
    candidates = pd.DataFrame({"x": x, "y": y}).dropna()
    candidates = candidates[candidates["x"].le(REFERENCE_DISTANCE)]
    if candidates.empty:
        return
    y0 = float(candidates.loc[candidates["y"].abs().idxmax(), "y"])
    ax.axhline(y0, color="#882255", linestyle="--", linewidth=1.1)
    ax.text(
        0.98,
        y0,
        f"{y0:.2g}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom" if y0 >= 0 else "top",
        fontsize=13,
        color="#882255",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
    )


def make_figure(metric: str, upper_quantile: float = 0.99):
    frames = _load_frames(metric)
    pmp_frames = [
        pd.read_csv(directory / f"pmp_ambiguity_frame_{tag}.csv")
        for _, directory, tag in PANEL_SPECS[metric]
    ]
    model_number = ASSUMED_MODEL[1:]

    fig, axes = plt.subplots(2, 3, figsize=(24.0, 10.6), squeeze=False)
    fig.subplots_adjust(
        left=0.075,
        right=0.935,
        bottom=0.20,
        top=0.90,
        wspace=0.24,
        hspace=0.20,
    )

    posterior_axes = []
    logml_axes = []
    pmp_axes = []
    plotted_logml_y = []
    x_intervals = []

    for row, ((summary_label, posterior, logml), pmp) in enumerate(
        zip(frames, pmp_frames, strict=False)
    ):
        del summary_label

        posterior_ax = axes[row, 0]
        posterior_sub = posterior[posterior["assumed_model"].eq(ASSUMED_MODEL)].copy()
        posterior_x = _rho(posterior_sub)
        posterior_cutoff = float(np.quantile(posterior_x, upper_quantile))
        posterior_data_min = float(posterior_x.min() * 0.8)
        posterior_data_max = posterior_cutoff * 1.05
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
        posterior_ax.set_xscale("symlog", linthresh=X_LINTHRESH)
        posterior_ax.set_yscale("symlog", linthresh=POSTERIOR_Y_LINTHRESH)
        posterior_interval = _set_independent_aligned_xlim(
            posterior_ax,
            posterior_data_min,
            posterior_data_max,
        )
        posterior_axes.append(posterior_ax)

        logml_ax = axes[row, 1]
        logml_sub = logml[logml["assumed_model"].eq(ASSUMED_MODEL)].copy()
        logml_x = _rho(logml_sub)
        logml_cutoff = float(np.quantile(logml_x, upper_quantile))
        logml_data_min = float(logml_x.min() * 0.8)
        logml_data_max = logml_cutoff * 1.05
        logml_keep = logml_x <= logml_cutoff
        logml_sub = logml_sub.loc[logml_keep].copy()
        logml_x = logml_x.loc[logml_keep]
        # Cached log marginal-likelihood errors use natural logs. Convert the
        # plotted difference to base 10 without taking a log of the x values.
        logml_y = logml_sub["signed_logml_error"] / LOGML_BASE_CHANGE
        plotted_logml_y.append(logml_y.to_numpy(float))
        logml_low = float((logml_sub["dm_low"] / logml_sub["dm_high"]).iloc[0])
        _add_distance_regions(logml_ax, logml_low, 1.0)
        _scatter_source_with_mean(logml_ax, logml_sub, logml_x, logml_y)
        logml_ax.axhline(0, color="0.35", linewidth=0.8)
        for threshold in (-LOGML_ERROR_BOUND, LOGML_ERROR_BOUND):
            logml_ax.axhline(
                threshold,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
        logml_ax.set_xscale("symlog", linthresh=X_LINTHRESH)
        logml_ax.set_yscale("symlog", linthresh=LOGML_Y_LINTHRESH)
        logml_ylim_values = pd.concat(
            [
                logml_y,
                pd.Series([-LOGML_ERROR_BOUND, LOGML_ERROR_BOUND]),
            ],
            ignore_index=True,
        )
        _set_tight_symlog_ylim(
            logml_ax,
            logml_ylim_values,
            pad_fraction=0.025,
        )
        logml_interval = _set_independent_aligned_xlim(
            logml_ax,
            logml_data_min,
            logml_data_max,
        )
        logml_axes.append(logml_ax)

        pmp_ax = axes[row, 2]
        pmp_x = pmp[f"d_{ASSUMED_MODEL}"] / pmp[f"dm_high_{ASSUMED_MODEL}"]
        pmp_y = pmp[f"signed_pmp_error_npe_{ASSUMED_MODEL}"]
        pmp_cutoff = float(np.quantile(pmp_x, upper_quantile))
        pmp_data_min = float(pmp_x.min() * 0.8)
        pmp_data_max = pmp_cutoff * 1.05
        pmp_keep = pmp_x <= pmp_cutoff
        pmp_sub = pmp.loc[pmp_keep].copy()
        pmp_x = pmp_x.loc[pmp_keep]
        pmp_y = pmp_y.loc[pmp_keep]
        pmp_low = float(
            (pmp_sub[f"dm_low_{ASSUMED_MODEL}"] / pmp_sub[f"dm_high_{ASSUMED_MODEL}"]).iloc[0]
        )
        _add_distance_regions(pmp_ax, pmp_low, 1.0)
        _scatter_pmp_by_class(pmp_ax, pmp_sub, pmp_x, pmp_y)
        _add_first_large_pmp_error(pmp_ax, pmp_sub, pmp_x, pmp_y)
        _add_max_pmp_error_within_typical_set(pmp_ax, pmp_x, pmp_y)
        pmp_ax.axhline(0, color="0.35", linewidth=0.8)
        for threshold in (-PMP_ERROR_BOUND, PMP_ERROR_BOUND):
            pmp_ax.axhline(
                threshold,
                color="0.35",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
        pmp_ax.set_xscale("symlog", linthresh=X_LINTHRESH)
        pmp_ax.set_yscale("symlog", linthresh=PMP_Y_LINTHRESH)
        pmp_ylim_values = pd.concat(
            [
                pmp_y,
                pd.Series([-PMP_ERROR_BOUND, PMP_ERROR_BOUND]),
            ],
            ignore_index=True,
        )
        _set_tight_symlog_ylim(pmp_ax, pmp_ylim_values, pad_fraction=0.12)
        pmp_interval = _set_independent_aligned_xlim(
            pmp_ax,
            pmp_data_min,
            pmp_data_max,
        )
        pmp_axes.append(pmp_ax)
        x_intervals.append((posterior_interval, logml_interval, pmp_interval))

    _shared_limits(posterior_axes)
    for ax in pmp_axes:
        pmp_y_min, pmp_y_max = ax.get_ylim()
        pmp_ticks = [
            tick
            for tick in (-1.0, -0.1, 0.0, 0.1, 1.0)
            if pmp_y_min <= tick <= pmp_y_max
        ]
        ax.yaxis.set_major_locator(FixedLocator(pmp_ticks))
        ax.yaxis.set_major_formatter(FuncFormatter(_signed_power_formatter))
        ax.yaxis.set_minor_locator(NullLocator())

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
        ax.tick_params(labelsize=19)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)

    axes[1, 0].set_xlabel(sd.DIAGNOSTIC_XLABELS[metric])
    axes[1, 1].set_xlabel(sd.DIAGNOSTIC_XLABELS[metric])
    axes[1, 2].set_xlabel(sd.DIAGNOSTIC_XLABELS[metric])
    _add_shared_column_ylabels(
        fig,
        axes,
        [
            "Posterior MMD",
            rf"$\log_{{10}} \widehat{{p}}(y\mid M_{{{model_number}}})"
            rf"-\log_{{10}} p(y\mid M_{{{model_number}}})$",
            rf"$\widehat{{p}}(M_{{{model_number}}}\mid y)"
            rf"-p(M_{{{model_number}}}\mid y)$",
        ],
    )

    _add_facet_strips(
        fig,
        axes,
        [
            rf"Assumed $M_{{{model_number}}}$: Posterior MMD",
            rf"Assumed $M_{{{model_number}}}$: $\log_{{10}}$ ML error",
            rf"$p(M_{{{model_number}}}\mid y)$ error",
        ],
        ["S=D", "S=4D"],
        top_height=0.052,
        column_fontsize=22,
        row_fontsize=20,
    )
    _add_compact_legend(fig)

    png_path = OUTPUT_DIR / f"{ASSUMED_MODEL}_{metric}_posterior_logml_pmp_2x3.png"
    pdf_path = OUTPUT_DIR / f"{ASSUMED_MODEL}_{metric}_posterior_logml_pmp_2x3.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"{metric}: independent x intervals; distance=1 centered in every panel")
    for (summary_label, _, _), intervals in zip(frames, x_intervals, strict=False):
        print(
            f"  {summary_label}: posterior={intervals[0]}, "
            f"logml={intervals[1]}, pmp={intervals[2]}"
        )
    print(png_path)
    print(pdf_path)
    return png_path, pdf_path


if __name__ == "__main__":
    for diagnostic_metric in ("l2", "linf"):
        make_figure(diagnostic_metric)
