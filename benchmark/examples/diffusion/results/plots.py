from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import FIGURE_DIR
from .summary_diagnostic import MODELS, logml_diagnostic_frame, pmp_diagnostic_frame


DATASET_COLORS = {
    "empirical": "#000000",
    "m0": "#4C78A8",
    "m1": "#F58518",
    "m2": "#54A24B",
    "m3": "#B279A2",
    "simulated_from_m0": "#4C78A8",
    "simulated_from_m1": "#F58518",
    "simulated_from_m2": "#54A24B",
    "simulated_from_m3": "#B279A2",
}
REGIME_COLORS = {
    "low surprise": "#E69F00",
    "in_distribution": "#0072B2",
    "high surprise": "#CC79A7",
}
MARKERS = {
    False: ("o", 34, "all high surprise"),
    True: ("D", 48, "at least one not high surprise"),
}
PLOT_FONT = {"title": 16, "label": 16, "tick": 14, "legend": 13}


def _model_index(model: str) -> str:
    return model.removeprefix("m")


def _model_symbol(model: str) -> str:
    return rf"M_{{{_model_index(model)}}}"


def _save(fig, filename):
    if filename is None:
        return
    path = Path(filename)
    if not path.is_absolute():
        path = FIGURE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")


def _style_axes(axes) -> None:
    for ax in np.atleast_1d(axes):
        ax.title.set_fontsize(PLOT_FONT["title"])
        ax.xaxis.label.set_size(PLOT_FONT["label"])
        ax.yaxis.label.set_size(PLOT_FONT["label"])
        ax.tick_params(labelsize=PLOT_FONT["tick"])


def _add_rho_regions(ax, rho_low: float, x_max: float) -> None:
    ax.axvspan(0.0, rho_low, color=REGIME_COLORS["low surprise"], alpha=0.08)
    ax.axvspan(rho_low, 1.0, color=REGIME_COLORS["in_distribution"], alpha=0.07)
    ax.axvspan(1.0, x_max, color=REGIME_COLORS["high surprise"], alpha=0.07)
    ax.axvline(rho_low, color=REGIME_COLORS["low surprise"], linestyle="--", linewidth=1.0)
    ax.axvline(1.0, color=REGIME_COLORS["high surprise"], linestyle="--", linewidth=1.0)


def _add_distance_regions(ax, low: float, high: float, x_max: float, x_min: float = 0.0) -> None:
    ax.axvspan(x_min, low, color=REGIME_COLORS["low surprise"], alpha=0.08)
    ax.axvspan(low, high, color=REGIME_COLORS["in_distribution"], alpha=0.07)
    ax.axvspan(high, x_max, color=REGIME_COLORS["high surprise"], alpha=0.07)
    ax.axvline(low, color=REGIME_COLORS["low surprise"], linestyle="--", linewidth=1.0)
    ax.axvline(high, color=REGIME_COLORS["high surprise"], linestyle="--", linewidth=1.0)


def _scatter_by_dataset(ax, data: pd.DataFrame, x: str, y: str):
    for dataset, group in data.groupby("dataset", sort=False):
        ax.scatter(
            group[x],
            group[y],
            s=36,
            color=DATASET_COLORS.get(dataset, "0.45"),
            alpha=0.78,
            edgecolors="black",
            linewidths=0.4,
        )


def _scatter_pmp(ax, data: pd.DataFrame, x: str):
    for flag, (marker, size, _) in MARKERS.items():
        group = data[data["at_least_one_not_high_surprise"] == flag]
        if group.empty:
            continue
        colors = group["dataset"].map(lambda value: DATASET_COLORS.get(value, "0.45"))
        ax.scatter(
            group[x],
            group["signed_pmp_error"],
            s=size,
            c=colors,
            marker=marker,
            alpha=0.78,
            edgecolors="black",
            linewidths=0.55,
        )


def _dataset_handles(frame: pd.DataFrame) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=DATASET_COLORS.get(dataset, "0.45"),
            markeredgecolor="black",
            markersize=7,
            label=str(dataset),
        )
        for dataset in frame["dataset"].drop_duplicates()
    ]


def _marker_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="0.55",
            markeredgecolor="black",
            markersize=7,
            label=label,
        )
        for marker, _, label in MARKERS.values()
    ]


def _pmp_x_values(sub: pd.DataFrame, model: str, x: str):
    index = _model_index(model)
    if x == "distance":
        return (
            sub["distance"],
            float(sub["distance_low"].iloc[0]),
            float(sub["distance_high"].iloc[0]),
            rf"$d_{{{index}}}(y)$",
            0.0,
        )
    if x == "rho":
        return (
            sub["rho"],
            float(sub["rho_low"].iloc[0]),
            1.0,
            rf"$\rho_{{{index}}}(y)$",
            0.0,
        )
    values = np.log(sub["distance"])
    return (
        values,
        float(np.log(sub["distance_low"].iloc[0])),
        float(np.log(sub["distance_high"].iloc[0])),
        rf"$\log d_{{{index}}}(y)$",
        min(float(values.min()), float(np.log(sub["distance_low"].iloc[0]))) - 0.05,
    )


def _signed_max_error_within_rho(
    data: pd.DataFrame,
    y_col: str = "signed_pmp_error",
    rho_col: str = "rho",
    rho_max: float = 1.0,
) -> float | None:
    subset = data.loc[data[rho_col].le(rho_max), [rho_col, y_col]].dropna()
    if subset.empty:
        return None
    return float(subset.loc[subset[y_col].abs().idxmax(), y_col])


def _add_rho_max_error_line(ax, data: pd.DataFrame, y_col: str = "signed_pmp_error") -> None:
    y0 = _signed_max_error_within_rho(data, y_col=y_col)
    if y0 is None:
        return
    ax.axhline(y0, color="#882255", linestyle="--", linewidth=1.1)
    va = "bottom" if y0 >= 0 else "top"
    ax.text(
        0.98,
        y0,
        f"{y0:.2g}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va=va,
        fontsize=11,
        color="#882255",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
    )


def plot_logml_error_vs_rho(frame: pd.DataFrame, filename: str | Path | None = None):
    data = logml_diagnostic_frame(frame)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.1 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, MODELS, strict=False):
        sub = data[data["model"] == model]
        x_max = max(float(sub["rho"].max()) * 1.08, 1.15)
        _add_rho_regions(ax, float(sub["rho_low"].iloc[0]), x_max)
        _scatter_by_dataset(ax, sub, "rho", "signed_logml_error")
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xlim(0.0, x_max)
        ax.set_title(rf"Assumed ${_model_symbol(model)}$")
        ax.set_xlabel(rf"$\rho_{{{_model_index(model)}}}(y)$")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(r"$\log\widehat{ p}(y\mid M_j)-\log p(y\mid M_j)$")
    _style_axes(axes)
    fig.legend(handles=_dataset_handles(data), loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=PLOT_FONT["legend"])
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, filename)
    return fig, axes


def plot_pmp_estimates_vs_distance(
    frame: pd.DataFrame,
    x: str = "distance",
    y: str = "estimate",
    filename: str | Path | None = None,
):
    if x not in {"distance", "rho", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance', 'rho', or 'log_distance'")
    if y not in {"estimate", "signed_error", "signed_pmp_error"}:
        raise ValueError("y must be 'estimate' or 'signed_pmp_error'")

    x = "log_distance" if x == "logdistance" else x
    y = "signed_error" if y == "signed_pmp_error" else y
    data = pmp_diagnostic_frame(frame)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.1 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    last = None

    for ax, model in zip(axes, MODELS, strict=False):
        sub = data[data["model"] == model]
        x_values, low, high, xlabel, x_min = _pmp_x_values(sub, model, x)
        x_max = max(float(x_values.max()) * 1.08, high * 1.08)
        if x == "rho":
            _add_rho_regions(ax, low, x_max)
        else:
            _add_distance_regions(ax, low, high, x_max, x_min=x_min)

        if y == "estimate":
            last = ax.scatter(
                x_values,
                sub["p_npe"],
                c=sub["p_gold"],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=42,
                alpha=0.78,
                marker="o",
                edgecolors="black",
                linewidths=0.45,
                label="NPE",
            )
            ax.scatter(
                x_values,
                sub["p_gold"],
                c=sub["p_gold"],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=88,
                alpha=0.78,
                marker="*",
                edgecolors="black",
                linewidths=0.55,
                label="gold",
            )
            ax.set_ylim(-0.02, 1.02)
        else:
            last = ax.scatter(
                x_values,
                sub["signed_pmp_error"],
                c=sub["p_gold"],
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=42,
                alpha=0.78,
                marker="o",
                edgecolors="black",
                linewidths=0.45,
                label="NPE - gold",
            )
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            _add_rho_max_error_line(ax, sub)
            ax.set_ylim(-1.02, 1.02)

        ax.set_xlim(x_min, x_max)
        ax.set_title(rf"$p({_model_symbol(model)}\mid y)$")
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.18)

    axes[0].set_ylabel("Estimated PMP" if y == "estimate" else "Signed PMP error")
    _style_axes(axes)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, fontsize=PLOT_FONT["legend"])
    if last is not None:
        fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02)
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.18, top=0.88, wspace=0.08)
    _save(fig, filename)
    return fig, axes


def plot_pmp_error_vs_rho(frame: pd.DataFrame, filename: str | Path | None = None):
    data = pmp_diagnostic_frame(frame)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.1 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, MODELS, strict=False):
        sub = data[data["model"] == model]
        x_max = max(float(sub["rho"].max()) * 1.08, 1.15)
        _add_rho_regions(ax, float(sub["rho_low"].iloc[0]), x_max)
        _scatter_pmp(ax, sub, "rho")
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        _add_rho_max_error_line(ax, sub)
        ax.set_xlim(0.0, x_max)
        ax.set_title(rf"$p({_model_symbol(model)}\mid y)$")
        ax.set_xlabel(rf"$\rho_{{{_model_index(model)}}}(y)$")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$")
    _style_axes(axes)
    handles = _marker_handles() + _dataset_handles(data)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=min(len(handles), 4), frameon=False, fontsize=PLOT_FONT["legend"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, filename)
    return fig, axes


def plot_pmp_error_vs_log_ambiguity(frame: pd.DataFrame, filename: str | Path | None = None):
    data = pmp_diagnostic_frame(frame)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.1 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    x_max = max(float(data["log1p_summary_ambiguity"].max()) * 1.08, 0.25)
    for ax, model in zip(axes, MODELS, strict=False):
        sub = data[data["model"] == model]
        _scatter_pmp(ax, sub, "log1p_summary_ambiguity")
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.0)
        ax.set_xlim(-0.05, x_max)
        ax.set_title(rf"$p({_model_symbol(model)}\mid y)$")
        ax.set_xlabel(r"$\log(1+A(y))$")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$")
    _style_axes(axes)
    handles = _marker_handles() + _dataset_handles(data)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=min(len(handles), 4), frameon=False, fontsize=PLOT_FONT["legend"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, filename)
    return fig, axes


def plot_posterior_metric_vs_rho(
    frame: pd.DataFrame,
    metric: str = "posterior_mmd",
    ylabel: str = "Posterior MMD",
    filename: str | Path | None = None,
):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(5.1 * len(MODELS), 4.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, MODELS, strict=False):
        sub = frame[frame["model"] == model]
        x_max = max(float(sub["rho"].max()) * 1.08, 1.15)
        _add_rho_regions(ax, float(sub["rho_low"].iloc[0]), x_max)
        _scatter_by_dataset(ax, sub, "rho", metric)
        ax.set_xlim(0.0, x_max)
        ax.set_title(rf"Assumed ${_model_symbol(model)}$")
        ax.set_xlabel(rf"$\rho_{{{_model_index(model)}}}(y)$")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(ylabel)
    _style_axes(axes)
    fig.legend(handles=_dataset_handles(frame), loc="lower center", bbox_to_anchor=(0.5, -0.11), ncol=5, frameon=False, fontsize=PLOT_FONT["legend"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, filename)
    return fig, axes
