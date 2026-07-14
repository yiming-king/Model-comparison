from __future__ import annotations

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..config import MODEL_TITLES, RESULT_DIR, TrainingConfig
from .multisource_pipeline import all_observed_paths
from .plots import DATASET_COLORS


SUMMARY_SPECS = (
    ("S=D", TrainingConfig(summary_multiplier=1)),
    ("S=2D", TrainingConfig(summary_multiplier=2)),
    ("S=4D", TrainingConfig(summary_multiplier=4)),
    ("S=6D", TrainingConfig(summary_multiplier=6)),
)
PAIR_MODELS = {
    "m0_m2": ("m0", "m2"),
    "m1_m3": ("m1", "m3"),
}
SOURCE_LABELS = {
    "empirical": "empirical",
    "simulated_from_m0": "simulated from M0",
    "simulated_from_m1": "simulated from M1",
    "simulated_from_m2": "simulated from M2",
    "simulated_from_m3": "simulated from M3",
    "m3_fast_30": "M3 fast 30",
    "m3_slow_30": "M3 slow 30",
    "m3_fast_slow_30": "M3 fast+slow 30",
}
REGIME_COLORS = {
    "low surprise": "#E69F00",
    "in_distribution": "#0072B2",
    "high surprise": "#CC79A7",
}
MARKERS = {
    False: {"marker": "o", "size": 18, "label": "all high surprise"},
    True: {"marker": "D", "size": 28, "label": "at least one not high surprise"},
}
PLOT_FONT = {"strip": 16, "label": 15, "ylabel": 16, "tick": 12, "legend": 11}
PMP_ERROR_BOUND = 0.05
LOGML_ERROR_BOUND = 10.0
POSTERIOR_MMD_BOUND = 0.40
RHO_X_MIN = -1.6
RHO_X_MAX = 4.0
BOUNDARY_LINE_COLOR = "#1300E0"
MAX_LINE_COLOR = "#7A0276"


def _metric_suffix(metric: str) -> str:
    return "" if metric == "l2" else f"_{metric}"


def _comparison_directory(metric: str) -> str:
    return "summary_dimension_comparison" if metric == "l2" else f"summary_dimension_comparison_{metric}_reference"


def _pair_paths(tag: str, metric: str, pair_key: str) -> dict[str, Path]:
    base = RESULT_DIR / "model_pairs" / pair_key
    suffix = _metric_suffix(metric)
    return {
        "diagnostic": base / f"npe_{tag}_all_observed_diagnostic{suffix}.csv",
        "posterior_plot": base / "posterior_diagnostics" / f"npe_{tag}_all_observed_posterior_plot{suffix}.csv",
    }


def _recompute_pair_diagnostics(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    metric: str,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Recompute pair-level surprise and ambiguity after replacing distances."""
    output = frame.copy()
    ranking_distances = []
    for model in models:
        distance = output[f"d_{model}"].to_numpy(dtype=float)
        if metric == "typical":
            low = output[f"dm_low_{model}"].to_numpy(dtype=float)
            high = output[f"dm_high_{model}"].to_numpy(dtype=float)
            distance = np.maximum(low - distance, distance - high).clip(min=0.0)
        ranking_distances.append(distance)

    distance_matrix = np.column_stack(ranking_distances)
    order = np.argsort(distance_matrix, axis=1)
    row_index = np.arange(len(output))
    output["closest_summary_model"] = [models[index] for index in order[:, 0]]
    output["d_min"] = distance_matrix[row_index, order[:, 0]]
    output["d_second"] = distance_matrix[row_index, order[:, 1]]
    output["summary_ambiguity_true"] = 1.0 / (
        np.abs(output["d_second"] - output["d_min"]) + eps
    )
    output["globally_high_surprise"] = output[
        [f"regime_{model}" for model in models]
    ].eq("high surprise").all(axis=1)
    output["at_least_one_not_high_surprise"] = ~output["globally_high_surprise"]
    output["summary_ambiguity"] = np.where(
        output["globally_high_surprise"], output["summary_ambiguity_true"], 0.0
    )
    output["log1p_summary_ambiguity"] = np.log1p(output["summary_ambiguity"])
    return output


def _pair_metric_fallback(
    config: TrainingConfig,
    metric: str,
    pair_key: str,
) -> dict[str, pd.DataFrame]:
    """Combine pairwise fit results with all-model distances for a missing metric cache."""
    if pair_key not in PAIR_MODELS:
        raise ValueError(f"Unknown pair_key: {pair_key}")

    models = PAIR_MODELS[pair_key]
    pair_paths = _pair_paths(config.summary_label, "l2", pair_key)
    all_paths = all_observed_paths(config.summary_label, metric)
    required = [*pair_paths.values(), all_paths["diagnostic"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing cached diagnostics:\n" + "\n".join(missing))

    pair_diagnostic = pd.read_csv(pair_paths["diagnostic"], keep_default_na=False)
    metric_diagnostic = pd.read_csv(all_paths["diagnostic"], keep_default_na=False)
    pair_posterior = pd.read_csv(pair_paths["posterior_plot"], keep_default_na=False)

    metric_columns = []
    for model in models:
        metric_columns.extend(
            [f"d_{model}", f"rho_{model}", f"dm_low_{model}", f"dm_high_{model}", f"regime_{model}"]
        )
        metric_columns.extend(
            column
            for column in metric_diagnostic.columns
            if column.startswith("typicality_") and column.endswith(f"_{model}")
        )
    metric_columns = list(dict.fromkeys(metric_columns))

    pair_diagnostic = pair_diagnostic.drop(columns=metric_columns, errors="ignore").merge(
        metric_diagnostic[["dataset", "id", *metric_columns]],
        on=["dataset", "id"],
        how="left",
        validate="one_to_one",
    )
    pair_diagnostic = _recompute_pair_diagnostics(pair_diagnostic, models, metric)

    metadata_columns = [
        "summary_ambiguity",
        "summary_ambiguity_true",
        "log1p_summary_ambiguity",
        "globally_high_surprise",
        "at_least_one_not_high_surprise",
        "rho",
        "rho_low",
    ]
    pair_posterior = pair_posterior.drop(columns=metadata_columns, errors="ignore")
    metadata = []
    for model in models:
        metadata.append(
            pair_diagnostic[
                [
                    "dataset",
                    "id",
                    "summary_ambiguity",
                    "summary_ambiguity_true",
                    "log1p_summary_ambiguity",
                    "globally_high_surprise",
                    "at_least_one_not_high_surprise",
                ]
            ].assign(
                model=model,
                rho=pair_diagnostic[f"rho_{model}"].to_numpy(),
                rho_low=(pair_diagnostic[f"dm_low_{model}"] / pair_diagnostic[f"dm_high_{model}"]).to_numpy(),
            )
        )
    pair_posterior = pair_posterior.merge(
        pd.concat(metadata, ignore_index=True),
        on=["dataset", "id", "model"],
        how="left",
        validate="one_to_one",
    )
    return {"diagnostic": pair_diagnostic, "posterior_plot": pair_posterior}


def _load_frames(config: TrainingConfig, metric: str, pair_key: str | None) -> dict[str, pd.DataFrame]:
    paths = (
        _pair_paths(config.summary_label, metric, pair_key)
        if pair_key
        else {name: path for name, path in all_observed_paths(config.summary_label, metric).items() if name in {"diagnostic", "posterior_plot"}}
    )
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        if pair_key and metric != "l2":
            return _pair_metric_fallback(config, metric, pair_key)
        raise FileNotFoundError("Missing cached diagnostics:\n" + "\n".join(missing))
    return {name: pd.read_csv(path, keep_default_na=False) for name, path in paths.items()}


def load_summary_dimension_data(
    metric: str = "l2",
    pair_key: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load cached diagnostic and posterior-plot frames for all summary dimensions."""
    if pair_key and metric != "l2":
        first_config = SUMMARY_SPECS[0][1]
        if not all(path.exists() for path in _pair_paths(first_config.summary_label, metric, pair_key).values()):
            warnings.warn(
                f"Pair-specific {metric!r} caches are missing for {pair_key}; "
                "using the pairwise fit results with matching all-model distance diagnostics.",
                stacklevel=2,
            )
    diagnostic = {}
    posterior = {}
    for label, config in SUMMARY_SPECS:
        frames = _load_frames(config, metric, pair_key)
        diagnostic[label] = frames["diagnostic"].assign(summary=label)
        posterior[label] = frames["posterior_plot"].assign(summary=label)
    return diagnostic, posterior


def _filter_sources(frames: dict[str, pd.DataFrame], source_group: str) -> dict[str, pd.DataFrame]:
    if source_group not in {"simulated", "empirical"}:
        raise ValueError("source_group must be 'simulated' or 'empirical'")
    return {
        label: frame.loc[frame["dataset"].ne("empirical") if source_group == "simulated" else frame["dataset"].eq("empirical")].copy()
        for label, frame in frames.items()
    }


def pmp_long(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model in models:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "model_title": MODEL_TITLES[model],
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "signed_pmp_error": frame[f"signed_pmp_error_{model}"],
                    "log1p_summary_ambiguity": frame["log1p_summary_ambiguity"],
                    "at_least_one_not_high_surprise": frame["at_least_one_not_high_surprise"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def logml_long(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for model in models:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "signed_logml_error": frame[f"signed_logml_error_{model}"],
                    "at_least_one_not_high_surprise": frame["at_least_one_not_high_surprise"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _posterior_long(
    diagnostic: pd.DataFrame,
    posterior: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    columns = ["dataset", "id", "model", "posterior_mmd"]
    posterior = posterior.loc[posterior["model"].isin(models), columns]
    flags = pmp_long(diagnostic, models)[
        ["dataset", "id", "model", "rho", "rho_low", "at_least_one_not_high_surprise"]
    ]
    return posterior.merge(flags, on=["dataset", "id", "model"], how="left", validate="one_to_one")


def _logml_posterior_long(
    diagnostic: pd.DataFrame,
    posterior: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    logml = logml_long(diagnostic, models)[
        ["dataset", "id", "model", "signed_logml_error", "at_least_one_not_high_surprise"]
    ]
    posterior = posterior.loc[posterior["model"].isin(models), ["dataset", "id", "model", "posterior_mmd"]]
    return posterior.merge(logml, on=["dataset", "id", "model"], how="left", validate="one_to_one")


def _pmp_posterior_long(
    diagnostic: pd.DataFrame,
    posterior: pd.DataFrame,
    models: tuple[str, ...],
) -> pd.DataFrame:
    pmp = pmp_long(diagnostic, models)
    posterior = posterior.loc[posterior["model"].isin(models), ["dataset", "id", "model", "posterior_mmd"]]
    return pmp.merge(posterior, on=["dataset", "id", "model"], how="left", validate="one_to_one")


def _padded_limits(values, include_zero: bool = False, padding: float = 0.12) -> tuple[float, float]:
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if include_zero:
        values = np.append(values, 0.0)
    if not len(values):
        return -1.0, 1.0
    low, high = float(values.min()), float(values.max())
    span = high - low
    margin = padding * span if span > 0 else max(abs(low) * padding, 0.1)
    return low - margin, high + margin


def _model_label(model: str) -> str:
    return rf"Assumed $M_{{{model.removeprefix('m')}}}$"


def _make_grid(row_labels: list[str], models: tuple[str, ...], sharey: bool = False):
    fig, axes = plt.subplots(
        len(row_labels),
        len(models),
        figsize=(4.8 * len(models) + 2.0, 3.2 * len(row_labels)),
        sharey=sharey,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.17, top=0.88, wspace=0.12, hspace=0.28)
    return fig, axes


def _add_facet_strips(fig, axes, row_labels: list[str], models: tuple[str, ...]) -> None:
    fig.canvas.draw()
    for ax, model in zip(axes[0], models, strict=False):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x0, pos.y1 + 0.006, pos.width, 0.046])
        strip.set_facecolor("#D9D9D9")
        strip.text(0.5, 0.5, _model_label(model), ha="center", va="center", fontsize=PLOT_FONT["strip"])
        strip.set_xticks([])
        strip.set_yticks([])
    for ax, label in zip(axes[:, -1], row_labels, strict=False):
        pos = ax.get_position()
        strip = fig.add_axes([pos.x1 + 0.006, pos.y0, 0.052, pos.height])
        strip.set_facecolor("#D9D9D9")
        strip.text(0.5, 0.5, label, ha="center", va="center", rotation=-90, fontsize=PLOT_FONT["strip"])
        strip.set_xticks([])
        strip.set_yticks([])


def _scatter(ax, data: pd.DataFrame, x: str, y: str, color_col: str | None = None, vmin=None, vmax=None):
    artist = None
    for flag, style in MARKERS.items():
        group = data.loc[data["at_least_one_not_high_surprise"].eq(flag)]
        if group.empty:
            continue
        if color_col:
            artist = ax.scatter(
                group[x], group[y], c=group[color_col], cmap="viridis", vmin=vmin, vmax=vmax,
                s=style["size"], marker=style["marker"], alpha=0.64,
                edgecolors="black", linewidths=0.5,
            )
        else:
            for dataset, part in group.groupby("dataset", sort=False):
                artist = ax.scatter(
                    part[x], part[y], color=DATASET_COLORS.get(dataset, "0.45"),
                    s=style["size"], marker=style["marker"], alpha=0.64,
                    edgecolors="black", linewidths=0.5,
                )
    return artist


def _scatter_with_means(ax, data: pd.DataFrame, x: str, y: str) -> None:
    _scatter(ax, data, x, y)
    for dataset, part in data.groupby("dataset", sort=False):
        ax.scatter(
            [part[x].mean()], [part[y].mean()], marker="*", s=145,
            color=DATASET_COLORS.get(dataset, "0.45"), edgecolors="black", linewidths=0.9, zorder=7,
        )


def _add_empirical_rug(ax, data: pd.DataFrame, x: str) -> None:
    """Show one marginal tick per empirical participant without altering coordinates."""
    if data.empty or not data["dataset"].eq("empirical").all():
        return
    ax.plot(
        data[x],
        np.full(len(data), 0.025),
        transform=ax.get_xaxis_transform(),
        linestyle="none",
        marker="|",
        markersize=6,
        markeredgewidth=0.9,
        color="black",
        alpha=0.65,
        zorder=8,
    )
    ax.text(
        0.02,
        0.96,
        f"n = {len(data)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="0.25",
    )


def _add_rho_regions(ax, rho_low: float, x_min: float, x_max: float) -> None:
    ax.axvspan(x_min, rho_low, color=REGIME_COLORS["low surprise"], alpha=0.08)
    ax.axvspan(rho_low, 1.0, color=REGIME_COLORS["in_distribution"], alpha=0.07)
    ax.axvspan(1.0, x_max, color=REGIME_COLORS["high surprise"], alpha=0.07)
    ax.axvline(rho_low, color=REGIME_COLORS["low surprise"], linestyle="--", linewidth=1.0)
    ax.axvline(1.0, color=REGIME_COLORS["high surprise"], linestyle="--", linewidth=1.0)


def _max_within_rho(data: pd.DataFrame, y_col: str, signed: bool = True):
    data = data.loc[data["rho"].le(1.0)]
    if data.empty:
        return None
    score = data[y_col].abs() if signed else data[y_col]
    return data.loc[score.idxmax()]


def _add_max_line(ax, data: pd.DataFrame, y_col: str, signed: bool) -> None:
    row = _max_within_rho(data, y_col, signed)
    if row is None:
        return
    y0 = float(row[y_col])
    ax.axhline(y0, color=MAX_LINE_COLOR, linestyle="--", linewidth=1.1)
    ax.text(
        0.98, y0, f"{y0:.2g}", transform=ax.get_yaxis_transform(), ha="right",
        va="bottom" if y0 >= 0 else "top", color=MAX_LINE_COLOR, fontsize=10,
        clip_on=True, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
    )


def _add_boundary_line(ax, data: pd.DataFrame, x_col: str, y_col: str, bound: float | None, signed: bool) -> None:
    if bound is None:
        return
    score = data[y_col].abs() if signed else data[y_col]
    hits = data.loc[score.gt(bound)].sort_values(x_col)
    if hits.empty:
        return
    x0 = float(hits.iloc[0][x_col])
    ax.axvline(x0, color=BOUNDARY_LINE_COLOR, linestyle=":", linewidth=1.5)
    ax.text(
        x0, 0.03, f"{x0:.2g}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
        fontsize=10, rotation=90, color=BOUNDARY_LINE_COLOR, clip_on=True,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 1.0},
    )


def _legend_handles(data: pd.DataFrame, include_means: bool = False) -> list[Line2D]:
    handles = [
        Line2D([0], [0], marker=style["marker"], color="none", markerfacecolor="0.5", markeredgecolor="black", markersize=7, label=style["label"])
        for style in MARKERS.values()
    ]
    if include_means:
        handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor="0.5", markeredgecolor="black", markersize=11, label="source mean"))
    handles.extend(
        Line2D([0], [0], marker="o", color="none", markerfacecolor=DATASET_COLORS.get(dataset, "0.45"), markeredgecolor="black", markersize=7, label=SOURCE_LABELS.get(dataset, dataset))
        for dataset in data["dataset"].drop_duplicates()
    )
    return handles


def _finish(fig, axes, row_labels, models, ylabel, data, include_means=False) -> None:
    for ax in axes.ravel():
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=PLOT_FONT["tick"])
    _add_facet_strips(fig, axes, row_labels, models)
    fig.supylabel(ylabel, x=0.035, fontsize=PLOT_FONT["ylabel"])
    handles = _legend_handles(data, include_means=include_means)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=min(5, len(handles)), frameon=False, fontsize=PLOT_FONT["legend"])


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_rho_grid(
    data_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    y_col: str,
    ylabel: str,
    path: Path,
    sharey: bool = True,
    color_col: str | None = None,
    max_line: bool = False,
    y_bound: float | None = None,
    signed_y: bool = True,
    source_means: bool = False,
) -> Path:
    row_labels = list(data_by_summary)
    all_data = pd.concat(data_by_summary.values(), ignore_index=True)
    fig, axes = _make_grid(row_labels, models, sharey=sharey)
    shared_y = _padded_limits(all_data[y_col], include_zero=True) if sharey else None
    vmin = float(all_data[color_col].min()) if color_col else None
    vmax = float(all_data[color_col].max()) if color_col else None
    last = None
    for r, label in enumerate(row_labels):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = data_by_summary[label].loc[data_by_summary[label]["model"].eq(model)]
            x_min, x_max = _padded_limits(sub["rho"], include_zero=True, padding=0.20)
            x_min, x_max = min(RHO_X_MIN, x_min), max(RHO_X_MAX, x_max)
            _add_rho_regions(ax, float(sub["rho_low"].iloc[0]), x_min, x_max)
            if source_means:
                _scatter_with_means(ax, sub, "rho", y_col)
            else:
                last = _scatter(ax, sub, "rho", y_col, color_col=color_col, vmin=vmin, vmax=vmax)
            _add_empirical_rug(ax, sub, "rho")
            if max_line:
                _add_max_line(ax, sub, y_col, signed_y)
            _add_boundary_line(ax, sub, "rho", y_col, y_bound, signed_y)
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_xscale("symlog", linthresh=1.0, linscale=1.0)
            ax.set_xlim(x_min, x_max)
            if shared_y:
                ax.set_ylim(*shared_y)
            else:
                ax.set_ylim(*_padded_limits(sub[y_col], include_zero=True))
            ax.set_xlabel(rf"signed $\rho_{{{model.removeprefix('m')}}}(y)$" if r == len(row_labels) - 1 else "")
    _finish(fig, axes, row_labels, models, ylabel, all_data, include_means=source_means)
    if color_col and last is not None:
        cbar = fig.colorbar(last, ax=axes, fraction=0.018, pad=0.02)
        cbar.set_label("Posterior MMD")
    return _save(fig, path)


def plot_logml_vs_posterior_mmd(
    data_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    path: Path,
) -> Path:
    row_labels = list(data_by_summary)
    all_data = pd.concat(data_by_summary.values(), ignore_index=True)
    fig, axes = _make_grid(row_labels, models, sharey=False)
    for r, label in enumerate(row_labels):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = data_by_summary[label].loc[data_by_summary[label]["model"].eq(model)]
            _scatter(ax, sub, "posterior_mmd", "signed_logml_error")
            ax.axvline(POSTERIOR_MMD_BOUND, color="0.30", linestyle="--", linewidth=1.0)
            ax.axhline(LOGML_ERROR_BOUND, color="0.30", linestyle="--", linewidth=1.0)
            ax.axhline(-LOGML_ERROR_BOUND, color="0.30", linestyle="--", linewidth=1.0)
            ax.axhline(0.0, color="0.45", linewidth=0.8)
            ax.set_xlim(*_padded_limits(sub["posterior_mmd"], include_zero=True))
            ax.set_ylim(*_padded_limits(sub["signed_logml_error"], include_zero=True))
            ax.set_xlabel("Posterior MMD" if r == len(row_labels) - 1 else "")
    _finish(
        fig, axes, row_labels, models,
        r"$\log\widehat{p}(y\mid M_j)-\log p(y\mid M_j)$", all_data,
    )
    return _save(fig, path)


def plot_ambiguity_grid(
    data_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    path: Path,
) -> Path:
    row_labels = list(data_by_summary)
    all_data = pd.concat(data_by_summary.values(), ignore_index=True)
    fig, axes = _make_grid(row_labels, models, sharey=True)
    x_limits = _padded_limits(all_data["log1p_summary_ambiguity"], include_zero=True)
    y_limits = _padded_limits(all_data["signed_pmp_error"], include_zero=True)
    for r, label in enumerate(row_labels):
        for c, model in enumerate(models):
            ax = axes[r, c]
            sub = data_by_summary[label].loc[data_by_summary[label]["model"].eq(model)]
            _scatter(ax, sub, "log1p_summary_ambiguity", "signed_pmp_error")
            _add_boundary_line(ax, sub, "log1p_summary_ambiguity", "signed_pmp_error", PMP_ERROR_BOUND, True)
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.axvline(0.0, color="0.35", linestyle=":", linewidth=1.0)
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.set_xlabel(r"$\log(1+A(y))$" if r == len(row_labels) - 1 else "")
    _finish(fig, axes, row_labels, models, r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$", all_data)
    return _save(fig, path)


def max_pmp_error_within_rho_table(
    data_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
) -> pd.DataFrame:
    rows = []
    for label, data in data_by_summary.items():
        for model in models:
            row = _max_within_rho(data.loc[data["model"].eq(model)], "signed_pmp_error", True)
            if row is not None:
                rows.append(
                    {
                        "summary": label,
                        "model": model,
                        "dataset": row["dataset"],
                        "id": row["id"],
                        "rho": row["rho"],
                        "signed_pmp_error": row["signed_pmp_error"],
                    }
                )
    return pd.DataFrame(rows)


def largest_error_table(
    diagnostic_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    n: int | None = None,
) -> pd.DataFrame:
    data = pd.concat(diagnostic_by_summary.values(), ignore_index=True)
    logml = [f"signed_logml_error_{model}" for model in models]
    pmp = [f"signed_pmp_error_{model}" for model in models]
    rho = [f"rho_{model}" for model in models]
    regimes = [f"regime_{model}" for model in models]
    data["max_abs_logml_error"] = data[logml].abs().max(axis=1)
    data["max_abs_pmp_error"] = data[pmp].abs().max(axis=1)
    columns = ["summary", "dataset", "id"] + regimes + rho + logml + pmp + ["max_abs_logml_error", "max_abs_pmp_error"]
    result = data[columns].sort_values("max_abs_logml_error", ascending=False)
    return result if n is None else result.head(n)


def _default_output_root(metric: str, pair_key: str | None) -> Path:
    base = RESULT_DIR / "model_pairs" / pair_key if pair_key else RESULT_DIR
    return base / "plots" / _comparison_directory(metric)


def run_summary_dimension_comparison(
    models: tuple[str, ...] = ("m0", "m1", "m2", "m3"),
    metric: str = "l2",
    pair_key: str | None = None,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    """Generate the same comparison suite separately for simulated and empirical data."""
    diagnostic_all, posterior_all = load_summary_dimension_data(metric=metric, pair_key=pair_key)
    output_root = Path(output_root) if output_root else _default_output_root(metric, pair_key)
    manifest = []
    tables = {}

    for source_group in ("simulated", "empirical"):
        output_dir = output_root / source_group
        diagnostic = _filter_sources(diagnostic_all, source_group)
        posterior = _filter_sources(posterior_all, source_group)
        pmp = {label: pmp_long(frame, models) for label, frame in diagnostic.items()}
        logml = {label: logml_long(frame, models) for label, frame in diagnostic.items()}
        posterior_mmd = {
            label: _posterior_long(diagnostic[label], posterior[label], models) for label in diagnostic
        }
        logml_posterior = {
            label: _logml_posterior_long(diagnostic[label], posterior[label], models) for label in diagnostic
        }
        pmp_posterior = {
            label: _pmp_posterior_long(diagnostic[label], posterior[label], models) for label in diagnostic
        }

        plot_specs = (
            (plot_logml_vs_posterior_mmd, logml_posterior, "combined_logml_error_vs_posterior_mmd.png", {}),
            (plot_rho_grid, posterior_mmd, "combined_posterior_mmd_vs_rho.png", {"y_col": "posterior_mmd", "ylabel": "Posterior MMD", "max_line": True, "y_bound": POSTERIOR_MMD_BOUND, "signed_y": False}),
            (plot_rho_grid, logml, "combined_logml_error_vs_rho.png", {"y_col": "signed_logml_error", "ylabel": r"$\log\widehat{p}(y\mid M_j)-\log p(y\mid M_j)$", "sharey": False, "max_line": True, "y_bound": LOGML_ERROR_BOUND}),
            (plot_rho_grid, pmp, "combined_pmp_error_vs_rho.png", {"y_col": "signed_pmp_error", "ylabel": r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$", "max_line": True, "y_bound": PMP_ERROR_BOUND}),
            (plot_rho_grid, pmp_posterior, "combined_pmp_error_vs_rho_colored_by_posterior_mmd.png", {"y_col": "signed_pmp_error", "ylabel": r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$", "color_col": "posterior_mmd", "y_bound": PMP_ERROR_BOUND}),
            (plot_rho_grid, posterior_mmd, "combined_posterior_mmd_vs_rho_source_means.png", {"y_col": "posterior_mmd", "ylabel": "Posterior MMD", "source_means": True, "max_line": True, "y_bound": POSTERIOR_MMD_BOUND, "signed_y": False}),
            (plot_rho_grid, logml, "combined_logml_error_vs_rho_source_means.png", {"y_col": "signed_logml_error", "ylabel": r"$\log\widehat{p}(y\mid M_j)-\log p(y\mid M_j)$", "sharey": False, "source_means": True, "max_line": True, "y_bound": LOGML_ERROR_BOUND}),
            (plot_rho_grid, pmp, "combined_pmp_error_vs_rho_source_means.png", {"y_col": "signed_pmp_error", "ylabel": r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$", "source_means": True, "max_line": True, "y_bound": PMP_ERROR_BOUND}),
            (plot_ambiguity_grid, pmp, "combined_pmp_error_vs_log_ambiguity.png", {}),
        )
        for function, data, filename, kwargs in plot_specs:
            path = output_dir / filename
            if function is plot_rho_grid:
                function(data, models=models, path=path, **kwargs)
            else:
                function(data, models=models, path=path)
            manifest.append({"source_group": source_group, "kind": "figure", "path": str(path)})

        group_tables = {
            "max_pmp_error_within_rho": max_pmp_error_within_rho_table(pmp, models),
            "largest_errors": largest_error_table(diagnostic, models),
        }
        for name, table in group_tables.items():
            path = output_dir / f"{name}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False)
            manifest.append({"source_group": source_group, "kind": "table", "path": str(path)})
            tables[f"{source_group}_{name}"] = table

    return {
        "manifest": pd.DataFrame(manifest),
        "tables": tables,
        "output_root": output_root,
    }


def display_summary_dimension_comparison(
    comparison: dict[str, object],
    width: int | None = None,
) -> None:
    """Display all generated figures inside a Jupyter notebook."""
    from IPython.display import Image, Markdown, display

    manifest = comparison["manifest"]
    figures = manifest.loc[manifest["kind"].eq("figure")]
    for source_group in figures["source_group"].drop_duplicates():
        display(Markdown(f"## {source_group.capitalize()} datasets"))
        paths = figures.loc[figures["source_group"].eq(source_group), "path"]
        for path in paths:
            image = Image(filename=str(path), width=width) if width else Image(filename=str(path))
            display(image)
