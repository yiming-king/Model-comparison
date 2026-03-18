from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy
from matplotlib.lines import Line2D

DEFAULT_DATASET_DIR = Path(
    "/Users/mandyking/benchmark/benchmark/examples/gaussian/results/datasets"
)
DATASET_GROUPS = ("m1", "m2", "m3")

MISSPECIFICATION_LABELS = {
    "m1": {
        "m1": r"$M_1$" + "\n(Well-specified)",
        "m2": r"$M_2$" + "\n(Prior misspecified)",
        "m3": r"$M_3$" + "\n(Likelihood misspecified)",
    },
    "m2": {
        "m1": r"$M_1$" + "\n(Prior misspecified)",
        "m2": r"$M_2$" + "\n(Well-specified)",
        "m3": r"$M_3$" + "\n(Joint misspecified)",
    },
    "m3": {
        "m1": r"$M_1$" + "\n(Likelihood misspecified)",
        "m2": r"$M_2$" + "\n(Joint misspecified)",
        "m3": r"$M_3$" + "\n(Well-specified)",
    },
}

MISSPECIFICATION_SUBTITLES = {
    "m1": {
        "m1": "Well\nspecified",
        "m2": "Prior\nmisspecified",
        "m3": "Likelihood\nmisspecified",
    },
    "m2": {
        "m1": "Prior\nmisspecified",
        "m2": "Well\nspecified",
        "m3": "Joint\nmisspecified",
    },
    "m3": {
        "m1": "Likelihood\nmisspecified",
        "m2": "Joint\nmisspecified",
        "m3": "Well\nspecified",
    },
}

ASSUMED_MODEL_TITLES = {
    "m1": r"$\mathbf{Assumed\ M_1}$",
    "m2": r"$\mathbf{Assumed\ M_2}$",
    "m3": r"$\mathbf{Assumed\ M_3}$",
}

COMPARISON_SOURCE_ORDER = (
    "npe_posterior_samples",
    "gold_posterior_samples",
)
COMPARISON_SOURCE_LABELS = {
    "npe_posterior_samples": "NPE",
    "gold_posterior_samples": "Gold",
}
COMPARISON_SOURCE_COLORS = {
    "npe_posterior_samples": "#8ecae6",
    "gold_posterior_samples": "#ffb703",
}
COMPARISON_SOURCE_OFFSETS = {
    "npe_posterior_samples": -0.35,
    "gold_posterior_samples": 0.35,
}


def load_pickle(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _shared_y_limits_from_frames(
    frames: list[pd.DataFrame],
    value_col: str,
    y_limits: tuple[float, float] | None = None,
) -> tuple[float, float] | None:
    if y_limits is not None:
        return y_limits

    if not frames:
        return None

    values = np.concatenate([frame[value_col].to_numpy() for frame in frames])
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return None

    ymin = float(finite_values.min())
    ymax = float(finite_values.max())

    if np.isclose(ymin, ymax):
        pad = 0.05 * max(abs(ymax), 1.0)
        return (ymin - pad, ymax + pad)

    pad = 0.05 * (ymax - ymin)
    lower = min(0.0, ymin) if ymin >= 0 else ymin - pad
    upper = ymax + pad
    return (lower, upper)


def _compact_tick_labels(groups: list[str]) -> list[str]:
    return [rf"$M_{group[-1]}$" for group in groups]


def _draw_misspecification_subtitles(
    ax,
    positions: list[float] | np.ndarray,
    assumed_model: str,
    order: list[str],
    tick_fontsize: int,
) -> None:
    for xpos, group in zip(positions, order, strict=False):
        ax.text(
            xpos,
            -0.10,
            MISSPECIFICATION_SUBTITLES[assumed_model][group],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
        )


def _build_single_metric_boxplot(
    df: pd.DataFrame,
    assumed_model: str,
    value_col: str,
    ylabel: str,
    title: str,
    label_map: dict[str, dict[str, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 18,
    tick_fontsize: int = 18,
    ax=None,
    show_ylabel: bool = True,
) -> None:
    label_map = label_map or MISSPECIFICATION_LABELS

    order = [group for group in DATASET_GROUPS if group in set(df["dataset_group"])]
    series = [df.loc[df["dataset_group"] == group, value_col].to_numpy() for group in order]
    tick_positions = np.arange(1, len(order) + 1)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_fig = True

    boxplot = ax.boxplot(
        series,
        positions=tick_positions,
        showmeans=True,
        patch_artist=True,
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor("none")
        patch.set_alpha(1.0)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(_compact_tick_labels(order), fontsize=tick_fontsize)
    _draw_misspecification_subtitles(
        ax=ax,
        positions=tick_positions,
        assumed_model=assumed_model,
        order=order,
        tick_fontsize=12,
    )

    ax.set_title(title, fontsize=title_fontsize)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if created_fig:
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
        plt.show()


def build_metric_boxplot(
    df: pd.DataFrame,
    assumed_model: str,
    value_col: str,
    ylabel: str,
    title: str,
    label_map: dict[str, dict[str, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 18,
    tick_fontsize: int = 18,
    ax=None,
    show_ylabel: bool = True,
) -> None:
    _build_single_metric_boxplot(
        df=df,
        assumed_model=assumed_model,
        value_col=value_col,
        ylabel=ylabel,
        title=title,
        label_map=label_map,
        y_limits=y_limits,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
        ax=ax,
        show_ylabel=show_ylabel,
    )


def build_metric_triptych(
    result_map: dict[str, dict[str, object]],
    value_col: str,
    ylabel: str,
    suptitle: str,
    label_map: dict[str, dict[str, str]] | None = None,
    panel_titles: dict[str, str] | None = None,
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14.5, 4.8),
    title_fontsize: int = 18,
    axis_label_fontsize: int = 18,
    tick_fontsize: int = 18,
) -> None:
    label_map = label_map or MISSPECIFICATION_LABELS
    panel_titles = panel_titles or ASSUMED_MODEL_TITLES

    model_order = [model for model in DATASET_GROUPS if model in result_map]
    frames = [result_map[model]["long_df"] for model in model_order]
    shared_y_limits = _shared_y_limits_from_frames(
        frames=frames,
        value_col=value_col,
        y_limits=y_limits,
    )

    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if len(model_order) == 1:
        axes = [axes]

    for idx, (ax, model_name) in enumerate(zip(axes, model_order, strict=False)):
        _build_single_metric_boxplot(
            df=result_map[model_name]["long_df"],
            assumed_model=model_name,
            value_col=value_col,
            ylabel=ylabel,
            title=panel_titles[model_name],
            label_map=label_map,
            y_limits=shared_y_limits,
            title_fontsize=title_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            ax=ax,
            show_ylabel=False,
        )
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.supylabel(ylabel, fontsize=axis_label_fontsize, x=0.06)
    fig.suptitle(suptitle, fontsize=22, y=1.02)
    fig.tight_layout(rect=[0.03, 0.10, 0.995, 0.95])
    plt.show()


def compute_group_logml_abs_error_rows(
    datasets: list[dict],
    dataset_group: str,
    assumed_model: str,
) -> list[dict]:
    key_gold = f"gold_log_marginal_{assumed_model}"
    key_npe = f"npe_log_marginal_{assumed_model}"

    rows = []
    for ds in datasets:
        gold = float(ds[key_gold])
        npe = float(ds[key_npe])
        rows.append(
            {
                "dataset_group": dataset_group,
                "id": ds["id"],
                "assumed_model": assumed_model,
                "abs_error": abs(npe - gold),
                "gold_log_marginal": gold,
                "npe_log_marginal": npe,
            }
        )
    return rows


def compute_group_mmd_rows(
    datasets: list[dict],
    dataset_group: str,
    assumed_model: str,
) -> list[dict]:
    key_gold = f"gold_post_samples_{assumed_model}"
    key_npe = f"npe_post_samples_{assumed_model}"
    mmd_metric = MaximumMeanDiscrepancy(kernel="gaussian")

    rows: list[dict] = []
    for ds in datasets:
        gold = tf.convert_to_tensor(np.asarray(ds[key_gold], dtype=np.float32))
        npe = tf.convert_to_tensor(np.asarray(ds[key_npe], dtype=np.float32))
        rows.append(
            {
                "dataset_group": dataset_group,
                "id": ds["id"],
                "assumed_model": assumed_model,
                "mmd": float(mmd_metric(npe, gold)),
                "gold_shape": tuple(np.asarray(ds[key_gold]).shape),
                "npe_shape": tuple(np.asarray(ds[key_npe]).shape),
            }
        )
    return rows


def compute_group_logml_comparison_rows(
    datasets: list[dict],
    dataset_group: str,
    assumed_model: str,
) -> list[dict]:
    key_gold = f"gold_log_marginal_{assumed_model}"
    key_npe = f"npe_log_marginal_{assumed_model}"
    key_gp = f"npe_log_marginal_gp_{assumed_model}"

    rows: list[dict] = []
    for ds in datasets:
        gold = float(ds[key_gold])
        npe = float(ds[key_npe])
        gp = float(ds[key_gp])

        for source_name, estimate in {
            "npe_posterior_samples": npe,
            "gold_posterior_samples": gp,
        }.items():
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "id": ds["id"],
                    "assumed_model": assumed_model,
                    "estimate_source": source_name,
                    "estimated_log_marginal": estimate,
                    "gold_log_marginal": gold,
                    "abs_error": abs(estimate - gold),
                    "signed_error": estimate - gold,
                }
            )
    return rows


def run_logml_abs_error_analysis(
    assumed_model: str = "m1",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, object]:
    rows = []

    for dataset_group in DATASET_GROUPS:
        dataset_path = dataset_dir / f"{dataset_group}.pkl"
        datasets = load_pickle(dataset_path)
        rows.extend(
            compute_group_logml_abs_error_rows(
                datasets=datasets,
                dataset_group=dataset_group,
                assumed_model=assumed_model,
            )
        )

    return {"long_df": pd.DataFrame(rows)}


def run_posterior_mmd_analysis(
    assumed_model: str = "m1",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, object]:
    rows = []

    for dataset_group in DATASET_GROUPS:
        dataset_path = dataset_dir / f"{dataset_group}.pkl"
        datasets = load_pickle(dataset_path)
        rows.extend(
            compute_group_mmd_rows(
                datasets=datasets,
                dataset_group=dataset_group,
                assumed_model=assumed_model,
            )
        )

    return {"long_df": pd.DataFrame(rows)}


def run_logml_comparison_analysis(
    assumed_model: str = "m1",
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, object]:
    rows = []

    for dataset_group in DATASET_GROUPS:
        dataset_path = dataset_dir / f"{dataset_group}.pkl"
        datasets = load_pickle(dataset_path)
        rows.extend(
            compute_group_logml_comparison_rows(
                datasets=datasets,
                dataset_group=dataset_group,
                assumed_model=assumed_model,
            )
        )

    return {"long_df": pd.DataFrame(rows)}


def _build_single_logml_comparison_boxplot(
    df: pd.DataFrame,
    assumed_model: str,
    value_col: str,
    ylabel: str,
    title: str,
    label_map: dict[str, dict[str, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    title_fontsize: int = 18,
    axis_label_fontsize: int = 18,
    tick_fontsize: int =18,
    ax=None,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> list[Line2D]:
    label_map = label_map or MISSPECIFICATION_LABELS
    order = [group for group in DATASET_GROUPS if group in set(df["dataset_group"])]
    group_centers = np.arange(len(order)) * 2.3

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
        created_fig = True

    legend_handles: list[Line2D] = []
    for source_name in COMPARISON_SOURCE_ORDER:
        positions = [center + COMPARISON_SOURCE_OFFSETS[source_name] for center in group_centers]
        series = [
            df.loc[
                (df["dataset_group"] == group)
                & (df["estimate_source"] == source_name),
                value_col,
            ].to_numpy()
            for group in order
        ]

        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=0.55,
            showmeans=True,
            patch_artist=True,
        )

        for patch in boxplot["boxes"]:
            patch.set_facecolor(COMPARISON_SOURCE_COLORS[source_name])
            patch.set_alpha(0.8)

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=COMPARISON_SOURCE_COLORS[source_name],
                lw=10,
                label=COMPARISON_SOURCE_LABELS[source_name],
            )
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(_compact_tick_labels(order), fontsize=tick_fontsize)
    _draw_misspecification_subtitles(
        ax=ax,
        positions=group_centers,
        assumed_model=assumed_model,
        order=order,
        tick_fontsize=12,
    )

    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    if show_legend:
        ax.legend(handles=legend_handles, frameon=False)

    if created_fig:
        fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
        plt.show()

    return legend_handles


def build_logml_comparison_boxplot(
    df: pd.DataFrame,
    assumed_model: str,
    value_col: str = "abs_error",
    ylabel: str = "Absolute log marginal likelihood error",
    title: str | None = None,
    label_map: dict[str, dict[str, str]] | None = None,
    y_limits: tuple[float, float] | None = None,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 16,
    ax=None,
    show_ylabel: bool = True,
    show_legend: bool = True,
) -> list[Line2D]:
    title = title or f"Assumed {assumed_model.upper()}: logml comparison"
    return _build_single_logml_comparison_boxplot(
        df=df,
        assumed_model=assumed_model,
        value_col=value_col,
        ylabel=ylabel,
        title=title,
        label_map=label_map,
        y_limits=y_limits,
        title_fontsize=title_fontsize,
        axis_label_fontsize=axis_label_fontsize,
        tick_fontsize=tick_fontsize,
        ax=ax,
        show_ylabel=show_ylabel,
        show_legend=show_legend,
    )


def build_logml_comparison_triptych(
    result_map: dict[str, dict[str, object]],
    value_col: str = "abs_error",
    ylabel: str = "Absolute log marginal likelihood error",
    suptitle: str = "Log marginal likelihood comparison",
    label_map: dict[str, dict[str, str]] | None = None,
    panel_titles: dict[str, str] | None = None,
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (14.5, 4.8),
    title_fontsize: int = 18,
    axis_label_fontsize: int = 18,
    tick_fontsize: int = 18,
) -> None:
    label_map = label_map or MISSPECIFICATION_LABELS
    panel_titles = panel_titles or ASSUMED_MODEL_TITLES

    model_order = [model for model in DATASET_GROUPS if model in result_map]
    frames = [result_map[model]["long_df"] for model in model_order]
    shared_y_limits = _shared_y_limits_from_frames(
        frames=frames,
        value_col=value_col,
        y_limits=y_limits,
    )

    fig, axes = plt.subplots(
        1,
        len(model_order),
        figsize=figsize,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )
    if len(model_order) == 1:
        axes = [axes]

    legend_handles: list[Line2D] | None = None
    for idx, (ax, model_name) in enumerate(zip(axes, model_order, strict=False)):
        legend_handles = _build_single_logml_comparison_boxplot(
            df=result_map[model_name]["long_df"],
            assumed_model=model_name,
            value_col=value_col,
            ylabel=ylabel,
            title=panel_titles[model_name],
            label_map=label_map,
            y_limits=shared_y_limits,
            title_fontsize=title_fontsize,
            axis_label_fontsize=axis_label_fontsize,
            tick_fontsize=tick_fontsize,
            ax=ax,
            show_ylabel=False,
            show_legend=False,
        )
        if idx > 0:
            ax.tick_params(axis="y", labelleft=False)

    fig.supylabel(ylabel, fontsize=axis_label_fontsize, x=0.06)
    fig.suptitle(suptitle, fontsize=22, y=1.02)
    if legend_handles is not None:
        axes[-1].legend(
            handles=legend_handles,
            loc="upper right",
            frameon=True,
            fontsize=12,
            handlelength=1.2,
            borderaxespad=0.8,
            borderpad=0.8,
        )



    fig.tight_layout(rect=[0.03, 0.10, 0.995, 0.92])
    plt.show()


def run_all_posterior_mmd_analysis(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, dict[str, object]]:
    return {
        model_name: run_posterior_mmd_analysis(
            assumed_model=model_name,
            dataset_dir=dataset_dir,
        )
        for model_name in DATASET_GROUPS
    }


def run_all_logml_abs_error_analysis(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, dict[str, object]]:
    return {
        model_name: run_logml_abs_error_analysis(
            assumed_model=model_name,
            dataset_dir=dataset_dir,
        )
        for model_name in DATASET_GROUPS
    }


def run_all_logml_comparison_analysis(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
) -> dict[str, dict[str, object]]:
    return {
        model_name: run_logml_comparison_analysis(
            assumed_model=model_name,
            dataset_dir=dataset_dir,
        )
        for model_name in DATASET_GROUPS
    }
