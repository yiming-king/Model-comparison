from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from bayesflow.metrics import MaximumMeanDiscrepancy

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


def load_pickle(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        return pickle.load(handle)

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
) -> None:
    label_map = label_map or MISSPECIFICATION_LABELS

    order = [group for group in DATASET_GROUPS if group in set(df["dataset_group"])]
    labels = [label_map[assumed_model][group] for group in order]
    series = [df.loc[df["dataset_group"] == group, value_col].to_numpy() for group in order]

    fig, ax = plt.subplots(figsize=(9, 6))
    boxplot = ax.boxplot(
        series,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
    )

    for patch in boxplot["boxes"]:
        patch.set_facecolor("none")
        patch.set_alpha(1.0)

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.tick_params(axis="x", labelsize=tick_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    fig.tight_layout()
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


def compute_group_logml_comparison_rows(
    datasets: list[dict],
    dataset_group: str,
    assumed_model: str,
) -> list[dict]:
    key_gold = f"gold_log_marginal_{assumed_model}"
    key_npe = f"npe_log_marginal_{assumed_model}"
    key_gp = f"npe_log_marginal_gp_{assumed_model}"

    rows = []
    for ds in datasets:
        gold = float(ds[key_gold])
        npe = float(ds[key_npe])
        gp = float(ds[key_gp])

        estimates = {
            "npe_posterior_samples": npe,
            "gold_posterior_samples": gp,
        }

        for estimate_source, estimate_value in estimates.items():
            rows.append(
                {
                    "dataset_group": dataset_group,
                    "id": ds["id"],
                    "assumed_model": assumed_model,
                    "estimate_source": estimate_source,
                    "abs_error": abs(estimate_value - gold),
                    "signed_error": estimate_value - gold,
                    "gold_log_marginal": gold,
                    "estimated_log_marginal": estimate_value,
                }
            )
    return rows


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


def build_logml_comparison_boxplot(
    df: pd.DataFrame,
    assumed_model: str,
    value_col: str = "abs_error",
    ylabel: str = "|Δ log(y)|",
    title: str | None = None,
    label_map: dict[str, dict[str, str]] | None = None,
    title_fontsize: int = 16,
    axis_label_fontsize: int = 16,
    tick_fontsize: int = 16,
) -> None:
    label_map = label_map or MISSPECIFICATION_LABELS
    title = title or f"|Δ log(y)| comparison under assumed model {assumed_model.upper()}"

    source_order = ("npe_posterior_samples", "gold_posterior_samples")
    source_labels = {
        "npe_posterior_samples": "NPE posterior samples",
        "gold_posterior_samples": "Gold posterior samples",
    }
    source_colors = {
        "npe_posterior_samples": "#8ecae6",
        "gold_posterior_samples": "#ffb703",
    }
    source_offsets = {
        "npe_posterior_samples": -0.4,
        "gold_posterior_samples": 0.4,
    }

    order = [group for group in DATASET_GROUPS if group in set(df["dataset_group"])]
    group_centers = np.arange(len(order)) * 3.0

    fig, ax = plt.subplots(figsize=(9, 6))

    legend_handles = []
    for source in source_order:
        positions = [center + source_offsets[source] for center in group_centers]
        series = [
            df.loc[
                (df["dataset_group"] == group) & (df["estimate_source"] == source),
                value_col,
            ].to_numpy()
            for group in order
        ]

        boxplot = ax.boxplot(
            series,
            positions=positions,
            widths=0.65,
            showmeans=True,
            patch_artist=True,
        )

        for patch in boxplot["boxes"]:
            patch.set_facecolor(source_colors[source])
            patch.set_alpha(0.8)

        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=source_colors[source],
                lw=10,
                label=source_labels[source],
            )
        )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [label_map[assumed_model][group] for group in order],
        fontsize=tick_fontsize,
    )
    ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
    ax.set_title(title, fontsize=title_fontsize)
    ax.tick_params(axis="y", labelsize=tick_fontsize)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(handles=legend_handles, frameon=False)

 
    fig.tight_layout()
    plt.show()
