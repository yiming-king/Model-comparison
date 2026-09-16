"""Plot saved Gaussian PMP results without running inference or diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from benchmark.examples.gaussian.config import ASSUMED_MODELS, SOURCE_MODELS


def load_saved_results(results_dir: str | Path, run_dir: str | Path):
    """Load saved training history and comparison tables for plotting."""
    results_dir, run_dir = Path(results_dir), Path(run_dir)
    history = json.loads((run_dir / "history.json").read_text())
    probabilities = pd.read_csv(results_dir / "probabilities.csv")
    paired = pd.read_csv(results_dir / "paired.csv")
    calibration = pd.read_csv(results_dir / "calibration.csv")
    return history, probabilities, paired, calibration


def _config_label(frame: pd.DataFrame, method: str) -> str:
    column = f"{method}_config"
    return "; ".join(str(value) for value in frame[column].drop_duplicates())


def _source_colors(frame: pd.DataFrame) -> dict:
    present = set(frame["source_model"])
    colors = plt.get_cmap("tab20")
    return {
        source: colors(index)
        for index, source in enumerate(SOURCE_MODELS)
        if source in present
    }


def plot_loss(history: dict):
    """Show saved training and independent-validation categorical losses."""
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for key, label in (("loss", "Training"), ("val_loss", "Validation")):
        values = history[key]
        ax.plot(range(1, len(values) + 1), values, marker=".", label=label)
    ax.set(xlabel="Epoch", ylabel="Categorical cross-entropy", title="Direct PMP loss")
    ax.legend()
    ax.grid(alpha=0.2)
    return fig


def plot_calibration(calibration: pd.DataFrame, probabilities: pd.DataFrame):
    """Show precomputed one-versus-rest calibration on independent ID data."""
    id_rows = calibration.loc[
        (calibration["source_pool"] == "ID") & (calibration["count"] > 0)
    ]
    if id_rows.empty:
        raise ValueError("No nonempty independent ID calibration bins are available")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), layout="constrained")
    for ax, model in zip(axes, ASSUMED_MODELS):
        ax.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
        for method, color in (("direct", "C0"), ("indirect", "C1")):
            rows = id_rows.loc[
                (id_rows["method"] == method) & (id_rows["model"] == model)
            ].sort_values("mean_probability")
            ax.plot(
                rows["mean_probability"], rows["observed_frequency"],
                "o-", color=color, label=method.capitalize(),
            )
        ax.set(
            title=model, xlabel="Mean predicted PMP", xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02), aspect="equal",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Observed model frequency")
    axes[-1].legend()
    fig.suptitle(
        "Independent ID test calibration (m1–m4; equal model prior)\n"
        f"Direct: {_config_label(probabilities, 'direct')}\n"
        f"Indirect: {_config_label(probabilities, 'indirect')}",
        fontsize=10,
    )
    return fig


def plot_pmp_recovery(probabilities: pd.DataFrame, method: str):
    """Compare saved probabilities with analytical gold for each candidate model."""
    if method not in ("direct", "indirect"):
        raise ValueError("method must be 'direct' or 'indirect'")
    colors = _source_colors(probabilities)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), layout="constrained")
    for ax, model in zip(axes, ASSUMED_MODELS):
        ax.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
        for source, color in colors.items():
            rows = probabilities.loc[
                (probabilities["model"] == model)
                & (probabilities["source_model"] == source)
            ]
            ax.scatter(
                rows["p_gold"], rows[f"p_{method}"], s=16, alpha=0.65,
                color=color, marker="o" if source in ASSUMED_MODELS else "x",
                label=f"{source} ({'ID' if source in ASSUMED_MODELS else 'OOD'})",
            )
        ax.set(
            title=model, xlabel="Analytical gold PMP", xlim=(-0.02, 1.02),
            ylim=(-0.02, 1.02), aspect="equal",
        )
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(f"{method.capitalize()} PMP")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle(
        f"{method.capitalize()} PMP recovery\n{_config_label(probabilities, method)}",
        fontsize=10,
    )
    return fig


def plot_paired_tv(paired: pd.DataFrame):
    """Show per-dataset paired TV errors and saved differences by source model."""
    colors = _source_colors(paired)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    axes[0].plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
    axes[1].axhline(0, linestyle="--", color="0.65", linewidth=1)
    for position, (source, color) in enumerate(colors.items()):
        rows = paired.loc[paired["source_model"] == source].sort_values("dataset_id")
        axes[0].scatter(
            rows["TV_indirect"], rows["TV_direct"], s=20, alpha=0.7,
            color=color, label=source,
        )
        # Deterministic offsets keep nearby datasets visible without changing values.
        offsets = [position + 0.03 * (index % 9 - 4) for index in range(len(rows))]
        axes[1].scatter(offsets, rows["delta_TV"], s=20, alpha=0.7, color=color)
    axes[0].set(
        xlabel="Indirect TV to gold", ylabel="Direct TV to gold",
        title="One point per matched dataset", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02),
        aspect="equal",
    )
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].set(
        xlabel="Source model", ylabel="TV direct − TV indirect",
        title="Negative values favor direct", ylim=(-1.02, 1.02),
        xticks=range(len(colors)), xticklabels=list(colors),
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"Direct: {_config_label(paired, 'direct')}\n"
        f"Indirect: {_config_label(paired, 'indirect')}", fontsize=10,
    )
    return fig
