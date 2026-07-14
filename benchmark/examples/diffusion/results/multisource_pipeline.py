from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..approximators.indirect import load_approximator
from ..config import MODELS, RESULT_DIR, TrainingConfig
from ..simulators import SIMULATORS
from .observed_datasets import OBSERVED_DATASETS, load_observed_dataset
from .posterior_diagnostic import (
    load_posterior_diagnostic,
    posterior_diagnostic_frame,
    save_posterior_diagnostic,
)
from .results import attach_gold_standard, estimate_model_comparison, load_gold_standard, save_results
from .summary_diagnostic import (
    add_summary_diagnostics,
    fit_references,
    load_references,
    pmp_diagnostic_frame,
    save_diagnostic,
    save_references,
)


def summary_tag(config: TrainingConfig) -> str:
    return config.summary_label


def load_approximators(config: TrainingConfig) -> dict[str, object]:
    return {model: load_approximator(model, config=config) for model in MODELS}


def reference_path(tag: str, metric: str = "l2") -> Path:
    suffix = "" if metric == "l2" else f"_{metric}"
    return RESULT_DIR / f"npe_{tag}_references{suffix}.pkl"


def all_observed_results_path(tag: str) -> Path:
    return RESULT_DIR / f"npe_{tag}_all_observed_results_gold.csv"


def all_observed_diagnostic_path(tag: str, metric: str = "l2") -> Path:
    suffix = "" if metric == "l2" else f"_{metric}"
    return RESULT_DIR / f"npe_{tag}_all_observed_diagnostic{suffix}.csv"


def all_observed_posterior_path(tag: str) -> Path:
    return RESULT_DIR / "posterior_diagnostics" / f"npe_{tag}_all_observed_posterior.csv"


def all_observed_posterior_plot_path(tag: str, metric: str = "l2") -> Path:
    suffix = "" if metric == "l2" else f"_{metric}"
    return RESULT_DIR / "posterior_diagnostics" / f"npe_{tag}_all_observed_posterior_plot{suffix}.csv"


def all_observed_paths(tag: str, metric: str = "l2") -> dict[str, Path]:
    return {
        "results": all_observed_results_path(tag),
        "diagnostic": all_observed_diagnostic_path(tag, metric=metric),
        "posterior": all_observed_posterior_path(tag),
        "posterior_plot": all_observed_posterior_plot_path(tag, metric=metric),
    }


def covers_observed_datasets(frame: pd.DataFrame) -> bool:
    return set(OBSERVED_DATASETS).issubset(set(frame["dataset"].unique()))


def load_cached_all_observed(config: TrainingConfig, metric: str = "l2") -> dict[str, pd.DataFrame]:
    paths = all_observed_paths(summary_tag(config), metric=metric)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Run multi_source_ood_diagnostic.ipynb first. Missing:\n" + "\n".join(missing))

    frames = {
        "results": pd.read_csv(paths["results"], keep_default_na=False),
        "diagnostic": pd.read_csv(paths["diagnostic"], keep_default_na=False),
        "posterior": load_posterior_diagnostic(paths["posterior"]),
        "posterior_plot": load_posterior_diagnostic(paths["posterior_plot"]),
    }
    incomplete = [name for name, frame in frames.items() if not covers_observed_datasets(frame)]
    if incomplete:
        raise ValueError(f"Cached files do not include all OBSERVED_DATASETS: {incomplete}")
    return frames


def load_or_fit_references(
    approximators: dict[str, object],
    config: TrainingConfig,
    metric: str = "l2",
    overwrite: bool = False,
):
    path = reference_path(summary_tag(config), metric=metric)
    if path.exists() and not overwrite:
        return load_references(path)
    references = fit_references(approximators, SIMULATORS, metric=metric)
    save_references(references, path)
    return references


def compute_all_observed_results(
    approximators: dict[str, object],
    num_samples: int = 2048,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    frames = []
    for dataset in OBSERVED_DATASETS:
        y, ids = load_observed_dataset(dataset)
        estimates = estimate_model_comparison(
            approximators,
            y,
            ids=ids,
            dataset=dataset,
            num_samples=num_samples,
            batch_size=batch_size,
            seed=seed,
        )
        frames.append(attach_gold_standard(estimates, load_gold_standard(dataset=dataset)))
    return pd.concat(frames, ignore_index=True)


def compute_all_observed_summary_diagnostics(
    results: pd.DataFrame,
    approximators: dict[str, object],
    references: dict[str, dict],
) -> pd.DataFrame:
    frames = []
    for dataset in OBSERVED_DATASETS:
        y, _ = load_observed_dataset(dataset)
        frame = results[results["dataset"] == dataset].reset_index(drop=True)
        frames.append(add_summary_diagnostics(frame, y, approximators, references))
    return pd.concat(frames, ignore_index=True)


def compute_all_observed_posterior_diagnostics(
    approximators: dict[str, object],
    num_samples: int = 2048,
    mmd_samples: int = 512,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    frames = []
    for dataset in OBSERVED_DATASETS:
        y, ids = load_observed_dataset(dataset)
        frames.append(
            posterior_diagnostic_frame(
                approximators,
                y,
                ids,
                dataset,
                num_samples=num_samples,
                mmd_samples=mmd_samples,
                batch_size=batch_size,
                seed=seed,
            )
        )
    return pd.concat(frames, ignore_index=True)


def posterior_plot_frame(diagnostic: pd.DataFrame, posterior: pd.DataFrame) -> pd.DataFrame:
    distance = pmp_diagnostic_frame(diagnostic)[
        [
            "dataset",
            "id",
            "model",
            "rho",
            "rho_low",
            "log1p_summary_ambiguity",
            "globally_high_surprise",
            "at_least_one_not_high_surprise",
        ]
    ]
    return posterior.merge(distance, on=["dataset", "id", "model"], how="left")


def compute_or_load_all_observed(
    config: TrainingConfig,
    metric: str = "l2",
    num_samples: int = 2048,
    mmd_samples: int = 512,
    batch_size: int | None = 8,
    recompute: bool = False,
    recompute_references: bool = False,
) -> dict[str, pd.DataFrame]:
    tag = summary_tag(config)
    approximators = load_approximators(config)
    references = load_or_fit_references(approximators, config, metric=metric, overwrite=recompute_references)

    paths = all_observed_paths(tag, metric=metric)
    results_path = paths["results"]
    diagnostic_path = paths["diagnostic"]
    posterior_path = paths["posterior"]
    posterior_plot_path = paths["posterior_plot"]

    if results_path.exists() and not recompute:
        results = pd.read_csv(results_path, keep_default_na=False)
        recompute_results = not covers_observed_datasets(results)
    else:
        recompute_results = True

    if recompute_results:
        results = compute_all_observed_results(approximators, num_samples=num_samples, batch_size=batch_size)
        save_results(results, results_path)

    if diagnostic_path.exists() and not recompute and not recompute_references:
        diagnostic = pd.read_csv(diagnostic_path, keep_default_na=False)
        recompute_diagnostic = not covers_observed_datasets(diagnostic)
    else:
        recompute_diagnostic = True

    if recompute_diagnostic:
        diagnostic = compute_all_observed_summary_diagnostics(results, approximators, references)
        save_diagnostic(diagnostic, diagnostic_path)

    if posterior_path.exists() and not recompute:
        posterior = load_posterior_diagnostic(posterior_path)
        recompute_posterior = not covers_observed_datasets(posterior)
    else:
        recompute_posterior = True

    if recompute_posterior:
        posterior = compute_all_observed_posterior_diagnostics(
            approximators,
            num_samples=num_samples,
            mmd_samples=mmd_samples,
            batch_size=batch_size,
        )
        save_posterior_diagnostic(posterior, posterior_path)

    if posterior_plot_path.exists() and not recompute and not recompute_references:
        posterior_plot = load_posterior_diagnostic(posterior_plot_path)
        recompute_posterior_plot = not covers_observed_datasets(posterior_plot)
    else:
        recompute_posterior_plot = True

    if recompute_posterior_plot:
        posterior_plot = posterior_plot_frame(diagnostic, posterior)
        save_posterior_diagnostic(posterior_plot, posterior_plot_path)

    return {
        "results": results,
        "diagnostic": diagnostic,
        "posterior": posterior,
        "posterior_plot": posterior_plot,
    }
