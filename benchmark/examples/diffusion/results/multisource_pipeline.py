"""Compute and cache diffusion diagnostics across datasets and summary sizes."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..approximators.indirect import load_approximator
from ..config import MODELS, RESULT_DIR, TrainingConfig
from ..simulators import SIMULATORS
from .observed_datasets import OBSERVED_DATASETS, load_observed_dataset
from .posterior_diagnostic import (
    DEFAULT_MMD_DRAWS,
    load_posterior_diagnostic,
    posterior_diagnostic_frame,
    save_posterior_diagnostic,
)
from .results import (
    attach_gold_standard,
    estimate_model_comparison,
    load_gold_standard,
    save_results,
)
from .summary_diagnostic import (
    REFERENCE_METRICS,
    add_summary_diagnostic_suite,
    fit_reference_suites,
    load_references,
    pmp_diagnostic_frame,
    save_diagnostic,
    save_references,
)


SUMMARY_MULTIPLIERS = (1, 2, 4, 6)
NPE_RESULT_DIR = RESULT_DIR / "NPE_results"


def load_approximators(config: TrainingConfig) -> dict[str, object]:
    return {model: load_approximator(model, config=config) for model in MODELS}


def reference_suite_path(tag: str) -> Path:
    return NPE_RESULT_DIR / f"npe_{tag}_reference_suite.pkl"


def results_path(tag: str) -> Path:
    return NPE_RESULT_DIR / f"npe_{tag}_all_observed_results_gold.csv"


def diagnostic_path(tag: str, metric: str) -> Path:
    return NPE_RESULT_DIR / f"npe_{tag}_all_observed_diagnostic_{metric}.csv"


def posterior_path(tag: str) -> Path:
    return (
        RESULT_DIR / "posterior_diagnostics" / f"npe_{tag}_all_observed_posterior.csv"
    )


def all_observed_paths(tag: str, metric: str) -> dict[str, Path]:
    return {
        "results": results_path(tag),
        "diagnostic": diagnostic_path(tag, metric),
        "posterior": posterior_path(tag),
    }


def covers_observed_datasets(frame: pd.DataFrame) -> bool:
    return "dataset" in frame and set(OBSERVED_DATASETS).issubset(
        frame["dataset"].unique()
    )


def load_or_fit_reference_suite(
    approximators: dict[str, object],
    config: TrainingConfig,
    metrics: tuple[str, ...] = REFERENCE_METRICS,
    overwrite: bool = False,
    reference_kwargs: dict | None = None,
) -> dict[str, dict[str, dict]]:
    path = reference_suite_path(config.summary_label)
    if path.exists() and not overwrite:
        references = load_references(path)
        complete = all(
            metric in references.get(model, {})
            for model in MODELS
            for metric in metrics
        )
        if complete:
            return references

    references = fit_reference_suites(
        approximators,
        SIMULATORS,
        metrics=metrics,
        **(reference_kwargs or {}),
    )
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
        frames.append(
            attach_gold_standard(estimates, load_gold_standard(dataset=dataset))
        )
    return pd.concat(frames, ignore_index=True)


def compute_all_observed_summary_diagnostics(
    results: pd.DataFrame,
    approximators: dict[str, object],
    references: dict[str, dict[str, dict]],
    metrics: tuple[str, ...] = REFERENCE_METRICS,
) -> dict[str, pd.DataFrame]:
    frames = {metric: [] for metric in metrics}
    for dataset in OBSERVED_DATASETS:
        y, _ = load_observed_dataset(dataset)
        dataset_results = results.loc[results["dataset"].eq(dataset)].reset_index(
            drop=True
        )
        suite = add_summary_diagnostic_suite(
            dataset_results,
            y,
            approximators,
            references,
            metrics=metrics,
        )
        for metric in metrics:
            frames[metric].append(suite[metric])
    return {
        metric: pd.concat(parts, ignore_index=True) for metric, parts in frames.items()
    }


def compute_all_observed_posterior_diagnostics(
    approximators: dict[str, object],
    num_samples: int = DEFAULT_MMD_DRAWS,
    mmd_samples: int = DEFAULT_MMD_DRAWS,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compute raw NPE-versus-MCMC posterior metrics for all observed data."""
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


def compute_or_load_posterior_diagnostic(
    config: TrainingConfig,
    *,
    approximators: dict[str, object] | None = None,
    num_samples: int = DEFAULT_MMD_DRAWS,
    mmd_samples: int = DEFAULT_MMD_DRAWS,
    batch_size: int | None = 8,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    """Compute or load raw NPE-versus-MCMC posterior metrics."""
    path = posterior_path(config.summary_label)
    posterior = (
        load_posterior_diagnostic(path)
        if path.exists() and not recompute
        else pd.DataFrame()
    )
    required = {
        "dataset",
        "id",
        "model",
        "observed_mmd2",
        "observed_mmd",
        "rbf_bandwidth2",
        "posterior_mean_rmse",
        "num_npe_draws",
        "num_stan_draws",
        "num_mmd_draws",
        "observed_seed",
    }
    complete = covers_observed_datasets(posterior) and required.issubset(
        posterior.columns
    )
    if complete:
        complete = (
            posterior["num_npe_draws"].eq(num_samples).all()
            and posterior["num_mmd_draws"].eq(mmd_samples).all()
            and posterior["observed_seed"].eq(seed).all()
        )
    if not complete:
        approximators = approximators or load_approximators(config)
        posterior = compute_all_observed_posterior_diagnostics(
            approximators,
            num_samples=num_samples,
            mmd_samples=mmd_samples,
            batch_size=batch_size,
            seed=seed,
        )
        save_posterior_diagnostic(posterior, path)
    return posterior


def posterior_plot_frame(
    diagnostic: pd.DataFrame, posterior: pd.DataFrame
) -> pd.DataFrame:
    metadata = pmp_diagnostic_frame(diagnostic)[
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
    return posterior.merge(
        metadata, on=["dataset", "id", "model"], how="left", validate="one_to_one"
    )


def compute_or_load_all_observed_suite(
    config: TrainingConfig,
    metrics: tuple[str, ...] = REFERENCE_METRICS,
    num_samples: int = 2048,
    posterior_num_samples: int = DEFAULT_MMD_DRAWS,
    mmd_samples: int = DEFAULT_MMD_DRAWS,
    batch_size: int | None = 8,
    recompute: bool = False,
    recompute_references: bool = False,
    reference_kwargs: dict | None = None,
) -> dict[str, object]:
    unknown = set(metrics) - set(REFERENCE_METRICS)
    if unknown:
        raise ValueError(f"Unknown reference metrics: {sorted(unknown)}")

    tag = config.summary_label
    approximators = load_approximators(config)
    references = load_or_fit_reference_suite(
        approximators,
        config,
        metrics=metrics,
        overwrite=recompute_references,
        reference_kwargs=reference_kwargs,
    )

    result_file = results_path(tag)
    if result_file.exists() and not recompute:
        results = pd.read_csv(result_file, keep_default_na=False)
    else:
        results = pd.DataFrame()
    if not covers_observed_datasets(results):
        results = compute_all_observed_results(
            approximators,
            num_samples=num_samples,
            batch_size=batch_size,
        )
        save_results(results, result_file)

    posterior = compute_or_load_posterior_diagnostic(
        config,
        approximators=approximators,
        num_samples=posterior_num_samples,
        mmd_samples=mmd_samples,
        batch_size=batch_size,
        recompute=recompute,
    )

    paths = {metric: diagnostic_path(tag, metric) for metric in metrics}
    use_cache = (
        not recompute
        and not recompute_references
        and all(path.exists() for path in paths.values())
    )
    if use_cache:
        diagnostics = {
            metric: pd.read_csv(path, keep_default_na=False)
            for metric, path in paths.items()
        }
        use_cache = all(
            covers_observed_datasets(frame) for frame in diagnostics.values()
        )
    if not use_cache:
        diagnostics = compute_all_observed_summary_diagnostics(
            results,
            approximators,
            references,
            metrics=metrics,
        )
        for metric, frame in diagnostics.items():
            save_diagnostic(frame, paths[metric])

    return {
        "results": results,
        "posterior": posterior,
        "diagnostics": diagnostics,
        "references": references,
    }


def compute_diagnostic_pipeline(
    summary_multipliers: tuple[int, ...] = SUMMARY_MULTIPLIERS,
    metrics: tuple[str, ...] = REFERENCE_METRICS,
    *,
    summary_base_distribution: str | None = "normal",
    run_suffix: str | None = None,
    embed_dim: int = 64,
    **kwargs,
) -> dict[str, dict[str, object]]:
    """Compute or load one diagnostic suite for each requested summary size."""
    return {
        config.summary_label: compute_or_load_all_observed_suite(
            config, metrics=metrics, **kwargs
        )
        for config in (
            TrainingConfig(
                summary_multiplier=value,
                embed_dim=embed_dim,
                summary_base_distribution=summary_base_distribution,
                run_suffix=run_suffix,
            )
            for value in summary_multipliers
        )
    }


def load_cached_all_observed(
    config: TrainingConfig,
    metric: str = "l2",
) -> dict[str, pd.DataFrame]:
    paths = all_observed_paths(config.summary_label, metric)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing diagnostic cache:\n" + "\n".join(missing))

    results = pd.read_csv(paths["results"], keep_default_na=False)
    diagnostic = pd.read_csv(paths["diagnostic"], keep_default_na=False)
    posterior = load_posterior_diagnostic(paths["posterior"])
    return {
        "results": results,
        "diagnostic": diagnostic,
        "posterior": posterior,
        "posterior_plot": posterior_plot_frame(diagnostic, posterior),
    }


def compute_or_load_all_observed(
    config: TrainingConfig,
    metric: str = "l2",
    **kwargs,
) -> dict[str, pd.DataFrame]:
    suite = compute_or_load_all_observed_suite(config, metrics=(metric,), **kwargs)
    diagnostic = suite["diagnostics"][metric]
    posterior = suite["posterior"]
    return {
        "results": suite["results"],
        "diagnostic": diagnostic,
        "posterior": posterior,
        "posterior_plot": posterior_plot_frame(diagnostic, posterior),
    }
