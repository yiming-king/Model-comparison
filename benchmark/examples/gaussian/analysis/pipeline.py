"""Compute Gaussian OOD inference caches and summary-space diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import bayesflow as bf
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "benchmark.examples.gaussian.analysis"

from ..config import (
    ASSUMED_MODELS,
    CALIBRATION_ROOT,
    MODEL_SPECS,
    NETWORK_DIR,
    NetworkSet,
    RESULT_DIR,
    SOURCE_MODELS,
    configuration_tag,
    discover_network_sets,
)
from ..datasets.calculation import Calculation
from ..datasets.datasets import GetDatasets
from ..direct.calculator import direct_get_probs, indirect_get_probs, softmax_stable
from . import summry_diagnostic as diagnostic


NONNEGATIVE_Y_MARGIN = 0.1
THRESHOLD_LINEWIDTH = 1.8
AXIS_SCALES = ("linear", "symlog")
LOG10_LOGML_ERROR_LABEL = (
    r"$\log_{10}\widehat{p}(y\mid M_j)-\log_{10}p(y\mid M_j)$"
)


@dataclass(frozen=True)
class OODPaths:
    """Cached computation and diagnostic paths for one summary network."""

    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    def raw_dataset(self, source: str) -> Path:
        return self.datasets / f"{source}_raw.pkl"

    def inference_dataset(self, source: str) -> Path:
        return self.datasets / f"{source}_logml_pmp.pkl"

    @property
    def inference(self) -> Path:
        return self.root / "inference"

    @property
    def posterior(self) -> Path:
        return self.inference / "posterior.csv"

    @property
    def logml(self) -> Path:
        return self.inference / "logml.csv"

    @property
    def pmp(self) -> Path:
        return self.inference / "pmp.csv"

    @property
    def inference_metadata(self) -> Path:
        return self.inference / "metadata.json"

    def diagnostic(self, metric: str) -> Path:
        return self.root / "diagnostics" / metric


def _save_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(value, file)


def _load_pickle(path: Path):
    with path.open("rb") as file:
        return pickle.load(file)


def load_approximators(
    network_set: NetworkSet,
) -> dict[str, object]:
    return {
        model: keras.saving.load_model(network_set.paths[model])
        for model in ASSUMED_MODELS
    }


def generate_ood_datasets(
    paths: OODPaths,
    network_set: NetworkSet,
    *,
    num_datasets: int = 50,
    seed: int = 2025,
    overwrite: bool = False,
) -> dict[str, list[dict]]:
    """Generate or reuse the shared m1--m12 Gaussian OOD dataset bank."""
    if num_datasets <= 1:
        raise ValueError("num_datasets must exceed one")
    output = {}
    for source_index, source in enumerate(SOURCE_MODELS):
        path = paths.raw_dataset(source)
        if path.exists() and not overwrite:
            datasets = _load_pickle(path)
            expected_shape = (network_set.num_obs, network_set.data_dim)
            if len(datasets) != num_datasets or any(
                np.asarray(item["x"]).shape != expected_shape for item in datasets
            ):
                raise ValueError(
                    f"Existing OOD data at {path} do not match "
                    f"{num_datasets} datasets with shape {expected_shape}. "
                    "Use --overwrite-datasets to replace them."
                )
            print(f"Reusing raw OOD datasets: {path}")
        else:
            spec = MODEL_SPECS[source]
            datasets = GetDatasets(
                obs_mu_prior_mean=spec["mu_prior_mean"],
                obs_mu_prior_std=spec["mu_prior_std"],
                num_dims=network_set.data_dim,
                num_obs=network_set.num_obs,
                obs_likelihood_std=spec["likelihood_std"],
                num_datasets=num_datasets,
                rng=np.random.default_rng(seed + 10_000 * source_index),
            ).get_datasets_normal()
            for item in datasets:
                item["source_model"] = source
            _save_pickle(datasets, path)
            print(f"Generated raw OOD datasets: {path}")
        output[source] = datasets
    return output


def _make_calculations(
    approximators: dict[str, object],
    network_set: NetworkSet,
    *,
    num_samples: int,
    logml_method: str,
    seed: int,
) -> dict[str, Calculation]:
    calculations = {}
    for model_index, model in enumerate(ASSUMED_MODELS):
        spec = MODEL_SPECS[model]
        calculations[model] = Calculation(
            approximator=approximators[model],
            mu_prior_mean=spec["mu_prior_mean"],
            mu_prior_std=spec["mu_prior_std"],
            num_dims=network_set.data_dim,
            num_obs=network_set.num_obs,
            likelihood_std=spec["likelihood_std"],
            num_samples=num_samples,
            assumed_model=model,
            rng=np.random.default_rng(seed + model_index),
            logml_method=logml_method,
        )
    return calculations


def _collect_inference_frames(
    datasets: dict[str, list[dict]],
    *,
    n_posterior_samples: int,
) -> dict[str, pd.DataFrame]:
    """Build plot-ready posterior, logML, and PMP tables before diagnostics."""
    posterior_rows = []
    logml_rows = []
    pmp_rows = []
    mmd = MaximumMeanDiscrepancy(kernel="gaussian")
    for source in SOURCE_MODELS:
        for item in datasets[source]:
            gold_logml = np.asarray(
                [item[f"gold_log_marginal_{model}"] for model in ASSUMED_MODELS],
                dtype=float,
            )
            npe_logml = np.asarray(
                [item[f"npe_log_marginal_{model}"] for model in ASSUMED_MODELS],
                dtype=float,
            )
            gold_pmp = softmax_stable(gold_logml)
            npe_pmp = softmax_stable(npe_logml)
            direct_pmp = np.asarray(item["p_direct"], dtype=float)
            if direct_pmp.shape != (len(ASSUMED_MODELS),):
                raise ValueError(
                    f"Direct PMP for {source}/{item['id']} has shape "
                    f"{direct_pmp.shape}; expected {(len(ASSUMED_MODELS),)}"
                )
            pmp_row = {"source_model": source, "id": int(item["id"])}
            for model_index, model in enumerate(ASSUMED_MODELS):
                gold = np.asarray(item[f"gold_post_samples_{model}"], dtype=np.float32)
                npe = np.asarray(item[f"npe_post_samples_{model}"], dtype=np.float32)
                sample_count = min(n_posterior_samples, len(gold), len(npe))
                if sample_count <= 1:
                    raise ValueError("Posterior MMD requires at least two draws")
                gold = gold[:sample_count]
                npe = npe[:sample_count]
                posterior_rows.append(
                    {
                        "source_model": source,
                        "id": int(item["id"]),
                        "assumed_model": model,
                        "posterior_mmd": float(
                            mmd(tf.convert_to_tensor(npe), tf.convert_to_tensor(gold))
                        ),
                        "posterior_mean_rmse": float(
                            np.sqrt(np.mean((npe.mean(0) - gold.mean(0)) ** 2))
                        ),
                        "n_posterior_samples": sample_count,
                    }
                )
                logml_rows.append(
                    {
                        "source_model": source,
                        "id": int(item["id"]),
                        "assumed_model": model,
                        "gold_logml": float(gold_logml[model_index]),
                        "npe_logml": float(npe_logml[model_index]),
                        "signed_logml_error": float(
                            npe_logml[model_index] - gold_logml[model_index]
                        ),
                    }
                )
                pmp_row[f"p_gold_{model}"] = float(gold_pmp[model_index])
                pmp_row[f"p_npe_{model}"] = float(npe_pmp[model_index])
                pmp_row[f"p_direct_{model}"] = float(direct_pmp[model_index])
                pmp_row[f"signed_pmp_error_npe_{model}"] = float(
                    npe_pmp[model_index] - gold_pmp[model_index]
                )
                pmp_row[f"signed_pmp_error_direct_{model}"] = float(
                    direct_pmp[model_index] - gold_pmp[model_index]
                )
            pmp_row["pmp_l1_error_npe"] = float(np.abs(npe_pmp - gold_pmp).sum())
            pmp_row["pmp_l1_error_direct"] = float(np.abs(direct_pmp - gold_pmp).sum())
            pmp_rows.append(pmp_row)
    return {
        "posterior": pd.DataFrame(posterior_rows),
        "logml": pd.DataFrame(logml_rows),
        "pmp": pd.DataFrame(pmp_rows),
    }


def save_inference_results(
    frames: dict[str, pd.DataFrame], paths: OODPaths, metadata: dict
) -> None:
    paths.inference.mkdir(parents=True, exist_ok=True)
    frames["posterior"].to_csv(paths.posterior, index=False)
    frames["logml"].to_csv(paths.logml, index=False)
    frames["pmp"].to_csv(paths.pmp, index=False)
    with paths.inference_metadata.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(f"Saved posterior/logML/PMP tables: {paths.inference}")


def load_cached_inference_results(
    result_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load computation-only posterior, logML, and PMP tables."""
    paths = OODPaths(Path(result_dir))
    files = {
        "posterior": paths.posterior,
        "logml": paths.logml,
        "pmp": paths.pmp,
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Gaussian inference cache:\n" + "\n".join(missing)
        )
    return {
        name: pd.read_csv(path, keep_default_na=False) for name, path in files.items()
    }


def compute_ood_inference(
    paths: OODPaths,
    network_set: NetworkSet,
    *,
    network_dir: str | Path = NETWORK_DIR,
    num_samples: int = 1000,
    logml_method: str = "log_mean_exp",
    seed: int = 2025,
    overwrite: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute NPE/analytical posterior, logML, indirect PMP, and direct PMP."""
    if num_samples <= 1:
        raise ValueError("num_samples must exceed one")
    diagnostic.quiet_bayesflow_progress()
    raw_signatures = {}
    for source in SOURCE_MODELS:
        path = paths.raw_dataset(source)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing raw OOD data: {path}. Run the generate stage first."
            )
        stat = path.stat()
        raw_signatures[source] = {
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    metadata = {
        "network_tag": network_set.network_tag,
        "summary_label": network_set.summary_label,
        "summary_dim": network_set.summary_dim,
        "data_dim": network_set.data_dim,
        "num_obs": network_set.num_obs,
        "num_posterior_samples": int(num_samples),
        "logml_method": logml_method,
        "seed": int(seed),
        "raw_datasets": raw_signatures,
    }
    paths_to_check = (paths.posterior, paths.logml, paths.pmp)
    inference_pickles = tuple(
        paths.inference_dataset(source) for source in SOURCE_MODELS
    )
    complete = all(path.exists() for path in (*paths_to_check, *inference_pickles))
    if not overwrite and complete and paths.inference_metadata.exists():
        with paths.inference_metadata.open(encoding="utf-8") as file:
            cached_metadata = json.load(file)
        if cached_metadata == metadata:
            print(f"Reusing posterior/logML/PMP tables: {paths.inference}")
            return load_cached_inference_results(paths.root)
        print(
            f"Inference settings or raw datasets changed; rebuilding {paths.inference}"
        )

    datasets = {
        source: _load_pickle(paths.raw_dataset(source)) for source in SOURCE_MODELS
    }
    approximators = load_approximators(network_set)
    calculations = _make_calculations(
        approximators,
        network_set,
        num_samples=num_samples,
        logml_method=logml_method,
        seed=seed + 1_000_000,
    )
    direct_path = Path(network_dir) / (
        f"direct_s_{configuration_tag(network_set.data_dim, network_set.num_obs)}.keras"
    )
    if not direct_path.exists():
        raise FileNotFoundError(
            f"Missing direct model-comparison network: {direct_path}"
        )
    direct_approximator = keras.saving.load_model(direct_path)

    for source_index, source in enumerate(SOURCE_MODELS):
        items = datasets[source]
        for model_index, model in enumerate(ASSUMED_MODELS):
            calculation = calculations[model]
            calculation.normal_analytical(items)
            calculation.npe_estimation(
                items,
                seed=seed + 2_000_000 + 10_000 * source_index + model_index,
            )
        direct_get_probs(items, direct_approximator)
        indirect_get_probs(items, ASSUMED_MODELS)
        _save_pickle(items, paths.inference_dataset(source))
        print(f"Computed posterior/logML/PMP: {network_set.summary_label} / {source}")

    frames = _collect_inference_frames(datasets, n_posterior_samples=num_samples)
    save_inference_results(frames, paths, metadata)
    return frames


def make_simulator(spec: dict[str, float], num_dims: int, num_obs: int, seed: int):
    rng = np.random.default_rng(seed)

    def prior():
        return {
            "mu": rng.normal(
                loc=spec["mu_prior_mean"],
                scale=spec["mu_prior_std"],
                size=num_dims,
            )
        }

    def likelihood(mu):
        return {
            "x": rng.normal(
                loc=mu,
                scale=spec["likelihood_std"],
                size=(num_obs, num_dims),
            )
        }

    return bf.make_simulator([prior, likelihood])


def load_cached_ood_datasets(
    result_dir: Path,
    sources: tuple[str, ...] = SOURCE_MODELS,
) -> dict[str, list[dict]]:
    datasets = {}
    for source in sources:
        path = result_dir / "datasets" / f"{source}_logml_pmp.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached Gaussian OOD results: {path}. Run the Python "
                "inference stage first."
            )
        with path.open("rb") as file:
            datasets[source] = pickle.load(file)
    return datasets


def load_or_fit_references(
    path: Path,
    approximators: dict[str, object],
    simulators: dict[str, object],
    metrics: tuple[str, ...],
    *,
    overwrite: bool,
    reference_kwargs: dict,
) -> dict[str, dict[str, dict]]:
    if path.exists() and not overwrite:
        references = diagnostic.load_reference_suites(path)
        if all(
            metric in references.get(model, {})
            for model in ASSUMED_MODELS
            for metric in metrics
        ):
            print(f"Reusing reference suite: {path}")
            return references
    references = diagnostic.fit_summary_reference_suites(
        approximators,
        simulators,
        assumed_models=ASSUMED_MODELS,
        metrics=metrics,
        **reference_kwargs,
    )
    diagnostic.save_reference_suites(references, path)
    print(f"Saved reference suite: {path}")
    return references


def _threshold_lookup(
    path: Path,
    configuration: str,
    network_set: NetworkSet,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration thresholds: {path}. Run Gaussian calibration first."
        )
    thresholds = pd.read_csv(path, keep_default_na=False)
    required_metadata = {
        "network_tag",
        "summary_label",
        "summary_dim",
        "lower_threshold",
        "threshold",
        "median",
    }
    missing_metadata = sorted(required_metadata.difference(thresholds.columns))
    if missing_metadata:
        raise ValueError(
            "Threshold table predates multi-summary calibration and is missing "
            f"{missing_metadata}. Re-run Gaussian calibration."
        )
    thresholds = thresholds.loc[
        thresholds["configuration"].eq(configuration)
        & thresholds["network_tag"].eq(network_set.network_tag)
        & thresholds["summary_dim"].eq(network_set.summary_dim)
    ]
    wide = thresholds.pivot(
        index="generating_model",
        columns="metric",
        values=["lower_threshold", "threshold", "median"],
    )
    wide.columns = [f"{metric}_{statistic}" for statistic, metric in wide.columns]
    required = {
        "posterior_mmd_threshold",
        "posterior_mmd_lower_threshold",
        "posterior_mmd_median",
        "signed_logml_error_lower_threshold",
        "signed_logml_error_threshold",
        "signed_logml_error_median",
        "signed_pmp_error_lower_threshold",
        "signed_pmp_error_threshold",
    }
    missing = sorted(required.difference(wide.columns))
    if missing:
        raise ValueError(f"Calibration thresholds are missing columns: {missing}")
    if wide["posterior_mmd_lower_threshold"].ne(0.0).any():
        raise ValueError("Posterior MMD lower thresholds must be zero")
    for low, high in (
        ("posterior_mmd_lower_threshold", "posterior_mmd_threshold"),
        ("signed_logml_error_lower_threshold", "signed_logml_error_threshold"),
    ):
        if wide[low].ge(wide[high]).any():
            raise ValueError(f"Calibration lower threshold must be below {high}")
    pmp_low = "signed_pmp_error_lower_threshold"
    pmp_high = "signed_pmp_error_threshold"
    if wide[pmp_low].gt(wide[pmp_high]).any():
        raise ValueError(f"Calibration lower threshold must not exceed {pmp_high}")
    wide["signed_pmp_error_degenerate_interval"] = wide[pmp_low].eq(
        wide[pmp_high]
    )
    return wide.reset_index().rename(columns={"generating_model": "assumed_model"})


def _scale_nonnegative_metric(values, high):
    """Scale a non-negative metric so its raw upper threshold maps to one."""
    values = np.asarray(values, dtype=float)
    upper = np.asarray(high, dtype=float)
    if np.any(upper <= 0.0):
        raise ValueError("Non-negative metric upper threshold must be positive")
    return values / upper


def _nonnegative_plot_limits(upper: float) -> tuple[float, float]:
    """Add visual space below zero without changing non-negative data values."""
    upper = float(upper)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("Non-negative plot upper limit must be positive and finite")
    return -NONNEGATIVE_Y_MARGIN * upper, upper


def _set_axis_scale(
    ax: plt.Axes,
    axis: str,
    scale: str,
    *,
    linthresh: float,
) -> None:
    """Apply a notebook-selectable linear or symmetrical-log axis scale."""
    if scale not in AXIS_SCALES:
        raise ValueError(f"{axis}scale must be one of {AXIS_SCALES}; got {scale!r}")
    if scale == "symlog" and linthresh <= 0.0:
        raise ValueError(f"{axis}_symlog_linthresh must be positive")
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if scale == "symlog":
        setter("symlog", base=10.0, linthresh=linthresh, linscale=1.0)
    else:
        setter("linear")


def _resolve_metric_scale(scale: str | dict[str, str], metric: str) -> str:
    """Resolve a shared y scale or a per-metric notebook mapping."""
    if isinstance(scale, str):
        return scale
    if metric not in scale:
        raise ValueError(f"yscale mapping has no entry for {metric!r}")
    return scale[metric]


def _scale_signed_by_lower_bound(values, low):
    """Scale a signed metric by the magnitude of its raw lower bound."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray(low, dtype=float)
    scale = np.abs(lower)
    if np.any(scale <= 0.0):
        raise ValueError("Signed-error lower threshold must be non-zero")
    return values / scale


METRIC_NORMALIZERS = {
    "posterior_mmd": _scale_nonnegative_metric,
    "logml": _scale_signed_by_lower_bound,
}


def _normalize_metric(metric: str, values, median, low, high):
    """Apply the independently replaceable display transform for one metric."""
    try:
        normalizer = METRIC_NORMALIZERS[metric]
    except KeyError as error:
        raise ValueError(f"No display normalizer registered for {metric!r}") from error
    if metric == "posterior_mmd":
        return normalizer(values, high)
    return normalizer(values, low)


def _metric_ylabel(base: str, normalized: bool) -> str:
    """Build a display label whose normalization suffix follows the switch."""
    suffix = "(NPE, MCMC; normalized)" if normalized else "(NPE, MCMC)"
    return f"{base}\n{suffix}"


def attach_calibrated_errors(
    posterior: pd.DataFrame,
    logml: pd.DataFrame,
    pmp: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Attach current calibration fields, replacing partial or legacy fields."""
    posterior = posterior.drop(
        columns=[
            column
            for column in posterior
            if column
            in {
                "posterior_mmd_lower_threshold",
                "posterior_mmd_threshold",
                "posterior_mmd_median",
            }
            or column.startswith("normalized_posterior_mmd")
        ],
        errors="ignore",
    ).copy()
    logml = logml.drop(
        columns=[
            column
            for column in logml
            if column
            in {
                "signed_logml_error_lower_threshold",
                "signed_logml_error_threshold",
                "signed_logml_error_median",
                "absolute_logml_error_threshold",
                "absolute_logml_error_median",
            }
            or column.startswith("normalized_logml_error")
        ],
        errors="ignore",
    ).copy()
    posterior = posterior.merge(
        thresholds[
            [
                "assumed_model",
                "posterior_mmd_lower_threshold",
                "posterior_mmd_threshold",
                "posterior_mmd_median",
            ]
        ],
        on="assumed_model",
        how="left",
        validate="many_to_one",
    )
    posterior["normalized_posterior_mmd"] = _normalize_metric(
        "posterior_mmd",
        posterior["posterior_mmd"],
        posterior["posterior_mmd_median"],
        posterior["posterior_mmd_lower_threshold"],
        posterior["posterior_mmd_threshold"],
    )
    posterior["normalized_posterior_mmd_low"] = _normalize_metric(
        "posterior_mmd",
        posterior["posterior_mmd_lower_threshold"],
        posterior["posterior_mmd_median"],
        posterior["posterior_mmd_lower_threshold"],
        posterior["posterior_mmd_threshold"],
    )
    posterior["normalized_posterior_mmd_high"] = _normalize_metric(
        "posterior_mmd",
        posterior["posterior_mmd_threshold"],
        posterior["posterior_mmd_median"],
        posterior["posterior_mmd_lower_threshold"],
        posterior["posterior_mmd_threshold"],
    )

    logml = logml.merge(
        thresholds[
            [
                "assumed_model",
                "signed_logml_error_lower_threshold",
                "signed_logml_error_threshold",
                "signed_logml_error_median",
            ]
        ],
        on="assumed_model",
        how="left",
        validate="many_to_one",
    )
    logml["log10_logml_error"] = logml["signed_logml_error"] / np.log(10.0)
    logml["log10_logml_error_lower_threshold"] = (
        logml["signed_logml_error_lower_threshold"] / np.log(10.0)
    )
    logml["log10_logml_error_upper_threshold"] = (
        logml["signed_logml_error_threshold"] / np.log(10.0)
    )
    logml["log10_logml_error_median"] = (
        logml["signed_logml_error_median"] / np.log(10.0)
    )
    logml["normalized_logml_error"] = _normalize_metric(
        "logml",
        logml["log10_logml_error"],
        logml["log10_logml_error_median"],
        logml["log10_logml_error_lower_threshold"],
        logml["log10_logml_error_upper_threshold"],
    )
    logml["normalized_logml_error_low"] = _normalize_metric(
        "logml",
        logml["log10_logml_error_lower_threshold"],
        logml["log10_logml_error_median"],
        logml["log10_logml_error_lower_threshold"],
        logml["log10_logml_error_upper_threshold"],
    )
    logml["normalized_logml_error_high"] = _normalize_metric(
        "logml",
        logml["log10_logml_error_upper_threshold"],
        logml["log10_logml_error_median"],
        logml["log10_logml_error_lower_threshold"],
        logml["log10_logml_error_upper_threshold"],
    )

    pmp = pmp.drop(
        columns=[
            column
            for column in pmp
            if column.startswith("normalized_pmp")
            or column.startswith("pmp_error_lower_threshold_")
            or column.startswith("pmp_error_upper_threshold_")
            or column.startswith("pmp_error_threshold_")
            or column.startswith("pmp_error_degenerate_interval_")
            or column.startswith("signed_pmp_error_median_")
        ],
        errors="ignore",
    ).copy()
    indexed = thresholds.set_index("assumed_model")
    for model in ASSUMED_MODELS:
        lower_threshold = float(
            indexed.loc[model, "signed_pmp_error_lower_threshold"]
        )
        upper_threshold = float(indexed.loc[model, "signed_pmp_error_threshold"])
        pmp[f"pmp_error_lower_threshold_{model}"] = lower_threshold
        pmp[f"pmp_error_upper_threshold_{model}"] = upper_threshold
        pmp[f"pmp_error_degenerate_interval_{model}"] = bool(
            indexed.loc[model, "signed_pmp_error_degenerate_interval"]
            if "signed_pmp_error_degenerate_interval" in indexed
            else lower_threshold == upper_threshold
        )
    return posterior, logml, pmp


def save_metric_frames(
    output_dir: Path,
    metric: str,
    configuration: str,
    network_set: NetworkSet,
    datasets: dict[str, list[dict]],
    posterior_error_frame: pd.DataFrame | None,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    logml = diagnostic.collect_logml_distance_frame(datasets)
    pmp = diagnostic.collect_pmp_ambiguity_frame(datasets)
    if posterior_error_frame is None:
        posterior = diagnostic.collect_posterior_distance_frame(datasets)
        posterior_error_frame = posterior[
            [
                "source_model",
                "id",
                "assumed_model",
                "posterior_mmd",
                "posterior_mean_rmse",
                "n_posterior_samples",
            ]
        ].copy()
    posterior = logml[
        [
            "source_model",
            "id",
            "assumed_model",
            "d_M",
            "dm_median",
            "dm_low",
            "dm_high",
            "distance_regime",
        ]
    ].merge(
        posterior_error_frame,
        on=["source_model", "id", "assumed_model"],
        how="left",
        validate="one_to_one",
    )
    posterior, logml, pmp = attach_calibrated_errors(posterior, logml, pmp, thresholds)
    for frame in (posterior, logml):
        frame["rho"] = (frame["d_M"] - frame["dm_median"]) / (
            frame["dm_high"] - frame["dm_median"]
        )
        frame["diagnostic"] = metric
        frame["configuration"] = configuration
        frame["network_tag"] = network_set.network_tag
        frame["summary_label"] = network_set.summary_label
        frame["summary_dim"] = network_set.summary_dim
    pmp["diagnostic"] = metric
    pmp["configuration"] = configuration
    pmp["network_tag"] = network_set.network_tag
    pmp["summary_label"] = network_set.summary_label
    pmp["summary_dim"] = network_set.summary_dim

    posterior.to_csv(output_dir / "posterior_distance_frame.csv", index=False)
    logml.to_csv(output_dir / "logml_distance_frame.csv", index=False)
    pmp.to_csv(output_dir / "pmp_ambiguity_frame.csv", index=False)
    logml_summary, pmp_summary = diagnostic.summarize_frames(logml, pmp)
    logml_summary.to_csv(output_dir / "logml_summary.csv", index=False)
    pmp_summary.to_csv(output_dir / "pmp_summary.csv", index=False)
    print(f"Saved {metric} diagnostic frames: {output_dir}")
    return posterior_error_frame


def _pmp_plot_long(pmp: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for model in ASSUMED_MODELS:
        frame = pd.DataFrame(
            {
                "source_model": pmp["source_model"],
                "id": pmp["id"],
                "assumed_model": model,
                "d_M": pmp[f"d_{model}"],
                "dm_median": pmp[f"dm_median_{model}"],
                "dm_low": pmp[f"dm_low_{model}"],
                "dm_high": pmp[f"dm_high_{model}"],
                "plot_error": pmp[f"signed_pmp_error_npe_{model}"],
                "plot_error_low": pmp[f"pmp_error_lower_threshold_{model}"],
                "plot_error_high": pmp[f"pmp_error_upper_threshold_{model}"],
            }
        )
        frame["rho"] = (frame["d_M"] - frame["dm_median"]) / (
            frame["dm_high"] - frame["dm_median"]
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _plot_normalized_error_grid(
    frame: pd.DataFrame,
    *,
    value_column: str,
    low_column: str | None,
    high_column: str | None,
    ylabel: str,
    output_path: Path,
    nonnegative: bool = False,
    xscale: str = "symlog",
    yscale: str = "linear",
    x_symlog_linthresh: float = 1.0,
    y_symlog_linthresh: float = 1.0,
) -> None:
    colors = dict(zip(SOURCE_MODELS, diagnostic.SOURCE_COLORS, strict=True))
    fig, axes = plt.subplots(1, len(ASSUMED_MODELS), figsize=(20, 5), sharey=True)
    for ax, model in zip(np.atleast_1d(axes), ASSUMED_MODELS, strict=True):
        panel = frame.loc[frame["assumed_model"].eq(model)]
        rho_low = float(
            (
                (panel["dm_low"] - panel["dm_median"])
                / (panel["dm_high"] - panel["dm_median"])
            ).median()
        )
        error_low = float(panel[low_column].median()) if low_column is not None else 0.0
        error_high = (
            float(panel[high_column].median()) if high_column is not None else 1.0
        )
        ax.axvspan(rho_low, 1.0, color=diagnostic.TYPICAL_SET_FILL, alpha=0.65)
        ax.axhspan(error_low, error_high, color=diagnostic.TYPICAL_SET_FILL, alpha=0.35)
        ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        if nonnegative:
            ax.axhline(
                error_high,
                color="0.25",
                linestyle="--",
                linewidth=THRESHOLD_LINEWIDTH,
            )
        else:
            ax.axhline(
                error_low,
                color="0.45",
                linestyle=":",
                linewidth=THRESHOLD_LINEWIDTH,
            )
            ax.axhline(
                error_high,
                color="0.25",
                linestyle="--",
                linewidth=THRESHOLD_LINEWIDTH,
            )
        for source in SOURCE_MODELS:
            source_frame = panel.loc[panel["source_model"].eq(source)]
            ax.scatter(
                source_frame["rho"],
                source_frame[value_column],
                s=20,
                alpha=0.7,
                color=colors[source],
                edgecolor="none",
                label=source.upper(),
            )
        _set_axis_scale(
            ax,
            "x",
            xscale,
            linthresh=x_symlog_linthresh,
        )
        _set_axis_scale(
            ax,
            "y",
            yscale,
            linthresh=y_symlog_linthresh,
        )
        ax.set_title(f"Assumed {model.upper()}")
        ax.set_xlabel(r"Normalized diagnostic $\rho_j(y)$")
        ax.grid(alpha=0.15)
    if nonnegative:
        upper = max(ax.get_ylim()[1] for ax in np.atleast_1d(axes))
        limits = (
            (-NONNEGATIVE_Y_MARGIN * y_symlog_linthresh, upper)
            if yscale == "symlog"
            else _nonnegative_plot_limits(upper)
        )
        for ax in np.atleast_1d(axes):
            ax.set_ylim(limits)
    np.atleast_1d(axes)[0].set_ylabel(ylabel, labelpad=20)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(SOURCE_MODELS),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def generate_calibrated_plots(
    output_dir: Path,
    normalize_metrics: bool = True,
    frames: dict[str, pd.DataFrame] | None = None,
    yscale: str | dict[str, str] = "linear",
    y_symlog_linthresh: float = 1.0,
) -> None:
    """Plot raw or transformed values with a shared or per-metric y scale."""
    if frames is None:
        frames = {
            "posterior": pd.read_csv(output_dir / "posterior_distance_frame.csv"),
            "logml": pd.read_csv(output_dir / "logml_distance_frame.csv"),
            "pmp": pd.read_csv(output_dir / "pmp_ambiguity_frame.csv"),
        }
    posterior = frames["posterior"]
    logml = frames["logml"]
    pmp = _pmp_plot_long(frames["pmp"])
    figure_dir = output_dir / "figures"
    posterior_columns = (
        (
            "normalized_posterior_mmd",
            "normalized_posterior_mmd_low",
            "normalized_posterior_mmd_high",
            _metric_ylabel("Posterior MMD", normalized=True),
        )
        if normalize_metrics
        else (
            "posterior_mmd",
            "posterior_mmd_lower_threshold",
            "posterior_mmd_threshold",
            _metric_ylabel("Posterior MMD", normalized=False),
        )
    )
    logml_columns = (
        (
            "normalized_logml_error",
            "normalized_logml_error_low",
            "normalized_logml_error_high",
            _metric_ylabel(LOG10_LOGML_ERROR_LABEL, normalized=True),
        )
        if normalize_metrics
        else (
            "log10_logml_error",
            "log10_logml_error_lower_threshold",
            "log10_logml_error_upper_threshold",
            _metric_ylabel(LOG10_LOGML_ERROR_LABEL, normalized=False),
        )
    )
    _plot_normalized_error_grid(
        posterior,
        value_column=posterior_columns[0],
        low_column=posterior_columns[1],
        high_column=posterior_columns[2],
        ylabel=posterior_columns[3],
        output_path=figure_dir / "posterior_mmd_vs_rho_calibrated.png",
        nonnegative=True,
        yscale=_resolve_metric_scale(yscale, "posterior_mmd"),
        y_symlog_linthresh=y_symlog_linthresh,
    )
    _plot_normalized_error_grid(
        logml,
        value_column=logml_columns[0],
        low_column=logml_columns[1],
        high_column=logml_columns[2],
        ylabel=logml_columns[3],
        output_path=figure_dir / "logml_error_vs_rho_calibrated.png",
        yscale=_resolve_metric_scale(yscale, "logml"),
        y_symlog_linthresh=y_symlog_linthresh,
    )
    _plot_normalized_error_grid(
        pmp,
        value_column="plot_error",
        low_column="plot_error_low",
        high_column="plot_error_high",
        ylabel=_metric_ylabel("Signed PMP error", normalized=False),
        output_path=figure_dir / "pmp_error_vs_rho_calibrated.png",
        yscale=_resolve_metric_scale(yscale, "pmp"),
        y_symlog_linthresh=y_symlog_linthresh,
    )
    print(f"Saved calibrated figures: {figure_dir}")


def load_cached_metric_frames(
    result_dir: str | Path,
    metric: str = "l2",
) -> dict[str, pd.DataFrame]:
    """Load plot frames from the current layout or the legacy L2/Linf layout."""
    result_dir = Path(result_dir)
    metric = metric.lower()
    if metric not in diagnostic.REFERENCE_METRICS:
        raise ValueError(f"metric must be one of {diagnostic.REFERENCE_METRICS}")

    output_dir = OODPaths(result_dir).diagnostic(metric)
    modern_files = {
        "posterior": output_dir / "posterior_distance_frame.csv",
        "logml": output_dir / "logml_distance_frame.csv",
        "pmp": output_dir / "pmp_ambiguity_frame.csv",
    }
    if all(path.exists() for path in modern_files.values()):
        files = modern_files
    elif metric in {"l2", "linf"}:
        network_tag = result_dir.name.removeprefix("ood_")
        legacy_root = (
            result_dir
            if metric == "l2"
            else result_dir.parent / f"{result_dir.name}_inf"
        )
        suffix = network_tag if metric == "l2" else f"{network_tag}_linf"
        files = {
            "posterior": legacy_root / f"posterior_distance_frame_{suffix}.csv",
            "logml": legacy_root / f"logml_distance_frame_{suffix}.csv",
            "pmp": legacy_root / f"pmp_ambiguity_frame_{suffix}.csv",
        }
    else:
        files = modern_files

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Gaussian diagnostic cache:\n" + "\n".join(missing)
        )
    frames = {
        name: pd.read_csv(path, keep_default_na=False) for name, path in files.items()
    }
    if metric in {"l2", "linf"}:
        # Older L2/Linf notebooks defined rho as d / d_high and did not always
        # persist a reference median. A zero center reproduces that definition:
        # (d - median) / (high - median) = d / high.
        for name in ("posterior", "logml"):
            if "dm_median" not in frames[name]:
                frames[name]["dm_median"] = 0.0
        for model in ASSUMED_MODELS:
            column = f"dm_median_{model}"
            if column not in frames["pmp"]:
                frames["pmp"][column] = 0.0
    for name in ("posterior", "logml"):
        frame = frames[name]
        if "rho" not in frame and {"d_M", "dm_median", "dm_high"}.issubset(
            frame.columns
        ):
            frame["rho"] = (frame["d_M"] - frame["dm_median"]) / (
                frame["dm_high"] - frame["dm_median"]
            )
    can_attach_thresholds = (
        {"assumed_model", "posterior_mmd"}.issubset(frames["posterior"].columns)
        and {"assumed_model", "signed_logml_error"}.issubset(
            frames["logml"].columns
        )
        and all(
            f"signed_pmp_error_npe_{model}" in frames["pmp"]
            for model in ASSUMED_MODELS
        )
    )
    required_calibration_columns = {
        "posterior": {
            "posterior_mmd_lower_threshold",
            "posterior_mmd_threshold",
            "posterior_mmd_median",
            "normalized_posterior_mmd",
            "normalized_posterior_mmd_low",
            "normalized_posterior_mmd_high",
        },
        "logml": {
            "signed_logml_error_lower_threshold",
            "signed_logml_error_threshold",
            "signed_logml_error_median",
            "normalized_logml_error",
            "normalized_logml_error_low",
            "normalized_logml_error_high",
        },
        "pmp": {
            column
            for model in ASSUMED_MODELS
            for column in (
                f"pmp_error_lower_threshold_{model}",
                f"pmp_error_upper_threshold_{model}",
                f"pmp_error_degenerate_interval_{model}",
            )
        },
    }
    needs_calibration_refresh = any(
        not required.issubset(frames[name].columns)
        for name, required in required_calibration_columns.items()
    )
    if needs_calibration_refresh and can_attach_thresholds:
        network_tag = result_dir.name.removeprefix("ood_")
        matches = tuple(
            network_set
            for network_set in discover_network_sets(network_tags=(network_tag,))
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one network set for {network_tag!r}; found {len(matches)}"
            )
        network_set = matches[0]
        configuration = configuration_tag(network_set.data_dim, network_set.num_obs)
        thresholds = _threshold_lookup(
            CALIBRATION_ROOT / configuration / "thresholds.csv",
            configuration,
            network_set,
        )
        posterior, logml, pmp = attach_calibrated_errors(
            frames["posterior"], frames["logml"], frames["pmp"], thresholds
        )
        frames = {"posterior": posterior, "logml": logml, "pmp": pmp}
    return frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        choices=("generate", "inference", "diagnostics", "all"),
        default="diagnostics",
        help=(
            "generate raw OOD data, compute posterior/logML/PMP, compute summary "
            "diagnostics, or run all stages"
        ),
    )
    parser.add_argument("--num-dims", type=int, default=20)
    parser.add_argument("--num-obs", type=int, default=10)
    parser.add_argument("--num-datasets", type=int, default=50)
    parser.add_argument("--num-posterior-samples", type=int, default=1000)
    parser.add_argument(
        "--logml-method",
        choices=("log_mean_exp", "mean_log"),
        default="log_mean_exp",
    )
    parser.add_argument(
        "--metrics", nargs="+", choices=diagnostic.REFERENCE_METRICS, default=["l2"]
    )
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument(
        "--summary-dims",
        type=int,
        nargs="+",
        help="Actual summary dimensions to evaluate; default: all available.",
    )
    parser.add_argument(
        "--network-tags",
        nargs="+",
        help="Optional saved-network tags, e.g. 20d_10n 40d_10n 80d_10n.",
    )
    parser.add_argument("--threshold-path", type=Path)
    parser.add_argument("--n-fit", type=int, default=2000)
    parser.add_argument("--n-calibration", type=int, default=2000)
    parser.add_argument("--n-density-validation", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--density-epochs", type=int, default=250)
    parser.add_argument("--density-batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument("--overwrite-inference", action="store_true")
    parser.add_argument("--overwrite-reference", action="store_true")
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also render calibrated figures; notebooks are the default plot entry point.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    diagnostic.quiet_bayesflow_progress()
    if args.num_datasets <= 1 or args.num_posterior_samples <= 1:
        raise ValueError("num_datasets and num_posterior_samples must exceed one")
    configuration = configuration_tag(args.num_dims, args.num_obs)
    network_sets = discover_network_sets(
        args.network_dir,
        data_dim=args.num_dims,
        num_obs=args.num_obs,
        summary_dims=tuple(args.summary_dims) if args.summary_dims else None,
        network_tags=tuple(args.network_tags) if args.network_tags else None,
    )
    if not network_sets:
        raise FileNotFoundError("No complete m1--m4 network sets match this request.")
    if args.result_dir is not None and len(network_sets) != 1:
        raise ValueError(
            "--result-dir is only valid when exactly one summary network is selected"
        )
    threshold_path = (
        args.threshold_path or CALIBRATION_ROOT / configuration / "thresholds.csv"
    )
    metrics = tuple(dict.fromkeys(args.metrics))
    for summary_index, network_set in enumerate(network_sets):
        result_dir = args.result_dir or RESULT_DIR / f"ood_{network_set.network_tag}"
        paths = OODPaths(result_dir)
        if args.stage in {"generate", "all"}:
            generate_ood_datasets(
                paths,
                network_set,
                num_datasets=args.num_datasets,
                seed=args.seed,
                overwrite=args.overwrite_datasets,
            )
        if args.stage in {"inference", "all"}:
            compute_ood_inference(
                paths,
                network_set,
                network_dir=args.network_dir,
                num_samples=args.num_posterior_samples,
                logml_method=args.logml_method,
                seed=args.seed + 100_000 * summary_index,
                overwrite=args.overwrite_inference,
            )
        if args.stage not in {"diagnostics", "all"}:
            continue

        output_root = result_dir / "diagnostics"
        reference_path = output_root / f"reference_suite_{network_set.network_tag}.pkl"
        approximators = load_approximators(network_set)
        simulators = {
            model: make_simulator(
                MODEL_SPECS[model],
                network_set.data_dim,
                network_set.num_obs,
                args.seed + 10_000 * summary_index + index,
            )
            for index, model in enumerate(ASSUMED_MODELS)
        }
        references = load_or_fit_references(
            reference_path,
            approximators,
            simulators,
            metrics,
            overwrite=args.overwrite_reference,
            reference_kwargs={
                "n_fit": args.n_fit,
                "n_calibration": args.n_calibration,
                "alpha": args.alpha,
                "n_boot": args.n_boot,
                "seed": args.seed + 10_000 * summary_index,
                "density_epochs": args.density_epochs,
                "density_batch_size": args.density_batch_size,
                "n_density_validation": args.n_density_validation,
            },
        )
        raw_datasets = load_cached_ood_datasets(result_dir)
        evaluated = diagnostic.add_summary_diagnostic_suite(
            raw_datasets,
            approximators,
            references,
            metrics=metrics,
        )
        thresholds = _threshold_lookup(threshold_path, configuration, network_set)
        posterior_errors = (
            load_cached_inference_results(result_dir)["posterior"]
            if paths.posterior.exists()
            else None
        )
        for metric in metrics:
            metric_output_dir = output_root / metric
            posterior_errors = save_metric_frames(
                metric_output_dir,
                metric,
                configuration,
                network_set,
                evaluated[metric],
                posterior_errors,
                thresholds,
            )
            if args.plots:
                generate_calibrated_plots(metric_output_dir)


if __name__ == "__main__":
    main()
