from __future__ import annotations

import os
import copy
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy
from matplotlib.lines import Line2D
from sklearn.covariance import LedoitWolf

import benchmark.examples.gaussian.direct.calculator as BF
from benchmark.examples.gaussian.config import (
    ASSUMED_MODELS,
    MODEL_SPECS,
    RESULT_DIR as GAUSSIAN_RESULT_DIR,
    SOURCE_MODELS,
)


RESULT_DIR = GAUSSIAN_RESULT_DIR / "ood"
FIGURE_DIR = RESULT_DIR / "figures"
SOURCE_COLORS = ("#F0E442", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#0072B2", "#E011CF", "#999999", "#1F03EE", "#882255", "#44AA99",)
TYPICAL_SET_FILL = "#DCEEDC"
DEFAULT_DISTANCE_METRIC = "l2"
REFERENCE_METRICS = ("l2", "linf", "mmd", "density")
DIAGNOSTIC_XLABELS = {
    "l2": r"Diagnostic: $L_2$-based",
    "linf": r"Diagnostic: $L_\infty$-based",
    "mmd": "Diagnostic: Kernel-based",
    "density": "Diagnostic: density-based",
}
NEAREST_TWO_CLASS_MARKERS = {
    "both extrapolative": ("o", 32),
    "only one not extrapolative": ("D", 46),
    "both not extrapolative": ("X", 52),
}
PLOT_FONT = {"title": 16, "label": 16, "tick": 16, "suptitle": 16, "legend": 15, "note": 16}


def misspec_label(assumed: str, source: str) -> str:
    a, s = MODEL_SPECS[assumed], MODEL_SPECS[source]
    prior_diff = not (np.isclose(a["mu_prior_mean"], s["mu_prior_mean"]) and np.isclose(a["mu_prior_std"], s["mu_prior_std"]))
    likelihood_diff = not np.isclose(a["likelihood_std"], s["likelihood_std"])
    return "J" if prior_diff and likelihood_diff else "P" if prior_diff else "L" if likelihood_diff else ""


MISSPEC_LABELS = {a: {s: misspec_label(a, s) for s in SOURCE_MODELS} for a in ASSUMED_MODELS}

def stack_obs(data: list[dict] | dict | np.ndarray, obs_key: str = "x") -> np.ndarray:
    """Stack observation arrays into shape (n_datasets, n_obs, n_dims)."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if isinstance(data, dict):
        return np.asarray(data[obs_key], dtype=np.float32)
    return np.stack([np.asarray(item[obs_key], dtype=np.float32) for item in data], axis=0)


def summary_outputs(approximator, x_batch: np.ndarray, obs_key: str = "x") -> np.ndarray:
    """Compute the standardized summary-network outputs used by the approximator."""
    z = approximator.summarize({obs_key: np.asarray(x_batch, dtype=np.float32)})
    return np.asarray(keras.ops.convert_to_numpy(z), dtype=np.float64)


def _rbf_kernel_mean(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth2: float,
    chunk_size: int = 512,
) -> float:
    """Return the mean RBF kernel value without materializing all x/y pairs."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    bandwidth2 = max(float(bandwidth2), 1e-8)
    total = 0.0
    count = 0
    for start in range(0, len(x), chunk_size):
        block = x[start : start + chunk_size]
        dist2 = np.sum((block[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        total += float(np.exp(-dist2 / (2.0 * bandwidth2)).sum())
        count += int(block.shape[0] * y.shape[0])
    return total / count


def mmd_rbf_bandwidth2(
    samples: np.ndarray,
    max_pairs: int = 500_000,
    seed: int = 2025,
) -> float:
    """Median positive squared-distance bandwidth used by the diffusion case study."""
    samples = np.atleast_2d(np.asarray(samples, dtype=np.float64))
    n = len(samples)
    if n < 2:
        return 1.0
    rng = np.random.default_rng(seed)
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        diff = samples[:, None, :] - samples[None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        positive = dist2[dist2 > 0.0]
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n - 1, size=max_pairs)
        j = j + (j >= i)
        diff = samples[i] - samples[j]
        positive = np.sum(diff * diff, axis=-1)
    if len(positive) == 0:
        return 1.0
    return max(float(np.median(positive)), 1e-8)


def mmd_reference_distance_from_summary(
    summaries: np.ndarray,
    reference_summary: np.ndarray,
    bandwidth2: float,
    reference_kernel_mean: float | None = None,
    chunk_size: int = 512,
) -> np.ndarray:
    """Biased RBF-MMD distance from each summary point to the reference law."""
    summaries = np.atleast_2d(np.asarray(summaries, dtype=np.float64))
    reference_summary = np.atleast_2d(
        np.asarray(reference_summary, dtype=np.float64)
    )
    bandwidth2 = max(float(bandwidth2), 1e-8)
    if reference_kernel_mean is None:
        reference_kernel_mean = _rbf_kernel_mean(
            reference_summary, reference_summary, bandwidth2
        )
    distances = np.empty(len(summaries), dtype=np.float64)
    for start in range(0, len(summaries), chunk_size):
        block = summaries[start : start + chunk_size]
        dist2 = np.sum(
            (block[:, None, :] - reference_summary[None, :, :]) ** 2,
            axis=-1,
        )
        kxy_mean = np.exp(-dist2 / (2.0 * bandwidth2)).mean(axis=1)
        mmd2 = 1.0 + float(reference_kernel_mean) - 2.0 * kxy_mean
        distances[start : start + len(block)] = np.sqrt(np.maximum(mmd2, 0.0))
    return distances


def squared_mmd_rbf_two_sample(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 512,
    seed: int = 2025,
) -> float:
    """Small validation MMD used to check a fitted density flow."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]
    bandwidth2 = mmd_rbf_bandwidth2(np.vstack([x, y]), seed=seed)
    return float(
        _rbf_kernel_mean(x, x, bandwidth2)
        + _rbf_kernel_mean(y, y, bandwidth2)
        - 2.0 * _rbf_kernel_mean(x, y, bandwidth2)
    )


def fit_typicality_flow(
    summaries: np.ndarray,
    epochs: int = 250,
    batch_size: int = 128,
    depth: int = 6,
    widths: tuple[int, ...] = (256, 256),
    learning_rate: float = 5e-4,
    patience: int = 20,
    start_from_epoch: int = 50,
):
    """Fit q_phi(z) to well-specified summary outputs."""
    import bayesflow as bf

    summaries = np.asarray(summaries, dtype=np.float32)
    dataset = bf.datasets.OfflineDataset(
        data={"z": summaries},
        batch_size=batch_size,
        adapter=bf.Adapter().rename("z", "inference_variables"),
    )
    flow = bf.approximators.ContinuousApproximator(
        inference_network=bf.networks.CouplingFlow(
            depth=depth,
            subnet_kwargs={"widths": widths, "activation": "mish", "norm": "layer"},
        ),
    )
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(1, epochs * dataset.num_batches),
        alpha=1e-6,
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=patience,
            start_from_epoch=start_from_epoch,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    flow.compile(keras.optimizers.Adam(learning_rate=schedule))
    history = flow.fit(dataset=dataset, epochs=epochs, callbacks=callbacks)
    return flow, history


def typicality_log_density(flow, summaries: np.ndarray) -> np.ndarray:
    summaries = np.asarray(summaries, dtype=np.float32)
    tensor = keras.ops.convert_to_tensor(summaries)
    standardized = flow.standardizer.maybe_standardize(
        tensor, key="inference_variables", stage="inference"
    )
    _, log_q = flow.inference_network(standardized, density=True, training=False)
    return np.asarray(keras.ops.convert_to_numpy(log_q), dtype=np.float64).reshape(-1)


def flow_summary_samples(flow, num_samples: int, seed: int = 2025) -> np.ndarray:
    samples = flow.sample(num_samples=num_samples, seed=seed)
    return np.asarray(samples["inference_variables"], dtype=np.float64)


def density_flow_validation(
    flow,
    heldout_summary: np.ndarray,
    num_flow_samples: int | None = None,
    max_mmd_samples: int = 512,
    seed: int = 2025,
) -> dict[str, float]:
    heldout_summary = np.asarray(heldout_summary, dtype=np.float64)
    generated = flow_summary_samples(
        flow, int(num_flow_samples or len(heldout_summary)), seed=seed
    )
    mmd2 = squared_mmd_rbf_two_sample(
        heldout_summary, generated, max_samples=max_mmd_samples, seed=seed
    )
    return {
        "density_validation_n_simulator": int(len(heldout_summary)),
        "density_validation_n_flow": int(len(generated)),
        "density_validation_mmd2": float(mmd2),
        "density_validation_mmd": float(np.sqrt(max(mmd2, 0.0))),
    }


def signed_typicality_from_summary(
    summaries: np.ndarray,
    flow,
    expected_log_density: float,
) -> np.ndarray:
    return typicality_log_density(flow, summaries) - float(expected_log_density)


def fit_reference(
    approximator,
    simulator,
    n_ref: int = 2000,
    alpha: float = 0.05,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
    n_boot: int = 1000,
    bootstrap_seed: int = 2025,
) -> dict:
    """Fit summary statistics and calibrate reference distances on independent datasets."""
    # First batch: estimate the reference mean and covariance.
    S_fit = summary_outputs(approximator, simulator.sample(n_ref)["x"])
    lw = LedoitWolf().fit(S_fit)
    reference = {
        "distance_metric": distance_metric,
        "summary_dim": int(S_fit.shape[1]),
        "mu_hat": lw.location_,
        "L_hat": np.linalg.cholesky(lw.covariance_),
    }

    # Second independent batch: calibrate the reference-distance distribution.
    x_calibration = simulator.sample(n_ref)["x"]
    S_calibration = summary_outputs(approximator, x_calibration)
    calibration_distances = summary_distance_from_summary(
        S_calibration,
        reference["mu_hat"],
        reference["L_hat"],
        metric=distance_metric,
    )
    stats = bootstrap_reference_stats(calibration_distances, alpha=alpha, n_boot=n_boot, seed=bootstrap_seed)
    reference.update({
        "median": stats["median"],
        "dm_low": stats["dm_low"],
        "dm_high": stats["dm_high"],
        "alpha": alpha,
    })
    return reference


def bootstrap_reference_stats(
    distances: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 2025,
) -> dict[str, float]:
    """Estimate reference quantiles by bootstrap resampling."""
    distances = np.asarray(distances, dtype=float)
    if n_boot <= 0:
        return {
            "median": float(np.median(distances)),
            "dm_low": float(np.percentile(distances, 100 * alpha / 2)),
            "dm_high": float(np.percentile(distances, 100 * (1 - alpha / 2))),
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(distances, size=(n_boot, len(distances)), replace=True)
    qs = np.percentile(samples, [50, 100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return {"median": float(qs[0].mean()), "dm_low": float(qs[1].mean()), "dm_high": float(qs[2].mean())}


def fit_reference_suite(
    approximator,
    simulator,
    metrics: tuple[str, ...] = REFERENCE_METRICS,
    n_fit: int = 2000,
    n_calibration: int = 2000,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = 2025,
    density_epochs: int = 250,
    density_batch_size: int = 128,
    n_density_validation: int = 2000,
) -> dict[str, dict]:
    """Fit l2, linf, MMD, and density references from shared summary samples."""
    metrics = tuple(metric.lower() for metric in metrics)
    unknown = set(metrics).difference(REFERENCE_METRICS)
    if unknown:
        raise ValueError(f"Unknown reference metrics: {sorted(unknown)}")
    if n_fit <= 1 or n_calibration <= 1:
        raise ValueError("n_fit and n_calibration must both be greater than one")

    fit_summary = summary_outputs(approximator, simulator.sample(n_fit)["x"])
    calibration_summary = summary_outputs(
        approximator, simulator.sample(n_calibration)["x"]
    )
    summary_dim = int(fit_summary.shape[1])
    references: dict[str, dict] = {}

    if set(metrics) & {"l2", "linf"}:
        covariance = LedoitWolf().fit(fit_summary)
        mean = covariance.location_
        chol = np.linalg.cholesky(covariance.covariance_)
        for metric in metrics:
            if metric not in {"l2", "linf"}:
                continue
            distances = summary_distance_from_summary(
                calibration_summary, mean, chol, metric=metric
            )
            references[metric] = {
                "distance_metric": metric,
                "metric": metric,
                "summary_dim": summary_dim,
                "mu_hat": mean,
                "L_hat": chol,
                "alpha": alpha,
                **bootstrap_reference_stats(
                    distances, alpha=alpha, n_boot=n_boot, seed=seed
                ),
            }

    if "mmd" in metrics:
        bandwidth2 = mmd_rbf_bandwidth2(fit_summary, seed=seed)
        kernel_mean = _rbf_kernel_mean(fit_summary, fit_summary, bandwidth2)
        distances = mmd_reference_distance_from_summary(
            calibration_summary, fit_summary, bandwidth2, kernel_mean
        )
        references["mmd"] = {
            "distance_metric": "mmd",
            "metric": "mmd",
            "summary_dim": summary_dim,
            "reference_summary": np.asarray(fit_summary, dtype=np.float64),
            "bandwidth2": float(bandwidth2),
            "reference_kernel_mean": float(kernel_mean),
            "alpha": alpha,
            **bootstrap_reference_stats(
                distances, alpha=alpha, n_boot=n_boot, seed=seed
            ),
        }

    if "density" in metrics:
        flow, history = fit_typicality_flow(
            fit_summary, epochs=density_epochs, batch_size=density_batch_size
        )
        calibration_log_q = typicality_log_density(flow, calibration_summary)
        expected_log_density = float(np.mean(calibration_log_q))
        calibration_typicality = calibration_log_q - expected_log_density
        signed_surprise_distance = -calibration_typicality
        stats = {
            "median": float(np.median(signed_surprise_distance)),
            "dm_low": float(
                np.percentile(signed_surprise_distance, 100 * alpha / 2)
            ),
            "dm_high": float(
                np.percentile(signed_surprise_distance, 100 * (1 - alpha / 2))
            ),
        }
        validation_summary = summary_outputs(
            approximator, simulator.sample(n_density_validation)["x"]
        )
        references["density"] = {
            "distance_metric": "density",
            "metric": "density",
            "summary_dim": summary_dim,
            "flow": flow,
            "expected_log_density": expected_log_density,
            "std_log_density": float(np.std(calibration_log_q)),
            "typicality_low": float(-stats["dm_high"]),
            "typicality_high": float(-stats["dm_low"]),
            "typicality_tau": float(
                max(abs(stats["dm_low"]), abs(stats["dm_high"]))
            ),
            "alpha": alpha,
            "density_epochs": int(density_epochs),
            "density_batch_size": int(density_batch_size),
            "density_validation_samples": int(n_density_validation),
            "density_training_history": {
                key: [float(value) for value in values]
                for key, values in getattr(history, "history", {}).items()
            },
            **stats,
            **density_flow_validation(
                flow,
                validation_summary,
                num_flow_samples=n_density_validation,
                seed=seed,
            ),
        }

    return {metric: references[metric] for metric in metrics}


def fit_summary_reference_suites(
    approximators: dict[str, object],
    simulators: dict[str, object],
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    **kwargs,
) -> dict[str, dict[str, dict]]:
    """Fit a complete diagnostic reference suite for every assumed model."""
    return {
        model: fit_reference_suite(
            approximators[model],
            simulators[model],
            seed=int(kwargs.get("seed", 2025)) + index,
            **{key: value for key, value in kwargs.items() if key != "seed"},
        )
        for index, model in enumerate(assumed_models)
    }


def save_reference_suites(references: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(references, file)
    return path


def load_reference_suites(path: str | Path) -> dict:
    with Path(path).open("rb") as file:
        return pickle.load(file)




def summary_distance_from_summary(
    S: np.ndarray,
    mu_hat: np.ndarray,
    L_hat: np.ndarray,
    metric: str = DEFAULT_DISTANCE_METRIC,
) -> np.ndarray:
    """Compute a dimension-normalized distance from whitened summary vectors."""
    whitened = np.linalg.solve(L_hat, (np.asarray(S, dtype=np.float64) - mu_hat).T).T
    summary_dim = whitened.shape[1]
    metric = metric.lower()
    if metric == "l2":
        return np.linalg.norm(whitened, axis=1) / np.sqrt(summary_dim)
    if metric == "linf":
        scale = np.sqrt(2 * np.log(summary_dim)) if summary_dim > 1 else 1.0
        return np.max(np.abs(whitened), axis=1) / scale
    raise ValueError("distance_metric must be 'l2' or 'linf'")


def summary_distance_from_obs(
    approximator,
    x_batch: list[dict] | dict | np.ndarray,
    reference: dict,
) -> np.ndarray:
    """Compute summary-space distances for observation datasets."""
    x_batch = stack_obs(x_batch)
    S = summary_outputs(approximator, x_batch)
    metric = reference.get(
        "metric", reference.get("distance_metric", DEFAULT_DISTANCE_METRIC)
    ).lower()
    if metric == "mmd":
        return mmd_reference_distance_from_summary(
            S,
            reference["reference_summary"],
            reference["bandwidth2"],
            reference.get("reference_kernel_mean"),
        )
    if metric == "density":
        return -signed_typicality_from_summary(
            S, reference["flow"], reference["expected_log_density"]
        )
    return summary_distance_from_summary(
        S,
        reference["mu_hat"],
        reference["L_hat"],
        metric=metric,
    )


def quiet_bayesflow_progress() -> None:
    from tqdm.auto import tqdm as original_tqdm
    import bayesflow.approximators.helpers.conditions as bf_conditions
    import bayesflow.approximators.helpers.samplers as bf_samplers

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    bf_samplers.tqdm = quiet_tqdm
    bf_conditions.tqdm = quiet_tqdm


def compute_logml_and_posteriors(
    datasets: dict[str, list[dict]],
    calculations: dict[str, object],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    logml_method: str | None = None,
) -> dict[str, list[dict]]:
    """Compute posterior samples and logmls using log-mean-exp aggregation."""
    if logml_method is not None and logml_method != "log_mean_exp":
        raise ValueError("logml_method must be 'log_mean_exp'")

    for assumed in assumed_models:
        calculation = calculations[assumed]
        if logml_method is not None:
            calculation.logml_method = logml_method
        for source in sources:
            datasets[source] = calculation.normal_analytical(datasets[source])
            datasets[source] = calculation.npe_estimation(datasets[source])
            datasets[source] = calculation.npe_estimation_use_gold_posterior(datasets[source])
    return datasets


def compute_model_probabilities(
    datasets: dict[str, list[dict]],
    direct_approximator,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> dict[str, list[dict]]:
    for source in sources:
        datasets[source] = BF.direct_get_probs(datasets[source], direct_approximator)
        datasets[source] = BF.indirect_get_probs(datasets[source], assumed_models)
    return datasets



def fit_summary_references(
    approximators: dict[str, object],
    simulators: dict[str, object],
    n_ref: int = 2000,
    alpha: float = 0.05,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
    n_boot: int = 1000,
    bootstrap_seed: int = 2025,
) -> dict[str, dict]:
    distance_metric = distance_metric.lower()
    if distance_metric not in REFERENCE_METRICS:
        raise ValueError(f"distance_metric must be one of {REFERENCE_METRICS}")
    if distance_metric in {"mmd", "density"}:
        return {
            model: fit_reference_suite(
                approximators[model],
                simulators[model],
                metrics=(distance_metric,),
                n_fit=n_ref,
                n_calibration=n_ref,
                alpha=alpha,
                n_boot=n_boot,
                seed=bootstrap_seed + index,
            )[distance_metric]
            for index, model in enumerate(assumed_models)
        }
    return {
        m: fit_reference(
            approximators[m],
            simulators[m],
            n_ref=n_ref,
            alpha=alpha,
            distance_metric=distance_metric,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed + i,
        )
        for i, m in enumerate(assumed_models)
    }


def distance_regime(distance: float, reference: dict) -> str:
    low = reference.get("dm_low", reference.get("low"))
    high = reference.get("dm_high", reference.get("high"))
    if low is None or high is None:
        raise KeyError("reference must contain dm_low/dm_high or low/high")
    if distance < low:
        return "interpolation"
    if distance > high:
        return "extrapolation"
    return "in_distribution"


def _ranking_distance(distances: np.ndarray, reference: dict) -> np.ndarray:
    """Use calibrated interval exceedance for the signed density diagnostic."""
    metric = reference.get(
        "metric", reference.get("distance_metric", DEFAULT_DISTANCE_METRIC)
    )
    distances = np.asarray(distances, dtype=float)
    if metric != "density":
        return distances
    return np.maximum(
        float(reference["dm_low"]) - distances,
        distances - float(reference["dm_high"]),
    ).clip(min=0.0)


def _add_precomputed_distances_and_regimes(
    datasets: dict[str, list[dict]],
    distances_by_source: dict[str, dict[str, np.ndarray]],
    references: dict[str, dict],
    sources: tuple[str, ...],
    assumed_models: tuple[str, ...],
    eps: float,
) -> dict[str, list[dict]]:
    for source in sources:
        distances = distances_by_source[source]
        for i, item in enumerate(datasets[source]):
            d_vec = np.array([distances[m][i] for m in assumed_models], dtype=float)
            ranking_vec = np.array(
                [
                    _ranking_distance(np.asarray([d_vec[j]]), references[m])[0]
                    for j, m in enumerate(assumed_models)
                ],
                dtype=float,
            )
            regimes = {
                m: distance_regime(d_vec[j], references[m])
                for j, m in enumerate(assumed_models)
            }
            order = np.argsort(ranking_vec, kind="stable")
            all_extra = all(v == "extrapolation" for v in regimes.values())
            item["summary_distances"] = {
                m: float(d_vec[j]) for j, m in enumerate(assumed_models)
            }
            item["summary_regimes"] = regimes
            item["summary_ci"] = {
                m: {
                    "median": float(references[m]["median"]),
                    "low": float(references[m]["dm_low"]),
                    "high": float(references[m]["dm_high"]),
                }
                for m in assumed_models
            }
            item["globally_extrapolative"] = bool(all_extra)
            item["at_least_one_not_extrapolative"] = not all_extra
            item["closest_summary_models"] = [
                assumed_models[j] for j in order[:2]
            ]
            item["d_min"] = float(ranking_vec[order[0]])
            item["d_second"] = float(ranking_vec[order[1]])
            item["ambiguity_score_true"] = float(
                1.0 / (abs(ranking_vec[order[1]] - ranking_vec[order[0]]) + eps)
            )
            item["ambiguity_score"] = (
                item["ambiguity_score_true"] if all_extra else 0.0
            )
    return datasets


def add_distances_and_regimes(
    datasets: dict[str, list[dict]],
    approximators: dict[str, object],
    references: dict[str, dict],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    eps: float = 1e-8,
) -> dict[str, list[dict]]:
    distances_by_source = {}
    for source in sources:
        x_batch = stack_obs(datasets[source])
        distances_by_source[source] = {
            model: summary_distance_from_obs(
                approximators[model], x_batch, references[model]
            )
            for model in assumed_models
        }
    return _add_precomputed_distances_and_regimes(
        datasets,
        distances_by_source,
        references,
        sources,
        assumed_models,
        eps,
    )


def add_summary_diagnostic_suite(
    datasets: dict[str, list[dict]],
    approximators: dict[str, object],
    reference_suites: dict[str, dict[str, dict]],
    metrics: tuple[str, ...] = REFERENCE_METRICS,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    eps: float = 1e-8,
) -> dict[str, dict[str, list[dict]]]:
    """Evaluate several diagnostics while computing each model summary only once."""
    unknown = set(metrics).difference(REFERENCE_METRICS)
    if unknown:
        raise ValueError(f"Unknown diagnostic metrics: {sorted(unknown)}")
    summaries = {
        source: {
            model: summary_outputs(approximators[model], stack_obs(datasets[source]))
            for model in assumed_models
        }
        for source in sources
    }
    output = {}
    for metric in metrics:
        references = {
            model: reference_suites[model][metric] for model in assumed_models
        }
        distances_by_source = {}
        for source in sources:
            distances_by_source[source] = {}
            for model in assumed_models:
                reference = references[model]
                summary = summaries[source][model]
                if metric == "mmd":
                    distance = mmd_reference_distance_from_summary(
                        summary,
                        reference["reference_summary"],
                        reference["bandwidth2"],
                        reference.get("reference_kernel_mean"),
                    )
                elif metric == "density":
                    distance = -signed_typicality_from_summary(
                        summary,
                        reference["flow"],
                        reference["expected_log_density"],
                    )
                else:
                    distance = summary_distance_from_summary(
                        summary,
                        reference["mu_hat"],
                        reference["L_hat"],
                        metric=metric,
                    )
                distances_by_source[source][model] = distance
        metric_datasets = {
            source: [copy.copy(item) for item in datasets[source]] for source in sources
        }
        output[metric] = _add_precomputed_distances_and_regimes(
            metric_datasets,
            distances_by_source,
            references,
            sources,
            assumed_models,
            eps,
        )
    return output


def collect_logml_distance_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        for item in datasets[source]:
            for assumed in assumed_models:
                gold = float(item[f"gold_log_marginal_{assumed}"])
                npe = float(item[f"npe_log_marginal_{assumed}"])
                rows.append({
                    "source_model": source,
                    "id": int(item["id"]),
                    "assumed_model": assumed,
                    "d_M": float(item["summary_distances"][assumed]),
                    "dm_median": float(item["summary_ci"][assumed]["median"]),
                    "dm_low": float(item["summary_ci"][assumed]["low"]),
                    "dm_high": float(item["summary_ci"][assumed]["high"]),
                    "distance_regime": item["summary_regimes"][assumed],
                    "gold_logml": gold,
                    "npe_logml": npe,
                    "signed_logml_error": npe - gold,
                })
    return pd.DataFrame(rows)


def collect_posterior_distance_frame(
    datasets: dict[str, list[dict]],
    n_samples: int = 1000,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    """Compare NPE and analytical posteriors for each dataset."""
    rows = []
    mmd = MaximumMeanDiscrepancy(kernel="gaussian")
    for source in sources:
        for item in datasets[source]:
            for assumed in assumed_models:
                gold = np.asarray(item[f"gold_post_samples_{assumed}"], dtype=np.float32)
                npe = np.asarray(item[f"npe_post_samples_{assumed}"], dtype=np.float32)
                sample_count = min(n_samples, len(gold), len(npe))
                if sample_count <= 1:
                    raise ValueError("MMD requires at least two posterior samples from each distribution")
                gold = gold[:sample_count]
                npe = npe[:sample_count]
                rows.append({
                    "source_model": source,
                    "id": int(item["id"]),
                    "assumed_model": assumed,
                    "d_M": float(item["summary_distances"][assumed]),
                    "dm_median": float(item["summary_ci"][assumed]["median"]),
                    "dm_low": float(item["summary_ci"][assumed]["low"]),
                    "dm_high": float(item["summary_ci"][assumed]["high"]),
                    "distance_regime": item["summary_regimes"][assumed],
                    "posterior_mmd": float(mmd(tf.convert_to_tensor(npe), tf.convert_to_tensor(gold))),
                    "posterior_mean_rmse": float(np.sqrt(np.mean((npe.mean(axis=0) - gold.mean(axis=0)) ** 2))),
                    "n_posterior_samples": sample_count,
                })
    return pd.DataFrame(rows)


def collect_pmp_ambiguity_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    n_models = len(assumed_models)
    for source in sources:
        for item in datasets[source]:
            gold = BF.softmax_stable([item[f"gold_log_marginal_{m}"] for m in assumed_models])
            npe = BF.softmax_stable([item[f"npe_log_marginal_{m}"] for m in assumed_models])
            direct = np.asarray(item["p_direct"], dtype=float)
            direct_ok = len(direct) == n_models
            row = {
                "source_model": source,
                "id": int(item["id"]),
                "globally_extrapolative": bool(item["globally_extrapolative"]),
                "at_least_one_not_extrapolative": bool(item["at_least_one_not_extrapolative"]),
                "not_extrapolative_count": sum(v != "extrapolation" for v in item["summary_regimes"].values()),
                "ambiguity_score": float(item["ambiguity_score"]),
                "ambiguity_score_true": float(item.get("ambiguity_score_true", 1.0 / (abs(item["d_second"] - item["d_min"]) + 1e-8))),
                "d_min": float(item["d_min"]),
                "d_second": float(item["d_second"]),
                "closest_summary_models": ",".join(item["closest_summary_models"]),
                "pmp_l1_error_npe": float(np.sum(np.abs(npe - gold))),
                "pmp_l1_error_direct": float(np.sum(np.abs(direct - gold))) if direct_ok else np.nan,
            }
            row["extrapolation_class"] = _extrapolation_class(row["not_extrapolative_count"])
            row["nearest_two_extrapolation_class"] = _nearest_two_extrapolation_class(item["closest_summary_models"], item["summary_regimes"])
            for j, assumed in enumerate(assumed_models):
                regime = item["summary_regimes"][assumed]
                row[f"d_{assumed}"] = float(item["summary_distances"][assumed])
                row[f"dm_median_{assumed}"] = float(item["summary_ci"][assumed]["median"])
                row[f"dm_low_{assumed}"] = float(item["summary_ci"][assumed]["low"])
                row[f"dm_high_{assumed}"] = float(item["summary_ci"][assumed]["high"])
                row[f"regime_{assumed}"] = regime
                row[f"p_gold_{assumed}"] = float(gold[j])
                row[f"p_npe_{assumed}"] = float(npe[j])
                row[f"p_direct_{assumed}"] = float(direct[j]) if direct_ok else np.nan
                row[f"signed_pmp_error_npe_{assumed}"] = float(npe[j] - gold[j])
                row[f"signed_pmp_error_direct_{assumed}"] = float(direct[j] - gold[j]) if direct_ok else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_frames(logml_df: pd.DataFrame, pmp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logml_summary = (
        logml_df.groupby(["assumed_model", "source_model", "distance_regime"], sort=False)
        .agg(n=("id", "count"), median_d_M=("d_M", "median"), median_signed_logml_error=("signed_logml_error", "median"))
        .reset_index()
    )
    pmp_summary = (
        pmp_df.groupby(["source_model", "at_least_one_not_extrapolative"], sort=False)
        .agg(n=("id", "count"), median_d_min=("d_min", "median"), median_A=("ambiguity_score", "median"), median_pmp_l1_error_npe=("pmp_l1_error_npe", "median"))
        .reset_index()
    )
    return logml_summary, pmp_summary


def _add_distance_regions(
    ax,
    low: float,
    high: float,
    x_max: float,
    x_min: float = 0.0,
    y_bounds: tuple[float, float] | None = None,
) -> None:
    """Highlight the calibrated diagnostic interval and optional error band."""
    del x_min, x_max  # Retained in the signature for compatibility with older notebooks.
    ax.axvspan(low, high, color=TYPICAL_SET_FILL, alpha=0.70, zorder=0)
    if y_bounds is not None:
        ax.axhspan(*y_bounds, color=TYPICAL_SET_FILL, alpha=0.70, zorder=0)
    ax.axvline(low, color="0.45", linestyle=":", linewidth=0.9, zorder=1)
    ax.axvline(high, color="0.25", linestyle="--", linewidth=0.9, zorder=1)


def _add_misspec_label(ax, assumed: str, source: str) -> None:
    label = MISSPEC_LABELS.get(assumed, {}).get(source, "")
    if label:
        ax.text(0.04, 0.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=PLOT_FONT["note"], fontweight="bold", bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 2})


def _set_shared_xlim(axes, x_max: float, x_min: float = 0.0) -> None:
    for ax in np.atleast_1d(axes):
        ax.set_xlim(x_min, x_max)


def _add_first_large_error(ax, data: pd.DataFrame, x_col: str, y_col: str, error_bound: float | None) -> None:
    if error_bound is None or data.empty:
        return
    ordered = data[[x_col, y_col]].dropna().sort_values(x_col)
    hit = ordered[np.abs(ordered[y_col]) > error_bound].head(1)
    if hit.empty:
        return
    x0 = float(hit[x_col].iloc[0])
    ax.axvline(x0, color="0.2", linestyle=":", linewidth=1.2)
    ax.text(x0, 0.96, f"{x0:.2g}", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=PLOT_FONT["note"], rotation=90)


def _error_subset_data(data: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    if subset is None:
        return data
    if subset == "all_extrapolative":
        return data[data["at_least_one_not_extrapolative"] == False]
    if subset == "not_all_extrapolative":
        return data[data["at_least_one_not_extrapolative"] == True]
    raise ValueError("error_subset must be None, 'all_extrapolative', or 'not_all_extrapolative'")


def _safe_log(x) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), 1e-12))


def _normalize_distance(x, median, high):
    return (x - median) / (high - median)


def _normalize_signed_error(x, median, low, high):
    """Map a signed-error median to 0 and its lower/upper bounds to -1/+1."""
    values = np.asarray(x, dtype=float)
    centers = np.asarray(median, dtype=float)
    lower = np.asarray(low, dtype=float)
    upper = np.asarray(high, dtype=float)
    lower_scale = centers - lower
    upper_scale = upper - centers
    if np.any(lower_scale <= 0.0) or np.any(upper_scale <= 0.0):
        raise ValueError("Signed-error median must lie strictly between its bounds")
    return np.where(
        values < centers,
        (values - centers) / lower_scale,
        (values - centers) / upper_scale,
    )


def _error_median(
    data: pd.DataFrame,
    value_column: str,
    model: str,
    supplied: float | dict[str, float] | None,
) -> float:
    """Use an explicit calibration median, or infer it from well-specified rows."""
    if isinstance(supplied, dict):
        return float(supplied[model])
    if supplied is not None:
        return float(supplied)
    values = pd.to_numeric(
        data.loc[data["source_model"].eq(model), value_column], errors="coerce"
    ).dropna()
    if values.empty:
        raise ValueError(f"No well-specified {value_column} values for {model}")
    return float(values.median())


def _style_axes(axes) -> None:
    for ax in np.atleast_1d(axes).ravel():
        ax.title.set_fontsize(PLOT_FONT["title"])
        ax.xaxis.label.set_size(PLOT_FONT["label"])
        ax.yaxis.label.set_size(PLOT_FONT["label"])
        ax.tick_params(labelsize=PLOT_FONT["tick"])


def _style_colorbar(cbar) -> None:
    cbar.ax.yaxis.label.set_size(PLOT_FONT["label"])
    cbar.ax.tick_params(labelsize=PLOT_FONT["tick"])


def _diagnostic_xlabel(distance_metric: str) -> str:
    metric = distance_metric.lower()
    if metric not in DIAGNOSTIC_XLABELS:
        raise ValueError(
            f"distance_metric must be one of {tuple(DIAGNOSTIC_XLABELS)}"
        )
    return DIAGNOSTIC_XLABELS[metric]


def _apply_axis_scales(
    ax,
    xscale: str = "linear",
    yscale: str = "linear",
    x_linthresh: float = 1.0,
    y_linthresh: float = 1.0,
) -> None:
    """Apply independently configurable linear, logarithmic, or symlog axes."""
    valid_scales = {"linear", "log", "symlog"}
    if xscale not in valid_scales or yscale not in valid_scales:
        raise ValueError("xscale and yscale must be 'linear', 'log', or 'symlog'")
    if xscale == "symlog":
        if x_linthresh <= 0:
            raise ValueError("x_linthresh must be positive")
        ax.set_xscale("symlog", linthresh=x_linthresh)
    else:
        ax.set_xscale(xscale)
    if yscale == "symlog":
        if y_linthresh <= 0:
            raise ValueError("y_linthresh must be positive")
        ax.set_yscale("symlog", linthresh=y_linthresh)
    else:
        ax.set_yscale(yscale)


def _extrapolation_class(n_not: int) -> str:
    if n_not == 0:
        return "all extrapolative"
    if n_not == 1:
        return "only one not extrapolative"
    return "two or more not extrapolative"


def _nearest_two_extrapolation_class(models: list[str] | tuple[str, ...] | str, regimes: dict[str, str]) -> str:
    models = models.split(",") if isinstance(models, str) else models
    n_not = sum(regimes[m] != "extrapolation" for m in models[:2])
    if n_not == 0:
        return "both extrapolative"
    if n_not == 1:
        return "only one not extrapolative"
    return "both not extrapolative"


def _with_extrapolation_class(df: pd.DataFrame) -> pd.DataFrame:
    if {"extrapolation_class", "nearest_two_extrapolation_class"}.issubset(df.columns):
        return df
    regimes = [f"regime_{m}" for m in ASSUMED_MODELS]
    d_cols = [f"d_{m}" for m in ASSUMED_MODELS]
    out = df.copy()
    if "extrapolation_class" not in out.columns:
        out["not_extrapolative_count"] = out[regimes].ne("extrapolation").sum(axis=1)
        out["extrapolation_class"] = out["not_extrapolative_count"].map(_extrapolation_class)
    if "nearest_two_extrapolation_class" not in out.columns:
        order = np.argsort(out[d_cols].to_numpy(float), axis=1)
        out["nearest_two_extrapolation_class"] = [
            _nearest_two_extrapolation_class([ASSUMED_MODELS[j] for j in idx[:2]], row)
            for idx, row in zip(order, out[regimes].rename(columns=lambda x: x.replace("regime_", "")).to_dict("records"), strict=False)
        ]
    return out




def plot_logml_error_vs_distance(
    logml_df: pd.DataFrame,
    color_by: str = "source",
    output_dir: str | Path | None = FIGURE_DIR,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    x: str = "rho",
    error_bound: float | None = None,
    error_median: float | dict[str, float] | None = None,
    x_min: float | None = None,
    filename: str | None = None,
    distance_metric: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    x_linthresh: float = 1.0,
    y_linthresh: float = 1.0,
):
    """Plot log marginal likelihood error vs distance, one panel per assumed model."""
    if x not in {"distance", "log_distance", "logdistance", "rho", "log_rho", "logrho"}:
        raise ValueError("x must be 'distance', 'log_distance', 'rho', or 'log_rho'")
    if color_by not in {"source", "gold_logml", "npe_logml", "signed_logml_error"}:
        raise ValueError("color_by must be 'source', 'gold_logml', 'npe_logml', or 'signed_logml_error'")

    use_log_distance = x in {"log_distance", "logdistance"}
    use_rho = x in {"rho", "log_rho", "logrho"}
    use_log_rho = x in {"log_rho", "logrho"}
    x_col = "_x"
    if x_min is None:
        x_min = -0.5 if use_log_distance or use_log_rho else 0.0

    fig, axes = plt.subplots(1, len(assumed_models), figsize=(5.1 * len(assumed_models), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
    last = None

    for ax, assumed in zip(axes, assumed_models, strict=False):
        sub = logml_df[logml_df["assumed_model"] == assumed].copy()
        normalized_error = error_bound is not None
        calibrated_columns = {
            "signed_logml_error_lower_threshold",
            "signed_logml_error_threshold",
        }
        calibrated_error = (
            not normalized_error
            and calibrated_columns.issubset(sub.columns)
        )
        y_col = "_log10_signed_logml_error"
        sub[y_col] = sub["signed_logml_error"] / np.log(10.0)
        error_low = error_high = None
        if normalized_error:
            median = _error_median(
                sub, "signed_logml_error", assumed, error_median
            )
            y_col = "_normalized_signed_logml_error"
            sub[y_col] = _normalize_signed_error(
                sub["signed_logml_error"], median, -error_bound, error_bound
            )
        elif calibrated_error:
            error_low = float(
                sub["signed_logml_error_lower_threshold"].median()
                / np.log(10.0)
            )
            error_high = float(
                sub["signed_logml_error_threshold"].median()
                / np.log(10.0)
            )
        if use_rho:
            rho = _normalize_distance(sub["d_M"], sub["dm_median"], sub["dm_high"])
            sub[x_col] = _safe_log(rho) if use_log_rho else rho
            rho_low = _normalize_distance(
                sub["dm_low"].iloc[0],
                sub["dm_median"].iloc[0],
                sub["dm_high"].iloc[0],
            )
            low = float(_safe_log(rho_low)) if use_log_rho else float(rho_low)
            high = 0.0 if use_log_rho else 1.0
            x_label = (
                _diagnostic_xlabel(distance_metric)
                if distance_metric is not None
                else rf"$\log \rho_{assumed[-1]}(y)$" if use_log_rho
                else rf"$\rho_{assumed[-1]}(y)$"
            )
        else:
            sub[x_col] = _safe_log(sub["d_M"]) if use_log_distance else sub["d_M"]
            low = float(_safe_log(sub["dm_low"].iloc[0])) if use_log_distance else float(sub["dm_low"].iloc[0])
            high = float(_safe_log(sub["dm_high"].iloc[0])) if use_log_distance else float(sub["dm_high"].iloc[0])
            x_label = r"$\log d_j(y)$" if use_log_distance else r"$d_j(y)$"
        plot_x_min = min(x_min, low) if use_rho else x_min
        x_max = max(float(sub[x_col].max()) * 1.05, high * 1.1)

        _add_distance_regions(
            ax,
            low,
            high,
            x_max,
            x_min=plot_x_min,
            y_bounds=(
                (-1.0, 1.0)
                if normalized_error
                else (error_low, error_high) if calibrated_error
                else None
            ),
        )
        if color_by == "source":
            for source in SOURCE_MODELS:
                group = sub[sub["source_model"] == source]
                if group.empty:
                    continue
                ax.scatter(
                    group[x_col],
                    group[y_col],
                    s=24,
                    color=color_map[source],
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=0.35,
                )
        else:
            last = ax.scatter(
                sub[x_col],
                sub[y_col],
                c=sub[color_by],
                cmap="viridis",
                s=24,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.35,
            )

        _add_first_large_error(
            ax,
            sub,
            x_col,
            y_col,
            1.0 if normalized_error else error_bound,
        )
        ax.axhline(0, color="0.35", linewidth=0.8)
        if normalized_error:
            for threshold in (-1.0, 1.0):
                ax.axhline(
                    threshold, color="0.35", linestyle=":", linewidth=0.9
                )
        elif calibrated_error:
            ax.axhline(
                error_low,
                color="0.45",
                linestyle=":",
                linewidth=1.2,
            )
            ax.axhline(
                error_high,
                color="0.25",
                linestyle="--",
                linewidth=1.2,
            )
        ax.set_xlim(plot_x_min, x_max)
        ax.set_title(rf"Assumed {assumed.upper()}")
        ax.set_xlabel(x_label)
        _apply_axis_scales(
            ax,
            xscale=xscale,
            yscale=yscale,
            x_linthresh=x_linthresh,
            y_linthresh=y_linthresh,
        )
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(
        "Normalized signed log marginal-likelihood error"
        if error_bound is not None
        else r"$\log_{10}\widehat{p}(y\mid M_j)-\log_{10}p(y\mid M_j)$"
    )
    _style_axes(axes)
    for ax in axes:
        offset = ax.yaxis.get_offset_text()
        offset.set_fontsize(PLOT_FONT["tick"])
        offset.set_fontweight("bold")
    if color_by == "source":
        _source_legend(fig, y=-0.035)
    elif last is not None:
        cbar = fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(color_by)
        _style_colorbar(cbar)

    fig.tight_layout(rect=(0, 0.065 if color_by == "source" else 0, 1, 1))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = {
            "distance": "distance",
            "log_distance": "log_distance",
            "logdistance": "log_distance",
            "rho": "rho",
            "log_rho": "log_rho",
            "logrho": "log_rho",
        }[x]
        fig.savefig(Path(output_dir) / (filename or f"logml_error_vs_{suffix}_by_assumed.png"), dpi=200, bbox_inches="tight")
    return fig, axes


def _pmp_long_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    pmp_df = _with_extrapolation_class(pmp_df).copy()
    if "ambiguity_score_true" not in pmp_df:
        pmp_df["ambiguity_score_true"] = 1.0 / (np.abs(pmp_df["d_second"] - pmp_df["d_min"]) + 1e-8)
    rows = []
    for model in ASSUMED_MODELS:
        cols = [
            "source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "ambiguity_score", "ambiguity_score_true", "d_min", "d_second",
            f"d_{model}", f"dm_median_{model}", f"dm_low_{model}", f"dm_high_{model}",
            f"p_gold_{model}", f"p_npe_{model}", f"p_direct_{model}", f"signed_pmp_error_{estimate}_{model}",
        ]
        part = pmp_df[cols].copy()
        part.columns = ["source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "A_raw", "A_true", "d_min", "d_second", "d_M", "dm_median", "dm_low", "dm_high", "gold", "npe", "direct", "signed_error"]
        part["rho_M"] = _normalize_distance(part["d_M"], part["dm_median"], part["dm_high"])
        part["rho_low"] = _normalize_distance(part["dm_low"], part["dm_median"], part["dm_high"])
        part["log_d_M"] = _safe_log(part["d_M"])
        part["log_d_min"] = _safe_log(part["d_min"])
        part["log_rho_M"] = _safe_log(part["rho_M"])
        part["model"] = model
        lower_threshold = f"pmp_error_lower_threshold_{model}"
        upper_threshold = f"pmp_error_upper_threshold_{model}"
        if {lower_threshold, upper_threshold}.issubset(pmp_df.columns):
            part["error_threshold_low"] = pmp_df[lower_threshold].to_numpy()
            part["error_threshold_high"] = pmp_df[upper_threshold].to_numpy()
        rows.append(part)
    return pd.concat(rows, ignore_index=True) # row number: 3 * 7 * 50


def _pmp_rmse_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    out = _with_extrapolation_class(pmp_df).copy()
    if "ambiguity_score_true" not in out:
        out["ambiguity_score_true"] = 1.0 / (np.abs(out["d_second"] - out["d_min"]) + 1e-8)
    error_cols = [f"signed_pmp_error_{estimate}_{m}" for m in ASSUMED_MODELS]
    out["pmp_rmse"] = np.sqrt(np.mean(np.square(out[error_cols].to_numpy(float)), axis=1))
    out["A_raw"] = out["ambiguity_score"]
    out["A_true"] = out["ambiguity_score_true"]
    out["log_d_min"] = _safe_log(out["d_min"])
    return out


def _nearest_distance_region(data: pd.DataFrame) -> tuple[float, float, float]: # return typical low/high distance values and max x for plotting based on nearest assumed model
    d_cols = [f"d_{m}" for m in ASSUMED_MODELS]
    nearest = data[d_cols].to_numpy().argmin(axis=1) # return indices of nearest assumed model for each row
    lows = [data[f"dm_low_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    highs = [data[f"dm_high_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    x_max = max(float(data["d_min"].max()) * 1.05, float(np.median(highs)) * 1.1)
    return float(np.median(lows)), float(np.median(highs)), x_max # median low/high distance values across rows based on nearest assumed model, and max x for plotting


def _marker_handles(marker_map: dict[object, tuple[str, int]]) -> list[Line2D]:
    labels = {False: "all high surprise", True: "at least one not high surprise"}
    return [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.55", markeredgecolor="black", markersize=7, label=labels.get(label, label))
        for label, (marker, _) in marker_map.items()
    ]


def _source_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLORS[i], markeredgecolor="black", markersize=7, label=source.upper())
        for i, source in enumerate(SOURCE_MODELS)
    ]


def _marker_legend(fig, marker_map: dict[object, tuple[str, int]], y: float = -0.07) -> None:
    fig.legend(handles=_marker_handles(marker_map), loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(marker_map), frameon=False, fontsize=PLOT_FONT["legend"])


def _source_legend(fig, y: float = -0.07) -> None:
    fig.legend(handles=_source_handles(), loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(SOURCE_MODELS), frameon=False, fontsize=PLOT_FONT["legend"])


def _right_rmse_legends(fig, group_by: str) -> None:
    marker_map = None
    if group_by in {"global_extrapolation", "global_extrapolation_source"}:
        marker_map = {False: ("o", 32), True: ("D", 48)}
    elif group_by in {"nearest_two", "nearest_two_source"}:
        marker_map = NEAREST_TWO_CLASS_MARKERS

    if marker_map is not None:
        fig.legend(
            handles=_marker_handles(marker_map),
            loc="center left",
            bbox_to_anchor=(0.76, 0.62),
            ncol=1,
            frameon=False,
            fontsize=PLOT_FONT["legend"],
        )

    if group_by in {"source_model", "global_extrapolation_source", "nearest_two_source"}:
        fig.legend(
            handles=_source_handles(),
            loc="center left",
            bbox_to_anchor=(0.76, 0.34),
            ncol=2,
            frameon=False,
            fontsize=PLOT_FONT["legend"],
        )


def _pmp_plot_data(pmp_df: pd.DataFrame, y: str, estimate: str) -> tuple[pd.DataFrame, bool]:
    if y == "signed_error":
        return _pmp_long_frame(pmp_df, estimate), True
    if y == "rmse":
        return _pmp_rmse_frame(pmp_df, estimate), False
    raise ValueError("y must be 'signed_error' or 'rmse'")


def _pmp_model_index(model: str) -> str:
    return model[1:] if model.lower().startswith("m") else model


def _pmp_x_column(
    x: str,
    model: str | None = None,
    distance_metric: str | None = None,
) -> tuple[str, str]:
    if x == "distance":
        if model is None:
            raise ValueError("x='distance' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "d_M", rf"$d_{j}(y)$"
    if x in {"log_distance", "logdistance"}:
        if model is None:
            raise ValueError("x='log_distance' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "log_d_M", rf"$\log d_{j}(y)$"
    if x == "rho":
        if model is None:
            raise ValueError("x='rho' is only available for model-wise PMP plots")
        label = (
            _diagnostic_xlabel(distance_metric)
            if distance_metric is not None
            else rf"$\rho_{_pmp_model_index(model)}(y)$"
        )
        return "rho_M", label
    if x in {"log_rho", "logrho"}:
        if model is None:
            raise ValueError("x='log_rho' is only available for model-wise PMP plots")
        label = (
            _diagnostic_xlabel(distance_metric)
            if distance_metric is not None
            else rf"$\log \rho_{_pmp_model_index(model)}(y)$"
        )
        return "log_rho_M", label
    if x == "A":
        return "A_raw", r"$A_{\mathrm{raw}}(y)$"
    if x in {"A_true", "trueA"}:
        return "A_true", r"$A_{\mathrm{true}}(y)$"
    if x == "d_min":
        return "d_min", r"$d_{\min}(y)$"
    if x == "log_d_min":
        return "log_d_min", r"$\log d_{\min}(y)$"
    raise ValueError("x must be 'distance', 'log_distance', 'rho', 'log_rho', 'A', 'A_true', 'd_min', or 'log_d_min'")


def _assumed_region_bounds(sub: pd.DataFrame, x: str) -> tuple[float, float]:
    if x == "rho":
        return float(sub["rho_low"].iloc[0]), 1.0
    if x in {"log_rho", "logrho"}:
        return float(_safe_log(sub["rho_low"].iloc[0])), 0.0
    low, high = float(sub["dm_low"].iloc[0]), float(sub["dm_high"].iloc[0])
    if x in {"log_distance", "logdistance"}:
        return float(_safe_log(low)), float(_safe_log(high))
    return low, high


def _signed_max_error_within_rho(
    data: pd.DataFrame,
    y_col: str,
    rho_col: str = "rho_M",
    rho_max: float = 1.0,
) -> float | None:
    if rho_col not in data or y_col not in data:
        return None
    subset = data.loc[data[rho_col].le(rho_max), [rho_col, y_col]].dropna()
    if subset.empty:
        return None
    return float(subset.loc[subset[y_col].abs().idxmax(), y_col])


def _add_rho_max_error_line(
    ax,
    data: pd.DataFrame,
    y_col: str,
    rho_max: float = 1.0,
) -> None:
    y0 = _signed_max_error_within_rho(data, y_col=y_col, rho_max=rho_max)
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




def _scatter_pmp(ax, data: pd.DataFrame, x_col: str, y_col: str, group_by: str):
    if group_by == "source_model":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        ax.scatter(data[x_col], data[y_col], c=data["source_model"].map(color_map), s=36, alpha=0.75, edgecolors="black", linewidths=0.4)
        return None
    if group_by == "global_extrapolation_source":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        marker_map = {False: ("o", 32), True: ("D", 48)}
        for group_value, (marker, size) in marker_map.items():
            group = data[data["at_least_one_not_extrapolative"] == group_value]
            ax.scatter(group[x_col], group[y_col], c=group["source_model"].map(color_map), s=size, alpha=0.75, marker=marker, edgecolors="black", linewidths=0.55)
        return None
    if group_by == "nearest_two_source":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        for group_value, (marker, size) in NEAREST_TWO_CLASS_MARKERS.items():
            group = data[data["nearest_two_extrapolation_class"] == group_value]
            ax.scatter(group[x_col], group[y_col], c=group["source_model"].map(color_map), s=size, alpha=0.75, marker=marker, edgecolors="black", linewidths=0.55)
        return None

    group_col = "at_least_one_not_extrapolative" if group_by == "global_extrapolation" else "nearest_two_extrapolation_class"
    marker_map = {
        "global_extrapolation": {False: ("o", 32), True: ("D", 48)},
        "nearest_two": NEAREST_TWO_CLASS_MARKERS,
    }[group_by]
    last = None
    for group_value, (marker, size) in marker_map.items():
        group = data[data[group_col] == group_value]
        if len(group) == 0:
            continue
        last = ax.scatter(group[x_col], group[y_col], c=group["gold"], cmap="viridis", vmin=0, vmax=1, s=size, alpha=0.72, marker=marker, edgecolors="black", linewidths=0.55)
    return last



def plot_pmp_diagnostic(
    pmp_df: pd.DataFrame,
    x: str,
    y: str = "signed_error",
    group_by: str = "global_extrapolation",
    estimate: str = "npe",
    output_dir: str | Path | None = FIGURE_DIR,
    filename: str | None = None,
    title: str | None = None,
    regions: str | None = None,
    sharex: bool = False,
    error_bound: float | None = None,
    error_median: float | dict[str, float] | None = None,
    error_subset: str | None = None,
    x_min: float = -0.5,
    show_rho_leq_one_max_error: bool = True,
    distance_metric: str | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    x_linthresh: float = 1.0,
    y_linthresh: float = 0.05,
):
    data, by_model = _pmp_plot_data(pmp_df, y, estimate)
    normalized_error = y == "signed_error" and error_bound is not None
    y_col = (
        "_normalized_signed_error"
        if normalized_error
        else "signed_error" if y == "signed_error" else "pmp_rmse"
    )
    y_label = (
        "Normalized signed PMP error"
        if normalized_error
        else r"$\hat{p}(M_j|y)-p(M_j|y)$" if y == "signed_error" else "PMP RMSE"
    )
    n_axes = len(ASSUMED_MODELS) if by_model else 1
    right_rmse_legend = y == "rmse" and group_by in {
        "source_model",
        "global_extrapolation",
        "global_extrapolation_source",
        "nearest_two",
        "nearest_two_source",
    }
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(4.9 * n_axes, 6) if by_model else (8.8, 5.1) if right_rmse_legend else (6.8, 5.1),
        sharex=sharex,
        sharey=by_model,
        constrained_layout=not right_rmse_legend,
    )
    if right_rmse_legend:
        fig.subplots_adjust(right=0.72)
    axes = np.atleast_1d(axes)
    last = None
    plot_data = []

    for model in ASSUMED_MODELS if by_model else [None]:
        sub = (data[data["model"] == model] if by_model else data).copy()
        calibrated_error = (
            y == "signed_error"
            and not normalized_error
            and {
                "error_threshold_low",
                "error_threshold_high",
            }.issubset(sub.columns)
        )
        threshold_low = threshold_high = None
        if normalized_error:
            median = _error_median(sub, "signed_error", model, error_median)
            sub[y_col] = _normalize_signed_error(
                sub["signed_error"], median, -error_bound, error_bound
            )
        elif calibrated_error:
            threshold_low = float(sub["error_threshold_low"].median())
            threshold_high = float(sub["error_threshold_high"].median())
        x_col, x_label = _pmp_x_column(
            x,
            model,
            distance_metric=distance_metric,
        )
        plot_x_min = x_min
        x_max = max(float(sub[x_col].max()) * 1.05, 1e-12)
        if regions == "assumed":
            low, high = _assumed_region_bounds(sub, x)
            if x == "rho":
                plot_x_min = min(plot_x_min, low)
            x_max = max(x_max, high * 1.1)
        elif regions == "nearest":
            _, _, x_max = _nearest_distance_region(pmp_df)
            if x == "log_d_min":
                x_max = float(_safe_log(x_max))
        plot_data.append(
            (
                model,
                sub,
                x_col,
                x_label,
                plot_x_min,
                x_max,
                calibrated_error,
                threshold_low,
                threshold_high,
            )
        )

    shared_x_max = max(item[5] for item in plot_data)
    shared_x_min = min(item[4] for item in plot_data)

    for ax, (
        model,
        sub,
        x_col,
        x_label,
        plot_x_min,
        x_max,
        calibrated_error,
        threshold_low,
        threshold_high,
    ) in zip(axes, plot_data, strict=False):
        plot_x_min = shared_x_min if sharex else plot_x_min
        plot_x_max = shared_x_max if sharex else x_max
        y_bounds = (
            (-1.0, 1.0)
            if normalized_error
            else (threshold_low, threshold_high) if calibrated_error
            else None
        )
        if regions == "assumed":
            low, high = _assumed_region_bounds(sub, x)
            _add_distance_regions(
                ax,
                low,
                high,
                plot_x_max,
                x_min=plot_x_min,
                y_bounds=y_bounds,
            )
            ax.set_xlim(plot_x_min, plot_x_max)
        elif regions == "nearest":
            low, high, _ = _nearest_distance_region(pmp_df)
            if x == "log_d_min":
                low, high = float(_safe_log(low)), float(_safe_log(high))
            _add_distance_regions(
                ax,
                low,
                high,
                plot_x_max,
                x_min=plot_x_min,
                y_bounds=y_bounds,
            )
            ax.set_xlim(plot_x_min, plot_x_max)
        elif normalized_error:
            ax.axhspan(-1.0, 1.0, color=TYPICAL_SET_FILL, alpha=0.70, zorder=0)
        elif calibrated_error:
            ax.axhspan(
                threshold_low,
                threshold_high,
                color=TYPICAL_SET_FILL,
                alpha=0.70,
                zorder=0,
            )

        last = _scatter_pmp(ax, sub, x_col, y_col, group_by) or last
        _add_first_large_error(
            ax,
            _error_subset_data(sub, error_subset),
            x_col,
            y_col,
            1.0 if normalized_error else error_bound,
        )
        if y == "signed_error":
            ax.axhline(0, color="0.35", linewidth=0.8)
            if normalized_error:
                for threshold in (-1.0, 1.0):
                    ax.axhline(
                        threshold, color="0.35", linestyle=":", linewidth=0.9
                    )
            elif calibrated_error:
                ax.axhline(
                    threshold_low,
                    color="0.45",
                    linestyle=":",
                    linewidth=1.2,
                )
                if not np.isclose(
                    threshold_low,
                    threshold_high,
                    rtol=0.0,
                    atol=1e-15,
                ):
                    ax.axhline(
                        threshold_high,
                        color="0.25",
                        linestyle="--",
                        linewidth=1.2,
                    )
            if by_model and show_rho_leq_one_max_error:
                _add_rho_max_error_line(ax, sub, y_col)
        ax.set_title(rf"$p(M_{model[-1]}\mid y)$" if by_model else title or "")
        ax.set_xlabel(x_label)
        _apply_axis_scales(
            ax,
            xscale=xscale,
            yscale=yscale,
            x_linthresh=x_linthresh,
            y_linthresh=y_linthresh,
        )
        ax.grid(alpha=0.2)

    if sharex and regions is None:
        _set_shared_xlim(axes, shared_x_max, x_min=x_min)

    axes[0].set_ylabel(y_label)
    _style_axes(axes)
    if title and by_model:
        fig.suptitle(title, y=1.04, fontsize=PLOT_FONT["suptitle"])
    if right_rmse_legend:
        _right_rmse_legends(fig, group_by)
    elif group_by == "source_model":
        _source_legend(fig)
    elif group_by == "global_extrapolation_source":
        _marker_legend(fig, {False: ("o", 32), True: ("D", 48)}, y=-0.07)
        _source_legend(fig, y=-0.15)
    elif group_by == "nearest_two_source":
        _marker_legend(fig, NEAREST_TWO_CLASS_MARKERS, y=-0.07)
        _source_legend(fig, y=-0.15)
    elif group_by == "global_extrapolation":
        _marker_legend(fig, {False: ("o", 32), True: ("D", 48)})
    elif group_by == "nearest_two":
        _marker_legend(fig, NEAREST_TWO_CLASS_MARKERS)
    if last is not None:
        _style_colorbar(fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02))
    if output_dir is not None and filename:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=200, bbox_inches="tight")
    return fig, axes if by_model else axes[0]


def plot_pmp_estimates_vs_distance(
    pmp_df: pd.DataFrame,
    source_model: str,
    x: str = "distance",
    y: str = "estimate",
    x_min: float | None = None,
    output_dir: str | Path | None = FIGURE_DIR,
    xscale: str = "linear",
    yscale: str = "linear",
    x_linthresh: float = 1.0,
    y_linthresh: float = 0.05,
):
    if x not in {"distance", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance' or 'log_distance'")
    if y not in {"estimate", "signed_error"}:
        raise ValueError("y must be 'estimate' or 'signed_error'")
    n_models = len(ASSUMED_MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.4 * n_models, 5.5), sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes)
    last = None
    data = pmp_df[pmp_df["source_model"] == source_model]
    use_log = x in {"log_distance", "logdistance"}
    x_min = -0.5 if x_min is None and use_log else 0.0 if x_min is None else x_min
    for ax, model in zip(axes, ASSUMED_MODELS, strict=False):
        x_values = _safe_log(data[f"d_{model}"]) if use_log else data[f"d_{model}"]
        low = float(_safe_log(data[f"dm_low_{model}"].iloc[0])) if use_log else float(data[f"dm_low_{model}"].iloc[0])
        high = float(_safe_log(data[f"dm_high_{model}"].iloc[0])) if use_log else float(data[f"dm_high_{model}"].iloc[0])
        x_max = max(float(np.max(x_values)) * 1.05, high * 1.1)
        _add_distance_regions(ax, low, high, x_max, x_min=x_min)
        if y == "estimate":
            npe_y = data[f"p_npe_{model}"]
            direct_y = data[f"p_direct_{model}"]
            ax.set_ylim(-0.02, 1.02)
        else:
            npe_y = data[f"p_npe_{model}"] - data[f"p_gold_{model}"]
            direct_y = data[f"p_direct_{model}"] - data[f"p_gold_{model}"]
            ax.axhline(0, color="0.35", linewidth=0.8)
            ax.set_ylim(-1.02, 1.02)
        last = ax.scatter(x_values, npe_y, c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=42, alpha=0.78, marker="o", label="NPE" if y == "estimate" else "NPE - gold")
        ax.scatter(x_values, direct_y, c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=88, alpha=0.78, marker="*", edgecolors="black", linewidths=0.5, label="NPMP" if y == "estimate" else "NPMP - gold")
        ax.set_title(rf"$p(M_{model[-1]} \mid y)$")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(rf"$\log d_{model[-1]}(y)$" if use_log else rf"$d_{model[-1]}(y)$")
        _apply_axis_scales(
            ax,
            xscale=xscale,
            yscale=yscale,
            x_linthresh=x_linthresh,
            y_linthresh=y_linthresh,
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated PMP" if y == "estimate" else "Signed PMP error")
    _style_axes(axes)
    fig.subplots_adjust(top=0.78, right=0.90, wspace=0.05)
    fig.suptitle(f"Observation datasets from {source_model.upper()}", y=0.96, fontsize=PLOT_FONT["suptitle"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=True, fontsize=PLOT_FONT["legend"])
    _style_colorbar(fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = "log_distance" if use_log else "distance"
        prefix = "pmp_estimates" if y == "estimate" else "signed_pmp_error"
        fig.savefig(Path(output_dir) / f"{prefix}_vs_{suffix}_{source_model}.png", dpi=200, bbox_inches="tight")
    return fig, axes
