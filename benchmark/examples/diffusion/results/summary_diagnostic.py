from __future__ import annotations

import pickle
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from ..config import DIAGNOSTIC_PATH, MODELS, MODEL_TITLES, REFERENCE_PATH
from .results import data_array, data_conditions


def summary_outputs(approximator, y) -> np.ndarray:
    summaries = approximator.summarize(data_conditions(y))
    return np.asarray(keras.ops.convert_to_numpy(summaries), dtype=np.float64)


def summary_distance_from_summary(
    summaries: np.ndarray,
    mean: np.ndarray,
    chol: np.ndarray,
    metric: str = "l2",
) -> np.ndarray:
    z = np.linalg.solve(chol, (np.asarray(summaries, dtype=np.float64) - mean).T).T
    dim = z.shape[1]
    if metric == "l2":
        return np.linalg.norm(z, axis=1) / np.sqrt(dim)
    if metric == "linf":
        return np.max(np.abs(z), axis=1) / np.sqrt(2.0 * np.log(dim))
    raise ValueError("metric must be 'l2' or 'linf'")


def _rbf_kernel_mean(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth2: float,
    chunk_size: int = 512,
) -> float:
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    bandwidth2 = max(float(bandwidth2), 1e-8)
    total = 0.0
    count = 0
    for start in range(0, len(x), chunk_size):
        block = x[start:start + chunk_size]
        dist2 = np.sum((block[:, None, :] - y[None, :, :]) ** 2, axis=-1)
        total += float(np.exp(-dist2 / (2.0 * bandwidth2)).sum())
        count += int(block.shape[0] * y.shape[0])
    return total / count


def mmd_rbf_bandwidth2(
    samples: np.ndarray,
    max_pairs: int = 500_000,
    seed: int = 2025,
) -> float:
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
    """Biased RBF-MMD distance from each singleton summary to a reference summary bank.

    This follows the MMD misspecification diagnostic setup where singleton observed
    datasets are allowed by using the biased MMD² estimator and plotting/saving its
    square root as the distance.
    """
    summaries = np.atleast_2d(np.asarray(summaries, dtype=np.float64))
    reference_summary = np.atleast_2d(np.asarray(reference_summary, dtype=np.float64))
    bandwidth2 = max(float(bandwidth2), 1e-8)
    if reference_kernel_mean is None:
        reference_kernel_mean = _rbf_kernel_mean(reference_summary, reference_summary, bandwidth2)

    distances = np.empty(len(summaries), dtype=np.float64)
    for start in range(0, len(summaries), chunk_size):
        block = summaries[start:start + chunk_size]
        dist2 = np.sum((block[:, None, :] - reference_summary[None, :, :]) ** 2, axis=-1)
        kxy_mean = np.exp(-dist2 / (2.0 * bandwidth2)).mean(axis=1)
        mmd2 = 1.0 + float(reference_kernel_mean) - 2.0 * kxy_mean
        distances[start:start + len(block)] = np.sqrt(np.maximum(mmd2, 0.0))
    return distances



def squared_mmd_rbf_two_sample(
    x: np.ndarray,
    y: np.ndarray,
    max_samples: int = 512,
    seed: int = 2025,
) -> float:
    """Biased two-sample RBF-MMD² with median-distance bandwidth."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]

    z = np.vstack([x, y])
    bandwidth2 = mmd_rbf_bandwidth2(z, seed=seed)
    kxx = _rbf_kernel_mean(x, x, bandwidth2)
    kyy = _rbf_kernel_mean(y, y, bandwidth2)
    kxy = _rbf_kernel_mean(x, y, bandwidth2)
    return float(kxx + kyy - 2.0 * kxy)


def flow_summary_samples(flow, num_samples: int, seed: int = 2025) -> np.ndarray:
    """Draw generated summaries from q_phi in the original summary space."""
    samples = flow.sample(num_samples=num_samples, seed=seed)
    return np.asarray(samples["inference_variables"], dtype=np.float64)


def c2st_accuracy(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.5,
    seed: int = 2025,
) -> float:
    """Classifier two-sample test accuracy; 0.5 means hard to distinguish."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    features = np.vstack([x, y])
    labels = np.concatenate([np.zeros(len(x), dtype=int), np.ones(len(y), dtype=int)])
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
    )
    classifier.fit(x_train, y_train)
    return float(accuracy_score(y_test, classifier.predict(x_test)))


def density_flow_validation(
    flow,
    heldout_summary: np.ndarray,
    num_flow_samples: int | None = None,
    max_mmd_samples: int = 512,
    seed: int = 2025,
) -> dict[str, float]:
    """Compare held-out simulator summaries to generated flow summaries."""
    heldout_summary = np.asarray(heldout_summary, dtype=np.float64)
    num_flow_samples = int(num_flow_samples or len(heldout_summary))
    generated_summary = flow_summary_samples(flow, num_flow_samples, seed=seed)
    mmd2 = squared_mmd_rbf_two_sample(
        heldout_summary,
        generated_summary,
        max_samples=max_mmd_samples,
        seed=seed,
    )
    return {
        "density_validation_n_simulator": int(len(heldout_summary)),
        "density_validation_n_flow": int(len(generated_summary)),
        "density_validation_mmd2": float(mmd2),
        "density_validation_mmd": float(np.sqrt(max(mmd2, 0.0))),
        "density_validation_c2st_accuracy": c2st_accuracy(
            heldout_summary,
            generated_summary,
            seed=seed,
        ),
    }



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
    """Fit q_phi(z) on summary-network outputs, following density_based.ipynb."""
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

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=max(1, epochs * dataset.num_batches),
        alpha=1e-6,
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=patience,
            start_from_epoch=start_from_epoch,
            restore_best_weights=True,
            verbose=1,
        )
    ]
    flow.compile(optimizer)
    history = flow.fit(dataset=dataset, epochs=epochs, callbacks=callbacks)
    return flow, history


def typicality_log_density(flow, summaries: np.ndarray) -> np.ndarray:
    """Evaluate log q_phi(z) with the same standardization used during flow inference."""
    summaries = np.asarray(summaries, dtype=np.float32)
    z_tensor = keras.ops.convert_to_tensor(summaries)
    z_std = flow.standardizer.maybe_standardize(
        z_tensor,
        key="inference_variables",
        stage="inference",
    )
    _, log_q = flow.inference_network(z_std, density=True, training=False)
    return np.asarray(keras.ops.convert_to_numpy(log_q), dtype=np.float64).reshape(-1)


def signed_typicality_from_summary(
    summaries: np.ndarray,
    flow,
    expected_log_density: float,
) -> np.ndarray:
    """T(z) = log q_phi(z) - E_ref[log q_phi(Z_ref)]."""
    return typicality_log_density(flow, summaries) - float(expected_log_density)


def typicality_regime(score: float, low: float, high: float) -> str:
    """Map signed typicality to a central 90% typical interval.

    The central interval [low, high] is the typical set. The lower 5% tail
    (low log density) is extrapolation / high surprise; the upper 5% tail
    (too high log density) is interpolation / low surprise.
    """
    if low <= score <= high:
        return "in_distribution"
    if score > high:
        return "low surprise"
    return "high surprise"


def bootstrap_interval(
    distances: np.ndarray,
    alpha: float = 0.1,
    n_boot: int = 1000,
    seed: int = 2025,
) -> dict[str, float]:
    distances = np.asarray(distances, dtype=np.float64)
    if n_boot <= 0:
        return {
            "median": float(np.median(distances)),
            "low": float(np.percentile(distances, 100 * alpha / 2)),
            "high": float(np.percentile(distances, 100 * (1 - alpha / 2))),
        }
    rng = np.random.default_rng(seed)
    boot = rng.choice(distances, size=(n_boot, len(distances)), replace=True)
    qs = np.percentile(boot, [50, 100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return {"median": float(qs[0].mean()), "low": float(qs[1].mean()), "high": float(qs[2].mean())}


def fit_reference(
    approximator,
    simulator,
    n_fit: int = 2000,
    n_calibration: int = 2000,
    alpha: float = 0.1,
    metric: str = "l2",
    n_boot: int = 1000,
    seed: int = 2025,
    density_epochs: int = 250,
    density_batch_size: int = 128,
    n_density_validation: int = 2000,
) -> dict:
    fit_data = simulator.sample(n_fit)
    fit_summary = summary_outputs(approximator, _sim_to_array(fit_data))

    if metric == "mmd":
        bandwidth2 = mmd_rbf_bandwidth2(fit_summary, seed=seed)
        reference_kernel_mean = _rbf_kernel_mean(fit_summary, fit_summary, bandwidth2)
        reference = {
            "reference_summary": np.asarray(fit_summary, dtype=np.float64),
            "summary_dim": int(fit_summary.shape[1]),
            "metric": metric,
            "alpha": alpha,
            "bandwidth2": float(bandwidth2),
            "reference_kernel_mean": float(reference_kernel_mean),
        }

        calibration = simulator.sample(n_calibration)
        calibration_summary = summary_outputs(approximator, _sim_to_array(calibration))
        distances = mmd_reference_distance_from_summary(
            calibration_summary,
            reference["reference_summary"],
            reference["bandwidth2"],
            reference["reference_kernel_mean"],
        )
        reference.update(bootstrap_interval(distances, alpha=alpha, n_boot=n_boot, seed=seed))
        return reference

    if metric == "typical":
        flow, history = fit_typicality_flow(
            fit_summary,
            epochs=density_epochs,
            batch_size=density_batch_size,
        )

        calibration = simulator.sample(n_calibration)
        calibration_summary = summary_outputs(approximator, _sim_to_array(calibration))
        calibration_log_q = typicality_log_density(flow, calibration_summary)
        expected_log_density = float(np.mean(calibration_log_q))
        calibration_typicality = calibration_log_q - expected_log_density
        typicality_low = float(np.percentile(calibration_typicality, 100 * alpha / 2))
        typicality_high = float(np.percentile(calibration_typicality, 100 * (1 - alpha / 2)))
        signed_reference_distance = -calibration_typicality

        validation = simulator.sample(n_density_validation)
        validation_summary = summary_outputs(approximator, _sim_to_array(validation))
        validation_metrics = density_flow_validation(
            flow,
            validation_summary,
            num_flow_samples=n_density_validation,
            seed=seed,
        )

        reference = {
            "flow": flow,
            "summary_dim": int(fit_summary.shape[1]),
            "metric": metric,
            "alpha": alpha,
            "expected_log_density": expected_log_density,
            "std_log_density": float(np.std(calibration_log_q)),
            "typicality_low": typicality_low,
            "typicality_high": typicality_high,
            "typicality_tau": float(max(abs(typicality_low), abs(typicality_high))),
            "median": float(np.median(signed_reference_distance)),
            # In signed surprise distance d=-T, high log density / interpolation
            # lies to the left and low log density / extrapolation lies to the right.
            "low": float(-typicality_high),
            "high": float(-typicality_low),
            "density_epochs": int(density_epochs),
            "density_batch_size": int(density_batch_size),
            "density_validation_samples": int(n_density_validation),
            "density_training_history": {
                key: [float(value) for value in values]
                for key, values in getattr(history, "history", {}).items()
            },
            **validation_metrics,
        }
        return reference

    if metric not in {"l2", "linf"}:
        raise ValueError("metric must be 'l2', 'linf', 'mmd', or 'typical'")

    covariance = LedoitWolf().fit(fit_summary)
    reference = {
        "mean": covariance.location_,
        "chol": np.linalg.cholesky(covariance.covariance_),
        "summary_dim": int(fit_summary.shape[1]),
        "metric": metric,
        "alpha": alpha,
    }

    calibration = simulator.sample(n_calibration)
    calibration_summary = summary_outputs(approximator, _sim_to_array(calibration))
    distances = summary_distance_from_summary(
        calibration_summary,
        reference["mean"],
        reference["chol"],
        metric=metric,
    )
    reference.update(bootstrap_interval(distances, alpha=alpha, n_boot=n_boot, seed=seed))
    return reference


def fit_references(
    approximators: dict[str, object],
    simulators: dict[str, object],
    **kwargs,
) -> dict[str, dict]:
    return {model: fit_reference(approximators[model], simulators[model], **kwargs) for model in MODELS}


def distance_regime(distance: float, reference: dict) -> str:
    if distance < reference["low"]:
        return "low surprise"
    if distance > reference["high"]:
        return "high surprise"
    return "in_distribution"


def add_summary_diagnostics(
    frame: pd.DataFrame,
    y,
    approximators: dict[str, object],
    references: dict[str, dict],
    eps: float = 1e-8,
) -> pd.DataFrame:
    y = data_array(y)
    output = frame.copy()
    distances = {}
    ranking_distances = {}
    for model in MODELS:
        summaries = summary_outputs(approximators[model], y)
        ref = references[model]
        if ref["metric"] == "mmd":
            distances[model] = mmd_reference_distance_from_summary(
                summaries,
                ref["reference_summary"],
                ref["bandwidth2"],
                ref.get("reference_kernel_mean"),
            )
            ranking_distances[model] = distances[model]
            regimes = [distance_regime(d, ref) for d in distances[model]]
        elif ref["metric"] == "typical":
            signed_typicality = signed_typicality_from_summary(
                summaries,
                ref["flow"],
                ref["expected_log_density"],
            )
            # Signed surprise distance. This keeps the two-tailed typical-set
            # diagnostic visible on the x-axis: low surprise is left of the
            # central interval and high surprise is right of it.
            distances[model] = -signed_typicality
            ranking_distances[model] = np.maximum(
                ref["low"] - distances[model],
                distances[model] - ref["high"],
            ).clip(min=0.0)
            output[f"typicality_{model}"] = signed_typicality
            output[f"typicality_low_{model}"] = ref["typicality_low"]
            output[f"typicality_high_{model}"] = ref["typicality_high"]
            output[f"typicality_tau_{model}"] = ref["typicality_tau"]
            output[f"typicality_expected_log_density_{model}"] = ref["expected_log_density"]
            regimes = [
                typicality_regime(score, ref["typicality_low"], ref["typicality_high"])
                for score in signed_typicality
            ]
        else:
            distances[model] = summary_distance_from_summary(summaries, ref["mean"], ref["chol"], metric=ref["metric"])
            ranking_distances[model] = distances[model]
            regimes = [distance_regime(d, ref) for d in distances[model]]
        output[f"d_{model}"] = distances[model]
        output[f"rho_{model}"] = distances[model] / ref["high"]
        output[f"dm_low_{model}"] = ref["low"]
        output[f"dm_high_{model}"] = ref["high"]
        output[f"regime_{model}"] = regimes

    distance_matrix = np.column_stack([ranking_distances[model] for model in MODELS])
    order = np.argsort(distance_matrix, axis=1)
    output["closest_summary_model"] = [MODELS[i] for i in order[:, 0]]
    output["d_min"] = distance_matrix[np.arange(len(output)), order[:, 0]]
    output["d_second"] = distance_matrix[np.arange(len(output)), order[:, 1]]
    output["summary_ambiguity_true"] = 1.0 / (np.abs(output["d_second"] - output["d_min"]) + eps)
    output["globally_high_surprise"] = output[[f"regime_{m}" for m in MODELS]].eq("high surprise").all(axis=1)
    output["at_least_one_not_high_surprise"] = ~output["globally_high_surprise"]
    output["summary_ambiguity"] = np.where(output["globally_high_surprise"], output["summary_ambiguity_true"], 0.0)
    output["log1p_summary_ambiguity"] = np.log1p(output["summary_ambiguity"])
    return output


def pmp_diagnostic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "model_title": MODEL_TITLES[model],
                    "distance": frame[f"d_{model}"],
                    "distance_low": frame[f"dm_low_{model}"],
                    "distance_high": frame[f"dm_high_{model}"],
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "p_npe": frame[f"pmp_{model}"],
                    "p_gold": frame[f"gold_pmp_{model}"],
                    "signed_pmp_error": frame[f"signed_pmp_error_{model}"],
                    "summary_ambiguity": frame["summary_ambiguity"],
                    "summary_ambiguity_true": frame["summary_ambiguity_true"],
                    "log1p_summary_ambiguity": np.log1p(frame["summary_ambiguity"]),
                    "globally_high_surprise": frame["globally_high_surprise"],
                    "at_least_one_not_high_surprise": frame["at_least_one_not_high_surprise"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def logml_diagnostic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "model_title": MODEL_TITLES[model],
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "signed_logml_error": frame[f"signed_logml_error_{model}"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def save_references(references: dict[str, dict], path: str | Path = REFERENCE_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(references, f)
    return path


def load_references(path: str | Path = REFERENCE_PATH) -> dict[str, dict]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_diagnostic(frame: pd.DataFrame, path: str | Path = DIAGNOSTIC_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_diagnostic(path: str | Path = DIAGNOSTIC_PATH) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def _sim_to_array(samples: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([samples["rt"], samples["conditions"]], axis=-1)
