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


def bootstrap_interval(
    distances: np.ndarray,
    alpha: float = 0.05,
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
    alpha: float = 0.05,
    metric: str = "l2",
    n_boot: int = 1000,
    seed: int = 2025,
) -> dict:
    fit_data = simulator.sample(n_fit)
    fit_summary = summary_outputs(approximator, _sim_to_array(fit_data))
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
    for model in MODELS:
        summaries = summary_outputs(approximators[model], y)
        ref = references[model]
        distances[model] = summary_distance_from_summary(summaries, ref["mean"], ref["chol"], metric=ref["metric"])
        output[f"d_{model}"] = distances[model]
        output[f"rho_{model}"] = distances[model] / ref["high"]
        output[f"dm_low_{model}"] = ref["low"]
        output[f"dm_high_{model}"] = ref["high"]
        output[f"regime_{model}"] = [distance_regime(d, ref) for d in distances[model]]

    distance_matrix = np.column_stack([distances[model] for model in MODELS])
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
