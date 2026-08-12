"""Estimate NPE model-comparison quantities and load Stan gold standards."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax

from ..config import BASE_DIR, MODELS, N_ALPHA
from ..dataset import wagenmakers
from ..distributions import Likelihood, Prior


def normalized_model_priors(
    model_priors: dict[str, float] | None = None,
) -> np.ndarray:
    """Return validated model-prior probabilities in ``MODELS`` order."""
    if model_priors is None:
        return np.full(len(MODELS), 1.0 / len(MODELS), dtype=np.float64)
    if set(model_priors) != set(MODELS):
        raise ValueError(f"model_priors must define exactly {MODELS}")
    priors = np.asarray([model_priors[model] for model in MODELS], dtype=np.float64)
    if not np.isfinite(priors).all() or (priors <= 0.0).any():
        raise ValueError("Every model prior must be finite and strictly positive")
    return priors / priors.sum()


def pmp_from_log_marginals(
    log_marginals: np.ndarray,
    model_priors: dict[str, float] | None = None,
) -> np.ndarray:
    """Compute PMPs from log marginal likelihoods and explicit model priors."""
    log_marginals = np.asarray(log_marginals, dtype=np.float64)
    if log_marginals.shape[-1] != len(MODELS):
        raise ValueError(f"Expected {len(MODELS)} log marginal likelihoods")
    return softmax(
        log_marginals + np.log(normalized_model_priors(model_priors)), axis=-1
    )


def data_array(data=None) -> np.ndarray:
    if data is None:
        data = wagenmakers.df_array
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[None, ...]
    if data.ndim != 3 or data.shape[-1] != 2:
        raise ValueError("data must have shape (datasets, trials, rt+condition)")
    return data


def data_conditions(data=None) -> dict[str, np.ndarray]:
    data = data_array(data)
    return {"rt": data[..., 0], "conditions": data[..., 1]}


def parameter_array(samples: dict[str, np.ndarray]) -> np.ndarray:
    if "inference_variables" in samples:
        return np.asarray(samples["inference_variables"], dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(samples["alpha"], dtype=np.float32),
            np.asarray(samples["nu"], dtype=np.float32),
            np.asarray(samples["tau"], dtype=np.float32),
        ],
        axis=-1,
    )


def parameter_dict(theta: np.ndarray, model: str) -> dict[str, np.ndarray]:
    theta = np.asarray(theta, dtype=np.float32)
    n_alpha = N_ALPHA[model]
    return {
        "alpha": theta[..., :n_alpha],
        "nu": theta[..., n_alpha : n_alpha + 2],
        "tau": theta[..., n_alpha + 2 : n_alpha + 3],
    }


def log_prior(theta: np.ndarray, model: str) -> np.ndarray:
    value = Prior(model=model).log_prob(
        keras.ops.convert_to_tensor(theta), conditions=None
    )
    return np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64).reshape(-1)


def log_likelihood(theta: np.ndarray, y: np.ndarray, model: str) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y[None, ...]
    y_rep = np.repeat(y, repeats=theta.shape[0], axis=0)
    value = Likelihood(model=model).log_prob(
        samples=keras.ops.convert_to_tensor(y_rep),
        conditions=keras.ops.convert_to_tensor(theta),
    )
    return np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64).reshape(-1)


def posterior_draws(
    approximator,
    y: np.ndarray,
    num_samples: int = 1024,
    seed: int | None = 2025,
    batch_size: int | None = None,
) -> np.ndarray:
    samples = approximator.sample(
        num_samples=num_samples,
        conditions=data_conditions(y),
        sample_shape=(),
        batch_size=batch_size,
        seed=seed,
    )
    return parameter_array(samples)


def posterior_log_prob(
    approximator,
    theta: np.ndarray,
    y: np.ndarray,
    model: str,
    batch_size: int | None = None,
) -> np.ndarray:
    params = parameter_dict(theta, model)
    y = data_array(y)[0]
    params["rt"] = np.repeat(y[None, :, 0], repeats=theta.shape[0], axis=0)
    params["conditions"] = np.repeat(y[None, :, 1], repeats=theta.shape[0], axis=0)
    value = approximator.log_prob(params, batch_size=batch_size)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def estimate_log_marginal(
    approximator,
    y: np.ndarray,
    model: str,
    num_samples: int = 1024,
    seed: int | None = 2025,
    batch_size: int | None = None,
) -> pd.DataFrame:
    y = data_array(y)
    draws = posterior_draws(
        approximator,
        y,
        num_samples=num_samples,
        seed=seed,
        batch_size=batch_size,
    )
    rows = []
    for i in range(y.shape[0]):
        theta = draws[i]
        log_weights = (
            log_prior(theta, model)
            + log_likelihood(theta, y[i], model)
            - posterior_log_prob(
                approximator, theta, y[i], model, batch_size=batch_size
            )
        )
        normalized = log_weights - logsumexp(log_weights)
        rows.append(
            {
                "row": i,
                "model": model,
                "log_ml": float(logsumexp(log_weights) - np.log(theta.shape[0])),
                "importance_ess": float(1.0 / np.sum(np.exp(2.0 * normalized))),
                "num_importance_samples": int(theta.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def estimate_model_comparison(
    approximators: dict[str, object],
    y=None,
    ids: list[str] | None = None,
    dataset: str = "empirical",
    num_samples: int = 1024,
    seed: int = 2025,
    batch_size: int | None = None,
    model_priors: dict[str, float] | None = None,
) -> pd.DataFrame:
    y = data_array(y)
    if ids is None:
        ids = (
            wagenmakers.ids
            if dataset == "empirical" and y.shape[0] == len(wagenmakers.ids)
            else [f"s{i}" for i in range(y.shape[0])]
        )

    output = pd.DataFrame({"row": np.arange(y.shape[0]), "dataset": dataset, "id": ids})
    log_ml_columns = []
    for offset, model in enumerate(MODELS):
        estimates = estimate_log_marginal(
            approximators[model],
            y,
            model,
            num_samples=num_samples,
            seed=seed + offset,
            batch_size=batch_size,
        )
        output = output.merge(
            estimates.rename(
                columns={
                    "log_ml": f"log_ml_{model}",
                    "importance_ess": f"importance_ess_{model}",
                }
            )[["row", f"log_ml_{model}", f"importance_ess_{model}"]],
            on="row",
        )
        log_ml_columns.append(f"log_ml_{model}")

    pmp = pmp_from_log_marginals(
        output[log_ml_columns].to_numpy(dtype=np.float64), model_priors
    )
    for i, model in enumerate(MODELS):
        output[f"pmp_{model}"] = pmp[:, i]
    return output.drop(columns="row")


def load_gold_standard(
    path: str | Path | None = None,
    dataset: str = "empirical",
    model_priors: dict[str, float] | None = None,
) -> pd.DataFrame:
    if path is None:
        path = BASE_DIR / "stan" / "results_4_models" / dataset
    path = Path(path)

    frames = []
    for model in MODELS:
        frame = pd.read_csv(path / model / "bridgesampling.csv", keep_default_na=False)
        frame["dataset"] = dataset
        frame["model"] = model
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    wide = table.pivot_table(
        index=["dataset", "id"], columns="model", values="estimate", aggfunc="first"
    ).reset_index()
    wide = wide.rename(columns={model: f"gold_log_ml_{model}" for model in MODELS})
    gold_logml = wide[[f"gold_log_ml_{model}" for model in MODELS]].to_numpy(
        dtype=np.float64
    )
    gold_pmp = pmp_from_log_marginals(gold_logml, model_priors)
    for i, model in enumerate(MODELS):
        wide[f"gold_pmp_{model}"] = gold_pmp[:, i]
    return wide


def attach_gold_standard(
    results: pd.DataFrame, gold: pd.DataFrame | None = None
) -> pd.DataFrame:
    if gold is None:
        gold = load_gold_standard()
    output = results.merge(gold, on=["dataset", "id"], how="left")
    for model in MODELS:
        output[f"signed_logml_error_{model}"] = (
            output[f"log_ml_{model}"] - output[f"gold_log_ml_{model}"]
        )
        output[f"abs_logml_error_{model}"] = np.abs(
            output[f"signed_logml_error_{model}"]
        )
        output[f"signed_pmp_error_{model}"] = (
            output[f"pmp_{model}"] - output[f"gold_pmp_{model}"]
        )
        output[f"abs_pmp_error_{model}"] = np.abs(output[f"signed_pmp_error_{model}"])
    return output


def save_results(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_results(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)
