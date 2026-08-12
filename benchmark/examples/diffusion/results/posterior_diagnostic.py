"""Direct NPE-versus-MCMC posterior diagnostics.

Threshold calibration lives in ``diffusion/calibration``.  This module only
computes raw posterior discrepancies and contains the shared RBF-MMD helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import MODELS, N_ALPHA
from .observed_datasets import load_stan_posterior_draws
from .results import posterior_draws


DEFAULT_MMD_DRAWS = 1024


def parameter_columns(model: str) -> list[str]:
    return [f"alpha_{i}" for i in range(N_ALPHA[model])] + ["nu_0", "nu_1", "tau"]


def stan_draw_array(dataset: str, model: str, dataset_id: str) -> np.ndarray:
    frame = load_stan_posterior_draws(dataset, model, dataset_id)
    return frame[parameter_columns(model)].to_numpy(dtype=np.float64)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    distances = (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(y * y, axis=1)[None, :]
        - 2.0 * (x @ y.T)
    )
    return np.maximum(distances, 0.0)


def rbf_bandwidth2(samples: np.ndarray) -> float:
    """Median positive squared distance over the supplied posterior draws."""
    distances = squared_distances(samples, samples)
    np.fill_diagonal(distances, 0.0)
    positive = distances[distances > 0.0]
    return max(float(np.median(positive)) if len(positive) else 1.0, 1e-8)


def rbf_kernel(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth2: float,
) -> np.ndarray:
    bandwidth2 = max(float(bandwidth2), 1e-8)
    return np.exp(-squared_distances(x, y) / (2.0 * bandwidth2))


def squared_mmd_rbf(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth2: float,
) -> float:
    """Biased squared RBF-MMD with a caller-supplied fixed bandwidth."""
    kxx = rbf_kernel(x, x, bandwidth2)
    kyy = rbf_kernel(y, y, bandwidth2)
    kxy = rbf_kernel(x, y, bandwidth2)
    return max(float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()), 0.0)


def posterior_diagnostic_frame(
    approximators: dict[str, object],
    y: np.ndarray,
    ids: list[str],
    dataset: str,
    num_samples: int = DEFAULT_MMD_DRAWS,
    mmd_samples: int = DEFAULT_MMD_DRAWS,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compute raw NPE-versus-MCMC posterior MMD for each dataset and model."""
    rows = []
    for model_offset, model in enumerate(MODELS):
        npe_draws = posterior_draws(
            approximators[model],
            y,
            num_samples=num_samples,
            seed=seed + model_offset,
            batch_size=batch_size,
        )
        for dataset_offset, dataset_id in enumerate(ids):
            stan = stan_draw_array(dataset, model, dataset_id)
            npe = np.asarray(npe_draws[dataset_offset], dtype=np.float64)
            if len(npe) < mmd_samples or len(stan) < mmd_samples:
                raise ValueError(
                    f"Need at least {mmd_samples} NPE and MCMC draws for "
                    f"{dataset}/{dataset_id}/{model}; received {len(npe)} and {len(stan)}"
                )

            rng = np.random.default_rng(
                seed + 100_000 + 1_000 * model_offset + dataset_offset
            )
            stan_eval = stan[rng.choice(len(stan), size=mmd_samples, replace=False)]
            npe_eval = npe[rng.choice(len(npe), size=mmd_samples, replace=False)]
            bandwidth2 = rbf_bandwidth2(stan)
            observed_mmd2 = squared_mmd_rbf(stan_eval, npe_eval, bandwidth2)
            rows.append(
                {
                    "dataset": dataset,
                    "id": dataset_id,
                    "model": model,
                    "observed_mmd2": observed_mmd2,
                    "observed_mmd": float(np.sqrt(observed_mmd2)),
                    "rbf_bandwidth2": bandwidth2,
                    "posterior_mean_rmse": float(
                        np.sqrt(np.mean((npe.mean(axis=0) - stan.mean(axis=0)) ** 2))
                    ),
                    "num_npe_draws": int(len(npe)),
                    "num_stan_draws": int(len(stan)),
                    "num_mmd_draws": int(mmd_samples),
                    "observed_seed": int(seed),
                }
            )
    return pd.DataFrame(rows)


def save_posterior_diagnostic(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_posterior_diagnostic(path: str | Path) -> pd.DataFrame:
    """Load a cache, including pre-calibration raw-MMD column names."""
    frame = pd.read_csv(path, keep_default_na=False)
    if "observed_mmd2" not in frame:
        if "posterior_mmd2" in frame:
            frame["observed_mmd2"] = frame["posterior_mmd2"]
        elif "posterior_mmd" in frame:
            frame["observed_mmd2"] = frame["posterior_mmd"]
    if "observed_mmd" not in frame and "observed_mmd2" in frame:
        frame["observed_mmd"] = np.sqrt(
            np.maximum(frame["observed_mmd2"].to_numpy(dtype=np.float64), 0.0)
        )
    return frame
