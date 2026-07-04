from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BASE_DIR, MODELS, N_ALPHA
from .observed_datasets import load_stan_posterior_draws
from .results import posterior_draws


POSTERIOR_DIAGNOSTIC_DIR = BASE_DIR / "results" / "posterior_diagnostics"


def parameter_columns(model: str) -> list[str]:
    return [f"alpha_{i}" for i in range(N_ALPHA[model])] + ["nu_0", "nu_1", "tau"]


def stan_draw_array(dataset: str, model: str, dataset_id: str) -> np.ndarray:
    frame = load_stan_posterior_draws(dataset, model, dataset_id)
    return frame[parameter_columns(model)].to_numpy(dtype=np.float64)


def squared_mmd_rbf(x: np.ndarray, y: np.ndarray, max_samples: int = 512, seed: int = 2025) -> float:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), size=max_samples, replace=False)]
    if len(y) > max_samples:
        y = y[rng.choice(len(y), size=max_samples, replace=False)]

    z = np.vstack([x, y])
    diff = z[:, None, :] - z[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    positive = dist2[dist2 > 0.0]
    bandwidth2 = np.median(positive) if len(positive) else 1.0
    bandwidth2 = max(float(bandwidth2), 1e-8)

    kxx = np.exp(-np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1) / (2.0 * bandwidth2))
    kyy = np.exp(-np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1) / (2.0 * bandwidth2))
    kxy = np.exp(-np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1) / (2.0 * bandwidth2))
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def posterior_diagnostic_frame(
    approximators: dict[str, object],
    y: np.ndarray,
    ids: list[str],
    dataset: str,
    num_samples: int = 2048,
    mmd_samples: int = 512,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    rows = []
    for offset, model in enumerate(MODELS):
        npe_draws = posterior_draws(
            approximators[model],
            y,
            num_samples=num_samples,
            seed=seed + offset,
            batch_size=batch_size,
        )
        for i, dataset_id in enumerate(ids):
            stan = stan_draw_array(dataset, model, dataset_id)
            npe = npe_draws[i]
            rows.append(
                {
                    "dataset": dataset,
                    "id": dataset_id,
                    "model": model,
                    "posterior_mmd": squared_mmd_rbf(npe, stan, max_samples=mmd_samples, seed=seed + i + offset),
                    "posterior_mean_rmse": float(np.sqrt(np.mean((npe.mean(axis=0) - stan.mean(axis=0)) ** 2))),
                    "num_npe_draws": int(npe.shape[0]),
                    "num_stan_draws": int(stan.shape[0]),
                    "num_mmd_draws": int(min(mmd_samples, npe.shape[0], stan.shape[0])),
                }
            )
    return pd.DataFrame(rows)


def save_posterior_diagnostic(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def load_posterior_diagnostic(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)
