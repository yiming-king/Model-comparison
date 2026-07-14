from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BASE_DIR
from ..dataset import wagenmakers


OBSERVED_DATASETS = (
    "empirical",
    "simulated_from_m0",
    "simulated_from_m1",
    "simulated_from_m2",
    "simulated_from_m3",
    "m3_fast_30",
    "m3_slow_30",
    "m3_fast_slow_30",
)


def _id_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    return path.stem.rstrip("0123456789"), int(match.group(1)) if match else -1


def dataset_json_dir(dataset: str) -> Path:
    return BASE_DIR / "dataset" / "json" / dataset


def load_observed_dataset(dataset: str) -> tuple[np.ndarray, list[str]]:
    if dataset == "empirical":
        return wagenmakers.df_array, wagenmakers.ids

    folder = dataset_json_dir(dataset)
    files = sorted(folder.glob("*.json"), key=_id_key)
    arrays = []
    ids = []
    for path in files:
        with path.open("r") as f:
            record = json.load(f)
        arrays.append(np.stack([record["rt"], record["condition"]], axis=-1))
        ids.append(path.stem)
    return np.asarray(arrays, dtype=np.float32), ids


def load_true_parameters(dataset: str) -> pd.DataFrame | None:
    path = dataset_json_dir(dataset) / "true_parameters.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, keep_default_na=False)


def posterior_draw_path(dataset: str, model: str, dataset_id: str) -> Path:
    return BASE_DIR / "stan" / "results_4_models" / dataset / model / "posterior_draws" / f"{dataset_id}.csv"


def load_stan_posterior_draws(dataset: str, model: str, dataset_id: str) -> pd.DataFrame:
    path = posterior_draw_path(dataset, model, dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Stan posterior draws not found: {path}")
    return pd.read_csv(path)
