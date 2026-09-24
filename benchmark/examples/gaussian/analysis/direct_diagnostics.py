"""Direct-model OOD diagnostics in each classifier's own learned summary space.

The indirect S=4D cache supplies observations and comparison probabilities only.
No indirect summary embeddings, reference distributions, or OOD labels are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from ..config import ASSUMED_MODELS, MODEL_SPECS, NETWORK_DIR, RESULT_DIR, SOURCE_MODELS
from .reference_datasets import ensure_reference_datasets, load_or_fit_model_reference


LOSSES = ("cross_entropy", "exponential", "logistic")
DIAGNOSTICS = ("l2", "linf", "density", "mmd")
DEFAULT_SETTINGS = {
    "n_fit": 2000,
    "n_calibration": 2000,
    "alpha": 0.1,
    "n_boot": 1000,
    "seed": 2025,
    "density_epochs": 250,
    "density_batch_size": 128,
    "n_density_validation": 2000,
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint_config(path: Path) -> tuple[int, int, int]:
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    if config["class_name"] != "ModelComparisonApproximator":
        raise ValueError(f"Expected a trained direct classifier: {path}")
    shape = config["build_config"]["input_shape"]["summary_variables"]
    summary_dim = config["config"]["summary_network"]["config"]["summary_dim"]
    return int(shape[1]), int(shape[2]), int(summary_dim)


def _load_observations(dataset_dir: Path, num_obs: int, data_dim: int):
    """Read only observations and scalar gold/indirect evidence into the new cache."""
    observations, rows = [], []
    for source in SOURCE_MODELS:
        with (dataset_dir / f"{source}_logml_pmp.pkl").open("rb") as stream:
            items = pickle.load(stream)
        for item in items:
            x = np.asarray(item["x"], dtype=np.float32)
            if x.shape != (num_obs, data_dim):
                raise ValueError(f"Unexpected observation shape {x.shape} for {source}")
            if item.get("source_model", source) != source:
                raise ValueError(f"Mismatched source label in {source}")
            observations.append(x)
            rows.append({
                "source_model": source,
                "id": int(item["id"]),
                **{f"gold_logml_{m}": item[f"gold_log_marginal_{m}"] for m in ASSUMED_MODELS},
                **{f"indirect_logml_{m}": item[f"npe_log_marginal_{m}"] for m in ASSUMED_MODELS},
            })
        del items
    frame = pd.DataFrame(rows)
    if frame.duplicated(["source_model", "id"]).any():
        raise ValueError("Observation identifiers must be unique")
    return np.stack(observations), frame


class _GaussianSimulator:
    """Independent prior-predictive draws for one candidate model."""

    def __init__(self, spec: dict, num_obs: int, data_dim: int, seed: int):
        self.spec, self.num_obs, self.data_dim = spec, num_obs, data_dim
        self.rng = np.random.default_rng(seed)

    def sample(self, count: int) -> dict:
        mu = self.rng.normal(
            self.spec["mu_prior_mean"], self.spec["mu_prior_std"],
            size=(count, 1, self.data_dim),
        )
        x = self.rng.normal(
            mu, self.spec["likelihood_std"],
            size=(count, self.num_obs, self.data_dim),
        )
        return {"x": x.astype(np.float32)}


def diagnostic_frame(index: pd.DataFrame, distances: dict, references: dict, loss: str):
    """Normalize each candidate's distances and classify using its own reference."""
    frames = []
    for metric in DIAGNOSTICS:
        all_high = np.logical_and.reduce([
            np.asarray(distances[model][metric]) > references[model][metric]["dm_high"]
            for model in ASSUMED_MODELS
        ])
        for model in ASSUMED_MODELS:
            reference = references[model][metric]
            distance = np.asarray(distances[model][metric], dtype=float)
            low, median, high = (float(reference[k]) for k in ("dm_low", "median", "dm_high"))
            if not np.isfinite([low, median, high]).all() or not low <= median < high:
                raise ValueError(f"Invalid diagnostic reference for {loss}/{model}/{metric}")
            if distance.shape != (len(index),) or not np.isfinite(distance).all():
                raise ValueError(f"Invalid distances for {loss}/{model}/{metric}")
            frame = index[["source_model", "id"]].copy()
            frame["loss_function"] = loss
            frame["diagnostic"] = metric
            frame["assumed_model"] = model
            frame["summary_owner"] = f"direct_{loss}"
            frame["summary_dim"] = reference["summary_dim"]
            frame["d_M"] = distance
            frame["dm_low"], frame["dm_median"], frame["dm_high"] = low, median, high
            frame["rho"] = (distance - median) / (high - median)
            frame["rho_low"] = (low - median) / (high - median)
            frame["distance_regime"] = np.where(distance > high, "extrapolation", np.where(distance < low, "interpolation", "in_distribution"))
            frame["globally_extrapolative"] = all_high
            frame["at_least_one_not_extrapolative"] = ~all_high
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _comparison_frame(index, approximator, observations):
    predictions = approximator.estimate(conditions={"x": observations})
    direct = np.asarray(predictions["model_probs"], dtype=float)
    gold = softmax(index[[f"gold_logml_{m}" for m in ASSUMED_MODELS]].to_numpy(), axis=1)
    indirect = softmax(index[[f"indirect_logml_{m}" for m in ASSUMED_MODELS]].to_numpy(), axis=1)
    comparison = index[["source_model", "id"]].copy()
    for method, values in (("gold", gold), ("indirect", indirect), ("direct", direct)):
        if values.shape != (len(index), 4) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {method} probabilities")
        if np.any(values < 0) or np.any(values > 1) or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
            raise ValueError(f"Unnormalized {method} probabilities")
        for j, model in enumerate(ASSUMED_MODELS):
            comparison[f"p_{method}_{model}"] = values[:, j]
    comparison["l1_indirect"] = np.abs(indirect - gold).sum(axis=1)
    comparison["l1_direct"] = np.abs(direct - gold).sum(axis=1)
    return comparison


def ensure_direct_diagnostics(
    loss: str, *, results_dir: str | Path = RESULT_DIR / "direct",
    dataset_dir: str | Path = RESULT_DIR / "ood_80d_10n" / "datasets",
    checkpoint: str | Path | None = None, overwrite: bool = False,
    reference_root: str | Path | None = None,
    **settings,
) -> pd.DataFrame:
    """Reuse valid direct caches or rebuild from the saved direct checkpoint.

    A checkpoint/content/settings change invalidates the cache. Rebuilding trains
    density reference flows only; it never retrains the direct classifier.
    """
    if loss not in LOSSES:
        raise ValueError(f"Unknown direct loss: {loss}")
    unknown = set(settings) - DEFAULT_SETTINGS.keys()
    if unknown:
        raise ValueError(f"Unknown reference settings: {sorted(unknown)}")
    settings = {**DEFAULT_SETTINGS, **settings}
    checkpoint = Path(checkpoint or NETWORK_DIR / f"direct_{loss}.keras").resolve()
    dataset_dir = Path(dataset_dir).resolve()
    num_obs, data_dim, summary_dim = _checkpoint_config(checkpoint)
    shared_data = ensure_reference_datasets(
        data_dim, num_obs, n_fit=settings["n_fit"], n_calibration=settings["n_calibration"],
        n_density_validation=settings["n_density_validation"], seed=settings["seed"], root=reference_root,
    )
    output = Path(results_dir) / f"direct_{loss}"
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "diagnostic_frame.csv"
    comparison_path = output / "pmp_comparison.csv"
    metadata_path = diagnostics / "metadata.json"
    reference_identity = {
        "schema_version": 2,
        "summary_owner": f"direct_{loss}",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "summary_dim": summary_dim, "data_dim": data_dim, "num_obs": num_obs,
        "model_specs": {m: MODEL_SPECS[m] for m in ASSUMED_MODELS},
        "settings": settings,
        "shared_reference_datasets": shared_data.identity,
        "reference_dataset_implementation_sha256": file_sha256(Path(__file__).with_name("reference_datasets.py")),
        "implementation_sha256": file_sha256(Path(__file__)),
        "diagnostic_implementation_sha256": file_sha256(Path(__file__).with_name("summry_diagnostic.py")),
    }
    identity = {
        **reference_identity,
        "dataset_directory": str(dataset_dir),
        "input_sha256": {f"{source}_logml_pmp.pkl": file_sha256(dataset_dir / f"{source}_logml_pmp.pkl") for source in SOURCE_MODELS},
    }
    if not overwrite and metadata_path.exists() and csv_path.exists() and comparison_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("identity") == identity
                and metadata.get("diagnostic_sha256") == file_sha256(csv_path)
                and metadata.get("comparison_sha256") == file_sha256(comparison_path)):
            return pd.read_csv(csv_path)

    # BayesFlow registers the serialized networks and chooses its configured backend.
    import bayesflow  # noqa: F401
    import keras
    from . import summry_diagnostic as sd

    sd.quiet_bayesflow_progress()
    approximator = keras.saving.load_model(checkpoint, compile=False)
    observations, index = _load_observations(dataset_dir, num_obs, data_dim)
    comparison = _comparison_frame(index, approximator, observations)
    observed_summary = sd.summary_outputs(approximator, observations)
    if observed_summary.shape != (len(index), summary_dim):
        raise ValueError("Direct checkpoint summary shape does not match its metadata")
    references, distances = {}, {}
    for model_index, model in enumerate(ASSUMED_MODELS):
        ref_path = diagnostics / f"reference_{model}.pkl"
        references[model] = load_or_fit_model_reference(
            ref_path, approximator, shared_data, model, checkpoint=checkpoint,
            settings={**settings, "seed": settings["seed"] + model_index},
            metrics=DIAGNOSTICS, overwrite=overwrite, allow_legacy_direct=True,
        )
        distances[model] = {}
        for metric, reference in references[model].items():
            if metric == "mmd":
                distance = sd.mmd_reference_distance_from_summary(observed_summary, reference["reference_summary"], reference["bandwidth2"], reference["reference_kernel_mean"])
            elif metric == "density":
                distance = -sd.signed_typicality_from_summary(observed_summary, reference["flow"], reference["expected_log_density"])
            else:
                distance = sd.summary_distance_from_summary(observed_summary, reference["mu_hat"], reference["L_hat"], metric=metric)
            distances[model][metric] = distance
    frame = diagnostic_frame(index, distances, references, loss)
    _write_csv(csv_path, frame)
    _write_csv(comparison_path, comparison)
    np.savez_compressed(diagnostics / "observed_summaries.npz", summary=observed_summary,
                        source_model=index["source_model"].to_numpy(dtype=str), id=index["id"].to_numpy())
    created = datetime.now(timezone.utc).isoformat()
    _write_json(output / "pmp_comparison.metadata.json", {
        "created_at_utc": created, "num_datasets": len(index),
        "loss_function": loss, "direct_checkpoint": str(checkpoint),
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "direct_summary_dimension": summary_dim,
        "model_order": list(ASSUMED_MODELS), "model_prior": [0.25] * 4,
        "input_shape": list(observations.shape), "dataset_directory": str(dataset_dir),
        "input_files": list(identity["input_sha256"]),
        "direct_source": "estimate from the saved direct checkpoint",
        "indirect_source": "S=4D cached NPE logML; used only for PMP comparison",
        "gold_source": "cached analytical logML",
        "diagnostic_source": "this direct checkpoint's own learned summary embeddings",
    })
    _write_json(metadata_path, {
        "created_at_utc": created, "identity": identity,
        "diagnostic_sha256": file_sha256(csv_path), "comparison_sha256": file_sha256(comparison_path),
        "num_datasets": len(index), "diagnostic_rows": len(frame),
        "classification": "all high surprise iff all four candidate distances exceed their own upper reference quantile",
        "normalization": "rho=(distance-reference_median)/(reference_high-reference_median)",
        "density_validation": {m: {k: v for k, v in references[m]["density"].items() if k.startswith("density_validation_")} for m in ASSUMED_MODELS},
    })
    print(f"Saved {len(frame)} direct diagnostic rows: {csv_path}", flush=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=LOSSES, action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for loss in args.loss or LOSSES:
        ensure_direct_diagnostics(loss, overwrite=args.overwrite)
