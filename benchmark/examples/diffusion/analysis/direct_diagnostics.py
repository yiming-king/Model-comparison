"""Diffusion direct diagnostics with a pooled reference in the shared summary.

Reuse the observations used by indirect inference. Metric calibration is handled
separately by direct_calibration. Only PMP comparison values come from an indirect cache; summary
embeddings, reference distributions, and diagnostic labels are recomputed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import BASE_DIR, MODELS, RESULT_DIR

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


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _write_csv(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _checkpoint_config(path, loss):
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    if config["class_name"] != "ModelComparisonApproximator":
        raise ValueError(f"Expected a trained direct classifier: {path}")
    score = config["config"]["inference_network"]["config"]["scoring_rules"][
        "scoring_rule"
    ]["class_name"]
    expected = {
        "cross_entropy": "CrossEntropyScore",
        "exponential": "ExponentialScore",
        "logistic": "LogisticScore",
    }
    if score != expected[loss]:
        raise ValueError(f"Checkpoint scoring rule {score} does not match {loss}")
    shape = config["build_config"]["input_shape"]["summary_variables"]
    dim = config["config"]["summary_network"]["config"]["summary_dim"]
    if int(shape[-1]) != 2:
        raise ValueError("Diffusion inputs must contain signed RT and condition")
    return int(shape[1]), int(dim)


def _validate_observations(observations, num_obs):
    values = np.asarray(observations, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (num_obs, 2):
        raise ValueError(f"Expected (datasets, {num_obs}, 2), got {values.shape}")
    if not np.isfinite(values).all() or not np.isin(values[..., 1], [0, 1]).all():
        raise ValueError("Invalid RT or condition observations")
    return values


def _load_observations(indirect_path, num_obs):
    from ..results.observed_datasets import load_observed_dataset

    index = pd.read_csv(indirect_path, dtype={"dataset": str, "id": str})
    keys = ["dataset", "id"]
    if index[keys].isna().any().any() or index.duplicated(keys).any() or index.empty:
        raise ValueError(
            "Indirect observations must have unique nonempty dataset/ID keys"
        )
    arrays = {}
    for dataset, rows in index.groupby("dataset", sort=False):
        y, ids = load_observed_dataset(dataset)
        if len(set(ids)) != len(ids) or set(ids) != set(rows["id"]):
            raise ValueError(f"Dataset IDs do not match indirect inference: {dataset}")
        arrays.update({(dataset, str(i)): a for i, a in zip(ids, y)})
    y = np.stack(
        [arrays[key] for key in index[keys].itertuples(index=False, name=None)]
    )
    return _validate_observations(y, num_obs), index


def _array_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class _PooledReferenceSimulator:
    """Replay equal draws from the four assumed models for one joint reference."""

    def __init__(self, manifest, counts, seed):
        self.manifest, self.counts, self.seed = manifest, counts, seed
        self.consumed_splits = []

    def sample(self, count):
        from ..results.shared_reference_data import PARTITIONS, load_reference_partition

        split = PARTITIONS[len(self.consumed_splits)]
        if count != self.counts[split]:
            raise ValueError(f"Expected {self.counts[split]} pooled {split} datasets")
        rng = np.random.default_rng(self.seed + len(self.consumed_splits))
        choices = np.tile(np.arange(len(MODELS)), count // len(MODELS))
        choices = np.concatenate(
            (choices, rng.permutation(len(MODELS))[: count % len(MODELS)])
        )
        rng.shuffle(choices)
        pooled = None
        for model_index, model in enumerate(MODELS):
            positions = np.flatnonzero(choices == model_index)
            values = load_reference_partition(self.manifest, model, split)[: len(positions)]
            if pooled is None:
                pooled = np.empty((count, *values.shape[1:]), dtype=values.dtype)
            pooled[positions] = values
        self.consumed_splits.append(split)
        return {"rt": pooled[..., 0], "conditions": pooled[..., 1]}


def _remove_legacy_references(directory):
    for suffix in ("pkl", "json"):
        for path in directory.glob(f"reference_m*.{suffix}"):
            path.unlink()


def diagnostic_frame(index, distances, references, loss):
    """One score per dataset and diagnostic in the joint direct summary space."""
    frames = []
    for metric in DIAGNOSTICS:
        ref = references[metric]
        distance = np.asarray(distances[metric], dtype=float)
        low, high = (float(ref[k]) for k in ("low", "high"))
        if not np.isfinite([low, high]).all() or not low < high or high <= 0:
            raise ValueError(f"Invalid pooled reference for {loss}/{metric}")
        if distance.shape != (len(index),) or not np.isfinite(distance).all():
            raise ValueError(f"Invalid diagnostic scores for {loss}/{metric}")
        frame = index[[c for c in ("dataset", "id", "generating_model") if c in index]].copy()
        frame["loss_function"] = loss
        frame["diagnostic"] = metric
        frame["summary_owner"] = f"direct_{loss}"
        frame["summary_dim"] = ref["summary_dim"]
        frame["distance"] = distance
        frame["reference_low"], frame["reference_high"] = low, high
        frame["rho"] = distance / high
        frame["rho_low"] = low / high
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _comparison_frame(index, approximator, observations, indirect_tag):
    from ..results.results import data_conditions

    direct = np.asarray(
        approximator.estimate(conditions=data_conditions(observations))["model_probs"],
        dtype=float,
    )
    gold = index[[f"gold_pmp_{m}" for m in MODELS]].to_numpy(float)
    indirect = index[[f"pmp_{m}" for m in MODELS]].to_numpy(float)
    frame = index[["dataset", "id"]].copy()
    for method, values in (("gold", gold), ("indirect", indirect), ("direct", direct)):
        if values.shape != (len(index), len(MODELS)) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {method} PMP values")
        if (
            np.any(values < 0)
            or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1, atol=1e-6)
        ):
            raise ValueError(f"Unnormalized {method} PMP values")
        for j, model in enumerate(MODELS):
            frame[f"p_{method}_{model}"] = values[:, j]
    frame["indirect_configuration"] = indirect_tag
    frame["l1_indirect"] = np.abs(indirect - gold).sum(axis=1)
    frame["l1_direct"] = np.abs(direct - gold).sum(axis=1)
    return frame


def _score_summaries(summary, references, sd):
    distances = {}
    for metric, ref in references.items():
        if metric == "mmd":
            distance = sd.mmd_reference_distance_from_summary(
                summary, ref["reference_summary"], ref["bandwidth2"], ref["reference_kernel_mean"]
            )
        elif metric == "density":
            distance = -sd.signed_typicality_from_summary(
                summary, ref["flow"], ref["expected_log_density"]
            )
        else:
            distance = sd.summary_distance_from_summary(
                summary, ref["mean"], ref["chol"], metric=metric
            )
        distances[metric] = distance
    return distances


def ensure_direct_diagnostics(
    loss,
    *,
    results_dir=RESULT_DIR / "direct",
    indirect_tag="S4D",
    checkpoint=None,
    shared_reference_root=None,
    overwrite=False,
    **settings,
):
    """Compute or reuse own-summary diagnostics without training the classifier.

    Reference fitting uses independent prior-predictive draws, never the scored
    metric-calibration or observed datasets. One density flow is fitted per classifier.
    """
    if loss not in LOSSES:
        raise ValueError(f"Unknown direct loss: {loss}")
    if set(settings) - DEFAULT_SETTINGS.keys():
        raise ValueError(
            f"Unknown reference settings: {set(settings) - DEFAULT_SETTINGS.keys()}"
        )
    settings = {**DEFAULT_SETTINGS, **settings}
    checkpoint = Path(
        checkpoint or BASE_DIR / "approximators" / "trained" / "direct"
        / f"direct_{'cent' if loss == 'cross_entropy' else loss}.keras"
    ).resolve()
    num_obs, summary_dim = _checkpoint_config(checkpoint, loss)
    indirect_path = (
        RESULT_DIR / "NPE_results" / f"npe_{indirect_tag}_all_observed_results_gold.csv"
    )
    observations, index = _load_observations(indirect_path, num_obs)
    from ..results.shared_reference_data import (
        ensure_shared_reference_data,
        reference_data_identity,
    )
    if min(settings[k] for k in ("n_fit", "n_calibration", "n_density_validation")) < 8:
        raise ValueError("Pooled reference partitions need at least eight datasets")
    shared_manifest = ensure_shared_reference_data(
        root=shared_reference_root,
        n_fit=math.ceil(settings["n_fit"] / len(MODELS)),
        n_calibration=math.ceil(settings["n_calibration"] / len(MODELS)),
        n_density_validation=math.ceil(settings["n_density_validation"] / len(MODELS)),
        seed=settings["seed"],
    )
    output = Path(results_dir) / f"direct_{loss}"
    diagnostics = output / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / "diagnostic_frame.csv"
    comparison_path = output / "pmp_comparison.csv"
    summary_path = diagnostics / "observed_summaries.npz"
    outputs = [csv_path, comparison_path, summary_path]
    metadata_path = diagnostics / "metadata.json"
    ref_path = diagnostics / "reference_pooled.pkl"
    reference_identity = {
        "schema_version": 3,
        "summary_owner": f"direct_{loss}",
        "shared_reference_data": reference_data_identity(shared_manifest),
        "pooling": "uniform balanced model indices; n_fit, n_calibration, n_density_validation are totals",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "summary_dim": summary_dim,
        "num_obs": num_obs,
        "model_order": list(MODELS),
        "settings": settings,
        "implementation_sha256": file_sha256(Path(__file__)),
        "reference_implementation_sha256": file_sha256(
            BASE_DIR / "results/summary_diagnostic.py"
        ),
        "simulator_sha256": file_sha256(BASE_DIR / "simulators/rdm_4.py"),
        "dataset_loader_sha256": file_sha256(BASE_DIR / "dataset/dataset.py"),
        "data_adapter_sha256": file_sha256(BASE_DIR / "results/results.py"),
        "observation_loader_sha256": file_sha256(
            BASE_DIR / "results/observed_datasets.py"
        ),
        "conditions_sha256": _array_sha256(observations[0, :, 1]),
    }
    identity = {
        **reference_identity,
        "indirect_configuration": indirect_tag,
        "indirect_path": str(indirect_path),
        "indirect_sha256": file_sha256(indirect_path),
        "observation_keys": index[["dataset", "id"]].values.tolist(),
        "observations_sha256": _array_sha256(observations),
    }
    if not overwrite and metadata_path.exists() and all(p.exists() for p in outputs):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") == identity and metadata.get("output_sha256") == {
            str(p.relative_to(output)): file_sha256(p) for p in outputs
        } and ref_path.exists() and metadata.get("pooled_reference_sha256") == file_sha256(ref_path):
            _remove_legacy_references(diagnostics)
            return pd.read_csv(csv_path, dtype={"dataset": str, "id": str})

    import bayesflow  # noqa: F401: register serialized direct networks
    import keras
    from ..results import summary_diagnostic as sd

    keras.utils.disable_interactive_logging()
    approximator = keras.saving.load_model(checkpoint, compile=False)
    comparison = _comparison_frame(index, approximator, observations, indirect_tag)
    # Save the completed PMP comparison even if reference fitting is interrupted.
    _write_csv(comparison_path, comparison)
    summary = sd.summary_outputs(approximator, observations)
    if summary.shape != (len(index), summary_dim) or not np.isfinite(summary).all():
        raise ValueError("Direct checkpoint returned invalid summary embeddings")
    ref_metadata_path = diagnostics / "reference_pooled.json"
    references = None
    if ref_path.exists() and ref_metadata_path.exists() and not overwrite:
        ref_metadata = json.loads(ref_metadata_path.read_text())
        if ref_metadata.get("identity") == reference_identity and ref_metadata.get(
            "sha256"
        ) == file_sha256(ref_path):
            references = sd.load_references(ref_path)
    if references is None:
        references = sd.fit_reference_suite(
            approximator,
            _PooledReferenceSimulator(
                shared_manifest,
                {"fit": settings["n_fit"], "calibration": settings["n_calibration"],
                 "validation": settings["n_density_validation"]},
                settings["seed"],
            ),
            metrics=DIAGNOSTICS,
            **settings,
        )
        for reference in references.values():
            reference.pop("median", None)
        sd.save_references(references, ref_path)
        _write_json(ref_metadata_path, {
            "identity": reference_identity,
            "sha256": file_sha256(ref_path),
        })
    frame = diagnostic_frame(
        index, _score_summaries(summary, references, sd), references, loss
    )
    _write_csv(csv_path, frame)
    np.savez_compressed(
        summary_path,
        summary=summary,
        dataset=index["dataset"].to_numpy(str),
        id=index["id"].to_numpy(str),
    )
    created = datetime.now(timezone.utc).isoformat()
    _write_json(
        output / "pmp_comparison.metadata.json",
        {
            "created_at_utc": created,
            "loss_function": loss,
            "num_datasets": len(index),
            "direct_checkpoint": str(checkpoint),
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "direct_summary_dimension": summary_dim,
            "model_order": list(MODELS),
            "model_prior": [0.25] * 4,
            "indirect_configuration": indirect_tag,
            "indirect_source": str(indirect_path),
            "gold_source": "shared Stan bridge-sampling PMP in the indirect observed cache",
            "diagnostic_source": "this direct checkpoint's own learned summaries",
        },
    )
    _write_json(
        metadata_path,
        {
            "created_at_utc": created,
            "identity": identity,
            "output_sha256": {
                str(p.relative_to(output)): file_sha256(p) for p in outputs
            },
            "num_datasets": len(index),
            "diagnostic_rows": len(frame),
            "normalization": "rho=distance/reference_high",
            "reference_data": "persisted raw fit/calibration/validation partitions shared with all indirect networks; separate from observed and metric-calibration data",
            "pooled_reference_sha256": file_sha256(ref_path),
            "density_validation": {
                k: v for k, v in references["density"].items()
                if k.startswith("density_validation_")
            },
        },
    )
    _remove_legacy_references(diagnostics)
    print(f"Saved {len(frame)} observed diagnostic rows for {loss}", flush=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=LOSSES, action="append")
    parser.add_argument("--indirect-tag", default="S4D")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for loss in args.loss or LOSSES:
        ensure_direct_diagnostics(
            loss, indirect_tag=args.indirect_tag, overwrite=args.overwrite
        )
