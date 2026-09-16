"""Shared test observations for the paired Gaussian PMP experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import pickle
import platform
from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, MODEL_SPECS, SOURCE_MODELS
from ..datasets.datasets import GetDatasets

MODEL_ORDER = list(ASSUMED_MODELS)
MODEL_PRIOR = [0.25] * 4
PREPROCESSING = {"input_dtype": "float32", "transform": "cast_only", "shape": "batch,N,D"}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value):
    """Hash dtype, shape, and observation bytes in C order."""
    value = np.ascontiguousarray(value)
    header = json.dumps([value.dtype.str, list(value.shape)], separators=(",", ":")).encode()
    return hashlib.sha256(header + b"\0" + value.tobytes()).hexdigest()


def versions():
    result = {"python": platform.python_version()}
    for name in ("bayesflow", "keras", "tensorflow", "numpy", "scipy", "pandas", "matplotlib", "nbformat", "nbclient"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def write_json(path, value):
    """Create a new JSON artifact; never replace an existing result."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_probabilities(probabilities, count):
    p = np.asarray(probabilities)
    if p.shape != (count, 4) or not np.all(np.isfinite(p)):
        raise ValueError(f"PMP must be finite with shape ({count}, 4), got {p.shape}")
    if np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("PMP must be nonnegative and each row must sum to one")
    return p


def record_key(record):
    return record["source_model"], record["dataset_id"]


def validate_records(records, expected_records):
    """Match datasets by source/id and reject missing, repeated, or changed data."""
    found = {record_key(record): i for i, record in enumerate(records)}
    expected = {record_key(record) for record in expected_records}
    if len(found) != len(records) or len(expected) != len(expected_records):
        raise ValueError("Duplicate dataset keys")
    if found.keys() != expected:
        raise ValueError("Missing or unexpected dataset keys")
    order = [found[record_key(record)] for record in expected_records]
    for i, record in zip(order, expected_records):
        if records[i]["observation_sha256"] != record["observation_sha256"]:
            raise ValueError(f"Different observations for {record_key(record)}")
    return order


def prepare_bank(path, *, data_dim=20, num_obs=10, num_datasets=50, seed=73001,
                 sources=SOURCE_MODELS, legacy_raw_dir=None):
    """Import trusted local raw pickles or generate once, then seal a new bank."""
    sources = tuple(sources)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Bank already exists and will not be changed: {path}")
    records, observations, origin = [], [], {}
    for source in sources:
        if legacy_raw_dir is not None:
            raw_path = Path(legacy_raw_dir) / f"{source}_raw.pkl"
            # Only explicitly supplied, trusted local project pickles are imported.
            with raw_path.open("rb") as stream:
                items = pickle.load(stream)
            if len(items) < num_datasets:
                raise ValueError(f"Too few datasets in {raw_path}")
            items = sorted(items, key=lambda item: item["id"])[:num_datasets]
            origin[source] = str(raw_path.resolve())
        else:
            spec = MODEL_SPECS[source]
            items = GetDatasets(
                obs_mu_prior_mean=spec["mu_prior_mean"], obs_mu_prior_std=spec["mu_prior_std"],
                num_dims=data_dim, num_obs=num_obs, obs_likelihood_std=spec["likelihood_std"],
                num_datasets=num_datasets,
                rng=np.random.default_rng(np.random.SeedSequence([seed, SOURCE_MODELS.index(source)])),
            ).get_datasets_normal()
        for item in items:
            if item.get("source_model", source) != source:
                raise ValueError(f"Legacy source_model disagrees with filename: {source}")
            raw = np.asarray(item["x"])
            if raw.dtype != np.float64 or raw.shape != (num_obs, data_dim) or not np.all(np.isfinite(raw)):
                raise ValueError(f"Invalid raw observation in {source}/{item['id']}: {raw.shape}, {raw.dtype}")
            canonical = np.ascontiguousarray(raw, dtype=np.float32)
            dataset_id = int(item["id"])
            observations.append(raw)
            records.append({"source_model": source, "dataset_id": dataset_id,
                            "raw_shape": list(raw.shape), "raw_dtype": raw.dtype.str,
                            "observation_sha256": array_sha256(raw), "input_sha256": array_sha256(canonical)})
    validate_records(records, records)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "observations.npz", x=np.stack(observations))
    metadata = {"schema": 1, "data_dim": data_dim, "num_obs": num_obs,
                "model_order": MODEL_ORDER, "model_prior": MODEL_PRIOR, "model_specs": MODEL_SPECS,
                "preprocessing": PREPROCESSING, "sources": list(sources), "num_datasets_per_source": num_datasets,
                "seed": seed if legacy_raw_dir is None else None, "origin": origin,
                "versions": versions(), "records": records,
                "data_sha256": file_sha256(path / "observations.npz")}
    metadata["bank_id"] = metadata["data_sha256"]
    write_json(path / "manifest.json", metadata)
    for artifact in path.iterdir():
        artifact.chmod(0o444)
    path.chmod(0o555)
    return metadata


def load_bank(path):
    """Load the bank once and cast the same observations for all three methods."""
    path = Path(path)
    metadata = json.loads((path / "manifest.json").read_text())
    records = metadata["records"]
    validate_records(records, records)
    with np.load(path / "observations.npz", allow_pickle=False) as data:
        raw = data["x"]
    if raw.shape != (len(records), metadata["num_obs"], metadata["data_dim"]):
        raise ValueError("Test observations do not match the recorded N/D/count")
    for value, record in zip(raw, records):
        if array_sha256(value) != record["observation_sha256"]:
            raise ValueError(f"Different observations for {record_key(record)}")
    x = np.ascontiguousarray(raw, dtype=np.float32)
    x.setflags(write=False)
    return x, records, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--legacy-raw-dir", type=Path, help="Import existing trusted *_raw.pkl observations; never inference caches")
    parser.add_argument("--data-dim", type=int, default=20, help="Raw observation dimension D")
    parser.add_argument("--num-obs", type=int, default=10, help="Observations per dataset N")
    parser.add_argument("--num-datasets", type=int, default=50, help="Datasets per source")
    parser.add_argument("--sources", nargs="+", choices=SOURCE_MODELS, default=SOURCE_MODELS)
    parser.add_argument("--seed", type=int, default=73001)
    args = parser.parse_args()
    metadata = prepare_bank(args.output_dir, data_dim=args.data_dim, num_obs=args.num_obs,
                            num_datasets=args.num_datasets, sources=args.sources, seed=args.seed,
                            legacy_raw_dir=args.legacy_raw_dir)
    print(f"Sealed {len(metadata['records'])} observations: {args.output_dir}; bank_id={metadata['bank_id']}")


if __name__ == "__main__":
    main()
