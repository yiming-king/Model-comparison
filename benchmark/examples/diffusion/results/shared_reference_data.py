"""Persisted raw diagnostic reference draws shared by every RDM summary network.

This bank is separate from the observed data and the PMP error benchmark. Its
fit/calibration/validation partitions are generated before any network is used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..config import BASE_DIR, MODELS

DEFAULT_ROOT = BASE_DIR / "reference_datasets" / "diagnostics"
PARTITIONS = ("fit", "calibration", "validation")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def array_sha256(values):
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(json.dumps(list(values.shape)).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def ensure_shared_reference_data(*, root=None, n_fit=2000, n_calibration=2000,
                                 n_density_validation=2000, seed=2025):
    """Return a verified content-addressed manifest; never silently replace data."""
    from ..dataset import wagenmakers
    from ..simulators.rdm_4 import RDM

    counts = {"fit": int(n_fit), "calibration": int(n_calibration),
              "validation": int(n_density_validation)}
    if any(count < 2 for count in counts.values()):
        raise ValueError("Each reference partition needs at least two datasets")
    bank = Path(root or DEFAULT_ROOT) / (
        f"seed{seed}_fit{n_fit}_cal{n_calibration}_validation{n_density_validation}"
    )
    generation = {
        "schema_version": 1, "seed": int(seed), "counts": counts,
        "models": list(MODELS), "num_obs": len(wagenmakers.conditions),
        "simulator_sha256": file_sha256(BASE_DIR / "simulators/rdm_4.py"),
        "conditions_sha256": array_sha256(np.asarray(wagenmakers.conditions)),
        "protocol": "per-model MT19937 seed+model_index; fit then calibration; validation reseeded at seed+1000000+model_index",
        "dtype": "float32",
    }
    manifest_path = bank / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("generation") != generation:
            raise ValueError(f"Shared reference generation changed: {manifest_path}; use a new root/seed")
        for model in MODELS:
            for part in PARTITIONS:
                entry = manifest["files"][model][part]
                path = bank / entry["path"]
                if file_sha256(path) != entry["file_sha256"]:
                    raise ValueError(f"Shared reference file was modified: {path}")
                values = load_reference_partition(manifest, model, part, bank=bank)
                if values.shape != (counts[part], generation["num_obs"], 2):
                    raise ValueError(f"Invalid reference shape: {path}")
        return {**manifest, "bank_directory": str(bank.resolve())}

    bank.mkdir(parents=True, exist_ok=True)
    files = {}
    state = np.random.get_state()
    try:
        for number, model in enumerate(MODELS):
            simulator = RDM(model=model, keep_params=False)
            np.random.seed(int(seed) + number)
            files[model] = {}
            for part in PARTITIONS:
                if part == "validation":
                    np.random.seed(int(seed) + 1_000_000 + number)
                data = simulator.sample(counts[part])
                values = np.stack([data["rt"], data["conditions"]], axis=-1).astype(np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite shared reference draw: {model}/{part}")
                path = bank / model / f"{part}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".npz.tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(stream, observations=values)
                temporary.replace(path)
                files[model][part] = {
                    "path": str(path.relative_to(bank)), "shape": list(values.shape),
                    "file_sha256": file_sha256(path), "array_sha256": array_sha256(values),
                }
    finally:
        np.random.set_state(state)
    manifest = {"generation": generation, "files": files}
    write_json(manifest_path, manifest)
    return {**manifest, "bank_directory": str(bank.resolve())}


def load_reference_partition(manifest, model, part, *, bank=None):
    path = Path(bank or manifest["bank_directory"]) / manifest["files"][model][part]["path"]
    with np.load(path, allow_pickle=False) as archive:
        values = np.asarray(archive["observations"], dtype=np.float32)
    if array_sha256(values) != manifest["files"][model][part]["array_sha256"]:
        raise ValueError(f"Shared reference array was modified: {path}")
    return values


def reference_data_identity(manifest, model=None):
    models = (model,) if model else MODELS
    return {
        "bank_directory": manifest["bank_directory"],
        "generation": manifest["generation"],
        "arrays": {m: {part: manifest["files"][m][part]["array_sha256"]
                       for part in PARTITIONS} for m in models},
    }


class SharedReferenceSimulator:
    """Replay fixed raw partitions in the order fit_reference_suite requests."""
    def __init__(self, manifest, model):
        self.manifest, self.model, self.position = manifest, model, 0

    def sample(self, count):
        if self.position >= len(PARTITIONS):
            raise ValueError("Reference suite requested more than three raw partitions")
        part = PARTITIONS[self.position]
        values = load_reference_partition(self.manifest, self.model, part)
        if len(values) != count:
            raise ValueError(f"{part} reference count is {len(values)}, requested {count}")
        self.position += 1
        return {"rt": values[..., 0], "conditions": values[..., 1]}
