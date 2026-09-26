"""Persist the exact Gaussian diagnostic reference observations shared by all nets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import ASSUMED_MODELS, MODEL_SPECS, RESULT_DIR, configuration_tag


SPLITS = ("fit", "calibration", "validation")
DEFAULT_ROOT = RESULT_DIR / "diagnostic_reference_datasets"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


class ReferenceSimulator:
    """Replay whole saved splits; never resample, shuffle, or wrap around."""

    def __init__(self, bank: "ReferenceDatasets", model: str):
        self.bank, self.model = bank, model
        self.spec = bank.metadata["configuration"]["model_specs"][model]
        self.consumed_splits = []

    def sample(self, count: int) -> dict:
        if len(self.consumed_splits) >= len(SPLITS):
            raise ValueError("All shared reference splits have already been consumed")
        split = SPLITS[len(self.consumed_splits)]
        values = self.bank.load(self.model, split)
        if len(values) != count:
            raise ValueError(
                f"Shared reference {split} contains {len(values)} observations, requested {count}"
            )
        self.consumed_splits.append(split)
        return {"x": values}


@dataclass(frozen=True)
class ReferenceDatasets:
    root: Path
    metadata: dict

    @property
    def identity(self) -> dict:
        return {"directory": str(self.root.resolve()), **self.metadata}

    def load(self, model: str, split: str) -> np.ndarray:
        if model not in ASSUMED_MODELS or split not in SPLITS:
            raise ValueError(f"Unknown shared reference model/split: {model}/{split}")
        relative = f"{model}/{split}.npy"
        path = self.root / relative
        if file_sha256(path) != self.metadata["file_sha256"][relative]:
            raise ValueError(f"Shared reference content changed: {path}")
        return np.load(path, allow_pickle=False)

    def simulator(self, model: str) -> ReferenceSimulator:
        return ReferenceSimulator(self, model)


def ensure_reference_datasets(
    data_dim: int = 20,
    num_obs: int = 10,
    *,
    n_fit: int = 2000,
    n_calibration: int = 2000,
    n_density_validation: int = 2000,
    seed: int = 2025,
    root: str | Path | None = None,
) -> ReferenceDatasets:
    """Create once or verify a content-addressed description of twelve raw batches.

    Generation matches the historical direct simulator: each model uses one RNG
    seeded by ``seed + 10000 * model_index`` and consumes whole fit, calibration,
    then validation batches, drawing all means before each batch's observations.
    """
    counts = dict(
        zip(SPLITS, (n_fit, n_calibration, n_density_validation), strict=True)
    )
    if data_dim < 1 or num_obs < 1 or min(counts.values()) < 2:
        raise ValueError(
            "Reference dimensions must be positive and each split requires at least two datasets"
        )
    configuration = {
        "generator": "gaussian_vectorized_sequential_batches_v1",
        "data_dim": int(data_dim),
        "num_obs": int(num_obs),
        "seed": int(seed),
        "counts": counts,
        "dtype": "float32",
        "model_specs": {model: MODEL_SPECS[model] for model in ASSUMED_MODELS},
    }
    folder = (
        Path(root)
        if root is not None
        else (
            DEFAULT_ROOT
            / configuration_tag(data_dim, num_obs)
            / f"seed_{seed}_fit_{n_fit}_calibration_{n_calibration}_validation_{n_density_validation}"
        )
    )
    manifest = folder / "manifest.json"
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if metadata.get("configuration") != configuration:
            raise ValueError(
                f"Existing shared reference configuration differs: {manifest}"
            )
        bank = ReferenceDatasets(folder, metadata)
        for model in ASSUMED_MODELS:
            for split in SPLITS:
                values = bank.load(model, split)
                if (
                    values.shape != (counts[split], num_obs, data_dim)
                    or values.dtype != np.float32
                    or not np.isfinite(values).all()
                ):
                    raise ValueError(f"Invalid shared reference array: {model}/{split}")
        return bank
    hashes = {}
    for model_index, model in enumerate(ASSUMED_MODELS):
        rng = np.random.default_rng(seed + 10_000 * model_index)
        spec = MODEL_SPECS[model]
        directory = folder / model
        directory.mkdir(parents=True, exist_ok=True)
        for split, count in counts.items():
            mu = rng.normal(
                spec["mu_prior_mean"], spec["mu_prior_std"], size=(count, 1, data_dim)
            )
            values = rng.normal(
                mu, spec["likelihood_std"], size=(count, num_obs, data_dim)
            ).astype(np.float32)
            path = directory / f"{split}.npy"
            # An incomplete previous creation is deterministic and may be resumed.
            if path.exists():
                if not np.array_equal(np.load(path, allow_pickle=False), values):
                    raise ValueError(
                        f"Unmanifested shared reference has different data: {path}"
                    )
            else:
                with path.with_suffix(".npy.tmp").open("wb") as stream:
                    np.save(stream, values, allow_pickle=False)
                path.with_suffix(".npy.tmp").replace(path)
            hashes[f"{model}/{split}.npy"] = file_sha256(path)
    metadata = {
        "schema_version": 1,
        "configuration": configuration,
        "file_sha256": hashes,
    }
    _write_json(manifest, metadata)
    return ReferenceDatasets(folder, metadata)


def validate_legacy_direct_reference(
    references, approximator, bank, model, settings
) -> dict:
    """Prove a legacy direct suite matches saved raw batches through recomputation.

    The complete fitting embedding matrix is retained by its MMD reference. Every
    calibration statistic is recomputed, including density centering/quantiles,
    and the saved density validation statistic must agree on the third batch.
    A discrepancy causes normal reference refitting; metadata alone is insufficient.
    """
    from sklearn.covariance import LedoitWolf
    from . import summry_diagnostic as sd

    summaries = {
        split: sd.summary_outputs(approximator, bank.load(model, split))
        for split in SPLITS
    }
    checks = {}

    def check(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=2e-6, atol=2e-6
        ):
            raise ValueError(
                f"Legacy reference does not match shared data: {model}/{name}"
            )
        checks[name] = float(np.max(np.abs(actual - expected)))

    fit, calibration = summaries["fit"], summaries["calibration"]
    check("full_fit_embedding_matrix", fit, references["mmd"]["reference_summary"])
    covariance = LedoitWolf().fit(fit)
    chol = np.linalg.cholesky(covariance.covariance_)
    bandwidth = sd.mmd_rbf_bandwidth2(fit, seed=settings["seed"])
    kernel_mean = sd._rbf_kernel_mean(fit, fit, bandwidth)
    check("mmd_bandwidth2", bandwidth, references["mmd"]["bandwidth2"])
    check(
        "mmd_reference_kernel_mean",
        kernel_mean,
        references["mmd"]["reference_kernel_mean"],
    )
    for metric in ("l2", "linf", "mmd"):
        reference = references[metric]
        if metric == "mmd":
            distances = sd.mmd_reference_distance_from_summary(
                calibration, fit, bandwidth, kernel_mean
            )
        else:
            check(f"{metric}_fit_mean", covariance.location_, reference["mu_hat"])
            check(f"{metric}_fit_cholesky", chol, reference["L_hat"])
            distances = sd.summary_distance_from_summary(
                calibration, covariance.location_, chol, metric=metric
            )
        statistics = sd.bootstrap_reference_stats(
            distances,
            alpha=settings["alpha"],
            n_boot=settings["n_boot"],
            seed=settings["seed"],
        )
        for key, value in statistics.items():
            check(f"{metric}_calibration_{key}", value, reference[key])
    density = references["density"]
    log_q = sd.typicality_log_density(density["flow"], calibration)
    check("density_expected_log_density", log_q.mean(), density["expected_log_density"])
    check("density_std_log_density", log_q.std(), density["std_log_density"])
    signed = log_q.mean() - log_q
    quantiles = np.percentile(
        signed, [50, 100 * settings["alpha"] / 2, 100 * (1 - settings["alpha"] / 2)]
    )
    for key, value in zip(("median", "dm_low", "dm_high"), quantiles, strict=True):
        check(f"density_calibration_{key}", value, density[key])
    validation = sd.density_flow_validation(
        density["flow"],
        summaries["validation"],
        num_flow_samples=settings["n_density_validation"],
        seed=settings["seed"],
    )
    for key, value in validation.items():
        check(key, value, density[key])
    return {
        "method": "recomputed embeddings, full MMD fitting matrix, all calibration statistics, density validation",
        "maximum_absolute_differences": checks,
        "recomputed_embedding_sha256": {
            split: _array_sha256(values) for split, values in summaries.items()
        },
        "rtol": 2e-6,
        "atol": 2e-6,
    }


def load_or_fit_model_reference(
    path: Path,
    approximator,
    bank: ReferenceDatasets,
    model: str,
    *,
    checkpoint: Path,
    settings: dict,
    metrics: tuple[str, ...],
    overwrite: bool = False,
    allow_legacy_direct: bool = False,
) -> dict:
    """Fit/reuse one network's reference, with shared raw input content in its key."""
    import keras
    from . import summry_diagnostic as sd

    path, checkpoint = Path(path), Path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".json")
    identity = {
        "schema_version": 2,
        "candidate_model": model,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "shared_reference_datasets": bank.identity,
        "settings": settings,
        "metrics": sorted(metrics),
        "implementation_sha256": file_sha256(Path(__file__)),
        "diagnostic_implementation_sha256": file_sha256(
            Path(__file__).with_name("summry_diagnostic.py")
        ),
        "checkpoint_loader_sha256": file_sha256(
            Path(__file__).parents[1] / "approximators" / "legacy_npe.py"
        ),
    }
    cached = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    valid_file = path.exists() and cached.get("sha256") == file_sha256(path)
    prior_identity = cached.get("identity", {})
    compatible_identity = {
        k: v for k, v in prior_identity.items() if k != "metrics"
    } == {k: v for k, v in identity.items() if k != "metrics"} and set(
        metrics
    ).issubset(prior_identity.get("metrics", []))
    if not overwrite and valid_file and compatible_identity:
        stored = sd.load_reference_suites(path)
        return {metric: stored[metric] for metric in metrics}
    migration = None
    if not overwrite and valid_file and allow_legacy_direct:
        old = cached.get("identity", {})
        expected_old_settings = {
            **settings,
            "seed": bank.metadata["configuration"]["seed"],
        }
        if (
            old.get("schema_version") == 1
            and str(old.get("summary_owner", "")).startswith("direct_")
            and old.get("checkpoint_sha256") == identity["checkpoint_sha256"]
            and old.get("settings") == expected_old_settings
        ):
            candidate = sd.load_reference_suites(path)
            try:
                migration = validate_legacy_direct_reference(
                    candidate, approximator, bank, model, settings
                )
            except (KeyError, ValueError, TypeError) as error:
                print(
                    f"Legacy {model} reference requires refitting: {error}", flush=True
                )
            else:
                migration["legacy_identity"] = old
                references = candidate
                print(
                    f"Verified legacy {model} reference against shared raw datasets",
                    flush=True,
                )
    if migration is None:
        keras.utils.set_random_seed(settings["seed"])
        simulator = bank.simulator(model)
        references = sd.fit_reference_suite(
            approximator, simulator, metrics=metrics, **settings
        )
        expected_splits = list(SPLITS if "density" in metrics else SPLITS[:2])
        if simulator.consumed_splits != expected_splits:
            raise ValueError(
                f"Reference fit did not consume expected shared splits: {simulator.consumed_splits}"
            )
        sd.save_reference_suites(references, path)
        provenance = {
            "method": "fitted using persisted shared raw datasets",
            "consumed_splits": simulator.consumed_splits,
        }
    else:
        provenance = migration
    _write_json(
        metadata_path,
        {"identity": identity, "sha256": file_sha256(path), "provenance": provenance},
    )
    return references
