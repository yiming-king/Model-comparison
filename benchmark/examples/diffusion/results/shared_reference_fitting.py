"""Checkpoint-specific diagnostic references over a shared raw reference bank."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..config import MODELS
from .shared_reference_data import (
    SharedReferenceSimulator, file_sha256, load_reference_partition,
    reference_data_identity, write_json,
)

DEFAULT_SETTINGS = {
    "n_fit": 2000, "n_calibration": 2000, "alpha": 0.1,
    "n_boot": 1000, "seed": 2025, "density_epochs": 250,
    "density_batch_size": 128, "n_density_validation": 2000,
}


def _validate_legacy_direct(references, approximator, manifest, model, settings, sd):
    """Verify fit embeddings and every calibration quantile before reuse.

    This is a one-time migration of the previously seeded direct references.
    Validation is always reevaluated using the newly persisted held-out partition.
    A mismatch returns False and the caller refits the complete reference suite.
    """
    from sklearn.covariance import LedoitWolf

    if set(sd.REFERENCE_METRICS) - set(references):
        return None
    fit = sd.summary_outputs(approximator, load_reference_partition(manifest, model, "fit"))
    calibration = sd.summary_outputs(approximator, load_reference_partition(manifest, model, "calibration"))
    prior = references["mmd"]["reference_summary"]
    if fit.shape != np.shape(prior) or not np.allclose(fit, prior, rtol=1e-6, atol=1e-7):
        return None
    covariance = LedoitWolf().fit(fit)
    mean, chol = covariance.location_, np.linalg.cholesky(covariance.covariance_)
    effective_seed = settings["seed"] + MODELS.index(model)
    checks = []

    def check(actual, expected):
        if not np.allclose(actual, expected, rtol=2e-5, atol=2e-6):
            raise ValueError("Legacy reference does not match shared-data recalculation")
        checks.append(float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))

    try:
        for metric in ("l2", "linf"):
            ref = references[metric]
            check(mean, ref["mean"])
            check(chol, ref["chol"])
            scores = sd.summary_distance_from_summary(calibration, mean, chol, metric=metric)
            stats = sd.bootstrap_interval(scores, alpha=settings["alpha"], n_boot=settings["n_boot"], seed=effective_seed)
            for key in ("low", "median", "high"):
                check(stats[key], ref[key])
        ref = references["mmd"]
        bandwidth = sd.mmd_rbf_bandwidth2(fit, seed=effective_seed)
        kernel_mean = sd._rbf_kernel_mean(fit, fit, bandwidth)
        check(bandwidth, ref["bandwidth2"])
        check(kernel_mean, ref["reference_kernel_mean"])
        scores = sd.mmd_reference_distance_from_summary(calibration, fit, bandwidth, kernel_mean)
        stats = sd.bootstrap_interval(scores, alpha=settings["alpha"], n_boot=settings["n_boot"], seed=effective_seed)
        for key in ("low", "median", "high"):
            check(stats[key], ref[key])
        ref = references["density"]
        log_q = sd.typicality_log_density(ref["flow"], calibration)
        expected = float(log_q.mean())
        check(expected, ref["expected_log_density"])
        values = -(log_q - expected)
        lower = settings["alpha"] / 2
        for key, value in zip(("low", "median", "high"), np.quantile(values, [lower, .5, 1 - lower])):
            check(value, ref[key])
    except (ValueError, KeyError, TypeError):
        return None
    validation = sd.summary_outputs(approximator, load_reference_partition(manifest, model, "validation"))
    references["density"].update(sd.density_flow_validation(
        references["density"]["flow"], validation,
        num_flow_samples=settings["n_density_validation"], seed=effective_seed,
    ))
    return {
        "kind": "legacy_direct_fit_and_calibration_verified",
        "fit_summary_max_abs_difference": float(np.max(np.abs(fit - prior))),
        "calibration_validation_max_abs_difference": max(checks),
        "verification_rtol": 2e-5, "verification_atol": 2e-6,
        "validation": "recomputed using persisted shared validation observations",
    }


def load_or_fit_shared_reference(approximator, *, checkpoint, model, owner, path,
                                 manifest, metrics, settings=None, overwrite=False,
                                 allow_legacy_direct=False):
    """Hash-check each candidate independently, so interrupted runs can resume."""
    import keras
    from . import summary_diagnostic as sd

    settings = {**DEFAULT_SETTINGS, **(settings or {})}
    path, checkpoint = Path(path), Path(checkpoint)
    metadata_path = path.with_suffix(".json")
    identity = {
        "schema_version": 2, "summary_owner": owner, "assumed_model": model,
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": file_sha256(checkpoint),
        "shared_reference_data": reference_data_identity(manifest, model),
        "settings": settings, "metrics": list(metrics),
        "effective_seed": settings["seed"] + MODELS.index(model),
        "reference_implementation_sha256": file_sha256(Path(sd.__file__)),
        "shared_fitting_implementation_sha256": file_sha256(Path(__file__)),
        "checkpoint_loader_sha256": file_sha256(
            Path(__file__).parents[2] / "gaussian/approximators/legacy_npe.py"
        ),
    }
    cached = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    valid_file = path.exists() and cached.get("sha256") == file_sha256(path)
    if not overwrite and valid_file and cached.get("identity") == identity:
        return sd.load_references(path)

    migration = None
    previous_identity = cached.get("identity", {})
    if (allow_legacy_direct and not overwrite and valid_file
            and previous_identity.get("schema_version") == 1
            and previous_identity.get("checkpoint_sha256") == identity["checkpoint_sha256"]
            and previous_identity.get("summary_owner") == owner
            and previous_identity.get("settings") == settings
            and previous_identity.get("simulator_sha256") == manifest["generation"]["simulator_sha256"]):
        references = sd.load_references(path)
        migration = _validate_legacy_direct(references, approximator, manifest, model, settings, sd)
    if migration is None:
        print(f"Fitting {owner}/{model} against the persisted shared raw reference bank", flush=True)
        keras.utils.set_random_seed(identity["effective_seed"])
        references = sd.fit_reference_suite(
            approximator, SharedReferenceSimulator(manifest, model), metrics=metrics,
            **{**settings, "seed": identity["effective_seed"]},
        )
    else:
        print(f"Verified {owner}/{model} legacy fit/calibration; refreshed shared validation", flush=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pkl.tmp")
    sd.save_references(references, temporary)
    temporary.replace(path)
    write_json(metadata_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": identity, "sha256": file_sha256(path),
        "migration": migration,
    })
    return references
