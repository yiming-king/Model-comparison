"""All Gaussian diagnostic networks consume the same persisted raw references."""

import copy
import json

import numpy as np
import pytest

from benchmark.examples.gaussian.analysis import reference_datasets as shared
from benchmark.examples.gaussian.analysis.direct_diagnostics import _GaussianSimulator


def _bank(tmp_path):
    return shared.ensure_reference_datasets(
        3,
        4,
        n_fit=8,
        n_calibration=9,
        n_density_validation=10,
        seed=2025,
        root=tmp_path / "raw_reference",
    )


def test_persisted_reference_batches_match_historical_direct_rng_exactly(tmp_path):
    bank = _bank(tmp_path)
    files_before = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in bank.root.rglob("*.npy")
    }
    second = _bank(tmp_path)
    assert second.identity == bank.identity
    for path, (mtime, contents) in files_before.items():
        assert path.stat().st_mtime_ns == mtime
        assert path.read_bytes() == contents
    for index, model in enumerate(shared.ASSUMED_MODELS):
        historical = _GaussianSimulator(
            shared.MODEL_SPECS[model], 4, 3, 2025 + 10_000 * index
        )
        for split, count in zip(shared.SPLITS, (8, 9, 10)):
            np.testing.assert_array_equal(
                bank.load(model, split), historical.sample(count)["x"]
            )
    assert len(bank.identity["file_sha256"]) == 12


def test_shared_reference_replays_whole_splits_without_resampling_or_wraparound(
    tmp_path,
):
    bank = _bank(tmp_path)
    direct, indirect = bank.simulator("m2"), bank.simulator("m2")
    with pytest.raises(ValueError, match="contains 8"):
        direct.sample(9)
    for split, count in zip(shared.SPLITS, (8, 9, 10)):
        direct_x, indirect_x = direct.sample(count)["x"], indirect.sample(count)["x"]
        np.testing.assert_array_equal(direct_x, indirect_x)
        np.testing.assert_array_equal(direct_x, bank.load("m2", split))
    assert direct.consumed_splits == list(shared.SPLITS)
    with pytest.raises(ValueError, match="already been consumed"):
        direct.sample(8)


def test_changed_shared_raw_content_or_configuration_cannot_be_silently_reused(
    tmp_path,
):
    bank = _bank(tmp_path)
    with pytest.raises(ValueError, match="configuration differs"):
        shared.ensure_reference_datasets(3, 4, root=bank.root)
    path = bank.root / "m1" / "calibration.npy"
    values = np.load(path)
    values[0, 0, 0] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="content changed"):
        bank.load("m1", "calibration")
    with pytest.raises(ValueError, match="content changed"):
        _bank(tmp_path)


@pytest.fixture
def reference_suite(tmp_path, monkeypatch):
    from benchmark.examples.gaussian.analysis import summry_diagnostic as sd

    class Approximator:
        def summarize(self, data):
            values = np.asarray(data["x"], dtype=np.float64)
            mean = values.mean(axis=(1, 2))
            return np.column_stack([mean, values.var(axis=(1, 2)) + mean**2])

    fit_calls = []

    def fit_flow(values, **kwargs):
        fit_calls.append(np.asarray(values).copy())
        return {"fixture_flow": True}, None

    def validation(flow, summary, num_flow_samples=None, seed=2025):
        return {"density_validation_mmd2": float(np.mean(np.square(summary)))}

    monkeypatch.setattr(sd, "fit_typicality_flow", fit_flow)
    monkeypatch.setattr(
        sd,
        "typicality_log_density",
        lambda flow, values: -np.square(values).sum(axis=1),
    )
    monkeypatch.setattr(sd, "density_flow_validation", validation)
    bank, approximator = _bank(tmp_path), Approximator()
    settings = {
        "n_fit": 8,
        "n_calibration": 9,
        "n_density_validation": 10,
        "alpha": 0.1,
        "n_boot": 0,
        "seed": 2025,
        "density_epochs": 1,
        "density_batch_size": 8,
    }
    refs = sd.fit_reference_suite(approximator, bank.simulator("m1"), **settings)
    return bank, approximator, refs, settings, sd, fit_calls


@pytest.mark.parametrize("changed_split", ["fit", "calibration", "validation"])
def test_legacy_reuse_requires_actual_embeddings_from_each_shared_batch(
    reference_suite, changed_split
):
    bank, approximator, refs, settings, _, _ = reference_suite
    report = shared.validate_legacy_direct_reference(
        refs, approximator, bank, "m1", settings
    )
    assert set(report["recomputed_embedding_sha256"]) == set(shared.SPLITS)
    assert max(report["maximum_absolute_differences"].values()) < 1e-12
    path = bank.root / "m1" / f"{changed_split}.npy"
    values = np.load(path)
    values += 3.0
    np.save(path, values)
    updated_metadata = copy.deepcopy(bank.metadata)
    updated_metadata["file_sha256"][f"m1/{changed_split}.npy"] = shared.file_sha256(
        path
    )
    changed_bank = shared.ReferenceDatasets(bank.root, updated_metadata)
    with pytest.raises(ValueError, match="does not match shared data"):
        shared.validate_legacy_direct_reference(
            refs, approximator, changed_bank, "m1", settings
        )


def test_verified_legacy_direct_flow_is_reused_and_evidence_is_recorded(
    reference_suite, tmp_path
):
    bank, approximator, refs, settings, sd, fit_calls = reference_suite
    checkpoint = tmp_path / "direct.keras"
    checkpoint.write_bytes(b"unchanged trained direct checkpoint")
    path = tmp_path / "reference_m1.pkl"
    sd.save_reference_suites(refs, path)
    legacy = {
        "identity": {
            "schema_version": 1,
            "summary_owner": "direct_cross_entropy",
            "checkpoint_sha256": shared.file_sha256(checkpoint),
            "settings": settings,
        },
        "sha256": shared.file_sha256(path),
    }
    path.with_suffix(".json").write_text(json.dumps(legacy))
    kwargs = dict(
        checkpoint=checkpoint,
        settings=settings,
        metrics=tuple(refs),
        allow_legacy_direct=True,
    )
    first = shared.load_or_fit_model_reference(path, approximator, bank, "m1", **kwargs)
    assert len(fit_calls) == 1  # Only the fixture fit; migration trains nothing.
    assert first["density"]["flow"] == refs["density"]["flow"]
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["identity"]["shared_reference_datasets"] == bank.identity
    assert metadata["identity"]["checkpoint_loader_sha256"]
    assert "legacy_identity" in metadata["provenance"]
    assert (
        "full_fit_embedding_matrix"
        in metadata["provenance"]["maximum_absolute_differences"]
    )
    shared.load_or_fit_model_reference(path, approximator, bank, "m1", **kwargs)
    assert len(fit_calls) == 1
    checkpoint.write_bytes(b"changed trained direct checkpoint")
    shared.load_or_fit_model_reference(path, approximator, bank, "m1", **kwargs)
    assert len(fit_calls) == 2


def test_unproven_legacy_indirect_reference_is_refitted(reference_suite, tmp_path):
    bank, approximator, refs, settings, sd, fit_calls = reference_suite
    checkpoint = tmp_path / "indirect.keras"
    checkpoint.write_bytes(b"trained indirect checkpoint")
    path = tmp_path / "reference_m1.pkl"
    sd.save_reference_suites(
        refs, path
    )  # Historical pickle has no shared-input provenance.
    shared.load_or_fit_model_reference(
        path,
        approximator,
        bank,
        "m1",
        checkpoint=checkpoint,
        settings=settings,
        metrics=tuple(refs),
    )
    assert len(fit_calls) == 2
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["provenance"]["consumed_splits"] == list(shared.SPLITS)
    assert metadata["identity"]["shared_reference_datasets"] == bank.identity


def test_indirect_metric_subset_preserves_other_verified_bundle_metrics(
    reference_suite, tmp_path
):
    from benchmark.examples.gaussian.analysis.pipeline import load_or_fit_references

    bank, approximator, _, settings, sd, fit_calls = reference_suite
    checkpoints = {}
    for model in shared.ASSUMED_MODELS:
        checkpoints[model] = tmp_path / f"{model}.keras"
        checkpoints[model].write_bytes(f"trained {model}".encode())
    approximators = {model: approximator for model in shared.ASSUMED_MODELS}
    path = tmp_path / "reference_suite_test.pkl"
    kwargs = dict(
        overwrite=False,
        reference_kwargs=settings,
        shared_reference_data=bank,
        checkpoint_paths=checkpoints,
    )
    all_metrics = ("l2", "linf", "density", "mmd")
    load_or_fit_references(path, approximators, {}, all_metrics, **kwargs)
    count = len(fit_calls)
    selected = load_or_fit_references(path, approximators, {}, ("l2",), **kwargs)
    assert len(fit_calls) == count
    assert all(set(suite) == {"l2"} for suite in selected.values())
    stored = sd.load_reference_suites(path)
    assert all(set(suite) == set(all_metrics) for suite in stored.values())
    for model in shared.ASSUMED_MODELS:
        child = json.loads(path.with_name(f"{path.stem}_{model}.json").read_text())
        assert child["identity"]["shared_reference_datasets"] == bank.identity
