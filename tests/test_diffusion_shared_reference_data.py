"""Raw-draw sharing and stale diagnostic-cache checks without network training."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import benchmark.examples.diffusion as diffusion


@pytest.fixture
def shared(tmp_path, monkeypatch):
    package_name = "benchmark.examples.diffusion.results"
    package = ModuleType(package_name)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setattr(diffusion, "results", package, raising=False)
    source = Path(diffusion.__file__).parent / "results/shared_reference_data.py"
    spec = importlib.util.spec_from_file_location(package_name + ".shared_reference_data", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "BASE_DIR", tmp_path)
    (tmp_path / "simulators").mkdir()
    (tmp_path / "simulators/rdm_4.py").write_text("# fixed simulator v1\n")
    calls = []

    class Simulator:
        def __init__(self, model, keep_params=False):
            self.model = model
            assert not keep_params

        def sample(self, count):
            calls.append((self.model, count))
            return {"rt": np.random.normal(size=(count, 2)),
                    "conditions": np.tile([0, 1], (count, 1))}

    monkeypatch.setitem(sys.modules, "benchmark.examples.diffusion.simulators.rdm_4", SimpleNamespace(RDM=Simulator))
    monkeypatch.setitem(sys.modules, "benchmark.examples.diffusion.dataset", SimpleNamespace(
        wagenmakers=SimpleNamespace(conditions=np.array([0, 1]))))

    def ensure(**kwargs):
        return module.ensure_shared_reference_data(root=tmp_path / "bank", n_fit=4,
            n_calibration=5, n_density_validation=6, **kwargs)

    return SimpleNamespace(module=module, ensure=ensure, calls=calls, root=tmp_path)


def test_reference_bank_is_persisted_and_identical_for_every_network(shared):
    module = shared.module
    before = np.random.get_state()
    manifest = shared.ensure()
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert len(shared.calls) == 4 * 3
    assert shared.ensure() == manifest
    assert len(shared.calls) == 4 * 3  # Loading another network never simulates.
    for model in module.MODELS:
        network_a = module.SharedReferenceSimulator(manifest, model)
        network_b = module.SharedReferenceSimulator(manifest, model)
        for count, part in zip((4, 5, 6), module.PARTITIONS):
            a, b = network_a.sample(count), network_b.sample(count)
            for key in ("rt", "conditions"):
                np.testing.assert_array_equal(a[key], b[key])
            saved = module.load_reference_partition(manifest, model, part)
            assert saved.dtype == np.float32 and saved.shape == (count, 2, 2)
            assert module.array_sha256(saved) == manifest["files"][model][part]["array_sha256"]
        with pytest.raises(ValueError, match="more than three"):
            network_a.sample(4)


def test_shared_reference_content_tampering_is_rejected(shared):
    manifest = shared.ensure()
    path = Path(manifest["bank_directory"]) / manifest["files"]["m0"]["fit"]["path"]
    with path.open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="modified"):
        shared.ensure()


def test_validation_random_stream_is_independent_of_fit_partition_length(shared):
    manifest = shared.ensure()
    different = shared.module.ensure_shared_reference_data(root=shared.root / "bank",
        n_fit=7, n_calibration=5, n_density_validation=6)
    for model in shared.module.MODELS:
        original = shared.module.load_reference_partition(manifest, model, "validation")
        other = shared.module.load_reference_partition(different, model, "validation")
        np.testing.assert_array_equal(original, other)
        assert manifest["files"][model]["calibration"]["array_sha256"] != different["files"][model]["calibration"]["array_sha256"]


def test_shared_reference_simulator_refuses_the_wrong_partition_count(shared):
    manifest = shared.ensure()
    simulator = shared.module.SharedReferenceSimulator(manifest, "m0")
    with pytest.raises(ValueError, match="requested 5"):
        simulator.sample(5)
    assert simulator.position == 0


def test_indirect_diagnostic_cache_tracks_reference_and_observation_content(shared):
    """Execute the real summary-only cache/ID logic with numerical network doubles."""
    module = shared.module
    source = Path(diffusion.__file__).parent / "results/multisource_pipeline.py"
    names = {"ensure_observed_summary_diagnostics", "compute_all_observed_summary_diagnostics"}
    functions = [node for node in ast.parse(source.read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    root = shared.root / "outputs"
    root.mkdir()
    reference = root / "reference.pkl"
    reference.write_bytes(b"old reference suite")
    metadata = reference.with_suffix(".json")
    module.write_json(metadata, {"identity": {"shared_reference_data": "raw bank A"},
                                 "sha256": module.file_sha256(reference)})
    result_path = root / "results.csv"
    results = pd.DataFrame({"dataset": ["empirical", "empirical"], "id": ["p1", "p0"]})
    results.to_csv(result_path, index=False)
    observations = np.array([[[10, 0], [10, 1]], [[20, 0], [20, 1]]], dtype=np.float32)
    calls = []

    def diagnostics(frame, y, approximators, references, metrics):
        calls.append(y.copy())
        output = frame.assign(d_m0=y[:, 0, 0], dm_median_m0=1.0)
        return {metric: output for metric in metrics}

    namespace = {
        "__file__": str(source), "Path": Path, "json": json, "np": np, "pd": pd,
        "REFERENCE_METRICS": ("l2",), "OBSERVED_DATASETS": ("empirical",),
        "NPE_RESULT_DIR": root,
        "reference_suite_path": lambda tag: reference,
        "results_path": lambda tag: result_path,
        "diagnostic_path": lambda tag, metric: root / f"{tag}_{metric}.csv",
        "load_observed_dataset": lambda dataset: (observations, ["p0", "p1"]),
        "add_summary_diagnostic_suite": diagnostics,
        "save_diagnostic": lambda frame, path: frame.to_csv(path, index=False),
        "array_sha256": module.array_sha256, "file_sha256": module.file_sha256,
        "write_json": module.write_json,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    config = SimpleNamespace(summary_label="S4D")
    run = lambda: namespace["ensure_observed_summary_diagnostics"](config, results, {}, {})
    first = run()["l2"]
    assert first["d_m0"].tolist() == [20, 10]  # Matches CSV IDs, not loader row order.
    pd.testing.assert_frame_equal(first, run()["l2"], check_dtype=False)
    assert len(calls) == 1
    reference.write_bytes(b"new shared-data reference suite")
    module.write_json(metadata, {"identity": {"shared_reference_data": "raw bank B"},
                                 "sha256": module.file_sha256(reference)})
    run()
    assert len(calls) == 2
    observations[0, :, 0] += 1
    updated = run()["l2"]
    assert updated["d_m0"].tolist() == [20, 11]
    assert len(calls) == 3
    (root / "S4D_l2.csv").write_text("corrupted\n")
    run()
    assert len(calls) == 4
    metadata.unlink()
    with pytest.raises(ValueError, match="lack shared-data provenance"):
        run()


def test_indirect_metric_subset_preserves_the_complete_verified_reference_suite(shared):
    module = shared.module
    source = Path(diffusion.__file__).parent / "results/multisource_pipeline.py"
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_or_fit_reference_suite")
    canonical = ("l2", "linf", "mmd", "density")
    path = shared.root / "npe_S4D_reference_suite.pkl"
    # A historical complete pickle without provenance must not be blindly reused.
    path.write_bytes(b"historical unverified suite")
    calls, saved = [], []
    manifest = shared.ensure()

    def verified_candidate(approximator, **kwargs):
        calls.append((kwargs["model"], kwargs["metrics"]))
        candidate_path = kwargs["path"]
        candidate_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_path.write_bytes(f"verified shared-bank {kwargs['model']}".encode())
        return {metric: {"from_verified_shared_bank": True} for metric in kwargs["metrics"]}

    def save_references(references, destination):
        saved.append(references)
        destination.write_text(json.dumps(references, sort_keys=True))

    namespace = {
        "Path": Path, "json": json, "TrainingConfig": object,
        "REFERENCE_METRICS": canonical, "MODELS": module.MODELS,
        "DEFAULT_SETTINGS": {"n_fit": 4, "n_calibration": 5, "n_density_validation": 6, "seed": 2025},
        "ensure_shared_reference_data": lambda **kwargs: manifest,
        "reference_suite_path": lambda tag: path,
        "get_name": lambda model, summary_label: f"{model}_{summary_label}",
        "get_path": lambda name: shared.root / f"{name}.keras",
        "load_or_fit_shared_reference": verified_candidate,
        "file_sha256": module.file_sha256, "reference_data_identity": module.reference_data_identity,
        "save_references": save_references, "write_json": module.write_json,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    config = SimpleNamespace(summary_label="S4D")
    approximators = dict.fromkeys(module.MODELS, object())
    run = namespace["load_or_fit_reference_suite"]
    full = run(approximators, config)
    digest = module.file_sha256(path)
    narrow = run(approximators, config, metrics=("l2",))
    assert full == narrow
    assert all(set(ref) == set(canonical) for ref in narrow.values())
    assert all(metrics == canonical for model, metrics in calls)
    assert len(calls) == 8  # Both routes go through the verified candidate cache.
    assert len(saved) == 1  # Narrow plotting did not replace the complete suite.
    assert module.file_sha256(path) == digest
