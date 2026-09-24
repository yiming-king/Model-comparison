"""Scientific and provenance checks for diffusion direct-summary diagnostics.

Density fitting is replaced by small deterministic doubles; no classifier or
reference flow is trained by these tests.
"""
from __future__ import annotations

import json
import importlib.util
import os
import pickle
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmark.examples.diffusion.analysis import direct_diagnostics as direct


SCORES = {
    "cross_entropy": "CrossEntropyScore",
    "exponential": "ExponentialScore",
    "logistic": "LogisticScore",
}


def write_checkpoint(path, loss="cross_entropy", factor=1):
    config = {
        "class_name": "ModelComparisonApproximator",
        "build_config": {"input_shape": {"summary_variables": [None, 2, 2]}},
        "config": {
            "summary_network": {"config": {"summary_dim": 2}},
            "inference_network": {"config": {"scoring_rules": {
                "scoring_rule": {"class_name": SCORES[loss]},
            }}},
        },
        "test_factor": factor,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))


@pytest.mark.parametrize("loss", direct.LOSSES)
def test_checkpoint_must_use_the_requested_scoring_rule(tmp_path, loss):
    checkpoint = tmp_path / "classifier.keras"
    write_checkpoint(checkpoint, loss)
    assert direct._checkpoint_config(checkpoint, loss) == (2, 2)
    for different_loss in set(direct.LOSSES) - {loss}:
        with pytest.raises(ValueError, match="does not match"):
            direct._checkpoint_config(checkpoint, different_loss)


def test_candidate_specific_rho_and_high_surprise_do_not_treat_low_tail_as_high():
    index = pd.DataFrame({"dataset": ["empirical"] * 4, "id": ["p0", "p1", "p2", "p3"]})
    references, distances = {}, {}
    for j, model in enumerate(direct.MODELS):
        references[model], distances[model] = {}, {}
        for k, metric in enumerate(direct.DIAGNOSTICS):
            low, median, high = -4 + j, 2 + j, 6 + 2 * j + k
            references[model][metric] = {
                "low": low, "median": median, "high": high, "summary_dim": 12,
            }
            distances[model][metric] = np.array(
                [low, median, high, high + 1] if j == 0
                else [high + 2, low - 1, high + 1, high + 2], dtype=float,
            )

    frame = direct.diagnostic_frame(index, distances, references, "cross_entropy")
    assert len(frame) == 4 * 4 * 4
    assert set(frame["summary_owner"]) == {"direct_cross_entropy"}
    for (metric, model), panel in frame.groupby(["diagnostic", "assumed_model"]):
        ref = references[model][metric]
        np.testing.assert_allclose(panel["rho"],
            (distances[model][metric] - ref["median"]) / (ref["high"] - ref["median"]))
        np.testing.assert_allclose(panel["rho_low"],
            (ref["low"] - ref["median"]) / (ref["high"] - ref["median"]))
        assert panel["globally_extrapolative"].tolist() == [False, False, False, True]
        assert panel["at_least_one_not_extrapolative"].tolist() == [True, True, True, False]
        if model == "m0":
            assert panel["rho"].iloc[2] == 1.0
            assert panel["distance_regime"].tolist() == [
                "in_distribution", "in_distribution", "in_distribution", "extrapolation",
            ]
        else:
            assert panel["distance_regime"].iloc[1] == "interpolation"

    other_scores = {m: {metric: x + 100 for metric, x in scores.items()}
                    for m, scores in distances.items()}
    other = direct.diagnostic_frame(index, other_scores, references, "logistic")
    assert other["globally_extrapolative"].all()
    assert set(other["summary_owner"]) == {"direct_logistic"}
    assert not np.array_equal(frame["rho"], other["rho"])


@pytest.fixture
def isolated_pipeline(tmp_path, monkeypatch):
    """Keep real loading, alignment, scoring and cache orchestration under test."""
    import benchmark.examples.diffusion as diffusion

    base = tmp_path / "diffusion"
    results = base / "results"
    for relative in ("results/summary_diagnostic.py", "results/results.py",
                     "results/observed_datasets.py", "results/shared_reference_fitting.py",
                     "simulators/rdm_4.py", "dataset/dataset.py"):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# reference implementation v1\n")
    monkeypatch.setattr(direct, "BASE_DIR", base)
    monkeypatch.setattr(direct, "RESULT_DIR", results)

    def observations(a, b):
        return np.array([[[a, 0], [a + 2, 1]], [[b, 0], [b + 2, 1]]], dtype=np.float32)

    datasets = {
        "empirical": (observations(1, 3), ["p0", "p1"]),
        "simulated_from_m0": (observations(5, 7), ["s0", "s1"]),
    }
    # Deliberately mix groups and reverse within-group row order.
    indirect = pd.DataFrame({
        "dataset": ["empirical", "simulated_from_m0", "empirical", "simulated_from_m0"],
        "id": ["p1", "s1", "p0", "s0"],
    })
    for j, m in enumerate(direct.MODELS):
        indirect[f"gold_pmp_{m}"] = [0.1, 0.2, 0.3, 0.4][j]
        indirect[f"pmp_{m}"] = 0.25
    indirect_path = results / "NPE_results" / "npe_S4D_all_observed_results_gold.csv"
    indirect_path.parent.mkdir(parents=True)
    indirect.to_csv(indirect_path, index=False)

    loaded, fitted, summarized, seeded = [], [], [], []

    def data_conditions(y):
        return {"rt": y[..., 0], "conditions": y[..., 1]}

    class Approximator:
        def __init__(self, path):
            self.path = Path(path)
            with zipfile.ZipFile(path) as archive:
                self.factor = json.loads(archive.read("config.json"))["test_factor"]

        def summarize(self, conditions):
            assert set(conditions) == {"rt", "conditions"}
            mean = np.asarray(conditions["rt"]).mean(axis=1)
            summarized.append((self.path.name, mean.copy()))
            return np.column_stack([mean * self.factor, mean**2 + self.factor])

        def estimate(self, conditions):
            assert set(conditions) == {"rt", "conditions"}
            values = np.array([self.factor, 2, 3, 4], dtype=float)
            return {"model_probs": np.tile(values / values.sum(), (len(conditions["rt"]), 1))}

    def load_model(path, compile=False):
        loaded.append(Path(path).name)
        return Approximator(path)

    class Simulator:
        def __init__(self, model, keep_params=False):
            self.model = model
            assert not keep_params

        def sample(self, count):
            value = direct.MODELS.index(self.model) + 1
            y = np.tile(np.array([[[value, 0], [value + 2, 1]]], dtype=np.float32), (count, 1, 1))
            return data_conditions(y)

    def fit_reference_suite(approximator, simulator, metrics, **settings):
        summary = approximator.summarize(simulator.sample(settings["n_fit"]))
        fitted.append((approximator.path.name, simulator.model, summary.copy()))
        return {metric: {
            "summary_dim": 2, "low": -1.0, "median": 0.0, "high": 3.0,
            "mean": summary.mean(axis=0), "chol": np.eye(2),
            "reference_summary": summary, "bandwidth2": 1.0, "reference_kernel_mean": 0.5,
            "flow": approximator.factor, "expected_log_density": 0.0,
        } for metric in metrics}

    def save_references(value, path):
        with Path(path).open("wb") as stream:
            pickle.dump(value, stream)

    def load_references(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    fake_sd = SimpleNamespace(
        __file__=str(base / "results/summary_diagnostic.py"),
        REFERENCE_METRICS=direct.DIAGNOSTICS,
        summary_outputs=lambda a, y: a.summarize(data_conditions(y)),
        fit_reference_suite=fit_reference_suite,
        save_references=save_references, load_references=load_references,
        summary_distance_from_summary=lambda s, mean, chol, metric: np.linalg.norm(s - mean, axis=1),
        mmd_reference_distance_from_summary=lambda s, ref, bandwidth, kernel: np.linalg.norm(s - ref.mean(axis=0), axis=1),
        signed_typicality_from_summary=lambda s, flow, expected: -s[:, 0] / flow,
    )
    def reject_benchmark_access(*args, **kwargs):
        pytest.fail("Summary diagnostics must not load the PMP calibration benchmark")

    original_open = Path.open

    def open_without_calibration_benchmark(path, *args, **kwargs):
        if "calibration_reference_100" in path.parts:
            reject_benchmark_access()
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_without_calibration_benchmark)
    observed = SimpleNamespace(
        load_observed_dataset=lambda name: datasets[name],
        load_dataset_directory=reject_benchmark_access,
    )
    fake_results = ModuleType("benchmark.examples.diffusion.results")
    fake_results.__path__ = []
    fake_results.summary_diagnostic = fake_sd
    fake_results.observed_datasets = observed
    fake_results.results = SimpleNamespace(data_conditions=data_conditions)
    monkeypatch.setitem(sys.modules, fake_results.__name__, fake_results)
    monkeypatch.setattr(diffusion, "results", fake_results, raising=False)
    for name, module in (("summary_diagnostic", fake_sd), ("observed_datasets", observed), ("results", fake_results.results)):
        monkeypatch.setitem(sys.modules, f"{fake_results.__name__}.{name}", module)
    # Run the real per-candidate cache orchestration with an in-memory raw bank.
    helpers = {}
    for name in ("shared_reference_data", "shared_reference_fitting"):
        module_name = f"{fake_results.__name__}.{name}"
        source = Path(direct.__file__).parents[1] / "results" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        setattr(fake_results, name, module)
        spec.loader.exec_module(module)
        helpers[name] = module

    def shared_manifest(root=None, **settings):
        return {
            "generation": settings, "bank_directory": str(base / "reference_datasets"),
            "files": {m: {part: {"array_sha256": f"{m}/{part}/{settings}"}
                          for part in ("fit", "calibration", "validation")} for m in direct.MODELS},
        }

    monkeypatch.setattr(helpers["shared_reference_data"], "ensure_shared_reference_data", shared_manifest)
    monkeypatch.setattr(helpers["shared_reference_fitting"], "SharedReferenceSimulator",
                        lambda manifest, model: Simulator(model))
    monkeypatch.setitem(sys.modules, "benchmark.examples.diffusion.simulators.rdm_4", SimpleNamespace(RDM=Simulator))
    monkeypatch.setitem(sys.modules, "bayesflow", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "keras", SimpleNamespace(
        saving=SimpleNamespace(load_model=load_model),
        utils=SimpleNamespace(set_random_seed=seeded.append, disable_interactive_logging=lambda: None),
    ))

    def run(loss, checkpoint, **settings):
        return direct.ensure_direct_diagnostics(
            loss, checkpoint=checkpoint, results_dir=results / "direct",
            **{"n_fit": 8, "n_calibration": 8, "n_boot": 0, "density_epochs": 1,
               "n_density_validation": 8, **settings},
        )

    return SimpleNamespace(
        run=run, loaded=loaded, fitted=fitted, summarized=summarized, seeded=seeded,
        base=base, root=tmp_path, results=results, datasets=datasets,
        indirect_path=indirect_path, indirect=indirect,
    )


def test_observations_follow_csv_dataset_id_order_and_reject_duplicates(isolated_pipeline):
    pipeline = isolated_pipeline
    y, index = direct._load_observations(pipeline.indirect_path, 2)
    assert index["id"].tolist() == ["p1", "s1", "p0", "s0"]
    np.testing.assert_array_equal(y[:, 0, 0], [3, 7, 1, 5])
    duplicate = pd.concat([pipeline.indirect, pipeline.indirect.iloc[:1]], ignore_index=True)
    duplicate.to_csv(pipeline.indirect_path, index=False)
    with pytest.raises(ValueError, match="unique nonempty"):
        direct._load_observations(pipeline.indirect_path, 2)


def test_observations_reject_missing_id_and_invalid_conditions(isolated_pipeline):
    pipeline = isolated_pipeline
    pipeline.indirect.iloc[:-1].to_csv(pipeline.indirect_path, index=False)
    with pytest.raises(ValueError, match="IDs do not match"):
        direct._load_observations(pipeline.indirect_path, 2)
    pipeline.indirect.to_csv(pipeline.indirect_path, index=False)
    pipeline.datasets["empirical"][0][0, 0, 1] = 2
    with pytest.raises(ValueError, match="Invalid RT or condition"):
        direct._load_observations(pipeline.indirect_path, 2)


def test_every_loss_scores_its_own_summary_and_four_conditional_references(isolated_pipeline):
    pipeline = isolated_pipeline
    for factor, loss in enumerate(direct.LOSSES, start=1):
        checkpoint = pipeline.root / f"direct_{loss}.keras"
        write_checkpoint(checkpoint, loss, factor)
        frame = pipeline.run(loss, checkpoint)
        assert set(frame["summary_owner"]) == {f"direct_{loss}"}
        calls = [call for call in pipeline.fitted if call[0] == checkpoint.name]
        assert [call[1] for call in calls] == list(direct.MODELS)
        assert not np.array_equal(calls[0][2], calls[1][2])
        output = pipeline.results / "direct" / f"direct_{loss}"
        with np.load(output / "diagnostics/observed_summaries.npz") as saved:
            expected_means = np.array([4, 8, 2, 6])
            np.testing.assert_array_equal(saved["id"], ["p1", "s1", "p0", "s0"])
            np.testing.assert_allclose(saved["summary"][:, 0], expected_means * factor)
        # Density distance is negative signed typicality, not absolute typicality.
        density = frame.query("diagnostic == 'density' and assumed_model == 'm0'")
        np.testing.assert_allclose(density["d_M"], expected_means)
        comparison = pd.read_csv(output / "pmp_comparison.csv")
        np.testing.assert_allclose(comparison["p_direct_m0"], factor / (factor + 9))
        assert len(frame) == len(pipeline.indirect) * 4 * 4
        assert not list((output / "diagnostics").glob("benchmark*"))
        metadata = json.loads((output / "diagnostics/metadata.json").read_text())
        assert not any(key.startswith("benchmark") for key in metadata["identity"])
        assert not any("benchmark" in path for path in metadata["output_sha256"])
        # One observed summary batch plus four independent reference-fit batches.
        assert len([call for call in pipeline.summarized if call[0] == checkpoint.name]) == 5
    assert not (pipeline.base / "calibration_reference_100").exists()
    assert len(pipeline.fitted) == 12
    assert pipeline.seeded == [2025, 2026, 2027, 2028] * 3


def test_cache_tracks_content_and_repairs_outputs_without_refitting(isolated_pipeline):
    pipeline = isolated_pipeline
    checkpoint = pipeline.root / "direct_cross_entropy.keras"
    write_checkpoint(checkpoint)
    original = pipeline.run("cross_entropy", checkpoint)
    cached = pipeline.run("cross_entropy", checkpoint)
    pd.testing.assert_frame_equal(original, cached, check_dtype=False)
    assert len(pipeline.loaded) == 1
    assert len(pipeline.fitted) == 4
    output = pipeline.results / "direct/direct_cross_entropy"
    for relative in ("pmp_comparison.csv", "diagnostics/diagnostic_frame.csv",
                     "diagnostics/observed_summaries.npz"):
        path = output / relative
        with path.open("ab") as stream:
            stream.write(b"tampered")
        previous_loads = len(pipeline.loaded)
        restored = pipeline.run("cross_entropy", checkpoint)
        assert len(pipeline.loaded) == previous_loads + 1
        assert len(pipeline.fitted) == 4
        pd.testing.assert_frame_equal(original, restored)
    before = checkpoint.stat()
    write_checkpoint(checkpoint, factor=2)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checkpoint.stat().st_size == before.st_size
    pipeline.run("cross_entropy", checkpoint)
    assert len(pipeline.fitted) == 8
    comparison = pd.read_csv(output / "pmp_comparison.csv")
    np.testing.assert_allclose(comparison["p_direct_m0"], 2 / 11)


def test_changed_observations_and_indirect_comparison_rescore_but_settings_refit(isolated_pipeline):
    pipeline = isolated_pipeline
    checkpoint = pipeline.root / "direct_cross_entropy.keras"
    write_checkpoint(checkpoint)
    pipeline.run("cross_entropy", checkpoint)
    pipeline.datasets["empirical"][0][1, :, 0] += 1
    pipeline.run("cross_entropy", checkpoint)
    assert len(pipeline.loaded) == 2 and len(pipeline.fitted) == 4
    pipeline.indirect["pmp_m0"] = 0.4
    pipeline.indirect["pmp_m1"] = 0.1
    pipeline.indirect.to_csv(pipeline.indirect_path, index=False)
    pipeline.run("cross_entropy", checkpoint)
    assert len(pipeline.loaded) == 3 and len(pipeline.fitted) == 4
    pipeline.run("cross_entropy", checkpoint, n_fit=10)
    assert len(pipeline.fitted) == 8
