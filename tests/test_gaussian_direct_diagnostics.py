"""Scientific invariants for diagnostics built in direct classifier summaries."""

from __future__ import annotations

import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmark.examples.gaussian.analysis import direct_diagnostics as direct


def test_pooled_score_has_one_row_per_dataset_and_diagnostic():
    index = pd.DataFrame({"source_model": ["m7"] * 4, "id": [0, 1, 2, 3]})
    references = {
        metric: {"dm_low": -4.0, "dm_high": 8.0, "summary_dim": 12}
        for metric in direct.DIAGNOSTICS
    }
    distances = {metric: np.array([-4., 0., 8., 10.]) for metric in direct.DIAGNOSTICS}
    frame = direct.diagnostic_frame(index, distances, references, "cross_entropy")
    assert len(frame) == len(index) * len(direct.DIAGNOSTICS)
    assert "assumed_model" not in frame
    assert "distance_regime" not in frame
    assert "median" not in " ".join(frame.columns)
    assert set(frame["summary_owner"]) == {"direct_cross_entropy"}
    np.testing.assert_allclose(frame["rho"], np.tile([-0.5, 0, 1, 1.25], 4))
    np.testing.assert_allclose(frame["rho_low"], -0.5)


def _write_checkpoint(path: Path, factor: float) -> None:
    config = {
        "class_name": "ModelComparisonApproximator",
        "build_config": {"input_shape": {"summary_variables": [None, 2, 1]}},
        "config": {"summary_network": {"config": {"summary_dim": 2}}},
        "test_factor": factor,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))


@pytest.fixture
def isolated_direct_pipeline(tmp_path, monkeypatch):
    """Exercise cache orchestration without training expensive density flows."""
    from benchmark.examples.gaussian import analysis

    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    for source in direct.SOURCE_MODELS:
        items = [
            {
                "x": np.array([[1.0], [3.0]], dtype=np.float32),
                "source_model": source,
                "id": 0,
                **{
                    f"gold_log_marginal_{m}": float(i)
                    for i, m in enumerate(direct.ASSUMED_MODELS)
                },
                **{
                    f"npe_log_marginal_{m}": float(i) / 2
                    for i, m in enumerate(direct.ASSUMED_MODELS)
                },
            }
        ]
        with (dataset_dir / f"{source}_logml_pmp.pkl").open("wb") as stream:
            pickle.dump(items, stream)

    fitted, loaded, summarized = [], [], []

    class Approximator:
        def __init__(self, checkpoint):
            self.checkpoint = Path(checkpoint)
            with zipfile.ZipFile(checkpoint) as archive:
                self.factor = json.loads(archive.read("config.json"))["test_factor"]

        def summarize(self, data):
            x = np.asarray(data["x"])
            mean = x.mean(axis=(1, 2))
            summarized.append((self.checkpoint.name, x.copy()))
            return np.column_stack([mean * self.factor, mean**2 + self.factor])

        def estimate(self, conditions):
            return {
                "model_probs": np.tile([0.1, 0.2, 0.3, 0.4], (len(conditions["x"]), 1))
            }

    def load_model(checkpoint, compile=False):
        loaded.append(Path(checkpoint).name)
        return Approximator(checkpoint)

    def fit_reference_suite(approximator, simulator, metrics, **settings):
        summary = approximator.summarize(simulator.sample(settings["n_fit"]))
        simulator.sample(settings["n_calibration"])
        if "density" in metrics:
            simulator.sample(settings["n_density_validation"])
        fitted.append(
            (approximator.checkpoint.name, list(simulator.consumed_splits), summary.copy())
        )
        return {
            metric: {
                "summary_dim": 2,
                "dm_low": -1.0,
                "median": 0.0,
                "dm_high": 3.0,
                "mu_hat": summary.mean(axis=0),
                "L_hat": np.eye(2),
                "reference_summary": summary,
                "bandwidth2": 1.0,
                "reference_kernel_mean": 0.5,
                "flow": approximator.factor,
                "expected_log_density": 0.0,
            }
            for metric in metrics
        }

    def save_references(value, path):
        with Path(path).open("wb") as stream:
            pickle.dump(value, stream)

    def load_references(path):
        with Path(path).open("rb") as stream:
            return pickle.load(stream)

    fake_sd = SimpleNamespace(
        quiet_bayesflow_progress=lambda: None,
        summary_outputs=lambda approximator, x: approximator.summarize({"x": x}),
        fit_reference_suite=fit_reference_suite,
        save_reference_suites=save_references,
        load_reference_suites=load_references,
        summary_distance_from_summary=lambda summary, mean, chol, metric: (
            np.linalg.norm(summary - mean, axis=1)
        ),
        mmd_reference_distance_from_summary=lambda summary, reference, bandwidth, kernel: (
            np.linalg.norm(summary - reference.mean(axis=0), axis=1)
        ),
        signed_typicality_from_summary=lambda summary, flow, expected: (
            -summary[:, 0] / flow
        ),
    )
    monkeypatch.setitem(sys.modules, "bayesflow", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "keras",
        SimpleNamespace(
            saving=SimpleNamespace(load_model=load_model),
            utils=SimpleNamespace(set_random_seed=lambda seed: None),
        ),
    )
    monkeypatch.setitem(
        sys.modules, "benchmark.examples.gaussian.analysis.summry_diagnostic", fake_sd
    )
    monkeypatch.setattr(analysis, "summry_diagnostic", fake_sd, raising=False)

    def run(loss, checkpoint):
        return direct.ensure_direct_diagnostics(
            loss,
            checkpoint=checkpoint,
            results_dir=tmp_path / "results",
            dataset_dir=dataset_dir,
            n_fit=8,
            n_calibration=8,
            n_boot=0,
            density_epochs=1,
            n_density_validation=8,
            reference_root=tmp_path / "shared_reference",
        )

    return SimpleNamespace(
        run=run,
        fitted=fitted,
        loaded=loaded,
        summarized=summarized,
        root=tmp_path,
        dataset_dir=dataset_dir,
    )


def test_each_loss_fits_one_balanced_pool_in_its_own_direct_summary(
    isolated_direct_pipeline,
):
    pipeline = isolated_direct_pipeline
    for factor, loss in enumerate(direct.LOSSES, start=1):
        checkpoint = pipeline.root / f"direct_{loss}.keras"
        _write_checkpoint(checkpoint, factor)
        frame = pipeline.run(loss, checkpoint)
        assert set(frame["summary_owner"]) == {f"direct_{loss}"}
        assert set(frame["summary_dim"]) == {2}
        calls = [call for call in pipeline.fitted if call[0] == checkpoint.name]
        assert len(calls) == 1
        assert calls[0][1] == ["fit", "calibration", "validation"]
        assert calls[0][2].shape == (8, 2)
        cache = (
            pipeline.root
            / "results"
            / f"direct_{loss}"
            / "diagnostics"
            / "observed_summaries.npz"
        )
        with np.load(cache) as saved:
            np.testing.assert_allclose(
                saved["summary"], np.tile([2 * factor, 4 + factor], (12, 1))
            )
        assert len(frame) == 12 * len(direct.DIAGNOSTICS)
    assert len(pipeline.fitted) == 3


def test_cache_tracks_checkpoint_content_and_output_integrity(isolated_direct_pipeline):
    pipeline = isolated_direct_pipeline
    checkpoint = pipeline.root / "direct_cross_entropy.keras"
    _write_checkpoint(checkpoint, 1)
    original = pipeline.run("cross_entropy", checkpoint)
    assert len(pipeline.loaded) == 1
    assert len(pipeline.fitted) == 1
    cached = pipeline.run("cross_entropy", checkpoint)
    pd.testing.assert_frame_equal(original, cached, check_dtype=False)
    assert len(pipeline.loaded) == 1

    output = pipeline.root / "results" / "direct_cross_entropy"
    for relative_path in ["diagnostics/diagnostic_frame.csv", "pmp_comparison.csv"]:
        path = output / relative_path
        with path.open("a") as stream:
            stream.write("tampered\n")
        count = len(pipeline.loaded)
        restored = pipeline.run("cross_entropy", checkpoint)
        assert len(pipeline.loaded) == count + 1
        assert len(pipeline.fitted) == 1  # Valid reference fits are still reusable.
        pd.testing.assert_frame_equal(original, restored)
        assert "tampered" not in path.read_text()

    before = checkpoint.stat()
    _write_checkpoint(checkpoint, 2)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checkpoint.stat().st_size == before.st_size
    count = len(pipeline.loaded)
    pipeline.run("cross_entropy", checkpoint)
    assert len(pipeline.loaded) == count + 1
    assert len(pipeline.fitted) == 2
    with np.load(output / "diagnostics" / "observed_summaries.npz") as saved:
        np.testing.assert_allclose(saved["summary"], np.tile([4, 6], (12, 1)))
