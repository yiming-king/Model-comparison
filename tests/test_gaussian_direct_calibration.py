"""Check direct PMP calibration against the established Gaussian benchmark rule."""

from __future__ import annotations

import json
import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import multivariate_normal

from benchmark.examples.gaussian.analysis import direct_calibration as direct
from benchmark.examples.gaussian.calibration.thresholds import calculate_thresholds


def _component_metrics():
    rows = []
    for loss_index, loss in enumerate(direct.LOSSES):
        for model_index, generator in enumerate(direct.ASSUMED_MODELS):
            for dataset_id, errors in enumerate((
                [0.01, 0.04, -0.02, -0.03],
                [0.1, -0.4, 0.2, 0.1],
            )):
                for candidate, error in zip(direct.ASSUMED_MODELS, errors):
                    rows.append({
                        "loss_function": loss, "generating_model": generator,
                        "dataset_id": dataset_id, "candidate_model": candidate,
                        "signed_pmp_error": error / (1 + loss_index + model_index),
                        "configuration": "20d_10n", "network_tag": f"direct_{loss}",
                        "summary_label": "direct", "summary_dim": 12,
                        "num_dims": 20, "num_obs": 10, "num_posterior_samples": 1000,
                        "logml_method": "log_mean_exp", "posterior_mmd": 0.1,
                        "signed_logml_error": 0.2,
                    })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("coverage", [0.9, 0.8])
def test_direct_thresholds_match_original_all_component_protocol(coverage):
    metrics = _component_metrics()
    actual = direct.calculate_direct_pmp_thresholds(metrics, coverage)
    original = calculate_thresholds(metrics, signed_error_coverage=coverage)
    original = original.loc[original.metric.eq("signed_pmp_error")]
    columns = [
        "lower_quantile", "upper_quantile", "quantile_method",
        "lower_threshold", "threshold", "median", "n_values",
        "aggregation", "degenerate_interval",
    ]
    keys = ["network_tag", "generating_model"]
    pd.testing.assert_frame_equal(
        actual.set_index(keys)[columns].sort_index(),
        original.set_index(keys)[columns].sort_index(),
    )
    assert len(actual) == 3 * 4
    assert actual.n_values.eq(8).all()
    assert actual.num_datasets.eq(2).all()
    assert set(actual.metric) == {"signed_pmp_error"}

    matching_only = metrics.loc[
        metrics.loss_function.eq("cross_entropy")
        & metrics.generating_model.eq("m1")
        & metrics.candidate_model.eq("m1"), "signed_pmp_error"
    ]
    pooled = actual.set_index(keys).loc[("direct_cross_entropy", "m1")]
    assert pooled.lower_threshold < matching_only.min()
    assert pooled.threshold > matching_only.max()


def test_calibration_rejects_incomplete_or_duplicate_probability_vectors():
    metrics = _component_metrics()
    with pytest.raises(ValueError, match="all four PMP components"):
        direct.calculate_direct_pmp_thresholds(metrics.iloc[1:])
    with pytest.raises(ValueError, match="unique"):
        direct.calculate_direct_pmp_thresholds(pd.concat([metrics, metrics.iloc[:1]]))


@pytest.fixture
def saved_benchmark(tmp_path):
    root = tmp_path / "benchmark"
    arrays, indices = [], []
    for number, model in enumerate(direct.ASSUMED_MODELS):
        directory = root / "datasets" / model
        directory.mkdir(parents=True)
        x = np.arange(6, dtype=float).reshape(3, 2, 1) / 7 + number / 3
        np.save(directory / "x.npy", x)
        manifest = pd.DataFrame({
            "dataset_id": [91, 4, 23], "generating_model": model,
            "generation_seed": 2025 + 10_000 * number,
        })
        manifest.to_csv(directory / "manifest.csv", index=False)
        arrays.append(x)
        indices.append(manifest[["generating_model", "dataset_id"]])
    return root, np.concatenate(arrays), pd.concat(indices, ignore_index=True)


def test_benchmark_loader_preserves_saved_values_manifest_ids_and_order(saved_benchmark):
    root, expected_x, expected_index = saved_benchmark
    x, index = direct._load_benchmark(root, num_obs=2, data_dim=1)
    np.testing.assert_array_equal(x, expected_x)
    assert x.dtype == np.float64
    pd.testing.assert_frame_equal(index, expected_index)
    with pytest.raises(ValueError, match="shape/manifest mismatch"):
        direct._load_benchmark(root, num_obs=3, data_dim=1)


def test_analytical_pmp_uses_equal_model_prior_and_integrated_gaussian_likelihood():
    observations = np.array([[[0.0, 0.4], [0.2, -0.1]], [[2.5, 3.3], [3.2, 2.9]]])
    probabilities, evidence = direct._analytical_probabilities(observations)
    expected = np.empty_like(evidence)
    for row, observation in enumerate(observations):
        for column, model in enumerate(direct.ASSUMED_MODELS):
            spec = direct.MODEL_SPECS[model]
            covariance = spec["likelihood_std"]**2 * np.eye(2) + spec["mu_prior_std"]**2 * np.ones((2, 2))
            expected[row, column] = sum(
                multivariate_normal.logpdf(
                    observation[:, dimension],
                    mean=np.full(2, spec["mu_prior_mean"]), cov=covariance,
                )
                for dimension in range(2)
            )
    np.testing.assert_allclose(evidence, expected, atol=1e-12, rtol=0)
    np.testing.assert_allclose(probabilities, softmax(expected, axis=1), atol=1e-12, rtol=0)


def _write_checkpoint(path, probability):
    config = {
        "class_name": "ModelComparisonApproximator",
        "build_config": {"input_shape": {"summary_variables": [None, 2, 1]}},
        "config": {"summary_network": {"config": {"summary_dim": 12}}},
        "test_probability": probability,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))


def test_calibration_cache_evaluates_saved_data_and_invalidates_checkpoint_content(
    saved_benchmark, tmp_path, monkeypatch,
):
    root, observations, expected_index = saved_benchmark
    checkpoint = tmp_path / "direct_cross_entropy.keras"
    _write_checkpoint(checkpoint, 0.1)
    input_bytes = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    evaluated = []

    def load_model(path, compile=False):
        with zipfile.ZipFile(path) as archive:
            probability = json.loads(archive.read("config.json"))["test_probability"]

        def estimate(conditions):
            evaluated.append(conditions["x"].copy())
            return {"model_probs": np.tile(
                [probability, 0.2, 0.3, 0.5 - probability], (len(conditions["x"]), 1),
            )}

        return SimpleNamespace(estimate=estimate)

    monkeypatch.setitem(sys.modules, "bayesflow", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "keras", SimpleNamespace(saving=SimpleNamespace(load_model=load_model)))
    kwargs = {"checkpoint": checkpoint, "benchmark_dir": root, "results_dir": tmp_path / "results"}
    first = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    cached = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 1
    pd.testing.assert_frame_equal(first, cached, check_dtype=False)
    np.testing.assert_array_equal(evaluated[0], observations.astype(np.float32))
    assert set(first.metric) == {"signed_pmp_error"}
    assert first.n_values.eq(12).all()
    assert first.num_datasets.eq(3).all()

    output = tmp_path / "results" / "direct_cross_entropy" / "calibration"
    metrics = pd.read_csv(output / "per_dataset_metrics.csv")
    assert len(metrics) == 4 * len(observations)
    for model, frame in metrics.groupby("candidate_model", sort=False):
        pd.testing.assert_frame_equal(
            frame[["generating_model", "dataset_id"]].reset_index(drop=True), expected_index,
        )
    np.testing.assert_allclose(metrics.signed_pmp_error, metrics.direct_pmp - metrics.analytical_pmp)
    for path, content in input_bytes.items():
        assert path.read_bytes() == content

    before = checkpoint.stat()
    _write_checkpoint(checkpoint, 0.2)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checkpoint.stat().st_size == before.st_size
    second = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 2
    updated_metrics = pd.read_csv(output / "per_dataset_metrics.csv")
    np.testing.assert_allclose(
        updated_metrics.loc[updated_metrics.candidate_model.eq("m1"), "direct_pmp"], 0.2,
    )
    assert not np.array_equal(metrics.direct_pmp, updated_metrics.direct_pmp)
    with (output / "per_dataset_results.csv").open("a") as stream:
        stream.write("tampered\n")
    repaired = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 3
    pd.testing.assert_frame_equal(second, repaired)
    assert "tampered" not in (output / "per_dataset_results.csv").read_text()
