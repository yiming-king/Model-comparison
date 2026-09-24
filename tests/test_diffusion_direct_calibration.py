"""Direct diffusion PMP calibration preserves the indirect benchmark protocol."""

import json
import os
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from benchmark.examples.diffusion.analysis import direct_calibration as direct
from benchmark.examples.diffusion.calibration.thresholds import calculate_thresholds


def _metrics():
    rows = []
    for loss_index, loss in enumerate(direct.LOSSES):
        for generator in direct.MODELS:
            for dataset_id, errors in enumerate((
                [0.01, 0.04, -0.02, -0.03],
                [0.1, -0.4, 0.2, 0.1],
                [0.9, -0.9, 0.0, 0.0],
            )):
                for candidate, error in zip(direct.MODELS, errors):
                    rows.append({
                        "loss_function": loss, "generating_model": generator,
                        "dataset_id": f"s{dataset_id:03d}", "candidate_model": candidate,
                        "signed_pmp_error": error / (loss_index + 1),
                        "converged": not (dataset_id == 2 and candidate == "m3"),
                        "all_candidate_models_converged": dataset_id != 2,
                        "generating_model_label": generator.upper(),
                        "npe_configuration": f"direct_{loss}", "training_setting": "direct",
                        "summary_multiplier": 1, "model_prior": 0.25,
                        "posterior_mmd": 0.1, "signed_logml_error": 0.2,
                    })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("coverage", [0.9, 0.8])
def test_thresholds_match_indirect_convergence_filtered_component_pooling(coverage):
    metrics = _metrics()
    result = direct.calculate_direct_pmp_thresholds(metrics, coverage)
    old = calculate_thresholds(metrics, signed_error_coverage=coverage)
    old = old.loc[old.metric.eq("signed_pmp_error")].rename(columns={"npe_configuration": "network_tag"})
    columns = [
        "lower_quantile", "upper_quantile", "quantile_method", "lower_threshold",
        "threshold", "median", "n_values", "aggregation",
    ]
    keys = ["network_tag", "generating_model"]
    pd.testing.assert_frame_equal(
        result.set_index(keys)[columns].sort_index(), old.set_index(keys)[columns].sort_index(),
    )
    assert result.num_datasets.eq(3).all()
    assert result.num_converged_datasets.eq(2).all()
    assert result.n_values.eq(8).all()
    assert set(result.metric) == {"signed_pmp_error"}
    first = result.set_index(keys).loc[("direct_cross_entropy", "m0")]
    assert first.lower_threshold < 0  # Retains errors from nonmatching candidates.
    assert first.threshold < 0.9  # Excludes every component of the failed dataset.


def test_missing_candidates_and_inconsistent_convergence_cannot_enter_thresholds():
    metrics = _metrics()
    with pytest.raises(ValueError, match="all four PMP components"):
        direct.calculate_direct_pmp_thresholds(metrics.iloc[1:])
    metrics.loc[0, "all_candidate_models_converged"] = False
    with pytest.raises(ValueError, match="Inconsistent"):
        direct.calculate_direct_pmp_thresholds(metrics)
    missing_generator = _metrics().loc[lambda x: ~(x.loss_function.eq("logistic") & x.generating_model.eq("m3"))]
    with pytest.raises(ValueError, match="Each direct loss"):
        direct.calculate_direct_pmp_thresholds(missing_generator)


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "reference"
    ids = ["s009", "s002", "s017"]
    observations, manifests = [], []
    for generator_index, generator in enumerate(direct.MODELS):
        directory = root / "datasets" / generator
        directory.mkdir(parents=True)
        pd.DataFrame({"id": ids, "dataset_seed": [31, 32, 33]}).to_csv(directory / "true_parameters.csv", index=False)
        manifests.append(pd.DataFrame({"generating_model": generator, "id": ids, "dataset_seed": [31, 32, 33]}))
        for number, dataset_id in enumerate(ids):
            record = {"N": 2, "rt": [0.4 + number / 10, -0.5 - generator_index / 10], "condition": [1, 0]}
            (directory / f"{dataset_id}.json").write_text(json.dumps(record))
            observations.append(np.stack([record["rt"], record["condition"]], axis=-1))
        for candidate_index, candidate in enumerate(direct.MODELS):
            folder = root / "mcmc" / generator / candidate
            folder.mkdir(parents=True)
            # Deliberately shuffle the MCMC rows; loading must join by saved ID.
            pd.DataFrame({
                "id": ids[::-1], "estimate": [candidate_index + 2, candidate_index + 1, candidate_index],
                "sd": 0.01,
            }).to_csv(folder / "bridgesampling.csv", index=False)
            pd.DataFrame({
                "id": ids, "converged": [True, True, candidate != "m3"],
            }).to_csv(folder / "convergence_diagnostics.csv", index=False)
    pd.concat(manifests, ignore_index=True).to_csv(root / "dataset_manifest.csv", index=False)
    return root, np.asarray(observations, dtype=np.float32)


def test_loader_preserves_saved_rt_condition_ids_and_mcmc_alignment(benchmark):
    root, expected = benchmark
    observations, index = direct._load_benchmark(root, 2)
    np.testing.assert_array_equal(observations, expected)
    assert observations.dtype == np.float32
    assert index.dataset_id.tolist() == ["s009", "s002", "s017"] * 4
    assert index.generating_model.tolist() == [m for m in direct.MODELS for _ in range(3)]
    assert index.all_candidate_models_converged.tolist() == [True, True, False] * 4
    np.testing.assert_allclose(index.bridge_log_ml_m0, [0, 1, 2] * 4)
    with pytest.raises(ValueError, match="shape/content"):
        direct._load_benchmark(root, 384)
    manifest = pd.read_csv(root / "dataset_manifest.csv")
    manifest.loc[0, "dataset_seed"] += 1
    manifest.to_csv(root / "dataset_manifest.csv", index=False)
    with pytest.raises(ValueError, match="Shared benchmark manifest"):
        direct._load_benchmark(root, 2)


def _checkpoint(path, probability):
    config = {
        "class_name": "ModelComparisonApproximator",
        "build_config": {"input_shape": {"summary_variables": [None, 2, 2], "inference_variables": [None, 4]}},
        "config": {
            "summary_network": {"config": {"summary_dim": 12}},
            "inference_network": {"config": {"scoring_rules": {"scoring_rule": {"class_name": "CrossEntropyScore"}}}},
        },
        "test_probability": probability,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("config.json", json.dumps(config))


def test_cached_calibration_uses_diffusion_adapter_and_tracks_checkpoint_and_gold(
    benchmark, tmp_path, monkeypatch,
):
    root, expected = benchmark
    checkpoint = tmp_path / "direct_cross_entropy.keras"
    _checkpoint(checkpoint, 0.1)
    inputs = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    evaluated = []

    def load_model(path, compile=False):
        with zipfile.ZipFile(path) as archive:
            probability = json.loads(archive.read("config.json"))["test_probability"]

        def estimate(conditions):
            assert set(conditions) == {"rt", "conditions"}
            evaluated.append(np.stack([conditions["rt"], conditions["conditions"]], axis=-1))
            return {"model_probs": np.tile([probability, 0.2, 0.3, 0.5 - probability], (len(conditions["rt"]), 1))}

        return SimpleNamespace(estimate=estimate)

    monkeypatch.setitem(sys.modules, "bayesflow", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "keras", SimpleNamespace(saving=SimpleNamespace(load_model=load_model)))
    kwargs = {"checkpoint": checkpoint, "benchmark_dir": root, "results_dir": tmp_path / "results"}
    first = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    cached = direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    pd.testing.assert_frame_equal(first, cached, check_dtype=False)
    assert len(evaluated) == 1
    np.testing.assert_array_equal(evaluated[0], expected)
    for path, content in inputs.items():
        assert path.read_bytes() == content

    output = tmp_path / "results" / "direct_cross_entropy" / "calibration"
    metrics = pd.read_csv(output / "per_dataset_metrics.csv")
    assert len(metrics) == 12 * 4
    assert set(first.metric) == {"signed_pmp_error"}
    assert first.n_values.eq(8).all()
    for candidate_index, model in enumerate(direct.MODELS):
        frame = metrics.loc[metrics.candidate_model.eq(model)]
        np.testing.assert_allclose(frame.gold_pmp, softmax(np.arange(4))[candidate_index])
    np.testing.assert_allclose(metrics.signed_pmp_error, metrics.direct_pmp - metrics.gold_pmp)

    before = checkpoint.stat()
    _checkpoint(checkpoint, 0.2)
    os.utime(checkpoint, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert checkpoint.stat().st_size == before.st_size
    direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 2
    updated = pd.read_csv(output / "per_dataset_metrics.csv")
    np.testing.assert_allclose(updated.loc[updated.candidate_model.eq("m0"), "direct_pmp"], 0.2)

    gold_path = root / "mcmc" / "m0" / "m0" / "bridgesampling.csv"
    gold = pd.read_csv(gold_path)
    gold["estimate"] += 2
    gold.to_csv(gold_path, index=False)
    direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 3
    with (output / "thresholds.csv").open("a") as stream:
        stream.write("tampered\n")
    direct.ensure_direct_pmp_thresholds("cross_entropy", **kwargs)
    assert len(evaluated) == 4
    assert "tampered" not in (output / "thresholds.csv").read_text()


def test_missing_direct_checkpoint_is_explicit_and_never_creates_placeholder(tmp_path):
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        direct.ensure_direct_pmp_thresholds(
            "exponential", checkpoint=tmp_path / "missing.keras", results_dir=tmp_path / "results",
        )
    assert not (tmp_path / "results").exists()


def test_cross_entropy_checkpoint_cannot_stand_in_for_missing_other_loss(tmp_path):
    path = tmp_path / "direct_cross_entropy.keras"
    _checkpoint(path, 0.1)
    with pytest.raises(ValueError, match="scoring rule does not match"):
        direct.ensure_direct_pmp_thresholds("logistic", checkpoint=path, results_dir=tmp_path / "results")
    assert not (tmp_path / "results").exists()
