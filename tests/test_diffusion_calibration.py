import numpy as np
import pandas as pd
import pytest

from benchmark.examples.diffusion.calibration import pipeline, thresholds as threshold_module
from benchmark.examples.diffusion.calibration.thresholds import calculate_thresholds
from benchmark.examples.diffusion.config import MODEL_LABELS, TrainingConfig
from benchmark.examples.diffusion.results.summary_dimension_comparison import (
    METRIC_PLOTS,
    THRESHOLD_LINEWIDTH,
    _add_filename_suffix,
    _attach_calibration_thresholds,
    _compact_tick_label,
    _legend_layout,
    _metric_ylabel,
    _nonnegative_plot_limits,
    _normalize_distance,
    _scale_signed_by_lower_bound,
    _select_plot_summaries,
    _threshold_band,
    calculate_diagnostic_classification,
    load_calibration_thresholds,
)
from benchmark.examples.diffusion.results.results import pmp_from_log_marginals


def test_calibration_defaults_use_100_shared_datasets_and_current_variants():
    args = pipeline._parse_args(["generate"])
    assert args.k == 100
    assert args.summary_multipliers == [1, 2, 4]
    assert args.training_settings == ["without_mmd"]
    assert args.without_mmd_run_suffix == "noMMD"
    assert args.reference_root.name == "calibration_reference_100"
    assert args.output_root.name == "calibration_outputs_100_noMMD"

    paths = pipeline.CalibrationPaths()
    assert paths.datasets == args.reference_root / "datasets"
    assert paths.mcmc == args.reference_root / "mcmc"
    assert paths.manifest == args.reference_root / "dataset_manifest.csv"
    assert paths.thresholds == args.output_root / "thresholds.csv"


@pytest.mark.parametrize("variant", ["withMMD", "noMMD", "noMMD_rerun1"])
def test_calibration_variant_routes_checkpoints_and_thresholds_together(variant):
    args = pipeline._parse_args(["all", "--variant", variant])
    configs = pipeline.build_npe_configs(
        args.summary_multipliers,
        args.training_settings,
        args.without_mmd_run_suffix,
        args.embed_dim,
    )
    assert args.output_root.name == f"calibration_outputs_100_{variant}"
    assert all(
        config.training.run_suffix == (None if variant == "withMMD" else variant)
        for config in configs
    )
    threshold_args = threshold_module._parse_args(["--variant", variant])
    assert threshold_args.calibration_root == args.output_root


def test_calibration_explicit_roots_and_legacy_training_flags_are_unambiguous(tmp_path):
    args = pipeline._parse_args(
        [
            "all", "--training-settings", "without_mmd",
            "--without-mmd-run-suffix", "noMMD_rerun1",
        ]
    )
    assert args.variant == "noMMD_rerun1"
    assert args.output_root.name == "calibration_outputs_100_noMMD_rerun1"

    reference = tmp_path / "reference"
    output = tmp_path / "results"
    args = pipeline._parse_args(
        [
            "all", "--reference-root", str(reference), "--output-root", str(output),
            "--training-settings", "with_mmd", "without_mmd", "--k", "7",
        ]
    )
    paths = pipeline.CalibrationPaths(args.output_root, args.reference_root)
    assert args.k == 7
    assert paths.datasets == reference / "datasets"
    assert paths.manifest == reference / "dataset_manifest.csv"
    assert paths.per_dataset_metrics == output / "per_dataset_metrics.csv"
    assert threshold_module._parse_args(
        ["--calibration-root", str(output)]
    ).calibration_root == output
    with pytest.raises(SystemExit):
        pipeline._parse_args(
            ["metrics", "--variant", "withMMD", "--training-settings", "without_mmd"]
        )
    with pytest.raises(SystemExit):
        pipeline._parse_args(
            ["metrics", "--training-settings", "with_mmd", "without_mmd"]
        )


def test_generate_default_100_datasets_is_deterministic_and_protects_cache(tmp_path, monkeypatch):
    from benchmark.examples.diffusion import simulators

    class SmallSimulator:
        def sample(self, count):
            assert count == 1
            return {
                "rt": np.random.random((1, 384)),
                "conditions": np.zeros((1, 384), dtype=int),
                "alpha": np.ones((1, 1)),
                "nu": np.ones((1, 1)),
                "tau": np.ones((1, 1)),
            }

    monkeypatch.setattr(
        simulators, "SIMULATORS", {model: SmallSimulator() for model in pipeline.MODELS}
    )
    args = pipeline._parse_args(["generate"])
    paths = pipeline.CalibrationPaths(tmp_path / "results", tmp_path / "reference")
    kwargs = {
        "models": list(pipeline.MODELS), "k": args.k, "base_seed": args.base_seed,
    }
    manifest = pipeline.generate_datasets(paths, **kwargs, overwrite=False)
    assert len(manifest) == 400
    assert manifest.groupby("generating_model").size().to_dict() == {
        model: 100 for model in pipeline.MODELS
    }
    assert manifest["dataset_seed"].nunique() == 400
    expected_ids = [f"s{index:03d}" for index in range(100)]
    for model in pipeline.MODELS:
        assert sorted(path.stem for path in (paths.datasets / model).glob("s*.json")) == expected_ids
        assert manifest.loc[manifest["generating_model"].eq(model), "dataset_seed"].tolist() == [
            pipeline.dataset_seed(args.base_seed, model, index) for index in range(100)
        ]
    sample_path = paths.datasets / "m0" / "s099.json"
    original = sample_path.read_bytes()
    pipeline.generate_datasets(paths, **kwargs, overwrite=True)
    assert sample_path.read_bytes() == original
    sample_mtime = sample_path.stat().st_mtime_ns
    pipeline.generate_datasets(paths, **kwargs, overwrite=False)
    assert sample_path.stat().st_mtime_ns == sample_mtime
    with pytest.raises(ValueError, match="do not match"):
        pipeline.generate_datasets(paths, **{**kwargs, "k": 30}, overwrite=False)
    assert sample_path.read_bytes() == original
    assert not paths.root.exists()


def test_calibration_fit_resume_requires_used_draws_and_all_candidate_diagnostics(tmp_path):
    diagnostic = {
        "id": "s000", "mcmc_seed": 2025, "mcmc_elapsed_seconds": 1.0,
        "bridge_elapsed_seconds": 1.0, "fit_elapsed_seconds": 2.0,
        "max_rhat": 1.2, "min_n_eff": 100, "min_n_eff_ratio": 0.5,
        "num_chains": 4, "num_postwarmup_draws": 2048, "num_divergent": 0,
        "num_max_treedepth": 0, "min_ebfmi": 0.5, "converged": False,
    }
    pd.DataFrame([{"id": "s000", "estimate": 0.0, "sd": 0.1}]).to_csv(
        tmp_path / "bridgesampling.csv", index=False
    )
    diagnostic_path = tmp_path / "convergence_diagnostics.csv"
    pd.DataFrame([diagnostic]).to_csv(diagnostic_path, index=False)
    assert pipeline._fit_is_complete(tmp_path, {"s000"}, require_posterior_draws=False)
    assert not pipeline._fit_is_complete(tmp_path, {"s000"})
    draws = tmp_path / "posterior_draws"
    draws.mkdir()
    (draws / "s000.csv").write_text("parameter\n" + "1\n" * pipeline.GOLD_POSTERIOR_DRAWS)
    assert pipeline._fit_is_complete(tmp_path, {"s000"})
    assert not (tmp_path / "parameter_diagnostics").exists()
    assert not pipeline._fit_is_complete(tmp_path, {"s000", "s001"}, require_posterior_draws=False)
    pd.DataFrame([{key: value for key, value in diagnostic.items() if key != "converged"}]).to_csv(
        diagnostic_path, index=False
    )
    assert not pipeline._fit_is_complete(tmp_path, {"s000"}, require_posterior_draws=False)


def test_model_priors_are_used_in_pmp():
    pmp = pmp_from_log_marginals(
        np.zeros((1, 4)), {"m0": 0.1, "m1": 0.2, "m2": 0.3, "m3": 0.4}
    )
    np.testing.assert_allclose(pmp[0], [0.1, 0.2, 0.3, 0.4])


def test_model_labels_match_zero_based_model_codes():
    assert MODEL_LABELS == {"m0": "M0", "m1": "M1", "m2": "M2", "m3": "M3"}


def test_requested_rho_normalization_uses_d_over_d_high():
    normalized = _normalize_distance(
        np.array([-2.0, 2.0, 4.0]),
        median=1.0,
        high=4.0,
        method="upper_threshold",
    )
    np.testing.assert_allclose(normalized, [-0.5, 0.5, 1.0])


def test_figure_filename_suffix_preserves_extension():
    assert (
        _add_filename_suffix("combined_pmp_error_vs_rho.png", "trial")
        == "combined_pmp_error_vs_rho_trial.png"
    )


def test_logml_normalization_uses_only_lower_bound_scale():
    normalized = _scale_signed_by_lower_bound(
        np.array([-2.0, 0.0, 3.0]),
        low=-2.0,
    )
    np.testing.assert_allclose(normalized, [-1.0, 0.0, 1.5])


def test_posterior_mmd_axis_keeps_space_below_zero():
    assert _nonnegative_plot_limits(10.0) == (-0.5, 10.0)


def test_plot_overlays_hide_six_d_without_dropping_result_rows():
    data = pd.DataFrame({"summary": ["S=D", "S=2D", "S=4D", "S=6D"]})

    plotted = _select_plot_summaries(
        data,
        ("S=D", "S=2D", "S=4D"),
        "summary",
    )

    assert plotted["summary"].tolist() == ["S=D", "S=2D", "S=4D"]
    assert data["summary"].tolist()[-1] == "S=6D"


def test_two_model_legend_keeps_summary_entries_on_first_row():
    handles = ["S=D", "S=2D", "S=4D", "misspecified", "well-specified", "typical"]

    ordered, columns, rows = _legend_layout(
        handles,
        summary_count=3,
        model_count=2,
    )

    assert ordered == [
        "S=D",
        "misspecified",
        "S=2D",
        "well-specified",
        "S=4D",
        "typical",
    ]
    assert columns == 3
    assert rows == 2


def test_plot_labels_and_asymmetric_background_follow_display_rules():
    assert "normalized" in _metric_ylabel("Signed logML error", normalized=True)
    assert "normalized" not in _metric_ylabel("Signed logML error", normalized=False)
    assert _threshold_band((-1.0, 2.0, -0.5, 1.5)) == (-1.0, 2.0)
    assert _threshold_band((0.0, 0.3, 0.0, 0.8)) == (0.0, 0.8)
    assert METRIC_PLOTS["logml"]["sharey"] is True
    assert THRESHOLD_LINEWIDTH == 1.8


def test_symlog_tick_labels_are_compact():
    assert _compact_tick_label(0.129353) == "0.13"
    assert _compact_tick_label(-0.429574) == "-0.43"
    assert _compact_tick_label(0.0) == "0"


def test_thresholds_filter_nonconverged_matching_fit():
    rows = []
    for dataset, converged, mmd, logml in [
        ("s0", True, 1.0, 2.0),
        ("s1", False, 99.0, 99.0),
    ]:
        for candidate_index, candidate in enumerate(("m0", "m1", "m2", "m3")):
            rows.append(
                {
                    "generating_model": "m0",
                    "generating_model_label": "M0",
                    "dataset_id": dataset,
                    "candidate_model": candidate,
                    "model_prior": 0.25,
                    "npe_configuration": "S1D",
                    "training_setting": "with_mmd",
                    "summary_multiplier": 1,
                    "converged": converged if candidate == "m0" else True,
                    "all_candidate_models_converged": converged,
                    "importance_ess": 500.0,
                    "num_npe_logml_draws": 1000,
                    "posterior_mmd": mmd if candidate == "m0" else np.nan,
                    "absolute_logml_error": logml if candidate == "m0" else 0.0,
                    "signed_logml_error": logml if candidate == "m0" else 0.0,
                    "absolute_pmp_error": 0.1 * (candidate_index + 1),
                    "signed_pmp_error": 0.1 * (candidate_index + 1),
                }
            )
    thresholds = calculate_thresholds(pd.DataFrame(rows)).set_index("metric")
    assert thresholds.loc["posterior_mmd", "threshold"] == 1.0
    assert thresholds.loc["posterior_mmd", "lower_threshold"] == 0.0
    assert thresholds.loc["signed_logml_error", "threshold"] == 2.0
    assert thresholds.loc["signed_pmp_error", "n_values"] == 4


def test_posterior_uses_q95_and_signed_errors_use_central_90_percent():
    rows = []
    posterior_values = np.arange(20, dtype=float)
    logml_values = np.arange(-10, 10, dtype=float)
    pmp_values = []
    for dataset_id, (mmd, logml) in enumerate(
        zip(posterior_values, logml_values, strict=True)
    ):
        for candidate_index, candidate in enumerate(("m0", "m1", "m2", "m3")):
            pmp_error = float(4 * dataset_id + candidate_index)
            pmp_values.append(pmp_error)
            rows.append(
                {
                    "generating_model": "m0",
                    "generating_model_label": "M0",
                    "dataset_id": dataset_id,
                    "candidate_model": candidate,
                    "model_prior": 0.25,
                    "npe_configuration": "S1D",
                    "training_setting": "with_mmd",
                    "summary_multiplier": 1,
                    "converged": True,
                    "all_candidate_models_converged": True,
                    "importance_ess": 500.0,
                    "num_npe_logml_draws": 1000,
                    "posterior_mmd": mmd if candidate == "m0" else np.nan,
                    "signed_logml_error": logml if candidate == "m0" else 0.0,
                    "signed_pmp_error": pmp_error,
                }
            )

    thresholds = calculate_thresholds(pd.DataFrame(rows)).set_index("metric")

    assert thresholds.loc["posterior_mmd", "quantile"] == 0.95
    assert thresholds.loc["signed_logml_error", "quantile"] == 0.90
    assert thresholds.loc["signed_pmp_error", "quantile"] == 0.90
    np.testing.assert_allclose(
        thresholds.loc["posterior_mmd", "threshold"],
        np.quantile(posterior_values, 0.95, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_logml_error", "lower_threshold"],
        np.quantile(logml_values, 0.05, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_logml_error", "threshold"],
        np.quantile(logml_values, 0.95, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_pmp_error", "lower_threshold"],
        np.quantile(pmp_values, 0.05, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_pmp_error", "threshold"],
        np.quantile(pmp_values, 0.95, method="linear"),
    )


def test_plot_threshold_loader_converts_logml_to_base_10(tmp_path):
    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [
            {
                "generating_model": "m0",
                "npe_configuration": "S1D_noMMD",
                "metric": metric,
                "lower_threshold": lower_threshold,
                "threshold": threshold,
                "median": 0.0,
            }
            for metric, lower_threshold, threshold in (
                ("posterior_mmd", 0.0, 0.25),
                ("signed_logml_error", -np.log(10.0), 2.0 * np.log(10.0)),
                ("signed_pmp_error", -0.02, 0.05),
            )
        ]
    ).to_csv(path, index=False)

    thresholds = load_calibration_thresholds(
        (
            (
                "S=D",
                TrainingConfig(
                    summary_multiplier=1,
                    summary_base_distribution=None,
                    run_suffix="noMMD",
                ),
            ),
        ),
        path,
    ).iloc[0]

    assert thresholds["summary"] == "S=D"
    assert thresholds["posterior_mmd_upper_threshold"] == 0.25
    assert thresholds["posterior_mmd_lower_threshold"] == 0.0
    assert thresholds["log10_logml_error_lower_threshold"] == -1.0
    assert thresholds["log10_logml_error_upper_threshold"] == 2.0
    assert thresholds["pmp_error_lower_threshold"] == -0.02
    assert thresholds["pmp_error_upper_threshold"] == 0.05


def test_posterior_mmd_normalization_scales_by_q95(tmp_path):
    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [
            {
                "generating_model": "m0",
                "npe_configuration": "S1D_noMMD",
                "metric": metric,
                "lower_threshold": lower_threshold,
                "threshold": threshold,
                "median": median,
            }
            for metric, lower_threshold, threshold, median in (
                ("posterior_mmd", 0.0, 0.8, 0.2),
                ("signed_logml_error", -np.log(10.0), 2.0 * np.log(10.0), 0.0),
                ("signed_pmp_error", -0.05, 0.1, 0.0),
            )
        ]
    ).to_csv(path, index=False)
    data = pd.DataFrame(
        {
            "model": ["m0"] * 3,
            "summary": ["S=D"] * 3,
            "observed_mmd": [0.2, 0.5, 0.8],
            "logml_error": [0.0] * 3,
            "pmp_error": [0.0] * 3,
            "normalized_pmp_error": [999.0] * 3,
        }
    )

    output = _attach_calibration_thresholds(
        data,
        (
            (
                "S=D",
                TrainingConfig(
                    summary_multiplier=1,
                    summary_base_distribution=None,
                    run_suffix="noMMD",
                ),
            ),
        ),
        path,
    )

    np.testing.assert_allclose(output["normalized_mmd"], [0.25, 0.625, 1.0])
    np.testing.assert_allclose(output["normalized_mmd_low"], [0.0] * 3)
    np.testing.assert_allclose(output["normalized_mmd_high"], [1.0] * 3)
    np.testing.assert_allclose(output["normalized_logml_error_low"], [-1.0] * 3)
    np.testing.assert_allclose(output["normalized_logml_error_high"], [2.0] * 3)
    np.testing.assert_allclose(output["pmp_error_lower_threshold"], [-0.05] * 3)
    np.testing.assert_allclose(output["pmp_error_upper_threshold"], [0.1] * 3)
    assert not any(column.startswith("normalized_pmp") for column in output)


def test_calibrated_classification_uses_shared_error_definitions():
    data = pd.DataFrame(
        {
            "dataset": ["simulated_from_m0"] * 4,
            "id": range(4),
            "model": ["m0"] * 4,
            "summary": ["S=D"] * 4,
            "rho": [2.0, 2.0, 0.5, 0.5],
            "observed_mmd": [2.0, 0.5, 2.0, 0.5],
            "posterior_mmd_lower_threshold": [0.0] * 4,
            "posterior_mmd_upper_threshold": [1.0] * 4,
            "logml_error": [-2.0, 0.5, -2.0, 0.5],
            "log10_logml_error_lower_threshold": [-1.0] * 4,
            "log10_logml_error_upper_threshold": [1.0] * 4,
            "pmp_error": [2.0, 0.5, -2.0, 0.5],
            "pmp_error_lower_threshold": [-1.0] * 4,
            "pmp_error_upper_threshold": [1.0] * 4,
        }
    )

    metrics = calculate_diagnostic_classification(data, "l2")

    assert set(metrics["error_lower_threshold"]) == {-1.0, 0.0}
    assert set(metrics["error_upper_threshold"]) == {1.0}
    assert (metrics[["tp", "fp", "fn", "tn"]] == 1).all().all()
    np.testing.assert_allclose(metrics["false_positive_rate"], 0.5)
    np.testing.assert_allclose(metrics["false_negative_rate"], 0.5)
    np.testing.assert_allclose(metrics["f1"], 0.5)
    np.testing.assert_allclose(metrics["auc"], 0.5)

    display_mutation = data.assign(
        normalized_mmd=999.0,
        normalized_logml_error=-999.0,
    )
    pd.testing.assert_frame_equal(
        metrics,
        calculate_diagnostic_classification(display_mutation, "l2"),
    )
