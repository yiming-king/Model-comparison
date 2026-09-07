from __future__ import annotations

import numpy as np
import pandas as pd

from benchmark.examples.gaussian.config import NetworkSet, discover_network_sets
from benchmark.examples.gaussian.calibration.pipeline import CalibrationPaths
from benchmark.examples.gaussian.calibration.thresholds import (
    add_thresholds_to_metrics,
    calculate_thresholds,
)
from benchmark.examples.gaussian.analysis.summry_diagnostic import (
    mmd_rbf_bandwidth2,
    mmd_reference_distance_from_summary,
)
from benchmark.examples.gaussian.analysis.pipeline import (
    OODPaths,
    THRESHOLD_LINEWIDTH,
    _collect_inference_frames,
    _metric_ylabel,
    _nonnegative_plot_limits,
    _scale_signed_by_lower_bound,
    _threshold_lookup,
    attach_calibrated_errors,
    load_cached_inference_results,
    load_cached_metric_frames,
)
def _calibration_frame() -> pd.DataFrame:
    rows = []
    for dataset_id in range(3):
        for candidate_index, candidate_model in enumerate(("m1", "m2", "m3", "m4")):
            matching = candidate_model == "m1"
            signed_logml = (-1.0, 0.0, 3.0)[dataset_id] if matching else 99.0
            signed_pmp = 0.01 * (dataset_id * 4 + candidate_index + 1)
            rows.append(
                {
                    "configuration": "20d_10n",
                    "network_tag": "20d_10n",
                    "summary_label": "S=1D",
                    "summary_dim": 20,
                    "generating_model": "m1",
                    "num_dims": 20,
                    "num_obs": 10,
                    "num_posterior_samples": 1000,
                    "logml_method": "log_mean_exp",
                    "dataset_id": dataset_id,
                    "candidate_model": candidate_model,
                    "posterior_mmd": (0.1, 0.2, 0.5)[dataset_id]
                    if matching
                    else np.nan,
                    "signed_logml_error": signed_logml,
                    "absolute_logml_error": abs(signed_logml),
                    "signed_pmp_error": signed_pmp,
                    "absolute_pmp_error": abs(signed_pmp),
                }
            )
    return pd.DataFrame(rows)


def test_thresholds_use_matching_posterior_and_logml_but_pool_pmp_components():
    thresholds = calculate_thresholds(_calibration_frame()).set_index("metric")

    assert thresholds.loc["posterior_mmd", "n_values"] == 3
    assert thresholds.loc["signed_logml_error", "n_values"] == 3
    assert thresholds.loc["signed_pmp_error", "n_values"] == 12
    assert thresholds.loc["posterior_mmd", "median"] == 0.2
    assert thresholds.loc["signed_logml_error", "median"] == 0.0
    assert thresholds.loc["posterior_mmd", "quantile"] == 0.95
    assert thresholds.loc["signed_logml_error", "quantile"] == 0.90
    assert thresholds.loc["signed_pmp_error", "quantile"] == 0.90
    np.testing.assert_allclose(
        thresholds.loc["posterior_mmd", "threshold"],
        np.quantile([0.1, 0.2, 0.5], 0.95, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["posterior_mmd", "lower_threshold"],
        0.0,
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_pmp_error", "threshold"],
        np.quantile(np.arange(1, 13) / 100.0, 0.95, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_logml_error", "lower_threshold"],
        np.quantile([-1.0, 0.0, 3.0], 0.05, method="linear"),
    )
    np.testing.assert_allclose(
        thresholds.loc["signed_logml_error", "threshold"],
        np.quantile([-1.0, 0.0, 3.0], 0.95, method="linear"),
    )


def test_thresholds_use_all_values_without_importance_ess_fields():
    frame = _calibration_frame()

    thresholds = calculate_thresholds(frame).set_index("metric")

    assert thresholds.loc["posterior_mmd", "n_values"] == 3
    assert thresholds.loc["signed_logml_error", "n_values"] == 3
    assert thresholds.loc["signed_pmp_error", "n_values"] == 12
    assert not any("ess" in column.lower() for column in thresholds.columns)


def test_zero_width_pmp_interval_is_marked_degenerate():
    frame = _calibration_frame()
    frame["signed_pmp_error"] = 0.0

    thresholds = calculate_thresholds(frame).set_index("metric")

    assert thresholds.loc["signed_pmp_error", "lower_threshold"] == 0.0
    assert thresholds.loc["signed_pmp_error", "threshold"] == 0.0
    assert bool(thresholds.loc["signed_pmp_error", "degenerate_interval"])


def test_threshold_lookup_accepts_zero_width_pmp_interval(tmp_path):
    thresholds = calculate_thresholds(_calibration_frame())
    pmp = thresholds["metric"].eq("signed_pmp_error")
    thresholds.loc[pmp, ["lower_threshold", "threshold", "median"]] = 0.0
    path = tmp_path / "thresholds.csv"
    thresholds.to_csv(path, index=False)
    network_set = NetworkSet(
        network_tag="20d_10n",
        data_dim=20,
        num_obs=10,
        summary_dim=20,
        paths={},
    )

    output = _threshold_lookup(path, "20d_10n", network_set)

    assert output["signed_pmp_error_degenerate_interval"].all()


def test_logml_normalization_uses_only_lower_bound_scale():
    normalized = _scale_signed_by_lower_bound(
        np.array([-2.0, 0.0, 3.0]),
        low=-2.0,
    )
    np.testing.assert_allclose(normalized, [-1.0, 0.0, 1.5])


def test_posterior_mmd_axis_keeps_space_below_zero():
    assert _nonnegative_plot_limits(10.0) == (-0.5, 10.0)


def test_plot_label_suffix_follows_normalization_switch():
    assert "normalized" in _metric_ylabel("Signed logML error", normalized=True)
    assert "normalized" not in _metric_ylabel("Signed logML error", normalized=False)
    assert THRESHOLD_LINEWIDTH == 1.8


def test_posterior_mmd_normalization_scales_by_q95():
    thresholds = pd.DataFrame(
        {
            "assumed_model": ["m1", "m2", "m3", "m4"],
            "posterior_mmd_threshold": [0.8] * 4,
            "posterior_mmd_lower_threshold": [0.0] * 4,
            "posterior_mmd_median": [0.2] * 4,
            "signed_logml_error_lower_threshold": [-1.0] * 4,
            "signed_logml_error_threshold": [2.0] * 4,
            "signed_logml_error_median": [0.0] * 4,
            "signed_pmp_error_lower_threshold": [-0.05] * 4,
            "signed_pmp_error_threshold": [0.1] * 4,
            "signed_pmp_error_median": [0.0] * 4,
        }
    )
    posterior = pd.DataFrame(
        {
            "assumed_model": ["m1"] * 3,
            "posterior_mmd": [0.2, 0.5, 0.8],
        }
    )
    logml = pd.DataFrame(
        {"assumed_model": ["m1"], "signed_logml_error": [0.0]}
    )
    pmp = pd.DataFrame(
        {
            **{
                f"signed_pmp_error_npe_{model}": [0.0]
                for model in ("m1", "m2", "m3", "m4")
            },
            "normalized_pmp_error_m1": [999.0],
        }
    )

    normalized, normalized_logml, attached_pmp = attach_calibrated_errors(
        posterior, logml, pmp, thresholds
    )

    np.testing.assert_allclose(
        normalized["normalized_posterior_mmd"], [0.25, 0.625, 1.0]
    )
    np.testing.assert_allclose(
        normalized["normalized_posterior_mmd_low"], [0.0] * 3
    )
    np.testing.assert_allclose(
        normalized["normalized_posterior_mmd_high"], [1.0] * 3
    )
    np.testing.assert_allclose(normalized_logml["normalized_logml_error_low"], [-1.0])
    np.testing.assert_allclose(normalized_logml["normalized_logml_error_high"], [2.0])
    for model in ("m1", "m2", "m3", "m4"):
        np.testing.assert_allclose(
            attached_pmp[f"pmp_error_lower_threshold_{model}"], [-0.05]
        )
        np.testing.assert_allclose(
            attached_pmp[f"pmp_error_upper_threshold_{model}"], [0.1]
        )
        assert not attached_pmp[f"pmp_error_degenerate_interval_{model}"].any()
    assert not any(column.startswith("normalized_pmp") for column in attached_pmp)


def test_attach_calibrated_errors_replaces_partial_legacy_fields():
    thresholds = pd.DataFrame(
        {
            "assumed_model": ["m1", "m2", "m3", "m4"],
            "posterior_mmd_lower_threshold": [0.0] * 4,
            "posterior_mmd_threshold": [0.8] * 4,
            "posterior_mmd_median": [0.2] * 4,
            "signed_logml_error_lower_threshold": [-1.0] * 4,
            "signed_logml_error_threshold": [2.0] * 4,
            "signed_logml_error_median": [0.0] * 4,
            "signed_pmp_error_lower_threshold": [-0.05] * 4,
            "signed_pmp_error_threshold": [0.1] * 4,
            "signed_pmp_error_degenerate_interval": [False] * 4,
        }
    )
    posterior = pd.DataFrame(
        {
            "assumed_model": ["m1"],
            "posterior_mmd": [0.4],
            "posterior_mmd_threshold": [999.0],
            "posterior_mmd_median": [999.0],
            "normalized_posterior_mmd": [999.0],
        }
    )
    logml = pd.DataFrame(
        {
            "assumed_model": ["m1"],
            "signed_logml_error": [1.0],
            "absolute_logml_error_threshold": [999.0],
            "absolute_logml_error_median": [999.0],
            "normalized_logml_error": [999.0],
        }
    )
    pmp = pd.DataFrame(
        {
            "pmp_error_threshold_m1": [999.0],
            "signed_pmp_error_median_m1": [999.0],
            "normalized_pmp_error_m1": [999.0],
        }
    )

    refreshed_posterior, refreshed_logml, refreshed_pmp = attach_calibrated_errors(
        posterior, logml, pmp, thresholds
    )

    assert refreshed_posterior.loc[0, "posterior_mmd_threshold"] == 0.8
    assert refreshed_posterior.loc[0, "posterior_mmd_median"] == 0.2
    assert refreshed_posterior.loc[0, "normalized_posterior_mmd"] == 0.5
    assert refreshed_logml.loc[0, "signed_logml_error_threshold"] == 2.0
    assert refreshed_logml.loc[0, "normalized_logml_error"] == 1.0
    assert "absolute_logml_error_threshold" not in refreshed_logml
    assert not any(column.endswith(("_x", "_y")) for column in refreshed_posterior)
    assert "pmp_error_threshold_m1" not in refreshed_pmp
    assert "normalized_pmp_error_m1" not in refreshed_pmp
    assert not refreshed_pmp["pmp_error_degenerate_interval_m1"].any()


def test_attach_thresholds_is_many_to_one_by_full_configuration():
    frame = _calibration_frame()
    thresholds = calculate_thresholds(frame)
    output = add_thresholds_to_metrics(frame, thresholds)

    assert len(output) == len(frame)
    assert output["posterior_mmd_threshold"].notna().all()
    assert output["signed_logml_error_median"].eq(0.0).all()
    assert output["signed_pmp_error_threshold"].nunique() == 1
    assert not output["signed_pmp_error_degenerate_interval"].any()


def test_threshold_quantile_is_validated():
    frame = _calibration_frame()
    for quantile in (0.0, 1.0):
        try:
            calculate_thresholds(frame, quantile=quantile)
        except ValueError as error:
            assert "between zero and one" in str(error)
        else:
            raise AssertionError("invalid quantile was accepted")
    for coverage in (0.0, 1.0):
        try:
            calculate_thresholds(frame, signed_error_coverage=coverage)
        except ValueError as error:
            assert "between zero and one" in str(error)
        else:
            raise AssertionError("invalid signed-error coverage was accepted")


def test_summary_mmd_reference_distance_is_finite_and_nonnegative():
    reference = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    observed = np.asarray([[0.25, 0.25], [4.0, 4.0]])
    bandwidth2 = mmd_rbf_bandwidth2(reference, max_pairs=100)
    distances = mmd_reference_distance_from_summary(
        observed, reference, bandwidth2, chunk_size=1
    )

    assert distances.shape == (2,)
    assert np.isfinite(distances).all()
    assert (distances >= 0.0).all()
    assert distances[1] > distances[0]


def test_thresholds_are_separate_for_each_summary_configuration():
    s1d = _calibration_frame()
    s2d = _calibration_frame().assign(
        network_tag="40d_10n", summary_label="S=2D", summary_dim=40
    )
    thresholds = calculate_thresholds(pd.concat([s1d, s2d], ignore_index=True))

    assert (
        len(thresholds) == 6
    )  # 2 summaries × fixture's one generating model × 3 metrics
    assert set(thresholds["summary_label"]) == {"S=1D", "S=2D"}
    assert thresholds.groupby(["summary_label", "generating_model"]).size().eq(3).all()


def test_network_discovery_reads_actual_summary_dimensions():
    network_sets = discover_network_sets(data_dim=20, num_obs=10)
    observed = {
        (network_set.summary_label, network_set.summary_dim, network_set.network_tag)
        for network_set in network_sets
    }

    assert {
        ("S=1D", 20, "20d_10n"),
        ("S=2D", 40, "40d_10n"),
        ("S=4D", 80, "80d_10n"),
    }.issubset(observed)


def test_calibration_paths_separate_shared_analytical_and_summary_npe(tmp_path):
    paths = CalibrationPaths(tmp_path)
    network_sets = discover_network_sets(data_dim=20, num_obs=10)
    by_label = {network_set.summary_label: network_set for network_set in network_sets}

    assert paths.analytical_draws == tmp_path / "analytical" / "posterior_draws"
    assert paths.npe_draws(by_label["S=1D"]) == (
        tmp_path / "metrics" / "S1D" / "npe_posterior_draws"
    )
    assert paths.npe_draws(by_label["S=4D"]) == (
        tmp_path / "metrics" / "S4D" / "npe_posterior_draws"
    )


def test_ood_paths_separate_inference_tables_from_diagnostic_tables(tmp_path):
    paths = OODPaths(tmp_path)

    assert paths.raw_dataset("m1") == tmp_path / "datasets" / "m1_raw.pkl"
    assert paths.inference_dataset("m1") == (tmp_path / "datasets" / "m1_logml_pmp.pkl")
    assert paths.posterior == tmp_path / "inference" / "posterior.csv"
    assert paths.logml == tmp_path / "inference" / "logml.csv"
    assert paths.pmp == tmp_path / "inference" / "pmp.csv"
    assert paths.diagnostic("l2") == tmp_path / "diagnostics" / "l2"


def test_collect_inference_frames_keeps_indirect_and_direct_pmp_separate():
    item = {"id": 0, "p_direct": np.asarray([0.4, 0.3, 0.2, 0.1])}
    for index, model in enumerate(("m1", "m2", "m3", "m4")):
        item[f"gold_log_marginal_{model}"] = -float(index)
        item[f"npe_log_marginal_{model}"] = -float(index) + 0.1
        item[f"gold_post_samples_{model}"] = np.zeros((4, 2), dtype=np.float32)
        item[f"npe_post_samples_{model}"] = np.full(
            (4, 2), 0.01 * (index + 1), dtype=np.float32
        )
    datasets = {
        source: [dict(item, source_model=source)]
        for source in (
            "m1",
            "m2",
            "m3",
            "m4",
            "m5",
            "m6",
            "m7",
            "m8",
            "m9",
            "m10",
            "m11",
            "m12",
        )
    }

    frames = _collect_inference_frames(datasets, n_posterior_samples=4)

    assert set(frames) == {"posterior", "logml", "pmp"}
    assert len(frames["posterior"]) == 12 * 4
    assert len(frames["logml"]) == 12 * 4
    assert len(frames["pmp"]) == 12
    assert {
        "p_npe_m1",
        "p_direct_m1",
        "signed_pmp_error_npe_m1",
        "signed_pmp_error_direct_m1",
    }.issubset(frames["pmp"].columns)


def test_load_cached_inference_results_reads_only_saved_tables(tmp_path):
    paths = OODPaths(tmp_path)
    paths.inference.mkdir(parents=True)
    expected = pd.DataFrame({"id": [0], "value": [1.0]})
    for path in (paths.posterior, paths.logml, paths.pmp):
        expected.to_csv(path, index=False)

    loaded = load_cached_inference_results(tmp_path)

    assert set(loaded) == {"posterior", "logml", "pmp"}
    for frame in loaded.values():
        pd.testing.assert_frame_equal(frame, expected)


def test_metric_loader_supports_modern_diagnostic_layout(tmp_path):
    result_dir = tmp_path / "ood_20d_10n"
    output_dir = result_dir / "diagnostics" / "mmd"
    output_dir.mkdir(parents=True)
    expected = pd.DataFrame({"id": [0], "value": [1.0]})
    for filename in (
        "posterior_distance_frame.csv",
        "logml_distance_frame.csv",
        "pmp_ambiguity_frame.csv",
    ):
        expected.to_csv(output_dir / filename, index=False)

    loaded = load_cached_metric_frames(result_dir, "mmd")

    for frame in loaded.values():
        pd.testing.assert_frame_equal(frame, expected)


def test_metric_loader_supports_legacy_l2_and_linf_layouts(tmp_path):
    result_dir = tmp_path / "ood_20d_10n"
    linf_dir = tmp_path / "ood_20d_10n_inf"
    result_dir.mkdir()
    linf_dir.mkdir()
    expected = pd.DataFrame({"id": [0], "value": [1.0]})
    file_stems = (
        "posterior_distance_frame",
        "logml_distance_frame",
        "pmp_ambiguity_frame",
    )
    for stem in file_stems:
        expected.to_csv(result_dir / f"{stem}_20d_10n.csv", index=False)
        expected.to_csv(linf_dir / f"{stem}_20d_10n_linf.csv", index=False)

    for metric in ("l2", "linf"):
        loaded = load_cached_metric_frames(result_dir, metric)
        assert loaded["posterior"]["dm_median"].eq(0.0).all()
        assert loaded["logml"]["dm_median"].eq(0.0).all()
        for model in ("m1", "m2", "m3", "m4"):
            assert loaded["pmp"][f"dm_median_{model}"].eq(0.0).all()
