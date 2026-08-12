import numpy as np
import pandas as pd

from benchmark.examples.diffusion.calibration.thresholds import calculate_thresholds
from benchmark.examples.diffusion.config import TrainingConfig
from benchmark.examples.diffusion.results.summary_dimension_comparison import (
    calculate_diagnostic_classification,
    load_calibration_thresholds,
)
from benchmark.examples.diffusion.results.results import pmp_from_log_marginals


def test_model_priors_are_used_in_pmp():
    pmp = pmp_from_log_marginals(
        np.zeros((1, 4)), {"m0": 0.1, "m1": 0.2, "m2": 0.3, "m3": 0.4}
    )
    np.testing.assert_allclose(pmp[0], [0.1, 0.2, 0.3, 0.4])


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
                    "generating_model_label": "M1",
                    "dataset_id": dataset,
                    "candidate_model": candidate,
                    "model_prior": 0.25,
                    "npe_configuration": "S1D",
                    "training_setting": "with_mmd",
                    "summary_multiplier": 1,
                    "converged": converged if candidate == "m0" else True,
                    "all_candidate_models_converged": converged,
                    "posterior_mmd": mmd if candidate == "m0" else np.nan,
                    "absolute_logml_error": logml if candidate == "m0" else 0.0,
                    "absolute_pmp_error": 0.1 * (candidate_index + 1),
                }
            )
    thresholds = calculate_thresholds(pd.DataFrame(rows)).set_index("metric")
    assert thresholds.loc["posterior_mmd", "threshold"] == 1.0
    assert thresholds.loc["absolute_logml_error", "threshold"] == 2.0
    assert thresholds.loc["absolute_pmp_error", "n_values"] == 4


def test_plot_threshold_loader_converts_logml_to_base_10(tmp_path):
    path = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [
            {
                "generating_model": "m0",
                "npe_configuration": "S1D_noMMD",
                "metric": metric,
                "threshold": threshold,
            }
            for metric, threshold in (
                ("posterior_mmd", 0.25),
                ("absolute_logml_error", np.log(10.0)),
                ("absolute_pmp_error", 0.05),
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
    assert thresholds["posterior_mmd_threshold"] == 0.25
    assert thresholds["log10_logml_error_threshold"] == 1.0
    assert thresholds["pmp_error_threshold"] == 0.05


def test_calibrated_classification_uses_shared_error_definitions():
    data = pd.DataFrame(
        {
            "dataset": ["simulated_from_m0"] * 4,
            "id": range(4),
            "model": ["m0"] * 4,
            "summary": ["S=D"] * 4,
            "rho": [2.0, 2.0, 0.5, 0.5],
            "normalized_mmd": [2.0, 0.5, 2.0, 0.5],
            "logml_error": [-2.0, 0.5, -2.0, 0.5],
            "pmp_error": [0.2, 0.05, -0.2, 0.05],
        }
    )

    metrics = calculate_diagnostic_classification(data, "l2")

    assert set(metrics["error_threshold"]) == {0.1, 1.0}
    assert (metrics[["tp", "fp", "fn", "tn"]] == 1).all().all()
    np.testing.assert_allclose(metrics["false_positive_rate"], 0.5)
    np.testing.assert_allclose(metrics["false_negative_rate"], 0.5)
    np.testing.assert_allclose(metrics["f1"], 0.5)
