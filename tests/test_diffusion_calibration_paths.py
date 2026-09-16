from dataclasses import replace

import pandas as pd
import pytest

from benchmark.examples.diffusion.config import TrainingConfig
from benchmark.examples.diffusion.results import summary_dimension_comparison as comparison


@pytest.mark.parametrize(
    ("distribution", "suffix", "variant"),
    [
        ("normal", None, "withMMD"),
        (None, "noMMD", "noMMD"),
        (None, "noMMD_rerun1", "noMMD_rerun1"),
    ],
)
def test_threshold_loader_selects_matching_100_dataset_run(
    tmp_path, monkeypatch, distribution, suffix, variant
):
    monkeypatch.setattr(comparison, "BASE_DIR", tmp_path)
    config = TrainingConfig(
        summary_multiplier=1,
        summary_base_distribution=distribution,
        run_suffix=suffix,
    )
    path = tmp_path / f"calibration_outputs_100_{variant}" / "thresholds.csv"
    path.parent.mkdir()
    pd.DataFrame(
        [
            {
                "generating_model": "m0",
                "npe_configuration": config.summary_label,
                "metric": metric,
                "lower_threshold": lower,
                "threshold": upper,
                "median": median,
            }
            for metric, lower, upper, median in (
                ("posterior_mmd", 0.0, 2.0, 1.0),
                ("signed_logml_error", -2.0, 3.0, 0.0),
                ("signed_pmp_error", -0.1, 0.2, 0.0),
            )
        ]
    ).to_csv(path, index=False)

    result = comparison.load_calibration_thresholds((("S=D", config),))

    assert result["npe_configuration"].tolist() == [config.summary_label]
    assert result["posterior_mmd_upper_threshold"].tolist() == [2.0]
    custom = replace(config, run_suffix="custom")
    if distribution is not None:
        with pytest.raises(ValueError, match="Pass threshold_path"):
            comparison.load_calibration_thresholds((("S=D", custom),))


def test_default_summary_specs_match_complete_calibration_dimensions():
    for specs in (comparison.WITH_MMD_SUMMARY_SPECS, comparison.NO_MMD_SUMMARY_SPECS):
        assert [config.summary_multiplier for _, config in specs] == [1, 2, 4]
    assert comparison._make_summary_specs(summary_multipliers=(6,))[0][1].summary_multiplier == 6
