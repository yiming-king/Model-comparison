from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmark.examples.diffusion.results import summary_dimension_comparison as comparison


def _threshold_table(dataset_count=100):
    return pd.DataFrame(
        [
            {"metric": metric, "lower_quantile": low, "upper_quantile": high, "n_values": count}
            for metric, low, high, count in (
                ("posterior_mmd", 0.0, 0.95, dataset_count),
                ("signed_logml_error", 0.05, 0.95, dataset_count),
                ("signed_pmp_error", 0.05, 0.95, 4 * dataset_count),
            )
        ]
    )


@pytest.mark.parametrize("variant", ["withMMD", "noMMD", "noMMD_rerun1"])
def test_notebook_workflow_keeps_training_run_and_figure_layout_together(
    tmp_path, monkeypatch, variant
):
    monkeypatch.setattr(comparison, "BASE_DIR", tmp_path)
    monkeypatch.setattr(comparison, "RESULT_DIR", tmp_path / "results")
    threshold_path = tmp_path / f"calibration_outputs_100_{variant}" / "thresholds.csv"
    threshold_path.parent.mkdir()
    _threshold_table().to_csv(threshold_path, index=False)
    original = threshold_path.read_bytes()
    comparison_calls, grid_calls = [], []

    def compare(**kwargs):
        comparison_calls.append(kwargs)
        return {"all_l2": {"output_root": kwargs["output_root"]}}

    def grid(metric, **kwargs):
        grid_calls.append((metric, kwargs))
        return {"png": Path(kwargs["output_root"]) / "grid.png"}

    monkeypatch.setattr(comparison, "run_calibrated_comparison_pipeline", compare)
    monkeypatch.setattr(comparison, "generate_calibrated_gold_pmp_lowess_grid", grid)
    result = comparison.run_diagnostic_notebook("l2", variant, refresh_thresholds=False)

    with_mmd = variant == "withMMD"
    assert threshold_path.read_bytes() == original
    assert not (tmp_path / "results").exists()
    assert result["threshold_path"] == threshold_path
    assert comparison_calls[0]["plot_style"] == ("standard" if with_mmd else "combined")
    assert comparison_calls[0]["y_symlog_linthresh"] == (1.0 if with_mmd else 0.1)
    configs = [config for _, config in comparison_calls[0]["summary_specs"]]
    assert [config.summary_multiplier for config in configs] == [1, 2, 4]
    assert {config.run_suffix for config in configs} == {None if with_mmd else variant}
    assert len(grid_calls) == (1 if with_mmd else 2)
    expected_grid_variant = {
        "withMMD": "with_mmd", "noMMD": "without_mmd", "noMMD_rerun1": "without_mmd_rerun1"
    }[variant]
    assert {options["output_variant"] for _, options in grid_calls} == {expected_grid_variant}
    # The comparison's logML scale must not change the independent PMP grid scale.
    assert all("y_symlog_linthresh" not in options for _, options in grid_calls)
    if not with_mmd:
        assert grid_calls[-1][1]["models"] == comparison.MODELS


def test_notebook_refuses_old_calibration_before_rendering(tmp_path, monkeypatch):
    threshold_path = tmp_path / "thresholds.csv"
    _threshold_table(dataset_count=30).to_csv(threshold_path, index=False)

    def unexpected_plot(**kwargs):
        pytest.fail("Plotting must not start with the wrong calibration sample")

    monkeypatch.setattr(comparison, "run_calibrated_comparison_pipeline", unexpected_plot)
    with pytest.raises(ValueError, match="100 calibration values"):
        comparison.run_diagnostic_notebook(
            "density", threshold_path=threshold_path, refresh_thresholds=False
        )


def test_notebook_threshold_validation_respects_requested_intervals():
    thresholds = _threshold_table()
    signed = thresholds["metric"].str.startswith("signed_")
    thresholds.loc[signed, "lower_quantile"] = 0.025
    thresholds.loc[signed, "upper_quantile"] = 0.975
    comparison._validate_diagnostic_notebook_thresholds(
        thresholds,
        posterior_mmd_quantile=0.95,
        signed_error_coverage=0.95,
        expected_calibration_datasets=100,
    )
    with pytest.raises(ValueError, match="signed_logml_error"):
        comparison._validate_diagnostic_notebook_thresholds(
            thresholds,
            posterior_mmd_quantile=0.95,
            signed_error_coverage=0.90,
            expected_calibration_datasets=100,
        )


@pytest.mark.parametrize("include_empirical", [False, True])
def test_combined_plot_keeps_empirical_points_when_excluded_from_loess(
    tmp_path, monkeypatch, include_empirical
):
    data = pd.DataFrame(
        {
            "model": ["m0"] * 5,
            "summary": ["S=D"] * 5,
            "dataset": ["m0", "m0", "contaminated_m0", "contaminated_m0", "empirical"],
            "model_matched": [True, True, False, False, False],
            "rho": [0.1, 0.2, 0.3, 0.4, 5.0],
            "rho_low": [0.0] * 5,
            "observed_mmd": [1.0, 2.0, 3.0, 4.0, 8.0],
        }
    )
    original = data.copy(deep=True)
    fits, figures = [], []

    def fit(x, y, frac):
        fits.append((x.to_numpy(), y.to_numpy()))
        return x.to_numpy(), y.to_numpy()

    monkeypatch.setattr(comparison, "_lowess_curve", fit)
    monkeypatch.setattr(
        comparison.plt.Figure, "savefig", lambda self, *args, **kwargs: figures.append(self)
    )
    comparison.plot_rho_error_overlay(
        data, ("m0",), "posterior_mmd", "simulated", tmp_path / "simulated" / "plot.png",
        diagnostic="density", overlay_order=("S=D",), plot_style="combined",
        include_empirical_in_loess=include_empirical, xscale="linear",
    )

    pd.testing.assert_frame_equal(data, original)
    np.testing.assert_allclose(fits[0][0], data["rho"] if include_empirical else data["rho"][:4])
    axes = figures[0].axes[0]
    offsets = np.concatenate([np.asarray(collection.get_offsets()) for collection in axes.collections])
    assert np.any(np.all(np.isclose(offsets, [5.0, 8.0]), axis=1))
    # Each simulated source has its own median, including contamination variants.
    medians = np.concatenate(
        [np.asarray(collection.get_offsets()) for collection in axes.collections if collection.get_zorder() == 4]
    )
    assert set(map(tuple, np.round(medians, 8))) == {(0.15, 1.5), (0.35, 3.5)}
