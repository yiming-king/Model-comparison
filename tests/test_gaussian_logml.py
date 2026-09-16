"""Regression checks for the supported Gaussian marginal-likelihood estimator."""

import importlib
import sys

import numpy as np
import pytest

from benchmark.examples.gaussian.datasets.calculation import Calculation
from benchmark.examples.gaussian.npe.estimation import MarginalLikelihoodEstimator


def _estimator():
    return MarginalLikelihoodEstimator(
        approximator=None,
        mu=np.zeros((3, 1)),
        obs_data=np.zeros((1, 1)),
        mu_prior_mean=0.0,
        mu_prior_std=1.0,
        num_dims=1,
        likelihood_std=1.0,
    )


@pytest.mark.parametrize("offset", [0.0, 1000.0, -1000.0])
def test_logml_uses_stable_log_of_mean_weight_and_preserves_ess(monkeypatch, offset):
    estimator = _estimator()
    weights = np.array([1.0, 3.0, 5.0])
    monkeypatch.setattr(estimator, "log_prior_mu", lambda: np.zeros(3))
    monkeypatch.setattr(
        estimator, "log_likelihood_x_given_mu", lambda: np.log(weights) + offset
    )
    monkeypatch.setattr(estimator, "log_q_phi", lambda: np.zeros(3))

    expected = np.log(weights.mean()) + offset
    assert estimator.log_marginal_npe() == pytest.approx(expected)
    assert estimator.log_marginal_npe(method="log_mean_exp") == pytest.approx(expected)
    assert estimator.importance_ess == pytest.approx(
        weights.sum() ** 2 / np.square(weights).sum()
    )
    assert estimator.num_importance_samples == 3


def test_removed_method_is_rejected_before_density_evaluation(monkeypatch):
    estimator = _estimator()

    def unexpected_evaluation():
        pytest.fail("Unsupported methods must fail before model evaluation")

    monkeypatch.setattr(estimator, "log_prior_mu", unexpected_evaluation)
    with pytest.raises(ValueError, match="log_mean_exp"):
        estimator.log_marginal_npe(method="mean_log")


def test_calculation_rejects_removed_method_before_posterior_sampling():
    with pytest.raises(ValueError, match="log_mean_exp"):
        Calculation(
            approximator=None,
            mu_prior_mean=0.0,
            mu_prior_std=1.0,
            num_dims=1,
            num_obs=1,
            likelihood_std=1.0,
            num_samples=3,
            assumed_model="m1",
            logml_method="mean_log",
        )


def test_diagnostic_helper_rejects_removed_method():
    from benchmark.examples.gaussian.analysis.summry_diagnostic import (
        compute_logml_and_posteriors,
    )

    with pytest.raises(ValueError, match="log_mean_exp"):
        compute_logml_and_posteriors({}, {}, logml_method="mean_log")


@pytest.mark.parametrize("pipeline", ["analysis", "calibration"])
def test_pipeline_cli_keeps_standard_method_and_rejects_removed_method(
    monkeypatch, pipeline
):
    module = importlib.import_module(f"benchmark.examples.gaussian.{pipeline}.pipeline")
    monkeypatch.setattr(
        sys, "argv", ["gaussian", "all", "--logml-method", "log_mean_exp"]
    )
    assert module._parse_args().logml_method == "log_mean_exp"
    monkeypatch.setattr(sys, "argv", ["gaussian", "all", "--logml-method", "mean_log"])
    with pytest.raises(SystemExit) as error:
        module._parse_args()
    assert error.value.code == 2
