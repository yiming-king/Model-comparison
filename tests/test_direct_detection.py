"""Checks for detecting candidate-specific signed PMP errors from pooled scores."""

import numpy as np
import pandas as pd

from benchmark.examples.direct_detection import evaluate_direct_detection


def test_detection_counts_and_roc_use_one_score_per_dataset():
    amplitudes = np.array([0.1, 0.5, 0.8, 0.2])
    comparison = pd.DataFrame({
        "source_model": ["m5"] * 4,
        "id": [0, 1, 2, 3],
    }).set_index(["source_model", "id"])
    for model, gold, sign in (("m1", 0.5, 1), ("m2", 0.5, -1),
                              ("m3", 0.0, 0), ("m4", 0.0, 0)):
        comparison[f"p_gold_{model}"] = gold
        comparison[f"p_direct_{model}"] = gold + sign * amplitudes / 2
    diagnostics = pd.DataFrame([
        {"source_model": "m5", "id": i, "diagnostic": method,
         "loss_function": "cross_entropy", "rho": rho, "rho_low": 0.1}
        for method in ("l2", "linf", "density", "mmd")
        for i, rho in enumerate((0.2, 1.2, 0.8, 1.5))
    ])
    calibration = pd.DataFrame([
        {"loss_function": "cross_entropy", "candidate_model": model,
         "metric": "signed_pmp_error", "lower_quantile": 0.05,
         "upper_quantile": 0.95, "n_values": 120,
         "lower_threshold": -0.2, "threshold": 0.2}
        for model in ("m1", "m2", "m3", "m4")
    ])
    data, metrics = evaluate_direct_detection(
        diagnostics, comparison, calibration,
        keys=("source_model", "id"), loss="cross_entropy",
    )
    assert len(data) == 16
    np.testing.assert_allclose(data["signed_pmp_error_m1"].to_numpy(),
                               np.tile([0.05, 0.25, 0.4, 0.1], 4))
    assert set(metrics["n_high_error"]) == {2}
    assert set(metrics[["TP", "FP", "TN", "FN"]].itertuples(index=False, name=None)) == {(1, 1, 1, 1)}
    np.testing.assert_allclose(metrics[["FNR", "FPR", "ROC_AUC"]], 0.5)
