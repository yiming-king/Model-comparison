"""Calibrate direct PMP errors on the saved indirect benchmark observations.

Direct model-comparison networks estimate model probabilities, not parameter
posteriors or absolute marginal likelihoods. Only PMP error is calibrated here.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

from ..analytic.analytic import GaussianAnalytical
from ..calibration.thresholds import DEFAULT_SIGNED_ERROR_COVERAGE
from ..config import ASSUMED_MODELS, CALIBRATION_ROOT, MODEL_SPECS, NETWORK_DIR, RESULT_DIR
from .direct_diagnostics import LOSSES, _checkpoint_config, _write_csv, _write_json, file_sha256


def calculate_direct_pmp_thresholds(
    metrics: pd.DataFrame,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
) -> pd.DataFrame:
    """Match the indirect protocol: pool four PMP components per generator."""
    if not 0 < signed_error_coverage < 1:
        raise ValueError("signed_error_coverage must be between zero and one")
    required = {"loss_function", "generating_model", "dataset_id", "candidate_model", "signed_pmp_error"}
    if required - set(metrics):
        raise ValueError(f"Missing calibration columns: {sorted(required - set(metrics))}")
    keys = ["loss_function", "generating_model", "dataset_id", "candidate_model"]
    if metrics.empty or metrics[keys].isna().any().any() or metrics.duplicated(keys).any():
        raise ValueError("Calibration rows must have unique nonempty dataset/component keys")
    if set(metrics["generating_model"]) != set(ASSUMED_MODELS):
        raise ValueError("Calibration requires all four generating models")
    if not np.isfinite(metrics["signed_pmp_error"]).all():
        raise ValueError("Calibration errors must be finite")
    components = metrics.groupby(keys[:-1])["candidate_model"].agg(set)
    if not components.map(lambda value: value == set(ASSUMED_MODELS)).all():
        raise ValueError("Each dataset must contain all four PMP components")
    lower_q = (1 - signed_error_coverage) / 2
    rows = []
    for (loss, model), group in metrics.groupby(["loss_function", "generating_model"], sort=False):
        values = group["signed_pmp_error"].to_numpy(dtype=float)
        low, median, high = np.quantile(values, [lower_q, 0.5, 1 - lower_q], method="linear")
        rows.append({
            "loss_function": loss, "network_tag": f"direct_{loss}",
            "generating_model": model, "metric": "signed_pmp_error",
            "quantile": signed_error_coverage, "lower_quantile": lower_q,
            "upper_quantile": 1 - lower_q, "quantile_method": "linear",
            "lower_threshold": low, "threshold": high, "median": median,
            "degenerate_interval": bool(low == high), "n_values": len(values),
            "num_datasets": group["dataset_id"].nunique(),
            "aggregation": "all_four_pmp_components_from_well_specified_datasets",
            "gold_standard": "analytical",
        })
    return pd.DataFrame(rows)


def _load_benchmark(root: Path, num_obs: int, data_dim: int):
    observations, index = [], []
    for model in ASSUMED_MODELS:
        directory = root / "datasets" / model
        x = np.load(directory / "x.npy", allow_pickle=False)
        manifest = pd.read_csv(directory / "manifest.csv")
        if x.ndim != 3 or x.shape[1:] != (num_obs, data_dim) or len(x) != len(manifest):
            raise ValueError(f"Benchmark shape/manifest mismatch: {directory}")
        if not np.isfinite(x).all() or len(x) < 2:
            raise ValueError(f"Benchmark observations must be finite and nonempty: {directory}")
        if manifest.dataset_id.isna().any() or manifest.dataset_id.duplicated().any():
            raise ValueError(f"Benchmark IDs must be unique: {directory}")
        if not manifest.generating_model.eq(model).all():
            raise ValueError(f"Benchmark generator mismatch: {directory}")
        observations.append(x)
        index.append(manifest[["generating_model", "dataset_id"]].copy())
    return np.concatenate(observations), pd.concat(index, ignore_index=True)


def _analytical_probabilities(observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logml = np.array([
        [GaussianAnalytical(
            observation, num_dims=observation.shape[1], num_obs=observation.shape[0],
            num_samples=0, **MODEL_SPECS[model],
        ).log_marginal_analytical() for model in ASSUMED_MODELS]
        for observation in observations
    ])
    return softmax(logml, axis=1), logml


def ensure_direct_pmp_thresholds(
    loss: str, *, results_dir: str | Path = RESULT_DIR / "direct",
    benchmark_dir: str | Path = CALIBRATION_ROOT / "20d_10n",
    checkpoint: str | Path | None = None,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Reuse the exact saved benchmark; evaluate the current direct checkpoint."""
    if loss not in LOSSES:
        raise ValueError(f"Unknown direct loss: {loss}")
    if not 0 < signed_error_coverage < 1:
        raise ValueError("signed_error_coverage must be between zero and one")
    benchmark_dir = Path(benchmark_dir).resolve()
    checkpoint = Path(checkpoint or NETWORK_DIR / f"direct_{loss}.keras").resolve()
    num_obs, data_dim, summary_dim = _checkpoint_config(checkpoint)
    output = Path(results_dir) / f"direct_{loss}" / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / f"{name}.csv" for name in ("thresholds", "per_dataset_metrics", "per_dataset_results")}
    metadata_path = output / "metadata.json"
    identity = {
        "schema_version": 1, "loss_function": loss,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "summary_dim": summary_dim, "data_dim": data_dim, "num_obs": num_obs,
        "benchmark_directory": str(benchmark_dir),
        "benchmark_sha256": {
            str(path.relative_to(benchmark_dir)): file_sha256(path)
            for model in ASSUMED_MODELS
            for path in (benchmark_dir / "datasets" / model / "x.npy", benchmark_dir / "datasets" / model / "manifest.csv")
        },
        "model_specs": {model: MODEL_SPECS[model] for model in ASSUMED_MODELS},
        "model_order": list(ASSUMED_MODELS), "model_prior": [0.25] * 4,
        "signed_error_coverage": signed_error_coverage,
        "implementation_sha256": file_sha256(Path(__file__)),
        "analytical_implementation_sha256": file_sha256(Path(__file__).parents[1] / "analytic" / "analytic.py"),
        "inference_helper_sha256": file_sha256(Path(__file__).with_name("direct_diagnostics.py")),
    }
    if not overwrite and metadata_path.exists() and all(path.exists() for path in paths.values()):
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("identity") == identity
                and metadata.get("output_sha256") == {name: file_sha256(path) for name, path in paths.items()}):
            return pd.read_csv(paths["thresholds"])

    import bayesflow  # noqa: F401
    import keras

    observations, index = _load_benchmark(benchmark_dir, num_obs, data_dim)
    gold, gold_logml = _analytical_probabilities(observations)
    approximator = keras.saving.load_model(checkpoint, compile=False)
    direct = np.asarray(approximator.estimate(conditions={"x": observations.astype(np.float32)})["model_probs"], dtype=float)
    if (direct.shape != gold.shape or not np.isfinite(direct).all()
            or np.any(direct < 0) or np.any(direct > 1)
            or not np.allclose(direct.sum(axis=1), 1, rtol=0, atol=1e-6)):
        raise ValueError("Invalid direct model probabilities on the benchmark")
    frames = []
    for j, model in enumerate(ASSUMED_MODELS):
        frame = index.copy()
        frame["candidate_model"] = model
        frame["loss_function"] = loss
        frame["network_tag"] = f"direct_{loss}"
        frame["summary_dim"] = summary_dim
        frame["num_dims"], frame["num_obs"] = data_dim, num_obs
        frame["analytical_logml"] = gold_logml[:, j]
        frame["analytical_pmp"] = gold[:, j]
        frame["direct_pmp"] = direct[:, j]
        frame["signed_pmp_error"] = direct[:, j] - gold[:, j]
        frame["absolute_pmp_error"] = np.abs(direct[:, j] - gold[:, j])
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    thresholds = calculate_direct_pmp_thresholds(metrics, signed_error_coverage)
    attached = metrics.merge(
        thresholds[["loss_function", "generating_model", "lower_threshold", "threshold", "median"]],
        on=["loss_function", "generating_model"], how="left", validate="many_to_one",
    ).rename(columns={"lower_threshold": "signed_pmp_error_lower_threshold", "threshold": "signed_pmp_error_threshold", "median": "signed_pmp_error_median"})
    for name, frame in (("thresholds", thresholds), ("per_dataset_metrics", metrics), ("per_dataset_results", attached)):
        _write_csv(paths[name], frame)
    _write_json(metadata_path, {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
        "output_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "num_datasets": len(index), "datasets_per_generating_model": index.groupby("generating_model").size().to_dict(),
        "dataset_policy": "exact saved indirect benchmark observations reused without regeneration",
        "gold_standard": "analytical marginal likelihoods with equal model prior",
        "aggregation": "per generating model, pool all four signed PMP components; central 90% by default",
        "not_applicable": ["posterior_mmd", "signed_logml_error"],
        "not_applicable_reason": "ModelComparisonApproximator outputs model probabilities, not parameter posteriors or absolute marginal likelihoods",
    })
    print(f"Calibrated {loss} on {len(index)} saved benchmark datasets: {paths['thresholds']}", flush=True)
    return thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=LOSSES, action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for loss in args.loss or LOSSES:
        ensure_direct_pmp_thresholds(loss, overwrite=args.overwrite)
