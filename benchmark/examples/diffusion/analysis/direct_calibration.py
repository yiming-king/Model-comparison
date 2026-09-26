"""Calibrate direct PMP error on the retained diffusion benchmark and MCMC gold.

The same saved observations and all-candidate convergence filter used by the
indirect benchmark are reused. Direct classifiers have no posterior/logML errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..calibration.thresholds import DEFAULT_SIGNED_ERROR_COVERAGE
from ..config import BASE_DIR, MODEL_LABELS, MODELS, RESULT_DIR
from ...direct_pmp_calibration import calibrate_direct_pmp_signed, load_indirect_gold


LOSSES = ("cross_entropy", "exponential", "logistic")
CHECKPOINT_NAMES = {"cross_entropy": "cent", "exponential": "exponential", "logistic": "logistic"}
DEFAULT_BENCHMARK_DIR = BASE_DIR / "calibration_reference_100"
DEFAULT_GOLD_METRICS = BASE_DIR / "calibration_outputs_100_withMMD" / "per_dataset_metrics.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if isinstance(value, pd.DataFrame):
        value.to_csv(temporary, index=False)
    else:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _as_bool(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip().str.lower()
    if not text.isin(["true", "t", "1", "false", "f", "0"]).all():
        raise ValueError("Convergence flags must be explicit booleans")
    return text.isin(["true", "t", "1"])


def calculate_direct_pmp_thresholds(
    metrics: pd.DataFrame,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
) -> pd.DataFrame:
    """Calibrate a signed 90% interval for each candidate PMP component."""
    return calibrate_direct_pmp_signed(
        metrics, MODELS, "gold_pmp", signed_error_coverage
    )[1]


def _benchmark_inputs(root: Path) -> list[Path]:
    paths = (
        [root / "dataset_manifest.csv"]
        if (root / "dataset_manifest.csv").exists()
        else []
    )
    for model in MODELS:
        paths.append(root / "datasets" / model / "true_parameters.csv")
        paths.extend(sorted((root / "datasets" / model).glob("*.json")))
        for candidate in MODELS:
            paths.extend(
                root / "mcmc" / model / candidate / name
                for name in (
                    "bridgesampling.csv",
                    "convergence_diagnostics.csv",
                )
            )
    return paths


def _load_benchmark(root: Path, num_obs: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Preserve manifest IDs and load matching existing bridge/convergence rows."""
    observations, rows = [], []
    for model in MODELS:
        directory = root / "datasets" / model
        manifest = pd.read_csv(directory / "true_parameters.csv", dtype={"id": str})
        if (
            manifest.empty
            or manifest["id"].isna().any()
            or manifest["id"].duplicated().any()
            or set(manifest["id"]) != {p.stem for p in directory.glob("*.json")}
        ):
            raise ValueError(f"Benchmark manifest/JSON IDs mismatch: {directory}")
        gold = {}
        for candidate in MODELS:
            folder = root / "mcmc" / model / candidate
            bridge = pd.read_csv(folder / "bridgesampling.csv", dtype={"id": str})
            convergence = pd.read_csv(
                folder / "convergence_diagnostics.csv", dtype={"id": str}
            )
            for table in (bridge, convergence):
                if table["id"].duplicated().any() or set(table["id"]) != set(
                    manifest["id"]
                ):
                    raise ValueError(
                        f"Incomplete or duplicate MCMC benchmark IDs: {folder}"
                    )
            table = bridge[["id", "estimate"]].merge(
                convergence[["id", "converged"]],
                on="id",
                validate="one_to_one",
            )
            table["converged"] = _as_bool(table["converged"])
            if not np.isfinite(table["estimate"]).all():
                raise ValueError(f"Nonfinite benchmark log evidence: {folder}")
            gold[candidate] = table.set_index("id")
        for item in manifest.to_dict("records"):
            dataset_id = item["id"]
            record = json.loads((directory / f"{dataset_id}.json").read_text())
            x = np.stack([record["rt"], record["condition"]], axis=-1).astype(
                np.float32
            )
            if (
                x.shape != (num_obs, 2)
                or not np.isfinite(x).all()
                or int(record["N"]) != num_obs
            ):
                raise ValueError(
                    f"Invalid benchmark observation shape/content: {model}/{dataset_id}"
                )
            if not np.isin(x[:, 1], [0, 1]).all():
                raise ValueError(f"Invalid binary conditions: {model}/{dataset_id}")
            observations.append(x)
            rows.append(
                {
                    "generating_model": model,
                    "dataset_id": dataset_id,
                    "dataset_seed": item["dataset_seed"],
                    **{
                        f"bridge_log_ml_{m}": float(gold[m].loc[dataset_id, "estimate"])
                        for m in MODELS
                    },
                    **{
                        f"converged_{m}": bool(gold[m].loc[dataset_id, "converged"])
                        for m in MODELS
                    },
                    "all_candidate_models_converged": all(
                        bool(gold[m].loc[dataset_id, "converged"]) for m in MODELS
                    ),
                }
            )
    index = pd.DataFrame(rows)
    main_manifest = root / "dataset_manifest.csv"
    if main_manifest.exists():
        manifest = pd.read_csv(main_manifest, dtype={"id": str}).rename(
            columns={"id": "dataset_id"}
        )
        keys = ["generating_model", "dataset_id"]
        if manifest.duplicated(keys).any():
            raise ValueError("Duplicate keys in the shared benchmark manifest")
        expected = index.set_index(keys)["dataset_seed"].sort_index()
        actual = manifest.set_index(keys)["dataset_seed"].sort_index()
        if not expected.index.equals(actual.index) or not np.array_equal(
            expected.to_numpy(), actual.to_numpy()
        ):
            raise ValueError(
                "Shared benchmark manifest disagrees with model dataset IDs/seeds"
            )
    return np.stack(observations), index


def _checkpoint_metadata(path: Path, loss: str) -> tuple[int, int]:
    with zipfile.ZipFile(path) as archive:
        config = json.loads(archive.read("config.json"))
    if config["class_name"] != "ModelComparisonApproximator":
        raise ValueError(f"Expected a trained direct classifier: {path}")
    expected_rule = {
        "cross_entropy": "CrossEntropyScore",
        "exponential": "ExponentialScore",
        "logistic": "LogisticScore",
    }[loss]
    rules = config["config"]["inference_network"]["config"]["scoring_rules"]
    if len(rules) != 1 or {rule.get("class_name") for rule in rules.values()} != {
        expected_rule
    }:
        raise ValueError(
            f"Checkpoint scoring rule does not match requested direct loss {loss}: {path}"
        )
    shapes = config["build_config"]["input_shape"]
    if shapes["summary_variables"][-1] != 2 or shapes["inference_variables"][-1] != 4:
        raise ValueError(
            "Diffusion direct checkpoint must accept rt/condition and four candidates"
        )
    return int(shapes["summary_variables"][1]), int(
        config["config"]["summary_network"]["config"]["summary_dim"]
    )


def ensure_direct_pmp_thresholds(
    loss: str,
    *,
    results_dir: str | Path = RESULT_DIR / "direct",
    benchmark_dir: str | Path = DEFAULT_BENCHMARK_DIR,
    gold_metrics_path: str | Path = DEFAULT_GOLD_METRICS,
    checkpoint: str | Path | None = None,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
    expected_per_model: int = 100,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Evaluate saved direct networks; never generate data, run MCMC, or train."""
    if loss not in LOSSES:
        raise ValueError(f"Unknown direct loss: {loss}")
    if not 0 < signed_error_coverage < 1:
        raise ValueError("signed_error_coverage must be between zero and one")
    checkpoint = Path(
        checkpoint or BASE_DIR / "approximators" / "trained" / "direct" / f"direct_{CHECKPOINT_NAMES[loss]}.keras"
    ).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Saved diffusion direct checkpoint is missing: {checkpoint}. Train/save this loss in its direct notebook first."
        )
    benchmark_dir = Path(benchmark_dir).resolve()
    gold_metrics_path = Path(gold_metrics_path).resolve()
    num_obs, summary_dim = _checkpoint_metadata(checkpoint, loss)
    output = Path(results_dir) / f"direct_{loss}" / "calibration"
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output / f"{name}.csv"
        for name in (
            "thresholds", "per_dataset_metrics", "per_dataset_results"
        )
    }
    metadata_path = output / "metadata.json"
    identity = {
        "schema_version": 3,
        "loss_function": loss,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "num_obs": num_obs,
        "summary_dim": summary_dim,
        "benchmark_directory": str(benchmark_dir),
        "indirect_gold_metrics": str(gold_metrics_path),
        "indirect_gold_sha256": _sha256(gold_metrics_path),
        "calibration_helper_sha256": _sha256(
            Path(__file__).parents[2] / "direct_pmp_calibration.py"
        ),
        "expected_per_model": expected_per_model,
        "benchmark_sha256": {
            str(path.relative_to(benchmark_dir)): _sha256(path)
            for path in _benchmark_inputs(benchmark_dir)
        },
        "model_order": list(MODELS),
        "model_prior": [0.25] * 4,
        "signed_error_coverage": signed_error_coverage,
        "implementation_sha256": _sha256(Path(__file__)),
    }
    if (
        not overwrite
        and metadata_path.exists()
        and all(path.exists() for path in paths.values())
    ):
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") == identity and metadata.get("output_sha256") == {
            name: _sha256(path) for name, path in paths.items()
        }:
            return pd.read_csv(paths["thresholds"])

    import bayesflow  # noqa: F401
    import keras

    observations, index = _load_benchmark(benchmark_dir, num_obs)
    counts = index.groupby("generating_model").size()
    if set(counts.index) != set(MODELS) or not counts.eq(expected_per_model).all():
        raise ValueError(f"Expected {expected_per_model} saved datasets per model")
    if not index["all_candidate_models_converged"].all():
        raise ValueError("All saved benchmark datasets must have converged gold PMPs")
    gold = load_indirect_gold(gold_metrics_path, index, MODELS, "gold_pmp")
    approximator = keras.saving.load_model(checkpoint, compile=False)
    probabilities = np.asarray(
        approximator.estimate(
            conditions={
                "rt": observations[..., 0],
                "conditions": observations[..., 1],
            }
        )["model_probs"],
        dtype=float,
    )
    if (
        probabilities.shape != (len(index), len(MODELS))
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6)
    ):
        raise ValueError(
            "Invalid direct model probabilities on the diffusion benchmark"
        )
    frames = []
    for j, model in enumerate(MODELS):
        frame = index[
            [
                "generating_model",
                "dataset_id",
                "dataset_seed",
                "all_candidate_models_converged",
            ]
        ].copy()
        frame["candidate_model"], frame["loss_function"] = model, loss
        frame["network_tag"] = f"direct_{loss}"
        frame["generating_model_label"] = frame["generating_model"].map(MODEL_LABELS)
        frame["model_prior"], frame["summary_dim"], frame["num_obs"] = (
            0.25,
            summary_dim,
            num_obs,
        )
        frame["bridge_log_ml"] = index[f"bridge_log_ml_{model}"]
        frame["converged"] = index[f"converged_{model}"]
        frame["gold_pmp"] = gold.loc[
            gold.candidate_model.eq(model), "gold_pmp"
        ].to_numpy()
        frame["direct_pmp"] = probabilities[:, j]
        frame["signed_pmp_error"] = frame["direct_pmp"] - frame["gold_pmp"]
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    errors, thresholds = calibrate_direct_pmp_signed(
        metrics, MODELS, "gold_pmp", signed_error_coverage
    )
    if len(errors) != expected_per_model * len(MODELS) ** 2:
        raise ValueError("PMP calibration component count changed")
    attached = metrics.merge(
        thresholds[["loss_function", "candidate_model", "lower_threshold", "threshold"]],
        on=["loss_function", "candidate_model"],
        how="left",
        validate="many_to_one",
    ).rename(columns={"lower_threshold": "signed_pmp_error_lower_threshold",
                      "threshold": "signed_pmp_error_upper_threshold"})
    for name, frame in (
        ("thresholds", thresholds),
        ("per_dataset_metrics", metrics),
        ("per_dataset_results", attached),
    ):
        _write(paths[name], frame)
    _write(
        metadata_path,
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "output_sha256": {name: _sha256(path) for name, path in paths.items()},
            "num_datasets": len(index),
            "datasets_per_generating_model": index.groupby("generating_model")
            .size()
            .to_dict(),
            "converged_datasets_per_generating_model": index.groupby(
                "generating_model"
            )["all_candidate_models_converged"]
            .sum()
            .to_dict(),
            "dataset_policy": "exact saved indirect benchmark observations reused without regeneration",
            "gold_standard": "indirect calibration's saved MCMC bridge-sampling PMP vector",
            "aggregation": "one signed PMP error interval per candidate model across all saved datasets",
            "not_applicable": ["posterior_mmd", "signed_logml_error"],
            "not_applicable_reason": "Direct classifiers estimate model probabilities, not parameter posteriors or absolute evidence",
        },
    )
    print(
        f"Calibrated {loss} on {len(index)} saved diffusion datasets: {paths['thresholds']}",
        flush=True,
    )
    return thresholds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=LOSSES, action="append")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for loss in args.loss or LOSSES:
        ensure_direct_pmp_thresholds(loss, overwrite=args.overwrite)
