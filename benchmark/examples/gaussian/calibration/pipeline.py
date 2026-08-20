"""Generate well-specified Gaussian data and calibrate NPE-analytical errors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "benchmark.examples.gaussian.calibration"

from ..config import (
    ASSUMED_MODELS,
    CALIBRATION_ROOT,
    MODEL_SPECS,
    NETWORK_DIR,
    NetworkSet,
    configuration_tag,
    discover_network_sets,
)
from ..datasets.calculation import Calculation
from ..datasets.datasets import GetDatasets
from ..direct.calculator import softmax_stable
from .thresholds import (
    DEFAULT_POSTERIOR_MMD_QUANTILE,
    DEFAULT_SIGNED_ERROR_COVERAGE,
    add_thresholds_to_metrics,
    calculate_thresholds,
)


@dataclass(frozen=True)
class CalibrationPaths:
    root: Path

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def analytical_draws(self) -> Path:
        """Summary-independent analytical posterior draws."""
        return self.root / "analytical" / "posterior_draws"

    @property
    def metric_root(self) -> Path:
        """Summary-specific NPE draws and per-model metric caches."""
        return self.root / "metrics"

    def metric_config(self, network_set: NetworkSet) -> Path:
        return self.metric_root / network_set.summary_slug

    def npe_draws(self, network_set: NetworkSet) -> Path:
        return self.metric_config(network_set) / "npe_posterior_draws"

    @property
    def metrics(self) -> Path:
        return self.root / "per_dataset_metrics.csv"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds.csv"

    @property
    def results(self) -> Path:
        return self.root / "per_dataset_results.csv"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"


def dataset_seed(base_seed: int, model_index: int) -> int:
    return int(base_seed + 10_000 * model_index)


def generate_datasets(
    paths: CalibrationPaths,
    *,
    num_dims: int,
    num_obs: int,
    num_datasets: int,
    base_seed: int,
    overwrite: bool = False,
) -> None:
    """Generate and persist independent well-specified datasets for m1--m4."""
    paths.datasets.mkdir(parents=True, exist_ok=True)
    for model_index, model in enumerate(ASSUMED_MODELS):
        model_dir = paths.datasets / model
        x_path = model_dir / "x.npy"
        mu_path = model_dir / "mu.npy"
        manifest_path = model_dir / "manifest.csv"
        if all(path.exists() for path in (x_path, mu_path, manifest_path)) and not overwrite:
            shape = np.load(x_path, mmap_mode="r").shape
            expected = (num_datasets, num_obs, num_dims)
            if shape != expected:
                raise ValueError(
                    f"Existing calibration data at {x_path} has shape {shape}; "
                    f"expected {expected}. Use --overwrite-datasets to replace it."
                )
            print(f"Reusing calibration datasets: {model_dir}")
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        seed = dataset_seed(base_seed, model_index)
        spec = MODEL_SPECS[model]
        datasets = GetDatasets(
            obs_mu_prior_mean=spec["mu_prior_mean"],
            obs_mu_prior_std=spec["mu_prior_std"],
            num_dims=num_dims,
            num_obs=num_obs,
            obs_likelihood_std=spec["likelihood_std"],
            num_datasets=num_datasets,
            rng=np.random.default_rng(seed),
        ).get_datasets_normal()
        np.save(x_path, np.stack([item["x"] for item in datasets]))
        np.save(mu_path, np.stack([item["mu"] for item in datasets]))
        pd.DataFrame(
            {
                "dataset_id": [int(item["id"]) for item in datasets],
                "generating_model": model,
                "generation_seed": seed,
            }
        ).to_csv(manifest_path, index=False)
        print(f"Generated {num_datasets} datasets: {model_dir}")


def _load_datasets(paths: CalibrationPaths, model: str) -> list[dict]:
    model_dir = paths.datasets / model
    x_path = model_dir / "x.npy"
    if not x_path.exists():
        raise FileNotFoundError(
            f"Missing calibration data: {x_path}. Run the generate stage first."
        )
    x = np.load(x_path)
    return [
        {"id": int(index), "source_model": model, "x": observation}
        for index, observation in enumerate(x)
    ]


def load_approximators(
    network_set: NetworkSet,
) -> dict[str, object]:
    return {
        model: keras.saving.load_model(network_set.paths[model])
        for model in ASSUMED_MODELS
    }


def _calculations(
    approximators: dict[str, object],
    *,
    num_dims: int,
    num_obs: int,
    num_posterior_samples: int,
    logml_method: str,
    base_seed: int,
    generating_index: int,
) -> dict[str, Calculation]:
    output = {}
    for candidate_index, model in enumerate(ASSUMED_MODELS):
        spec = MODEL_SPECS[model]
        output[model] = Calculation(
            approximator=approximators[model],
            mu_prior_mean=spec["mu_prior_mean"],
            mu_prior_std=spec["mu_prior_std"],
            num_dims=num_dims,
            num_obs=num_obs,
            likelihood_std=spec["likelihood_std"],
            num_samples=num_posterior_samples,
            assumed_model=model,
            rng=np.random.default_rng(
                base_seed + 1_000_000 + 10_000 * generating_index + candidate_index
            ),
            logml_method=logml_method,
        )
    return output


def _save_matching_analytical_draw(
    paths: CalibrationPaths,
    generating_model: str,
    item: dict,
) -> None:
    output_dir = paths.analytical_draws / generating_model
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = int(item["id"])
    np.save(
        output_dir / f"{dataset_id:04d}.npy",
        np.asarray(item[f"gold_post_samples_{generating_model}"], dtype=np.float32),
    )


def _save_matching_npe_draw(
    paths: CalibrationPaths,
    network_set: NetworkSet,
    generating_model: str,
    item: dict,
) -> None:
    output_dir = paths.npe_draws(network_set) / generating_model
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = int(item["id"])
    np.save(
        output_dir / f"{dataset_id:04d}.npy",
        np.asarray(item[f"npe_post_samples_{generating_model}"], dtype=np.float32),
    )


def _compute_analytical_references(
    paths: CalibrationPaths,
    *,
    num_dims: int,
    num_obs: int,
    num_posterior_samples: int,
    logml_method: str,
    base_seed: int,
    save_posterior_draws: bool,
) -> dict[str, list[dict]]:
    """Compute one shared analytical reference bank for every summary network."""
    references = {}
    placeholder_approximators = {model: None for model in ASSUMED_MODELS}
    for generating_index, generating_model in enumerate(ASSUMED_MODELS):
        datasets = _load_datasets(paths, generating_model)
        calculations = _calculations(
            placeholder_approximators,
            num_dims=num_dims,
            num_obs=num_obs,
            num_posterior_samples=num_posterior_samples,
            logml_method=logml_method,
            base_seed=base_seed,
            generating_index=generating_index,
        )
        for candidate_model in ASSUMED_MODELS:
            calculations[candidate_model].normal_analytical(datasets)
        if save_posterior_draws:
            for item in datasets:
                _save_matching_analytical_draw(paths, generating_model, item)
        references[generating_model] = datasets
        print(f"Computed shared analytical references: {generating_model}")
    return references


def compute_metrics(
    paths: CalibrationPaths,
    network_sets: tuple[NetworkSet, ...],
    *,
    num_dims: int,
    num_obs: int,
    num_posterior_samples: int,
    logml_method: str,
    base_seed: int,
    save_posterior_draws: bool = True,
) -> pd.DataFrame:
    """Compute per-dataset NPE/analytical posterior, logML, and PMP errors."""
    configuration = configuration_tag(num_dims, num_obs)
    rows = []
    analytical_references = _compute_analytical_references(
        paths,
        num_dims=num_dims,
        num_obs=num_obs,
        num_posterior_samples=num_posterior_samples,
        logml_method=logml_method,
        base_seed=base_seed,
        save_posterior_draws=save_posterior_draws,
    )
    for summary_index, network_set in enumerate(network_sets):
        if network_set.data_dim != num_dims or network_set.num_obs != num_obs:
            raise ValueError(
                f"Network set {network_set.network_tag} expects D={network_set.data_dim}, "
                f"n={network_set.num_obs}; requested D={num_dims}, n={num_obs}"
            )
        approximators = load_approximators(network_set)
        posterior_mmd = MaximumMeanDiscrepancy(kernel="gaussian")
        for generating_index, generating_model in enumerate(ASSUMED_MODELS):
            datasets = [
                dict(item) for item in analytical_references[generating_model]
            ]
            calculations = _calculations(
                approximators,
                num_dims=num_dims,
                num_obs=num_obs,
                num_posterior_samples=num_posterior_samples,
                logml_method=logml_method,
                base_seed=base_seed + 100_000 * summary_index,
                generating_index=generating_index,
            )
            for candidate_index, candidate_model in enumerate(ASSUMED_MODELS):
                calculation = calculations[candidate_model]
                calculation.npe_estimation(
                    datasets,
                    seed=base_seed
                    + 2_000_000
                    + 100_000 * summary_index
                    + 10_000 * generating_index
                    + candidate_index,
                )

            for item in datasets:
                gold_logml = np.asarray(
                    [item[f"gold_log_marginal_{model}"] for model in ASSUMED_MODELS],
                    dtype=float,
                )
                npe_logml = np.asarray(
                    [item[f"npe_log_marginal_{model}"] for model in ASSUMED_MODELS],
                    dtype=float,
                )
                gold_pmp = softmax_stable(gold_logml)
                npe_pmp = softmax_stable(npe_logml)
                if save_posterior_draws:
                    _save_matching_npe_draw(
                        paths, network_set, generating_model, item
                    )

                for candidate_index, candidate_model in enumerate(ASSUMED_MODELS):
                    matching = candidate_model == generating_model
                    mmd_value = np.nan
                    if matching:
                        npe = tf.convert_to_tensor(
                            np.asarray(
                                item[f"npe_post_samples_{candidate_model}"],
                                dtype=np.float32,
                            )
                        )
                        analytical = tf.convert_to_tensor(
                            np.asarray(
                                item[f"gold_post_samples_{candidate_model}"],
                                dtype=np.float32,
                            )
                        )
                        mmd_value = float(posterior_mmd(npe, analytical))
                    signed_logml_error = float(
                        npe_logml[candidate_index] - gold_logml[candidate_index]
                    )
                    signed_pmp_error = float(
                        npe_pmp[candidate_index] - gold_pmp[candidate_index]
                    )
                    rows.append(
                        {
                            "configuration": configuration,
                            "network_tag": network_set.network_tag,
                            "summary_label": network_set.summary_label,
                            "summary_dim": network_set.summary_dim,
                            "generating_model": generating_model,
                            "dataset_id": int(item["id"]),
                            "candidate_model": candidate_model,
                            "num_dims": int(num_dims),
                            "num_obs": int(num_obs),
                            "num_posterior_samples": int(num_posterior_samples),
                            "logml_method": logml_method,
                            "matching_model": bool(matching),
                            "gold_standard": "analytical",
                            "posterior_mmd": mmd_value,
                            "npe_logml": float(npe_logml[candidate_index]),
                            "analytical_logml": float(gold_logml[candidate_index]),
                            "importance_ess": float(
                                item[f"importance_ess_{candidate_model}"]
                            ),
                            "num_npe_logml_draws": int(
                                item[f"num_npe_logml_draws_{candidate_model}"]
                            ),
                            "signed_logml_error": signed_logml_error,
                            "absolute_logml_error": abs(signed_logml_error),
                            "npe_pmp": float(npe_pmp[candidate_index]),
                            "analytical_pmp": float(gold_pmp[candidate_index]),
                            "signed_pmp_error": signed_pmp_error,
                            "absolute_pmp_error": abs(signed_pmp_error),
                        }
                    )
            model_frame = pd.DataFrame(
                rows[-len(datasets) * len(ASSUMED_MODELS) :]
            )
            cache = (
                paths.metric_config(network_set)
                / generating_model
                / "per_dataset_metrics.csv"
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            model_frame.to_csv(cache, index=False)
            print(
                f"Computed NPE-analytical metrics: {network_set.summary_label} "
                f"({network_set.network_tag}) / {generating_model}"
            )

    frame = pd.DataFrame(rows)
    paths.root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.metrics, index=False)
    return frame


def save_thresholds(
    paths: CalibrationPaths,
    frame: pd.DataFrame,
    quantile: float = DEFAULT_POSTERIOR_MMD_QUANTILE,
    signed_error_coverage: float = DEFAULT_SIGNED_ERROR_COVERAGE,
) -> pd.DataFrame:
    thresholds = calculate_thresholds(
        frame,
        quantile=quantile,
        signed_error_coverage=signed_error_coverage,
    )
    thresholds.to_csv(paths.thresholds, index=False)
    add_thresholds_to_metrics(frame, thresholds).to_csv(paths.results, index=False)
    return thresholds


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "metrics", "thresholds", "all"))
    parser.add_argument("--num-dims", type=int, default=20)
    parser.add_argument("--num-obs", type=int, default=10)
    parser.add_argument("--num-datasets", type=int, default=30)
    parser.add_argument("--num-posterior-samples", type=int, default=1000)
    parser.add_argument(
        "--logml-method", choices=("log_mean_exp", "mean_log"), default="log_mean_exp"
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=DEFAULT_POSTERIOR_MMD_QUANTILE,
        help="One-sided upper quantile for posterior MMD (default: 0.95).",
    )
    parser.add_argument(
        "--signed-error-coverage",
        type=float,
        default=DEFAULT_SIGNED_ERROR_COVERAGE,
        help="Central coverage for signed logML/PMP errors (default: 0.90).",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument(
        "--summary-dims",
        type=int,
        nargs="+",
        help="Actual DeepSet summary dimensions to calibrate; default: all available.",
    )
    parser.add_argument(
        "--network-tags",
        nargs="+",
        help="Optional saved-network tags, e.g. 20d_10n 40d_10n 80d_10n.",
    )
    parser.add_argument("--calibration-root", type=Path, default=CALIBRATION_ROOT)
    parser.add_argument("--overwrite-datasets", action="store_true")
    parser.add_argument("--no-save-posterior-draws", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_datasets <= 1 or args.num_posterior_samples <= 1:
        raise ValueError("num_datasets and num_posterior_samples must exceed one")
    configuration = configuration_tag(args.num_dims, args.num_obs)
    network_sets = discover_network_sets(
        args.network_dir,
        data_dim=args.num_dims,
        num_obs=args.num_obs,
        summary_dims=tuple(args.summary_dims) if args.summary_dims else None,
        network_tags=tuple(args.network_tags) if args.network_tags else None,
    )
    if not network_sets:
        raise FileNotFoundError(
            "No complete m1--m4 network sets match the requested data and summary "
            "dimensions. Inspect --network-dir, --num-dims, --num-obs, and "
            "--summary-dims."
        )
    paths = CalibrationPaths(args.calibration_root / configuration)
    paths.root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "configuration": configuration,
        "num_dims": args.num_dims,
        "num_obs": args.num_obs,
        "num_datasets": args.num_datasets,
        "num_posterior_samples": args.num_posterior_samples,
        "logml_method": args.logml_method,
        "threshold_quantile": args.quantile,
        "posterior_mmd_quantile": args.quantile,
        "signed_error_coverage": args.signed_error_coverage,
        "seed": args.seed,
        "gold_standard": "analytical",
        "output_layout": "diffusion_style_v2",
        "summary_networks": [
            {
                "network_tag": network_set.network_tag,
                "summary_label": network_set.summary_label,
                "summary_dim": network_set.summary_dim,
            }
            for network_set in network_sets
        ],
    }
    if args.stage in {"generate", "all"}:
        generate_datasets(
            paths,
            num_dims=args.num_dims,
            num_obs=args.num_obs,
            num_datasets=args.num_datasets,
            base_seed=args.seed,
            overwrite=args.overwrite_datasets,
        )
    if args.stage in {"metrics", "all"}:
        frame = compute_metrics(
            paths,
            network_sets,
            num_dims=args.num_dims,
            num_obs=args.num_obs,
            num_posterior_samples=args.num_posterior_samples,
            logml_method=args.logml_method,
            base_seed=args.seed,
            save_posterior_draws=not args.no_save_posterior_draws,
        )
    elif args.stage == "thresholds":
        if not paths.metrics.exists():
            raise FileNotFoundError(
                f"Missing metrics: {paths.metrics}. Run the metrics stage first."
            )
        frame = pd.read_csv(paths.metrics, keep_default_na=False)
    else:
        frame = None
    if args.stage in {"thresholds", "all"}:
        thresholds = save_thresholds(
            paths,
            frame,
            quantile=args.quantile,
            signed_error_coverage=args.signed_error_coverage,
        )
        print(f"Thresholds: {paths.thresholds} ({len(thresholds)} rows)")
        print(f"Metrics with thresholds: {paths.results}")
    if frame is not None and not frame.empty:
        metadata["num_datasets"] = int(
            frame.groupby("generating_model")["dataset_id"].nunique().min()
        )
        metadata["num_posterior_samples"] = int(
            pd.to_numeric(frame["num_posterior_samples"]).iloc[0]
        )
        metadata["logml_method"] = str(frame["logml_method"].iloc[0])
    else:
        x_path = paths.datasets / ASSUMED_MODELS[0] / "x.npy"
        if x_path.exists():
            metadata["num_datasets"] = int(np.load(x_path, mmap_mode="r").shape[0])
    with paths.metadata.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)


if __name__ == "__main__":
    main()
