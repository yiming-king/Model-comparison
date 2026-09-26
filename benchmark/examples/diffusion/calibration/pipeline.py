"""Run isolated, well-specified posterior/logML/PMP calibration."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "benchmark.examples.diffusion.calibration"

from ..config import BASE_DIR, MODEL_LABELS, MODELS, TrainingConfig
from .thresholds import (
    CALIBRATION_VARIANTS,
    DEFAULT_CALIBRATION_ROOT,
    DEFAULT_CALIBRATION_VARIANT,
    DEFAULT_POSTERIOR_MMD_QUANTILE,
    DEFAULT_SIGNED_ERROR_COVERAGE,
    add_thresholds_to_metrics,
    calculate_thresholds,
)


DEFAULT_OUTPUT_ROOT = DEFAULT_CALIBRATION_ROOT
DEFAULT_REFERENCE_ROOT = BASE_DIR / "calibration_reference_100"
DEFAULT_DATASETS_PER_MODEL = 100
GOLD_POSTERIOR_DRAWS = 2048
MMD_DRAWS = 1024
NPE_LOGML_DRAWS = 2048


@dataclass(frozen=True)
class CalibrationPaths:
    root: Path = DEFAULT_OUTPUT_ROOT
    reference_root: Path | None = DEFAULT_REFERENCE_ROOT

    @property
    def inputs(self) -> Path:
        return self.reference_root or self.root

    @property
    def datasets(self) -> Path:
        return self.inputs / "datasets"

    @property
    def mcmc(self) -> Path:
        return self.inputs / "mcmc"

    @property
    def metrics(self) -> Path:
        return self.root / "metrics"

    @property
    def manifest(self) -> Path:
        return self.inputs / "dataset_manifest.csv"

    @property
    def per_dataset_metrics(self) -> Path:
        return self.root / "per_dataset_metrics.csv"

    @property
    def per_dataset_results(self) -> Path:
        return self.root / "per_dataset_results.csv"

    @property
    def thresholds(self) -> Path:
        return self.root / "thresholds.csv"


@dataclass(frozen=True)
class NPEConfig:
    training_setting: str
    training: TrainingConfig

    @property
    def label(self) -> str:
        return self.training.summary_label


def dataset_seed(base_seed: int, model: str, index: int) -> int:
    return int(base_seed + 100_000 * MODELS.index(model) + index)


def _config_seed_offset(config: NPEConfig) -> int:
    setting_offset = 0 if config.training_setting == "with_mmd" else 100
    return setting_offset + int(config.training.summary_multiplier)


def _npe_seed(base_seed: int, config: NPEConfig, model: str, index: int) -> int:
    return int(
        base_seed
        + 10_000_000
        + 1_000_000 * _config_seed_offset(config)
        + 10_000 * MODELS.index(model)
        + index
    )


def _mmd_seed(base_seed: int, config: NPEConfig, model: str, index: int) -> int:
    return int(
        base_seed
        + 20_000_000
        + 1_000_000 * _config_seed_offset(config)
        + 10_000 * MODELS.index(model)
        + index
    )


def build_npe_configs(
    summary_multipliers: list[int],
    training_settings: list[str],
    without_mmd_run_suffix: str = "noMMD",
    embed_dim: int = 64,
) -> list[NPEConfig]:
    output = []
    for setting in training_settings:
        if setting not in {"with_mmd", "without_mmd"}:
            raise ValueError(f"Unknown training setting: {setting}")
        for multiplier in summary_multipliers:
            output.append(
                NPEConfig(
                    training_setting=setting,
                    training=TrainingConfig(
                        summary_multiplier=multiplier,
                        embed_dim=embed_dim,
                        summary_base_distribution=(
                            "normal" if setting == "with_mmd" else None
                        ),
                        run_suffix=(
                            None if setting == "with_mmd" else without_mmd_run_suffix
                        ),
                    ),
                )
            )
    return output


def generate_datasets(
    paths: CalibrationPaths,
    *,
    models: list[str],
    k: int,
    base_seed: int,
    overwrite: bool,
) -> pd.DataFrame:
    from ..dataset.generate_synthetic import save_seeded_datasets
    from ..simulators import SIMULATORS

    if k < 1:
        raise ValueError("k must be positive")
    frames = []
    for model in models:
        seeds = [dataset_seed(base_seed, model, index) for index in range(k)]
        model_dir = paths.datasets / model
        manifest_path = model_dir / "true_parameters.csv"
        if manifest_path.exists() and not overwrite:
            frame = pd.read_csv(manifest_path, keep_default_na=False)
            expected_ids = [f"s{index:03d}" for index in range(k)]
            json_ids = sorted(path.stem for path in model_dir.glob("s*.json"))
            if (
                list(frame["id"]) != expected_ids
                or list(frame["dataset_seed"]) != seeds
                or json_ids != expected_ids
            ):
                raise ValueError(
                    f"Cached generated datasets in {model_dir} do not match "
                    f"k={k}, base_seed={base_seed}; use --overwrite to replace them"
                )
        else:
            frame = save_seeded_datasets(
                SIMULATORS[model], model_dir, seeds, overwrite=overwrite
            )
        frame.insert(0, "generating_model", model)
        frame.insert(1, "generating_model_label", MODEL_LABELS[model])
        frames.append(frame)
    if paths.manifest.exists():
        previous = pd.read_csv(paths.manifest, keep_default_na=False)
        previous = previous.loc[~previous["generating_model"].isin(models)]
        if len(previous):
            frames.append(previous)
    manifest = pd.concat(frames, ignore_index=True).sort_values(
        ["generating_model", "id"]
    )
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(paths.manifest, index=False)
    return manifest


def validate_generated_datasets(
    paths: CalibrationPaths,
    *,
    models: list[str],
    k: int,
    base_seed: int,
) -> None:
    for model in models:
        manifest_path = paths.datasets / model / "true_parameters.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing generated-data manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path, keep_default_na=False)
        expected_ids = [f"s{index:03d}" for index in range(k)]
        expected_seeds = [dataset_seed(base_seed, model, index) for index in range(k)]
        json_ids = sorted(
            path.stem for path in (paths.datasets / model).glob("s*.json")
        )
        if (
            list(manifest["id"]) != expected_ids
            or list(manifest["dataset_seed"]) != expected_seeds
            or json_ids != expected_ids
        ):
            raise ValueError(
                f"Generated datasets in {manifest_path} do not match "
                f"k={k}, base_seed={base_seed}"
            )


def _fit_is_complete(
    path: Path,
    expected_ids: set[str],
    *,
    require_posterior_draws: bool = True,
) -> bool:
    """Check the bridge/diagnostic tables and draws used by calibration.

    Only the matching candidate's posterior draws enter posterior MMD. Detailed
    per-parameter diagnostics and nonmatching draws are optional archival output;
    convergence decisions still come from every candidate's diagnostic table.
    """
    bridge_path = path / "bridgesampling.csv"
    diagnostic_path = path / "convergence_diagnostics.csv"
    if not bridge_path.exists() or not diagnostic_path.exists():
        return False
    bridge = pd.read_csv(bridge_path, keep_default_na=False)
    diagnostics = pd.read_csv(diagnostic_path, keep_default_na=False)
    required_diagnostics = {
        "id",
        "mcmc_seed",
        "mcmc_elapsed_seconds",
        "bridge_elapsed_seconds",
        "fit_elapsed_seconds",
        "max_rhat",
        "min_n_eff",
        "min_n_eff_ratio",
        "num_chains",
        "num_postwarmup_draws",
        "num_divergent",
        "num_max_treedepth",
        "min_ebfmi",
        "converged",
    }
    if not {"id", "estimate", "sd"}.issubset(bridge.columns):
        return False
    if not required_diagnostics.issubset(diagnostics.columns):
        return False
    for table in (bridge, diagnostics):
        if table["id"].duplicated().any() or set(table["id"]) != expected_ids:
            return False
    if require_posterior_draws:
        draws = {item.stem: item for item in (path / "posterior_draws").glob("*.csv")}
        if set(draws) != expected_ids:
            return False
        for draw_path in draws.values():
            with draw_path.open() as stream:
                if sum(1 for _ in stream) - 1 != GOLD_POSTERIOR_DRAWS:
                    return False
    return True


def run_mcmc(
    paths: CalibrationPaths,
    *,
    generating_models: list[str],
    candidate_models: list[str],
    base_seed: int,
    rscript: str,
    overwrite: bool,
) -> None:
    script = BASE_DIR / "stan" / "fit_4_models.R"
    for generating_model in generating_models:
        manifest = pd.read_csv(
            paths.datasets / generating_model / "true_parameters.csv"
        )
        expected_ids = set(manifest["id"])
        pending = [
            model
            for model in candidate_models
            if overwrite
            or not _fit_is_complete(
                paths.mcmc / generating_model / model,
                expected_ids,
                require_posterior_draws=model == generating_model,
            )
        ]
        if not pending:
            continue
        model_args = ["all"] if set(pending) == set(MODELS) else pending
        for model_arg in model_args:
            subprocess.run(
                [
                    rscript,
                    str(script),
                    model_arg,
                    f"calibration_{generating_model}",
                    str(GOLD_POSTERIOR_DRAWS),
                    str(base_seed + 1_000_000 * MODELS.index(generating_model)),
                    str(paths.datasets / generating_model),
                    str(paths.mcmc / generating_model),
                    str(not overwrite).upper(),
                ],
                check=True,
            )


def _read_mcmc_tables(
    paths: CalibrationPaths, generating_model: str, expected_ids: list[str]
) -> dict[str, pd.DataFrame]:
    tables = {}
    for candidate_model in MODELS:
        model_dir = paths.mcmc / generating_model / candidate_model
        bridge = pd.read_csv(model_dir / "bridgesampling.csv", keep_default_na=False)
        diagnostics = pd.read_csv(
            model_dir / "convergence_diagnostics.csv", keep_default_na=False
        )
        table = bridge[["id", "estimate", "sd"]].merge(
            diagnostics, on="id", validate="one_to_one"
        )
        if set(table["id"]) != set(expected_ids):
            raise ValueError(f"Incomplete MCMC output in {model_dir}")
        tables[candidate_model] = table.set_index("id", verify_integrity=True)
    return tables


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "t", "1"}


def _compute_model_metrics(
    paths: CalibrationPaths,
    generating_model: str,
    config: NPEConfig,
    approximators: dict[str, object],
    *,
    base_seed: int,
    batch_size: int | None,
    model_priors: dict[str, float],
    dataset_ids: set[str] | None = None,
) -> pd.DataFrame:
    from ..results.observed_datasets import load_dataset_directory
    from ..results.posterior_diagnostic import (
        parameter_columns,
        rbf_bandwidth2,
        squared_mmd_rbf,
    )
    from ..results.results import (
        estimate_model_comparison,
        pmp_from_log_marginals,
        posterior_draws,
    )

    dataset_dir = paths.datasets / generating_model
    y, ids = load_dataset_directory(dataset_dir)
    manifest = pd.read_csv(dataset_dir / "true_parameters.csv").set_index("id")
    mcmc = _read_mcmc_tables(paths, generating_model, ids)
    requested_ids = set(ids) if dataset_ids is None else set(dataset_ids)
    unknown_ids = sorted(requested_ids.difference(ids))
    if unknown_ids:
        raise ValueError(
            f"Requested datasets are not in the {generating_model} manifest: {unknown_ids}"
        )
    rows = []
    for index, (dataset_id, observation) in enumerate(zip(ids, y, strict=True)):
        if dataset_id not in requested_ids:
            continue
        npe_seed = _npe_seed(base_seed, config, generating_model, index)
        estimates = estimate_model_comparison(
            approximators,
            observation[None, ...],
            ids=[dataset_id],
            dataset=f"calibration_{generating_model}",
            num_samples=NPE_LOGML_DRAWS,
            batch_size=batch_size,
            seed=npe_seed,
            model_priors=model_priors,
        ).iloc[0]
        bridge_logml = np.asarray(
            [mcmc[model].loc[dataset_id, "estimate"] for model in MODELS],
            dtype=np.float64,
        )
        gold_pmp = pmp_from_log_marginals(bridge_logml, model_priors)
        all_converged = all(
            _bool(mcmc[model].loc[dataset_id, "converged"]) for model in MODELS
        )
        all_models_mcmc_seconds = sum(
            float(mcmc[model].loc[dataset_id, "mcmc_elapsed_seconds"])
            for model in MODELS
        )
        all_models_bridge_seconds = sum(
            float(mcmc[model].loc[dataset_id, "bridge_elapsed_seconds"])
            for model in MODELS
        )

        mmd_seed = _mmd_seed(base_seed, config, generating_model, index)
        npe = posterior_draws(
            approximators[generating_model],
            observation[None, ...],
            num_samples=MMD_DRAWS,
            seed=mmd_seed,
            batch_size=batch_size,
        )[0].astype(np.float64)
        posterior_path = (
            paths.metrics
            / config.label
            / "npe_posterior_draws"
            / generating_model
            / f"{dataset_id}.npy"
        )
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(posterior_path, npe)

        stan_path = (
            paths.mcmc
            / generating_model
            / generating_model
            / "posterior_draws"
            / f"{dataset_id}.csv"
        )
        stan_frame = pd.read_csv(stan_path)
        stan = stan_frame[parameter_columns(generating_model)].to_numpy(
            dtype=np.float64
        )
        if len(stan) != GOLD_POSTERIOR_DRAWS or len(npe) != MMD_DRAWS:
            raise ValueError(
                f"Expected {GOLD_POSTERIOR_DRAWS} MCMC and {MMD_DRAWS} NPE draws "
                f"for {generating_model}/{dataset_id}; got {len(stan)} and {len(npe)}"
            )
        bandwidth2 = rbf_bandwidth2(stan)
        rng = np.random.default_rng(mmd_seed)
        stan_eval = stan[rng.choice(len(stan), size=MMD_DRAWS, replace=False)]
        npe_eval = npe[rng.choice(len(npe), size=MMD_DRAWS, replace=False)]
        mmd2 = squared_mmd_rbf(stan_eval, npe_eval, bandwidth2)

        for candidate_index, candidate_model in enumerate(MODELS):
            diagnostic = mcmc[candidate_model].loc[dataset_id]
            npe_logml = float(estimates[f"log_ml_{candidate_model}"])
            gold_logml = float(diagnostic["estimate"])
            signed_logml_error = npe_logml - gold_logml
            estimated_pmp = float(estimates[f"pmp_{candidate_model}"])
            signed_pmp_error = estimated_pmp - float(gold_pmp[candidate_index])
            matching = candidate_model == generating_model
            rows.append(
                {
                    "generating_model": generating_model,
                    "generating_model_label": MODEL_LABELS[generating_model],
                    "dataset_id": dataset_id,
                    "dataset_seed": int(manifest.loc[dataset_id, "dataset_seed"]),
                    "candidate_model": candidate_model,
                    "candidate_model_label": MODEL_LABELS[candidate_model],
                    "npe_configuration": config.label,
                    "training_setting": config.training_setting,
                    "summary_multiplier": config.training.summary_multiplier,
                    "summary_dimension": config.training.summary_dim_for(
                        candidate_model
                    ),
                    "model_prior": model_priors[candidate_model],
                    "mcmc_seed": int(diagnostic["mcmc_seed"]),
                    "mcmc_elapsed_seconds": float(diagnostic["mcmc_elapsed_seconds"]),
                    "bridge_elapsed_seconds": float(
                        diagnostic["bridge_elapsed_seconds"]
                    ),
                    "fit_elapsed_seconds": float(diagnostic["fit_elapsed_seconds"]),
                    "all_models_mcmc_elapsed_seconds": all_models_mcmc_seconds,
                    "all_models_bridge_elapsed_seconds": all_models_bridge_seconds,
                    "all_models_fit_elapsed_seconds": (
                        all_models_mcmc_seconds + all_models_bridge_seconds
                    ),
                    "npe_seed": npe_seed + candidate_index,
                    "mmd_seed": mmd_seed if matching else np.nan,
                    "num_chains": int(diagnostic["num_chains"]),
                    "num_postwarmup_draws": int(diagnostic["num_postwarmup_draws"]),
                    "max_rhat": float(diagnostic["max_rhat"]),
                    "min_n_eff": float(diagnostic["min_n_eff"]),
                    "min_n_eff_ratio": float(diagnostic["min_n_eff_ratio"]),
                    "num_divergent": int(diagnostic["num_divergent"]),
                    "num_max_treedepth": int(diagnostic["num_max_treedepth"]),
                    "min_ebfmi": float(diagnostic["min_ebfmi"]),
                    "converged": _bool(diagnostic["converged"]),
                    "all_candidate_models_converged": all_converged,
                    "num_gold_posterior_draws": GOLD_POSTERIOR_DRAWS
                    if matching
                    else np.nan,
                    "num_npe_mmd_draws": MMD_DRAWS if matching else np.nan,
                    "num_mmd_draws_per_sample": MMD_DRAWS if matching else np.nan,
                    "parameter_space": "log_alpha+log_nu+logit_tau" if matching else "",
                    "mmd_kernel": "rbf" if matching else "",
                    "mmd_bandwidth_rule": "median_positive_squared_distance_full_mcmc"
                    if matching
                    else "",
                    "mmd_estimator": "biased" if matching else "",
                    "rbf_bandwidth2": bandwidth2 if matching else np.nan,
                    "posterior_mmd2": mmd2 if matching else np.nan,
                    "posterior_mmd": float(np.sqrt(mmd2)) if matching else np.nan,
                    "num_npe_logml_draws": NPE_LOGML_DRAWS,
                    "npe_log_ml": npe_logml,
                    "bridge_log_ml": gold_logml,
                    "bridge_log_ml_sd": float(diagnostic["sd"]),
                    "importance_ess": float(
                        estimates[f"importance_ess_{candidate_model}"]
                    ),
                    "signed_logml_error": signed_logml_error,
                    "absolute_logml_error": abs(signed_logml_error),
                    "estimated_pmp": estimated_pmp,
                    "gold_pmp": float(gold_pmp[candidate_index]),
                    "signed_pmp_error": signed_pmp_error,
                    "absolute_pmp_error": abs(signed_pmp_error),
                }
            )
    return pd.DataFrame(rows)


def _metric_cache_valid(
    frame: pd.DataFrame,
    *,
    generating_model: str,
    config: NPEConfig,
    manifest: pd.DataFrame,
    model_priors: dict[str, float],
) -> bool:
    required = {
        "generating_model",
        "dataset_id",
        "dataset_seed",
        "candidate_model",
        "npe_configuration",
        "training_setting",
        "summary_dimension",
        "model_prior",
        "mcmc_elapsed_seconds",
        "posterior_mmd",
        "absolute_logml_error",
        "absolute_pmp_error",
    }
    if not required.issubset(frame.columns):
        return False
    expected_keys = {
        (dataset_id, candidate_model)
        for dataset_id in manifest["id"]
        for candidate_model in MODELS
    }
    actual_keys = set(
        frame[["dataset_id", "candidate_model"]].itertuples(index=False, name=None)
    )
    prior_lookup = frame.groupby("candidate_model")["model_prior"].first().to_dict()
    return (
        actual_keys == expected_keys
        and frame["generating_model"].eq(generating_model).all()
        and frame["npe_configuration"].eq(config.label).all()
        and frame["training_setting"].eq(config.training_setting).all()
        and all(
            np.isclose(float(prior_lookup.get(model, np.nan)), model_priors[model])
            for model in MODELS
        )
        and frame[["dataset_id", "dataset_seed"]]
        .drop_duplicates()
        .set_index("dataset_id")["dataset_seed"]
        .to_dict()
        == manifest.set_index("id")["dataset_seed"].to_dict()
    )


def _compatible_cached_metrics(
    frame: pd.DataFrame,
    *,
    generating_model: str,
    config: NPEConfig,
    manifest: pd.DataFrame,
    model_priors: dict[str, float],
) -> pd.DataFrame:
    """Return complete compatible dataset blocks from a partial/global cache."""
    required = {
        "generating_model",
        "dataset_id",
        "dataset_seed",
        "candidate_model",
        "npe_configuration",
        "training_setting",
        "summary_dimension",
        "model_prior",
        "mcmc_elapsed_seconds",
        "posterior_mmd",
        "absolute_logml_error",
        "absolute_pmp_error",
    }
    if frame.empty or not required.issubset(frame.columns):
        return pd.DataFrame()

    manifest_ids = set(manifest["id"].astype(str))
    selected = frame.loc[
        frame["generating_model"].eq(generating_model)
        & frame["npe_configuration"].eq(config.label)
        & frame["training_setting"].eq(config.training_setting)
        & frame["dataset_id"].astype(str).isin(manifest_ids)
        & frame["candidate_model"].isin(MODELS)
    ].copy()
    if selected.empty:
        return selected
    key_columns = ["dataset_id", "candidate_model"]
    if selected.duplicated(key_columns).any():
        duplicates = selected.loc[
            selected.duplicated(key_columns, keep=False), key_columns
        ]
        raise ValueError(
            "Duplicate rows in reusable metric cache: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')}"
        )

    complete_ids = [
        dataset_id
        for dataset_id, group in selected.groupby("dataset_id", sort=False)
        if set(group["candidate_model"]) == set(MODELS) and len(group) == len(MODELS)
    ]
    selected = selected.loc[selected["dataset_id"].isin(complete_ids)].copy()
    if selected.empty:
        return selected

    seed_lookup = manifest.set_index("id")["dataset_seed"].to_dict()
    expected_seeds = pd.to_numeric(selected["dataset_id"].map(seed_lookup))
    actual_seeds = pd.to_numeric(selected["dataset_seed"], errors="coerce")
    if not np.array_equal(actual_seeds.to_numpy(), expected_seeds.to_numpy()):
        raise ValueError(
            f"Reusable metric cache has incompatible dataset seeds for "
            f"{generating_model}/{config.label}"
        )
    expected_priors = selected["candidate_model"].map(model_priors).astype(float)
    actual_priors = pd.to_numeric(selected["model_prior"], errors="coerce")
    if not np.isclose(actual_priors, expected_priors).all():
        raise ValueError(
            f"Reusable metric cache has incompatible model priors for "
            f"{generating_model}/{config.label}"
        )
    expected_dimensions = selected["candidate_model"].map(
        {model: config.training.summary_dim_for(model) for model in MODELS}
    )
    actual_dimensions = pd.to_numeric(selected["summary_dimension"], errors="coerce")
    if not np.array_equal(actual_dimensions.to_numpy(), expected_dimensions.to_numpy()):
        raise ValueError(
            f"Reusable metric cache has incompatible summary dimensions for "
            f"{generating_model}/{config.label}"
        )
    return selected.sort_values(key_columns).reset_index(drop=True)


def compute_metrics(
    paths: CalibrationPaths,
    *,
    generating_models: list[str],
    configs: list[NPEConfig],
    base_seed: int,
    batch_size: int | None,
    model_priors: dict[str, float],
    overwrite: bool,
    reuse_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frames = []
    for config in configs:
        approximators = None
        for generating_model in generating_models:
            cache = (
                paths.metrics
                / config.label
                / generating_model
                / "per_dataset_metrics.csv"
            )
            manifest = pd.read_csv(
                paths.datasets / generating_model / "true_parameters.csv",
                keep_default_na=False,
            )
            local_frame = (
                pd.read_csv(cache, keep_default_na=False)
                if cache.exists() and not overwrite
                else pd.DataFrame()
            )
            cached_frames = []
            if not overwrite:
                for candidate in (local_frame, reuse_metrics):
                    if candidate is None:
                        continue
                    compatible = _compatible_cached_metrics(
                        candidate,
                        generating_model=generating_model,
                        config=config,
                        manifest=manifest,
                        model_priors=model_priors,
                    )
                    if len(compatible):
                        cached_frames.append(compatible)
            frame = (
                pd.concat(cached_frames, ignore_index=True).drop_duplicates(
                    ["dataset_id", "candidate_model"], keep="first"
                )
                if cached_frames
                else pd.DataFrame()
            )
            cached_ids = set(frame["dataset_id"]) if len(frame) else set()
            missing_ids = set(manifest["id"].astype(str)).difference(cached_ids)
            print(
                f"{config.label}/{generating_model}: reuse {len(cached_ids)}, "
                f"compute {len(missing_ids)} datasets"
            )
            if missing_ids:
                if approximators is None:
                    from ..results.multisource_pipeline import load_approximators

                    approximators = load_approximators(config.training)
                computed = _compute_model_metrics(
                    paths,
                    generating_model,
                    config,
                    approximators,
                    base_seed=base_seed,
                    batch_size=batch_size,
                    model_priors=model_priors,
                    dataset_ids=missing_ids,
                )
                frame = pd.concat([frame, computed], ignore_index=True)
            frame = frame.sort_values(["dataset_id", "candidate_model"]).reset_index(
                drop=True
            )
            if not _metric_cache_valid(
                frame,
                generating_model=generating_model,
                config=config,
                manifest=manifest,
                model_priors=model_priors,
            ):
                raise ValueError(
                    f"Incomplete metric cache after resume for "
                    f"{generating_model}/{config.label}"
                )
            cache.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    paths.root.mkdir(parents=True, exist_ok=True)
    output.to_csv(paths.per_dataset_metrics, index=False)
    return output


def save_thresholds(paths: CalibrationPaths, frame: pd.DataFrame) -> pd.DataFrame:
    thresholds = calculate_thresholds(frame)
    thresholds.to_csv(paths.thresholds, index=False)
    output = add_thresholds_to_metrics(frame, thresholds)
    output.to_csv(paths.per_dataset_results, index=False)
    return thresholds


def _model_priors(values: list[float]) -> dict[str, float]:
    from ..results.results import normalized_model_priors

    priors = normalized_model_priors(dict(zip(MODELS, values, strict=True)))
    return dict(zip(MODELS, priors, strict=True))


def _validate_models(models: list[str]) -> list[str]:
    unknown = set(models) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")
    return [model for model in MODELS if model in models]


def _select_requested_metrics(
    frame: pd.DataFrame,
    *,
    models: list[str],
    configs: list[NPEConfig],
    model_priors: dict[str, float],
) -> pd.DataFrame:
    labels = [config.label for config in configs]
    selected = frame.loc[
        frame["generating_model"].isin(models) & frame["npe_configuration"].isin(labels)
    ].copy()
    expected = {(model, label) for model in models for label in labels}
    actual = set(
        selected[["generating_model", "npe_configuration"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if actual != expected:
        missing = sorted(expected - actual)
        raise ValueError(f"Missing per-dataset metric groups: {missing}")
    for model in MODELS:
        values = pd.to_numeric(
            selected.loc[selected["candidate_model"].eq(model), "model_prior"],
            errors="coerce",
        )
        if not len(values) or not np.isclose(values, model_priors[model]).all():
            raise ValueError(f"Cached metrics use a different prior for {model}")
    return selected


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("generate", "mcmc", "metrics", "thresholds", "all")
    )
    parser.add_argument(
        "--variant",
        choices=CALIBRATION_VARIANTS,
        help="Checkpoint/result variant (default: noMMD).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override calibration_outputs_100_<variant> for metrics and thresholds.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Shared datasets/MCMC root for every stage (default: calibration_reference_100).",
    )
    parser.add_argument(
        "--reuse-metrics",
        type=Path,
        help=(
            "Reuse compatible dataset rows from an existing per_dataset_metrics.csv "
            "(or a directory containing it) and compute only missing datasets."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_DATASETS_PER_MODEL,
        help="Number of independently seeded datasets per generating model (default: 100).",
    )
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--candidate-models", nargs="+", default=list(MODELS))
    parser.add_argument("--summary-multipliers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument(
        "--training-settings",
        nargs="+",
        choices=("with_mmd", "without_mmd"),
        help="Optional training override; mixed settings require --output-root.",
    )
    parser.add_argument(
        "--without-mmd-run-suffix",
        help="Checkpoint suffix for the without-MMD networks (for example noMMD_rerun1).",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=TrainingConfig().embed_dim,
        help="DeepSet embedding dimension used by the selected checkpoints.",
    )
    parser.add_argument("--model-priors", nargs=4, type=float, default=[0.25] * 4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.embed_dim < 1:
        parser.error("--embed-dim must be at least 1")
    if args.k < 1:
        parser.error("--k must be at least 1")
    if any(multiplier < 1 for multiplier in args.summary_multipliers):
        parser.error("--summary-multipliers must be positive")
    if args.reuse_metrics is not None and args.stage != "metrics":
        parser.error("--reuse-metrics is only valid for the metrics stage")

    requested_variant = args.variant
    variant = requested_variant or DEFAULT_CALIBRATION_VARIANT
    default_setting = "with_mmd" if variant == "withMMD" else "without_mmd"
    settings = args.training_settings or [default_setting]
    suffix = args.without_mmd_run_suffix or (
        variant if variant != "withMMD" else "noMMD"
    )
    if settings == ["with_mmd"]:
        selected_variant = "withMMD"
        if args.without_mmd_run_suffix is not None:
            parser.error("--without-mmd-run-suffix requires without_mmd training")
    elif settings == ["without_mmd"] and suffix in CALIBRATION_VARIANTS[1:]:
        selected_variant = suffix
    else:
        selected_variant = "custom"
    if requested_variant is not None and selected_variant != requested_variant:
        parser.error(
            "--variant conflicts with the selected training settings or suffix"
        )
    if selected_variant == "custom" and args.output_root is None:
        parser.error("Custom/mixed training settings require an explicit --output-root")
    args.variant = selected_variant
    args.training_settings = settings
    args.without_mmd_run_suffix = suffix
    if args.output_root is None:
        args.output_root = BASE_DIR / f"calibration_outputs_100_{selected_variant}"
    return args


def main() -> None:
    args = _parse_args()
    reuse_metrics_path = args.reuse_metrics
    if reuse_metrics_path is not None and reuse_metrics_path.is_dir():
        reuse_metrics_path = reuse_metrics_path / "per_dataset_metrics.csv"
    if reuse_metrics_path is not None and not reuse_metrics_path.exists():
        raise FileNotFoundError(f"Reusable metrics do not exist: {reuse_metrics_path}")
    reusable_metrics = (
        pd.read_csv(reuse_metrics_path, keep_default_na=False)
        if reuse_metrics_path is not None
        else None
    )
    paths = CalibrationPaths(
        args.output_root.resolve(),
        args.reference_root.resolve() if args.reference_root is not None else None,
    )
    models = _validate_models(args.models)
    candidate_models = _validate_models(args.candidate_models)
    configs = build_npe_configs(
        args.summary_multipliers,
        args.training_settings,
        args.without_mmd_run_suffix,
        args.embed_dim,
    )
    priors = _model_priors(args.model_priors)
    config_root = paths.inputs if args.stage in {"generate", "mcmc"} else paths.root
    config_root.mkdir(parents=True, exist_ok=True)
    with (config_root / "run_config.json").open("w") as stream:
        json.dump(
            {
                "stage": args.stage,
                "variant": args.variant,
                "reference_root": (
                    str(paths.reference_root)
                    if paths.reference_root is not None
                    else None
                ),
                "reuse_metrics": (
                    str(reuse_metrics_path.resolve())
                    if reuse_metrics_path is not None
                    else None
                ),
                "k": args.k,
                "base_seed": args.base_seed,
                "models": models,
                "candidate_models": candidate_models,
                "model_priors": priors,
                "gold_posterior_draws": GOLD_POSTERIOR_DRAWS,
                "mmd_draws_per_sample": MMD_DRAWS,
                "npe_logml_draws": NPE_LOGML_DRAWS,
                "threshold_intervals": {
                    "posterior_mmd": [0.0, DEFAULT_POSTERIOR_MMD_QUANTILE],
                    "signed_logml_error": DEFAULT_SIGNED_ERROR_COVERAGE,
                    "signed_pmp_error": DEFAULT_SIGNED_ERROR_COVERAGE,
                },
                "npe_configs": [
                    {"training_setting": item.training_setting, **asdict(item.training)}
                    for item in configs
                ],
            },
            stream,
            indent=2,
        )

    if args.stage in {"generate", "all"}:
        generate_datasets(
            paths,
            models=models,
            k=args.k,
            base_seed=args.base_seed,
            overwrite=args.overwrite,
        )
    if args.stage in {"mcmc", "metrics", "all"}:
        validate_generated_datasets(
            paths, models=models, k=args.k, base_seed=args.base_seed
        )
    if args.stage in {"mcmc", "all"}:
        run_mcmc(
            paths,
            generating_models=models,
            candidate_models=candidate_models,
            base_seed=args.base_seed,
            rscript=args.rscript,
            overwrite=args.overwrite,
        )
    if args.stage in {"metrics", "all"}:
        frame = compute_metrics(
            paths,
            generating_models=models,
            configs=configs,
            base_seed=args.base_seed,
            batch_size=args.batch_size,
            model_priors=priors,
            overwrite=args.overwrite,
            reuse_metrics=reusable_metrics,
        )
    elif args.stage == "thresholds":
        if not paths.per_dataset_metrics.exists():
            raise FileNotFoundError(
                f"Run the metrics stage first; missing {paths.per_dataset_metrics}"
            )
        frame = pd.read_csv(paths.per_dataset_metrics, keep_default_na=False)
    else:
        frame = None
    if args.stage in {"thresholds", "all"}:
        frame = _select_requested_metrics(
            frame, models=models, configs=configs, model_priors=priors
        )
        save_thresholds(paths, frame)


if __name__ == "__main__":
    main()
