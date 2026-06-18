from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy
from matplotlib.lines import Line2D
from sklearn.covariance import LedoitWolf

import benchmark.examples.gaussian.direct.calculator as BF


RESULT_DIR = Path("/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/results/ood")
FIGURE_DIR = RESULT_DIR / "figures"
MODEL_SPECS = {
    "m1": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m2": {"mu_prior_mean": 3.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m3": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 3.0},
    "m4": {"mu_prior_mean": 0.1, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m5": {"mu_prior_mean": 1.5, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m6": {"mu_prior_mean": 5.0, "mu_prior_std": 1.0, "likelihood_std": 1.0},
    "m7": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 5.0},
    "m8": {"mu_prior_mean": 0.0, "mu_prior_std": 3.0, "likelihood_std": 1.0},
    "m9": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 0.1},
    "m10": {"mu_prior_mean": 0.0, "mu_prior_std": 1.0, "likelihood_std": 0.01},
    "m11": {"mu_prior_mean": 0.0, "mu_prior_std": 0.1, "likelihood_std": 1},
    "m12": {"mu_prior_mean": 0.0, "mu_prior_std": 0.01, "likelihood_std": 1},

}
ASSUMED_MODELS = tuple(MODEL_SPECS)[:4]
SOURCE_MODELS = tuple(MODEL_SPECS)
SOURCE_COLORS = ("#F0E442", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#0072B2", "#E011CF", "#999999", "#1F03EE", "#882255", "#44AA99",)
REGIME_COLORS = {"interpolation": "#E69F00", "in_distribution": "#0072B2", "extrapolation": "#CC79A7"}
DEFAULT_DISTANCE_METRIC = "l2"
NEAREST_TWO_CLASS_MARKERS = {
    "both extrapolative": ("o", 32),
    "only one not extrapolative": ("D", 46),
    "both not extrapolative": ("X", 52),
}
PLOT_FONT = {"title": 16, "label": 16, "tick": 16, "suptitle": 16, "legend": 15, "note": 16}


def misspec_label(assumed: str, source: str) -> str:
    a, s = MODEL_SPECS[assumed], MODEL_SPECS[source]
    prior_diff = not (np.isclose(a["mu_prior_mean"], s["mu_prior_mean"]) and np.isclose(a["mu_prior_std"], s["mu_prior_std"]))
    likelihood_diff = not np.isclose(a["likelihood_std"], s["likelihood_std"])
    return "J" if prior_diff and likelihood_diff else "P" if prior_diff else "L" if likelihood_diff else ""


MISSPEC_LABELS = {a: {s: misspec_label(a, s) for s in SOURCE_MODELS} for a in ASSUMED_MODELS}

def stack_obs(data: list[dict] | dict | np.ndarray, obs_key: str = "x") -> np.ndarray:
    """Stack observation arrays into shape (n_datasets, n_obs, n_dims)."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if isinstance(data, dict):
        return np.asarray(data[obs_key], dtype=np.float32)
    return np.stack([np.asarray(item[obs_key], dtype=np.float32) for item in data], axis=0)


def summary_outputs(approximator, x_batch: np.ndarray, obs_key: str = "x") -> np.ndarray:
    """Compute the standardized summary-network outputs used by the approximator."""
    z = approximator.summarize({obs_key: np.asarray(x_batch, dtype=np.float32)})
    return np.asarray(keras.ops.convert_to_numpy(z), dtype=np.float64)


def fit_reference(
    approximator,
    simulator,
    n_ref: int = 2000,
    alpha: float = 0.05,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
    n_boot: int = 1000,
    bootstrap_seed: int = 2025,
) -> dict:
    """Fit summary statistics and calibrate reference distances on independent datasets."""
    # First batch: estimate the reference mean and covariance.
    S_fit = summary_outputs(approximator, simulator.sample(n_ref)["x"])
    lw = LedoitWolf().fit(S_fit)
    reference = {
        "distance_metric": distance_metric,
        "summary_dim": int(S_fit.shape[1]),
        "mu_hat": lw.location_,
        "L_hat": np.linalg.cholesky(lw.covariance_),
    }

    # Second independent batch: calibrate the reference-distance distribution.
    S_calibration = summary_outputs(approximator, simulator.sample(n_ref)["x"])
    calibration_distances = summary_distance_from_summary(
        S_calibration,
        reference["mu_hat"],
        reference["L_hat"],
        metric=distance_metric,
    )
    stats = bootstrap_reference_stats(calibration_distances, alpha=alpha, n_boot=n_boot, seed=bootstrap_seed)
    reference.update({
        "median": stats["median"],
        "dm_low": stats["dm_low"],
        "dm_high": stats["dm_high"],
        "alpha": alpha,
    })
    return reference


def bootstrap_reference_stats(
    distances: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 2025,
) -> dict[str, float]:
    """Estimate reference quantiles by bootstrap resampling."""
    distances = np.asarray(distances, dtype=float)
    if n_boot <= 0:
        return {
            "median": float(np.median(distances)),
            "dm_low": float(np.percentile(distances, 100 * alpha / 2)),
            "dm_high": float(np.percentile(distances, 100 * (1 - alpha / 2))),
        }
    rng = np.random.default_rng(seed)
    samples = rng.choice(distances, size=(n_boot, len(distances)), replace=True)
    qs = np.percentile(samples, [50, 100 * alpha / 2, 100 * (1 - alpha / 2)], axis=1)
    return {"median": float(qs[0].mean()), "dm_low": float(qs[1].mean()), "dm_high": float(qs[2].mean())}


def summary_distance_from_summary(
    S: np.ndarray,
    mu_hat: np.ndarray,
    L_hat: np.ndarray,
    metric: str = DEFAULT_DISTANCE_METRIC,
) -> np.ndarray:
    """Compute a dimension-normalized distance from whitened summary vectors."""
    whitened = np.linalg.solve(L_hat, (np.asarray(S, dtype=np.float64) - mu_hat).T).T
    summary_dim = whitened.shape[1]
    metric = metric.lower()
    if metric == "l2":
        return np.linalg.norm(whitened, axis=1) / np.sqrt(summary_dim)
    if metric == "linf":
        return np.max(np.abs(whitened), axis=1) / np.sqrt(2 * np.log(summary_dim))
    raise ValueError("distance_metric must be 'l2' or 'linf'")


def summary_distance_from_obs(
    approximator,
    x_batch: list[dict] | dict | np.ndarray,
    reference: dict,
) -> np.ndarray:
    """Compute summary-space distances for observation datasets."""
    x_batch = stack_obs(x_batch)
    S = summary_outputs(approximator, x_batch)
    return summary_distance_from_summary(
        S,
        reference["mu_hat"],
        reference["L_hat"],
        metric=reference.get("distance_metric", DEFAULT_DISTANCE_METRIC),
    )


def quiet_bayesflow_progress() -> None:
    from tqdm.auto import tqdm as original_tqdm
    import bayesflow.approximators.helpers.conditions as bf_conditions
    import bayesflow.approximators.helpers.samplers as bf_samplers

    def quiet_tqdm(*args, **kwargs):
        kwargs["disable"] = True
        return original_tqdm(*args, **kwargs)

    bf_samplers.tqdm = quiet_tqdm
    bf_conditions.tqdm = quiet_tqdm


def compute_logml_and_posteriors(
    datasets: dict[str, list[dict]],
    calculations: dict[str, object],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> dict[str, list[dict]]:
    for assumed in assumed_models:
        calculation = calculations[assumed]
        for source in sources:
            datasets[source] = calculation.normal_analytical(datasets[source])
            datasets[source] = calculation.npe_estimation(datasets[source])
            datasets[source] = calculation.npe_estimation_use_gold_posterior(datasets[source])
    return datasets


def compute_model_probabilities(
    datasets: dict[str, list[dict]],
    direct_approximator,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> dict[str, list[dict]]:
    for source in sources:
        datasets[source] = BF.direct_get_probs(datasets[source], direct_approximator)
        datasets[source] = BF.indirect_get_probs(datasets[source], assumed_models)
    return datasets


def fit_summary_references(
    approximators: dict[str, object],
    simulators: dict[str, object],
    n_ref: int = 2000,
    alpha: float = 0.05,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    distance_metric: str = DEFAULT_DISTANCE_METRIC,
    n_boot: int = 1000,
    bootstrap_seed: int = 2025,
) -> dict[str, dict]:
    return {
        m: fit_reference(
            approximators[m],
            simulators[m],
            n_ref=n_ref,
            alpha=alpha,
            distance_metric=distance_metric,
            n_boot=n_boot,
            bootstrap_seed=bootstrap_seed + i,
        )
        for i, m in enumerate(assumed_models)
    }


def distance_regime(distance: float, reference: dict) -> str:
    if distance < reference["dm_low"]:
        return "interpolation"
    if distance > reference["dm_high"]:
        return "extrapolation"
    return "in_distribution"


def add_distances_and_regimes(
    datasets: dict[str, list[dict]],
    approximators: dict[str, object],
    references: dict[str, dict],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    eps: float = 1e-8,
) -> dict[str, list[dict]]:
    for source in sources:
        x_batch = stack_obs(datasets[source])
        distances = {m: summary_distance_from_obs(approximators[m], x_batch, references[m]) for m in assumed_models}
        for i, item in enumerate(datasets[source]):
            d_vec = np.array([distances[m][i] for m in assumed_models], dtype=float)
            regimes = {m: distance_regime(d_vec[j], references[m]) for j, m in enumerate(assumed_models)}
            order = np.argsort(d_vec)
            all_extra = all(v == "extrapolation" for v in regimes.values())
            item["summary_distances"] = {m: float(d_vec[j]) for j, m in enumerate(assumed_models)}
            item["summary_regimes"] = regimes
            item["summary_ci"] = {m: {"low": float(references[m]["dm_low"]), "high": float(references[m]["dm_high"])} for m in assumed_models}
            item["globally_extrapolative"] = bool(all_extra)
            item["at_least_one_not_extrapolative"] = not all_extra
            item["closest_summary_models"] = [assumed_models[j] for j in order[:2]]
            item["d_min"] = float(d_vec[order[0]])
            item["d_second"] = float(d_vec[order[1]])
            item["ambiguity_score_true"] = float(1.0 / (abs(d_vec[order[1]] - d_vec[order[0]]) + eps))
            item["ambiguity_score"] = item["ambiguity_score_true"] if all_extra else 0.0
    return datasets


def collect_logml_distance_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        for item in datasets[source]:
            for assumed in assumed_models:
                gold = float(item[f"gold_log_marginal_{assumed}"])
                npe = float(item[f"npe_log_marginal_{assumed}"])
                rows.append({
                    "source_model": source,
                    "id": int(item["id"]),
                    "assumed_model": assumed,
                    "d_M": float(item["summary_distances"][assumed]),
                    "dm_low": float(item["summary_ci"][assumed]["low"]),
                    "dm_high": float(item["summary_ci"][assumed]["high"]),
                    "distance_regime": item["summary_regimes"][assumed],
                    "gold_logml": gold,
                    "npe_logml": npe,
                    "signed_logml_error": npe - gold,
                })
    return pd.DataFrame(rows)


def collect_posterior_distance_frame(
    datasets: dict[str, list[dict]],
    n_samples: int = 1000,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    """Compare NPE and analytical posteriors for each dataset."""
    rows = []
    mmd = MaximumMeanDiscrepancy(kernel="gaussian")
    for source in sources:
        for item in datasets[source]:
            for assumed in assumed_models:
                gold = np.asarray(item[f"gold_post_samples_{assumed}"], dtype=np.float32)
                npe = np.asarray(item[f"npe_post_samples_{assumed}"], dtype=np.float32)
                sample_count = min(n_samples, len(gold), len(npe))
                if sample_count <= 1:
                    raise ValueError("MMD requires at least two posterior samples from each distribution")
                gold = gold[:sample_count]
                npe = npe[:sample_count]
                rows.append({
                    "source_model": source,
                    "id": int(item["id"]),
                    "assumed_model": assumed,
                    "d_M": float(item["summary_distances"][assumed]),
                    "dm_low": float(item["summary_ci"][assumed]["low"]),
                    "dm_high": float(item["summary_ci"][assumed]["high"]),
                    "distance_regime": item["summary_regimes"][assumed],
                    "posterior_mmd": float(mmd(tf.convert_to_tensor(npe), tf.convert_to_tensor(gold))),
                    "posterior_mean_rmse": float(np.sqrt(np.mean((npe.mean(axis=0) - gold.mean(axis=0)) ** 2))),
                    "n_posterior_samples": sample_count,
                })
    return pd.DataFrame(rows)


def collect_pmp_ambiguity_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    n_models = len(assumed_models)
    for source in sources:
        for item in datasets[source]:
            gold = BF.softmax_stable([item[f"gold_log_marginal_{m}"] for m in assumed_models])
            npe = BF.softmax_stable([item[f"npe_log_marginal_{m}"] for m in assumed_models])
            direct = np.asarray(item["p_direct"], dtype=float)
            direct_ok = len(direct) == n_models
            row = {
                "source_model": source,
                "id": int(item["id"]),
                "globally_extrapolative": bool(item["globally_extrapolative"]),
                "at_least_one_not_extrapolative": bool(item["at_least_one_not_extrapolative"]),
                "not_extrapolative_count": sum(v != "extrapolation" for v in item["summary_regimes"].values()),
                "ambiguity_score": float(item["ambiguity_score"]),
                "ambiguity_score_true": float(item.get("ambiguity_score_true", 1.0 / (abs(item["d_second"] - item["d_min"]) + 1e-8))),
                "d_min": float(item["d_min"]),
                "d_second": float(item["d_second"]),
                "closest_summary_models": ",".join(item["closest_summary_models"]),
                "pmp_l1_error_npe": float(np.sum(np.abs(npe - gold))),
                "pmp_l1_error_direct": float(np.sum(np.abs(direct - gold))) if direct_ok else np.nan,
            }
            row["extrapolation_class"] = _extrapolation_class(row["not_extrapolative_count"])
            row["nearest_two_extrapolation_class"] = _nearest_two_extrapolation_class(item["closest_summary_models"], item["summary_regimes"])
            for j, assumed in enumerate(assumed_models):
                regime = item["summary_regimes"][assumed]
                row[f"d_{assumed}"] = float(item["summary_distances"][assumed])
                row[f"dm_low_{assumed}"] = float(item["summary_ci"][assumed]["low"])
                row[f"dm_high_{assumed}"] = float(item["summary_ci"][assumed]["high"])
                row[f"regime_{assumed}"] = regime
                row[f"p_gold_{assumed}"] = float(gold[j])
                row[f"p_npe_{assumed}"] = float(npe[j])
                row[f"p_direct_{assumed}"] = float(direct[j]) if direct_ok else np.nan
                row[f"signed_pmp_error_npe_{assumed}"] = float(npe[j] - gold[j])
                row[f"signed_pmp_error_direct_{assumed}"] = float(direct[j] - gold[j]) if direct_ok else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_frames(logml_df: pd.DataFrame, pmp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    logml_summary = (
        logml_df.groupby(["assumed_model", "source_model", "distance_regime"], sort=False)
        .agg(n=("id", "count"), median_d_M=("d_M", "median"), median_signed_logml_error=("signed_logml_error", "median"))
        .reset_index()
    )
    pmp_summary = (
        pmp_df.groupby(["source_model", "at_least_one_not_extrapolative"], sort=False)
        .agg(n=("id", "count"), median_d_min=("d_min", "median"), median_A=("ambiguity_score", "median"), median_pmp_l1_error_npe=("pmp_l1_error_npe", "median"))
        .reset_index()
    )
    return logml_summary, pmp_summary


def _add_distance_regions(ax, low: float, high: float, x_max: float, x_min: float = 0.0) -> None:
    ax.axvspan(x_min, low, color=REGIME_COLORS["interpolation"], alpha=0.08)
    ax.axvspan(low, high, color=REGIME_COLORS["in_distribution"], alpha=0.07)
    ax.axvspan(high, x_max, color=REGIME_COLORS["extrapolation"], alpha=0.07)
    ax.axvline(low, color=REGIME_COLORS["interpolation"], linestyle="--", linewidth=1)
    ax.axvline(high, color=REGIME_COLORS["extrapolation"], linestyle="--", linewidth=1)


def _add_misspec_label(ax, assumed: str, source: str) -> None:
    label = MISSPEC_LABELS.get(assumed, {}).get(source, "")
    if label:
        ax.text(0.04, 0.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=PLOT_FONT["note"], fontweight="bold", bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 2})


def _set_shared_xlim(axes, x_max: float, x_min: float = 0.0) -> None:
    for ax in np.atleast_1d(axes):
        ax.set_xlim(x_min, x_max)


def _add_first_large_error(ax, data: pd.DataFrame, x_col: str, y_col: str, error_bound: float | None) -> None:
    if error_bound is None or data.empty:
        return
    ordered = data[[x_col, y_col]].dropna().sort_values(x_col)
    hit = ordered[np.abs(ordered[y_col]) > error_bound].head(1)
    if hit.empty:
        return
    x0 = float(hit[x_col].iloc[0])
    ax.axvline(x0, color="0.2", linestyle=":", linewidth=1.2)
    ax.text(x0, 0.96, f"{x0:.2g}", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=PLOT_FONT["note"], rotation=90)


def _error_subset_data(data: pd.DataFrame, subset: str | None) -> pd.DataFrame:
    if subset is None:
        return data
    if subset == "all_extrapolative":
        return data[data["at_least_one_not_extrapolative"] == False]
    if subset == "not_all_extrapolative":
        return data[data["at_least_one_not_extrapolative"] == True]
    raise ValueError("error_subset must be None, 'all_extrapolative', or 'not_all_extrapolative'")


def _safe_log(x) -> np.ndarray:
    return np.log(np.maximum(np.asarray(x, dtype=float), 1e-12))


def _style_axes(axes) -> None:
    for ax in np.atleast_1d(axes).ravel():
        ax.title.set_fontsize(PLOT_FONT["title"])
        ax.xaxis.label.set_size(PLOT_FONT["label"])
        ax.yaxis.label.set_size(PLOT_FONT["label"])
        ax.tick_params(labelsize=PLOT_FONT["tick"])


def _style_colorbar(cbar) -> None:
    cbar.ax.yaxis.label.set_size(PLOT_FONT["label"])
    cbar.ax.tick_params(labelsize=PLOT_FONT["tick"])


def _extrapolation_class(n_not: int) -> str:
    if n_not == 0:
        return "all extrapolative"
    if n_not == 1:
        return "only one not extrapolative"
    return "two or more not extrapolative"


def _nearest_two_extrapolation_class(models: list[str] | tuple[str, ...] | str, regimes: dict[str, str]) -> str:
    models = models.split(",") if isinstance(models, str) else models
    n_not = sum(regimes[m] != "extrapolation" for m in models[:2])
    if n_not == 0:
        return "both extrapolative"
    if n_not == 1:
        return "only one not extrapolative"
    return "both not extrapolative"


def _with_extrapolation_class(df: pd.DataFrame) -> pd.DataFrame:
    if {"extrapolation_class", "nearest_two_extrapolation_class"}.issubset(df.columns):
        return df
    regimes = [f"regime_{m}" for m in ASSUMED_MODELS]
    d_cols = [f"d_{m}" for m in ASSUMED_MODELS]
    out = df.copy()
    if "extrapolation_class" not in out.columns:
        out["not_extrapolative_count"] = out[regimes].ne("extrapolation").sum(axis=1)
        out["extrapolation_class"] = out["not_extrapolative_count"].map(_extrapolation_class)
    if "nearest_two_extrapolation_class" not in out.columns:
        order = np.argsort(out[d_cols].to_numpy(float), axis=1)
        out["nearest_two_extrapolation_class"] = [
            _nearest_two_extrapolation_class([ASSUMED_MODELS[j] for j in idx[:2]], row)
            for idx, row in zip(order, out[regimes].rename(columns=lambda x: x.replace("regime_", "")).to_dict("records"), strict=False)
        ]
    return out



def plot_logml_error_vs_distance(
    logml_df: pd.DataFrame,
    color_by: str = "source",
    output_dir: str | Path | None = FIGURE_DIR,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    x: str = "distance",
    error_bound: float | None = None,
    x_min: float | None = None,
    filename: str | None = None,
):
    """Plot log marginal likelihood error vs distance, one panel per assumed model."""
    if x not in {"distance", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance' or 'log_distance'")
    if color_by not in {"source", "gold_logml", "npe_logml", "signed_logml_error"}:
        raise ValueError("color_by must be 'source', 'gold_logml', 'npe_logml', or 'signed_logml_error'")

    use_log = x in {"log_distance", "logdistance"}
    x_col = "_x"
    x_min = -0.5 if x_min is None and use_log else 0.0 if x_min is None else x_min

    fig, axes = plt.subplots(1, len(assumed_models), figsize=(5.1 * len(assumed_models), 5.2), sharey=True)
    axes = np.atleast_1d(axes)
    color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
    last = None

    for ax, assumed in zip(axes, assumed_models, strict=False):
        sub = logml_df[logml_df["assumed_model"] == assumed].copy()
        sub[x_col] = _safe_log(sub["d_M"]) if use_log else sub["d_M"]
        low = float(_safe_log(sub["dm_low"].iloc[0])) if use_log else float(sub["dm_low"].iloc[0])
        high = float(_safe_log(sub["dm_high"].iloc[0])) if use_log else float(sub["dm_high"].iloc[0])
        x_max = max(float(sub[x_col].max()) * 1.05, high * 1.1)

        _add_distance_regions(ax, low, high, x_max, x_min=x_min)
        if color_by == "source":
            for source in SOURCE_MODELS:
                group = sub[sub["source_model"] == source]
                if group.empty:
                    continue
                ax.scatter(
                    group[x_col],
                    group["signed_logml_error"],
                    s=24,
                    color=color_map[source],
                    alpha=0.75,
                    edgecolors="black",
                    linewidths=0.35,
                )
        else:
            last = ax.scatter(
                sub[x_col],
                sub["signed_logml_error"],
                c=sub[color_by],
                cmap="viridis",
                s=24,
                alpha=0.75,
                edgecolors="black",
                linewidths=0.35,
            )

        _add_first_large_error(ax, sub, x_col, "signed_logml_error", error_bound)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xlim(x_min, x_max)
        ax.set_title(rf"Assumed {assumed.upper()}")
        ax.set_xlabel(r"$\log d_j(y)$" if use_log else r"$d_j(y)$")
        ax.grid(alpha=0.18)

    axes[0].set_ylabel(r"$\widehat{\log p}(y\mid M_j)-\log p(y\mid M_j)$")
    _style_axes(axes)
    if color_by == "source":
        _source_legend(fig, y=-0.10)
    elif last is not None:
        cbar = fig.colorbar(last, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(color_by)
        _style_colorbar(cbar)

    fig.tight_layout(rect=(0, 0.12 if color_by == "source" else 0, 1, 1))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        default_name = "logml_error_vs_distance_by_assumed.png" if not use_log else "logml_error_vs_log_distance_by_assumed.png"
        fig.savefig(Path(output_dir) / (filename or default_name), dpi=200, bbox_inches="tight")
    return fig, axes

def plot_signed_logml_error_grid(
    logml_df: pd.DataFrame,
    output_dir: str | Path | None = FIGURE_DIR,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    sharey: bool = True,
    x: str = "distance",
    error_bound: float | None = None,
    x_min: float | None = None,
    filename: str | None = None,
):
    if x not in {"distance", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance' or 'log_distance'")
    use_log = x in {"log_distance", "logdistance"}
    x_min = -0.5 if x_min is None and use_log else 0.0 if x_min is None else x_min
    fig, axes = plt.subplots(len(assumed_models), len(sources), figsize=(3.1 * len(sources), 2.9 * len(assumed_models)), sharey=sharey)
    axes = np.atleast_2d(axes)
    for r, assumed in enumerate(assumed_models):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            sub = logml_df[(logml_df["assumed_model"] == assumed) & (logml_df["source_model"] == source)]
            x_values = _safe_log(sub["d_M"]) if use_log else sub["d_M"]
            low = float(_safe_log(sub["dm_low"].iloc[0])) if use_log else float(sub["dm_low"].iloc[0])
            high = float(_safe_log(sub["dm_high"].iloc[0])) if use_log else float(sub["dm_high"].iloc[0])
            x_max = max(float(np.max(x_values)) * 1.05, high * 1.1)
            _add_distance_regions(ax, low, high, x_max, x_min=x_min)
            plot_sub = sub.assign(_x=x_values)
            ax.scatter(plot_sub["_x"], plot_sub["signed_logml_error"], s=20, color="0.15", alpha=0.75)
            _add_first_large_error(ax, plot_sub, "_x", "signed_logml_error", error_bound)
            ax.axhline(0, color="0.35", linewidth=0.8)
            ax.set_xlim(x_min, x_max)
            ax.grid(alpha=0.16)
            _add_misspec_label(ax, assumed, source)
            if r == 0:
                ax.set_title(source.upper())
            if c == 0:
                ax.set_ylabel(f"Assumed {assumed.upper()}")
    _style_axes(axes)
    for ax in axes.ravel():
        ax.title.set_fontsize(18)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        ax.tick_params(labelsize=18)
    fig.supylabel(r"$\hat{p}(y\mid M_j)-p(y\mid M_j)$", fontsize=18)
    fig.supxlabel(r"$\log d_j(y)$" if use_log else r"$d_j(y)$", fontsize=18)
    fig.tight_layout(rect=(0.015, 0.025, 1, 1))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        default_name = (
            ("signed_logml_error_grid_log_distance.png" if sharey else "signed_logml_error_grid_log_distance_free_y.png")
            if use_log else
            ("signed_logml_error_grid.png" if sharey else "signed_logml_error_grid_free_y.png")
        )
        fig.savefig(Path(output_dir) / (filename or default_name), dpi=200, bbox_inches="tight")
    return fig, axes


def plot_posterior_metric_grid(
    posterior_df: pd.DataFrame,
    metric: str = "mmd",
    output_dir: str | Path | None = FIGURE_DIR,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    sharey: bool = False,
    x: str = "distance",
    x_min: float | None = None,
    filename: str | None = None,
):
    """Plot a posterior-quality metric against summary distance."""
    metrics = {
        "mmd": ("posterior_mmd", "Posterior MMD", "Gaussian MMD"),
        "mean_rmse": ("posterior_mean_rmse", "Posterior mean RMSE", "posterior mean RMSE"),
    }
    if metric not in metrics:
        raise ValueError("metric must be 'mmd' or 'mean_rmse'")
    if x not in {"distance", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance' or 'log_distance'")
    metric_col, y_label, title_metric = metrics[metric]
    use_log = x in {"log_distance", "logdistance"}
    x_min = -0.5 if x_min is None and use_log else 0.0 if x_min is None else x_min
    fig, axes = plt.subplots(
        len(assumed_models),
        len(sources),
        figsize=(3.1 * len(sources), 2.9 * len(assumed_models)),
        sharey=sharey,
    )
    axes = np.atleast_2d(axes)
    for r, assumed in enumerate(assumed_models):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            sub = posterior_df[(posterior_df["assumed_model"] == assumed) & (posterior_df["source_model"] == source)]
            if sub.empty:
                ax.set_visible(False)
                continue
            x_values = _safe_log(sub["d_M"]) if use_log else sub["d_M"]
            low = float(_safe_log(sub["dm_low"].iloc[0])) if use_log else float(sub["dm_low"].iloc[0])
            high = float(_safe_log(sub["dm_high"].iloc[0])) if use_log else float(sub["dm_high"].iloc[0])
            x_max = max(float(np.max(x_values)) * 1.05, high * 1.1)
            _add_distance_regions(ax, low, high, x_max, x_min=x_min)
            ax.scatter(x_values, sub[metric_col], s=20, color="0.15", alpha=0.75)
            ax.set_xlim(x_min, x_max)
            ax.grid(alpha=0.16)
            _add_misspec_label(ax, assumed, source)
            if r == 0:
                ax.set_title(source.upper())
            if c == 0:
                ax.set_ylabel(f"Assumed {assumed.upper()}\n{y_label}")
            if r == len(assumed_models) - 1:
                ax.set_xlabel(r"$\log d_j(y)$" if use_log else r"$d_j(y)$")
    _style_axes(axes)
    fig.suptitle(f"NPE posterior vs analytical posterior ({title_metric})", y=1.01, fontsize=22)
    fig.tight_layout()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = "_log_distance" if use_log else ""
        share_suffix = "" if sharey else "_free_y"
        fig.savefig(Path(output_dir) / (filename or f"posterior_{metric}_vs_distance_grid{suffix}{share_suffix}.png"), dpi=200, bbox_inches="tight")
    return fig, axes


def _pmp_long_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    pmp_df = _with_extrapolation_class(pmp_df).copy()
    if "ambiguity_score_true" not in pmp_df:
        pmp_df["ambiguity_score_true"] = 1.0 / (np.abs(pmp_df["d_second"] - pmp_df["d_min"]) + 1e-8)
    rows = []
    for model in ASSUMED_MODELS:
        cols = [
            "source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "ambiguity_score", "ambiguity_score_true", "d_min", "d_second",
            f"d_{model}", f"dm_low_{model}", f"dm_high_{model}",
            f"p_gold_{model}", f"p_npe_{model}", f"p_direct_{model}", f"signed_pmp_error_{estimate}_{model}",
        ]
        part = pmp_df[cols].copy()
        part.columns = ["source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "A_raw", "A_true", "d_min", "d_second", "d_M", "dm_low", "dm_high", "gold", "npe", "direct", "signed_error"]
        part["A_score"] = np.log1p(part["A_raw"])
        part["A_true_score"] = np.log1p(part["A_true"])
        part["rho_M"] = part["d_M"] / part["dm_high"]
        part["rho_low"] = part["dm_low"] / part["dm_high"]
        part["log_d_M"] = _safe_log(part["d_M"])
        part["log_d_min"] = _safe_log(part["d_min"])
        part["log_rho_M"] = _safe_log(part["rho_M"])
        part["model"] = model
        rows.append(part)
    return pd.concat(rows, ignore_index=True) # row number: 3 * 7 * 50


def _pmp_rmse_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    out = _with_extrapolation_class(pmp_df).copy()
    if "ambiguity_score_true" not in out:
        out["ambiguity_score_true"] = 1.0 / (np.abs(out["d_second"] - out["d_min"]) + 1e-8)
    error_cols = [f"signed_pmp_error_{estimate}_{m}" for m in ASSUMED_MODELS]
    out["pmp_rmse"] = np.sqrt(np.mean(np.square(out[error_cols].to_numpy(float)), axis=1))
    out["A_raw"] = out["ambiguity_score"]
    out["A_true"] = out["ambiguity_score_true"]
    out["A_score"] = np.log1p(out["A_raw"])
    out["A_true_score"] = np.log1p(out["A_true"])
    out["log_d_min"] = _safe_log(out["d_min"])
    return out


def _nearest_distance_region(data: pd.DataFrame) -> tuple[float, float, float]: # return typical low/high distance values and max x for plotting based on nearest assumed model
    d_cols = [f"d_{m}" for m in ASSUMED_MODELS]
    nearest = data[d_cols].to_numpy().argmin(axis=1) # return indices of nearest assumed model for each row
    lows = [data[f"dm_low_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    highs = [data[f"dm_high_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    x_max = max(float(data["d_min"].max()) * 1.05, float(np.median(highs)) * 1.1)
    return float(np.median(lows)), float(np.median(highs)), x_max # median low/high distance values across rows based on nearest assumed model, and max x for plotting


def _marker_handles(marker_map: dict[object, tuple[str, int]]) -> list[Line2D]:
    labels = {False: "all high surprise", True: "at least one not high surprise"}
    return [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.55", markeredgecolor="black", markersize=7, label=labels.get(label, label))
        for label, (marker, _) in marker_map.items()
    ]


def _source_handles() -> list[Line2D]:
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLORS[i], markeredgecolor="black", markersize=7, label=source.upper())
        for i, source in enumerate(SOURCE_MODELS)
    ]


def _marker_legend(fig, marker_map: dict[object, tuple[str, int]], y: float = -0.07) -> None:
    fig.legend(handles=_marker_handles(marker_map), loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(marker_map), frameon=False, fontsize=PLOT_FONT["legend"])


def _source_legend(fig, y: float = -0.07) -> None:
    fig.legend(handles=_source_handles(), loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(SOURCE_MODELS), frameon=False, fontsize=PLOT_FONT["legend"])


def _right_rmse_legends(fig, group_by: str) -> None:
    marker_map = None
    if group_by in {"global_extrapolation", "global_extrapolation_source"}:
        marker_map = {False: ("o", 32), True: ("D", 48)}
    elif group_by in {"nearest_two", "nearest_two_source"}:
        marker_map = NEAREST_TWO_CLASS_MARKERS

    if marker_map is not None:
        fig.legend(
            handles=_marker_handles(marker_map),
            loc="center left",
            bbox_to_anchor=(0.76, 0.62),
            ncol=1,
            frameon=False,
            fontsize=PLOT_FONT["legend"],
        )

    if group_by in {"source_model", "global_extrapolation_source", "nearest_two_source"}:
        fig.legend(
            handles=_source_handles(),
            loc="center left",
            bbox_to_anchor=(0.76, 0.34),
            ncol=2,
            frameon=False,
            fontsize=PLOT_FONT["legend"],
        )


def _pmp_plot_data(pmp_df: pd.DataFrame, y: str, estimate: str) -> tuple[pd.DataFrame, bool]:
    if y == "signed_error":
        return _pmp_long_frame(pmp_df, estimate), True
    if y == "rmse":
        return _pmp_rmse_frame(pmp_df, estimate), False
    raise ValueError("y must be 'signed_error' or 'rmse'")


def _pmp_model_index(model: str) -> str:
    return model[1:] if model.lower().startswith("m") else model


def _pmp_x_column(x: str, model: str | None = None) -> tuple[str, str]:
    if x == "distance":
        if model is None:
            raise ValueError("x='distance' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "d_M", rf"$d_{j}(y)$"
    if x in {"log_distance", "logdistance"}:
        if model is None:
            raise ValueError("x='log_distance' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "log_d_M", rf"$\log d_{j}(y)$"
    if x == "rho":
        if model is None:
            raise ValueError("x='rho' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "rho_M", rf"$\rho_{j}(y)$"
    if x in {"log_rho", "logrho"}:
        if model is None:
            raise ValueError("x='log_rho' is only available for model-wise PMP plots")
        j = _pmp_model_index(model)
        return "log_rho_M", rf"$\log \rho_{j}(y)$"
    if x == "A":
        return "A_raw", r"$A_{\mathrm{raw}}(y)$"
    if x in {"logA", "x"}:
        return "A_score", r"$\log(1+A(y))$"
    if x in {"A_true", "trueA"}:
        return "A_true", r"$A_{\mathrm{true}}(y)$"
    if x in {"logA_true", "true_logA"}:
        return "A_true_score", r"$A_{\mathrm{true}}(y)$"
    if x == "d_min":
        return "d_min", r"$d_{\min}(y)$"
    if x == "log_d_min":
        return "log_d_min", r"$\log d_{\min}(y)$"
    raise ValueError("x must be 'distance', 'log_distance', 'rho', 'log_rho', 'A', 'logA', 'A_true', 'logA_true', 'x', 'd_min', or 'log_d_min'")


def _assumed_region_bounds(sub: pd.DataFrame, x: str) -> tuple[float, float]:
    if x == "rho":
        return float(sub["rho_low"].iloc[0]), 1.0
    if x in {"log_rho", "logrho"}:
        return float(_safe_log(sub["rho_low"].iloc[0])), 0.0
    low, high = float(sub["dm_low"].iloc[0]), float(sub["dm_high"].iloc[0])
    if x in {"log_distance", "logdistance"}:
        return float(_safe_log(low)), float(_safe_log(high))
    return low, high


def _scatter_pmp(ax, data: pd.DataFrame, x_col: str, y_col: str, group_by: str):
    if group_by == "source_model":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        ax.scatter(data[x_col], data[y_col], c=data["source_model"].map(color_map), s=36, alpha=0.75, edgecolors="black", linewidths=0.4)
        return None
    if group_by == "global_extrapolation_source":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        marker_map = {False: ("o", 32), True: ("D", 48)}
        for group_value, (marker, size) in marker_map.items():
            group = data[data["at_least_one_not_extrapolative"] == group_value]
            ax.scatter(group[x_col], group[y_col], c=group["source_model"].map(color_map), s=size, alpha=0.75, marker=marker, edgecolors="black", linewidths=0.55)
        return None
    if group_by == "nearest_two_source":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        for group_value, (marker, size) in NEAREST_TWO_CLASS_MARKERS.items():
            group = data[data["nearest_two_extrapolation_class"] == group_value]
            ax.scatter(group[x_col], group[y_col], c=group["source_model"].map(color_map), s=size, alpha=0.75, marker=marker, edgecolors="black", linewidths=0.55)
        return None

    group_col = "at_least_one_not_extrapolative" if group_by == "global_extrapolation" else "nearest_two_extrapolation_class"
    marker_map = {
        "global_extrapolation": {False: ("o", 32), True: ("D", 48)},
        "nearest_two": NEAREST_TWO_CLASS_MARKERS,
    }[group_by]
    last = None
    for group_value, (marker, size) in marker_map.items():
        group = data[data[group_col] == group_value]
        if len(group) == 0:
            continue
        last = ax.scatter(group[x_col], group[y_col], c=group["gold"], cmap="viridis", vmin=0, vmax=1, s=size, alpha=0.72, marker=marker, edgecolors="black", linewidths=0.55)
    return last


def plot_pmp_diagnostic(
    pmp_df: pd.DataFrame,
    x: str,
    y: str = "signed_error",
    group_by: str = "global_extrapolation",
    estimate: str = "npe",
    output_dir: str | Path | None = FIGURE_DIR,
    filename: str | None = None,
    title: str | None = None,
    regions: str | None = None,
    sharex: bool = False,
    error_bound: float | None = None,
    error_subset: str | None = None,
    x_min: float = -0.5,
):
    data, by_model = _pmp_plot_data(pmp_df, y, estimate)
    y_col = "signed_error" if y == "signed_error" else "pmp_rmse"
    y_label = r"$\hat{p}(M_j|y)-p(M_j|y)$" if y == "signed_error" else "PMP RMSE"
    n_axes = len(ASSUMED_MODELS) if by_model else 1
    right_rmse_legend = y == "rmse" and group_by in {
        "source_model",
        "global_extrapolation",
        "global_extrapolation_source",
        "nearest_two",
        "nearest_two_source",
    }
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(4.9 * n_axes, 6) if by_model else (8.8, 5.1) if right_rmse_legend else (6.8, 5.1),
        sharex=sharex,
        sharey=by_model,
        constrained_layout=not right_rmse_legend,
    )
    if right_rmse_legend:
        fig.subplots_adjust(right=0.72)
    axes = np.atleast_1d(axes)
    last = None
    plot_data = []

    for model in ASSUMED_MODELS if by_model else [None]:
        sub = data[data["model"] == model] if by_model else data
        x_col, x_label = _pmp_x_column(x, model)
        x_max = max(float(sub[x_col].max()) * 1.05, 1e-12)
        if regions == "assumed":
            _, high = _assumed_region_bounds(sub, x)
            x_max = max(x_max, high * 1.1)
        elif regions == "nearest":
            _, _, x_max = _nearest_distance_region(pmp_df)
            if x == "log_d_min":
                x_max = float(_safe_log(x_max))
        plot_data.append((model, sub, x_col, x_label, x_max))

    shared_x_max = max(item[-1] for item in plot_data)

    for ax, (model, sub, x_col, x_label, x_max) in zip(axes, plot_data, strict=False):
        plot_x_max = shared_x_max if sharex else x_max
        if regions == "assumed": # regions based on the assumed model in each subplot
            low, high = _assumed_region_bounds(sub, x)
            _add_distance_regions(ax, low, high, plot_x_max, x_min=x_min)
            ax.set_xlim(x_min, plot_x_max)
        if regions == "nearest": # d_min regions based on the nearest assumed model for each point, same across subplots
            low, high, _ = _nearest_distance_region(pmp_df)
            if x == "log_d_min":
                low, high = float(_safe_log(low)), float(_safe_log(high))
            _add_distance_regions(ax, low, high, plot_x_max, x_min=x_min)
            ax.set_xlim(x_min, plot_x_max)
        last = _scatter_pmp(ax, sub, x_col, y_col, group_by) or last
        _add_first_large_error(ax, _error_subset_data(sub, error_subset), x_col, y_col, error_bound)
        if y == "signed_error":
            ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_title(rf"$p(M_{model[-1]}\mid y)$" if by_model else title or "")
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.2)

    if sharex and regions is None:
        _set_shared_xlim(axes, shared_x_max, x_min=x_min)

    axes[0].set_ylabel(y_label)
    _style_axes(axes)
    if title and by_model:
        fig.suptitle(title, y=1.04, fontsize=PLOT_FONT["suptitle"])
    if right_rmse_legend:
        _right_rmse_legends(fig, group_by)
    elif group_by == "source_model":
        _source_legend(fig)
    elif group_by == "global_extrapolation_source":
        _marker_legend(fig, {False: ("o", 32), True: ("D", 48)}, y=-0.07)
        _source_legend(fig, y=-0.15)
    elif group_by == "nearest_two_source":
        _marker_legend(fig, NEAREST_TWO_CLASS_MARKERS, y=-0.07)
        _source_legend(fig, y=-0.15)
    elif group_by == "global_extrapolation":
        _marker_legend(fig, {False: ("o", 32), True: ("D", 48)})
    elif group_by == "nearest_two":
        _marker_legend(fig, NEAREST_TWO_CLASS_MARKERS)
        _source_legend(fig, y=-0.15)
    if last is not None:
        _style_colorbar(fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02))
    if output_dir is not None and filename:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=200, bbox_inches="tight")
    return fig, axes if by_model else axes[0]


def plot_pmp_estimates_vs_distance(
    pmp_df: pd.DataFrame,
    source_model: str,
    x: str = "distance",
    y: str = "estimate",
    x_min: float | None = None,
    output_dir: str | Path | None = FIGURE_DIR,
):
    if x not in {"distance", "log_distance", "logdistance"}:
        raise ValueError("x must be 'distance' or 'log_distance'")
    if y not in {"estimate", "signed_error"}:
        raise ValueError("y must be 'estimate' or 'signed_error'")
    n_models = len(ASSUMED_MODELS)
    fig, axes = plt.subplots(1, n_models, figsize=(4.4 * n_models, 5.5), sharey=True, constrained_layout=False)
    axes = np.atleast_1d(axes)
    last = None
    data = pmp_df[pmp_df["source_model"] == source_model]
    use_log = x in {"log_distance", "logdistance"}
    x_min = -0.5 if x_min is None and use_log else 0.0 if x_min is None else x_min
    for ax, model in zip(axes, ASSUMED_MODELS, strict=False):
        x_values = _safe_log(data[f"d_{model}"]) if use_log else data[f"d_{model}"]
        low = float(_safe_log(data[f"dm_low_{model}"].iloc[0])) if use_log else float(data[f"dm_low_{model}"].iloc[0])
        high = float(_safe_log(data[f"dm_high_{model}"].iloc[0])) if use_log else float(data[f"dm_high_{model}"].iloc[0])
        x_max = max(float(np.max(x_values)) * 1.05, high * 1.1)
        _add_distance_regions(ax, low, high, x_max, x_min=x_min)
        if y == "estimate":
            npe_y = data[f"p_npe_{model}"]
            direct_y = data[f"p_direct_{model}"]
            ax.set_ylim(-0.02, 1.02)
        else:
            npe_y = data[f"p_npe_{model}"] - data[f"p_gold_{model}"]
            direct_y = data[f"p_direct_{model}"] - data[f"p_gold_{model}"]
            ax.axhline(0, color="0.35", linewidth=0.8)
            ax.set_ylim(-1.02, 1.02)
        last = ax.scatter(x_values, npe_y, c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=42, alpha=0.78, marker="o", label="NPE" if y == "estimate" else "NPE - gold")
        ax.scatter(x_values, direct_y, c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=88, alpha=0.78, marker="*", edgecolors="black", linewidths=0.5, label="NPMP" if y == "estimate" else "NPMP - gold")
        ax.set_title(rf"$p(M_{model[-1]} \mid y)$")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(rf"$\log d_{model[-1]}(y)$" if use_log else rf"$d_{model[-1]}(y)$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated PMP" if y == "estimate" else "Signed PMP error")
    _style_axes(axes)
    fig.subplots_adjust(top=0.78, right=0.90, wspace=0.05)
    fig.suptitle(f"Observation datasets from {source_model.upper()}", y=0.96, fontsize=PLOT_FONT["suptitle"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2, frameon=True, fontsize=PLOT_FONT["legend"])
    _style_colorbar(fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02))
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = "log_distance" if use_log else "distance"
        prefix = "pmp_estimates" if y == "estimate" else "signed_pmp_error"
        fig.savefig(Path(output_dir) / f"{prefix}_vs_{suffix}_{source_model}.png", dpi=200, bbox_inches="tight")
    return fig, axes
