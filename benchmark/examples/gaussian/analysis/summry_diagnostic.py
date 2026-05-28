from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import keras
import bayesflow as bf  # registers BayesFlow custom classes for keras.saving.load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

import benchmark.examples.gaussian.direct.calculator as BF


RESULT_DIR = Path("/Users/yimingzang/Documents/thesis/benchmark2/benchmark/examples/gaussian/results/m1_m7_ood")
FIGURE_DIR = RESULT_DIR / "figures"
ASSUMED_MODELS = ("m1", "m2", "m3")
SOURCE_MODELS = ("m1", "m2", "m3", "m4", "m5", "m6", "m7")
SOURCE_COLORS = ("#0072B2", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000")
REGIME_COLORS = {"interpolation": "#E69F00", "in_distribution": "#0072B2", "extrapolation": "#CC79A7"}
NEAREST_TWO_CLASS_MARKERS = {
    "both extrapolative": ("o", 32),
    "only one not extrapolative": ("D", 46),
    "both not extrapolative": ("X", 52),
}
MISSPEC_LABELS = {
    "m1": {"m2": "P", "m3": "L", "m4": "P", "m5": "P", "m6": "L", "m7": "P"},
    "m2": {"m1": "P", "m3": "J", "m4": "P", "m5": "P", "m6": "J", "m7": "P"},
    "m3": {"m1": "L", "m2": "J", "m4": "J", "m5": "J", "m6": "L", "m7": "J"},
}

def stack_obs(data: list[dict] | dict | np.ndarray, obs_key: str = "x") -> np.ndarray:
    """Stack observation arrays into shape (n_datasets, n_obs, n_dims)."""
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    if isinstance(data, dict):
        return np.asarray(data[obs_key], dtype=np.float32)
    return np.stack([np.asarray(item[obs_key], dtype=np.float32) for item in data], axis=0)


def summary_outputs(approximator, x_batch: np.ndarray, obs_key: str = "x", batch_size: int = 128) -> np.ndarray:
    """Compute the standardized summary-network outputs used by the approximator."""
    z = approximator.summarize({obs_key: np.asarray(x_batch, dtype=np.float32)})
    return np.asarray(keras.ops.convert_to_numpy(z), dtype=np.float64)


def fit_reference(
    approximator,
    simulator,
    n_ref: int = 2000,
    eps: float = 1e-4,
    alpha: float = 0.05,
) -> dict:
    """Simulate reference datasets and fit the summary-space Mahalanobis null distribution."""
    ref_sims = simulator.sample(n_ref)  # {"mu": (n_ref, num_dims), "x": (n_ref, num_obs, num_dims)}
    S_ref = summary_outputs(approximator, ref_sims["x"]) # (n_ref, summary_dim)

    mu_hat = S_ref.mean(axis=0) # (summary_dim,)
    S_centered = S_ref - mu_hat # (n_ref, summary_dim)
    Sigma_hat = (S_centered.T @ S_centered) / (n_ref - 1) # (summary_dim, summary_dim)
    Sigma_hat += eps * np.eye(Sigma_hat.shape[0]) 
    L_hat = np.linalg.cholesky(Sigma_hat)
    ref_dM = mahalanobis_from_summary(S_ref, mu_hat, L_hat) # (n_ref,)

    return {
        "S_ref": S_ref,
        "mu_hat": mu_hat, # mean summary vector across reference datasets
        "L_hat": L_hat,   # Cholesky factor of summary covariance across reference datasets
        "ref_dM": ref_dM, # (n_ref,) Mahalanobis distances of reference datasets in summary space
        "median": float(np.median(ref_dM)), # median of reference d_M, a typical radius in summary space
        # 95% ci 
        "dm_low": float(np.percentile(ref_dM, 100 * (alpha / 2))), # (alpha/2) quantile of reference d_M
        "dm_high": float(np.percentile(ref_dM, 100 * (1 - alpha / 2))), # (1-alpha/2) quantile of reference d_M
        "threshold": float(np.percentile(ref_dM, 100 * (1 - alpha/2))), # (1-alpha) quantile of reference d_M, used as OOD threshold
        "alpha": alpha,
    }


def mahalanobis_from_summary(S: np.ndarray, mu_hat: np.ndarray, L_hat: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distances from summary vectors."""
    whitened = np.linalg.solve(L_hat, (np.asarray(S, dtype=np.float64) - mu_hat).T).T
    return np.linalg.norm(whitened, axis=1)


def mahalanobis_from_obs(
    approximator,
    x_batch: list[dict] | dict | np.ndarray,
    reference: dict,
) -> np.ndarray:
    """Compute summary-space Mahalanobis distances for observation datasets."""
    x_batch = stack_obs(x_batch)
    S = summary_outputs(approximator, x_batch)
    return mahalanobis_from_summary(S, reference["mu_hat"], reference["L_hat"])


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
) -> dict[str, list[dict]]:
    for source in sources:
        datasets[source] = BF.direct_get_probs(datasets[source], direct_approximator)
        datasets[source] = BF.indirect_get_probs(datasets[source])
    return datasets


def fit_summary_references(
    approximators: dict[str, object],
    simulators: dict[str, object],
    n_ref: int = 2000,
    alpha: float = 0.05,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> dict[str, dict]:
    return {m: fit_reference(approximators[m], simulators[m], n_ref=n_ref, alpha=alpha) for m in assumed_models}


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
        distances = {m: mahalanobis_from_obs(approximators[m], x_batch, references[m]) for m in assumed_models}
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
            item["ambiguity_score"] = float(1.0 / (abs(d_vec[order[1]] - d_vec[order[0]]) + eps)) if all_extra else 0.0
    return datasets


def collect_logml_distance_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        source_index = sources.index(source)
        for item in datasets[source]:
            for assumed in assumed_models:
                gold = float(item[f"gold_log_marginal_{assumed}"])
                npe = float(item[f"npe_log_marginal_{assumed}"])
                rows.append({
                    "source_model": source,
                    "source_index": source_index,
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


def collect_pmp_ambiguity_frame(
    datasets: dict[str, list[dict]],
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
) -> pd.DataFrame:
    rows = []
    for source in sources:
        for item in datasets[source]:
            gold = np.asarray(item["p_gold"], dtype=float)
            npe = np.asarray(item["p_npe"], dtype=float)
            direct = np.asarray(item["p_direct"], dtype=float)
            row = {
                "source_model": source,
                "id": int(item["id"]),
                "globally_extrapolative": bool(item["globally_extrapolative"]),
                "at_least_one_not_extrapolative": bool(item["at_least_one_not_extrapolative"]),
                "not_extrapolative_count": sum(v != "extrapolation" for v in item["summary_regimes"].values()),
                "ambiguity_score": float(item["ambiguity_score"]),
                "d_min": float(item["d_min"]),
                "d_second": float(item["d_second"]),
                "closest_summary_models": ",".join(item["closest_summary_models"]),
                "pmp_l1_error_npe": float(np.sum(np.abs(npe - gold))),
                "pmp_l1_error_direct": float(np.sum(np.abs(direct - gold))),
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
                row[f"p_direct_{assumed}"] = float(direct[j])
                row[f"signed_pmp_error_npe_{assumed}"] = float(npe[j] - gold[j])
                row[f"signed_pmp_error_direct_{assumed}"] = float(direct[j] - gold[j])
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


def _add_distance_regions(ax, low: float, high: float, x_max: float) -> None:
    ax.axvspan(0, low, color=REGIME_COLORS["interpolation"], alpha=0.08)
    ax.axvspan(low, high, color=REGIME_COLORS["in_distribution"], alpha=0.07)
    ax.axvspan(high, x_max, color=REGIME_COLORS["extrapolation"], alpha=0.07)
    ax.axvline(low, color=REGIME_COLORS["interpolation"], linestyle="--", linewidth=1)
    ax.axvline(high, color=REGIME_COLORS["extrapolation"], linestyle="--", linewidth=1)


def _add_misspec_label(ax, assumed: str, source: str) -> None:
    label = MISSPEC_LABELS.get(assumed, {}).get(source, "")
    if label:
        ax.text(0.04, 0.08, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "pad": 2})


def _source_mappable():
    cmap = ListedColormap(SOURCE_COLORS[: len(SOURCE_MODELS)])
    norm = BoundaryNorm(np.arange(-0.5, len(SOURCE_MODELS) + 0.5), cmap.N)
    return cmap, norm


def _add_labeled_colorbar(fig, axes, mappable, labels: list[str], label: str) -> None:
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_ticks(range(len(labels)))
    cbar.set_ticklabels(labels)
    cbar.set_label(label)


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


def plot_signed_logml_error_grid(
    logml_df: pd.DataFrame,
    output_dir: str | Path | None = FIGURE_DIR,
    sources: tuple[str, ...] = SOURCE_MODELS,
    assumed_models: tuple[str, ...] = ASSUMED_MODELS,
    sharey: bool = True,
    filename: str | None = None,
):
    fig, axes = plt.subplots(len(assumed_models), len(sources), figsize=(20, 8.5), sharey=sharey)
    for r, assumed in enumerate(assumed_models):
        for c, source in enumerate(sources):
            ax = axes[r, c]
            sub = logml_df[(logml_df["assumed_model"] == assumed) & (logml_df["source_model"] == source)]
            x_max = max(float(sub["d_M"].max()) * 1.05, float(sub["dm_high"].iloc[0]) * 1.1)
            _add_distance_regions(ax, float(sub["dm_low"].iloc[0]), float(sub["dm_high"].iloc[0]), x_max)
            ax.scatter(sub["d_M"], sub["signed_logml_error"], s=20, color="0.15", alpha=0.75)
            ax.axhline(0, color="0.35", linewidth=0.8)
            ax.set_xlim(0, x_max)
            ax.grid(alpha=0.16)
            _add_misspec_label(ax, assumed, source)
            if r == 0:
                ax.set_title(source.upper(), fontsize=11)
            if c == 0:
                ax.set_ylabel(f"Assumed {assumed.upper()}\nNPE - analytical")
            if r == len(assumed_models) - 1:
                ax.set_xlabel(r"$d_j(y)$")
    fig.suptitle("Signed log marginal likelihood error", y=1.01)
    fig.tight_layout()
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / (filename or ("signed_logml_error_grid.png" if sharey else "signed_logml_error_grid_free_y.png")), dpi=200, bbox_inches="tight")
    return fig, axes


def plot_logml_error_vs_distance(
    logml_df: pd.DataFrame,
    color_by: str = "source",
    output_dir: str | Path | None = FIGURE_DIR,
    sharey: bool = True,
    filename: str | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=sharey, constrained_layout=True)
    cmap, norm = _source_mappable() if color_by == "source" else ("viridis", None)
    last = None
    for ax, assumed in zip(axes, ASSUMED_MODELS, strict=False):
        sub = logml_df[logml_df["assumed_model"] == assumed]
        x_max = max(float(sub["d_M"].max()) * 1.05, float(sub["dm_high"].iloc[0]) * 1.1)
        _add_distance_regions(ax, float(sub["dm_low"].iloc[0]), float(sub["dm_high"].iloc[0]), x_max)
        values = sub["source_index"] if color_by == "source" else sub["gold_logml"]
        last = ax.scatter(sub["d_M"], sub["signed_logml_error"], c=values, cmap=cmap, norm=norm, s=28, alpha=0.75)
        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_xlim(0, x_max)
        ax.set_title(f"Assumed {assumed.upper()}")
        ax.set_xlabel(r"$d_j(y)$")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel("NPE logml - analytical logml")
    if color_by == "source":
        _add_labeled_colorbar(fig, axes, last, list(SOURCE_MODELS), "observation dataset")
    else:
        fig.colorbar(last, ax=axes, label="analytical logml", fraction=0.025, pad=0.02)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        suffix = "" if sharey else "_free_y"
        fig.savefig(Path(output_dir) / (filename or f"logml_error_vs_distance_{color_by}{suffix}.png"), dpi=200, bbox_inches="tight")
    return fig, axes


def _pmp_long_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    pmp_df = _with_extrapolation_class(pmp_df)
    rows = []
    for model in ASSUMED_MODELS:
        cols = [
            "source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "ambiguity_score", "d_min",
            f"d_{model}", f"dm_low_{model}", f"dm_high_{model}",
            f"p_gold_{model}", f"p_npe_{model}", f"p_direct_{model}", f"signed_pmp_error_{estimate}_{model}",
        ]
        part = pmp_df[cols].copy()
        part.columns = ["source_model", "id", "at_least_one_not_extrapolative", "extrapolation_class", "nearest_two_extrapolation_class", "A", "d_min", "d_M", "dm_low", "dm_high", "gold", "npe", "direct", "signed_error"]
        part["log_A"] = np.log1p(part["A"])
        part["model"] = model
        rows.append(part)
    return pd.concat(rows, ignore_index=True) # row number: 3 * 7 * 50


def _pmp_rmse_frame(pmp_df: pd.DataFrame, estimate: str = "npe") -> pd.DataFrame:
    out = _with_extrapolation_class(pmp_df).copy()
    error_cols = [f"signed_pmp_error_{estimate}_{m}" for m in ASSUMED_MODELS]
    out["pmp_rmse"] = np.sqrt(np.mean(np.square(out[error_cols].to_numpy(float)), axis=1))
    out["log_A"] = np.log1p(out["ambiguity_score"])
    return out


def _nearest_distance_region(data: pd.DataFrame) -> tuple[float, float, float]: # return typical low/high distance values and max x for plotting based on nearest assumed model
    d_cols = [f"d_{m}" for m in ASSUMED_MODELS]
    nearest = data[d_cols].to_numpy().argmin(axis=1) # return indices of nearest assumed model for each row
    lows = [data[f"dm_low_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    highs = [data[f"dm_high_{ASSUMED_MODELS[j]}"].iloc[i] for i, j in enumerate(nearest)]
    x_max = max(float(data["d_min"].max()) * 1.05, float(np.median(highs)) * 1.1)
    return float(np.median(lows)), float(np.median(highs)), x_max # median low/high distance values across rows based on nearest assumed model, and max x for plotting


def _marker_legend(fig, marker_map: dict[object, tuple[str, int]], y: float = -0.07) -> None:
    labels = {False: "all extrapolative", True: "at least one not extrapolative"}
    handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor="0.55", markeredgecolor="black", markersize=7, label=labels.get(label, label))
        for label, (marker, _) in marker_map.items()
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(marker_map), frameon=False)


def _source_legend(fig, y: float = -0.07) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SOURCE_COLORS[i], markeredgecolor="black", markersize=7, label=source.upper())
        for i, source in enumerate(SOURCE_MODELS)
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=len(SOURCE_MODELS), frameon=False)


def _pmp_plot_data(pmp_df: pd.DataFrame, y: str, estimate: str) -> tuple[pd.DataFrame, bool]:
    if y == "signed_error":
        return _pmp_long_frame(pmp_df, estimate), True
    if y == "rmse":
        return _pmp_rmse_frame(pmp_df, estimate), False
    raise ValueError("y must be 'signed_error' or 'rmse'")


def _pmp_x_column(x: str, model: str | None = None) -> tuple[str, str]:
    if x == "distance":
        if model is None:
            raise ValueError("x='distance' is only available for model-wise PMP plots")
        return "d_M", rf"$d_{model[-1]}(y)$"
    if x == "A":
        return ("A" if model else "ambiguity_score"), r"$A(y)$"
    if x == "logA":
        return "log_A", r"$\log(1 + A(y))$"
    if x == "d_min":
        return "d_min", r"$d_{\min}(y)$"
    raise ValueError("x must be 'distance', 'A', 'logA', or 'd_min'")


def _scatter_pmp(ax, data: pd.DataFrame, x_col: str, y_col: str, group_by: str):
    if group_by == "source_model":
        color_map = dict(zip(SOURCE_MODELS, SOURCE_COLORS, strict=False))
        ax.scatter(data[x_col], data[y_col], c=data["source_model"].map(color_map), s=36, alpha=0.75, edgecolors="black", linewidths=0.4)
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
):
    data, by_model = _pmp_plot_data(pmp_df, y, estimate)
    y_col = "signed_error" if y == "signed_error" else "pmp_rmse"
    y_label = r"$\hat{p}(M_j|y)-p(M_j|y)$" if y == "signed_error" else "PMP RMSE"
    fig, axes = plt.subplots(1, 3 if by_model else 1, figsize=(15, 4.6) if by_model else (6.8, 5.1), sharey=by_model, constrained_layout=True)
    axes = np.atleast_1d(axes)
    last = None

    for ax, model in zip(axes, ASSUMED_MODELS if by_model else [None], strict=False):
        sub = data[data["model"] == model] if by_model else data
        x_col, x_label = _pmp_x_column(x, model)
        if regions == "assumed": # d1(y), d2(y), d3(y) regions based on the assumed model in each subplot
            x_max = max(float(sub[x_col].max()) * 1.05, float(sub["dm_high"].iloc[0]) * 1.1)
            _add_distance_regions(ax, float(sub["dm_low"].iloc[0]), float(sub["dm_high"].iloc[0]), x_max)
            ax.set_xlim(0, x_max)
        if regions == "nearest": # d_min regions based on the nearest assumed model for each point, same across subplots
            low, high, x_max = _nearest_distance_region(pmp_df)
            _add_distance_regions(ax, low, high, x_max)
            ax.set_xlim(0, x_max)
        last = _scatter_pmp(ax, sub, x_col, y_col, group_by) or last
        if y == "signed_error":
            ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set_title(f"{model.upper()} PMP" if by_model else title or "")
        ax.set_xlabel(x_label)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel(y_label)
    if title and by_model:
        fig.suptitle(title, y=1.04)
    if group_by == "source_model":
        _source_legend(fig)
    elif group_by == "global_extrapolation":
        _marker_legend(fig, {False: ("o", 32), True: ("D", 48)})
    elif group_by == "nearest_two":
        _marker_legend(fig, NEAREST_TWO_CLASS_MARKERS)
    if last is not None:
        fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02)
    if output_dir is not None and filename:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / filename, dpi=200, bbox_inches="tight")
    return fig, axes if by_model else axes[0]


def plot_pmp_estimates_vs_distance(
    pmp_df: pd.DataFrame,
    source_model: str,
    output_dir: str | Path | None = FIGURE_DIR,
):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, constrained_layout=True)
    last = None
    data = pmp_df[pmp_df["source_model"] == source_model]
    for ax, model in zip(axes, ASSUMED_MODELS, strict=False):
        x_max = max(float(data[f"d_{model}"].max()) * 1.05, float(data[f"dm_high_{model}"].iloc[0]) * 1.1)
        _add_distance_regions(ax, float(data[f"dm_low_{model}"].iloc[0]), float(data[f"dm_high_{model}"].iloc[0]), x_max)
        last = ax.scatter(data[f"d_{model}"], data[f"p_npe_{model}"], c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=42, alpha=0.78, marker="o", label="NPE")
        ax.scatter(data[f"d_{model}"], data[f"p_direct_{model}"], c=data[f"p_gold_{model}"], cmap="viridis", vmin=0, vmax=1, s=88, alpha=0.78, marker="*", edgecolors="black", linewidths=0.5, label="NPMP")
        ax.set_title(rf"$p(M_{model[-1]} \mid y)$")
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(rf"$d_{model[-1]}(y)$")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Estimated PMP")
    fig.suptitle(f"Observation datasets from {source_model.upper()}", y=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, frameon=True)
    fig.colorbar(last, ax=axes, label="gold PMP", fraction=0.025, pad=0.02)
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(output_dir) / f"pmp_estimates_vs_distance_{source_model}.png", dpi=200, bbox_inches="tight")
    return fig, axes
