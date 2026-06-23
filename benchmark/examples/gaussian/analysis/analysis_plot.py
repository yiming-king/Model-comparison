from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from bayesflow.metrics import MaximumMeanDiscrepancy
from matplotlib.lines import Line2D


dataset_dir = Path("/Users/yimingzang/Documents/Project/benchmark2/benchmark/examples/gaussian/results/datasets")
assumed_models = ("m1", "m2", "m3")
source_models = ("m1", "m2", "m3", "m4")
bf_pairs = ("12", "13", "23")

misspec = {
    "m1": {"m1": "Well\nspecified", "m2": "Prior\nmisspecified", "m3": "Likelihood\nmisspecified", "m4": "Open\nworld"},
    "m2": {"m1": "Prior\nmisspecified", "m2": "Well\nspecified", "m3": "Joint\nmisspecified", "m4": "Open\nworld"},
    "m3": {"m1": "Likelihood\nmisspecified", "m2": "Joint\nmisspecified", "m3": "Well\nspecified", "m4": "Open\nworld"},
}


def load_data(source_model: str, data_dir: str | Path = dataset_dir) -> list[dict]:
    """Load one source-model dataset."""
    with (Path(data_dir) / f"{source_model}.pkl").open("rb") as file:
        return pickle.load(file)


def load_all(data_dir: str | Path = dataset_dir, sources=source_models) -> dict[str, list[dict]]:
    """Load all selected source-model datasets."""
    return {source: load_data(source, data_dir) for source in sources}


def norm_prob(prob, eps: float = 1e-12) -> np.ndarray:
    """Normalize one probability vector."""
    prob = np.clip(np.asarray(prob, dtype=float), eps, 1.0)
    return prob / prob.sum()


def box_summary(values) -> dict:
    """Return boxplot summary statistics."""
    x = np.asarray(values, dtype=float).reshape(-1)
    q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    iqr = q3 - q1
    lo = q1 - 1.5 * iqr
    hi = q3 + 1.5 * iqr
    inside = x[(x >= lo) & (x <= hi)]
    return {
        "count": len(x),
        "mean": x.mean(),
        "median": median,
        "std": x.std(ddof=1),
        "min": x.min(),
        "q1": q1,
        "q3": q3,
        "max": x.max(),
        "iqr": iqr,
        "lower_fence": lo,
        "upper_fence": hi,
        "whisker_low": inside.min(),
        "whisker_high": inside.max(),
        "outlier_count": len(x) - len(inside),
    }


def summarize(df: pd.DataFrame, value_col: str, group_cols: list[str], metric: str) -> pd.DataFrame:
    """Summarize one metric by selected columns."""
    rows = []
    for keys, group in df.groupby(group_cols, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        row["metric"] = metric
        row.update(box_summary(group[value_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_logml(data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Collect gold and NPE log marginal likelihoods."""
    rows = []
    for assumed in assumed_models:
        for source, data in load_all(data_dir, sources).items():
            for item in data:
                gold = float(item[f"gold_log_marginal_{assumed}"])
                npe = float(item[f"npe_log_marginal_{assumed}"])
                rows.append(
                    {
                        "assumed_model": assumed,
                        "source_model": source,
                        "id": item["id"],
                        "gold": gold,
                        "npe": npe,
                        "signed_error": npe - gold,
                        "abs_error": abs(npe - gold),
                    }
                )
    return pd.DataFrame(rows)


def collect_logml_comparison(data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Collect logml estimates from NPE samples and gold posterior samples."""
    rows = []
    for assumed in assumed_models:
        for source, data in load_all(data_dir, sources).items():
            for item in data:
                gold = float(item[f"gold_log_marginal_{assumed}"])
                estimates = {
                    "npe": float(item[f"npe_log_marginal_{assumed}"]),
                    "gold_posterior": float(item[f"npe_log_marginal_gp_{assumed}"]),
                }
                for method, value in estimates.items():
                    rows.append(
                        {
                            "assumed_model": assumed,
                            "source_model": source,
                            "method": method,
                            "id": item["id"],
                            "gold": gold,
                            "estimate": value,
                            "signed_error": value - gold,
                            "abs_error": abs(value - gold),
                        }
                    )
    return pd.DataFrame(rows)


def posterior_rmse(samples, reference, moment: str) -> float:
    """Compute posterior mean or variance RMSE."""
    samples = np.asarray(samples, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if moment == "mean":
        return float(np.sqrt(np.mean((samples.mean(axis=0) - reference.mean(axis=0)) ** 2)))
    return float(np.sqrt(np.mean((samples.var(axis=0, ddof=1) - reference.var(axis=0, ddof=1)) ** 2)))


def collect_posterior(data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Collect posterior MMD and moment RMSE metrics."""
    rows = []
    mmd = MaximumMeanDiscrepancy(kernel="gaussian")
    for assumed in assumed_models:
        for source, data in load_all(data_dir, sources).items():
            for item in data:
                gold = np.asarray(item[f"gold_post_samples_{assumed}"], dtype=np.float32)
                npe = np.asarray(item[f"npe_post_samples_{assumed}"], dtype=np.float32)
                rows.append(
                    {
                        "assumed_model": assumed,
                        "source_model": source,
                        "id": item["id"],
                        "mmd": float(mmd(tf.convert_to_tensor(npe), tf.convert_to_tensor(gold))),
                        "mean_rmse": posterior_rmse(npe, gold, "mean"),
                        "variance_rmse": posterior_rmse(npe, gold, "variance"),
                    }
                )
    return pd.DataFrame(rows)


def collect_bayes_factor(data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Collect log Bayes factor errors for NPE and direct estimates."""
    rows = []
    for source, data in load_all(data_dir, sources).items():
        for item in data:
            for pair in bf_pairs:
                gold = float(item[f"logBF_{pair}_gold"])
                estimates = {"npe": float(item[f"logBF_{pair}_npe"]), "direct": float(item[f"logBF_{pair}_direct"])}
                for method, value in estimates.items():
                    rows.append(
                        {
                            "source_model": source,
                            "bf_pair": pair,
                            "method": method,
                            "id": item["id"],
                            "gold": gold,
                            "estimate": value,
                            "signed_error": value - gold,
                            "abs_error": abs(value - gold),
                        }
                    )
    return pd.DataFrame(rows)


def collect_pmp(data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Collect posterior model probabilities."""
    rows = []
    for source, data in load_all(data_dir, sources).items():
        for item in data:
            gold = norm_prob(item["p_gold"])
            npe = norm_prob(item["p_npe"])
            direct = norm_prob(item["p_direct"])
            for j in range(3):
                rows.append(
                    {
                        "source_model": source,
                        "model": f"m{j + 1}",
                        "id": item["id"],
                        "gold": gold[j],
                        "npe": npe[j],
                        "direct": direct[j],
                    }
                )
    return pd.DataFrame(rows)


def logml_summary(data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Summarize absolute logml error."""
    return summarize(collect_logml(data_dir), "signed_error", ["assumed_model", "source_model"], "logml_signed_error")


def logml_comparison_summary(data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Summarize logml comparison error."""
    return summarize(collect_logml_comparison(data_dir), "signed_error", ["assumed_model", "source_model", "method"], "logml_comparison_signed_error")


def posterior_summary(metric: str = "mmd", data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Summarize one posterior metric."""
    return summarize(collect_posterior(data_dir), metric, ["assumed_model", "source_model"], f"posterior_{metric}")


def bayes_factor_summary(data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Summarize signed log Bayes factor error."""
    return summarize(collect_bayes_factor(data_dir), "signed_error", ["source_model", "bf_pair", "method"], "bf_signed_error")


def pmp_summary(data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Summarize mean and median PMPs."""
    rows = []
    for source, data in load_all(data_dir).items():
        for key in ("p_gold", "p_npe", "p_direct"):
            arr = np.array([norm_prob(item[key]) for item in data])
            rows.append(
                {
                    "source_model": source,
                    "probability": key,
                    "mean_m1": arr[:, 0].mean(),
                    "mean_m2": arr[:, 1].mean(),
                    "mean_m3": arr[:, 2].mean(),
                    "median_m1": np.median(arr[:, 0]),
                    "median_m2": np.median(arr[:, 1]),
                    "median_m3": np.median(arr[:, 2]),
                }
            )
    return pd.DataFrame(rows)


def plot_triptych_box(
    df: pd.DataFrame,
    value_col: str,
    sources=source_models,
    title: str = "",
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot a three-panel boxplot for all assumed models."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True, gridspec_kw={"wspace": 0.05})
    for ax, assumed in zip(axes, assumed_models, strict=False):
        values = [df.loc[(df["assumed_model"] == assumed) & (df["source_model"] == source), value_col].to_numpy() for source in sources]
        ax.boxplot(values, showmeans=True)
        ax.set_title(rf"Assumed $M_{assumed[-1]}$", fontsize=18, fontweight="bold")
        ax.set_xticks(range(1, len(sources) + 1))
        ax.set_xticklabels([rf"$M_{source[-1]}$" + "\n" + misspec[assumed][source] for source in sources], fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(ylabel or value_col, fontsize=18)
    fig.suptitle(title or value_col, fontsize=24, y=1.04)
    fig.tight_layout()
    plt.show()


def plot_logml_error(data_dir: str | Path = dataset_dir, sources=source_models, ylim=None) -> pd.DataFrame:
    """Plot absolute logml error for all assumed models."""
    df = collect_logml(data_dir, sources)
    plot_triptych_box(df, "signed_error", sources, r"$\Delta \log p(y)$", r"$\Delta \log p(y)$", ylim)
    return df


def plot_posterior_metric(metric: str = "mmd", data_dir: str | Path = dataset_dir, sources=source_models, ylim=None) -> pd.DataFrame:
    """Plot one posterior metric for all assumed models."""
    df = collect_posterior(data_dir, sources)
    plot_triptych_box(df, metric, sources, metric, metric, ylim)
    return df


def plot_logml_comparison(data_dir: str | Path = dataset_dir, sources=source_models, ylim=None) -> pd.DataFrame:
    """Plot logml error from NPE samples and gold posterior samples."""
    df = collect_logml_comparison(data_dir, sources)
    methods = list(dict.fromkeys(df["method"]))
    colors = {"npe": "#8ecae6", "gold_posterior": "#ffb703"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, sharex=True, gridspec_kw={"wspace": 0.05})
    for ax, assumed in zip(axes, assumed_models, strict=False):
        centers = np.arange(len(sources)) * 2.4
        for i, method in enumerate(methods):
            values = [
                df.loc[(df["assumed_model"] == assumed) & (df["source_model"] == source) & (df["method"] == method), "signed_error"].to_numpy()
                for source in sources
            ]
            box = ax.boxplot(values, positions=centers + (i - 0.5) * 0.7, widths=0.55, showmeans=True, patch_artist=True)
            for patch in box["boxes"]:
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.8)
        ax.set_title(rf"Assumed $M_{assumed[-1]}$", fontsize=18, fontweight="bold")
        ax.set_xticks(centers)
        ax.set_xticklabels([rf"$M_{source[-1]}$" + "\n" + misspec[assumed][source] for source in sources], fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if ylim is not None:
            ax.set_ylim(*ylim)
    axes[0].set_ylabel(r"$\Delta \log p(y)$", fontsize=18)
    axes[-1].legend([Line2D([0], [0], color=colors[m], lw=10) for m in methods], methods, frameon=False)
    fig.suptitle("Logml error comparison", fontsize=22, y=1.04)
    fig.tight_layout()
    plt.show()
    return df


def plot_logml_comparison_scatter(source_model: str, data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Plot gold-vs-logml-estimate scatter for NPE samples and gold posterior samples."""
    df = collect_logml_comparison(data_dir, (source_model,))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"wspace": 0.12},sharey=True, sharex=True)
    global_lo = min(df["gold"].min(), df["estimate"].min())
    global_hi = max(df["gold"].max(), df["estimate"].max())
    global_pad = 0.05 * (global_hi - global_lo if global_hi > global_lo else 1.0)
    for ax, assumed in zip(axes, assumed_models, strict=False):
        sub = df[df["assumed_model"] == assumed]
        npe = sub[sub["method"] == "npe"]
        gold_posterior = sub[sub["method"] == "gold_posterior"]
        ax.scatter(npe["gold"], npe["estimate"], s=70, alpha=0.78, color="#6FA4FF", edgecolor="#4477CC", linewidth=0.8)
        ax.scatter(gold_posterior["gold"], gold_posterior["estimate"], s=150, alpha=0.78, marker="*", color="#FF8A7A", edgecolor="#D96A5C", linewidth=0.8)
        ax.plot([global_lo - global_pad, global_hi + global_pad], [global_lo - global_pad, global_hi + global_pad], "--", color="black", linewidth=1.6, dashes=(4, 3))
        ax.set_xlim(global_lo - global_pad, global_hi + global_pad)
        ax.set_ylim(global_lo - global_pad, global_hi + global_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.28)
        ax.set_title(rf"Assumed $M_{assumed[-1]}$", fontsize=18, pad=12)
    axes[0].set_ylabel("Approximated log p(y)", fontsize=18, fontweight="bold")
    fig.supxlabel("Gold", fontsize=18, y=0.04)
    fig.suptitle(rf"Observation Datasets from $M_{source_model[-1]}$", fontsize=22, y=1.02)
    fig.legend(handles=logml_comparison_legend(), loc="upper right", bbox_to_anchor=(0.98, 1.05), ncol=2, frameon=True, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.9])
    plt.show()
    return df


def plot_all_logml_comparison_scatter(data_dir: str | Path = dataset_dir, sources=source_models) -> dict[str, pd.DataFrame]:
    """Plot logml comparison scatter for all source models."""
    return {source: plot_logml_comparison_scatter(source, data_dir) for source in sources}


def logml_comparison_legend() -> list[Line2D]:
    """Create NPE and gold posterior legend handles."""
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="#6FA4FF", markeredgecolor="#4477CC", label="NPE", alpha=0.85),
        Line2D([0], [0], marker="*", linestyle="None", markersize=14, markerfacecolor="#FF8A7A", markeredgecolor="#D96A5C", label="Gold posterior", alpha=0.85),
    ]


def plot_pmp(source_model: str, data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Plot PMP gold-vs-approximation scatter for one source model."""
    df = collect_pmp(data_dir, (source_model,))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharex=True, sharey=True, gridspec_kw={"wspace": 0.12})
    for j, ax in enumerate(axes, start=1):
        sub = df[df["model"] == f"m{j}"]
        ax.scatter(sub["gold"], sub["npe"], s=70, alpha=0.78, color="#6FA4FF", edgecolor="#4477CC", linewidth=0.8)
        ax.scatter(sub["gold"], sub["direct"], s=150, alpha=0.78, marker="*", color="#FF8A7A", edgecolor="#D96A5C", linewidth=0.8)
        ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1.6, dashes=(4, 3))
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.28)
        ax.set_title(rf"$\mathbf{{p(M_{j}\mid y)}}$", fontsize=18, pad=12)
    axes[0].set_ylabel("Approximated", fontsize=18, fontweight="bold")
    fig.supxlabel("Gold", fontsize=18, y=0.04)
    fig.suptitle(rf"Observation Datasets from $M_{source_model[-1]}$", fontsize=22, y=1.02)
    fig.legend(handles=pmp_legend(), loc="upper right", bbox_to_anchor=(0.98, 1.05), ncol=2, frameon=True, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.9])
    plt.show()
    return df


def plot_all_pmp(data_dir: str | Path = dataset_dir, sources=source_models) -> dict[str, pd.DataFrame]:
    """Plot PMP scatter for all source models."""
    return {source: plot_pmp(source, data_dir) for source in sources}


def plot_logml_scatter(assumed_model: str, data_dir: str | Path = dataset_dir, sources=source_models) -> pd.DataFrame:
    """Plot gold-vs-NPE logml diagonal scatter for one assumed model."""
    df = collect_logml(data_dir, sources)
    df = df[df["assumed_model"] == assumed_model]
    fig, axes = plt.subplots(1, len(sources), figsize=(3.8 * len(sources), 4.5),sharey=True,sharex=True, gridspec_kw={"wspace": 0.12})
    global_lo = min(df["gold"].min(), df["npe"].min())
    global_hi = max(df["gold"].max(), df["npe"].max())
    global_pad = 0.05 * (global_hi - global_lo if global_hi > global_lo else 1.0)
    axes = np.atleast_1d(axes)
    for ax, source in zip(axes, sources, strict=False):
        sub = df[df["source_model"] == source]
        ax.scatter(sub["gold"], sub["npe"], s=70, alpha=0.82, color="#6FA4FF", edgecolor="#4477CC", linewidth=0.8)
        ax.plot([global_lo - global_pad, global_hi + global_pad], [global_lo - global_pad, global_hi + global_pad], "--", color="black", linewidth=1.6, dashes=(4, 3))
        ax.set_xlim(global_lo - global_pad, global_hi + global_pad)
        ax.set_ylim(global_lo - global_pad, global_hi + global_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.28)
        ax.set_title(rf"Obs from $M_{source[-1]}$", fontsize=16)
    axes[0].set_ylabel("NPE log marginal", fontsize=16, fontweight="bold")
    fig.supxlabel("Gold log marginal", fontsize=16, y=0.02)
    fig.suptitle(rf"Log marginal scatter under assumed $M_{assumed_model[-1]}$", fontsize=20, y=1.02)
    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.92])
    plt.show()
    return df


def plot_all_logml_scatter(data_dir: str | Path = dataset_dir, sources=source_models) -> dict[str, pd.DataFrame]:
    """Plot logml diagonal scatter for all assumed models."""
    return {assumed: plot_logml_scatter(assumed, data_dir, sources) for assumed in assumed_models}


def plot_bayes_factor(source_model: str, data_dir: str | Path = dataset_dir) -> pd.DataFrame:
    """Plot gold-vs-approximate log Bayes factor scatter for one source model."""
    df = collect_bayes_factor(data_dir, (source_model,))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8),sharey=True, sharex=True, gridspec_kw={"wspace": 0.12})
    global_lo = min(df["gold"].min(), df["estimate"].min())
    global_hi = max(df["gold"].max(), df["estimate"].max())
    global_pad = 0.05 * (global_hi - global_lo if global_hi > global_lo else 1.0)
    for ax, pair in zip(axes, bf_pairs, strict=False):
        sub = df[df["bf_pair"] == pair]
        npe = sub[sub["method"] == "npe"]
        direct = sub[sub["method"] == "direct"]
        ax.scatter(npe["gold"], npe["estimate"], s=70, alpha=0.78, color="#6FA4FF", edgecolor="#4477CC", linewidth=0.8)
        ax.scatter(direct["gold"], direct["estimate"], s=150, alpha=0.78, marker="*", color="#FF8A7A", edgecolor="#D96A5C", linewidth=0.8)
        ax.plot([global_lo - global_pad, global_hi + global_pad], [global_lo - global_pad, global_hi + global_pad], "--", color="black", linewidth=1.6, dashes=(4, 3))
        ax.set_xlim(global_lo - global_pad, global_hi + global_pad)
        ax.set_ylim(global_lo - global_pad, global_hi + global_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.28)
        ax.set_title(rf"$\log BF_{{{pair}}}$", fontsize=18, pad=12)
    axes[0].set_ylabel("Approximate log BF", fontsize=18, fontweight="bold")
    fig.supxlabel("Gold log BF", fontsize=18, y=0.01)
    fig.suptitle(rf"Observation Datasets from $M_{source_model[-1]}$", fontsize=22, y=1.02)
    fig.legend(handles=pmp_legend(), loc="upper right", bbox_to_anchor=(0.98, 1.05), ncol=2, frameon=True, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.9])
    plt.show()
    return df


def pmp_legend() -> list[Line2D]:
    """Create NPE and NPMP legend handles."""
    return [
        Line2D([0], [0], marker="o", linestyle="None", markersize=10, markerfacecolor="#6FA4FF", markeredgecolor="#4477CC", label="NPE", alpha=0.85),
        Line2D([0], [0], marker="*", linestyle="None", markersize=14, markerfacecolor="#FF8A7A", markeredgecolor="#D96A5C", label="NPMP", alpha=0.85),
    ]
