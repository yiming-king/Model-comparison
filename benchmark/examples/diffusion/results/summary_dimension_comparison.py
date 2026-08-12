"""Rho-vs-error plots across diffusion summary dimensions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.special import softmax

from ..config import BASE_DIR, MODELS, MODEL_TITLES, RESULT_DIR, TrainingConfig
from .multisource_pipeline import all_observed_paths
from .posterior_diagnostic import load_posterior_diagnostic


SummarySpecs = tuple[tuple[str, TrainingConfig], ...]

SUMMARY_LABELS = {1: "S=D", 2: "S=2D", 4: "S=4D", 6: "S=6D"}
SUMMARY_COLORS = {
    "S=D": "#002FB2",
    "S=2D": "#E69F00",
    "S=4D": "#009E73",
    "S=6D": "#CC79A7",
}


def _make_summary_specs(
    summary_base_distribution: str | None = "normal",
    run_suffix: str | None = None,
) -> SummarySpecs:
    return tuple(
        (
            label,
            TrainingConfig(
                summary_multiplier=multiplier,
                summary_base_distribution=summary_base_distribution,
                run_suffix=run_suffix,
            ),
        )
        for multiplier, label in SUMMARY_LABELS.items()
    )


WITH_MMD_SUMMARY_SPECS = _make_summary_specs()
NO_MMD_SUMMARY_SPECS = _make_summary_specs(
    summary_base_distribution=None,
    run_suffix="noMMD",
)

MODEL_SETS = {"all": MODELS, "m1_m3": ("m1", "m3")}
PMP_SOURCES = {"all": "four_model", "m1_m3": "m1_m3"}

PMP_ERROR_BOUND = 0.1
LOGML_ERROR_BOUND = 1.0  # signed base-10 log marginal-likelihood error
RHO_XSCALE = "symlog"
RHO_SYMLOG_BASE = 10.0
RHO_SYMLOG_LINTHRESH = 1.0
RHO_SYMLOG_LINSCALE = 1.0
LOWESS_FRAC = {"simulated": 0.7, "empirical": 1}
LOGML_CENTRAL_INTERVAL = 0.9
TYPICAL_SET_FILL = "#DCEEDC"
M1_M3_COMPARISON_FIGSIZE = (8.0, 5)

MODEL_MATCH_STYLES = {
    False: {"marker": "o", "size": 30, "label": "misspecified datasets"},
    True: {"marker": "s", "size": 40, "label": "well-specified datasets"},
}
TRAINING_METRICS = ("l2", "linf", "mmd", "density")
TRAINING_METRIC_LABELS = {
    "l2": r"$L_2$",
    "linf": r"$L_\infty$",
    "mmd": "MMD",
    "density": "Density",
}
TRAINING_VARIANT_COLORS = {
    "With MMD loss": "#0072B2",
    "Without MMD loss": "#D55E00",
}

METRIC_PLOTS = {
    "posterior_mmd": {
        "column": "observed_mmd",
        "ylabel": "Posterior MMD",
        "xscale": RHO_XSCALE,
        "thresholds": (),
        "filename": "combined_posterior_mmd_vs_rho.png",
        "extrema": False,
        "sharey": True,
    },
    "logml": {
        "column": "logml_error",
        "ylabel": r"$\log_{10}\widehat{p}(y\mid M_j)-\log_{10}p(y\mid M_j)$",
        "xscale": RHO_XSCALE,
        "thresholds": (-1.0, 1.0),
        "filename": "combined_logml_error_vs_rho.png",
        "extrema": False,
        "sharey": False,
    },
    "pmp": {
        "column": "pmp_error",
        "ylabel": r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        "xscale": RHO_XSCALE,
        "thresholds": (-0.1, 0.1),
        "filename": "combined_pmp_error_vs_rho.png",
        "extrema": False,
        "sharey": False,
    },
}

CALIBRATION_THRESHOLD_PATH = BASE_DIR / "calibration_outputs" / "thresholds.csv"
CALIBRATED_METRIC_PLOTS = {
    "posterior_mmd": {
        **METRIC_PLOTS["posterior_mmd"],
        "column": "normalized_mmd",
        "ylabel": "Normalized posterior MMD",
        "threshold_column": None,
        "common_thresholds": (1.0,),
        "symmetric_threshold": False,
        "filename": "combined_normalized_posterior_mmd_vs_rho.png",
    },
    "logml": {
        **METRIC_PLOTS["logml"],
        "threshold_column": None,
        "common_thresholds": (-1.0, 1.0),
        "symmetric_threshold": True,
    },
    "pmp": {
        **METRIC_PLOTS["pmp"],
        "threshold_column": None,
        "common_thresholds": (-PMP_ERROR_BOUND, PMP_ERROR_BOUND),
        "symmetric_threshold": True,
    },
}


def load_calibration_thresholds(
    summary_specs: SummarySpecs,
    threshold_path: str | Path = CALIBRATION_THRESHOLD_PATH,
) -> pd.DataFrame:
    """Load one 95% threshold per assumed model and NPE configuration."""
    summary_to_config = {
        summary: config.summary_label for summary, config in summary_specs
    }
    thresholds = pd.read_csv(threshold_path, keep_default_na=False)
    required = {"generating_model", "npe_configuration", "metric", "threshold"}
    missing = sorted(required.difference(thresholds.columns))
    if missing:
        raise ValueError(f"Calibration threshold table is missing columns: {missing}")
    thresholds = thresholds.loc[
        thresholds["npe_configuration"].isin(summary_to_config.values())
    ]
    wide = thresholds.pivot(
        index=["generating_model", "npe_configuration"],
        columns="metric",
        values="threshold",
    ).rename(
        columns={
            "posterior_mmd": "posterior_mmd_threshold",
            "absolute_logml_error": "absolute_logml_error_threshold",
            "absolute_pmp_error": "pmp_error_threshold",
        }
    )
    threshold_columns = [
        "posterior_mmd_threshold",
        "absolute_logml_error_threshold",
        "pmp_error_threshold",
    ]
    missing_metrics = sorted(set(threshold_columns).difference(wide.columns))
    if missing_metrics:
        raise ValueError(f"Calibration thresholds are missing metrics: {missing_metrics}")
    wide = wide.reset_index().rename(columns={"generating_model": "model"})
    config_to_summary = {config: summary for summary, config in summary_to_config.items()}
    wide["summary"] = wide["npe_configuration"].map(config_to_summary)
    wide["log10_logml_error_threshold"] = (
        wide["absolute_logml_error_threshold"] / np.log(10.0)
    )
    output_columns = [
        "model",
        "summary",
        "npe_configuration",
        "posterior_mmd_threshold",
        "log10_logml_error_threshold",
        "pmp_error_threshold",
    ]
    output = wide[output_columns].copy()
    numeric_columns = output_columns[3:]
    if output[numeric_columns].isna().any().any():
        raise ValueError("Calibration threshold table contains missing values")
    if output[numeric_columns].le(0.0).any().any():
        raise ValueError("Calibration thresholds must be strictly positive")
    return output


def _attach_calibration_thresholds(
    data: pd.DataFrame,
    summary_specs: SummarySpecs,
    threshold_path: str | Path = CALIBRATION_THRESHOLD_PATH,
) -> pd.DataFrame:
    thresholds = load_calibration_thresholds(summary_specs, threshold_path)
    output = data.merge(
        thresholds,
        on=["model", "summary"],
        how="left",
        validate="many_to_one",
    )
    threshold_columns = [
        "posterior_mmd_threshold",
        "log10_logml_error_threshold",
        "pmp_error_threshold",
    ]
    if output[threshold_columns].isna().any().any():
        missing = output.loc[
            output[threshold_columns].isna().any(axis=1), ["model", "summary"]
        ].drop_duplicates()
        raise ValueError(
            "Missing calibration thresholds for:\n" + missing.to_string(index=False)
        )
    if "observed_mmd" in output:
        output["normalized_mmd"] = (
            output["observed_mmd"] / output["posterior_mmd_threshold"]
        )
    return output


CALIBRATED_CLASSIFICATION_SPECS = {
    "posterior_mmd": {
        "value_column": "normalized_mmd",
        "threshold": 1.0,
        "absolute": False,
    },
    "logml": {
        "value_column": "logml_error",
        "threshold": 1.0,
        "absolute": True,
    },
    "pmp": {
        "value_column": "pmp_error",
        "threshold": PMP_ERROR_BOUND,
        "absolute": True,
    },
}


def diagnostic_classification_rows(
    data: pd.DataFrame,
    diagnostic: str,
) -> pd.DataFrame:
    """Classify rho > 1 against each calibrated approximation-error definition."""
    frames = []
    for metric, spec in CALIBRATED_CLASSIFICATION_SPECS.items():
        value_column = spec["value_column"]
        required = {"dataset", "id", "model", "summary", "rho", value_column}
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Classification data is missing columns: {missing}")
        frame = data[["dataset", "id", "model", "summary", "rho", value_column]].copy()
        frame = frame.dropna(subset=["rho", value_column])
        frame["diagnostic"] = diagnostic
        frame["source_group"] = np.where(
            frame["dataset"].eq("empirical"), "empirical", "simulated"
        )
        frame["error_metric"] = metric
        frame["error_value"] = frame[value_column]
        comparison = frame[value_column].abs() if spec["absolute"] else frame[value_column]
        frame["error_threshold"] = float(spec["threshold"])
        frame["error_positive"] = comparison.gt(spec["threshold"])
        frame["diagnostic_positive"] = frame["rho"].gt(1.0)
        frame["classification"] = np.select(
            [
                frame["diagnostic_positive"] & frame["error_positive"],
                frame["diagnostic_positive"] & ~frame["error_positive"],
                ~frame["diagnostic_positive"] & frame["error_positive"],
            ],
            ["TP", "FP", "FN"],
            default="TN",
        )
        frames.append(
            frame[
                [
                    "dataset",
                    "id",
                    "model",
                    "summary",
                    "diagnostic",
                    "source_group",
                    "error_metric",
                    "rho",
                    "error_value",
                    "error_threshold",
                    "error_positive",
                    "diagnostic_positive",
                    "classification",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def calculate_diagnostic_classification(
    data: pd.DataFrame,
    diagnostic: str,
) -> pd.DataFrame:
    """Calculate FP/FN/F1 results by source, assumed model, and summary."""
    rows = diagnostic_classification_rows(data, diagnostic)
    group_columns = [
        "diagnostic",
        "source_group",
        "model",
        "summary",
        "error_metric",
        "error_threshold",
    ]
    counts = (
        rows.groupby(group_columns, observed=True)["classification"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["TP", "FP", "FN", "TN"], fill_value=0)
        .reset_index()
        .rename(columns={"TP": "tp", "FP": "fp", "FN": "fn", "TN": "tn"})
    )
    tp = counts["tp"].to_numpy(dtype=float)
    fp = counts["fp"].to_numpy(dtype=float)
    fn = counts["fn"].to_numpy(dtype=float)
    tn = counts["tn"].to_numpy(dtype=float)

    def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 0.0,
        )

    counts["n"] = (tp + fp + fn + tn).astype(int)
    counts["false_positive_rate"] = divide(fp, fp + tn)
    counts["false_negative_rate"] = divide(fn, fn + tp)
    counts["precision"] = divide(tp, tp + fp)
    counts["recall"] = divide(tp, tp + fn)
    counts["f1"] = divide(2.0 * tp, 2.0 * tp + fp + fn)
    return counts.sort_values(
        ["source_group", "model", "summary", "error_metric"]
    ).reset_index(drop=True)


def _add_model_match_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Mark simulated data generated by the currently assumed model."""
    dataset = frame["dataset"].astype("string")
    generating_model = dataset.str.extract(r"^simulated_from_(m[0-3])$", expand=False)
    return frame.assign(
        model_matched=generating_model.notna()
        & generating_model.eq(frame["model"].astype("string"))
    )


def _model_set_name(models: tuple[str, ...]) -> str:
    for name, configured in MODEL_SETS.items():
        if tuple(models) == tuple(configured):
            return name
    raise ValueError(f"Unsupported model set: {models}")


def _infer_pmp_source(models: tuple[str, ...]) -> str:
    return PMP_SOURCES[_model_set_name(models)]


def _pmp_view(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    pmp_source: str,
) -> pd.DataFrame:
    """Select saved four-model PMP or normalize saved M1/M3 logML values."""
    output = frame.copy()
    if pmp_source == "four_model":
        return output
    if pmp_source != "m1_m3" or tuple(models) != ("m1", "m3"):
        raise ValueError("m1_m3 PMP must be used with M1/M3")

    estimated = softmax(
        output[["log_ml_m1", "log_ml_m3"]].to_numpy(dtype=float), axis=1
    )
    gold = softmax(
        output[["gold_log_ml_m1", "gold_log_ml_m3"]].to_numpy(dtype=float),
        axis=1,
    )
    for index, model in enumerate(models):
        output[f"pmp_{model}"] = estimated[:, index]
        output[f"gold_pmp_{model}"] = gold[:, index]
        output[f"signed_pmp_error_{model}"] = estimated[:, index] - gold[:, index]
    return output


def model_set_view(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    metric: str | None = None,
    pmp_source: str | None = None,
) -> pd.DataFrame:
    """Compatibility helper returning the selected PMP and surprise categories."""
    del metric
    output = _pmp_view(frame, models, pmp_source or _infer_pmp_source(models))
    high_surprise = output[[f"regime_{model}" for model in models]].eq("high surprise")
    output["globally_high_surprise"] = high_surprise.all(axis=1)
    output["at_least_one_not_high_surprise"] = ~output["globally_high_surprise"]
    return output


def _load_cached_frames(
    metric: str,
    summary_specs: SummarySpecs,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    diagnostics: dict[str, pd.DataFrame] = {}
    posteriors: dict[str, pd.DataFrame] = {}
    for label, config in summary_specs:
        paths = all_observed_paths(config.summary_label, metric)
        missing = [
            str(paths[key])
            for key in ("diagnostic", "posterior")
            if not paths[key].exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing cached diagnostics:\n" + "\n".join(missing)
            )
        diagnostics[label] = pd.read_csv(
            paths["diagnostic"], keep_default_na=False
        ).assign(summary=label)
        posteriors[label] = load_posterior_diagnostic(paths["posterior"]).assign(
            summary=label
        )
    return diagnostics, posteriors


def load_summary_dimension_data(
    metric: str = "l2",
    models: tuple[str, ...] = MODELS,
    summary_specs: SummarySpecs = WITH_MMD_SUMMARY_SPECS,
    pmp_source: str | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load existing diagnostic and posterior caches; never refit any model."""
    diagnostics, posteriors = _load_cached_frames(metric, summary_specs)
    source = pmp_source or _infer_pmp_source(models)
    diagnostics = {
        label: model_set_view(frame, models, pmp_source=source)
        for label, frame in diagnostics.items()
    }
    posteriors = {
        label: frame.loc[frame["model"].isin(models)].reset_index(drop=True)
        for label, frame in posteriors.items()
    }
    return diagnostics, posteriors


def pmp_long(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    """Long PMP view retained for downstream notebooks."""
    rows = []
    for model in models:
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "model_title": MODEL_TITLES[model],
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "gold_pmp": frame[f"gold_pmp_{model}"],
                    "signed_pmp_error": frame[f"signed_pmp_error_{model}"],
                    "at_least_one_not_high_surprise": frame[
                        "at_least_one_not_high_surprise"
                    ],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def logml_long(frame: pd.DataFrame, models: tuple[str, ...]) -> pd.DataFrame:
    """Long logML view with both natural-log and requested base-10 errors."""
    rows = []
    for model in models:
        signed_ln = frame[f"signed_logml_error_{model}"].to_numpy(dtype=float)
        rows.append(
            pd.DataFrame(
                {
                    "dataset": frame["dataset"],
                    "id": frame["id"],
                    "model": model,
                    "rho": frame[f"rho_{model}"],
                    "rho_low": frame[f"dm_low_{model}"] / frame[f"dm_high_{model}"],
                    "signed_logml_error": signed_ln,
                    "logml_error": signed_ln / np.log(10.0),
                    "at_least_one_not_high_surprise": frame[
                        "at_least_one_not_high_surprise"
                    ],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def prepare_rho_error_data(
    diagnostic_by_summary: dict[str, pd.DataFrame],
    posterior_by_summary: dict[str, pd.DataFrame],
    models: tuple[str, ...],
    pmp_source: str | None = None,
    posterior_mmd_column: str = "observed_mmd",
) -> pd.DataFrame:
    """Create one participant-level table shared by all three plot types."""
    source = pmp_source or _infer_pmp_source(models)
    frames = []
    for summary, raw_diagnostic in diagnostic_by_summary.items():
        diagnostic = model_set_view(raw_diagnostic, models, pmp_source=source)
        posterior = posterior_by_summary[summary]
        if posterior_mmd_column not in posterior:
            raise ValueError(
                f"{summary} posterior cache has no {posterior_mmd_column}; rerun "
                "the posterior diagnostic pipeline before plotting"
            )
        posterior_columns = ["dataset", "id", "model", posterior_mmd_column]
        posterior = posterior.loc[
            posterior["model"].isin(models),
            posterior_columns,
        ]

        rows = []
        for model in models:
            signed_ln = diagnostic[f"signed_logml_error_{model}"].to_numpy(dtype=float)
            globally_high = (
                diagnostic[[f"regime_{candidate}" for candidate in models]]
                .eq("high surprise")
                .all(axis=1)
            )
            rows.append(
                pd.DataFrame(
                    {
                        "dataset": diagnostic["dataset"],
                        "id": diagnostic["id"],
                        "summary": summary,
                        "model": model,
                        "rho": diagnostic[f"rho_{model}"],
                        "rho_low": (
                            diagnostic[f"dm_low_{model}"]
                            / diagnostic[f"dm_high_{model}"]
                        ),
                        "logml_error": signed_ln / np.log(10.0),
                        "pmp_error": diagnostic[f"signed_pmp_error_{model}"],
                        "all_high_surprise": globally_high,
                    }
                )
            )
        long = pd.concat(rows, ignore_index=True).merge(
            posterior,
            on=["dataset", "id", "model"],
            how="left",
            validate="one_to_one",
        )
        frames.append(_add_model_match_flag(long))

    data = pd.concat(frames, ignore_index=True)
    data["summary"] = pd.Categorical(
        data["summary"],
        categories=list(diagnostic_by_summary),
        ordered=True,
    )
    return data.sort_values(["model", "summary", "dataset", "id"]).reset_index(
        drop=True
    )


def _lowess_curve(
    x_values: pd.Series,
    y_values: pd.Series,
    frac: float,
    n_grid: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear LOWESS on untransformed rho and y."""
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 3:
        return x, y

    neighbors = min(len(x), max(3, int(np.ceil(frac * len(x)))))
    grid = np.linspace(x.min(), x.max(), n_grid)
    fitted = np.empty_like(grid)
    for index, center in enumerate(grid):
        distance = np.abs(x - center)
        bandwidth = np.partition(distance, neighbors - 1)[neighbors - 1]
        if bandwidth <= np.finfo(float).eps:
            fitted[index] = np.median(y[distance <= np.finfo(float).eps])
            continue
        weights = np.clip(1.0 - (distance / bandwidth) ** 3, 0.0, None) ** 3
        design = np.column_stack([np.ones_like(x), x - center])
        root_weight = np.sqrt(weights)
        fitted[index] = np.linalg.lstsq(
            design * root_weight[:, None], y * root_weight, rcond=None
        )[0][0]
    return grid, np.clip(fitted, y.min(), y.max())


def _symlog_lowess_curve(
    x_values: pd.Series,
    y_values: pd.Series,
    frac: float,
    n_grid: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-linear LOWESS on rho after applying the displayed symlog scale."""
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    transform = SymmetricalLogTransform(
        base=RHO_SYMLOG_BASE,
        linthresh=RHO_SYMLOG_LINTHRESH,
        linscale=RHO_SYMLOG_LINSCALE,
    )
    transformed_grid, fitted = _lowess_curve(
        pd.Series(transform.transform(x)),
        y_values,
        frac=frac,
        n_grid=n_grid,
    )
    return transform.inverted().transform(transformed_grid), fitted


def _limits(values: pd.Series, include: tuple[float, ...] = ()) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    array = np.concatenate([array, np.asarray(include, dtype=float)])
    if not len(array):
        return -1.0, 1.0
    lower, upper = float(array.min()), float(array.max())
    span = max(upper - lower, np.finfo(float).eps)
    return lower - 0.07 * span, upper + 0.07 * span


def _central_interval(
    data: pd.DataFrame,
    value_column: str,
    group_columns: list[str],
    coverage: float,
) -> pd.DataFrame:
    """Keep the central interval independently within every plotted curve."""
    tail = (1.0 - coverage) / 2.0
    grouped = data.groupby(group_columns, observed=True)[value_column]
    lower = grouped.transform(lambda values: values.quantile(tail))
    upper = grouped.transform(lambda values: values.quantile(1.0 - tail))
    return data.loc[data[value_column].between(lower, upper)].copy()


def _summary_handles(
    overlay_order: tuple[str, ...],
    overlay_colors: dict[str, str],
    *,
    show_line: bool = True,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color=overlay_colors[label] if show_line else "none",
            markerfacecolor=overlay_colors[label],
            markeredgecolor="black" if not show_line else overlay_colors[label],
            markeredgewidth=0.7,
            linewidth=2.0 if show_line else 0.0,
            markersize=8 if not show_line else 6,
            label=label,
        )
        for label in overlay_order
    ]


def _scatter_model_matches(
    ax: plt.Axes,
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    *,
    alpha: float,
    default_size: float = 20,
) -> None:
    """Use one color and a distinct marker for model-matched simulations."""
    matched = (
        data["model_matched"].fillna(False).astype(bool)
        if "model_matched" in data
        else pd.Series(False, index=data.index)
    )
    for is_matched, points in data.groupby(matched, sort=False):
        style = MODEL_MATCH_STYLES[bool(is_matched)]
        ax.scatter(
            points[x_column],
            points[y_column],
            s=style["size"] if is_matched else default_size,
            marker=style["marker"],
            color=color,
            alpha=alpha,
            edgecolors="black" if is_matched else "none",
            linewidths=0.7 if is_matched else 0.0,
            zorder=2,
        )


def _model_match_handles(data: pd.DataFrame) -> list[Line2D]:
    if "model_matched" not in data or not data["model_matched"].any():
        return []
    return [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor="0.4",
            markeredgecolor="black" if is_matched else "none",
            markeredgewidth=0.8 if is_matched else 0.0,
            markersize=np.sqrt(style["size"]),
            label=style["label"],
        )
        for is_matched, style in MODEL_MATCH_STYLES.items()
    ]


def _add_pmp_extremum(
    ax: plt.Axes,
    data: pd.DataFrame,
    color: str,
) -> None:
    within = data.loc[data["rho"].le(1.0)].dropna(subset=["pmp_error"])
    if within.empty:
        return
    value = float(within.loc[within["pmp_error"].abs().idxmax(), "pmp_error"])
    ax.axhline(
        value,
        color=color,
        linestyle="--",
        linewidth=0.9,
        alpha=0.8,
        zorder=1,
    )


def _calibrated_panel_thresholds(
    panel: pd.DataFrame,
    threshold_column: str,
    overlay_order: tuple[str, ...],
    overlay_column: str,
    symmetric: bool,
) -> dict[str, tuple[float, ...]]:
    """Return the single calibrated threshold assigned to each plotted overlay."""
    output = {}
    for label in overlay_order:
        values = (
            pd.to_numeric(
                panel.loc[panel[overlay_column].eq(label), threshold_column],
                errors="coerce",
            )
            .dropna()
            .unique()
        )
        if len(values) != 1:
            raise ValueError(
                f"Expected one {threshold_column} for {label}; found {len(values)}"
            )
        threshold = float(values[0])
        output[label] = (-threshold, threshold) if symmetric else (threshold,)
    return output


def plot_rho_error_overlay(
    data: pd.DataFrame,
    models: tuple[str, ...],
    value: str,
    source_group: str,
    path: str | Path,
    *,
    diagnostic: str,
    overlay_order: tuple[str, ...] = tuple(SUMMARY_COLORS),
    overlay_colors: dict[str, str] = SUMMARY_COLORS,
    overlay_column: str = "summary",
    loess_frac: float | None = None,
    show_loess: bool = True,
    calibrated_thresholds: bool = False,
) -> Path:
    """Plot colored overlays, optionally with LOWESS curves."""
    if diagnostic not in DIAGNOSTIC_XLABELS:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")
    if value not in METRIC_PLOTS:
        raise ValueError(f"Unknown value: {value}")
    plot_specs = CALIBRATED_METRIC_PLOTS if calibrated_thresholds else METRIC_PLOTS
    spec = plot_specs[value]
    y_column = spec["column"]
    thresholds = (
        tuple(spec.get("common_thresholds", ()))
        if calibrated_thresholds
        else tuple(spec["thresholds"])
    )
    threshold_column = spec.get("threshold_column")
    all_thresholds: tuple[float, ...] = thresholds
    if calibrated_thresholds and threshold_column is not None:
        if threshold_column not in data:
            raise ValueError(f"Plot data has no {threshold_column}")
        positive = tuple(
            pd.to_numeric(data[threshold_column], errors="coerce").dropna().unique()
        )
        all_thresholds = (
            tuple(value for threshold in positive for value in (-threshold, threshold))
            if spec["symmetric_threshold"]
            else positive
        )
    frac = LOWESS_FRAC[source_group] if loess_frac is None else loess_frac
    reference_data = data.loc[
        data["dataset"].ne("empirical")
        if source_group == "simulated"
        else data["dataset"].eq("empirical")
    ].copy()
    plot_data = (
        _central_interval(
            reference_data,
            y_column,
            ["model", overlay_column],
            LOGML_CENTRAL_INTERVAL,
        )
        if value == "logml"
        else reference_data
    )

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(
            M1_M3_COMPARISON_FIGSIZE if len(models) == 2 else (4.0 * len(models), 3.35)
        ),
        sharey=spec["sharey"],
        squeeze=False,
    )
    axes = axes[0]
    x_limits = _limits(
        pd.concat([reference_data["rho"], reference_data["rho_low"]]),
        include=(1.0,),
    )
    shared_y = (
        _limits(plot_data[y_column], include=all_thresholds)
        if spec["sharey"]
        else None
    )

    for ax, model in zip(axes, models, strict=True):
        reference_panel = reference_data.loc[reference_data["model"].eq(model)]
        panel = plot_data.loc[plot_data["model"].eq(model)]
        rho_low = (
            reference_panel.groupby(overlay_column, observed=True)["rho_low"]
            .median()
            .reindex(overlay_order)
            .dropna()
        )
        if not rho_low.empty:
            ax.axvspan(
                float(rho_low.min()),
                1.0,
                color=TYPICAL_SET_FILL,
                alpha=0.70,
                zorder=0,
            )
            for label, lower_bound in rho_low.items():
                ax.axvline(
                    float(lower_bound),
                    color=overlay_colors[label],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.9,
                    zorder=1,
                )
        ax.axvline(1.0, color="0.2", linestyle="--", linewidth=1.0, zorder=1)
        panel_thresholds = None
        if calibrated_thresholds and threshold_column is not None:
            panel_thresholds = _calibrated_panel_thresholds(
                reference_panel,
                threshold_column,
                overlay_order,
                overlay_column,
                spec["symmetric_threshold"],
            )
            for label, values in panel_thresholds.items():
                for threshold in values:
                    ax.axhline(
                        threshold,
                        color=overlay_colors[label],
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.9,
                        zorder=1,
                    )
        else:
            for threshold in thresholds:
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle=":",
                    linewidth=1.0,
                    zorder=1,
                )

        for label in overlay_order:
            overlay = panel.loc[panel[overlay_column].eq(label)].dropna(
                subset=["rho", y_column]
            )
            color = overlay_colors[label]
            _scatter_model_matches(
                ax,
                overlay,
                "rho",
                y_column,
                color,
                alpha=0.50,
            )

            if show_loess:
                if value == "logml":
                    curve_x, curve_y = _symlog_lowess_curve(
                        overlay["rho"], overlay[y_column], frac=frac
                    )
                else:
                    curve_x, curve_y = _lowess_curve(
                        overlay["rho"], overlay[y_column], frac=frac
                    )
                ax.plot(curve_x, curve_y, color=color, linewidth=2.0, zorder=3)
            if spec["extrema"]:
                _add_pmp_extremum(ax, overlay, color)

        ax.set_title(MODEL_TITLES[model], fontsize=12)
        if spec["xscale"] == "symlog":
            ax.set_xscale(
                "symlog",
                base=RHO_SYMLOG_BASE,
                linthresh=RHO_SYMLOG_LINTHRESH,
                linscale=RHO_SYMLOG_LINSCALE,
            )
        else:
            ax.set_xscale(spec["xscale"])
        ax.set_xlim(x_limits)
        if shared_y is not None:
            ax.set_ylim(shared_y)
        else:
            panel_limits = all_thresholds
            if panel_thresholds is not None:
                panel_limits = tuple(
                    value
                    for values in panel_thresholds.values()
                    for value in values
                )
            ax.set_ylim(_limits(panel[y_column], include=panel_limits))
        ax.set_xlabel(DIAGNOSTIC_XLABELS[diagnostic], fontsize=11)
        ax.grid(color="0.90", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.supylabel(spec["ylabel"], x=0.01, fontsize=11)
    fig.suptitle(
        "Simulated datasets" if source_group == "simulated" else "Empirical dataset",
        y=0.99,
        fontsize=12,
    )
    handles = _summary_handles(
        overlay_order,
        overlay_colors,
        show_line=show_loess,
    )
    handles.extend(_model_match_handles(plot_data))
    handles.append(
        Patch(
            facecolor=TYPICAL_SET_FILL,
            edgecolor="none",
            alpha=0.70,
            label="typical set",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(handles), 4 if len(models) == 2 else 7),
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.10 if len(models) == 2 else 0.07,
        right=0.99,
        bottom=0.31 if len(models) == 2 else 0.24,
        top=0.83,
        wspace=0.22,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _default_output_root(metric: str, models: tuple[str, ...]) -> Path:
    return (
        RESULT_DIR / "plots" / "summary_diagnostics" / _model_set_name(models) / metric
    )


def run_summary_dimension_comparison(
    models: tuple[str, ...] = MODELS,
    metric: str = "l2",
    summary_specs: SummarySpecs = WITH_MMD_SUMMARY_SPECS,
    output_root: str | Path | None = None,
    pmp_source: str | None = None,
    cached_frames: tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
    | None = None,
    calibration_threshold_path: str | Path | None = None,
) -> dict[str, object]:
    """Generate the requested three figures from existing saved results."""
    source = pmp_source or _infer_pmp_source(models)
    diagnostics, posteriors = cached_frames or _load_cached_frames(
        metric, summary_specs
    )
    calibrated = calibration_threshold_path is not None
    data = prepare_rho_error_data(
        diagnostics,
        posteriors,
        models=models,
        pmp_source=source,
        posterior_mmd_column="observed_mmd",
    )
    if calibrated:
        data = _attach_calibration_thresholds(
            data,
            summary_specs,
            calibration_threshold_path,
        )
    output_root = (
        Path(output_root) if output_root else _default_output_root(metric, models)
    )
    classification = None
    classification_path = None
    if calibrated:
        classification = calculate_diagnostic_classification(data, metric)
        classification_path = output_root / "classification_metrics.csv"
        classification_path.parent.mkdir(parents=True, exist_ok=True)
        classification.to_csv(classification_path, index=False)

    manifest = []
    for source_group in ("simulated", "empirical"):
        plot_specs = CALIBRATED_METRIC_PLOTS if calibrated else METRIC_PLOTS
        for value, spec in plot_specs.items():
            path = plot_rho_error_overlay(
                data,
                models,
                value,
                source_group,
                output_root / source_group / spec["filename"],
                diagnostic=metric,
                calibrated_thresholds=calibrated,
            )
            manifest.append(
                {
                    "source_group": source_group,
                    "metric": value,
                    "kind": "figure",
                    "path": str(path),
                }
            )
    return {
        "manifest": pd.DataFrame(manifest),
        "models": tuple(models),
        "pmp_source": source,
        "output_root": output_root,
        "data": data,
        "classification": classification,
        "classification_path": classification_path,
    }


def run_comparison_pipeline(
    metrics: tuple[str, ...] = ("l2", "linf", "mmd", "density"),
    model_sets: dict[str, tuple[str, ...]] = MODEL_SETS,
    summary_specs: SummarySpecs = WITH_MMD_SUMMARY_SPECS,
    output_root: str | Path | None = None,
    calibration_threshold_path: str | Path | None = None,
) -> dict[str, dict[str, object]]:
    """Create four-model and M1/M3 figures while loading each cache only once."""
    comparisons: dict[str, dict[str, object]] = {}
    for metric in metrics:
        cached = _load_cached_frames(metric, summary_specs)
        for name, models in model_sets.items():
            pmp_source = (
                "four_model"
                if calibration_threshold_path is not None
                else PMP_SOURCES[name]
            )
            comparisons[f"{name}_{metric}"] = run_summary_dimension_comparison(
                models=models,
                metric=metric,
                summary_specs=summary_specs,
                pmp_source=pmp_source,
                cached_frames=cached,
                calibration_threshold_path=calibration_threshold_path,
                output_root=(
                    Path(output_root) / name / metric
                    if output_root is not None
                    else None
                ),
            )
    return comparisons


def run_calibrated_comparison_pipeline(
    metrics: tuple[str, ...] = ("l2", "linf", "mmd", "density"),
    model_sets: dict[str, tuple[str, ...]] = MODEL_SETS,
    summary_specs: SummarySpecs = NO_MMD_SUMMARY_SPECS,
    output_root: str | Path | None = None,
    threshold_path: str | Path = CALIBRATION_THRESHOLD_PATH,
) -> dict[str, dict[str, object]]:
    """Run the original comparison layout with well-specified 95% thresholds."""
    return run_comparison_pipeline(
        metrics=metrics,
        model_sets=model_sets,
        summary_specs=summary_specs,
        output_root=output_root,
        calibration_threshold_path=threshold_path,
    )


def display_summary_dimension_comparison(
    comparison: dict[str, object],
    width: int | None = None,
) -> None:
    """Display generated figures inside a Jupyter notebook."""
    from IPython.display import Image, Markdown, display

    manifest = comparison["manifest"]
    display_width = width or (800 if len(comparison["models"]) == 2 else None)
    for source_group in manifest["source_group"].drop_duplicates():
        display(Markdown(f"## {source_group.capitalize()} datasets"))
        for path in manifest.loc[manifest["source_group"].eq(source_group), "path"]:
            display(
                Image(filename=str(path), width=display_width)
                if display_width
                else Image(filename=str(path))
            )


def _training_variants(summary_multiplier: int) -> SummarySpecs:
    if summary_multiplier not in SUMMARY_LABELS:
        raise ValueError(
            f"summary_multiplier must be one of {tuple(SUMMARY_LABELS)}; "
            f"got {summary_multiplier!r}"
        )
    return (
        (
            "With MMD loss",
            TrainingConfig(summary_multiplier=summary_multiplier),
        ),
        (
            "Without MMD loss",
            TrainingConfig(
                summary_multiplier=summary_multiplier,
                summary_base_distribution=None,
                run_suffix="noMMD",
            ),
        ),
    )


def run_training_variant_comparison(
    metric: str,
    output_root: str | Path | None = None,
    summary_multiplier: int = 4,
) -> dict[str, object]:
    """Plot cached with/without-MMD-loss results using the common plotter."""
    if metric not in TRAINING_METRICS:
        raise ValueError(f"Unknown diagnostic metric: {metric}")

    variants = _training_variants(summary_multiplier)
    diagnostics, posteriors = load_summary_dimension_data(
        metric=metric,
        models=MODELS,
        summary_specs=variants,
        pmp_source="four_model",
    )
    data = prepare_rho_error_data(
        diagnostics,
        posteriors,
        models=MODELS,
        pmp_source="four_model",
    )
    root = (
        Path(
            output_root
            or RESULT_DIR / "plots" / f"S{summary_multiplier}D_MMD_vs_noMMD_same_style"
        )
        / metric
    )
    labels = tuple(label for label, _ in variants)
    manifest = []
    for source_group in ("simulated", "empirical"):
        for value, spec in METRIC_PLOTS.items():
            path = plot_rho_error_overlay(
                data,
                MODELS,
                value,
                source_group,
                root / source_group / spec["filename"],
                diagnostic=metric,
                overlay_order=labels,
                overlay_colors=TRAINING_VARIANT_COLORS,
                overlay_column="summary",
            )
            manifest.append(
                {
                    "metric": metric,
                    "source_group": source_group,
                    "kind": "figure",
                    "path": str(path),
                }
            )
    return {
        "manifest": pd.DataFrame(manifest),
        "output_root": root,
        "summary_multiplier": summary_multiplier,
        "data": data,
    }


def run_training_variant_suite(
    metrics: tuple[str, ...] = TRAINING_METRICS,
    output_root: str | Path | None = None,
    summary_multiplier: int = 4,
) -> dict[str, dict[str, object]]:
    return {
        metric: run_training_variant_comparison(
            metric,
            output_root=output_root,
            summary_multiplier=summary_multiplier,
        )
        for metric in metrics
    }


def display_training_variant_suite(
    comparisons: dict[str, dict[str, object]],
) -> None:
    from IPython.display import Image, Markdown, display

    for metric, comparison in comparisons.items():
        display(Markdown(f"# {TRAINING_METRIC_LABELS[metric]} reference diagnostic"))
        manifest = comparison["manifest"]
        for source_group in manifest["source_group"].drop_duplicates():
            display(Markdown(f"## {source_group.capitalize()} datasets"))
            for path in manifest.loc[manifest["source_group"].eq(source_group), "path"]:
                display(Image(filename=str(path)))


# Diagnostic visualization suite used by notebooks/loess_regression.ipynb.
VISUALIZATION_MODELS = ("m1", "m3")
DIAGNOSTICS = ("density", "l2", "linf", "mmd")
VISUALIZATION_SUMMARY_LABELS = tuple(SUMMARY_COLORS)
VISUALIZATION_SOURCES = (
    "simulated_from_m0",
    "simulated_from_m1",
    "simulated_from_m2",
    "simulated_from_m3",
    "m3_fast_30",
    "m3_slow_30",
    "m3_fast_slow_30",
)
LOSS_VARIANTS = {
    "with_mmd": ("With MMD loss", WITH_MMD_SUMMARY_SPECS),
    "without_mmd": ("Without MMD loss", NO_MMD_SUMMARY_SPECS),
}
DIAGNOSTIC_LABELS = {
    "density": "Density",
    "l2": r"$L_2$",
    "linf": r"$L_\infty$",
    "mmd": "MMD",
}
DIAGNOSTIC_XLABELS = {
    "density": "Diagnostic: density-based",
    "l2": r"Diagnostic: $L_2$-based",
    "linf": r"Diagnostic: $L_\infty$-based",
    "mmd": "Diagnostic: MMD-based",
}
VISUALIZATION_VALUE_SPECS = {
    "observed_mmd": {
        "ylabel": "Posterior MMD",
        "bounds": (),
        "signed": False,
        "nonnegative": True,
        "independent_y": False,
    },
    "signed_logml_error": {
        "ylabel": r"$\log\widehat{p}(y\mid M_j)-\log p(y\mid M_j)$",
        "bounds": (-LOGML_ERROR_BOUND, LOGML_ERROR_BOUND),
        "signed": True,
        "nonnegative": False,
        "independent_y": True,
    },
    "signed_pmp_error": {
        "ylabel": r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        "bounds": (-PMP_ERROR_BOUND, PMP_ERROR_BOUND),
        "signed": True,
        "nonnegative": False,
        "independent_y": True,
    },
    "absolute_logml_error": {
        "ylabel": (
            r"$\left|\log\widehat{p}(y\mid M_j)"
            r"-\log p(y\mid M_j)\right|$"
        ),
        "bounds": (LOGML_ERROR_BOUND,),
        "signed": False,
        "nonnegative": True,
        "independent_y": True,
    },
    "absolute_pmp_error": {
        "ylabel": r"$\left|\widehat{p}(M_j\mid y)-p(M_j\mid y)\right|$",
        "bounds": (PMP_ERROR_BOUND,),
        "signed": False,
        "nonnegative": True,
        "independent_y": True,
    },
}
VISUALIZATION_PLOT_SPECS = (
    ("posterior_mmd", "observed_mmd", False, None),
    ("logml_error", "signed_logml_error", False, None),
    ("pmp_error", "signed_pmp_error", False, None),
    ("logml_error_trimmed", "signed_logml_error", True, None),
    ("pmp_error_trimmed", "signed_pmp_error", True, None),
    ("logml_absolute_all_linear", "absolute_logml_error", False, None),
    ("logml_absolute_all_symlog", "absolute_logml_error", False, 1.0),
    ("logml_absolute_trimmed_linear", "absolute_logml_error", True, None),
    ("logml_absolute_trimmed_symlog", "absolute_logml_error", True, 1.0),
    ("pmp_absolute_all_linear", "absolute_pmp_error", False, None),
    ("pmp_absolute_trimmed_linear", "absolute_pmp_error", True, None),
)
GOLD_PMP_GRID_KEY = "pmp_error_vs_rho_colored_by_gold_pmp"
SIGNED_PMP_KEYS = tuple(
    name for name, value, *_ in VISUALIZATION_PLOT_SPECS if value == "signed_pmp_error"
) + (GOLD_PMP_GRID_KEY,)
ABSOLUTE_LOGML_KEYS = tuple(
    name
    for name, value, *_ in VISUALIZATION_PLOT_SPECS
    if value == "absolute_logml_error"
)
ABSOLUTE_PMP_KEYS = tuple(
    name
    for name, value, *_ in VISUALIZATION_PLOT_SPECS
    if value == "absolute_pmp_error"
)
VISUALIZATION_SCATTER_ALPHA = 0.4
VISUALIZATION_LOWESS_FRAC = 0.7
VISUALIZATION_TRIM_QUANTILES = (0.05, 0.95)
VISUALIZATION_Y_PADDING = 0.06
VISUALIZATION_NONNEGATIVE_LOWER_PADDING = 0.03
VISUALIZATION_ABSOLUTE_ERROR_YMIN = -0.1
GOLD_PMP_POINT_STYLES = {
    False: {"marker": "o", "size": 44, "label": "all high surprise"},
    True: {
        "marker": "D",
        "size": 64,
        "label": "at least one not high surprise",
    },
}
GOLD_PMP_LOWESS_LINEWIDTH = 3.0
GOLD_PMP_HIGH_CONTRAST_CMAP = "turbo"
EMPIRICAL_GRID_ROW_STRIP_WIDTH = 0.075
SHOW_GOLD_PMP_MAX_ERROR = True
SHOW_GOLD_PMP_BOUNDARY_HIT = False


def load_visualization_data(
    diagnostic: str,
    summary_specs: SummarySpecs,
    models: tuple[str, ...] = VISUALIZATION_MODELS,
    sources: tuple[str, ...] = VISUALIZATION_SOURCES,
    posterior_mmd_column: str = "observed_mmd",
) -> pd.DataFrame:
    """Load participant-level diagnostics for the visualization suite."""
    if diagnostic not in DIAGNOSTICS:
        raise ValueError(f"diagnostic must be one of {DIAGNOSTICS}")
    diagnostic_by_summary, posterior_by_summary = load_summary_dimension_data(
        metric=diagnostic,
        models=models,
        summary_specs=summary_specs,
    )
    frames = []
    for summary in VISUALIZATION_SUMMARY_LABELS:
        if posterior_mmd_column not in posterior_by_summary[summary]:
            raise ValueError(
                f"{summary} posterior cache has no {posterior_mmd_column}"
            )
        posterior_columns = ["dataset", "id", "model", posterior_mmd_column]
        posterior = posterior_by_summary[summary].loc[
            lambda frame: frame["model"].isin(models),
            posterior_columns,
        ]
        pmp = pmp_long(diagnostic_by_summary[summary], models)[
            [
                "dataset",
                "id",
                "model",
                "rho",
                "rho_low",
                "gold_pmp",
                "signed_pmp_error",
                "at_least_one_not_high_surprise",
            ]
        ]
        logml = logml_long(diagnostic_by_summary[summary], models)[
            ["dataset", "id", "model", "signed_logml_error"]
        ]
        frames.append(
            posterior.merge(
                pmp,
                on=["dataset", "id", "model"],
                how="left",
                validate="one_to_one",
            )
            .merge(
                logml,
                on=["dataset", "id", "model"],
                how="left",
                validate="one_to_one",
            )
            .assign(summary=summary)
        )
    data = _add_model_match_flag(pd.concat(frames, ignore_index=True))
    data = data.loc[data["dataset"].isin(sources)].copy()
    data["summary"] = pd.Categorical(
        data["summary"],
        categories=VISUALIZATION_SUMMARY_LABELS,
        ordered=True,
    )
    return data.sort_values(["model", "summary", "dataset", "id"]).reset_index(
        drop=True
    )


def _save_visualization_figure(
    fig: plt.Figure,
    output_stem: str | Path,
    dpi: int = 220,
) -> dict[str, Path]:
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"png": stem.with_suffix(".png"), "pdf": stem.with_suffix(".pdf")}
    fig.savefig(paths["png"], dpi=dpi)
    fig.savefig(paths["pdf"])
    return paths


def _visualization_limits(
    values: pd.Series,
    *,
    include: tuple[float, ...] = (),
    nonnegative: bool = False,
) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    array = np.concatenate((array, np.asarray(include, dtype=float)))
    if not len(array):
        return -1.0, 1.0
    lower, upper = float(array.min()), float(array.max())
    span = max(upper - lower, np.finfo(float).eps)
    if nonnegative:
        return (
            -VISUALIZATION_NONNEGATIVE_LOWER_PADDING * span,
            upper + VISUALIZATION_Y_PADDING * span,
        )
    return (
        lower - VISUALIZATION_Y_PADDING * span,
        upper + VISUALIZATION_Y_PADDING * span,
    )


def _trim_visualization_extremes(
    data: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    grouped = data.groupby(["model", "summary"], observed=True)[value_column]
    lower = grouped.transform(
        lambda values: values.quantile(VISUALIZATION_TRIM_QUANTILES[0])
    )
    upper = grouped.transform(
        lambda values: values.quantile(VISUALIZATION_TRIM_QUANTILES[1])
    )
    return data.loc[data[value_column].between(lower, upper)].copy()


def _prepare_visualization_plot_data(
    data: pd.DataFrame,
    value_column: str,
    trimmed: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signed_column = {
        "absolute_logml_error": "signed_logml_error",
        "absolute_pmp_error": "signed_pmp_error",
    }.get(value_column)
    reference = (
        data.assign(**{value_column: data[signed_column].abs()})
        if signed_column
        else data
    )
    plot_data = (
        _trim_visualization_extremes(reference, value_column) if trimmed else reference
    )
    return plot_data, reference


def _tidy_visualization_symlog_ticks(
    ax: plt.Axes,
    y_limits: tuple[float, float],
    linthresh: float,
) -> None:
    lower, upper = y_limits
    if upper < linthresh:
        positive = MaxNLocator(nbins=4).tick_values(0.0, upper)
    else:
        positive = 10.0 ** np.arange(
            0,
            np.floor(np.log10(upper)).astype(int) + 1,
        )
    ticks = np.concatenate(([lower, 0.0], positive))
    ax.set_yticks(np.unique(ticks[(ticks >= lower) & (ticks <= upper)]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def _visualization_summary_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            color=SUMMARY_COLORS[summary],
            markerfacecolor=SUMMARY_COLORS[summary],
            linewidth=2.0,
            label=summary,
        )
        for summary in VISUALIZATION_SUMMARY_LABELS
    ]


def plot_visualization_overlay(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    diagnostic: str,
    loss_label: str,
    value_column: str,
    y_symlog_linthresh: float | None,
    output_stem: str | Path,
) -> tuple[plt.Figure, dict[str, Path]]:
    """Plot summary-colored scatter and LOWESS pooled across sources."""
    value_spec = VISUALIZATION_VALUE_SPECS[value_column]
    bounds = value_spec["bounds"]
    independent_y = value_spec["independent_y"]
    fig, axes = plt.subplots(
        1,
        len(VISUALIZATION_MODELS),
        figsize=M1_M3_COMPARISON_FIGSIZE,
        sharey=not independent_y,
    )
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.3, top=0.86, wspace=0.16)
    rho_limits = _visualization_limits(reference_data["rho"])
    shared_y_limits = (
        None
        if independent_y
        else _visualization_limits(
            data[value_column],
            include=bounds,
            nonnegative=value_spec["nonnegative"],
        )
    )
    for ax, model in zip(axes, VISUALIZATION_MODELS, strict=True):
        panel = data.loc[data["model"].eq(model)].dropna(
            subset=["rho", "rho_low", value_column]
        )
        reference_panel = reference_data.loc[reference_data["model"].eq(model)].dropna(
            subset=["rho", "rho_low"]
        )
        y_limits = (
            _visualization_limits(
                panel[value_column],
                nonnegative=value_spec["nonnegative"],
            )
            if independent_y
            else shared_y_limits
        )
        if value_column.startswith("absolute_"):
            y_limits = (VISUALIZATION_ABSOLUTE_ERROR_YMIN, y_limits[1])
        rho_low = (
            reference_panel.groupby("summary", observed=True)["rho_low"]
            .median()
            .reindex(VISUALIZATION_SUMMARY_LABELS)
        )
        ax.axvspan(
            float(rho_low.min()),
            1.0,
            color=TYPICAL_SET_FILL,
            alpha=0.70,
            zorder=0,
        )
        ax.axvline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        if value_spec["signed"]:
            ax.axhline(0.0, color="0.35", linewidth=0.8)
        for bound in bounds:
            ax.axhline(bound, color="#7A0276", linestyle=":", linewidth=1.0)
        for summary in VISUALIZATION_SUMMARY_LABELS:
            summary_data = panel.loc[panel["summary"].eq(summary)]
            color = SUMMARY_COLORS[summary]
            ax.axvline(
                float(rho_low[summary]),
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.8,
            )
            _scatter_model_matches(
                ax,
                summary_data,
                "rho",
                value_column,
                color,
                alpha=VISUALIZATION_SCATTER_ALPHA,
                default_size=17,
            )
            curve_x, curve_y = _lowess_curve(
                summary_data["rho"],
                summary_data[value_column],
                frac=VISUALIZATION_LOWESS_FRAC,
            )
            ax.plot(curve_x, curve_y, color=color, linewidth=2.0, zorder=3)
        ax.set_title(
            rf"Assumed $M_{{{model.removeprefix('m')}}}$",
            fontsize=12,
        )
        ax.set_xscale("symlog", linthresh=1.0)
        if y_symlog_linthresh is not None:
            ax.set_yscale("symlog", linthresh=y_symlog_linthresh)
        ax.set_xlim(rho_limits)
        ax.set_ylim(y_limits)
        if value_column == "absolute_logml_error" and y_symlog_linthresh is not None:
            _tidy_visualization_symlog_ticks(
                ax,
                y_limits,
                y_symlog_linthresh,
            )
        ax.grid(color="0.88", linewidth=0.6, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(loss_label, y=0.98, fontsize=11)
    fig.supylabel(value_spec["ylabel"], x=0.025, fontsize=12)
    fig.supxlabel(DIAGNOSTIC_XLABELS[diagnostic], y=0.10, fontsize=12)
    fig.legend(
        handles=[
            *_visualization_summary_handles(),
            *_model_match_handles(data),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
    )
    return fig, _save_visualization_figure(fig, output_stem)


def _symlog_padded_limits(
    values: pd.Series,
    linthresh: float = 1.0,
    padding: float = 0.08,
) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return -linthresh, linthresh
    lower, upper = float(array.min()), float(array.max())
    return (
        lower - padding * max(abs(lower), linthresh),
        upper + padding * max(abs(upper), linthresh),
    )


def _aligned_symlog_limits_by_group(
    data: pd.DataFrame,
    group_column: str,
    group_order: tuple[str, ...],
    value_columns: tuple[str, ...] = ("rho", "rho_low"),
    anchor: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """Give each group its own limits while aligning one symlog anchor."""
    transform = SymmetricalLogTransform(
        base=RHO_SYMLOG_BASE,
        linthresh=RHO_SYMLOG_LINTHRESH,
        linscale=RHO_SYMLOG_LINSCALE,
    )
    all_values = pd.concat(
        [*(data[column] for column in value_columns), pd.Series([anchor])],
        ignore_index=True,
    )
    global_limits = _symlog_padded_limits(all_values)
    global_transformed = transform.transform(np.asarray(global_limits))
    transformed_anchor = float(transform.transform(np.asarray([anchor]))[0])
    anchor_fraction = (transformed_anchor - float(global_transformed[0])) / (
        float(global_transformed[1]) - float(global_transformed[0])
    )
    anchor_fraction = float(np.clip(anchor_fraction, 0.15, 0.85))

    limits = {}
    for group in group_order:
        group_data = data.loc[data[group_column].eq(group)]
        group_values = pd.concat(
            [
                *(group_data[column] for column in value_columns),
                pd.Series([anchor]),
            ],
            ignore_index=True,
        )
        padded_limits = _symlog_padded_limits(group_values)
        transformed_limits = transform.transform(np.asarray(padded_limits))
        left_span = transformed_anchor - float(transformed_limits[0])
        right_span = float(transformed_limits[1]) - transformed_anchor
        total_span = max(
            left_span / anchor_fraction,
            right_span / (1.0 - anchor_fraction),
            np.finfo(float).eps,
        )
        aligned_transformed = np.asarray(
            [
                transformed_anchor - anchor_fraction * total_span,
                transformed_anchor + (1.0 - anchor_fraction) * total_span,
            ]
        )
        aligned_limits = transform.inverted().transform(aligned_transformed)
        limits[group] = (float(aligned_limits[0]), float(aligned_limits[1]))
    return limits


def _gold_pmp_grid_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color="none",
            markerfacecolor="0.5",
            markeredgecolor="none",
            markersize=np.sqrt(style["size"]),
            label=style["label"],
        )
        for style in GOLD_PMP_POINT_STYLES.values()
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="black",
            markeredgecolor="none",
            markersize=7,
            label="empirical",
        )
    )
    return handles


def _add_gold_pmp_max_error(
    ax: plt.Axes,
    panel: pd.DataFrame,
    value_column: str = "signed_pmp_error",
) -> None:
    """Mark the largest absolute signed PMP error among rows with rho <= 1."""
    within_typical = panel.loc[panel["rho"].le(1.0)]
    if within_typical.empty:
        return
    extreme = within_typical.loc[within_typical[value_column].abs().idxmax()]
    extreme_y = float(extreme[value_column])
    ax.axhline(
        extreme_y,
        color="#7A0276",
        linestyle="--",
        linewidth=1.1,
        zorder=1,
    )
    ax.text(
        0.02,
        extreme_y,
        f"{extreme_y:.2g}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        color="#7A0276",
        fontsize=11,
        clip_on=False,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.65,
            "pad": 1.0,
        },
    )


def plot_gold_pmp_colored_pmp_error_grid(
    data: pd.DataFrame,
    diagnostic: str,
    output_stem: str | Path,
    *,
    threshold_column: str | None = None,
    value_column: str = "signed_pmp_error",
    fixed_threshold: float = PMP_ERROR_BOUND,
    ylabel: str = r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
) -> dict[str, Path]:
    """Plot the empirical PMP-error grid with one pooled LOWESS per panel."""
    if diagnostic not in DIAGNOSTIC_LABELS:
        raise ValueError(f"diagnostic must be one of {tuple(DIAGNOSTIC_LABELS)}")
    required = {
        "summary",
        "model",
        "rho",
        "rho_low",
        "gold_pmp",
        value_column,
        "at_least_one_not_high_surprise",
    }
    if threshold_column is not None:
        required.add(threshold_column)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Gold-PMP grid is missing columns: {missing}")
    plot_data = data.loc[
        data["summary"].isin(VISUALIZATION_SUMMARY_LABELS)
        & data["model"].isin(VISUALIZATION_MODELS)
    ].copy()
    if plot_data.empty:
        raise ValueError("Gold-PMP grid has no data to plot")

    fig, axes = plt.subplots(
        len(VISUALIZATION_SUMMARY_LABELS),
        len(VISUALIZATION_MODELS),
        figsize=(9, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.80,
        bottom=0.16,
        top=0.88,
        wspace=0.28,
        hspace=0.15,
    )
    x_values = pd.concat(
        [plot_data["rho"], plot_data["rho_low"], pd.Series([1.0])],
        ignore_index=True,
    )
    x_limits = _symlog_padded_limits(x_values)
    threshold_values = (
        pd.to_numeric(plot_data[threshold_column], errors="coerce").dropna().unique()
        if threshold_column
        else np.asarray([fixed_threshold])
    )
    y_limits = _visualization_limits(
        plot_data[value_column],
        include=tuple(
            [0.0]
            + [value for threshold in threshold_values for value in (-threshold, threshold)]
        ),
    )
    last_scatter = None
    for row, summary in enumerate(VISUALIZATION_SUMMARY_LABELS):
        for column, model in enumerate(VISUALIZATION_MODELS):
            ax = axes[row, column]
            panel = plot_data.loc[
                plot_data["summary"].eq(summary) & plot_data["model"].eq(model)
            ].dropna(subset=["rho", "rho_low", "gold_pmp", value_column])
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
            panel_threshold = fixed_threshold
            if threshold_column is not None and not panel.empty:
                values = pd.to_numeric(panel[threshold_column], errors="coerce").dropna().unique()
                if len(values) != 1:
                    raise ValueError(
                        f"Expected one {threshold_column} for {summary}/{model}; "
                        f"found {len(values)}"
                    )
                panel_threshold = float(values[0])
            for threshold in (-panel_threshold, panel_threshold):
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
            if not panel.empty:
                rho_low = float(panel["rho_low"].median())
                ax.axvspan(
                    rho_low,
                    1.0,
                    color=TYPICAL_SET_FILL,
                    alpha=0.55,
                    zorder=0,
                )
                ax.axvline(
                    1.0,
                    color="black",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
                if SHOW_GOLD_PMP_MAX_ERROR:
                    _add_gold_pmp_max_error(ax, panel, value_column)
                if SHOW_GOLD_PMP_BOUNDARY_HIT:
                    boundary_hits = panel.loc[
                        panel[value_column].abs().gt(panel_threshold)
                    ].sort_values("rho")
                    if not boundary_hits.empty:
                        boundary_x = float(boundary_hits.iloc[0]["rho"])
                        ax.axvline(
                            boundary_x,
                            color="#1300E0",
                            linestyle=":",
                            linewidth=1.5,
                            zorder=1,
                        )
                        ax.text(
                            boundary_x,
                            0.03,
                            f"{boundary_x:.2g}",
                            transform=ax.get_xaxis_transform(),
                            ha="center",
                            va="bottom",
                            rotation=90,
                            color="#1300E0",
                            fontsize=11,
                            clip_on=True,
                            bbox={
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.70,
                                "pad": 1.0,
                            },
                        )
                for flag, style in GOLD_PMP_POINT_STYLES.items():
                    points = panel.loc[panel["at_least_one_not_high_surprise"].eq(flag)]
                    if points.empty:
                        continue
                    last_scatter = ax.scatter(
                        points["rho"],
                        points[value_column],
                        c=points["gold_pmp"],
                        cmap=GOLD_PMP_HIGH_CONTRAST_CMAP,
                        vmin=0.0,
                        vmax=1.0,
                        s=style["size"],
                        marker=style["marker"],
                        alpha=0.65,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=3,
                    )
                curve_x, curve_y = _lowess_curve(
                    panel["rho"],
                    panel[value_column],
                    frac=VISUALIZATION_LOWESS_FRAC,
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    color="black",
                    linestyle="--",
                    linewidth=GOLD_PMP_LOWESS_LINEWIDTH,
                    zorder=4,
                )
            ax.grid(alpha=0.18)
            ax.tick_params(
                labelsize=14,
                axis="x",
                labelbottom=row == len(VISUALIZATION_SUMMARY_LABELS) - 1,
            )
            ax.set_xlabel(
                DIAGNOSTIC_XLABELS[diagnostic]
                if row == len(VISUALIZATION_SUMMARY_LABELS) - 1
                else "",
                fontsize=13,
            )

    fig.canvas.draw()
    for ax, model in zip(axes[0], VISUALIZATION_MODELS, strict=True):
        position = ax.get_position()
        strip = fig.add_axes([position.x0, position.y1 + 0.006, position.width, 0.046])
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            rf"Assumed $M_{{{model.removeprefix('m')}}}$",
            ha="center",
            va="center",
            fontsize=20,
        )
        strip.set_xticks([])
        strip.set_yticks([])
    for ax, summary in zip(
        axes[:, -1],
        VISUALIZATION_SUMMARY_LABELS,
        strict=True,
    ):
        position = ax.get_position()
        strip = fig.add_axes(
            [
                position.x1 + 0.006,
                position.y0,
                EMPIRICAL_GRID_ROW_STRIP_WIDTH,
                position.height,
            ]
        )
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            f"{summary}\n{DIAGNOSTIC_LABELS[diagnostic]}",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=18,
        )
        strip.set_xticks([])
        strip.set_yticks([])
    fig.supylabel(ylabel, x=0.015, fontsize=18)
    fig.legend(
        handles=_gold_pmp_grid_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    if last_scatter is not None:
        colorbar_ax = fig.add_axes([0.90, 0.20, 0.016, 0.62])
        colorbar = fig.colorbar(last_scatter, cax=colorbar_ax)
        colorbar.set_label("Gold-standard PMP", fontsize=16)
        colorbar.ax.tick_params(labelsize=14)
    paths = _save_visualization_figure(fig, output_stem)
    plt.close(fig)
    return paths


def generate_diagnostic_visualization_suite(
    output_dir: str | Path,
    diagnostic: str,
    loss_variant: str,
) -> dict[str, dict[str, Path]]:
    """Generate the retained plot suite for one diagnostic and loss variant."""
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"loss_variant must be one of {tuple(LOSS_VARIANTS)}")
    loss_label, summary_specs = LOSS_VARIANTS[loss_variant]
    data = load_visualization_data(diagnostic, summary_specs)
    outputs = {}
    for index, (name, value_column, trimmed, linthresh) in enumerate(
        VISUALIZATION_PLOT_SPECS,
        start=1,
    ):
        plot_data, reference_data = _prepare_visualization_plot_data(
            data,
            value_column,
            trimmed,
        )
        fig, outputs[name] = plot_visualization_overlay(
            plot_data,
            reference_data,
            diagnostic,
            loss_label,
            value_column,
            linthresh,
            Path(output_dir) / f"{index:02d}_{name}_overlay",
        )
        plt.close(fig)
    empirical_data = load_visualization_data(
        diagnostic,
        summary_specs,
        sources=("empirical",),
    )
    outputs[GOLD_PMP_GRID_KEY] = plot_gold_pmp_colored_pmp_error_grid(
        empirical_data,
        diagnostic,
        Path(output_dir)
        / f"{len(VISUALIZATION_PLOT_SPECS) + 1:02d}_{GOLD_PMP_GRID_KEY}",
    )
    return outputs


def generate_gold_pmp_lowess_grid(
    diagnostic: str,
    loss_variant: str,
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Generate only the empirical 4x2 Gold-PMP-colored LOWESS grid."""
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"loss_variant must be one of {tuple(LOSS_VARIANTS)}")
    _, summary_specs = LOSS_VARIANTS[loss_variant]
    empirical_data = load_visualization_data(
        diagnostic,
        summary_specs,
        sources=("empirical",),
    )
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "diagnostic_overlays"
    )
    output_stem = root / loss_variant / diagnostic / f"12_{GOLD_PMP_GRID_KEY}"
    return plot_gold_pmp_colored_pmp_error_grid(
        empirical_data,
        diagnostic,
        output_stem,
    )


def display_gold_pmp_lowess_grid(
    paths: dict[str, Path],
    width: int | None = 800,
) -> None:
    """Display the generated empirical Gold-PMP LOWESS grid in a notebook."""
    from IPython.display import Image, Markdown, display

    display(Markdown("## Empirical PMP error vs. diagnostic $\\rho$"))
    display(
        Image(filename=str(paths["png"]), width=width)
        if width
        else Image(filename=str(paths["png"]))
    )


S4D_DIAGNOSTIC_ORDER = ("l2", "linf", "density", "mmd")
S4D_SUMMARY_LABEL = "S=4D"


def load_s4d_empirical_diagnostic_data(
    loss_variant: str = "without_mmd",
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
) -> pd.DataFrame:
    """Load empirical S=4D rows for several diagnostics into one table."""
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"loss_variant must be one of {tuple(LOSS_VARIANTS)}")
    unknown = tuple(
        diagnostic for diagnostic in diagnostics if diagnostic not in DIAGNOSTICS
    )
    if unknown:
        raise ValueError(f"Unknown diagnostics: {unknown}")
    _, summary_specs = LOSS_VARIANTS[loss_variant]
    frames = []
    for diagnostic in diagnostics:
        frame = load_visualization_data(
            diagnostic,
            summary_specs,
            sources=("empirical",),
        )
        frames.append(
            frame.loc[frame["summary"].eq(S4D_SUMMARY_LABEL)]
            .copy()
            .assign(diagnostic=diagnostic)
        )
    data = pd.concat(frames, ignore_index=True)
    data["diagnostic"] = pd.Categorical(
        data["diagnostic"],
        categories=diagnostics,
        ordered=True,
    )
    return data.sort_values(["diagnostic", "model", "dataset", "id"]).reset_index(
        drop=True
    )


def plot_s4d_diagnostic_gold_pmp_grid(
    data: pd.DataFrame,
    output_stem: str | Path,
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
    gold_pmp_cmap: str = GOLD_PMP_HIGH_CONTRAST_CMAP,
    show_max_error: bool = False,
) -> dict[str, Path]:
    """Plot a 4x2 empirical grid: S=4D only, with one row per diagnostic."""
    required = {
        "diagnostic",
        "summary",
        "model",
        "rho",
        "rho_low",
        "gold_pmp",
        "signed_pmp_error",
        "at_least_one_not_high_surprise",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"S=4D diagnostic grid is missing columns: {missing}")
    plot_data = data.loc[
        data["summary"].eq(S4D_SUMMARY_LABEL)
        & data["diagnostic"].isin(diagnostics)
        & data["model"].isin(VISUALIZATION_MODELS)
    ].copy()
    if plot_data.empty:
        raise ValueError("S=4D diagnostic grid has no data to plot")

    fig, axes = plt.subplots(
        len(diagnostics),
        len(VISUALIZATION_MODELS),
        figsize=(9, 13),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.80,
        bottom=0.16,
        top=0.88,
        wspace=0.28,
        hspace=0.48,
    )
    panel_order = tuple(
        f"{diagnostic}|{model}"
        for diagnostic in diagnostics
        for model in VISUALIZATION_MODELS
    )
    limit_data = plot_data.assign(
        _rho_panel=(
            plot_data["diagnostic"].astype("string")
            + "|"
            + plot_data["model"].astype("string")
        )
    )
    x_limits_by_panel = _aligned_symlog_limits_by_group(
        limit_data,
        "_rho_panel",
        panel_order,
    )
    y_limits = _visualization_limits(
        plot_data["signed_pmp_error"],
        include=(-PMP_ERROR_BOUND, 0.0, PMP_ERROR_BOUND),
    )
    norm = Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap(gold_pmp_cmap)

    for row, diagnostic in enumerate(diagnostics):
        for column, model in enumerate(VISUALIZATION_MODELS):
            ax = axes[row, column]
            panel = plot_data.loc[
                plot_data["diagnostic"].eq(diagnostic) & plot_data["model"].eq(model)
            ].dropna(subset=["rho", "rho_low", "gold_pmp", "signed_pmp_error"])
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlim(x_limits_by_panel[f"{diagnostic}|{model}"])
            ax.set_ylim(y_limits)
            ax.axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
            for threshold in (-PMP_ERROR_BOUND, PMP_ERROR_BOUND):
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=1,
                )
            if not panel.empty:
                ax.axvspan(
                    float(panel["rho_low"].median()),
                    1.0,
                    color=TYPICAL_SET_FILL,
                    alpha=0.55,
                    zorder=0,
                )
                if show_max_error:
                    _add_gold_pmp_max_error(ax, panel)
                for flag, style in GOLD_PMP_POINT_STYLES.items():
                    points = panel.loc[panel["at_least_one_not_high_surprise"].eq(flag)]
                    if points.empty:
                        continue
                    ax.scatter(
                        points["rho"],
                        points["signed_pmp_error"],
                        c=points["gold_pmp"],
                        cmap=cmap,
                        norm=norm,
                        s=style["size"],
                        marker=style["marker"],
                        alpha=0.78,
                        edgecolors="black",
                        linewidths=0.5,
                        zorder=3,
                    )
                curve_x, curve_y = _lowess_curve(
                    panel["rho"],
                    panel["signed_pmp_error"],
                    frac=VISUALIZATION_LOWESS_FRAC,
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    color="black",
                    linestyle="--",
                    linewidth=GOLD_PMP_LOWESS_LINEWIDTH,
                    zorder=4,
                )
            ax.axvline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            ax.grid(alpha=0.18)
            ax.tick_params(
                labelsize=14,
                axis="x",
                labelbottom=True,
            )
            ax.set_xlabel(
                DIAGNOSTIC_XLABELS[diagnostic],
                fontsize=12,
            )

    fig.canvas.draw()
    for ax, model in zip(axes[0], VISUALIZATION_MODELS, strict=True):
        position = ax.get_position()
        strip = fig.add_axes([position.x0, position.y1 + 0.006, position.width, 0.046])
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            rf"Assumed $M_{{{model.removeprefix('m')}}}$",
            ha="center",
            va="center",
            fontsize=20,
        )
        strip.set_xticks([])
        strip.set_yticks([])
    for ax, diagnostic in zip(axes[:, -1], diagnostics, strict=True):
        position = ax.get_position()
        strip = fig.add_axes(
            [
                position.x1 + 0.006,
                position.y0,
                EMPIRICAL_GRID_ROW_STRIP_WIDTH,
                position.height,
            ]
        )
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            f"{S4D_SUMMARY_LABEL}\n{DIAGNOSTIC_LABELS[diagnostic]}",
            ha="center",
            va="center",
            rotation=-90,
            fontsize=18,
        )
        strip.set_xticks([])
        strip.set_yticks([])

    fig.supylabel(
        r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        x=0.015,
        fontsize=18,
    )
    fig.legend(
        handles=_gold_pmp_grid_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    colorbar_ax = fig.add_axes([0.90, 0.20, 0.016, 0.62])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=colorbar_ax,
    )
    colorbar.set_label("Gold-standard PMP", fontsize=16)
    colorbar.ax.tick_params(labelsize=14)
    paths = _save_visualization_figure(fig, output_stem)
    plt.close(fig)
    return paths


def generate_s4d_diagnostic_gold_pmp_grid(
    loss_variant: str = "without_mmd",
    output_root: str | Path | None = None,
    gold_pmp_cmap: str = GOLD_PMP_HIGH_CONTRAST_CMAP,
    show_max_error: bool = False,
) -> dict[str, Path]:
    """Load data and generate the S=4D empirical four-diagnostic PMP grid."""
    data = load_s4d_empirical_diagnostic_data(loss_variant=loss_variant)
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "s4d_diagnostic_comparison"
    )
    return plot_s4d_diagnostic_gold_pmp_grid(
        data,
        root / loss_variant / "s4d_gold_pmp_by_diagnostic",
        gold_pmp_cmap=gold_pmp_cmap,
        show_max_error=show_max_error,
    )


def load_empirical_diagnostic_comparison_data(
    loss_variant: str = "without_mmd",
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
) -> pd.DataFrame:
    """Load all empirical summary dimensions for a diagnostic comparison."""
    if loss_variant not in LOSS_VARIANTS:
        raise ValueError(f"loss_variant must be one of {tuple(LOSS_VARIANTS)}")
    unknown = tuple(
        diagnostic for diagnostic in diagnostics if diagnostic not in DIAGNOSTICS
    )
    if unknown:
        raise ValueError(f"Unknown diagnostics: {unknown}")
    _, summary_specs = LOSS_VARIANTS[loss_variant]
    frames = []
    for diagnostic in diagnostics:
        frames.append(
            load_visualization_data(
                diagnostic,
                summary_specs,
                sources=("empirical",),
            ).assign(diagnostic=diagnostic)
        )
    data = pd.concat(frames, ignore_index=True).assign(
        log10_logml_error=lambda frame: frame["signed_logml_error"] / np.log(10.0)
    )
    data["diagnostic"] = pd.Categorical(
        data["diagnostic"],
        categories=diagnostics,
        ordered=True,
    )
    return data.sort_values(
        ["diagnostic", "model", "summary", "dataset", "id"]
    ).reset_index(drop=True)


def _plot_empirical_metric_diagnostic_loess_grid(
    data: pd.DataFrame,
    output_stem: str | Path,
    value_column: str,
    ylabel: str,
    thresholds: tuple[float, ...],
    *,
    central_interval: float | None = None,
    loess_on_symlog_x: bool = False,
    nonnegative: bool = False,
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
    loess_frac: float = LOWESS_FRAC["empirical"],
) -> dict[str, Path]:
    """Plot a 4x2 empirical metric grid with colored summary LOWESS curves."""
    required = {
        "diagnostic",
        "summary",
        "model",
        "rho",
        "rho_low",
        value_column,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Diagnostic LOWESS grid is missing columns: {missing}")
    reference_data = data.loc[
        data["diagnostic"].isin(diagnostics)
        & data["summary"].isin(VISUALIZATION_SUMMARY_LABELS)
        & data["model"].isin(VISUALIZATION_MODELS)
    ].copy()
    plot_data = (
        _central_interval(
            reference_data,
            value_column,
            ["diagnostic", "model", "summary"],
            central_interval,
        )
        if central_interval is not None
        else reference_data
    )
    if plot_data.empty:
        raise ValueError("Diagnostic LOWESS grid has no data to plot")

    fig, axes = plt.subplots(
        len(diagnostics),
        len(VISUALIZATION_MODELS),
        figsize=(9, 13),
        sharex=False,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.14,
        right=0.88,
        bottom=0.16,
        top=0.88,
        wspace=0.20,
        hspace=0.48,
    )
    panel_order = tuple(
        f"{diagnostic}|{model}"
        for diagnostic in diagnostics
        for model in VISUALIZATION_MODELS
    )
    limit_data = reference_data.assign(
        _rho_panel=(
            reference_data["diagnostic"].astype("string")
            + "|"
            + reference_data["model"].astype("string")
        )
    )
    x_limits_by_panel = _aligned_symlog_limits_by_group(
        limit_data,
        "_rho_panel",
        panel_order,
    )
    y_limits = _visualization_limits(
        plot_data[value_column],
        include=(*thresholds, 0.0),
        nonnegative=nonnegative,
    )

    for row, diagnostic in enumerate(diagnostics):
        for column, model in enumerate(VISUALIZATION_MODELS):
            ax = axes[row, column]
            reference_panel = reference_data.loc[
                reference_data["diagnostic"].eq(diagnostic)
                & reference_data["model"].eq(model)
            ]
            panel = plot_data.loc[
                plot_data["diagnostic"].eq(diagnostic) & plot_data["model"].eq(model)
            ]
            rho_low = (
                reference_panel.groupby("summary", observed=True)["rho_low"]
                .median()
                .reindex(VISUALIZATION_SUMMARY_LABELS)
                .dropna()
            )
            if not rho_low.empty:
                ax.axvspan(
                    float(rho_low.min()),
                    1.0,
                    color=TYPICAL_SET_FILL,
                    alpha=0.70,
                    zorder=0,
                )
                for summary, lower_bound in rho_low.items():
                    ax.axvline(
                        float(lower_bound),
                        color=SUMMARY_COLORS[summary],
                        linestyle="--",
                        linewidth=0.9,
                        alpha=0.9,
                        zorder=1,
                    )
            ax.axvline(
                1.0,
                color="0.2",
                linestyle="--",
                linewidth=1.0,
                zorder=1,
            )
            for threshold in thresholds:
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle=":",
                    linewidth=1.0,
                    zorder=1,
                )
            ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)

            for summary in VISUALIZATION_SUMMARY_LABELS:
                points = panel.loc[panel["summary"].eq(summary)].dropna(
                    subset=["rho", value_column]
                )
                if points.empty:
                    continue
                color = SUMMARY_COLORS[summary]
                ax.scatter(
                    points["rho"],
                    points[value_column],
                    s=20,
                    color=color,
                    alpha=0.50,
                    edgecolors="none",
                    zorder=2,
                )
                curve_x, curve_y = (
                    _symlog_lowess_curve(
                        points["rho"],
                        points[value_column],
                        frac=loess_frac,
                    )
                    if loess_on_symlog_x
                    else _lowess_curve(
                        points["rho"],
                        points[value_column],
                        frac=loess_frac,
                    )
                )
                ax.plot(
                    curve_x,
                    curve_y,
                    color=color,
                    linewidth=2.0,
                    zorder=3,
                )

            ax.set_xscale(
                "symlog",
                base=RHO_SYMLOG_BASE,
                linthresh=RHO_SYMLOG_LINTHRESH,
                linscale=RHO_SYMLOG_LINSCALE,
            )
            ax.set_xlim(x_limits_by_panel[f"{diagnostic}|{model}"])
            ax.set_ylim(y_limits)
            ax.grid(color="0.90", linewidth=0.6, alpha=0.7)
            ax.tick_params(
                labelsize=14,
                axis="x",
                labelbottom=True,
            )
            ax.set_xlabel(
                DIAGNOSTIC_XLABELS[diagnostic],
                fontsize=12,
            )

    fig.canvas.draw()
    for ax, model in zip(axes[0], VISUALIZATION_MODELS, strict=True):
        position = ax.get_position()
        strip = fig.add_axes([position.x0, position.y1 + 0.006, position.width, 0.046])
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            rf"Assumed $M_{{{model.removeprefix('m')}}}$",
            ha="center",
            va="center",
            fontsize=20,
        )
        strip.set_xticks([])
        strip.set_yticks([])
    for ax, diagnostic in zip(axes[:, -1], diagnostics, strict=True):
        position = ax.get_position()
        strip = fig.add_axes(
            [
                position.x1 + 0.006,
                position.y0,
                EMPIRICAL_GRID_ROW_STRIP_WIDTH,
                position.height,
            ]
        )
        strip.set_facecolor("#D9D9D9")
        strip.text(
            0.5,
            0.5,
            DIAGNOSTIC_LABELS[diagnostic],
            ha="center",
            va="center",
            rotation=-90,
            fontsize=18,
        )
        strip.set_xticks([])
        strip.set_yticks([])

    fig.supylabel(ylabel, x=0.015, fontsize=18)
    handles = _summary_handles(
        VISUALIZATION_SUMMARY_LABELS,
        SUMMARY_COLORS,
        show_line=True,
    )
    handles.append(
        Patch(
            facecolor=TYPICAL_SET_FILL,
            edgecolor="none",
            alpha=0.70,
            label="typical set",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        fontsize=14,
    )
    paths = _save_visualization_figure(fig, output_stem)
    plt.close(fig)
    return paths


def plot_log10_logml_diagnostic_loess_grid(
    data: pd.DataFrame,
    output_stem: str | Path,
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
    loess_frac: float = LOWESS_FRAC["empirical"],
) -> dict[str, Path]:
    """Plot empirical base-10 logML error for four diagnostics."""
    return _plot_empirical_metric_diagnostic_loess_grid(
        data,
        output_stem,
        "log10_logml_error",
        (
            r"$\log_{10}\widehat{p}(y\mid M_j)"
            r"-\log_{10}p(y\mid M_j)$"
        ),
        (-LOGML_ERROR_BOUND, LOGML_ERROR_BOUND),
        central_interval=LOGML_CENTRAL_INTERVAL,
        loess_on_symlog_x=True,
        diagnostics=diagnostics,
        loess_frac=loess_frac,
    )


def plot_posterior_mmd_diagnostic_loess_grid(
    data: pd.DataFrame,
    output_stem: str | Path,
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
    loess_frac: float = LOWESS_FRAC["empirical"],
) -> dict[str, Path]:
    """Plot empirical raw posterior MMD for four diagnostics."""
    return _plot_empirical_metric_diagnostic_loess_grid(
        data,
        output_stem,
        "observed_mmd",
        "Posterior MMD",
        (),
        nonnegative=True,
        diagnostics=diagnostics,
        loess_frac=loess_frac,
    )


def plot_pmp_diagnostic_loess_grid(
    data: pd.DataFrame,
    output_stem: str | Path,
    diagnostics: tuple[str, ...] = S4D_DIAGNOSTIC_ORDER,
    loess_frac: float = LOWESS_FRAC["empirical"],
) -> dict[str, Path]:
    """Plot empirical signed PMP error for four diagnostics."""
    return _plot_empirical_metric_diagnostic_loess_grid(
        data,
        output_stem,
        "signed_pmp_error",
        r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        (-PMP_ERROR_BOUND, PMP_ERROR_BOUND),
        diagnostics=diagnostics,
        loess_frac=loess_frac,
    )


def generate_log10_logml_diagnostic_loess_grid(
    loss_variant: str = "without_mmd",
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Load data and generate the four-diagnostic empirical log10-ML grid."""
    data = load_empirical_diagnostic_comparison_data(loss_variant=loss_variant)
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "s4d_diagnostic_comparison"
    )
    return plot_log10_logml_diagnostic_loess_grid(
        data,
        root / loss_variant / "log10_logml_error_by_diagnostic",
    )


def generate_posterior_mmd_diagnostic_loess_grid(
    loss_variant: str = "without_mmd",
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Load data and generate the four-diagnostic posterior-MMD grid."""
    data = load_empirical_diagnostic_comparison_data(loss_variant=loss_variant)
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "s4d_diagnostic_comparison"
    )
    return plot_posterior_mmd_diagnostic_loess_grid(
        data,
        root / loss_variant / "posterior_mmd_by_diagnostic",
    )


def generate_pmp_diagnostic_loess_grid(
    loss_variant: str = "without_mmd",
    output_root: str | Path | None = None,
) -> dict[str, Path]:
    """Load data and generate the four-diagnostic empirical PMP-error grid."""
    data = load_empirical_diagnostic_comparison_data(loss_variant=loss_variant)
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "s4d_diagnostic_comparison"
    )
    return plot_pmp_diagnostic_loess_grid(
        data,
        root / loss_variant / "pmp_error_by_diagnostic",
    )


def load_calibrated_visualization_data(
    diagnostic: str,
    summary_specs: SummarySpecs,
    threshold_path: str | Path = CALIBRATION_THRESHOLD_PATH,
    sources: tuple[str, ...] = ("empirical",),
) -> pd.DataFrame:
    """Load observed metrics and attach matching well-specified thresholds."""
    data = load_visualization_data(
        diagnostic,
        summary_specs,
        models=MODELS,
        sources=sources,
        posterior_mmd_column="observed_mmd",
    )
    data = data.loc[data["model"].isin(VISUALIZATION_MODELS)].copy()
    data["logml_error"] = data["signed_logml_error"] / np.log(10.0)
    data["pmp_error"] = data["signed_pmp_error"]
    data = _attach_calibration_thresholds(
        data,
        summary_specs,
        threshold_path,
    )
    if "observed_mmd" not in data:
        raise ValueError("Posterior cache has no observed_mmd")
    return data.sort_values(["model", "summary", "id"]).reset_index(drop=True)


def generate_calibrated_gold_pmp_lowess_grid(
    diagnostic: str,
    output_root: str | Path | None = None,
    threshold_path: str | Path = CALIBRATION_THRESHOLD_PATH,
    summary_specs: SummarySpecs = NO_MMD_SUMMARY_SPECS,
    output_variant: str = "without_mmd",
) -> dict[str, Path]:
    """Generate the original empirical PMP grid with calibrated PMP cutoffs."""
    data = load_calibrated_visualization_data(
        diagnostic,
        summary_specs,
        threshold_path=threshold_path,
    )
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "diagnostic_overlays_calibrated_noMMD"
    )
    return plot_gold_pmp_colored_pmp_error_grid(
        data,
        diagnostic,
        root / output_variant / diagnostic / f"12_{GOLD_PMP_GRID_KEY}",
    )


def generate_all_diagnostic_visualizations(
    output_root: str | Path,
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    loss_variants: tuple[str, ...] = tuple(LOSS_VARIANTS),
) -> dict[str, dict[str, dict[str, dict[str, Path]]]]:
    """Generate every diagnostic × loss-variant visualization suite."""
    output_root = Path(output_root)
    return {
        loss_variant: {
            diagnostic: generate_diagnostic_visualization_suite(
                output_root / loss_variant / diagnostic,
                diagnostic,
                loss_variant,
            )
            for diagnostic in diagnostics
        }
        for loss_variant in loss_variants
    }
