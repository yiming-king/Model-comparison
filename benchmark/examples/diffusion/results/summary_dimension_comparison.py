"""Rho-vs-error plots across diffusion summary dimensions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import FuncNorm, Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FuncFormatter, MaxNLocator
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from ..config import BASE_DIR, MODELS, MODEL_TITLES, RESULT_DIR, TrainingConfig
from .multisource_pipeline import all_observed_paths, reference_suite_path
from .posterior_diagnostic import load_posterior_diagnostic
from .summary_diagnostic import load_references


SummarySpecs = tuple[tuple[str, TrainingConfig], ...]
AxisScaleSpec = str | Mapping[str, str]

SUMMARY_LABELS = {1: "S=D", 2: "S=2D", 4: "S=4D", 6: "S=6D"}
SUMMARY_COLORS = {
    "S=D":  "#0072B2",
    "S=2D": "#E69F00",
    "S=4D": "#CC79A7",
    "S=6D": "#009E73",
}
PLOT_SUMMARY_LABELS = ("S=D", "S=2D", "S=4D")
LEGACY_CALIBRATED_FIGURE_FILENAMES = {
    "combined_normalized_logml_error_vs_rho.png",
    "combined_normalized_pmp_error_vs_rho.png",
}



def _make_summary_specs(
    summary_base_distribution: str | None = "normal",
    run_suffix: str | None = None,
    embed_dim: int = 64,
    epochs: int = 100,
    summary_multipliers: tuple[int, ...] = (1, 2, 4),
) -> SummarySpecs:
    return tuple(
        (
            SUMMARY_LABELS[multiplier],
            TrainingConfig(
                summary_multiplier=multiplier,
                embed_dim=embed_dim,
                epochs=epochs,
                summary_base_distribution=summary_base_distribution,
                run_suffix=run_suffix,
            ),
        )
        for multiplier in summary_multipliers
    )


WITH_MMD_SUMMARY_SPECS = _make_summary_specs()
NO_MMD_SUMMARY_SPECS = _make_summary_specs(
    summary_base_distribution=None,
    run_suffix="noMMD",
)

MODEL_SETS = {"all": MODELS, "m1_m3": ("m1", "m3")}
PMP_SOURCES = {"all": "four_model", "m1_m3": "m1_m3"}

# These values control only the linear region of symlog axes. They are not
# diagnostic thresholds; all diagnostic bounds come from NPE--MCMC references.
PMP_SYMLOG_LINTHRESH = 0.1
LOGML_SYMLOG_LINTHRESH = 1.0
RHO_XSCALE = "symlog"
RHO_SYMLOG_BASE = 10.0
RHO_SYMLOG_LINTHRESH = 1.0
RHO_SYMLOG_LINSCALE = 1.0
AXIS_SCALES = ("linear", "symlog")
LOWESS_FRAC = {"simulated": 0.7, "empirical": 1}
LOGML_CENTRAL_INTERVAL = 0.9
TYPICAL_SET_FILL = "#DCEEDC"
M1_M3_COMPARISON_FIGSIZE = (8.0, 5)
NONNEGATIVE_Y_MARGIN = 0.05
THRESHOLD_LINEWIDTH = 1.8


def _compact_tick_label(value: float, _: int | None = None) -> str:
    """Format axis values compactly without exposing full floating-point limits."""
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    return f"{value:.2g}"


RHO_NORMALIZATIONS = ("centered", "upper_threshold")


def _normalize_distance(x, median, high, method: str = "centered"):
    """Normalize a diagnostic using either the legacy or d / d_high rule."""
    if method not in RHO_NORMALIZATIONS:
        raise ValueError(
            f"rho normalization must be one of {RHO_NORMALIZATIONS}; got {method!r}"
        )
    values = np.asarray(x, dtype=float)
    upper = np.asarray(high, dtype=float)
    if method == "upper_threshold":
        if np.any(np.isclose(upper, 0.0)):
            raise ValueError("Diagnostic upper threshold must be non-zero")
        return values / upper
    center = np.asarray(median, dtype=float)
    scale = upper - center
    if np.any(np.isclose(scale, 0.0)):
        raise ValueError("Diagnostic upper threshold must differ from its median")
    return (values - center) / scale


def _scale_nonnegative_metric(x, high):
    """Scale a non-negative metric so its raw upper threshold maps to one."""
    values = np.asarray(x, dtype=float)
    upper = np.asarray(high, dtype=float)
    if np.any(upper <= 0.0):
        raise ValueError("Non-negative metric upper threshold must be positive")
    return values / upper


def _nonnegative_plot_limits(upper: float) -> tuple[float, float]:
    """Add visual space below zero without changing non-negative data values."""
    upper = float(upper)
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("Non-negative plot upper limit must be positive and finite")
    return -NONNEGATIVE_Y_MARGIN * upper, upper


def _set_axis_scale(
    ax: plt.Axes,
    axis: str,
    scale: str,
    *,
    linthresh: float,
) -> None:
    """Apply a notebook-selectable linear or symmetrical-log axis scale."""
    if scale not in AXIS_SCALES:
        raise ValueError(f"{axis}scale must be one of {AXIS_SCALES}; got {scale!r}")
    if scale == "symlog" and linthresh <= 0.0:
        raise ValueError(f"{axis}_symlog_linthresh must be positive")
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    if scale == "symlog":
        setter(
            "symlog",
            base=RHO_SYMLOG_BASE,
            linthresh=linthresh,
            linscale=RHO_SYMLOG_LINSCALE,
        )
    else:
        setter("linear")


def _resolve_metric_scale(scale: AxisScaleSpec, value: str) -> str:
    """Resolve a shared axis scale or a per-result scale mapping."""
    if isinstance(scale, str):
        return scale
    if value not in scale:
        raise ValueError(f"yscale mapping has no entry for {value!r}")
    return scale[value]


def _scale_signed_by_lower_bound(x, low):
    """Scale a signed metric by the magnitude of its raw lower bound."""
    values = np.asarray(x, dtype=float)
    lower = np.asarray(low, dtype=float)
    scale = np.abs(lower)
    if np.any(scale <= 0.0):
        raise ValueError("Signed-error lower threshold must be non-zero")
    return values / scale


METRIC_NORMALIZERS = {
    "posterior_mmd": _scale_nonnegative_metric,
    "logml": _scale_signed_by_lower_bound,
}


def _normalize_metric(metric: str, values, median, low, high):
    """Apply the independently replaceable display transform for one metric."""
    try:
        normalizer = METRIC_NORMALIZERS[metric]
    except KeyError as error:
        raise ValueError(f"No display normalizer registered for {metric!r}") from error
    if metric == "posterior_mmd":
        return normalizer(values, high)
    return normalizer(values, low)


def _metric_ylabel(base: str, normalized: bool) -> str:
    """Build a display label whose normalization suffix follows the switch."""
    suffix = "(NPE, MCMC; normalized)" if normalized else "(NPE, MCMC)"
    return f"{base}\n{suffix}"


def _threshold_band(thresholds) -> tuple[float, float] | None:
    """Convert the active error thresholds to one shaded acceptance band."""
    values = np.asarray(tuple(thresholds), dtype=float)
    if not len(values):
        return None
    if len(values) == 1:
        return (min(0.0, float(values[0])), max(0.0, float(values[0])))
    return float(values.min()), float(values.max())


def _shade_typical_set(
    ax: plt.Axes,
    x_low: float,
    y_band: tuple[float, float] | None = None,
    alpha: float = 0.70,
) -> None:
    """Shade the diagnostic interval and the active error-threshold band."""
    ax.axvspan(x_low, 1.0, color=TYPICAL_SET_FILL, alpha=alpha, zorder=0)
    if y_band is not None:
        ax.axhspan(*y_band, color=TYPICAL_SET_FILL, alpha=alpha, zorder=0)

MODEL_MATCH_STYLES = {
    False: {"marker": "o", "size": 30, "label": "misspecified datasets"},
    True: {"marker": "s", "size": 40, "label": "well-specified datasets"},
}
TRAINING_METRICS = ("l2", "linf", "mmd", "density")
TRAINING_METRIC_LABELS = {
    "l2": r"$L_2$",
    "linf": r"$L_\infty$",
    "mmd": "Kernel",
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
        "thresholds": (),
        "filename": "combined_logml_error_vs_rho.png",
        "extrema": False,
        "sharey": True,
    },
    "pmp": {
        "column": "pmp_error",
        "ylabel": r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        "xscale": RHO_XSCALE,
        "thresholds": (),
        "filename": "combined_pmp_error_vs_rho.png",
        "extrema": False,
        "sharey": False,
    },
}

CALIBRATION_THRESHOLD_PATH = BASE_DIR / "calibration_outputs_100_noMMD" / "thresholds.csv"
CALIBRATED_METRIC_PLOTS = {
    "posterior_mmd": {
        **METRIC_PLOTS["posterior_mmd"],
        "column": "normalized_mmd",
        "ylabel": _metric_ylabel("Posterior MMD", normalized=True),
        "threshold_columns": (
            "normalized_mmd_low",
            "normalized_mmd_high",
        ),
        "common_thresholds": (),
        "filename": "combined_normalized_posterior_mmd_vs_rho.png",
    },
    "logml": {
        **METRIC_PLOTS["logml"],
        "column": "normalized_logml_error",
        "ylabel": _metric_ylabel(METRIC_PLOTS["logml"]["ylabel"], normalized=True),
        "threshold_columns": (
            "normalized_logml_error_low",
            "normalized_logml_error_high",
        ),
        "common_thresholds": (),
    },
    "pmp": {
        **METRIC_PLOTS["pmp"],
        "ylabel": _metric_ylabel(METRIC_PLOTS["pmp"]["ylabel"], normalized=False),
        "threshold_columns": (
            "pmp_error_lower_threshold",
            "pmp_error_upper_threshold",
        ),
        "common_thresholds": (),
    },
}

RAW_CALIBRATED_METRIC_PLOTS = {
    "posterior_mmd": {
        **METRIC_PLOTS["posterior_mmd"],
        "ylabel": _metric_ylabel("Posterior MMD", normalized=False),
        "threshold_columns": (
            "posterior_mmd_lower_threshold",
            "posterior_mmd_upper_threshold",
        ),
        "common_thresholds": (),
    },
    "logml": {
        **METRIC_PLOTS["logml"],
        "ylabel": _metric_ylabel(METRIC_PLOTS["logml"]["ylabel"], normalized=False),
        "threshold_columns": (
            "log10_logml_error_lower_threshold",
            "log10_logml_error_upper_threshold",
        ),
        "common_thresholds": (),
    },
    "pmp": {
        **METRIC_PLOTS["pmp"],
        "ylabel": _metric_ylabel(METRIC_PLOTS["pmp"]["ylabel"], normalized=False),
        "threshold_columns": (
            "pmp_error_lower_threshold",
            "pmp_error_upper_threshold",
        ),
        "common_thresholds": (),
    },
}


def _calibrated_plot_specs(normalize_metrics: bool) -> dict[str, dict]:
    """Select display columns without changing raw diagnostic decisions."""
    return CALIBRATED_METRIC_PLOTS if normalize_metrics else RAW_CALIBRATED_METRIC_PLOTS


def load_calibration_thresholds(
    summary_specs: SummarySpecs,
    threshold_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load 100-dataset bounds from each configuration's matching training run."""
    summary_to_config = {
        summary: config.summary_label for summary, config in summary_specs
    }
    if threshold_path is None:
        paths = set()
        for _, config in summary_specs:
            if config.summary_base_distribution is None:
                variant = config.run_suffix or "noMMD"
            elif config.run_suffix is None:
                variant = "withMMD"
            else:
                raise ValueError("Pass threshold_path for custom with-MMD training runs")
            paths.add(BASE_DIR / f"calibration_outputs_100_{variant}" / "thresholds.csv")
        thresholds = pd.concat(
            [pd.read_csv(path, keep_default_na=False) for path in sorted(paths)],
            ignore_index=True,
        )
    else:
        thresholds = pd.read_csv(threshold_path, keep_default_na=False)
    required = {
        "generating_model",
        "npe_configuration",
        "metric",
        "lower_threshold",
        "threshold",
        "median",
    }
    missing = sorted(required.difference(thresholds.columns))
    if missing:
        raise ValueError(f"Calibration threshold table is missing columns: {missing}")
    thresholds = thresholds.loc[
        thresholds["npe_configuration"].isin(summary_to_config.values())
    ]
    threshold_wide = thresholds.pivot(
        index=["generating_model", "npe_configuration"],
        columns="metric",
        values="threshold",
    ).rename(
        columns={
            "posterior_mmd": "posterior_mmd_upper_threshold",
            "signed_logml_error": "signed_logml_error_upper_threshold",
            "signed_pmp_error": "pmp_error_upper_threshold",
        }
    )
    lower_threshold_wide = thresholds.pivot(
        index=["generating_model", "npe_configuration"],
        columns="metric",
        values="lower_threshold",
    ).rename(
        columns={
            "posterior_mmd": "posterior_mmd_lower_threshold",
            "signed_logml_error": "signed_logml_error_lower_threshold",
            "signed_pmp_error": "pmp_error_lower_threshold",
        }
    )
    median_wide = thresholds.pivot(
        index=["generating_model", "npe_configuration"],
        columns="metric",
        values="median",
    ).rename(
        columns={
            "posterior_mmd": "posterior_mmd_median",
            "signed_logml_error": "signed_logml_error_median",
            "signed_pmp_error": "signed_pmp_error_median",
        }
    )
    wide = threshold_wide.join(lower_threshold_wide).join(median_wide)
    threshold_columns = [
        "posterior_mmd_lower_threshold",
        "posterior_mmd_upper_threshold",
        "signed_logml_error_lower_threshold",
        "signed_logml_error_upper_threshold",
        "pmp_error_lower_threshold",
        "pmp_error_upper_threshold",
    ]
    missing_metrics = sorted(set(threshold_columns).difference(wide.columns))
    if missing_metrics:
        raise ValueError(f"Calibration thresholds are missing metrics: {missing_metrics}")
    wide = wide.reset_index().rename(columns={"generating_model": "model"})
    config_to_summary = {config: summary for summary, config in summary_to_config.items()}
    wide["summary"] = wide["npe_configuration"].map(config_to_summary)
    wide["log10_logml_error_lower_threshold"] = (
        wide["signed_logml_error_lower_threshold"] / np.log(10.0)
    )
    wide["log10_logml_error_upper_threshold"] = (
        wide["signed_logml_error_upper_threshold"] / np.log(10.0)
    )
    wide["log10_logml_error_median"] = (
        wide["signed_logml_error_median"] / np.log(10.0)
    )
    output_columns = [
        "model",
        "summary",
        "npe_configuration",
        "posterior_mmd_upper_threshold",
        "posterior_mmd_lower_threshold",
        "log10_logml_error_lower_threshold",
        "log10_logml_error_upper_threshold",
        "pmp_error_lower_threshold",
        "pmp_error_upper_threshold",
        "posterior_mmd_median",
        "log10_logml_error_median",
        "signed_pmp_error_median",
    ]
    output = wide[output_columns].copy()
    numeric_columns = output_columns[3:]
    if output[numeric_columns].isna().any().any():
        raise ValueError("Calibration threshold table contains missing values")
    if output["posterior_mmd_lower_threshold"].ne(0.0).any():
        raise ValueError("Posterior MMD lower thresholds must be zero")
    interval_pairs = (
        ("posterior_mmd_lower_threshold", "posterior_mmd_upper_threshold"),
        ("log10_logml_error_lower_threshold", "log10_logml_error_upper_threshold"),
        ("pmp_error_lower_threshold", "pmp_error_upper_threshold"),
    )
    for low, high in interval_pairs:
        if output[low].ge(output[high]).any():
            raise ValueError(f"Calibration lower threshold must be below {high}")
    return output


def _attach_calibration_thresholds(
    data: pd.DataFrame,
    summary_specs: SummarySpecs,
    threshold_path: str | Path | None = None,
) -> pd.DataFrame:
    data = data.drop(
        columns=[column for column in data if column.startswith("normalized_pmp")]
    )
    thresholds = load_calibration_thresholds(summary_specs, threshold_path)
    output = data.merge(
        thresholds,
        on=["model", "summary"],
        how="left",
        validate="many_to_one",
    )
    threshold_columns = [
        "posterior_mmd_upper_threshold",
        "log10_logml_error_lower_threshold",
        "log10_logml_error_upper_threshold",
        "pmp_error_lower_threshold",
        "pmp_error_upper_threshold",
    ]
    if output[threshold_columns].isna().any().any():
        missing = output.loc[
            output[threshold_columns].isna().any(axis=1), ["model", "summary"]
        ].drop_duplicates()
        raise ValueError(
            "Missing calibration thresholds for:\n" + missing.to_string(index=False)
        )
    if "observed_mmd" in output:
        output["normalized_mmd"] = _normalize_metric(
            "posterior_mmd",
            output["observed_mmd"],
            output["posterior_mmd_median"],
            output["posterior_mmd_lower_threshold"],
            output["posterior_mmd_upper_threshold"],
        )
        output["normalized_mmd_low"] = _normalize_metric(
            "posterior_mmd",
            output["posterior_mmd_lower_threshold"],
            output["posterior_mmd_median"],
            output["posterior_mmd_lower_threshold"],
            output["posterior_mmd_upper_threshold"],
        )
        output["normalized_mmd_high"] = _normalize_metric(
            "posterior_mmd",
            output["posterior_mmd_upper_threshold"],
            output["posterior_mmd_median"],
            output["posterior_mmd_lower_threshold"],
            output["posterior_mmd_upper_threshold"],
        )
    error_specs = {"logml": ("logml_error", "log10_logml_error")}
    for metric, (value_column, prefix) in error_specs.items():
        lower_threshold_column = f"{prefix}_lower_threshold"
        upper_threshold_column = f"{prefix}_upper_threshold"
        median_column = "log10_logml_error_median"
        low = output[lower_threshold_column]
        high = output[upper_threshold_column]
        output[f"normalized_{metric}_error"] = _normalize_metric(
            metric,
            output[value_column], output[median_column], low, high
        )
        output[f"normalized_{metric}_error_low"] = _normalize_metric(
            metric, low, output[median_column], low, high
        )
        output[f"normalized_{metric}_error_high"] = _normalize_metric(
            metric, high, output[median_column], low, high
        )
    return output


CALIBRATED_CLASSIFICATION_SPECS = {
    "posterior_mmd": {
        "value_column": "observed_mmd",
        "bounds": (
            "posterior_mmd_lower_threshold",
            "posterior_mmd_upper_threshold",
        ),
    },
    "logml": {
        "value_column": "logml_error",
        "bounds": (
            "log10_logml_error_lower_threshold",
            "log10_logml_error_upper_threshold",
        ),
    },
    "pmp": {
        "value_column": "pmp_error",
        "bounds": ("pmp_error_lower_threshold", "pmp_error_upper_threshold"),
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
        bound_columns = spec.get("bounds")
        required = {"dataset", "id", "model", "summary", "rho", value_column}
        if bound_columns:
            required.update(bound_columns)
        missing = sorted(required.difference(data.columns))
        if missing:
            raise ValueError(f"Classification data is missing columns: {missing}")
        columns = ["dataset", "id", "model", "summary", "rho", value_column]
        if bound_columns:
            columns.extend(bound_columns)
        frame = data[columns].copy()
        frame = frame.dropna(subset=["rho", value_column])
        frame["diagnostic"] = diagnostic
        frame["source_group"] = np.where(
            frame["dataset"].eq("empirical"), "empirical", "simulated"
        )
        frame["error_metric"] = metric
        frame["error_value"] = frame[value_column]
        if bound_columns:
            low_column, high_column = bound_columns
            frame["error_lower_threshold"] = frame[low_column]
            frame["error_upper_threshold"] = frame[high_column]
            frame["error_positive"] = frame[value_column].lt(
                frame[low_column]
            ) | frame[value_column].gt(frame[high_column])
        else:
            frame["error_lower_threshold"] = 0.0
            frame["error_upper_threshold"] = float(spec["threshold"])
            frame["error_positive"] = frame[value_column].gt(spec["threshold"])
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
                    "error_lower_threshold",
                    "error_upper_threshold",
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
        "error_lower_threshold",
        "error_upper_threshold",
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
    auc = (
        rows.groupby(group_columns, observed=True)
        .apply(
            lambda group: (
                roc_auc_score(group["error_positive"], group["rho"])
                if group["error_positive"].nunique() == 2
                else np.nan
            ),
            include_groups=False,
        )
        .rename("auc")
        .reset_index()
    )
    counts = counts.merge(auc, on=group_columns, how="left", validate="one_to_one")
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
    """Recompute PMP from either all four models or the M1/M3 subset."""
    output = frame.copy()
    if pmp_source == "four_model":
        estimated = softmax(
            output[[f"log_ml_{model}" for model in MODELS]].to_numpy(dtype=float),
            axis=1,
        )
        gold = softmax(
            output[
                [f"gold_log_ml_{model}" for model in MODELS]
            ].to_numpy(dtype=float),
            axis=1,
        )
        for index, model in enumerate(MODELS):
            output[f"pmp_{model}"] = estimated[:, index]
            output[f"gold_pmp_{model}"] = gold[:, index]
            output[f"signed_pmp_error_{model}"] = (
                estimated[:, index] - gold[:, index]
            )
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
    rho_normalization: str = "centered",
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
        diagnostic = pd.read_csv(
            paths["diagnostic"], keep_default_na=False
        ).assign(summary=label)
        references = load_references(reference_suite_path(config.summary_label))
        for model in MODELS:
            median = float(references[model][metric]["median"])
            diagnostic[f"dm_median_{model}"] = median
            diagnostic[f"rho_{model}"] = _normalize_distance(
                diagnostic[f"d_{model}"],
                median,
                diagnostic[f"dm_high_{model}"],
                method=rho_normalization,
            )
            diagnostic[f"rho_low_{model}"] = _normalize_distance(
                diagnostic[f"dm_low_{model}"],
                median,
                diagnostic[f"dm_high_{model}"],
                method=rho_normalization,
            )
        diagnostics[label] = diagnostic
        posteriors[label] = load_posterior_diagnostic(paths["posterior"]).assign(
            summary=label
        )
    return diagnostics, posteriors


def load_summary_dimension_data(
    metric: str = "l2",
    models: tuple[str, ...] = MODELS,
    summary_specs: SummarySpecs = WITH_MMD_SUMMARY_SPECS,
    pmp_source: str | None = None,
    rho_normalization: str = "centered",
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Load existing diagnostic and posterior caches; never refit any model."""
    diagnostics, posteriors = _load_cached_frames(
        metric,
        summary_specs,
        rho_normalization=rho_normalization,
    )
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
                    "rho_low": frame[f"rho_low_{model}"],
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
                    "rho_low": frame[f"rho_low_{model}"],
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
                        "rho_low": diagnostic[f"rho_low_{model}"],
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
    marker: str = "o",
    markersize: float | None = None,
) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=overlay_colors[label] if show_line else "none",
            markerfacecolor=overlay_colors[label],
            markeredgecolor="black" if not show_line else overlay_colors[label],
            markeredgewidth=0.7,
            linewidth=2.0 if show_line else 0.0,
            markersize=(8 if not show_line else 6) if markersize is None else markersize,
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


def _scatter_source_median(
    ax: plt.Axes,
    points: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    *,
    marker: str,
    empirical: bool,
) -> None:
    median_x = pd.to_numeric(points[x_column], errors="coerce").median()
    median_y = pd.to_numeric(points[y_column], errors="coerce").median()
    if np.isfinite(median_x) and np.isfinite(median_y):
        ax.scatter(
            [median_x], [median_y],
            s=150 if empirical else 55,
            marker=marker,
            color=color,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.0 if empirical else 0.8,
            zorder=5 if empirical else 4,
        )


def _scatter_combined_sources(
    ax: plt.Axes,
    points: pd.DataFrame,
    x_column: str,
    y_column: str,
    color: str,
    *,
    alpha: float,
) -> None:
    """Plot source-specific observations and one median per simulated source."""
    empirical = points["dataset"].eq("empirical")
    simulated_points = points.loc[~empirical]
    if not simulated_points.empty:
        matched = simulated_points["model_matched"].fillna(False).astype(bool)
        for is_matched, group in simulated_points.groupby(matched, sort=False):
            ax.scatter(
                group[x_column], group[y_column],
                s=36 if is_matched else 20,
                marker="^" if is_matched else "o",
                facecolors=to_rgba(color, alpha=0.28 if is_matched else alpha),
                edgecolors="none", linewidths=0.0, zorder=2,
            )
        for _, group in simulated_points.groupby(
            simulated_points["dataset"].astype("string"), observed=True, sort=True
        ):
            is_matched = group["model_matched"].fillna(False).astype(bool).all()
            _scatter_source_median(
                ax, group, x_column, y_column, color,
                marker="^" if is_matched else "o", empirical=False,
            )
    empirical_points = points.loc[empirical]
    if not empirical_points.empty:
        ax.scatter(
            empirical_points[x_column], empirical_points[y_column],
            s=60, marker="*",
            facecolors=to_rgba(color, alpha=0.45),
            edgecolors=to_rgba("black", alpha=0.42),
            linewidths=0.6, zorder=2.5,
        )
        _scatter_source_median(
            ax, empirical_points, x_column, y_column, color,
            marker="*", empirical=True,
        )


def _combined_source_handles(
    points: pd.DataFrame,
    *,
    empirical_only: bool,
    empirical_legend_alpha: float,
) -> list[Line2D]:
    handles = []
    if not empirical_only:
        for marker, alpha, median, size, label in (
            ("o", 0.5, False, 6, "other simulated datasets"),
            ("^", 0.28, False, 7, "well-specified datasets"),
            ("o", 1.0, True, 7, "other simulated median"),
            ("^", 1.0, True, 7, "well-specified median"),
        ):
            handles.append(
                Line2D(
                    [0], [0], marker=marker, color="none",
                    markerfacecolor="0.4" if median else to_rgba("0.4", alpha=alpha),
                    markeredgecolor="black" if median else "none",
                    markeredgewidth=0.8 if median else 1.0,
                    markersize=size, label=label,
                )
            )
    if points["dataset"].eq("empirical").any():
        handles.extend(
            [
                Line2D(
                    [0], [0], marker="*", color="none",
                    markerfacecolor=to_rgba(
                        "0.65", alpha=empirical_legend_alpha if empirical_only else 0.45
                    ),
                    markeredgecolor=to_rgba("black", alpha=0.42),
                    markeredgewidth=0.6, markersize=9, label="empirical datasets",
                ),
                Line2D(
                    [0], [0], marker="*", color="none", markerfacecolor="0.4",
                    markeredgecolor="black", markeredgewidth=0.8,
                    markersize=12, label="empirical median",
                ),
            ]
        )
    return handles


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
    threshold_columns: tuple[str, ...],
    overlay_order: tuple[str, ...],
    overlay_column: str,
) -> dict[str, tuple[float, ...]]:
    """Return the calibrated bounds assigned to each plotted overlay."""
    output = {}
    for label in overlay_order:
        bounds = []
        for threshold_column in threshold_columns:
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
            bounds.append(float(values[0]))
        output[label] = tuple(bounds)
    return output


def _select_plot_summaries(
    data: pd.DataFrame,
    overlay_order: tuple[str, ...],
    overlay_column: str,
) -> pd.DataFrame:
    """Keep S=6D in result tables while excluding it from displayed overlays."""
    return data.loc[data[overlay_column].isin(overlay_order)].copy()


def _legend_layout(
    handles: list,
    *,
    summary_count: int,
    model_count: int,
) -> tuple[list, int, int]:
    """Keep three summaries together above three context legend entries."""
    context_count = len(handles) - summary_count
    if model_count == 2 and summary_count == 3 and context_count == 3:
        summaries = handles[:summary_count]
        context = handles[summary_count:]
        interleaved = [item for pair in zip(summaries, context, strict=True) for item in pair]
        return interleaved, 3, 2
    columns = min(len(handles), 4 if model_count == 2 else 7)
    rows = int(np.ceil(len(handles) / columns))
    return handles, columns, rows


def _remove_stale_metric_figures(
    directory: Path,
    current_filenames: set[str],
) -> None:
    """Remove known generated variants that are not part of the active display."""
    known_filenames = {
        *(spec["filename"] for spec in CALIBRATED_METRIC_PLOTS.values()),
        *(spec["filename"] for spec in RAW_CALIBRATED_METRIC_PLOTS.values()),
        *LEGACY_CALIBRATED_FIGURE_FILENAMES,
    }
    for filename in known_filenames.difference(current_filenames):
        (directory / filename).unlink(missing_ok=True)


def _add_filename_suffix(filename: str, suffix: str | None) -> str:
    """Add an optional run identifier without changing the file extension."""
    if not suffix:
        return filename
    path = Path(filename)
    return f"{path.stem}_{suffix}{path.suffix}"


def plot_rho_error_overlay(
    data: pd.DataFrame,
    models: tuple[str, ...],
    value: str,
    source_group: str,
    path: str | Path,
    *,
    diagnostic: str,
    overlay_order: tuple[str, ...] = PLOT_SUMMARY_LABELS,
    overlay_colors: dict[str, str] = SUMMARY_COLORS,
    overlay_column: str = "summary",
    loess_frac: float | None = None,
    show_loess: bool = True,
    calibrated_thresholds: bool = False,
    normalize_metrics: bool = True,
    xscale: str | None = None,
    yscale: str = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = 1.0,
    plot_style: str = "standard",
    show_empirical_only_loess: bool = True,
    include_empirical_in_loess: bool = True,
    loess_frac_by_fit_source: Mapping[str, float] | None = None,
    empirical_legend_alpha: float = 0.65,
) -> Path:
    """Plot standard or combined-source overlays through the shared drawing core."""
    if plot_style not in {"standard", "combined"}:
        raise ValueError(f"Unknown plot style: {plot_style}")
    if diagnostic not in DIAGNOSTIC_XLABELS:
        raise ValueError(f"Unknown diagnostic: {diagnostic}")
    if value not in METRIC_PLOTS:
        raise ValueError(f"Unknown value: {value}")
    plot_specs = (
        _calibrated_plot_specs(normalize_metrics)
        if calibrated_thresholds
        else METRIC_PLOTS
    )
    spec = plot_specs[value]
    y_column = spec["column"]
    thresholds = (
        tuple(spec.get("common_thresholds", ()))
        if calibrated_thresholds
        else tuple(spec["thresholds"])
    )
    threshold_columns = spec.get("threshold_columns")
    threshold_column = spec.get("threshold_column")
    if threshold_columns is None and threshold_column is not None:
        threshold_columns = (threshold_column,)
    all_thresholds: tuple[float, ...] = thresholds
    if calibrated_thresholds and threshold_columns is not None:
        missing = sorted(set(threshold_columns).difference(data.columns))
        if missing:
            raise ValueError(f"Plot data is missing threshold columns: {missing}")
        all_thresholds = tuple(
            value
            for column in threshold_columns
            for value in pd.to_numeric(data[column], errors="coerce").dropna().unique()
        )
    combined = plot_style == "combined"
    empirical_only = combined and source_group == "empirical"
    active_show_loess = show_empirical_only_loess if empirical_only else show_loess
    frac = LOWESS_FRAC[source_group] if loess_frac is None else loess_frac
    if combined and not empirical_only:
        fit_source = (
            "simulated_and_empirical" if include_empirical_in_loess else "simulated_only"
        )
        fractions = {"simulated_and_empirical": 1.0, "simulated_only": 1.0}
        if loess_frac_by_fit_source is not None:
            fractions.update(loess_frac_by_fit_source)
        frac = fractions[fit_source]
        path = Path(path).parent.parent / "simulated_and_empirical" / Path(path).name
    if not 0.0 < frac <= 1.0:
        raise ValueError("LOWESS fraction must be greater than zero and at most one")
    source_data = _select_plot_summaries(data, overlay_order, overlay_column)
    if combined and not empirical_only:
        reference_data = source_data
    else:
        reference_data = source_data.loc[
            source_data["dataset"].ne("empirical")
            if source_group == "simulated"
            else source_data["dataset"].eq("empirical")
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
    simulated_only_fit = combined and not empirical_only and not include_empirical_in_loess
    fit_data = plot_data
    if simulated_only_fit:
        # Fit-only trimming uses the original simulated population, independently
        # of the combined trimming used for visible observations and medians.
        fit_data = reference_data.loc[
            reference_data["dataset"].ne("empirical")
        ].dropna(subset=["rho", y_column])
        if value == "logml":
            fit_data = _central_interval(
                fit_data, y_column, ["model", overlay_column], LOGML_CENTRAL_INTERVAL
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
    if value == "posterior_mmd" and shared_y is not None:
        shared_y = _nonnegative_plot_limits(shared_y[1])

    for ax, model in zip(axes, models, strict=True):
        reference_panel = reference_data.loc[reference_data["model"].eq(model)]
        panel = plot_data.loc[plot_data["model"].eq(model)]
        rho_low = (
            reference_panel.groupby(overlay_column, observed=True)["rho_low"]
            .median()
            .reindex(overlay_order)
            .dropna()
        )
        panel_thresholds = None
        if calibrated_thresholds and threshold_columns is not None:
            panel_thresholds = _calibrated_panel_thresholds(
                reference_panel,
                threshold_columns,
                overlay_order,
                overlay_column,
            )
            active_thresholds = tuple(
                threshold
                for values in panel_thresholds.values()
                for threshold in values
            )
            panel_bounds_vary = len(set(panel_thresholds.values())) > 1
            threshold_sets = (
                ((None, tuple(sorted(set(active_thresholds)))),)
                if normalize_metrics and not panel_bounds_vary
                else tuple(panel_thresholds.items())
            )
            for label, bounds in threshold_sets:
                plotted_bounds = (
                    bounds[1:]
                    if value == "posterior_mmd" and not normalize_metrics
                    else bounds
                )
                for threshold in plotted_bounds:
                    ax.axhline(
                        threshold,
                        color="0.35" if label is None else overlay_colors[label],
                        linestyle=":",
                        linewidth=THRESHOLD_LINEWIDTH,
                        alpha=0.9,
                        zorder=1,
                    )
        else:
            active_thresholds = thresholds
            for threshold in thresholds:
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle=":",
                    linewidth=THRESHOLD_LINEWIDTH,
                    zorder=1,
                )
        if not rho_low.empty:
            y_band = _threshold_band(active_thresholds)
            _shade_typical_set(
                ax,
                float(rho_low.min()),
                y_band,
            )
        if value == "posterior_mmd":
            ax.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
        ax.axvline(1.0, color="0.2", linestyle="--", linewidth=1.0, zorder=1)

        for label in overlay_order:
            overlay = panel.loc[panel[overlay_column].eq(label)].dropna(
                subset=["rho", y_column]
            )
            color = overlay_colors[label]
            scatter = _scatter_combined_sources if combined else _scatter_model_matches
            scatter(
                ax,
                overlay,
                "rho",
                y_column,
                color,
                alpha=0.50,
            )

            if active_show_loess:
                fit_overlay = overlay
                if simulated_only_fit and not overlay.empty:
                    fit_overlay = fit_data.loc[
                        fit_data["model"].eq(model) & fit_data[overlay_column].eq(label)
                    ]
                if value == "logml":
                    curve_x, curve_y = _symlog_lowess_curve(
                        fit_overlay["rho"], fit_overlay[y_column], frac=frac
                    )
                else:
                    curve_x, curve_y = _lowess_curve(
                        fit_overlay["rho"], fit_overlay[y_column], frac=frac
                    )
                ax.plot(curve_x, curve_y, color=color, linewidth=2.0, zorder=3)
            if spec["extrema"]:
                _add_pmp_extremum(ax, overlay, color)

        ax.set_title(MODEL_TITLES[model], fontsize=12)
        active_xscale = spec["xscale"] if xscale is None else xscale
        _set_axis_scale(
            ax,
            "x",
            active_xscale,
            linthresh=x_symlog_linthresh,
        )
        _set_axis_scale(
            ax,
            "y",
            yscale,
            linthresh=y_symlog_linthresh,
        )
        if yscale == "symlog":
            ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))
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
            panel_y_limits = _limits(panel[y_column], include=panel_limits)
            if value == "posterior_mmd":
                panel_y_limits = _nonnegative_plot_limits(panel_y_limits[1])
            ax.set_ylim(panel_y_limits)
        ax.set_xlabel(DIAGNOSTIC_XLABELS[diagnostic], fontsize=11)
        ax.grid(color="0.90", linewidth=0.6, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)

    fig.supylabel(
        spec["ylabel"],
        x=0.04 if len(models) == 2 else 0.025,
        fontsize=11,
        ha="center",
        va="center",
        multialignment="center",
    )
    source_title = "Simulated datasets" if source_group == "simulated" else "Empirical dataset"
    if combined:
        source_title = "Empirical datasets" if empirical_only else "Simulated and empirical datasets"
    fig.suptitle(source_title, y=0.99, fontsize=12)
    summary_handles = _summary_handles(
        overlay_order,
        overlay_colors,
        show_line=active_show_loess and not empirical_only,
        marker="*" if empirical_only else "o",
        markersize=9 if empirical_only else None,
    )
    handles = list(summary_handles)
    handles.extend(
        _combined_source_handles(
            plot_data, empirical_only=empirical_only,
            empirical_legend_alpha=empirical_legend_alpha,
        )
        if combined else _model_match_handles(plot_data)
    )
    handles.append(
        Patch(
            facecolor=TYPICAL_SET_FILL,
            edgecolor="none",
            alpha=0.70,
            label="typical set",
        )
    )
    if combined and not empirical_only and len(models) in {2, 4}:
        legend_columns = len(handles) if len(models) == 4 else 5
        legend_rows = 1 if len(models) == 4 else 2
    else:
        handles, legend_columns, legend_rows = _legend_layout(
            handles,
            summary_count=len(summary_handles),
            model_count=len(models),
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=legend_columns,
        frameon=False,
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.14 if len(models) == 2 else 0.09,
        right=0.99,
        bottom=(0.29 if legend_rows == 2 else 0.24) if len(models) == 2 else 0.24,
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
    xscale: str | None = None,
    yscale: AxisScaleSpec = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = 1.0,
    normalize_metrics: bool = True,
    rho_normalization: str = "centered",
    filename_suffix: str | None = None,
    plot_style: str = "standard",
    show_loess: bool = True,
    show_empirical_only_loess: bool = True,
    include_empirical_in_loess: bool = True,
    loess_frac_by_fit_source: Mapping[str, float] | None = None,
    empirical_legend_alpha: float = 0.65,
) -> dict[str, object]:
    """Generate figures using well-specified NPE--MCMC reference thresholds."""
    if plot_style not in {"standard", "combined"}:
        raise ValueError(f"Unknown plot style: {plot_style}")
    source = pmp_source or _infer_pmp_source(models)
    diagnostics, posteriors = cached_frames or _load_cached_frames(
        metric,
        summary_specs,
        rho_normalization=rho_normalization,
    )
    data = prepare_rho_error_data(
        diagnostics,
        posteriors,
        models=models,
        pmp_source=source,
        posterior_mmd_column="observed_mmd",
    )
    data = _attach_calibration_thresholds(
        data,
        summary_specs,
        calibration_threshold_path,
    )
    output_root = (
        Path(output_root) if output_root else _default_output_root(metric, models)
    )
    classification = calculate_diagnostic_classification(data, metric)
    classification_path = output_root / "classification_metrics.csv"
    classification_path.parent.mkdir(parents=True, exist_ok=True)
    classification.to_csv(classification_path, index=False)

    manifest = []
    for source_group in ("simulated", "empirical"):
        plot_specs = _calibrated_plot_specs(normalize_metrics)
        for value, spec in plot_specs.items():
            figure_filename = _add_filename_suffix(
                spec["filename"],
                filename_suffix,
            )
            path = plot_rho_error_overlay(
                data,
                models,
                value,
                source_group,
                output_root / source_group / figure_filename,
                diagnostic=metric,
                calibrated_thresholds=True,
                normalize_metrics=normalize_metrics,
                xscale=xscale,
                yscale=_resolve_metric_scale(yscale, value),
                x_symlog_linthresh=x_symlog_linthresh,
                y_symlog_linthresh=y_symlog_linthresh,
                plot_style=plot_style,
                show_loess=show_loess,
                show_empirical_only_loess=show_empirical_only_loess,
                include_empirical_in_loess=include_empirical_in_loess,
                loess_frac_by_fit_source=loess_frac_by_fit_source,
                empirical_legend_alpha=empirical_legend_alpha,
            )
            manifest_source = (
                {"simulated": "simulated and empirical", "empirical": "empirical only"}[
                    source_group
                ]
                if plot_style == "combined" else source_group
            )
            manifest.append(
                {
                    "source_group": manifest_source,
                    "metric": value,
                    "kind": "figure",
                    "path": str(path),
                }
            )
        _remove_stale_metric_figures(
            path.parent,
            {
                _add_filename_suffix(spec["filename"], filename_suffix)
                for spec in plot_specs.values()
            },
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
    xscale: str | None = None,
    yscale: AxisScaleSpec = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = 1.0,
    normalize_metrics: bool = True,
    rho_normalization: str = "centered",
    filename_suffix: str | None = None,
    plot_style: str = "standard",
    show_loess: bool = True,
    show_empirical_only_loess: bool = True,
    include_empirical_in_loess: bool = True,
    loess_frac_by_fit_source: Mapping[str, float] | None = None,
    empirical_legend_alpha: float = 0.65,
) -> dict[str, dict[str, object]]:
    """Create figures using well-specified NPE--MCMC reference thresholds."""
    comparisons: dict[str, dict[str, object]] = {}
    for metric in metrics:
        cached = _load_cached_frames(
            metric,
            summary_specs,
            rho_normalization=rho_normalization,
        )
        for name, models in model_sets.items():
            pmp_source = "four_model"
            comparisons[f"{name}_{metric}"] = run_summary_dimension_comparison(
                models=models,
                metric=metric,
                summary_specs=summary_specs,
                pmp_source=pmp_source,
                cached_frames=cached,
                calibration_threshold_path=calibration_threshold_path,
                xscale=xscale,
                yscale=yscale,
                x_symlog_linthresh=x_symlog_linthresh,
                y_symlog_linthresh=y_symlog_linthresh,
                normalize_metrics=normalize_metrics,
                rho_normalization=rho_normalization,
                filename_suffix=filename_suffix,
                plot_style=plot_style,
                show_loess=show_loess,
                show_empirical_only_loess=show_empirical_only_loess,
                include_empirical_in_loess=include_empirical_in_loess,
                loess_frac_by_fit_source=loess_frac_by_fit_source,
                empirical_legend_alpha=empirical_legend_alpha,
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
    threshold_path: str | Path | None = None,
    xscale: str | None = None,
    yscale: AxisScaleSpec = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = 1.0,
    normalize_metrics: bool = True,
    rho_normalization: str = "centered",
    filename_suffix: str | None = None,
    plot_style: str = "standard",
    show_loess: bool = True,
    show_empirical_only_loess: bool = True,
    include_empirical_in_loess: bool = True,
    loess_frac_by_fit_source: Mapping[str, float] | None = None,
    empirical_legend_alpha: float = 0.65,
) -> dict[str, dict[str, object]]:
    """Run the comparison layout with calibrated metric-specific bounds."""
    return run_comparison_pipeline(
        metrics=metrics,
        model_sets=model_sets,
        summary_specs=summary_specs,
        output_root=output_root,
        calibration_threshold_path=threshold_path,
        xscale=xscale,
        yscale=yscale,
        x_symlog_linthresh=x_symlog_linthresh,
        y_symlog_linthresh=y_symlog_linthresh,
        normalize_metrics=normalize_metrics,
        rho_normalization=rho_normalization,
        filename_suffix=filename_suffix,
        plot_style=plot_style,
        show_loess=show_loess,
        show_empirical_only_loess=show_empirical_only_loess,
        include_empirical_in_loess=include_empirical_in_loess,
        loess_frac_by_fit_source=loess_frac_by_fit_source,
        empirical_legend_alpha=empirical_legend_alpha,
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
VISUALIZATION_SUMMARY_LABELS = PLOT_SUMMARY_LABELS
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
    "mmd": "Kernel",
}
DIAGNOSTIC_XLABELS = {
    "density": "Diagnostic: density-based",
    "l2": r"Diagnostic: $L_2$-based",
    "linf": r"Diagnostic: $L_\infty$-based",
    "mmd": "Diagnostic: Kernel-based",
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
        "bounds": (),
        "signed": True,
        "nonnegative": False,
        "independent_y": True,
    },
    "signed_pmp_error": {
        "ylabel": r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        "bounds": (),
        "signed": True,
        "nonnegative": False,
        "independent_y": True,
    },
    "absolute_logml_error": {
        "ylabel": (
            r"$\left|\log\widehat{p}(y\mid M_j)"
            r"-\log p(y\mid M_j)\right|$"
        ),
        "bounds": (),
        "signed": False,
        "nonnegative": True,
        "independent_y": True,
    },
    "absolute_pmp_error": {
        "ylabel": r"$\left|\widehat{p}(M_j\mid y)-p(M_j\mid y)\right|$",
        "bounds": (),
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
GOLD_PMP_ALL_MODEL_COLOR_CUTOFF = 1e-2
GOLD_PMP_ALL_MODEL_LOW_COLOR_FRACTION = 0.30
GOLD_PMP_ALL_MODEL_COLORBAR_TICKS = (
    0.0,
    1e-40,
    1e-20,
    1e-8,
    1e-2,
    1e-1,
    5e-1,
    1.0,
)
EMPIRICAL_GRID_ROW_STRIP_WIDTH = 0.075
SHOW_GOLD_PMP_MAX_ERROR = True
SHOW_GOLD_PMP_BOUNDARY_HIT = False


def load_visualization_data(
    diagnostic: str,
    summary_specs: SummarySpecs,
    models: tuple[str, ...] = VISUALIZATION_MODELS,
    sources: tuple[str, ...] = VISUALIZATION_SOURCES,
    posterior_mmd_column: str = "observed_mmd",
    rho_normalization: str = "centered",
) -> pd.DataFrame:
    """Load participant-level diagnostics for the visualization suite."""
    if diagnostic not in DIAGNOSTICS:
        raise ValueError(f"diagnostic must be one of {DIAGNOSTICS}")
    diagnostic_by_summary, posterior_by_summary = load_summary_dimension_data(
        metric=diagnostic,
        models=models,
        summary_specs=summary_specs,
        rho_normalization=rho_normalization,
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
        _shade_typical_set(
            ax,
            float(rho_low.min()),
            _threshold_band(bounds),
        )
        ax.axvline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        if value_spec["signed"]:
            ax.axhline(0.0, color="0.35", linewidth=0.8)
        for bound in bounds:
            ax.axhline(
                bound,
                color="#7A0276",
                linestyle=":",
                linewidth=THRESHOLD_LINEWIDTH,
            )
        for summary in VISUALIZATION_SUMMARY_LABELS:
            summary_data = panel.loc[panel["summary"].eq(summary)]
            color = SUMMARY_COLORS[summary]
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


def _gold_pmp_grid_handles(*, include_line_guides: bool = False) -> list[Line2D]:
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
    if include_line_guides:
        handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    linewidth=GOLD_PMP_LOWESS_LINEWIDTH,
                    label="LOESS regression",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#7A0276",
                    linestyle="--",
                    linewidth=1.1,
                    label=r"max $|$error$|$ for $\rho\leq1$",
                ),
            ]
        )
    return handles


def _gold_pmp_color_norm(
    models: tuple[str, ...],
    values: pd.Series,
) -> Normalize:
    """Keep four-model PMP colors legible without changing their values."""
    if tuple(models) == tuple(MODELS):
        positive = pd.to_numeric(values, errors="coerce")
        positive = positive.loc[positive.gt(0.0) & np.isfinite(positive)]
        color_floor = max(
            float(positive.min()) if not positive.empty else np.finfo(float).tiny,
            np.finfo(float).tiny,
        )
        cutoff = GOLD_PMP_ALL_MODEL_COLOR_CUTOFF
        low_fraction = GOLD_PMP_ALL_MODEL_LOW_COLOR_FRACTION
        log_floor = np.log10(color_floor)
        log_cutoff = np.log10(cutoff)

        def forward(probabilities):
            probabilities = np.asarray(probabilities, dtype=float)
            clipped = np.clip(probabilities, 0.0, 1.0)
            low_values = low_fraction * (
                (np.log10(np.maximum(clipped, color_floor)) - log_floor)
                / (log_cutoff - log_floor)
            )
            high_values = low_fraction + (1.0 - low_fraction) * (
                (clipped - cutoff) / (1.0 - cutoff)
            )
            return np.where(
                clipped <= 0.0,
                0.0,
                np.where(clipped <= cutoff, low_values, high_values),
            )

        def inverse(colors):
            colors = np.asarray(colors, dtype=float)
            clipped = np.clip(colors, 0.0, 1.0)
            low_values = 10.0 ** (
                log_floor
                + (clipped / low_fraction) * (log_cutoff - log_floor)
            )
            high_values = cutoff + (
                (clipped - low_fraction) / (1.0 - low_fraction)
            ) * (1.0 - cutoff)
            return np.where(
                clipped <= 0.0,
                0.0,
                np.where(clipped <= low_fraction, low_values, high_values),
            )

        return FuncNorm(
            (forward, inverse),
            vmin=0.0,
            vmax=1.0,
            clip=True,
        )
    return Normalize(vmin=0.0, vmax=1.0, clip=True)


def _scatter_gold_pmp_points(
    ax: plt.Axes,
    panel: pd.DataFrame,
    value_column: str,
    color_norm: Normalize,
    *,
    size_scale: float = 1.0,
    linewidth: float = 0.5,
):
    """Draw Gold-PMP-colored points using one shared normalization."""
    last_scatter = None
    for flag, style in GOLD_PMP_POINT_STYLES.items():
        points = panel.loc[panel["at_least_one_not_high_surprise"].eq(flag)]
        if points.empty:
            continue
        last_scatter = ax.scatter(
            points["rho"],
            points[value_column],
            c=points["gold_pmp"],
            cmap=GOLD_PMP_HIGH_CONTRAST_CMAP,
            norm=color_norm,
            s=style["size"] * size_scale,
            marker=style["marker"],
            alpha=0.65,
            edgecolors="black",
            linewidths=linewidth,
            zorder=3,
        )
    return last_scatter


def _add_gold_pmp_max_error(
    ax: plt.Axes,
    panel: pd.DataFrame,
    value_column: str = "signed_pmp_error",
    *,
    fontsize: float = 11,
    linewidth: float = 1.1,
) -> bool:
    """Mark the largest absolute signed PMP error among rows with rho <= 1."""
    within_typical = panel.loc[panel["rho"].le(1.0)]
    if within_typical.empty:
        return False
    extreme = within_typical.loc[within_typical[value_column].abs().idxmax()]
    extreme_y = float(extreme[value_column])
    ax.axhline(
        extreme_y,
        color="#7A0276",
        linestyle="--",
        linewidth=linewidth,
        zorder=4,
    )
    ax.text(
        0.02,
        extreme_y,
        f"{extreme_y:.2g}",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="top",
        color="#7A0276",
        fontsize=fontsize,
        clip_on=False,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.65,
            "pad": 1.0,
        },
    )
    return True


def plot_gold_pmp_colored_pmp_error_grid(
    data: pd.DataFrame,
    diagnostic: str,
    output_stem: str | Path,
    *,
    models: tuple[str, ...] = VISUALIZATION_MODELS,
    threshold_columns: tuple[str, str] | None = None,
    value_column: str = "signed_pmp_error",
    ylabel: str = r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
    xscale: str = RHO_XSCALE,
    yscale: str = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = PMP_SYMLOG_LINTHRESH,
) -> dict[str, Path]:
    """Plot the empirical PMP-error grid with one pooled LOWESS per panel."""
    if diagnostic not in DIAGNOSTIC_LABELS:
        raise ValueError(f"diagnostic must be one of {tuple(DIAGNOSTIC_LABELS)}")
    models = tuple(models)
    if not models or len(models) != len(set(models)):
        raise ValueError("models must contain one or more unique model names")
    unknown_models = tuple(model for model in models if model not in MODELS)
    if unknown_models:
        raise ValueError(f"Unknown models: {unknown_models}")
    required = {
        "summary",
        "model",
        "rho",
        "rho_low",
        "gold_pmp",
        value_column,
        "at_least_one_not_high_surprise",
    }
    if threshold_columns is not None:
        required.update(threshold_columns)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Gold-PMP grid is missing columns: {missing}")
    plot_data = data.loc[
        data["summary"].isin(VISUALIZATION_SUMMARY_LABELS)
        & data["model"].isin(models)
    ].copy()
    if plot_data.empty:
        raise ValueError("Gold-PMP grid has no data to plot")

    fig, axes = plt.subplots(
        len(VISUALIZATION_SUMMARY_LABELS),
        len(models),
        figsize=(9, 10) if len(models) == 2 else (3.5 * len(models) + 1.5, 10),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    all_model_layout = len(models) == len(MODELS)
    color_norm = _gold_pmp_color_norm(models, plot_data["gold_pmp"])
    plot_right = 0.86 if all_model_layout else 0.80
    row_strip_width = 0.04 if all_model_layout else EMPIRICAL_GRID_ROW_STRIP_WIDTH
    colorbar_left = 0.94 if all_model_layout else 0.90
    fig.subplots_adjust(
        left=0.14,
        right=plot_right,
        bottom=0.16,
        top=0.88,
        wspace=0.28,
        hspace=0.15,
    )
    x_values = pd.concat(
        [plot_data["rho"], plot_data["rho_low"], pd.Series([1.0])],
        ignore_index=True,
    )
    x_limits = (
        _symlog_padded_limits(x_values, linthresh=x_symlog_linthresh)
        if xscale == "symlog"
        else _visualization_limits(x_values, include=(1.0,))
    )
    if threshold_columns is not None:
        threshold_values = np.concatenate(
            [
                pd.to_numeric(plot_data[column], errors="coerce").dropna().unique()
                for column in threshold_columns
            ]
        )
    else:
        threshold_values = np.asarray([], dtype=float)
    y_limits = _visualization_limits(
        plot_data[value_column],
        include=(0.0, *threshold_values),
    )
    last_scatter = None
    for row, summary in enumerate(VISUALIZATION_SUMMARY_LABELS):
        for column, model in enumerate(models):
            ax = axes[row, column]
            panel = plot_data.loc[
                plot_data["summary"].eq(summary) & plot_data["model"].eq(model)
            ].dropna(subset=["rho", "rho_low", "gold_pmp", value_column])
            _set_axis_scale(
                ax,
                "x",
                xscale,
                linthresh=x_symlog_linthresh,
            )
            _set_axis_scale(
                ax,
                "y",
                yscale,
                linthresh=y_symlog_linthresh,
            )
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.axhline(0.0, color="0.35", linewidth=0.8, zorder=1)
            panel_bounds: tuple[float, ...] = ()
            active_columns = threshold_columns or ()
            if active_columns and not panel.empty:
                values = []
                for column_name in active_columns:
                    unique = pd.to_numeric(
                        panel[column_name], errors="coerce"
                    ).dropna().unique()
                    if len(unique) != 1:
                        raise ValueError(
                            f"Expected one {column_name} for {summary}/{model}; "
                            f"found {len(unique)}"
                        )
                    values.append(float(unique[0]))
                panel_bounds = tuple(values)
            for threshold in panel_bounds:
                ax.axhline(
                    threshold,
                    color="0.35",
                    linestyle="--",
                    linewidth=THRESHOLD_LINEWIDTH,
                    zorder=1,
                )
            if not panel.empty:
                rho_low = float(panel["rho_low"].median())
                _shade_typical_set(
                    ax,
                    rho_low,
                    panel_bounds,
                    alpha=0.55,
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
                        panel[value_column].lt(panel_bounds[0])
                        | panel[value_column].gt(panel_bounds[-1])
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
                panel_scatter = _scatter_gold_pmp_points(
                    ax,
                    panel,
                    value_column,
                    color_norm,
                )
                if panel_scatter is not None:
                    last_scatter = panel_scatter
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
    for ax, model in zip(axes[0], models, strict=True):
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
                row_strip_width,
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
    fig.supylabel(
        ylabel,
        x=0.04,
        fontsize=18,
        ha="center",
        va="center",
        multialignment="center",
    )
    fig.legend(
        handles=_gold_pmp_grid_handles(include_line_guides=all_model_layout),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5 if all_model_layout else 3,
        frameon=False,
        fontsize=12 if all_model_layout else 14,
    )
    if last_scatter is not None:
        colorbar_ax = fig.add_axes([colorbar_left, 0.20, 0.016, 0.62])
        colorbar = fig.colorbar(last_scatter, cax=colorbar_ax)
        colorbar_label = "Gold-standard PMP"
        if all_model_layout:
            colorbar.set_ticks(GOLD_PMP_ALL_MODEL_COLORBAR_TICKS)
            colorbar.ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _: (
                        "0"
                        if value == 0.0
                        else f"{value:.0e}"
                        if value < 0.01
                        else f"{value:g}"
                    )
                )
            )
            colorbar_label += (
                "\n(log color spacing below "
                f"{GOLD_PMP_ALL_MODEL_COLOR_CUTOFF:g})"
            )
        colorbar.set_label(colorbar_label, fontsize=16)
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
    *,
    summary_specs: SummarySpecs | None = None,
    threshold_path: str | Path | None = None,
    xscale: str = RHO_XSCALE,
    yscale: AxisScaleSpec = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = PMP_SYMLOG_LINTHRESH,
    normalize_metrics: bool = True,
) -> dict[str, Path]:
    """Generate only the empirical 4x2 Gold-PMP-colored LOWESS grid."""
    if summary_specs is None:
        if loss_variant not in LOSS_VARIANTS:
            raise ValueError(f"loss_variant must be one of {tuple(LOSS_VARIANTS)}")
        _, summary_specs = LOSS_VARIANTS[loss_variant]
    empirical_data = load_visualization_data(
        diagnostic,
        summary_specs,
        sources=("empirical",),
    )
    empirical_data["logml_error"] = (
        empirical_data["signed_logml_error"] / np.log(10.0)
    )
    empirical_data["pmp_error"] = empirical_data["signed_pmp_error"]
    empirical_data = _attach_calibration_thresholds(
        empirical_data,
        summary_specs,
        threshold_path,
    )
    value_column = "pmp_error"
    threshold_columns = ("pmp_error_lower_threshold", "pmp_error_upper_threshold")
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
        value_column=value_column,
        threshold_columns=threshold_columns,
        ylabel=r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        xscale=xscale,
        yscale=_resolve_metric_scale(yscale, "pmp"),
        x_symlog_linthresh=x_symlog_linthresh,
        y_symlog_linthresh=y_symlog_linthresh,
    )


def display_gold_pmp_lowess_grid(
    paths: dict[str, Path],
    width: int | None = 800,
    models: tuple[str, ...] = VISUALIZATION_MODELS,
) -> None:
    """Display the generated empirical Gold-PMP LOWESS grid in a notebook."""
    from IPython.display import Image, Markdown, display

    model_label = (
        " — all four assumed models"
        if tuple(models) == tuple(MODELS)
        else ""
    )
    display(
        Markdown(
            "## Empirical PMP error vs. diagnostic $\\rho$"
            f"{model_label}"
        )
    )
    display(
        Image(filename=str(paths["png"]), width=width)
        if width
        else Image(filename=str(paths["png"]))
    )


def load_calibrated_visualization_data(
    diagnostic: str,
    summary_specs: SummarySpecs,
    threshold_path: str | Path | None = None,
    sources: tuple[str, ...] = ("empirical",),
    models: tuple[str, ...] = VISUALIZATION_MODELS,
    rho_normalization: str = "centered",
) -> pd.DataFrame:
    """Load observed metrics and attach matching well-specified thresholds."""
    data = load_visualization_data(
        diagnostic,
        summary_specs,
        models=MODELS,
        sources=sources,
        posterior_mmd_column="observed_mmd",
        rho_normalization=rho_normalization,
    )
    data = data.loc[data["model"].isin(models)].copy()
    if tuple(models) == tuple(MODELS):
        group_columns = ["summary", "dataset", "id"]
        model_counts = data.groupby(group_columns, observed=True)["model"].nunique()
        if not model_counts.eq(len(MODELS)).all():
            raise ValueError("Four-model PMP data is missing one or more assumed models")
        gold_sums = data.groupby(group_columns, observed=True)["gold_pmp"].sum()
        estimated_sums = (
            data.assign(
                estimated_pmp=data["gold_pmp"] + data["signed_pmp_error"]
            )
            .groupby(group_columns, observed=True)["estimated_pmp"]
            .sum()
        )
        if not np.allclose(gold_sums, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("Four-model gold PMP values do not sum to one")
        if not np.allclose(estimated_sums, 1.0, rtol=0.0, atol=1e-10):
            raise ValueError("Four-model estimated PMP values do not sum to one")
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
    threshold_path: str | Path | None = None,
    summary_specs: SummarySpecs = NO_MMD_SUMMARY_SPECS,
    output_variant: str = "without_mmd",
    models: tuple[str, ...] = VISUALIZATION_MODELS,
    xscale: str = RHO_XSCALE,
    yscale: AxisScaleSpec = "linear",
    x_symlog_linthresh: float = RHO_SYMLOG_LINTHRESH,
    y_symlog_linthresh: float = PMP_SYMLOG_LINTHRESH,
    normalize_metrics: bool = True,
    rho_normalization: str = "centered",
) -> dict[str, Path]:
    """Generate the original empirical PMP grid with calibrated PMP cutoffs."""
    data = load_calibrated_visualization_data(
        diagnostic,
        summary_specs,
        threshold_path=threshold_path,
        models=models,
        rho_normalization=rho_normalization,
    )
    root = (
        Path(output_root)
        if output_root is not None
        else RESULT_DIR / "plots" / "diagnostic_overlays_calibrated_noMMD"
    )
    value_column = "pmp_error"
    threshold_columns = ("pmp_error_lower_threshold", "pmp_error_upper_threshold")
    model_suffix = "_all_models" if tuple(models) == tuple(MODELS) else ""
    return plot_gold_pmp_colored_pmp_error_grid(
        data,
        diagnostic,
        root
        / output_variant
        / diagnostic
        / f"12_{GOLD_PMP_GRID_KEY}{model_suffix}",
        models=models,
        threshold_columns=threshold_columns,
        value_column=value_column,
        ylabel=r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
        xscale=xscale,
        yscale=_resolve_metric_scale(yscale, "pmp"),
        x_symlog_linthresh=x_symlog_linthresh,
        y_symlog_linthresh=y_symlog_linthresh,
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


def _validate_diagnostic_notebook_thresholds(
    thresholds: pd.DataFrame,
    *,
    posterior_mmd_quantile: float,
    signed_error_coverage: float,
    expected_calibration_datasets: int,
) -> None:
    """Check the calibration interval and sample-count contract used by notebooks."""
    required = {"metric", "lower_quantile", "upper_quantile", "n_values"}
    missing = required.difference(thresholds.columns)
    if missing:
        raise ValueError(f"Calibration thresholds are missing columns: {sorted(missing)}")
    signed_tail = (1.0 - signed_error_coverage) / 2.0
    for metric, low, high, count in (
        ("posterior_mmd", 0.0, posterior_mmd_quantile, expected_calibration_datasets),
        ("signed_logml_error", signed_tail, 1.0 - signed_tail, expected_calibration_datasets),
        ("signed_pmp_error", signed_tail, 1.0 - signed_tail, len(MODELS) * expected_calibration_datasets),
    ):
        rows = thresholds.loc[thresholds["metric"].eq(metric)]
        if rows.empty or not (
            np.isclose(rows["lower_quantile"], low).all()
            and np.isclose(rows["upper_quantile"], high).all()
            and rows["n_values"].eq(count).all()
        ):
            raise ValueError(
                f"{metric} thresholds must use interval [{low:g}, {high:g}] "
                f"and {count} calibration values per group"
            )


def run_diagnostic_notebook(
    metric: str,
    variant: str = "noMMD",
    *,
    xscale: str = "symlog",
    yscale: AxisScaleSpec | None = None,
    normalize_metrics: bool = False,
    y_symlog_linthresh: float | None = None,
    show_loess: bool = True,
    show_empirical_only_loess: bool = True,
    include_empirical_in_loess: bool = True,
    loess_frac_by_fit_source: Mapping[str, float] | None = None,
    empirical_legend_alpha: float = 0.45,
    posterior_mmd_quantile: float = 0.95,
    signed_error_coverage: float = 0.90,
    expected_calibration_datasets: int = 100,
    output_root: str | Path | None = None,
    overlay_output_root: str | Path | None = None,
    threshold_path: str | Path | None = None,
    refresh_thresholds: bool = True,
) -> dict[str, object]:
    """Run one of the twelve calibrated diagnostic notebook workflows.

    ``metric`` selects density/l2/linf/mmd; ``variant`` selects the checkpoint,
    100-dataset thresholds, and output directories together. With-MMD figures
    show simulated and empirical data separately. No-MMD figures overlay both
    sources, also show empirical-only figures, and include the all-model PMP
    grid. All options are passed explicitly; no plotting globals are replaced.

    Set ``refresh_thresholds=False`` to read the existing table without rewriting
    calibration results. Figure output roots can be redirected independently.
    """
    from ..calibration.thresholds import CALIBRATION_VARIANTS, calculate_and_save_thresholds

    if metric not in DIAGNOSTICS:
        raise ValueError(f"Unknown diagnostic: {metric!r}; choose one of {DIAGNOSTICS}")
    if variant not in CALIBRATION_VARIANTS:
        raise ValueError(f"Unknown calibration variant: {variant!r}")
    if expected_calibration_datasets < 1:
        raise ValueError("expected_calibration_datasets must be positive")
    with_mmd = variant == "withMMD"
    summary_specs = _make_summary_specs(
        summary_base_distribution="normal" if with_mmd else None,
        run_suffix=None if with_mmd else variant,
    )
    output_variant = {
        "withMMD": "with_mmd",
        "noMMD": "without_mmd",
        "noMMD_rerun1": "without_mmd_rerun1",
    }[variant]
    threshold_path = (
        Path(threshold_path) if threshold_path is not None
        else BASE_DIR / f"calibration_outputs_100_{variant}" / "thresholds.csv"
    )
    output_root = (
        Path(output_root) if output_root is not None
        else RESULT_DIR / "plots" / f"summary_diagnostics_{variant}_calibrated_thresholds"
    )
    overlay_output_root = (
        Path(overlay_output_root) if overlay_output_root is not None
        else RESULT_DIR / "plots" / f"diagnostic_overlays_calibrated_{variant}"
    )
    if yscale is None:
        yscale = {"posterior_mmd": "linear", "logml": "symlog", "pmp": "linear"}
    if y_symlog_linthresh is None:
        y_symlog_linthresh = 1.0 if with_mmd else 0.1
    if refresh_thresholds:
        thresholds, _ = calculate_and_save_thresholds(
            input_path=threshold_path.parent / "per_dataset_metrics.csv",
            thresholds_path=threshold_path,
            results_path=threshold_path.parent / "per_dataset_results.csv",
            quantile=posterior_mmd_quantile,
            signed_error_coverage=signed_error_coverage,
        )
    else:
        thresholds = pd.read_csv(threshold_path)
    _validate_diagnostic_notebook_thresholds(
        thresholds,
        posterior_mmd_quantile=posterior_mmd_quantile,
        signed_error_coverage=signed_error_coverage,
        expected_calibration_datasets=expected_calibration_datasets,
    )
    comparisons = run_calibrated_comparison_pipeline(
        metrics=(metric,),
        model_sets=MODEL_SETS,
        summary_specs=summary_specs,
        output_root=output_root,
        threshold_path=threshold_path,
        xscale=xscale,
        yscale=yscale,
        y_symlog_linthresh=y_symlog_linthresh,
        normalize_metrics=normalize_metrics,
        plot_style="standard" if with_mmd else "combined",
        show_loess=show_loess,
        show_empirical_only_loess=show_empirical_only_loess,
        include_empirical_in_loess=include_empirical_in_loess,
        loess_frac_by_fit_source=loess_frac_by_fit_source,
        empirical_legend_alpha=empirical_legend_alpha,
    )
    grid_options = dict(
        output_root=overlay_output_root,
        threshold_path=threshold_path,
        summary_specs=summary_specs,
        output_variant=output_variant,
        xscale=xscale,
        yscale=yscale,
        normalize_metrics=normalize_metrics,
    )
    gold_pmp_paths = generate_calibrated_gold_pmp_lowess_grid(metric, **grid_options)
    all_model_paths = (
        generate_calibrated_gold_pmp_lowess_grid(metric, models=MODELS, **grid_options)
        if not with_mmd else None
    )
    return {
        "metric": metric,
        "variant": variant,
        "thresholds": thresholds,
        "threshold_path": threshold_path,
        "comparisons": comparisons,
        "gold_pmp_paths": gold_pmp_paths,
        "all_model_gold_pmp_paths": all_model_paths,
    }


def display_diagnostic_notebook(result: dict[str, object]) -> None:
    """Display shared workflow figures in the original notebook order."""
    print(f"Calibration thresholds: {result['threshold_path']}")
    for name, comparison in result["comparisons"].items():
        print(name, comparison["output_root"])
        display_summary_dimension_comparison(comparison)
    display_gold_pmp_lowess_grid(result["gold_pmp_paths"])
    if result["all_model_gold_pmp_paths"] is not None:
        display_gold_pmp_lowess_grid(
            result["all_model_gold_pmp_paths"], width=1200, models=MODELS
        )
