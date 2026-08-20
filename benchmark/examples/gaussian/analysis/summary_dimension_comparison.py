"""Compare Gaussian approximation errors across summary dimensions.

This module is plotting-only. It reads cached OOD diagnostic tables and the
Gaussian calibration thresholds; it never loads a network or reruns inference.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.scale import SymmetricalLogTransform
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..config import ASSUMED_MODELS, CALIBRATION_ROOT, RESULT_DIR


DIAGNOSTICS = ("l2", "mmd", "density", "linf")
DIAGNOSTIC_LABELS = {
    "l2": r"$L_2$-based",
    "mmd": "MMD-based",
    "density": "density-based",
    "linf": r"$L_\infty$-based",
}
ERROR_METRICS = ("posterior_mmd", "log10_logml_error", "pmp_error")
ERROR_LABELS = {
    "posterior_mmd": "Posterior MMD",
    "log10_logml_error": (
        r"$\log_{10}\widehat{p}(y\mid M_j)-\log_{10}p(y\mid M_j)$"
    ),
    "pmp_error": r"$\widehat{p}(M_j\mid y)-p(M_j\mid y)$",
}
ERROR_CACHE_NAMES = {
    "posterior_mmd": "posterior_distance_frame.csv",
    "log10_logml_error": "logml_distance_frame.csv",
    "pmp_error": "pmp_ambiguity_frame.csv",
}
CALIBRATION_METRICS = {
    "posterior_mmd": "posterior_mmd",
    "log10_logml_error": "signed_logml_error",
    "pmp_error": "signed_pmp_error",
}
SUMMARY_SPECS = (
    ("S=D", "20d_10n"),
    ("S=2D", "40d_10n"),
    ("S=4D", "80d_10n"),
)
SUMMARY_COLORS = {
    "S=D": "#0072B2",
    "S=2D": "#E69F00",
    "S=4D": "#CC79A7",
}
COMPARISON_MODELS = tuple(ASSUMED_MODELS)
TYPICAL_SET_FILL = "#DCEEDC"
DEFAULT_OUTPUT_DIR = RESULT_DIR / "plots" / "summary_dimension_comparison"
DEFAULT_LOWESS_FRAC = 0.70
AXIS_SCALES = ("linear", "symlog")
DEFAULT_YSCALES = {
    "posterior_mmd": "linear",
    "log10_logml_error": "symlog",
    "pmp_error": "linear",
}
DEFAULT_Y_SYMLOG_LINTHRESH = {
    "posterior_mmd": 0.1,
    "log10_logml_error": 1.0,
    "pmp_error": 0.01,
}
DEFAULT_X_SYMLOG_LINTHRESH = 1.0
RHO_ALIGNMENT_FRACTION = 0.20
RHO_LEFT_PADDING_FRACTION = 0.10
FULL_RANGE_QUANTILES = (0.0, 1.0)
POSTERIOR_MMD_Y_MIN = -0.5

AxisScaleSpec = str | Mapping[str, str]
NumericSpec = float | Mapping[str, float]


def _validate_selection(
    diagnostics: tuple[str, ...],
    error_metrics: tuple[str, ...],
    models: tuple[str, ...],
) -> None:
    unknown_diagnostics = sorted(set(diagnostics).difference(DIAGNOSTICS))
    unknown_errors = sorted(set(error_metrics).difference(ERROR_METRICS))
    unknown_models = sorted(set(models).difference(ASSUMED_MODELS))
    if unknown_diagnostics:
        raise ValueError(f"Unknown Gaussian diagnostics: {unknown_diagnostics}")
    if unknown_errors:
        raise ValueError(f"Unknown Gaussian error metrics: {unknown_errors}")
    if unknown_models:
        raise ValueError(f"Unknown assumed models: {unknown_models}")
    if not diagnostics or not error_metrics or not models:
        raise ValueError("Diagnostics, error metrics, and models cannot be empty")


def _resolve_spec(spec: str | float | Mapping, key: str):
    if isinstance(spec, Mapping):
        if key not in spec:
            raise ValueError(f"No plotting option supplied for {key!r}")
        return spec[key]
    return spec


def _cache_path(
    diagnostic: str,
    network_tag: str,
    error_metric: str,
) -> tuple[Path, bool]:
    """Return the best available cache and whether it uses legacy rho."""
    result_root = RESULT_DIR / f"ood_{network_tag}"
    modern = (
        result_root
        / "diagnostics"
        / diagnostic
        / ERROR_CACHE_NAMES[error_metric]
    )
    if modern.exists():
        return modern, False
    stem = ERROR_CACHE_NAMES[error_metric].removesuffix(".csv")
    if diagnostic == "l2":
        return result_root / f"{stem}_{network_tag}.csv", True
    if diagnostic == "linf":
        legacy_root = RESULT_DIR / f"ood_{network_tag}_inf"
        return legacy_root / f"{stem}_{network_tag}_linf.csv", True
    return modern, False


def _calibration_thresholds() -> pd.DataFrame:
    path = CALIBRATION_ROOT / "20d_10n" / "thresholds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Gaussian calibration thresholds: {path}")
    thresholds = pd.read_csv(path)
    calibration_to_error = {
        calibration: error for error, calibration in CALIBRATION_METRICS.items()
    }
    thresholds = thresholds.loc[
        thresholds["metric"].isin(calibration_to_error),
        [
            "network_tag",
            "generating_model",
            "metric",
            "lower_threshold",
            "threshold",
        ],
    ].rename(
        columns={
            "generating_model": "assumed_model",
            "lower_threshold": "error_lower_threshold",
            "threshold": "error_upper_threshold",
        }
    )
    thresholds["error_metric"] = thresholds["metric"].map(calibration_to_error)
    thresholds = thresholds.drop(columns="metric")
    if thresholds.duplicated(
        ["network_tag", "assumed_model", "error_metric"]
    ).any():
        raise ValueError("Gaussian calibration thresholds are not unique")
    return thresholds


def _long_error_frame(
    frame: pd.DataFrame,
    error_metric: str,
) -> pd.DataFrame:
    common = ["source_model", "id"]
    if error_metric == "posterior_mmd":
        columns = common + [
            "assumed_model",
            "d_M",
            "dm_median",
            "dm_low",
            "dm_high",
            "posterior_mmd",
        ]
        available = [column for column in columns if column in frame]
        output = frame[available].copy()
        output["error_value"] = pd.to_numeric(
            output.pop("posterior_mmd"), errors="coerce"
        )
        return output

    if error_metric == "log10_logml_error":
        columns = common + [
            "assumed_model",
            "d_M",
            "dm_median",
            "dm_low",
            "dm_high",
            "signed_logml_error",
        ]
        available = [column for column in columns if column in frame]
        output = frame[available].copy()
        output["error_value"] = (
            pd.to_numeric(output.pop("signed_logml_error"), errors="coerce")
            / np.log(10.0)
        )
        return output

    rows = []
    for model in ASSUMED_MODELS:
        required = {
            f"d_{model}",
            f"dm_low_{model}",
            f"dm_high_{model}",
            f"signed_pmp_error_npe_{model}",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"PMP cache is missing columns: {missing}")
        rows.append(
            pd.DataFrame(
                {
                    "source_model": frame["source_model"],
                    "id": frame["id"],
                    "assumed_model": model,
                    "d_M": frame[f"d_{model}"],
                    "dm_median": (
                        frame[f"dm_median_{model}"]
                        if f"dm_median_{model}" in frame
                        else 0.0
                    ),
                    "dm_low": frame[f"dm_low_{model}"],
                    "dm_high": frame[f"dm_high_{model}"],
                    "error_value": frame[f"signed_pmp_error_npe_{model}"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _normalization_scale(data: pd.DataFrame) -> pd.Series:
    nonnegative = data["error_metric"].eq("posterior_mmd")
    scale = np.maximum(
        data["error_lower_threshold"].abs(),
        data["error_upper_threshold"].abs(),
    )
    scale = scale.where(~nonnegative, data["error_upper_threshold"])
    if (~np.isfinite(scale)).any():
        raise ValueError("All error-normalization scales must be finite")
    # Some analytically exact PMP components have the degenerate calibrated
    # interval [0, 0]. There is no non-zero calibration scale in that case, so
    # retain raw signed errors instead of dividing by zero.
    return scale.mask(scale.le(0.0), 1.0)


def _normalize_error_values(data: pd.DataFrame) -> pd.Series:
    return data["error_value"] / _normalization_scale(data)


def load_comparison_data(
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    error_metrics: tuple[str, ...] = ERROR_METRICS,
    summary_specs: tuple[tuple[str, str], ...] = SUMMARY_SPECS,
) -> pd.DataFrame:
    """Load all requested diagnostic, summary, and approximation-error rows."""
    diagnostics = tuple(diagnostics)
    error_metrics = tuple(error_metrics)
    _validate_selection(diagnostics, error_metrics, ASSUMED_MODELS)
    frames = []
    missing_paths = []

    for diagnostic in diagnostics:
        for summary, network_tag in summary_specs:
            for error_metric in error_metrics:
                path, legacy_rho = _cache_path(
                    diagnostic, network_tag, error_metric
                )
                if not path.exists():
                    missing_paths.append(str(path))
                    continue
                frame = _long_error_frame(
                    pd.read_csv(path, keep_default_na=False), error_metric
                )
                required = {
                    "source_model",
                    "id",
                    "assumed_model",
                    "d_M",
                    "dm_low",
                    "dm_high",
                    "error_value",
                }
                absent = sorted(required.difference(frame.columns))
                if absent:
                    raise ValueError(f"{path} is missing columns: {absent}")

                # Legacy L2/Linf notebooks used rho=d/d_high. Force the same
                # zero center for all three legacy summary sizes.
                median = (
                    np.zeros(len(frame), dtype=float)
                    if legacy_rho
                    else pd.to_numeric(frame["dm_median"], errors="coerce")
                )
                high = pd.to_numeric(frame["dm_high"], errors="coerce")
                denominator = high - median
                if denominator.le(0.0).any() or (~np.isfinite(denominator)).any():
                    raise ValueError(f"Invalid diagnostic reference interval in {path}")
                frame = frame.assign(
                    diagnostic=diagnostic,
                    summary=summary,
                    network_tag=network_tag,
                    error_metric=error_metric,
                    rho=(pd.to_numeric(frame["d_M"], errors="coerce") - median)
                    / denominator,
                    rho_low=(pd.to_numeric(frame["dm_low"], errors="coerce") - median)
                    / denominator,
                )
                frame["well_specified"] = frame["source_model"].eq(
                    frame["assumed_model"]
                )
                frames.append(frame)

    if missing_paths:
        raise FileNotFoundError(
            "Missing Gaussian diagnostic caches:\n" + "\n".join(missing_paths)
        )
    data = pd.concat(frames, ignore_index=True)
    data = data.merge(
        _calibration_thresholds(),
        on=["network_tag", "assumed_model", "error_metric"],
        how="left",
        validate="many_to_one",
    )
    threshold_columns = ["error_lower_threshold", "error_upper_threshold"]
    if data[threshold_columns].isna().any().any():
        missing = data.loc[
            data[threshold_columns].isna().any(axis=1),
            ["network_tag", "assumed_model", "error_metric"],
        ].drop_duplicates()
        raise ValueError("Missing error thresholds for:\n" + missing.to_string(index=False))

    logml = data["error_metric"].eq("log10_logml_error")
    data.loc[logml, threshold_columns] = (
        data.loc[logml, threshold_columns] / np.log(10.0)
    )
    data["normalized_error_value"] = _normalize_error_values(data)
    scale = _normalization_scale(data)
    data["degenerate_error_threshold"] = (
        data["error_lower_threshold"].eq(0.0)
        & data["error_upper_threshold"].eq(0.0)
    )
    data["normalized_error_lower_threshold"] = (
        data["error_lower_threshold"] / scale
    )
    data["normalized_error_upper_threshold"] = (
        data["error_upper_threshold"] / scale
    )
    data["summary"] = pd.Categorical(
        data["summary"],
        categories=[label for label, _ in summary_specs],
        ordered=True,
    )
    data["diagnostic"] = pd.Categorical(
        data["diagnostic"], categories=list(diagnostics), ordered=True
    )
    data["error_metric"] = pd.Categorical(
        data["error_metric"], categories=list(error_metrics), ordered=True
    )
    return data.sort_values(
        [
            "error_metric",
            "diagnostic",
            "assumed_model",
            "summary",
            "source_model",
            "id",
        ]
    ).reset_index(drop=True)


def _lowess_curve(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    frac: float = DEFAULT_LOWESS_FRAC,
    n_grid: int = 180,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a dependency-free local-linear LOWESS curve."""
    if not 0.0 < frac <= 1.0:
        raise ValueError("LOWESS frac must lie in (0, 1]")
    x = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    order = np.argsort(x)
    x, y = x[order], y[order]
    if len(x) < 3:
        return x, y

    neighbors = min(len(x), max(3, int(np.ceil(frac * len(x)))))
    grid = np.linspace(float(x.min()), float(x.max()), n_grid)
    fitted = np.empty_like(grid)
    epsilon = np.finfo(float).eps
    for index, center in enumerate(grid):
        distance = np.abs(x - center)
        bandwidth = np.partition(distance, neighbors - 1)[neighbors - 1]
        if bandwidth <= epsilon:
            fitted[index] = np.median(y[distance <= epsilon])
            continue
        weights = np.clip(1.0 - (distance / bandwidth) ** 3, 0.0, None) ** 3
        design = np.column_stack([np.ones_like(x), x - center])
        root_weight = np.sqrt(weights)
        fitted[index] = np.linalg.lstsq(
            design * root_weight[:, None], y * root_weight, rcond=None
        )[0][0]
    return grid, np.clip(fitted, float(y.min()), float(y.max()))


def _padded_limits(values: pd.Series, include: float = 1.0) -> tuple[float, float]:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    array = np.append(array, include)
    lower, upper = float(array.min()), float(array.max())
    span = max(upper - lower, np.finfo(float).eps)
    return lower - 0.04 * span, upper + 0.04 * span


def _rho_aligned_limits(
    values: pd.Series,
    *,
    anchor: float = 1.0,
    anchor_fraction: float = RHO_ALIGNMENT_FRACTION,
    left_padding_fraction: float = RHO_LEFT_PADDING_FRACTION,
    quantiles: tuple[float, float] = FULL_RANGE_QUANTILES,
) -> tuple[float, float]:
    """Return full-range linear limits with extra room at the left edge."""
    if not 0.0 < anchor_fraction < 1.0:
        raise ValueError("anchor_fraction must lie in (0, 1)")
    if left_padding_fraction < 0.0:
        raise ValueError("left_padding_fraction must be non-negative")
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("Cannot determine rho limits from empty data")
    lower, upper = np.quantile(array, quantiles)
    left_span = max(anchor - float(lower), np.finfo(float).eps) * 1.06
    right_span = max(float(upper) - anchor, np.finfo(float).eps) * 1.06
    total_span = max(
        left_span / anchor_fraction,
        right_span / (1.0 - anchor_fraction),
    )
    lower = anchor - anchor_fraction * total_span
    upper = anchor + (1.0 - anchor_fraction) * total_span
    return lower - left_padding_fraction * (upper - lower), upper


def _robust_limits(
    values: pd.Series,
    *extras: pd.Series | np.ndarray,
    quantiles: tuple[float, float] = FULL_RANGE_QUANTILES,
    include_zero: bool = False,
    lower_bound: float | None = None,
) -> tuple[float, float]:
    """Choose padded limits that include every finite observation."""
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("Cannot determine axis limits from empty data")
    selected = list(np.quantile(array, quantiles))
    for extra in extras:
        extra_array = pd.to_numeric(
            pd.Series(np.asarray(extra).ravel()), errors="coerce"
        ).to_numpy(dtype=float)
        selected.extend(extra_array[np.isfinite(extra_array)])
    if include_zero:
        selected.append(0.0)
    lower, upper = float(np.min(selected)), float(np.max(selected))
    span = max(upper - lower, 0.08 * max(abs(lower), abs(upper), 1.0))
    lower -= 0.055 * span
    upper += 0.055 * span
    if lower_bound is not None:
        lower = max(lower_bound, lower)
    return lower, upper


def _robust_symlog_limits(
    values: pd.Series,
    *extras: pd.Series | np.ndarray,
    linthresh: float,
    quantiles: tuple[float, float] = FULL_RANGE_QUANTILES,
    include_zero: bool = False,
) -> tuple[float, float]:
    """Choose full-range asymmetric symlog limits in transformed space."""
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("Cannot determine axis limits from empty data")
    selected = list(np.quantile(array, quantiles))
    for extra in extras:
        extra_array = pd.to_numeric(
            pd.Series(np.asarray(extra).ravel()), errors="coerce"
        ).to_numpy(dtype=float)
        selected.extend(extra_array[np.isfinite(extra_array)])
    if include_zero:
        selected.append(0.0)
    transform = SymmetricalLogTransform(
        base=10.0, linthresh=linthresh, linscale=1.0
    )
    transformed = transform.transform_non_affine(np.asarray(selected))
    lower_t, upper_t = float(transformed.min()), float(transformed.max())
    span = max(upper_t - lower_t, 1.0)
    limits = transform.inverted().transform_non_affine(
        np.asarray([lower_t - 0.045 * span, upper_t + 0.045 * span])
    )
    return float(limits[0]), float(limits[1])


def _compact_tick_label(value: float, _: int | None = None) -> str:
    if np.isclose(value, 0.0, atol=1e-12):
        return "0"
    absolute = abs(value)
    if absolute >= 10_000 or absolute < 0.001:
        return f"{value:.0e}".replace("e+", "e")
    if absolute >= 100:
        return f"{value:.0f}"
    if absolute >= 10:
        return f"{value:.1f}".rstrip("0").rstrip(".")
    if absolute >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _symlog_tick_label(value: float, _: int | None = None) -> str:
    """Format symmetric-log ticks as signed powers of ten."""
    if np.isclose(value, 0.0, atol=1e-12):
        return r"$0$"
    absolute = abs(value)
    exponent = int(np.floor(np.log10(absolute)))
    coefficient = absolute / (10.0**exponent)
    sign = "-" if value < 0.0 else ""
    if np.isclose(coefficient, 1.0, rtol=1e-10, atol=1e-12):
        return rf"${sign}10^{{{exponent}}}$"
    return rf"${sign}{coefficient:g}\times10^{{{exponent}}}$"


def _set_y_scale(ax: plt.Axes, scale: str, linthresh: float) -> None:
    if scale not in AXIS_SCALES:
        raise ValueError(f"yscale must be one of {AXIS_SCALES}; got {scale!r}")
    if scale == "symlog":
        if linthresh <= 0.0:
            raise ValueError("y_symlog_linthresh must be positive")
        ax.set_yscale(
            "symlog", base=10.0, linthresh=linthresh, linscale=1.0
        )
        ax.yaxis.get_major_locator().set_params(numticks=6)
        ax.yaxis.set_major_formatter(FuncFormatter(_symlog_tick_label))
    else:
        ax.set_yscale("linear")
        ax.yaxis.set_major_locator(
            MaxNLocator(nbins=5, min_n_ticks=4, steps=[1, 2, 2.5, 5, 10])
        )
        ax.yaxis.set_major_formatter(FuncFormatter(_compact_tick_label))


def _set_x_symlog_scale(ax: plt.Axes, linthresh: float) -> None:
    """Use a symmetric-log x-axis with power-of-ten tick labels."""
    if linthresh <= 0.0:
        raise ValueError("x_symlog_linthresh must be positive")
    ax.set_xscale(
        "symlog", base=10.0, linthresh=linthresh, linscale=1.0
    )
    ax.xaxis.set_major_formatter(FuncFormatter(_symlog_tick_label))


def _display_thresholds(
    panel: pd.DataFrame, normalized: bool
) -> pd.DataFrame:
    columns = (
        [
            "normalized_error_lower_threshold",
            "normalized_error_upper_threshold",
        ]
        if normalized
        else ["error_lower_threshold", "error_upper_threshold"]
    )
    output = panel.groupby("summary", observed=True)[columns].median()
    output.columns = ["low", "high"]
    return output


def _draw_panel(
    ax: plt.Axes,
    panel: pd.DataFrame,
    *,
    error_metric: str,
    normalized: bool,
    lowess_frac: float,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float] | None,
    x_symlog_linthresh: float,
    yscale: str,
    y_symlog_linthresh: float,
) -> None:
    y_column = "normalized_error_value" if normalized else "error_value"
    thresholds = _display_thresholds(panel, normalized)

    # Vertical green band: the diagnostic typical set. Different summary
    # dimensions can have slightly different lower bounds, so show their union.
    rho_low = panel.groupby("summary", observed=True)["rho_low"].median().min()
    ax.axvspan(rho_low, 1.0, color=TYPICAL_SET_FILL, alpha=0.70, zorder=0)
    ax.axvline(1.0, color="0.25", linestyle="--", linewidth=1.0, zorder=1)

    # Horizontal green band: the intersection of calibrated error acceptance
    # intervals across the three summary dimensions in this model panel.
    band_low = float(thresholds["low"].max())
    band_high = float(thresholds["high"].min())
    if band_low < band_high:
        ax.axhspan(
            band_low,
            band_high,
            color=TYPICAL_SET_FILL,
            alpha=0.48,
            zorder=0,
        )

    for summary, color in SUMMARY_COLORS.items():
        overlay = panel.loc[panel["summary"].eq(summary)].dropna(
            subset=["rho", y_column]
        )
        if overlay.empty:
            continue
        for well_specified, points in overlay.groupby("well_specified", sort=False):
            ax.scatter(
                points["rho"],
                points[y_column],
                color=color,
                marker="s" if well_specified else "o",
                s=72 if well_specified else 24,
                alpha=0.68 if well_specified else 0.34,
                edgecolors="black" if well_specified else "none",
                linewidths=0.9 if well_specified else 0.0,
                zorder=3 if well_specified else 2,
            )
        curve_x, curve_y = _lowess_curve(
            overlay["rho"], overlay[y_column], frac=lowess_frac
        )
        ax.plot(curve_x, curve_y, color=color, linewidth=2.5, zorder=3)

        summary_thresholds = thresholds.loc[summary]
        plotted = (
            (float(summary_thresholds["high"]),)
            if error_metric == "posterior_mmd"
            else (
                float(summary_thresholds["low"]),
                float(summary_thresholds["high"]),
            )
        )
        for threshold in plotted:
            ax.axhline(
                threshold,
                color=color,
                linestyle=":",
                linewidth=1.7,
                alpha=0.9,
                zorder=1,
            )

    if error_metric != "posterior_mmd":
        ax.axhline(0.0, color="0.45", linewidth=0.75, zorder=1)
    _set_x_symlog_scale(ax, x_symlog_linthresh)
    _set_y_scale(ax, yscale, y_symlog_linthresh)
    ax.set_xlim(x_limits)
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.grid(color="#D7DCE2", linewidth=0.6, alpha=0.65)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FCFCFD")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=11)


def _ylabel(error_metric: str, normalized: bool) -> str:
    comparison = "NPE, MCMC; normalized" if normalized else "NPE, MCMC"
    return f"{ERROR_LABELS[error_metric]}\n({comparison})"


def _add_model_strip(ax: plt.Axes, model: str) -> None:
    strip = Rectangle(
        (0.0, 1.02),
        1.0,
        0.115,
        transform=ax.transAxes,
        facecolor="0.85",
        edgecolor="0.35",
        linewidth=0.8,
        clip_on=False,
        zorder=8,
    )
    ax.add_patch(strip)
    ax.text(
        0.5,
        1.077,
        rf"Assumed $M_{{{model[1:]}}}$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        zorder=9,
    )


def _legend_handles() -> list[Line2D | Patch]:
    handles: list[Line2D | Patch] = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            linewidth=2.5,
            markersize=6,
            label=summary,
        )
        for summary, color in SUMMARY_COLORS.items()
    ]
    handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="0.42",
                markeredgecolor="none",
                markersize=6,
                label="misspecified datasets",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="0.55",
                markeredgecolor="black",
                markeredgewidth=0.9,
                markersize=10,
                label="well-specified datasets",
            ),
            Patch(
                facecolor=TYPICAL_SET_FILL,
                edgecolor="none",
                alpha=0.70,
                label="typical set / accepted error",
            ),
        ]
    )
    return handles


def plot_error_diagnostic(
    data: pd.DataFrame,
    diagnostic: str,
    error_metric: str,
    path: str | Path,
    *,
    models: tuple[str, ...] = COMPARISON_MODELS,
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: str = "linear",
    y_symlog_linthresh: float = 1.0,
) -> Path:
    """Plot one diagnostic and error metric with summary overlays."""
    _validate_selection((diagnostic,), (error_metric,), tuple(models))
    plot_data = data.loc[
        data["diagnostic"].eq(diagnostic)
        & data["error_metric"].eq(error_metric)
        & data["assumed_model"].isin(models)
    ].copy()
    if plot_data.empty:
        raise ValueError(f"No rows for {diagnostic}/{error_metric}/{models}")
    x_limits = _padded_limits(plot_data["rho"])
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.25 * len(models) + 1.0, 6.1),
        sharey=True,
        squeeze=False,
    )
    for ax, model in zip(axes[0], models, strict=True):
        panel = plot_data.loc[plot_data["assumed_model"].eq(model)]
        _draw_panel(
            ax,
            panel,
            error_metric=error_metric,
            normalized=normalized,
            lowess_frac=lowess_frac,
            x_limits=x_limits,
            y_limits=None,
            x_symlog_linthresh=x_symlog_linthresh,
            yscale=yscale,
            y_symlog_linthresh=y_symlog_linthresh,
        )
        ax.set_title(f"Assumed {model.upper()}", fontsize=16)
        ax.set_xlabel(
            f"Diagnostic: {DIAGNOSTIC_LABELS[diagnostic]}", fontsize=14
        )

    fig.supylabel(
        _ylabel(error_metric, normalized),
        x=0.025,
        fontsize=15,
        ha="center",
        va="center",
        multialignment="center",
    )
    fig.suptitle("Simulated datasets", fontsize=18, y=0.985)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.12,
        right=0.99,
        bottom=0.29,
        top=0.82,
        wspace=0.22,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_error_diagnostic_grid(
    data: pd.DataFrame,
    error_metric: str,
    path: str | Path,
    *,
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    models: tuple[str, ...] = tuple(ASSUMED_MODELS),
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: str = "linear",
    y_symlog_linthresh: float = 1.0,
    rho_alignment_fraction: float = RHO_ALIGNMENT_FRACTION,
) -> Path:
    """Plot diagnostics by rows and assumed models by columns in one figure."""
    diagnostics = tuple(diagnostics)
    models = tuple(models)
    _validate_selection(diagnostics, (error_metric,), models)
    plot_data = data.loc[
        data["diagnostic"].isin(diagnostics)
        & data["error_metric"].eq(error_metric)
        & data["assumed_model"].isin(models)
    ].copy()
    if plot_data.empty:
        raise ValueError(f"No rows for {diagnostics}/{error_metric}/{models}")

    fig, axes = plt.subplots(
        len(diagnostics),
        len(models),
        figsize=(5.25 * len(models) + 1.4, 3.15 * len(diagnostics) + 2.1),
        sharex=False,
        sharey=False,
        squeeze=False,
    )
    for row_index, diagnostic in enumerate(diagnostics):
        for column_index, model in enumerate(models):
            ax = axes[row_index, column_index]
            panel = plot_data.loc[
                plot_data["diagnostic"].eq(diagnostic)
                & plot_data["assumed_model"].eq(model)
            ]
            x_limits = _rho_aligned_limits(
                panel["rho"], anchor_fraction=rho_alignment_fraction
            )
            y_column = (
                "normalized_error_value" if normalized else "error_value"
            )
            threshold_columns = (
                [
                    "normalized_error_lower_threshold",
                    "normalized_error_upper_threshold",
                ]
                if normalized
                else ["error_lower_threshold", "error_upper_threshold"]
            )
            if yscale == "symlog":
                y_limits = _robust_symlog_limits(
                    panel[y_column],
                    panel[threshold_columns].to_numpy(),
                    linthresh=y_symlog_linthresh,
                    include_zero=True,
                )
            else:
                y_limits = _robust_limits(
                    panel[y_column],
                    panel[threshold_columns].to_numpy(),
                    include_zero=error_metric != "posterior_mmd",
                )
            if error_metric == "posterior_mmd":
                y_limits = (POSTERIOR_MMD_Y_MIN, y_limits[1])
            _draw_panel(
                ax,
                panel,
                error_metric=error_metric,
                normalized=normalized,
                lowess_frac=lowess_frac,
                x_limits=x_limits,
                y_limits=y_limits,
                x_symlog_linthresh=x_symlog_linthresh,
                yscale=yscale,
                y_symlog_linthresh=y_symlog_linthresh,
            )
            ax.set_xlabel(
                f"Diagnostic: {DIAGNOSTIC_LABELS[diagnostic]}", fontsize=12.5
            )
            if row_index == 0:
                _add_model_strip(ax, model)

    fig.supylabel(
        _ylabel(error_metric, normalized),
        x=0.018,
        fontsize=16,
        ha="center",
        va="center",
        multialignment="center",
    )
    fig.suptitle("Simulated datasets", fontsize=18, y=0.958)
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=3,
        frameon=False,
        fontsize=11,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.14,
        top=0.90,
        wspace=0.24,
        hspace=0.52,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def run_summary_dimension_comparison(
    diagnostics: tuple[str, ...] = DIAGNOSTICS,
    error_metrics: tuple[str, ...] = ERROR_METRICS,
    models: tuple[str, ...] = COMPARISON_MODELS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    normalized: bool = False,
    lowess_frac: float = DEFAULT_LOWESS_FRAC,
    x_symlog_linthresh: float = DEFAULT_X_SYMLOG_LINTHRESH,
    yscale: AxisScaleSpec = DEFAULT_YSCALES,
    y_symlog_linthresh: NumericSpec = DEFAULT_Y_SYMLOG_LINTHRESH,
) -> dict[str, object]:
    """Generate one reference-style figure per error metric and diagnostic."""
    diagnostics = tuple(diagnostics)
    error_metrics = tuple(error_metrics)
    models = tuple(models)
    _validate_selection(diagnostics, error_metrics, models)
    data = load_comparison_data(
        diagnostics=diagnostics,
        error_metrics=error_metrics,
    )
    output_dir = Path(output_dir)
    suffix = "normalized" if normalized else "raw"
    paths: dict[str, Path] = {}
    for error_metric in error_metrics:
        metric_scale = str(_resolve_spec(yscale, error_metric))
        metric_linthresh = float(
            _resolve_spec(y_symlog_linthresh, error_metric)
        )
        path = output_dir / (
            f"{error_metric}_all_diagnostics_summary_dimensions_{suffix}.png"
        )
        paths[error_metric] = plot_error_diagnostic_grid(
            data,
            error_metric,
            path,
            diagnostics=diagnostics,
            models=models,
            normalized=normalized,
            lowess_frac=lowess_frac,
            x_symlog_linthresh=x_symlog_linthresh,
            yscale=metric_scale,
            y_symlog_linthresh=metric_linthresh,
        )
    return {"data": data, "paths": paths}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics", nargs="+", choices=DIAGNOSTICS, default=list(DIAGNOSTICS)
    )
    parser.add_argument(
        "--error-metrics",
        nargs="+",
        choices=ERROR_METRICS,
        default=list(ERROR_METRICS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ASSUMED_MODELS, default=list(COMPARISON_MODELS)
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--normalized", action="store_true")
    parser.add_argument("--lowess-frac", type=float, default=DEFAULT_LOWESS_FRAC)
    parser.add_argument(
        "--x-symlog-linthresh", type=float, default=DEFAULT_X_SYMLOG_LINTHRESH
    )
    parser.add_argument(
        "--posterior-yscale", choices=AXIS_SCALES, default="linear"
    )
    parser.add_argument("--logml-yscale", choices=AXIS_SCALES, default="symlog")
    parser.add_argument("--pmp-yscale", choices=AXIS_SCALES, default="linear")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_summary_dimension_comparison(
        diagnostics=tuple(args.diagnostics),
        error_metrics=tuple(args.error_metrics),
        models=tuple(args.models),
        output_dir=args.output_dir,
        normalized=args.normalized,
        lowess_frac=args.lowess_frac,
        x_symlog_linthresh=args.x_symlog_linthresh,
        yscale={
            "posterior_mmd": args.posterior_yscale,
            "log10_logml_error": args.logml_yscale,
            "pmp_error": args.pmp_yscale,
        },
    )
    for error_metric, path in result["paths"].items():
        print(f"{error_metric}: {path}")


if __name__ == "__main__":
    main()
