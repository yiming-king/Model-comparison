"""Load cached Gaussian approximation errors across summary dimensions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ASSUMED_MODELS, CALIBRATION_ROOT, RESULT_DIR


DIAGNOSTICS = ("l2", "mmd", "density", "linf")
ERROR_METRICS = ("posterior_mmd", "log10_logml_error", "pmp_error")
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


def _cache_path(
    diagnostic: str,
    network_tag: str,
    error_metric: str,
) -> Path:
    """Return the best available diagnostic cache."""
    result_root = RESULT_DIR / f"ood_{network_tag}"
    modern = result_root / "diagnostics" / diagnostic / ERROR_CACHE_NAMES[error_metric]
    if modern.exists():
        return modern
    stem = ERROR_CACHE_NAMES[error_metric].removesuffix(".csv")
    if diagnostic == "l2":
        return result_root / f"{stem}_{network_tag}.csv"
    if diagnostic == "linf":
        legacy_root = RESULT_DIR / f"ood_{network_tag}_inf"
        return legacy_root / f"{stem}_{network_tag}_linf.csv"
    return modern


def _calibration_thresholds() -> pd.DataFrame:
    path = CALIBRATION_ROOT / "thresholds.csv"
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
    if thresholds.duplicated(["network_tag", "assumed_model", "error_metric"]).any():
        raise ValueError("Gaussian calibration thresholds are not unique")
    return thresholds


def _long_error_frame(
    frame: pd.DataFrame,
    error_metric: str,
) -> pd.DataFrame:
    common = ["source_model", "id", "at_least_one_not_extrapolative"]
    if error_metric == "posterior_mmd":
        columns = common + [
            "assumed_model",
            "d_M",
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
            "dm_low",
            "dm_high",
            "signed_logml_error",
        ]
        available = [column for column in columns if column in frame]
        output = frame[available].copy()
        output["error_value"] = pd.to_numeric(
            output.pop("signed_logml_error"), errors="coerce"
        ) / np.log(10.0)
        return output

    rows = []
    for model in ASSUMED_MODELS:
        required = {
            f"d_{model}",
            f"dm_low_{model}",
            f"dm_high_{model}",
            f"signed_pmp_error_npe_{model}",
            "at_least_one_not_extrapolative",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"PMP cache is missing columns: {missing}")
        rows.append(
            pd.DataFrame(
                {
                    "source_model": frame["source_model"],
                    "id": frame["id"],
                    "at_least_one_not_extrapolative": frame[
                        "at_least_one_not_extrapolative"
                    ],
                    "assumed_model": model,
                    "d_M": frame[f"d_{model}"],
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
                path = _cache_path(diagnostic, network_tag, error_metric)
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

                # Recalculate rho from raw distances so cached values cannot drift.
                high = pd.to_numeric(frame["dm_high"], errors="coerce")
                if high.le(0.0).any() or (~np.isfinite(high)).any():
                    raise ValueError(f"Invalid diagnostic reference interval in {path}")
                frame = frame.assign(
                    diagnostic=diagnostic,
                    summary=summary,
                    network_tag=network_tag,
                    error_metric=error_metric,
                    rho=pd.to_numeric(frame["d_M"], errors="coerce") / high,
                    rho_low=pd.to_numeric(frame["dm_low"], errors="coerce") / high,
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
        raise ValueError(
            "Missing error thresholds for:\n" + missing.to_string(index=False)
        )

    logml = data["error_metric"].eq("log10_logml_error")
    data.loc[logml, threshold_columns] = data.loc[logml, threshold_columns] / np.log(
        10.0
    )
    data["normalized_error_value"] = _normalize_error_values(data)
    scale = _normalization_scale(data)
    data["degenerate_error_threshold"] = data["error_lower_threshold"].eq(0.0) & data[
        "error_upper_threshold"
    ].eq(0.0)
    data["normalized_error_lower_threshold"] = data["error_lower_threshold"] / scale
    data["normalized_error_upper_threshold"] = data["error_upper_threshold"] / scale
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
