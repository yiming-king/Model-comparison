import numpy as np

from benchmark.examples.diffusion.results.summary_dimension_comparison import (
    LOGML_CENTRAL_INTERVAL,
    LOWESS_FRAC,
    MODELS,
    NO_MMD_SUMMARY_SPECS,
    PLOT_SUMMARY_LABELS,
    _attach_calibration_thresholds,
    _central_interval,
    _load_cached_frames,
    _lowess_curve,
    _symlog_lowess_curve,
    prepare_rho_error_data,
)
from benchmark.examples.diffusion.results.summary_dimension_comparison import (
    CALIBRATION_THRESHOLD_PATH,
)


diagnostics, posteriors = _load_cached_frames("density", NO_MMD_SUMMARY_SPECS)
data = prepare_rho_error_data(
    diagnostics,
    posteriors,
    models=MODELS,
    pmp_source="four_model",
)
data = _attach_calibration_thresholds(
    data,
    NO_MMD_SUMMARY_SPECS,
    CALIBRATION_THRESHOLD_PATH,
)

print("COUNTS")
print(
    data.assign(
        source=np.where(data["dataset"].eq("empirical"), "empirical", "simulated")
    )
    .groupby(["model", "summary", "source"], observed=True)
    .size()
    .unstack(fill_value=0)
    .to_string()
)

value_columns = {
    "posterior_mmd": "observed_mmd",
    "logml": "logml_error",
    "pmp": "pmp_error",
}
print("\nCURVE_DIFFERENCES")
for value, y_column in value_columns.items():
    simulated = data.loc[data["dataset"].ne("empirical")].copy()
    combined = data.copy()
    if value == "logml":
        simulated = _central_interval(
            simulated,
            y_column,
            ["model", "summary"],
            LOGML_CENTRAL_INTERVAL,
        )
        combined = _central_interval(
            combined,
            y_column,
            ["model", "summary"],
            LOGML_CENTRAL_INTERVAL,
        )
    curve_function = _symlog_lowess_curve if value == "logml" else _lowess_curve
    differences = []
    for model in MODELS:
        for summary in PLOT_SUMMARY_LABELS:
            sim_group = simulated.loc[
                simulated["model"].eq(model) & simulated["summary"].eq(summary)
            ].dropna(subset=["rho", y_column])
            combined_group = combined.loc[
                combined["model"].eq(model) & combined["summary"].eq(summary)
            ].dropna(subset=["rho", y_column])
            sim_x, sim_y = curve_function(
                sim_group["rho"], sim_group[y_column], frac=LOWESS_FRAC["simulated"]
            )
            combined_x, combined_y = curve_function(
                combined_group["rho"],
                combined_group[y_column],
                frac=LOWESS_FRAC["simulated"],
            )
            lower = max(float(np.min(sim_x)), float(np.min(combined_x)))
            upper = min(float(np.max(sim_x)), float(np.max(combined_x)))
            grid = np.linspace(lower, upper, 300)
            delta = np.abs(
                np.interp(grid, sim_x, sim_y)
                - np.interp(grid, combined_x, combined_y)
            )
            differences.append(
                (
                    model,
                    str(summary),
                    len(sim_group),
                    len(combined_group),
                    float(delta.mean()),
                    float(delta.max()),
                )
            )
    mean_delta = np.mean([row[4] for row in differences])
    max_delta = max(row[5] for row in differences)
    largest = max(differences, key=lambda row: row[5])
    print(
        f"{value}: mean_abs_delta={mean_delta:.6g}, max_abs_delta={max_delta:.6g}, "
        f"largest={largest[0]}/{largest[1]} (n={largest[2]}->{largest[3]})"
    )
