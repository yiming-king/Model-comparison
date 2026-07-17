from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from ..approximators.indirect import load_approximator
from ..config import MODELS, RESULT_DIR, TrainingConfig
from ..simulators import SIMULATORS
from .observed_datasets import OBSERVED_DATASETS, load_observed_dataset, load_stan_posterior_draws
from .results import parameter_dict, posterior_draws


PPC_DIR = RESULT_DIR / "posterior_predictive_checks"
PPC_PATH = PPC_DIR / "stan_ppc.csv"
PPC_DENSITY_PATH = PPC_DIR / "stan_ppc_empirical_density.csv"


def _true_model(dataset: pd.Series) -> pd.Series:
    return dataset.str.extract(r"simulated_from_(m[0-3])", expand=False)


def confusion_matrices(
    results_by_summary: dict[str, pd.DataFrame],
    gold_label: str = "Gold standard",
    models: tuple[str, ...] = MODELS,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Return hard-classification and mean-PMP matrices for clean simulated data."""
    first = next(iter(results_by_summary.values()))
    methods = {gold_label: (first, "gold_pmp_")}
    methods.update({label: (frame, "pmp_") for label, frame in results_by_summary.items()})
    simulated_sources = tuple(f"simulated_from_{model}" for model in models)

    hard = {}
    mean_pmp = {}
    for label, (frame, prefix) in methods.items():
        data = frame[frame["dataset"].isin(simulated_sources)].copy()
        data["true_model"] = _true_model(data["dataset"])
        pmp_columns = [f"{prefix}{model}" for model in models]
        data["selected_model"] = np.asarray(models)[data[pmp_columns].to_numpy().argmax(axis=1)]

        counts = pd.crosstab(data["true_model"], data["selected_model"], normalize="index")
        hard[label] = counts.reindex(index=models, columns=models, fill_value=0.0)

        means = data.groupby("true_model", observed=True)[pmp_columns].mean()
        means.columns = models
        mean_pmp[label] = means.reindex(index=models, columns=models)
    return hard, mean_pmp


def plot_matrix_grid(
    matrices: dict[str, pd.DataFrame],
    title: str,
    colorbar_label: str,
    filename: str | Path | None = None,
):
    labels = list(matrices)
    fig, axes = plt.subplots(1, len(labels), figsize=(3.5 * len(labels), 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for ax, label in zip(axes, labels):
        matrix = matrices[label]
        values = matrix.to_numpy(dtype=float)
        image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="Blues")
        for row, col in np.ndindex(values.shape):
            color = "white" if values[row, col] > 0.55 else "black"
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center", color=color, fontsize=9)
        ax.set_title(label)
        ax.set_xticks(range(len(matrix.columns)), [f"$M_{{{model[1:]}}}$" for model in matrix.columns])
        ax.set_yticks(range(len(matrix.index)), [f"$M_{{{model[1:]}}}$" for model in matrix.index])
        ax.set_xlabel("Selected model" if "classification" in colorbar_label.lower() else "Assumed model")

    axes[0].set_ylabel("Data-generating model")
    fig.suptitle(title, y=1.02)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(colorbar_label)
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, axes


def plot_empirical_pmp_error_boxplots(
    results_by_summary: dict[str, pd.DataFrame],
    dataset: str = "empirical",
    models: tuple[str, ...] = MODELS,
    filename: str | Path | None = None,
):
    """Compare empirical signed PMP errors across summary dimensions."""
    labels = list(results_by_summary)
    colors = ("#E69F00", "#009E73", "#56B4E9", "#CC79A7")
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 3.8), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, model in zip(axes, models):
        values = [
            frame.loc[frame["dataset"].eq(dataset), f"signed_pmp_error_{model}"].dropna().to_numpy()
            for frame in results_by_summary.values()
        ]
        boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.62)
        for box, color in zip(boxes["boxes"], colors):
            box.set_facecolor(color)
            box.set_alpha(0.65)
        for median in boxes["medians"]:
            median.set_color("black")
            median.set_linewidth(1.4)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_title(f"$p(M_{{{model[1:]}}}\\mid y)$")
        ax.set_xlabel("Summary dimension")
        ax.grid(axis="y", alpha=0.2)

    axes[0].set_ylabel(r"$\hat{p}(M_j\mid y)-p(M_j\mid y)$")
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, axes


def _summary_statistics(rt: np.ndarray, conditions: np.ndarray) -> dict[str, float]:
    statistics = {}
    correct = rt > 0.0
    absolute_rt = np.abs(rt)
    for condition in (0, 1):
        condition_mask = conditions == condition
        statistics[f"accuracy_c{condition}"] = float(correct[condition_mask].mean())
        for response, response_mask in (("correct", correct), ("error", ~correct)):
            values = absolute_rt[condition_mask & response_mask]
            for quantile in (0.1, 0.5, 0.9):
                statistics[f"rt_q{int(quantile * 100):02d}_c{condition}_{response}"] = (
                    float(np.quantile(values, quantile)) if values.size else np.nan
                )
    return statistics


def _posterior_predictive_draws(
    dataset: str,
    model: str,
    dataset_id: str,
    num_draws: int,
    seed: int,
    expected_conditions: np.ndarray | None = None,
) -> np.ndarray:
    posterior = load_stan_posterior_draws(dataset, model, dataset_id)
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(posterior), size=min(num_draws, len(posterior)), replace=False)
    simulator = SIMULATORS[model]
    parameters = {
        "alpha": posterior.loc[keep, [column for column in posterior if column.startswith("alpha_")]].to_numpy(),
        "nu": posterior.loc[keep, ["nu_0", "nu_1"]].to_numpy(),
        "tau": posterior.loc[keep, ["tau"]].to_numpy(),
    }
    constrained = simulator._constrain_parameters(**parameters)
    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed + 10_000)
        simulated = simulator.likelihood((len(keep),), **constrained)
    finally:
        np.random.set_state(numpy_state)
    if expected_conditions is not None:
        expected_conditions = np.asarray(expected_conditions, dtype=int)
        simulated_conditions = np.asarray(simulated["conditions"], dtype=int)
        if simulated_conditions.shape[1:] != expected_conditions.shape or not np.all(
            simulated_conditions == expected_conditions[None, :]
        ):
            raise ValueError("Posterior predictive simulation did not preserve the observed condition sequence")
    return simulated["rt"]


def _predictive_check_rows(
    observed: np.ndarray,
    replicated: np.ndarray,
    ids: list[str],
    dataset: str,
    model: str,
    interval: float,
) -> list[dict]:
    lower_probability = (1.0 - interval) / 2.0
    rows = []
    for observation_index, dataset_id in enumerate(ids):
        conditions = observed[observation_index, :, 1].astype(int)
        observed_statistics = _summary_statistics(observed[observation_index, :, 0], conditions)
        replicated_statistics = pd.DataFrame(
            [_summary_statistics(draw, conditions) for draw in replicated[observation_index]]
        )
        for statistic, observed_value in observed_statistics.items():
            values = replicated_statistics[statistic].dropna().to_numpy()
            if not values.size or not np.isfinite(observed_value):
                continue
            lower, median, upper = np.quantile(
                values,
                [lower_probability, 0.5, 1.0 - lower_probability],
            )
            rows.append(
                {
                    "dataset": dataset,
                    "id": dataset_id,
                    "model": model,
                    "statistic": statistic,
                    "observed": observed_value,
                    "predictive_lower": lower,
                    "predictive_median": median,
                    "predictive_upper": upper,
                    "predictive_percentile": float(np.mean(values <= observed_value)),
                    "outside_interval": observed_value < lower or observed_value > upper,
                    "num_predictive_draws": len(values),
                    "requested_num_draws": replicated.shape[1],
                    "interval": interval,
                }
            )
    return rows


def posterior_predictive_frame(
    datasets: tuple[str, ...] = OBSERVED_DATASETS,
    num_draws: int = 256,
    interval: float = 0.90,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compute Stan posterior predictive checks for every dataset and assumed model."""
    rows = []
    for dataset_index, dataset in enumerate(datasets):
        observed, ids = load_observed_dataset(dataset)
        for model_index, model in enumerate(MODELS):
            replicated = np.stack(
                [
                    _posterior_predictive_draws(
                        dataset,
                        model,
                        dataset_id,
                        num_draws,
                        seed + 10_000 * dataset_index + 1_000 * model_index + observation_index,
                    )
                    for observation_index, dataset_id in enumerate(ids)
                ]
            )
            rows.extend(
                _predictive_check_rows(
                    observed,
                    replicated,
                    ids,
                    dataset,
                    model,
                    interval,
                )
            )
    return pd.DataFrame(rows)


def npe_posterior_predictive_frame(
    config: TrainingConfig,
    datasets: tuple[str, ...] = OBSERVED_DATASETS,
    num_draws: int = 256,
    interval: float = 0.90,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compute NPE posterior predictive checks for every dataset and assumed model."""
    approximators = {model: load_approximator(model, config=config) for model in MODELS}
    rows = []
    for dataset_index, dataset in enumerate(datasets):
        observed, ids = load_observed_dataset(dataset)
        for model_index, model in enumerate(MODELS):
            draws = posterior_draws(
                approximators[model],
                observed,
                num_samples=num_draws,
                seed=seed + 10_000 * dataset_index + model_index,
                batch_size=batch_size,
            )
            simulator = SIMULATORS[model]
            replicated = []
            for dataset_draws in draws:
                parameters = parameter_dict(dataset_draws, model)
                constrained = simulator._constrain_parameters(**parameters)
                replicated.append(simulator.likelihood((num_draws,), **constrained)["rt"])
            rows.extend(
                _predictive_check_rows(
                    observed,
                    np.stack(replicated),
                    ids,
                    dataset,
                    model,
                    interval,
                )
            )
    frame = pd.DataFrame(rows)
    frame["summary"] = config.summary_label
    return frame


def compute_or_load_ppc(
    path: str | Path = PPC_PATH,
    datasets: tuple[str, ...] = OBSERVED_DATASETS,
    num_draws: int = 256,
    interval: float = 0.90,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not recompute:
        frame = pd.read_csv(path, keep_default_na=False)
        settings_columns = {"interval", "requested_num_draws"}
        matches_settings = (
            settings_columns.issubset(frame.columns)
            and set(datasets).issubset(frame["dataset"].unique())
            and np.isclose(frame["interval"].astype(float), interval).all()
            and frame["requested_num_draws"].astype(int).eq(num_draws).all()
        )
        if matches_settings:
            return frame

    frame = posterior_predictive_frame(datasets, num_draws=num_draws, interval=interval, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def compute_or_load_npe_ppc(
    config: TrainingConfig,
    path: str | Path | None = None,
    datasets: tuple[str, ...] = OBSERVED_DATASETS,
    num_draws: int = 256,
    interval: float = 0.90,
    batch_size: int | None = 8,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    if path is None:
        path = PPC_DIR / f"npe_{config.summary_label}_ppc.csv"
    path = Path(path)
    if path.exists() and not recompute:
        frame = pd.read_csv(path, keep_default_na=False)
        matches_settings = (
            set(datasets).issubset(frame["dataset"].unique())
            and frame["summary"].eq(config.summary_label).all()
            and frame["requested_num_draws"].astype(int).eq(num_draws).all()
            and np.isclose(frame["interval"].astype(float), interval).all()
        )
        if matches_settings:
            return frame

    frame = npe_posterior_predictive_frame(
        config=config,
        datasets=datasets,
        num_draws=num_draws,
        interval=interval,
        batch_size=batch_size,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def _signed_rt_kde(
    values: np.ndarray,
    grid: np.ndarray,
    bw_method: str | float,
    min_observations: int,
) -> np.ndarray | None:
    """Evaluate one joint signed-RT KDE, or return None when it is unstable."""
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if len(values) < min_observations or np.ptp(values) == 0.0:
        return None
    try:
        return gaussian_kde(values, bw_method=bw_method)(grid)
    except (ValueError, np.linalg.LinAlgError):
        return None


def _signed_rt_density_rows(
    observed: np.ndarray,
    replicated: dict[str, np.ndarray],
    ids: list[str],
    dataset: str,
    source: str,
    summary: str,
    conditions: tuple[int | None, ...] = (None, 0, 1),
    interval: float = 0.90,
    grid_size: int = 300,
    bw_method: str | float = "scott",
    min_observations: int = 5,
) -> pd.DataFrame:
    """Create participant-level signed-RT KDE curves on common grids.

    RT is already response-coded by the data pipeline: positive means correct
    and negative means incorrect.  The two sides are deliberately fitted as a
    single distribution so their relative probability mass is retained.
    """
    observed = np.asarray(observed)
    if observed.ndim != 3 or observed.shape[-1] != 2:
        raise ValueError("observed must have shape (participants, trials, rt+condition)")
    if len(ids) != len(observed):
        raise ValueError("ids and observed must contain the same number of participants")
    if not 0.0 < interval < 1.0:
        raise ValueError("interval must lie strictly between 0 and 1")
    for model, draws in replicated.items():
        if draws.ndim != 3 or draws.shape[0] != len(observed) or draws.shape[2] != observed.shape[1]:
            raise ValueError(
                f"replicated[{model!r}] must have shape "
                f"({len(observed)}, draws, {observed.shape[1]})"
            )

    lower_probability = (1.0 - interval) / 2.0
    rows = []
    for participant_index, participant_id in enumerate(ids):
        participant_conditions = observed[participant_index, :, 1].astype(int)
        for condition in conditions:
            mask = np.ones(observed.shape[1], dtype=bool) if condition is None else participant_conditions == condition
            condition_label = "pooled" if condition is None else str(condition)
            observed_values = observed[participant_index, mask, 0]
            grid_values = [observed_values]
            grid_values.extend(draws[participant_index, :, mask].reshape(-1) for draws in replicated.values())
            finite_values = np.concatenate(grid_values)
            finite_values = finite_values[np.isfinite(finite_values)]
            if len(finite_values) < min_observations:
                raise ValueError(f"Not enough finite signed RTs for {dataset}/{participant_id}/{condition_label}")
            grid_low, grid_high = np.quantile(finite_values, [0.002, 0.998])
            if grid_low == grid_high:
                raise ValueError(f"Signed RT grid is degenerate for {dataset}/{participant_id}/{condition_label}")
            grid = np.linspace(grid_low, grid_high, grid_size)
            observed_density = _signed_rt_kde(observed_values, grid, bw_method, min_observations)
            if observed_density is None:
                raise ValueError(f"Observed KDE is not estimable for {dataset}/{participant_id}/{condition_label}")

            for model, draws in replicated.items():
                replicate_densities = []
                skipped = 0
                for draw_index, values in enumerate(draws[participant_index][:, mask]):
                    density = _signed_rt_kde(values, grid, bw_method, min_observations)
                    if density is None:
                        skipped += 1
                        continue
                    replicate_densities.append(density)
                    rows.extend(
                        {
                            "dataset": dataset,
                            "id": participant_id,
                            "source": source,
                            "summary": summary,
                            "condition": condition_label,
                            "model": model,
                            "curve": "y_rep",
                            "replicate": draw_index,
                            "rt": x,
                            "density": y,
                            "num_requested_replicates": draws.shape[1],
                            "num_valid_replicates": draws.shape[1] - skipped,
                            "num_skipped_replicates": skipped,
                            "interval": interval,
                            "grid_size": grid_size,
                            "bw_method": str(bw_method),
                            "min_observations": min_observations,
                        }
                        for x, y in zip(grid, density)
                    )

                if replicate_densities:
                    density_array = np.asarray(replicate_densities)
                    lower, median, upper = np.quantile(
                        density_array,
                        [lower_probability, 0.5, 1.0 - lower_probability],
                        axis=0,
                    )
                else:
                    lower = median = upper = np.full(grid_size, np.nan)

                common = {
                    "dataset": dataset,
                    "id": participant_id,
                    "source": source,
                    "summary": summary,
                    "condition": condition_label,
                    "model": model,
                    "replicate": -1,
                    "num_requested_replicates": draws.shape[1],
                    "num_valid_replicates": len(replicate_densities),
                    "num_skipped_replicates": skipped,
                    "interval": interval,
                    "grid_size": grid_size,
                    "bw_method": str(bw_method),
                    "min_observations": min_observations,
                }
                for curve, values in (
                    ("observed", observed_density),
                    ("predictive_lower", lower),
                    ("predictive_median", median),
                    ("predictive_upper", upper),
                ):
                    rows.extend(
                        {**common, "curve": curve, "rt": x, "density": y}
                        for x, y in zip(grid, values)
                    )
    frame = pd.DataFrame(rows)
    # The skipped count is final only after all replicates have been attempted.
    keys = ["dataset", "id", "source", "summary", "condition", "model"]
    final_counts = frame.groupby(keys, observed=True)["num_skipped_replicates"].transform("max")
    frame["num_skipped_replicates"] = final_counts
    frame["num_valid_replicates"] = frame["num_requested_replicates"] - final_counts
    return frame


def posterior_predictive_density_overlay_frame(
    source: str = "stan",
    config: TrainingConfig | None = None,
    dataset: str = "empirical",
    num_replicates: int = 80,
    batch_size: int | None = 8,
    conditions: tuple[int | None, ...] = (None, 0, 1),
    interval: float = 0.90,
    grid_size: int = 300,
    bw_method: str | float = "scott",
    min_observations: int = 5,
    seed: int = 2025,
) -> pd.DataFrame:
    observed, ids = load_observed_dataset(dataset)
    replicated = {}
    if source == "stan":
        for model_index, model in enumerate(MODELS):
            replicated[model] = np.stack(
                [
                    _posterior_predictive_draws(
                        dataset,
                        model,
                        dataset_id,
                        num_replicates,
                        seed + 1_000 * model_index + observation_index,
                        expected_conditions=observed[observation_index, :, 1],
                    )
                    for observation_index, dataset_id in enumerate(ids)
                ]
            )
        summary = ""
    elif source == "npe":
        if config is None:
            raise ValueError("config is required when source='npe'")
        for model_index, model in enumerate(MODELS):
            approximator = load_approximator(model, config=config)
            draws = posterior_draws(
                approximator,
                observed,
                num_samples=num_replicates,
                seed=seed + model_index,
                batch_size=batch_size,
            )
            simulator = SIMULATORS[model]
            model_replicates = []
            for dataset_draws in draws:
                parameters = parameter_dict(dataset_draws, model)
                constrained = simulator._constrain_parameters(**parameters)
                simulated = simulator.likelihood((num_replicates,), **constrained)
                observation_index = len(model_replicates)
                expected_conditions = observed[observation_index, :, 1].astype(int)
                simulated_conditions = np.asarray(simulated["conditions"], dtype=int)
                if simulated_conditions.shape[1:] != expected_conditions.shape or not np.all(
                    simulated_conditions == expected_conditions[None, :]
                ):
                    raise ValueError("Posterior predictive simulation did not preserve the observed condition sequence")
                model_replicates.append(simulated["rt"])
            replicated[model] = np.stack(model_replicates)
        summary = config.summary_label
    else:
        raise ValueError("source must be 'stan' or 'npe'")

    return _signed_rt_density_rows(
        observed,
        replicated,
        ids,
        dataset=dataset,
        source=source,
        summary=summary,
        conditions=conditions,
        interval=interval,
        grid_size=grid_size,
        bw_method=bw_method,
        min_observations=min_observations,
    )


def compute_or_load_ppc_density_overlay(
    source: str = "stan",
    config: TrainingConfig | None = None,
    path: str | Path | None = None,
    dataset: str = "empirical",
    num_replicates: int = 80,
    batch_size: int | None = 8,
    conditions: tuple[int | None, ...] = (None, 0, 1),
    interval: float = 0.90,
    grid_size: int = 300,
    bw_method: str | float = "scott",
    min_observations: int = 5,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    summary = "" if config is None else config.summary_label
    if path is None:
        label = source if source == "stan" else f"npe_{summary}"
        path = PPC_DIR / f"{label}_{dataset}_density_overlay.csv"
    path = Path(path)
    if path.exists() and not recompute:
        frame = pd.read_csv(path, keep_default_na=False)
        required = {
            "dataset", "id", "source", "summary", "condition", "num_requested_replicates",
            "interval", "grid_size", "bw_method", "min_observations",
        }
        matches_settings = required.issubset(frame.columns) and (
            frame["dataset"].eq(dataset).all()
            and frame["source"].eq(source).all()
            and frame["summary"].eq(summary).all()
            and set("pooled" if c is None else str(c) for c in conditions).issubset(
                frame["condition"].astype(str).unique()
            )
            and frame["num_requested_replicates"].astype(int).eq(num_replicates).all()
            and np.isclose(frame["interval"].astype(float), interval).all()
            and frame["grid_size"].astype(int).eq(grid_size).all()
            and frame["bw_method"].astype(str).eq(str(bw_method)).all()
            and frame["min_observations"].astype(int).eq(min_observations).all()
        )
        if matches_settings:
            return frame

    frame = posterior_predictive_density_overlay_frame(
        source=source,
        config=config,
        dataset=dataset,
        num_replicates=num_replicates,
        batch_size=batch_size,
        conditions=conditions,
        interval=interval,
        grid_size=grid_size,
        bw_method=bw_method,
        min_observations=min_observations,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def posterior_predictive_density_frame(
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compare pooled observed RT densities with Stan posterior predictive densities."""
    observed, ids = load_observed_dataset(dataset)
    replicated = {}
    for model_index, model in enumerate(MODELS):
        replicated[model] = np.stack(
            [
                _posterior_predictive_draws(
                    dataset,
                    model,
                    dataset_id,
                    num_draws,
                    seed + 1_000 * model_index + observation_index,
                )
                for observation_index, dataset_id in enumerate(ids)
            ]
        )

    return _density_frame(observed, replicated, dataset, interval, num_bins)


def _density_frame(
    observed: np.ndarray,
    replicated: dict[str, np.ndarray],
    dataset: str,
    interval: float,
    num_bins: int,
) -> pd.DataFrame:
    conditions = observed[0, :, 1].astype(int)
    lower_probability = (1.0 - interval) / 2.0
    num_draws = next(iter(replicated.values())).shape[1]
    rows = []
    for condition in (0, 1):
        condition_mask = conditions == condition
        observed_condition = observed[:, condition_mask, 0].reshape(-1)
        for response, sign in (("correct", 1), ("error", -1)):
            observed_values = np.abs(observed_condition[np.sign(observed_condition) == sign])
            predictive_values = [
                np.abs(values[np.sign(values) == sign])
                for draws in replicated.values()
                for values in [draws[:, :, condition_mask]]
            ]
            upper_limit = 1.05 * np.quantile(
                np.concatenate([observed_values, *predictive_values]),
                0.995,
            )
            edges = np.linspace(0.0, upper_limit, num_bins + 1)
            centers = (edges[:-1] + edges[1:]) / 2.0
            observed_density, _ = np.histogram(observed_values, bins=edges, density=True)
            observed_density = gaussian_filter1d(observed_density, sigma=1.2)

            for model, draws in replicated.items():
                densities = []
                for draw_index in range(draws.shape[1]):
                    values = draws[:, draw_index, condition_mask].reshape(-1)
                    values = np.abs(values[np.sign(values) == sign])
                    density, _ = np.histogram(values, bins=edges, density=True)
                    densities.append(gaussian_filter1d(density, sigma=1.2))
                lower, median, upper = np.quantile(
                    np.asarray(densities),
                    [lower_probability, 0.5, 1.0 - lower_probability],
                    axis=0,
                )
                for bin_center, observed_value, low, middle, high in zip(
                    centers,
                    observed_density,
                    lower,
                    median,
                    upper,
                ):
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "condition": condition,
                            "response": response,
                            "rt": bin_center,
                            "observed_density": observed_value,
                            "predictive_lower": low,
                            "predictive_median": middle,
                            "predictive_upper": high,
                            "requested_num_draws": num_draws,
                            "interval": interval,
                            "num_bins": num_bins,
                        }
                    )
    return pd.DataFrame(rows)


def npe_posterior_predictive_density_frame(
    config: TrainingConfig,
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    batch_size: int | None = 8,
    seed: int = 2025,
) -> pd.DataFrame:
    """Compare observed RT densities with NPE posterior predictive densities."""
    observed, _ = load_observed_dataset(dataset)
    replicated = {}
    numpy_state = np.random.get_state()
    try:
        for model_index, model in enumerate(MODELS):
            approximator = load_approximator(model, config=config)
            draws = posterior_draws(
                approximator,
                observed,
                num_samples=num_draws,
                seed=seed + model_index,
                batch_size=batch_size,
            )
            simulator = SIMULATORS[model]
            np.random.seed(seed + 10_000 + model_index)
            model_replicates = []
            for dataset_draws in draws:
                parameters = parameter_dict(dataset_draws, model)
                constrained = simulator._constrain_parameters(**parameters)
                model_replicates.append(
                    simulator.likelihood((num_draws,), **constrained)["rt"]
                )
            replicated[model] = np.stack(model_replicates)
    finally:
        np.random.set_state(numpy_state)

    frame = _density_frame(observed, replicated, dataset, interval, num_bins)
    frame["summary"] = config.summary_label
    return frame


def compute_or_load_ppc_density(
    path: str | Path = PPC_DENSITY_PATH,
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    path = Path(path)
    if path.exists() and not recompute:
        frame = pd.read_csv(path, keep_default_na=False)
        matches_settings = (
            frame["dataset"].eq(dataset).all()
            and frame["requested_num_draws"].astype(int).eq(num_draws).all()
            and np.isclose(frame["interval"].astype(float), interval).all()
            and frame["num_bins"].astype(int).eq(num_bins).all()
        )
        if matches_settings:
            return frame

    frame = posterior_predictive_density_frame(
        dataset=dataset,
        num_draws=num_draws,
        interval=interval,
        num_bins=num_bins,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def compute_or_load_npe_ppc_density(
    config: TrainingConfig,
    path: str | Path | None = None,
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    batch_size: int | None = 8,
    seed: int = 2025,
    recompute: bool = False,
) -> pd.DataFrame:
    if path is None:
        path = PPC_DIR / f"npe_{config.summary_label}_empirical_density.csv"
    path = Path(path)
    if path.exists() and not recompute:
        frame = pd.read_csv(path, keep_default_na=False)
        matches_settings = (
            frame["dataset"].eq(dataset).all()
            and frame["summary"].eq(config.summary_label).all()
            and frame["requested_num_draws"].astype(int).eq(num_draws).all()
            and np.isclose(frame["interval"].astype(float), interval).all()
            and frame["num_bins"].astype(int).eq(num_bins).all()
        )
        if matches_settings:
            return frame

    frame = npe_posterior_predictive_density_frame(
        config=config,
        dataset=dataset,
        num_draws=num_draws,
        interval=interval,
        num_bins=num_bins,
        batch_size=batch_size,
        seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def ppc_misfit_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize the fraction of observed statistics outside the predictive interval."""
    return (
        frame.groupby(["dataset", "model"], observed=True)
        .agg(
            outside_rate=("outside_interval", "mean"),
            mean_tail_distance=(
                "predictive_percentile",
                lambda value: float(np.mean(np.abs(np.asarray(value, dtype=float) - 0.5))),
            ),
            num_checks=("outside_interval", "size"),
        )
        .reset_index()
    )


def plot_ppc_misfit_heatmap(
    frame: pd.DataFrame,
    filename: str | Path | None = None,
    datasets: tuple[str, ...] = OBSERVED_DATASETS,
    models: tuple[str, ...] = MODELS,
):
    summary = ppc_misfit_summary(frame)
    interval = float(frame["interval"].iloc[0])
    table = summary.pivot(index="dataset", columns="model", values="outside_rate")
    table = table.reindex(index=datasets, columns=models)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(table.to_numpy(), vmin=0.0, vmax=1.0, cmap="magma_r", aspect="auto")
    for row, col in np.ndindex(table.shape):
        value = table.iloc[row, col]
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black")
    ax.set_xticks(range(len(models)), [f"$M_{{{model[1:]}}}$" for model in models])
    ax.set_yticks(range(len(table.index)), [label.replace("_", " ") for label in table.index])
    ax.set_xlabel("Assumed model")
    ax.set_ylabel("Observed-data source")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.05, pad=0.03)
    colorbar.set_label(f"Fraction outside {interval:.0%} predictive interval")
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_empirical_ppc(
    frame: pd.DataFrame,
    filename: str | Path | None = None,
    models: tuple[str, ...] = MODELS,
):
    data = frame[frame["dataset"].eq("empirical")].copy()
    interval = float(data["interval"].iloc[0])
    grouped = (
        data.groupby(["model", "statistic"], observed=True)["outside_interval"]
        .mean()
        .unstack("statistic")
        .reindex(index=models)
    )

    fig, ax = plt.subplots(figsize=(11.0, 3.8))
    image = ax.imshow(grouped.to_numpy(), vmin=0.0, vmax=1.0, cmap="magma_r", aspect="auto")
    ax.set_yticks(range(len(models)), [f"$M_{{{model[1:]}}}$" for model in models])
    ax.set_xticks(range(len(grouped.columns)), grouped.columns, rotation=45, ha="right")
    ax.set_ylabel("Assumed model")
    ax.set_xlabel("Posterior predictive statistic")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label(f"Fraction of participants outside {interval:.0%} interval")
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, ax


def plot_ppc_density_overlay(
    frame: pd.DataFrame,
    filename: str | Path | None = None,
    models: tuple[str, ...] = MODELS,
    participant_id: str | None = None,
    conditions: tuple[str, ...] = ("pooled", "0", "1"),
    envelope: bool = False,
):
    """Plot one participant, with pooled and condition-specific rows."""
    ids = frame["id"].drop_duplicates().tolist()
    if participant_id is None:
        if len(ids) != 1:
            raise ValueError("participant_id is required when frame contains multiple participants")
        participant_id = ids[0]
    data_for_id = frame[frame["id"].eq(participant_id)]
    conditions = tuple(str(condition) for condition in conditions)
    missing = set(conditions) - set(data_for_id["condition"].astype(str).unique())
    if missing:
        raise ValueError(f"Missing conditions for {participant_id}: {sorted(missing)}")

    fig, axes = plt.subplots(
        len(conditions),
        len(models),
        figsize=(3.5 * len(models), 3.0 * len(conditions)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    row_labels = {"pooled": "Pooled", "0": "Accuracy condition", "1": "Speed condition"}
    for row, condition in enumerate(conditions):
        for col, model in enumerate(models):
            ax = axes[row, col]
            data = data_for_id[
                data_for_id["model"].eq(model) & data_for_id["condition"].astype(str).eq(condition)
            ]
            observed = data[data["curve"].eq("observed")].sort_values("rt")
            if envelope:
                lower = data[data["curve"].eq("predictive_lower")].sort_values("rt")
                median = data[data["curve"].eq("predictive_median")].sort_values("rt")
                upper = data[data["curve"].eq("predictive_upper")].sort_values("rt")
                ax.fill_between(
                    lower["rt"], lower["density"], upper["density"],
                    color="#9ECAE1", alpha=0.35, linewidth=0,
                )
                ax.plot(median["rt"], median["density"], color="#4292C6", linewidth=1.8)
            else:
                predictive = data[data["curve"].eq("y_rep")]
                for _, curve in predictive.groupby("replicate", sort=False):
                    curve = curve.sort_values("rt")
                    ax.plot(curve["rt"], curve["density"], color="#9ECAE1", linewidth=0.7, alpha=0.28)
            ax.plot(observed["rt"], observed["density"], color="#08306B", linewidth=2.6)
            if row == 0:
                ax.set_title(f"Assumed $M_{{{model[1:]}}}$")
            if col == 0:
                ax.set_ylabel(f"{row_labels.get(condition, f'Condition {condition}')}\nDensity")
            if row == len(conditions) - 1:
                ax.set_xlabel("Signed RT (s)")
            ax.axvline(0.0, color="0.55", linewidth=0.7, linestyle=":")
            ax.grid(alpha=0.18)

    handles = [plt.Line2D([], [], color="#08306B", linewidth=2.6, label=r"$y$")]
    if envelope:
        handles.extend(
            [
                plt.Line2D([], [], color="#4292C6", linewidth=1.8, label=r"median $y_{\mathrm{rep}}$"),
                plt.Rectangle((0, 0), 1, 1, color="#9ECAE1", alpha=0.35, label="90% envelope"),
            ]
        )
    else:
        handles.append(plt.Line2D([], [], color="#9ECAE1", linewidth=1.0, label=r"$y_{\mathrm{rep}}$"))
    fig.suptitle(f"Signed-RT posterior predictive check: {participant_id}", y=0.995)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(bottom=0.09, top=0.93, wspace=0.08, hspace=0.16)
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, axes


def save_ppc_density_overlay_figures(
    frame: pd.DataFrame,
    output_dir: str | Path,
    models: tuple[str, ...] = MODELS,
    conditions: tuple[str, ...] = ("pooled", "0", "1"),
    include_envelope: bool = True,
) -> tuple[list[Path], pd.DataFrame]:
    """Save participant-level spaghetti plots and optional 90% envelope plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for participant_id in frame["id"].drop_duplicates():
        safe_id = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(participant_id))
        spaghetti_path = output_dir / f"{safe_id}_signed_rt_ppc_spaghetti.png"
        fig, _ = plot_ppc_density_overlay(
            frame,
            filename=spaghetti_path,
            models=models,
            participant_id=participant_id,
            conditions=conditions,
            envelope=False,
        )
        plt.close(fig)
        paths.append(spaghetti_path)
        if include_envelope:
            envelope_path = output_dir / f"{safe_id}_signed_rt_ppc_envelope.png"
            fig, _ = plot_ppc_density_overlay(
                frame,
                filename=envelope_path,
                models=models,
                participant_id=participant_id,
                conditions=conditions,
                envelope=True,
            )
            plt.close(fig)
            paths.append(envelope_path)

    diagnostics = (
        frame.groupby(["dataset", "id", "source", "summary", "condition", "model"], observed=True)
        .agg(
            requested_replicates=("num_requested_replicates", "first"),
            valid_replicates=("num_valid_replicates", "first"),
            skipped_replicates=("num_skipped_replicates", "first"),
        )
        .reset_index()
    )
    diagnostics.to_csv(output_dir / "signed_rt_ppc_skipped_replicates.csv", index=False)
    return paths, diagnostics


def plot_ppc_densities(
    frame: pd.DataFrame,
    filename: str | Path | None = None,
    models: tuple[str, ...] = MODELS,
    title: str | None = None,
):
    colors = {"correct": "#0072B2", "error": "#D55E00"}
    labels = {0: "Accuracy condition", 1: "Speed condition"}
    fig, axes = plt.subplots(2, len(models), figsize=(3.5 * len(models), 6.0), sharex=True, sharey=True)

    for row, condition in enumerate((0, 1)):
        for col, model in enumerate(models):
            ax = axes[row, col]
            for response in ("correct", "error"):
                data = frame[
                    frame["model"].eq(model)
                    & frame["condition"].eq(condition)
                    & frame["response"].eq(response)
                ].sort_values("rt")
                color = colors[response]
                ax.fill_between(
                    data["rt"],
                    data["predictive_lower"],
                    data["predictive_upper"],
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
                ax.plot(data["rt"], data["predictive_median"], color=color, linewidth=1.8)
                ax.plot(data["rt"], data["observed_density"], color=color, linewidth=1.5, linestyle="--")
            if row == 0:
                ax.set_title(f"Assumed $M_{{{model[1:]}}}$")
            if col == 0:
                ax.set_ylabel(f"{labels[condition]}\nDensity")
            if row == 1:
                ax.set_xlabel("Absolute RT (s)")
            ax.grid(alpha=0.2)

    handles = [
        plt.Line2D([], [], color=colors["correct"], linewidth=1.8, label="PPC correct"),
        plt.Line2D([], [], color=colors["correct"], linewidth=1.5, linestyle="--", label="Observed correct"),
        plt.Line2D([], [], color=colors["error"], linewidth=1.8, label="PPC error"),
        plt.Line2D([], [], color=colors["error"], linewidth=1.5, linestyle="--", label="Observed error"),
    ]
    if title is not None:
        fig.suptitle(title, y=1.01)
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.16, wspace=0.08, hspace=0.18)
    if filename is not None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return fig, axes


def generate_npe_ppc_density_figures(
    summary_multipliers: tuple[int, ...] = (1, 2, 4, 6),
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    batch_size: int | None = 8,
    seed: int = 2025,
    recompute: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate the NPE RT-density PPC figure for every requested summary dimension."""
    output_dir = Path(output_dir) if output_dir else RESULT_DIR / "plots" / "model_checks"
    rows = []
    for multiplier in summary_multipliers:
        config = TrainingConfig(summary_multiplier=multiplier)
        cache_path = PPC_DIR / f"npe_{config.summary_label}_{dataset}_density.csv"
        frame = compute_or_load_npe_ppc_density(
            config=config,
            dataset=dataset,
            num_draws=num_draws,
            interval=interval,
            num_bins=num_bins,
            batch_size=batch_size,
            seed=seed,
            path=cache_path,
            recompute=recompute,
        )
        path = output_dir / f"npe_ppc_{dataset}_rt_densities_{config.summary_label}.png"
        fig, _ = plot_ppc_densities(
            frame,
            filename=path,
            title=f"NPE posterior predictive densities (S={multiplier}D)" if multiplier > 1
            else "NPE posterior predictive densities (S=D)",
        )
        plt.close(fig)
        rows.append(
            {
                "summary": config.summary_label,
                "density_cache": str(cache_path),
                "figure": str(path),
            }
        )
    return pd.DataFrame(rows)


def generate_stan_ppc_density_figure(
    dataset: str = "empirical",
    num_draws: int = 128,
    interval: float = 0.90,
    num_bins: int = 80,
    seed: int = 2025,
    recompute: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate the RT-density PPC figure based on Stan posterior draws."""
    output_dir = Path(output_dir) if output_dir else RESULT_DIR / "plots" / "model_checks"
    cache_path = PPC_DIR / f"stan_ppc_{dataset}_density.csv"
    frame = compute_or_load_ppc_density(
        path=cache_path,
        dataset=dataset,
        num_draws=num_draws,
        interval=interval,
        num_bins=num_bins,
        seed=seed,
        recompute=recompute,
    )
    figure_path = output_dir / f"stan_ppc_{dataset}_rt_densities.png"
    fig, _ = plot_ppc_densities(
        frame,
        filename=figure_path,
        title="Stan posterior predictive densities",
    )
    plt.close(fig)
    return pd.DataFrame(
        [
            {
                "source": "Stan",
                "density_cache": str(cache_path),
                "figure": str(figure_path),
            }
        ]
    )
