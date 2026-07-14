from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..config import MODELS, RESULT_DIR, TrainingConfig
from .multisource_pipeline import all_observed_diagnostic_path


REGIMES = {"low surprise", "in_distribution", "high surprise"}
DEFAULT_SUMMARY_MULTIPLIERS = (1, 2, 4, 6)
OUTPUT_DIR = RESULT_DIR / "ood_method_comparison"


def _load_diagnostic(config: TrainingConfig, metric: str) -> pd.DataFrame:
    path = all_observed_diagnostic_path(config.summary_label, metric=metric)
    if not path.exists():
        raise FileNotFoundError(f"Missing diagnostic cache: {path}")
    return pd.read_csv(path, keep_default_na=False)


def compare_methods(
    config: TrainingConfig,
    pmp_threshold: float = 0.05,
    datasets: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Compare paired Mahalanobis and density decisions on simulated data."""
    md = _load_diagnostic(config, metric="l2")
    density = _load_diagnostic(config, metric="typical")

    if datasets is None:
        datasets = tuple(name for name in md["dataset"].unique() if name != "empirical")

    keys = ["dataset", "id"]
    rows = []
    for model in MODELS:
        md_columns = keys + [f"regime_{model}", f"abs_pmp_error_{model}"]
        density_columns = keys + [f"regime_{model}"]
        paired = md[md_columns].merge(
            density[density_columns],
            on=keys,
            how="inner",
            validate="one_to_one",
            suffixes=("_md", "_density"),
        )
        paired = paired[paired["dataset"].isin(datasets)].copy()
        paired = paired.rename(
            columns={
                f"regime_{model}_md": "md_regime",
                f"regime_{model}_density": "density_regime",
                f"abs_pmp_error_{model}": "abs_pmp_error",
            }
        )
        observed_regimes = set(paired["md_regime"]) | set(paired["density_regime"])
        if not observed_regimes.issubset(REGIMES):
            raise ValueError(f"Unexpected regime labels: {sorted(observed_regimes - REGIMES)}")

        paired["summary"] = config.summary_label
        paired["model"] = model
        paired["large_pmp_error"] = paired["abs_pmp_error"] > pmp_threshold
        # Either tail is outside the calibrated typical set and therefore counts as detected.
        paired["md_detected"] = paired["md_regime"] != "in_distribution"
        paired["density_detected"] = paired["density_regime"] != "in_distribution"
        paired["md_missed_large_error"] = paired["large_pmp_error"] & ~paired["md_detected"]
        paired["density_missed_large_error"] = paired["large_pmp_error"] & ~paired["density_detected"]
        paired["same_regime"] = paired["md_regime"] == paired["density_regime"]
        paired["same_detection"] = paired["md_detected"] == paired["density_detected"]
        rows.append(paired)

    columns = [
        "summary",
        "dataset",
        "id",
        "model",
        "abs_pmp_error",
        "large_pmp_error",
        "md_regime",
        "density_regime",
        "md_detected",
        "density_detected",
        "md_missed_large_error",
        "density_missed_large_error",
        "same_regime",
        "same_detection",
    ]
    return pd.concat(rows, ignore_index=True)[columns]


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarize disagreement and missed large-PMP-error cases by dataset and model."""
    summary = (
        comparison.groupby(["summary", "dataset", "model"], observed=True)
        .agg(
            n=("id", "size"),
            n_large_pmp_error=("large_pmp_error", "sum"),
            md_missed_count=("md_missed_large_error", "sum"),
            density_missed_count=("density_missed_large_error", "sum"),
            exact_regime_agreement=("same_regime", "mean"),
            detection_agreement=("same_detection", "mean"),
        )
        .reset_index()
    )
    denominator = summary["n_large_pmp_error"].where(summary["n_large_pmp_error"] > 0)
    summary["md_miss_rate"] = summary["md_missed_count"] / denominator
    summary["density_miss_rate"] = summary["density_missed_count"] / denominator
    return summary


def run_comparison(
    summary_multipliers: tuple[int, ...] = DEFAULT_SUMMARY_MULTIPLIERS,
    pmp_threshold: float = 0.05,
    output_dir: str | Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Run the cached comparison and save concise machine-readable tables."""
    comparisons = [
        compare_methods(TrainingConfig(summary_multiplier=multiplier), pmp_threshold=pmp_threshold)
        for multiplier in summary_multipliers
    ]
    comparison = pd.concat(comparisons, ignore_index=True)
    missed = comparison[
        comparison["md_missed_large_error"] | comparison["density_missed_large_error"]
    ].copy()
    disagreements = comparison[~comparison["same_regime"]].copy()
    summary = summarize_comparison(comparison)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "comparison": comparison,
        "missed": missed,
        "disagreements": disagreements,
        "summary": summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Mahalanobis and density typical-set decisions on simulated datasets."
    )
    parser.add_argument("--pmp-threshold", type=float, default=0.05)
    parser.add_argument("--summary-multipliers", type=int, nargs="+", default=DEFAULT_SUMMARY_MULTIPLIERS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    outputs = run_comparison(
        summary_multipliers=tuple(args.summary_multipliers),
        pmp_threshold=args.pmp_threshold,
        output_dir=args.output_dir,
    )
    print(f"Saved comparison tables to {args.output_dir}")
    print(f"Compared rows: {len(outputs['comparison'])}")
    print(f"Missed large-error rows: {len(outputs['missed'])}")
    print(f"Exact three-class disagreements: {len(outputs['disagreements'])}")


if __name__ == "__main__":
    main()
