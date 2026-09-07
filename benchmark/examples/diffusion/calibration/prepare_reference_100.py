"""Combine the original 30 and incremental 70 calibration references.

The output follows the directory layout expected by ``pipeline.py metrics``::

    calibration_reference_100/
      datasets/m0/.../m3/          # s000--s099 plus one combined manifest
      mcmc/m0/m0/.../m3/...        # combined bridge/diagnostic tables

Dataset JSON files and Stan posterior CSV files are hard-linked by default, so
the combined reference does not duplicate the large source files.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = "benchmark.examples.diffusion.calibration"

from ..config import BASE_DIR, MODEL_LABELS, MODELS
from .pipeline import dataset_seed


DEFAULT_BASE_ROOT = BASE_DIR / "calibration_outputs"
DEFAULT_INCREMENT_ROOT = BASE_DIR / "calibration_increment_70"
DEFAULT_OUTPUT_ROOT = BASE_DIR / "calibration_reference_100"
TABLE_NAMES = ("bridgesampling.csv", "convergence_diagnostics.csv")


def _expected_ids(start: int, stop: int) -> list[str]:
    return [f"s{index:03d}" for index in range(start, stop)]


def _read_id_table(path: Path, expected_ids: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, keep_default_na=False)
    if "id" not in frame:
        raise ValueError(f"{path} has no id column")
    if frame["id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["id"].duplicated(), "id"].unique())
        raise ValueError(f"Duplicate ids in {path}: {duplicates}")
    actual_ids = frame["id"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError(
            f"Unexpected ids in {path}; expected {expected_ids[0]}--"
            f"{expected_ids[-1]}, got {actual_ids[:1]}--{actual_ids[-1:]}"
        )
    return frame


def _write_combined_table(
    base_path: Path,
    increment_path: Path,
    output_path: Path,
    *,
    base_ids: list[str],
    increment_ids: list[str],
) -> pd.DataFrame:
    base = _read_id_table(base_path, base_ids)
    increment = _read_id_table(increment_path, increment_ids)
    if list(base.columns) != list(increment.columns):
        raise ValueError(
            f"CSV schemas differ: {base_path} has {list(base.columns)}, "
            f"but {increment_path} has {list(increment.columns)}"
        )
    combined = pd.concat([base, increment], ignore_index=True)
    if combined["id"].duplicated().any():
        raise ValueError(f"Duplicate ids while combining {base_path} and {increment_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def _install_file(
    source: Path,
    destination: Path,
    *,
    link_mode: str,
    overwrite: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        try:
            if os.path.samefile(source, destination):
                return
        except (FileNotFoundError, OSError):
            pass
        if not overwrite:
            raise FileExistsError(
                f"{destination} already exists; use --overwrite to replace it"
            )
        destination.unlink()
    if link_mode == "hardlink":
        os.link(source, destination)
    elif link_mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    elif link_mode == "copy":
        shutil.copy2(source, destination)
    else:
        raise ValueError(f"Unknown link mode: {link_mode}")


def prepare_reference(
    *,
    base_root: Path,
    increment_root: Path,
    output_root: Path,
    base_count: int = 30,
    total_count: int = 100,
    base_seed: int = 2025,
    link_mode: str = "hardlink",
    overwrite: bool = False,
) -> None:
    """Create and validate one reference root containing all calibration data."""
    if not 0 < base_count < total_count:
        raise ValueError("Expected 0 < base_count < total_count")
    base_root = base_root.resolve()
    increment_root = increment_root.resolve()
    output_root = output_root.resolve()
    if output_root in {base_root, increment_root}:
        raise ValueError("output_root must differ from both source roots")

    base_ids = _expected_ids(0, base_count)
    increment_ids = _expected_ids(base_count, total_count)
    all_ids = base_ids + increment_ids
    manifest_frames = []

    for generating_model in MODELS:
        base_dataset_dir = base_root / "datasets" / generating_model
        increment_dataset_dir = increment_root / "datasets" / generating_model
        output_dataset_dir = output_root / "datasets" / generating_model
        manifest = _write_combined_table(
            base_dataset_dir / "true_parameters.csv",
            increment_dataset_dir / "true_parameters.csv",
            output_dataset_dir / "true_parameters.csv",
            base_ids=base_ids,
            increment_ids=increment_ids,
        )
        expected_seeds = [
            dataset_seed(base_seed, generating_model, index)
            for index in range(total_count)
        ]
        if pd.to_numeric(manifest["dataset_seed"]).tolist() != expected_seeds:
            raise ValueError(
                f"Dataset seeds do not match k={total_count}, base_seed={base_seed} "
                f"for {generating_model}"
            )
        for dataset_id in all_ids:
            source_dir = base_dataset_dir if dataset_id in base_ids else increment_dataset_dir
            _install_file(
                source_dir / f"{dataset_id}.json",
                output_dataset_dir / f"{dataset_id}.json",
                link_mode=link_mode,
                overwrite=overwrite,
            )

        labelled_manifest = manifest.copy()
        labelled_manifest.insert(0, "generating_model", generating_model)
        labelled_manifest.insert(1, "generating_model_label", MODEL_LABELS[generating_model])
        manifest_frames.append(labelled_manifest)

        for candidate_model in MODELS:
            base_mcmc_dir = base_root / "mcmc" / generating_model / candidate_model
            increment_mcmc_dir = (
                increment_root
                / "stan_results"
                / f"generating_{generating_model}"
                / candidate_model
            )
            output_mcmc_dir = output_root / "mcmc" / generating_model / candidate_model
            for table_name in TABLE_NAMES:
                combined = _write_combined_table(
                    base_mcmc_dir / table_name,
                    increment_mcmc_dir / table_name,
                    output_mcmc_dir / table_name,
                    base_ids=base_ids,
                    increment_ids=increment_ids,
                )
                if combined["id"].astype(str).tolist() != all_ids:
                    raise ValueError(f"Incomplete combined table: {output_mcmc_dir / table_name}")

            # Only the matching candidate posterior is used for posterior MMD.
            # Link non-matching draws too when they happen to be available.
            installed_draw_ids = []
            for dataset_id in all_ids:
                source_dir = base_mcmc_dir if dataset_id in base_ids else increment_mcmc_dir
                source = source_dir / "posterior_draws" / f"{dataset_id}.csv"
                if not source.exists():
                    continue
                _install_file(
                    source,
                    output_mcmc_dir / "posterior_draws" / source.name,
                    link_mode=link_mode,
                    overwrite=overwrite,
                )
                installed_draw_ids.append(dataset_id)
            if candidate_model == generating_model and installed_draw_ids != all_ids:
                missing = sorted(set(all_ids) - set(installed_draw_ids))
                raise ValueError(
                    f"Missing matching-model posterior draws for {generating_model}: {missing}"
                )

    output_root.mkdir(parents=True, exist_ok=True)
    pd.concat(manifest_frames, ignore_index=True).to_csv(
        output_root / "dataset_manifest.csv", index=False
    )
    print(f"Combined calibration reference: {output_root}")
    print(
        f"Datasets: {len(MODELS)} generating models x {total_count}; "
        f"Stan tables: {len(MODELS)} x {len(MODELS)} x {total_count}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, default=DEFAULT_BASE_ROOT)
    parser.add_argument("--increment-root", type=Path, default=DEFAULT_INCREMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--base-count", type=int, default=30)
    parser.add_argument("--total-count", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to install large JSON/posterior files (default: hardlink).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    prepare_reference(
        base_root=args.base_root,
        increment_root=args.increment_root,
        output_root=args.output_root,
        base_count=args.base_count,
        total_count=args.total_count,
        base_seed=args.base_seed,
        link_mode=args.link_mode,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
