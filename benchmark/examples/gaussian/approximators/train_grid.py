"""Run the six Gaussian dimension presets in separate training processes.

Example::

    python -m benchmark.examples.gaussian.approximators.train_grid --dry-run
    python -m benchmark.examples.gaussian.approximators.train_grid --max-workers 1
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import ASSUMED_MODELS, NETWORK_DIR
from .config import NOTEBOOK_PRESETS, TrainingConfig, model_path, validate_seed


PROJECT_ROOT = Path(__file__).resolve().parents[4]
LOG_DIR = NETWORK_DIR.parent / "training_logs"


def _command(args, preset: str, model: str) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "benchmark.examples.gaussian.approximators.indirect",
        "--preset",
        preset,
        "--models",
        model,
        "--output-dir",
        str(args.output_dir),
        "--validation-freq",
        str(args.validation_freq),
        "--validation-seed",
        str(args.validation_seed),
    ]
    # Let each worker resolve the same shared defaults unless overridden.
    for option in (
        "epochs",
        "batch_size",
        "num_batches",
        "learning_rate",
        "seed",
        "validation_size",
        "run_suffix",
    ):
        value = getattr(args, option)
        if value is not None:
            command.extend(["--" + option.replace("_", "-"), str(value)])
    if args.no_summary_mmd:
        command.append("--no-summary-mmd")
    if args.summary_mmd:
        command.append("--summary-mmd")
    if args.overwrite:
        command.append("--overwrite")
    return command


def _run_one(args, preset: str, model: str, target: Path) -> tuple[int, bool, Path]:
    log_path = args.log_dir / f"{target.stem}.log"
    # Check again inside the worker in case a queued job's model was created
    # while another job was running. Preserve its earlier training log.
    if target.exists() and not args.overwrite:
        return 0, True, log_path

    command = _command(args, preset, model)
    env = os.environ.copy()
    env["KERAS_BACKEND"] = args.backend
    threads = str(args.threads_per_worker)
    env["TF_NUM_INTRAOP_THREADS"] = threads
    env["TF_NUM_INTEROP_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = threads
    env["VECLIB_MAXIMUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"KERAS_BACKEND={args.backend} {shlex.join(command)}\n\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, False, log_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets",
        nargs="+",
        choices=tuple(NOTEBOOK_PRESETS),
        default=list(NOTEBOOK_PRESETS),
    )
    parser.add_argument(
        "--models", nargs="+", choices=ASSUMED_MODELS, default=list(ASSUMED_MODELS)
    )
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--backend", choices=("tensorflow", "jax", "torch"), default="tensorflow"
    )
    parser.add_argument("--epochs", type=int, help="Epochs for every model (default: 100).")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--validation-freq", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=2026)
    regularization = parser.add_mutually_exclusive_group()
    regularization.add_argument(
        "--summary-mmd", action="store_true",
        help="Opt into normal summary MMD regularization; adds _mmd to the network tag.",
    )
    regularization.add_argument(
        "--no-summary-mmd", action="store_true",
        help="Use the DeepSet default: no summary MMD regularization (already the default).",
    )
    parser.add_argument("--run-suffix", help="Extra suffix after the automatic _bf_default tag.")
    parser.add_argument("--output-dir", type=Path, default=NETWORK_DIR)
    parser.add_argument("--log-dir", type=Path, default=LOG_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for option in (
        "max_workers",
        "threads_per_worker",
        "validation_size",
        "validation_freq",
    ):
        value = getattr(args, option)
        if value is not None and value < 1:
            parser.error(f"--{option.replace('_', '-')} must be at least 1")
    try:
        validate_seed("validation_seed", args.validation_seed)
    except ValueError as error:
        parser.error(str(error))

    # Resolve relative paths before changing the child process working directory.
    args.output_dir = args.output_dir.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser().resolve()
    overrides = {
        name: getattr(args, name)
        for name in (
            "epochs",
            "batch_size",
            "num_batches",
            "learning_rate",
            "seed",
            "run_suffix",
        )
        if getattr(args, name) is not None
    }
    if args.summary_mmd:
        overrides["summary_base_distribution"] = "normal"

    jobs = []
    # Preserve the requested order while preventing duplicate concurrent writes.
    for preset in dict.fromkeys(args.presets):
        try:
            config = TrainingConfig.from_preset(preset, **overrides)
            targets = [
                (model, model_path(model, config, args.output_dir))
                for model in dict.fromkeys(args.models)
            ]
        except ValueError as error:
            parser.error(str(error))
        for model, target in targets:
            skipped = target.exists() and not args.overwrite
            status = "skip existing" if skipped else "train"
            print(
                f"[{status}] {model} / {preset}, epochs={config.epochs_for(model)} -> {target}",
                flush=True,
            )
            if args.dry_run:
                print(
                    f"  KERAS_BACKEND={args.backend} {shlex.join(_command(args, preset, model))}"
                )
            elif not skipped:
                jobs.append((preset, model, target))

    if args.dry_run or not jobs:
        return

    args.log_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(jobs))) as executor:
        futures = {
            executor.submit(_run_one, args, preset, model, target): (preset, model)
            for preset, model, target in jobs
        }
        for future in as_completed(futures):
            preset, model = futures[future]
            try:
                returncode, skipped, log_path = future.result()
            except OSError as error:
                failed = True
                print(
                    f"{model} / {preset} failed: {error}", file=sys.stderr, flush=True
                )
                continue
            if skipped:
                print(f"{model} / {preset} skipped: model already exists.", flush=True)
                continue
            status = "finished" if returncode == 0 else f"failed with code {returncode}"
            print(f"{model} / {preset} {status}. Log: {log_path}", flush=True)
            failed = failed or returncode != 0
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
