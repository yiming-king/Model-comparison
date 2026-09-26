from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import LOG_DIR, TrainingConfig, ensure_dirs


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _command(args, multiplier: int) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmark.examples.diffusion.approximators.indirect",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-batches",
        str(args.num_batches),
        "--summary-multiplier",
        str(multiplier),
        "--embed-dim",
        str(args.embed_dim),
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.no_summary_mmd:
        cmd.append("--no-summary-mmd")
    if args.run_suffix:
        cmd.extend(["--run-suffix", args.run_suffix])
    if args.validation_size is not None:
        cmd.extend(
            [
                "--validation-size",
                str(args.validation_size),
                "--validation-freq",
                str(args.validation_freq),
                "--validation-seed",
                str(args.validation_seed),
            ]
        )
    return cmd


def _run_one(args, multiplier: int) -> tuple[int, Path]:
    ensure_dirs()
    summary_label = f"S{multiplier}D"
    if args.run_suffix:
        summary_label = f"{summary_label}_{args.run_suffix}"
    log_path = LOG_DIR / f"{summary_label}.log"
    env = os.environ.copy()
    env["KERAS_BACKEND"] = args.backend
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
    threads = str(args.threads_per_worker)
    env["TF_NUM_INTRAOP_THREADS"] = threads
    env["TF_NUM_INTEROP_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = threads
    env["VECLIB_MAXIMUM_THREADS"] = threads
    env["OPENBLAS_NUM_THREADS"] = threads

    with log_path.open("w") as log:
        log.write(
            f"KERAS_BACKEND={args.backend} "
            + " ".join(_command(args, multiplier))
            + "\n\n"
        )
        log.flush()
        completed = subprocess.run(
            _command(args, multiplier),
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode, log_path


def main() -> None:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-multipliers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument(
        "--backend", choices=("tensorflow", "jax", "torch"), default="tensorflow"
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-batches", type=int, default=defaults.num_batches)
    parser.add_argument("--embed-dim", type=int, default=defaults.embed_dim)
    parser.add_argument("--no-summary-mmd", action="store_true")
    parser.add_argument("--run-suffix")
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--validation-freq", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.threads_per_worker < 1:
        parser.error("--threads-per-worker must be at least 1")
    if any(multiplier < 1 for multiplier in args.summary_multipliers):
        parser.error("--summary-multipliers must contain positive integers")
    if args.validation_size is not None and args.validation_size < 1:
        parser.error("--validation-size must be at least 1")
    if args.validation_freq < 1:
        parser.error("--validation-freq must be at least 1")
    if args.embed_dim < 1:
        parser.error("--embed-dim must be at least 1")
    if args.no_summary_mmd and not args.run_suffix:
        parser.error("--run-suffix is required with --no-summary-mmd")

    if args.dry_run:
        for multiplier in args.summary_multipliers:
            print(
                f"KERAS_BACKEND={args.backend} " + " ".join(_command(args, multiplier))
            )
        return

    workers = min(args.max_workers, len(args.summary_multipliers))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_one, args, multiplier): multiplier
            for multiplier in args.summary_multipliers
        }
        for future in as_completed(futures):
            multiplier = futures[future]
            returncode, log_path = future.result()
            status = "finished" if returncode == 0 else f"failed with code {returncode}"
            print(f"S={multiplier}D {status}. Log: {log_path}")
            if returncode != 0:
                raise SystemExit(returncode)


if __name__ == "__main__":
    main()


# cd /Users/yimingzang/Documents/Project/benchmark2

# /opt/anaconda3/envs/benchmark2/bin/python \
#   -m benchmark.examples.diffusion.approximators.train_grid \
#   --summary-multipliers 1 2 6 \
#   --max-workers 3 \
#   --threads-per-worker 4 \
#   --no-summary-mmd \
#   --run-suffix noMMD \
