from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import LOG_DIR, ensure_dirs


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
    ]
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _run_one(args, multiplier: int) -> tuple[int, Path]:
    ensure_dirs()
    log_path = LOG_DIR / f"S{multiplier}D.log"
    env = os.environ.copy()
    env.setdefault("KERAS_BACKEND", "tensorflow")
    env.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

    with log_path.open("w") as log:
        log.write(" ".join(_command(args, multiplier)) + "\n\n")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-multipliers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
