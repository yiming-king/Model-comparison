"""Recompute RDM summary diagnostics over shared raw references, without inference.

Run --all to process saved configurations sequentially in separate child
processes. Candidate reference caches allow interruption and resumption.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def configurations():
    # Current direct comparison first, then the other existing inference caches.
    return ["S4D", "S1D", "S2D", "S6D"] + [
        f"S{multiplier}D_{suffix}"
        for suffix in ("noMMD", "noMMD_rerun1") for multiplier in (1, 2, 4, 6)
    ]


def run_indirect(tag):
    import keras
    import pandas as pd
    from ..config import MODELS, TrainingConfig, get_name, get_path
    from ..results.multisource_pipeline import (
        load_approximators, load_or_fit_reference_suite, results_path,
        ensure_observed_summary_diagnostics,
    )
    from ..results.summary_diagnostic import REFERENCE_METRICS

    suffix = tag.split("_", 1)[1] if "_" in tag else None
    multiplier = int(tag.split("_", 1)[0][1:-1])
    config = TrainingConfig(summary_multiplier=multiplier,
                            summary_base_distribution=None if suffix else "normal",
                            run_suffix=suffix)
    if not all(get_path(get_name(m, summary_label=tag)).exists() for m in MODELS):
        raise FileNotFoundError(f"Incomplete checkpoint set for {tag}")
    result_file = results_path(tag)
    if not result_file.exists():
        raise FileNotFoundError(f"This command requires existing inference results: {result_file}")
    keras.utils.disable_interactive_logging()
    approximators = load_approximators(config)
    references = load_or_fit_reference_suite(approximators, config, metrics=REFERENCE_METRICS)
    results = pd.read_csv(result_file, keep_default_na=False)
    diagnostics = ensure_observed_summary_diagnostics(config, results, approximators, references)
    print(f"Completed {tag}: {len(results)} observed datasets, {len(diagnostics)} diagnostics", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--indirect-tag")
    group.add_argument("--direct-loss", choices=("cross_entropy", "exponential", "logistic"))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.indirect_tag:
        run_indirect(args.indirect_tag)
    elif args.direct_loss:
        from .direct_diagnostics import ensure_direct_diagnostics
        ensure_direct_diagnostics(args.direct_loss)
    else:
        from ..config import BASE_DIR
        from ..results.shared_reference_data import write_json
        logs = args.log_dir or BASE_DIR / "results/shared_reference_recompute"
        logs.mkdir(parents=True, exist_ok=True)
        tasks = [("direct_cross_entropy", ["--direct-loss", "cross_entropy"])] + [
            (f"indirect_{tag}", ["--indirect-tag", tag]) for tag in configurations()
        ]
        completed = []
        for name, flags in tasks:
            status = {"running": name, "completed": completed,
                      "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(logs / "status.json", status)
            print(f"Starting {name}; log {logs / (name + '.log')}", flush=True)
            environment = {**os.environ, "KERAS_BACKEND": "jax", "MPLCONFIGDIR": "/private/tmp/matplotlib"}
            with (logs / f"{name}.log").open("a") as stream:
                result = subprocess.run([sys.executable, "-m", __package__ + ".recompute_shared_diagnostics", *flags],
                                        stdout=stream, stderr=subprocess.STDOUT, env=environment)
            if result.returncode:
                write_json(logs / "status.json", {**status, "failed": name, "exit_code": result.returncode})
                raise SystemExit(result.returncode)
            completed.append(name)
        write_json(logs / "status.json", {"running": None, "completed": completed, "complete": True,
                                           "updated_at_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
