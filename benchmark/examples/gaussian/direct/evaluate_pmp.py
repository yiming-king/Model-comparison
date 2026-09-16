"""Evaluate Gaussian direct, existing NPE, and analytical PMP on one sealed bank.

No indirect training is performed. Cache filenames depend on the test data,
checkpoint weights, and inference settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd

from ..config import MODEL_SPECS, NETWORK_DIR, discover_network_sets
from ..analytic.analytic import GaussianAnalytical
from ..datasets.calculation import Calculation
from .pmp_data import (
    MODEL_ORDER, MODEL_PRIOR, file_sha256, load_bank, record_key,
    validate_probabilities, validate_records, versions, write_json,
)
from ..approximators.config import TrainingConfig
from ..approximators.direct import load_approximator


def _probabilities(logml):
    if logml.ndim != 2 or logml.shape[1] != 4 or not np.all(np.isfinite(logml)):
        raise ValueError("Natural logML must be a finite (datasets,4) matrix")
    # Reuse the existing stable normalization, with an explicit evaluation prior.
    from .calculator import softmax_stable
    return np.stack([softmax_stable(row + np.log(MODEL_PRIOR)) for row in logml])


def _base_metadata(method, bank, records, config_id, checkpoints, settings):
    metadata = {key: bank[key] for key in (
        "data_dim", "num_obs", "model_order", "model_prior", "preprocessing", "data_sha256",
    )}
    metadata.update(method=method, records=records, config_id=config_id,
                    checkpoints=checkpoints, settings=settings, versions=versions())
    # Changing data, weights, or inference settings selects a different cache file.
    key = [method, bank["data_sha256"], config_id,
           [checkpoints[name]["sha256"] for name in sorted(checkpoints)], settings]
    metadata["cache_id"] = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    return metadata


def _load_cache(path, bank):
    metadata = json.loads(path.with_suffix(".json").read_text())
    for field in ("model_order", "model_prior", "preprocessing"):
        if metadata[field] != bank[field]:
            raise ValueError(f"Comparison uses different {field}")
    order = validate_records(metadata["records"], bank["records"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if any(value.shape != (len(order), 4) for value in arrays.values()):
        raise ValueError("Prediction shape must be (number of datasets, 4)")
    arrays = {key: value[order] for key, value in arrays.items()}
    validate_probabilities(arrays["p"], len(order))
    return arrays, metadata


def _cached_compute(cache_dir, metadata, bank, compute):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{metadata['method']}-{metadata['cache_id']}.npz"
    if path.exists():
        arrays, saved = _load_cache(path, bank)
        print(f"Loaded {metadata['method']}: {path.name}")
        return path, arrays, saved
    arrays = compute()
    validate_probabilities(arrays["p"], len(bank["records"]))
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(path.with_suffix(".json"), metadata)
    print(f"Saved {metadata['method']}: {path.name}")
    return path, arrays, metadata


def gold_predictions(x, records, bank, cache_dir):
    metadata = _base_metadata("gold", bank, records, "analytical_gaussian", {}, {"log_base": "e"})

    def compute():
        logml = np.array([
            [GaussianAnalytical(
                obs_data=value.astype(np.float64), num_dims=bank["data_dim"], num_obs=bank["num_obs"],
                num_samples=0, **MODEL_SPECS[model],
            ).log_marginal_analytical() for model in MODEL_ORDER]
            for value in x
        ])
        return {"p": _probabilities(logml), "logml": logml}

    return _cached_compute(cache_dir, metadata, bank, compute)


def direct_predictions(x, records, bank, cache_dir, run_dir, summary_dim, batch_size):
    config = TrainingConfig(num_dims=bank["data_dim"], num_obs=bank["num_obs"], summary_dim=summary_dim)
    model = load_approximator(config, output_dir=run_dir)
    run = json.loads((Path(run_dir) / "metadata.json").read_text())
    checkpoint = Path(run_dir) / run["checkpoint"]
    metadata = _base_metadata("direct", bank, records, run["config_id"], {
        "classifier": {"path": str(checkpoint.resolve()), "sha256": file_sha256(checkpoint)}},
        {"summary_dim": summary_dim, "batch_size": batch_size})

    def compute():
        p = np.concatenate([np.asarray(model.predict(conditions={"x": x[start:start + batch_size]}, probs=True))
                            for start in range(0, len(x), batch_size)])
        return {"p": p}

    return _cached_compute(cache_dir, metadata, bank, compute)


def indirect_predictions(x, records, bank, cache_dir, network_set, num_samples, seed, batch_size):
    checkpoints = {model: {"path": str(path.resolve()), "sha256": file_sha256(path)}
                   for model, path in network_set.paths.items()}
    config_id = f"indirect_D{network_set.data_dim}_N{network_set.num_obs}_S{network_set.summary_dim}_{network_set.network_tag}"
    metadata = _base_metadata("indirect", bank, records, config_id, checkpoints,
                              {"summary_dim": network_set.summary_dim, "num_posterior_samples": num_samples,
                               "seed": seed, "batch_size": batch_size, "log_base": "e", "logml_method": "log_mean_exp"})

    def compute():
        from ..analysis.pipeline import load_approximators
        import keras
        keras.utils.set_random_seed(seed)
        approximators = load_approximators(network_set)
        logml, ess = np.empty((len(x), 4)), np.empty((len(x), 4))
        for j, candidate in enumerate(MODEL_ORDER):
            calculation = Calculation(approximators[candidate], **MODEL_SPECS[candidate],
                                      num_dims=bank["data_dim"], num_obs=bank["num_obs"],
                                      num_samples=num_samples, assumed_model=candidate,
                                      logml_method="log_mean_exp", rng=np.random.default_rng(seed + j))
            for start in range(0, len(x), batch_size):
                items = [{"x": value} for value in x[start:start + batch_size]]
                try:
                    calculation.npe_estimation(items, seed=seed + j * len(x) + start)
                except Exception as error:
                    keys = [record_key(record) for record in records[start:start + batch_size]]
                    raise RuntimeError(f"NPE inference failed for {candidate}, datasets {keys}") from error
                for i, item in enumerate(items, start):
                    logml[i, j] = item[f"npe_log_marginal_{candidate}"]
                    ess[i, j] = item[f"importance_ess_{candidate}"]
            print(f"Real NPE inference complete: {candidate}, S={network_set.summary_dim}")
        if not np.isfinite(ess).all() or np.any(ess <= 0):
            raise ValueError("NPE returned invalid importance ESS")
        return {"p": _probabilities(logml), "logml": logml, "ess": ess}

    return _cached_compute(cache_dir, metadata, bank, compute)


def compare(bank_path, gold_path, direct_path, indirect_path, output_dir, bins=10):
    """Match by source/id and persist all errors and calibration diagnostics."""
    _, records, bank = load_bank(bank_path)
    if bins < 1:
        raise ValueError("Calibration bins must be positive")
    results, metadata = {}, {}
    for method, path in (("gold", gold_path), ("direct", direct_path), ("indirect", indirect_path)):
        results[method], metadata[method] = _load_cache(Path(path), bank)
        if metadata[method]["method"] != method:
            raise ValueError(f"Wrong method cache provided for {method}")
    p_gold, p_direct, p_indirect = [results[method]["p"] for method in ("gold", "direct", "indirect")]
    errors = {"direct": p_direct - p_gold, "indirect": p_indirect - p_gold}
    tv = {method: 0.5 * np.abs(error).sum(axis=1) for method, error in errors.items()}
    labels = {"direct_config": metadata["direct"]["config_id"], "indirect_config": metadata["indirect"]["config_id"]}
    rows, paired = [], []
    for i, record in enumerate(records):
        identity = {**record, **labels}
        paired.append({**identity, "TV_direct": tv["direct"][i], "TV_indirect": tv["indirect"][i],
                       "delta_TV": tv["direct"][i] - tv["indirect"][i]})
        for j, candidate in enumerate(MODEL_ORDER):
            row = {**identity, "model": candidate, "p_gold": p_gold[i, j], "p_direct": p_direct[i, j],
                   "p_indirect": p_indirect[i, j], "logml_gold": results["gold"]["logml"][i, j],
                   "logml_indirect": results["indirect"]["logml"][i, j], "importance_ess": results["indirect"]["ess"][i, j],
                   "num_posterior_samples": metadata["indirect"]["settings"]["num_posterior_samples"],
                   "signed_logml_error": results["indirect"]["logml"][i, j] - results["gold"]["logml"][i, j]}
            for method in errors:
                row[f"signed_error_{method}"] = errors[method][i, j]
                row[f"absolute_error_{method}"] = abs(errors[method][i, j])
            rows.append(row)
    probabilities, paired = pd.DataFrame(rows), pd.DataFrame(paired)
    summary = paired.groupby(["source_model", *labels], sort=False).agg(
        count=("dataset_id", "size"), TV_direct=("TV_direct", "mean"), TV_indirect=("TV_indirect", "mean"),
        delta_TV=("delta_TV", "mean"), median_delta_TV=("delta_TV", "median"),
        direct_better_fraction=("delta_TV", lambda values: (values < 0).mean()),
    ).reset_index()
    model_summary = probabilities.groupby(["source_model", "model", *labels], sort=False).agg(
        count=("dataset_id", "size"), signed_error_direct=("signed_error_direct", "mean"),
        absolute_error_direct=("absolute_error_direct", "mean"), signed_error_indirect=("signed_error_indirect", "mean"),
        absolute_error_indirect=("absolute_error_indirect", "mean"), min_importance_ess=("importance_ess", "min"),
        mean_importance_ess=("importance_ess", "mean"),
    ).reset_index()
    id_indices = [i for i, record in enumerate(records) if record["source_model"] in MODEL_ORDER]
    id_counts = [sum(records[i]["source_model"] == model for i in id_indices) for model in MODEL_ORDER]
    if not all(id_counts) or len(set(id_counts)) != 1:
        raise ValueError("Uniform-prior ID calibration requires equal nonzero test counts for m1--m4")
    calibration, calibration_summary = [], []
    for method, p in (("direct", p_direct), ("indirect", p_indirect)):
        for j, candidate in enumerate(MODEL_ORDER):
            predicted = p[id_indices, j]
            observed = np.array([records[i]["source_model"] == candidate for i in id_indices], dtype=float)
            bin_indices = np.minimum((predicted * bins).astype(int), bins - 1)
            ece = 0.0
            for b in range(bins):
                mask = bin_indices == b
                count = int(mask.sum())
                mean = float(predicted[mask].mean()) if count else np.nan
                frequency = float(observed[mask].mean()) if count else np.nan
                gap = abs(mean - frequency)
                if count:
                    ece += count * gap / len(id_indices)
                calibration.append({**labels, "source_pool": "ID", "method": method, "model": candidate,
                                    "bin_lower": b / bins, "bin_upper": (b + 1) / bins, "count": count,
                                    "mean_probability": mean, "observed_frequency": frequency, "absolute_gap": gap})
            calibration_summary.append({**labels, "method": method, "model": candidate, "count": len(id_indices),
                                        "ece": ece, "brier": float(np.mean((predicted - observed)**2))})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames = {"probabilities": probabilities, "paired": paired, "summary": summary, "model_summary": model_summary,
              "calibration": pd.DataFrame(calibration), "calibration_summary": pd.DataFrame(calibration_summary)}
    for name, frame in frames.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_json(output_dir / "metadata.json", {
        "bank_id": bank["bank_id"], "data_sha256": bank["data_sha256"], "model_order": MODEL_ORDER,
        "model_prior": MODEL_PRIOR, "preprocessing": bank["preprocessing"], "count": len(records),
        "matched_by": ["source_model", "dataset_id"], "delta_TV_definition": "TV_direct - TV_indirect",
        "calibration": "independent balanced ID test only; fixed equal-width bins", "bins": bins,
        "failed_records": 0, "ess_filter": None, "methods": metadata,
    })
    print(f"Paired comparison saved: {output_dir}; {len(records)} datasets, no ESS filtering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Predict all methods and save a paired comparison")
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument("--direct-run", type=Path, required=True)
    run.add_argument("--summary-dim", type=int, choices=(20, 40, 80), required=True)
    run.add_argument("--indirect-tag", required=True, help="Exact existing m1--m4 checkpoint set tag")
    run.add_argument("--network-dir", type=Path, default=NETWORK_DIR)
    run.add_argument("--cache-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True, help="New comparison directory")
    run.add_argument("--num-posterior-samples", type=int, default=1000)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--seed", type=int, default=93001)
    run.add_argument("--calibration-bins", type=int, default=10)
    only = sub.add_parser("compare", help="Load three saved predictions and compute comparison tables")
    only.add_argument("--bank", type=Path, required=True)
    for method in ("gold", "direct", "indirect"):
        only.add_argument(f"--{method}", type=Path, required=True)
    only.add_argument("--output-dir", type=Path, required=True)
    only.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {args.output_dir}")
    if args.command == "compare":
        compare(args.bank, args.gold, args.direct, args.indirect, args.output_dir, args.calibration_bins)
        return
    if args.num_posterior_samples < 2 or args.batch_size < 1 or not 0 <= args.seed < 2**31:
        raise ValueError("Need >=2 posterior draws, positive batch size, and seed in [0,2**31)")
    x, records, bank = load_bank(args.bank)
    sets = discover_network_sets(args.network_dir, data_dim=bank["data_dim"], num_obs=bank["num_obs"],
                                 summary_dims=(args.summary_dim,), network_tags=(args.indirect_tag,))
    if len(sets) != 1:
        raise ValueError("Requested indirect tag must identify one complete m1--m4 set with matching actual D/N/S")
    direct = direct_predictions(x, records, bank, args.cache_dir, args.direct_run, args.summary_dim, args.batch_size)
    gold = gold_predictions(x, records, bank, args.cache_dir)
    indirect = indirect_predictions(x, records, bank, args.cache_dir, sets[0], args.num_posterior_samples, args.seed, args.batch_size)
    compare(args.bank, gold[0], direct[0], indirect[0], args.output_dir, args.calibration_bins)


if __name__ == "__main__":
    main()
