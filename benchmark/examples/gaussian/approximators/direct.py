"""Train a Gaussian model classifier with an independent, learned set summary.

Run with ``python -m benchmark.examples.gaussian.approximators.direct --help``.
Only raw observations enter the network; one-hot generating-model indices are
categorical cross-entropy targets. All four generating-model probabilities are
explicitly 1/4. Test-bank observations are used only to reject split overlap.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path

import numpy as np

from .config import TrainingConfig, positive_int, validate_seed
from .simulators import get_simulator
from ..config import MODEL_SPECS
from ..direct.pmp_data import (
    MODEL_ORDER,
    MODEL_PRIOR,
    PREPROCESSING,
    array_sha256,
    file_sha256,
    load_bank,
    versions,
    write_json,
)


def _runtime():
    """Use the installed TensorFlow backend and register BayesFlow serializers."""
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    import bayesflow as bf
    import keras

    return bf, keras


def build_adapter():
    """Use raw observations as inputs and generating-model indicators as targets."""
    bf, _ = _runtime()
    return bf.ModelComparisonApproximator.build_adapter(
        inference_variables="model_indices", summary_variables="x"
    )


def build_approximator(config: TrainingConfig = TrainingConfig()):
    """Build a fresh trainable summary and classifier with cross-entropy only."""
    if config.summary_base_distribution is not None:
        raise ValueError("Direct PMP uses categorical cross-entropy only; summary MMD is not supported")
    bf, keras = _runtime()
    keras.utils.set_random_seed(config.seed)
    approximator = bf.ModelComparisonApproximator(
        num_models=4,
        adapter=build_adapter(),
        summary_network=bf.networks.DeepSet(summary_dim=config.summary_dim, base_distribution=None),
        classifier_network=bf.networks.MLP(widths=(64, 64), dropout=None),
        standardize=None,
    )
    approximator.compile(optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate))
    return approximator


def load_approximator(
    config: TrainingConfig = TrainingConfig(), *, output_dir: str | Path
):
    """Load a direct checkpoint matching the requested raw D, N, and summary S."""
    if config.summary_base_distribution is not None:
        raise ValueError("Direct PMP does not use summary MMD")
    run_dir = Path(output_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    saved = metadata["config"]
    actual = (saved["num_dims"], saved["num_obs"], saved["summary_dim"])
    expected = (config.num_dims, config.num_obs, config.summary_dim)
    if actual != expected:
        raise ValueError(f"Direct checkpoint D/N/S {actual} do not match requested {expected}")
    _, keras = _runtime()
    model = keras.models.load_model(run_dir / metadata["checkpoint"], compile=False)
    if model.num_models != 4 or model.summary_network.summary_dim != config.summary_dim:
        raise ValueError("Saved network architecture disagrees with direct metadata")
    return model


def load_history(output_dir: str | Path) -> dict:
    """Read the saved training and validation loss history."""
    return json.loads((Path(output_dir) / "history.json").read_text())


def generate_data(config: TrainingConfig, size: int, seed: int) -> dict:
    """Sample the explicitly uniform model mixture using independent RNG streams."""
    positive_int("size", size)
    validate_seed("seed", seed)
    _runtime()
    streams = np.random.SeedSequence(seed).spawn(5)
    simulator_seeds = [int(s.generate_state(1)[0]) for s in streams[:4]]
    rng = np.random.default_rng(streams[4])
    labels = rng.choice(4, size=size, p=np.asarray(MODEL_PRIOR))
    x = np.empty((size, config.num_obs, config.num_dims), dtype=np.float32)
    for index, name in enumerate(MODEL_ORDER):
        selected = np.flatnonzero(labels == index)
        if selected.size:
            simulator = get_simulator(name, config=config, seed=simulator_seeds[index])
            x[selected] = simulator.sample(selected.size)["x"]
    return {"x": x, "model_indices": np.eye(4, dtype=np.float32)[labels]}


def train_approximator(
    config: TrainingConfig = TrainingConfig(),
    *,
    bank: str | Path,
    output_dir: str | Path,
    validation_size: int = 2048,
    validation_seed: int = 2026,
    verbose: int = 2,
):
    """Train on a fixed independent simulation bank; select by validation CCE."""
    _, bank_records, bank_metadata = load_bank(bank)
    config = replace(config, epochs=config.epochs or 100)
    validate_seed("validation_seed", validation_seed)
    if validation_size < 4:
        raise ValueError("validation_size must be at least four")
    if config.seed == validation_seed:
        raise ValueError("Training and validation seeds must differ")
    if bank_metadata.get("seed") in (config.seed, validation_seed):
        raise ValueError("Training/validation seed must differ from the test-bank seed")
    if (config.num_dims, config.num_obs) != (bank_metadata["data_dim"], bank_metadata["num_obs"]):
        raise ValueError("Requested raw D/N differ from the test bank")
    run_dir = Path(output_dir)
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing direct run: {run_dir}")
    model = build_approximator(config)
    bf, keras = _runtime()
    training = generate_data(config, config.batch_size * config.num_batches, config.seed)
    validation = generate_data(config, validation_size, validation_seed)
    training_hashes = [array_sha256(x) for x in training["x"]]
    validation_hashes = [array_sha256(x) for x in validation["x"]]
    bank_hashes = {r["input_sha256"] for r in bank_records}
    if set(training_hashes) & set(validation_hashes) or bank_hashes & (set(training_hashes) | set(validation_hashes)):
        raise ValueError("Training, validation, and final test observations must be disjoint")

    training_dataset = bf.datasets.OfflineDataset(training, config.batch_size, adapter=model.adapter, shuffle=True)
    validation_dataset = bf.datasets.OfflineDataset(validation, config.batch_size, adapter=model.adapter, shuffle=False)
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", {
        "config": asdict(config), "validation_size": validation_size,
        "validation_seed": validation_seed, "training_size": len(training_hashes),
        "model_order": list(MODEL_ORDER), "model_prior": list(MODEL_PRIOR),
        "evaluation_model_prior": list(MODEL_PRIOR), "preprocessing": PREPROCESSING,
        "test_bank_for_overlap_guard": str(Path(bank).resolve()),
        "training_bank_id": bank_metadata["bank_id"], "versions": versions(),
    })
    write_json(run_dir / "split_manifest.json", {
        "training_seed": config.seed, "validation_seed": validation_seed,
        "training_input_sha256": training_hashes, "validation_input_sha256": validation_hashes,
    })
    checkpoint = run_dir / "best.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min", save_best_only=True),
        keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
    history = model.fit(
        dataset=training_dataset, validation_data=validation_dataset,
        epochs=config.epochs, callbacks=callbacks, verbose=verbose,
    )
    history_data = {key: [float(v) for v in value] for key, value in history.history.items()}
    if not history_data.get("val_loss") or not all(np.isfinite(v).all() for v in history_data.values()):
        raise ValueError("Direct training did not produce finite training and validation histories")
    write_json(run_dir / "history.json", history_data)
    best_epoch = int(np.argmin(history_data["val_loss"])) + 1
    metadata = {
        "method": "direct", "config": asdict(config),
        "config_id": f"direct_D{config.num_dims}_N{config.num_obs}_S{config.summary_dim}_{run_dir.name}",
        "model_order": list(MODEL_ORDER), "model_prior": list(MODEL_PRIOR),
        "training_model_prior": list(MODEL_PRIOR), "model_specs": MODEL_SPECS,
        "preprocessing": PREPROCESSING, "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint), "best_epoch": best_epoch,
        "best_validation_loss": history_data["val_loss"][best_epoch - 1],
        "checkpoint_selection": "minimum independent validation categorical cross-entropy",
        "loss": "categorical_crossentropy", "auxiliary_losses": [],
        "summary_trainable": True, "classifier_trainable": True,
        "training_seed": config.seed, "validation_seed": validation_seed,
        "training_size": len(training_hashes), "validation_size": len(validation_hashes),
        "training_bank_id": bank_metadata["bank_id"],
        "training_bank_data_sha256": bank_metadata["data_sha256"],
        "versions": versions(), "backend": keras.backend.backend(),
    }
    write_json(run_dir / "metadata.json", metadata)
    print(f"Saved {metadata['config_id']} at {run_dir.resolve()} (best validation epoch {best_epoch})")
    return load_approximator(config, output_dir=run_dir), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="Immutable test bank, used only to guard split independence")
    parser.add_argument("--output-dir", required=True, help="New run directory; existing paths are rejected")
    parser.add_argument("--data-dim", type=int, default=20, help="Raw observation dimension D")
    parser.add_argument("--num-obs", type=int, default=10, help="Observations per data set N")
    parser.add_argument("--summary-dim", type=int, choices=(20, 40, 80), default=20, help="Learned summary dimension S")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-batches", type=int, default=128, help="Fixed training-bank size is this value times batch size")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=2)
    args = parser.parse_args()
    config = TrainingConfig(
        num_dims=args.data_dim, num_obs=args.num_obs, summary_dim=args.summary_dim,
        epochs=args.epochs, batch_size=args.batch_size, num_batches=args.num_batches,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    train_approximator(
        config, bank=args.bank, output_dir=args.output_dir,
        validation_size=args.validation_size, validation_seed=args.validation_seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
