"""Train and load Gaussian NPEs without executing any notebooks.

Run from the repository root with
``python -m benchmark.examples.gaussian.approximators.indirect --help``.
Heavy ML imports are deferred so --help and --dry-run do not initialize a backend.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import ASSUMED_MODELS, MODEL_SPECS, NETWORK_DIR, read_network_set_metadata
from .config import (
    NOTEBOOK_PRESETS,
    TrainingConfig,
    model_path,
    positive_int,
    validate_model,
    validate_seed,
)
from .simulators import get_simulator


def _runtime():
    os.environ.setdefault("KERAS_BACKEND", "tensorflow")
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "benchmark-matplotlib")
    )
    import bayesflow as bf
    import keras

    return bf, keras


@dataclass
class SavedHistory:
    history: dict[str, list[float]]


def build_adapter():
    bf, _ = _runtime()
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .rename("mu", "inference_variables")
        .rename("x", "summary_variables")
    )


def build_workflow(model: str, config: TrainingConfig = TrainingConfig()):
    validate_model(model)
    bf, keras = _runtime()
    keras.utils.set_random_seed(config.seed)
    summary_kwargs = {"summary_dim": config.summary_dim}
    if config.summary_base_distribution is not None:
        summary_kwargs["base_distribution"] = config.summary_base_distribution
    learning_rate = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.epochs_for(model) * config.num_batches,
    )
    return bf.BasicWorkflow(
        simulator=get_simulator(model, config),
        adapter=build_adapter(),
        summary_network=bf.networks.DeepSet(**summary_kwargs),
        inference_network=bf.networks.CouplingFlow(),
        standardize="all",
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    )


def generate_validation_data(
    model: str,
    config: TrainingConfig,
    validation_size: int,
    validation_seed: int = 2026,
):
    """Use a separate simulator so validation never advances the training RNG."""
    positive_int("validation_size", validation_size)
    validate_seed("validation_seed", validation_seed)
    _runtime()
    return get_simulator(model, config, seed=validation_seed).sample(validation_size)


def _check_dimensions(path: Path, config: TrainingConfig) -> None:
    actual = read_network_set_metadata(path)
    expected = (config.num_dims, config.num_obs, config.summary_dim)
    if actual != expected:
        raise ValueError(
            f"Saved model {path} has (data_dim, num_obs, summary_dim)={actual}, "
            f"requested {expected}. Use another output directory or run suffix."
        )


def load_approximator(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    *,
    output_dir: str | Path = NETWORK_DIR,
):
    path = model_path(model, config, output_dir)
    if not path.is_file():
        raise FileNotFoundError(f"Trained approximator not found: {path}")
    _check_dimensions(path, config)
    _, keras = _runtime()
    return keras.saving.load_model(path)


def load_history(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    *,
    output_dir: str | Path = NETWORK_DIR,
) -> SavedHistory:
    path = model_path(model, config, output_dir).with_suffix(".history.json")
    with path.open() as stream:
        return SavedHistory(json.load(stream))


def train_approximator(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    *,
    output_dir: str | Path = NETWORK_DIR,
    overwrite: bool = False,
    validation_size: int | None = None,
    validation_freq: int = 1,
    validation_seed: int = 2026,
    verbose: int = 2,
):
    """Fit one NPE, or load an existing one; overwrite means retrain from scratch."""
    path = model_path(model, config, output_dir)
    positive_int("validation_freq", validation_freq)
    validate_seed("validation_seed", validation_seed)
    if validation_size is not None:
        positive_int("validation_size", validation_size)
    if path.exists() and not overwrite:
        history = (
            load_history(model, config, output_dir=output_dir)
            if path.with_suffix(".history.json").exists()
            else None
        )
        return load_approximator(model, config, output_dir=output_dir), history

    workflow = build_workflow(model, config)
    fit_kwargs = {}
    if validation_size is not None:
        fit_kwargs["validation_data"] = generate_validation_data(
            model, config, validation_size, validation_seed
        )
        fit_kwargs["validation_freq"] = validation_freq
    history = workflow.fit_online(
        epochs=config.epochs_for(model),
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
        verbose=verbose,
        **fit_kwargs,
    )

    bf, keras = _runtime()
    metadata = {
        "model": model,
        "network_tag": config.network_tag,
        "training_config": {**asdict(config), "epochs": config.epochs_for(model)},
        "model_spec": MODEL_SPECS[model],
        "validation_size": validation_size,
        "validation_freq": validation_freq,
        "validation_seed": validation_seed,
        "optimizer": "Adam with CosineDecay",
        "flow": "default",
        "summary_network": workflow.approximator.summary_network.get_config(),
        "inference_network": workflow.approximator.inference_network.get_config(),
        "backend": keras.backend.backend(),
        "bayesflow_version": bf.__version__,
        "keras_version": keras.__version__,
    }
    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Stage the complete archive before replacing an existing trained model.
    with tempfile.TemporaryDirectory(
        prefix=".gaussian-training-", dir=path.parent
    ) as staging:
        staged_path = Path(staging) / path.name
        workflow.approximator.save(staged_path)
        staged_path.with_suffix(".history.json").write_text(
            json.dumps(serializable_history, indent=2) + "\n"
        )
        staged_path.with_suffix(".config.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        for suffix in (".history.json", ".config.json", ".keras"):
            staged_path.with_suffix(suffix).replace(path.with_suffix(suffix))
    return workflow.approximator, history


def train_approximators(
    config: TrainingConfig = TrainingConfig(),
    models=ASSUMED_MODELS,
    **kwargs,
):
    """Train a selected model set; use train_grid for isolated worker processes."""
    return {
        model: train_approximator(model, config, **kwargs)
        for model in dict.fromkeys(models)
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=NOTEBOOK_PRESETS, default="20d_10n")
    parser.add_argument(
        "--models", nargs="+", choices=ASSUMED_MODELS, default=list(ASSUMED_MODELS)
    )
    parser.add_argument(
        "--num-dims",
        type=int,
        help="Raw data and parameter dimension; not summary dimension.",
    )
    parser.add_argument("--num-obs", type=int)
    parser.add_argument("--summary-dim", type=int)
    parser.add_argument(
        "--epochs", type=int, help="Epochs for every model (default: 100)."
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-batches", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
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
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--validation-freq", type=int, default=1)
    parser.add_argument("--validation-seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved jobs without training or writing files.",
    )
    args = parser.parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in (
            "num_dims",
            "num_obs",
            "summary_dim",
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
    try:
        config = TrainingConfig.from_preset(args.preset, **overrides)
        positive_int("validation_freq", args.validation_freq)
        validate_seed("validation_seed", args.validation_seed)
        if args.validation_size is not None:
            positive_int("validation_size", args.validation_size)
    except ValueError as error:
        parser.error(str(error))

    for model in dict.fromkeys(args.models):
        path = model_path(model, config, args.output_dir).resolve()
        skip = path.exists() and not args.overwrite
        print(
            f"{'SKIP existing' if skip else 'TRAIN'} {model}: "
            f"D={config.num_dims}, n={config.num_obs}, S={config.summary_dim}, "
            f"epochs={config.epochs_for(model)}, batch_size={config.batch_size}, "
            f"num_batches={config.num_batches}, learning_rate={config.learning_rate:g}, "
            f"networks=bayesflow_default, summary_mmd={config.summary_base_distribution is not None} -> {path}",
            flush=True,
        )
        if args.dry_run:
            continue
        if skip:
            _check_dimensions(path, config)
            continue
        train_approximator(
            model,
            config,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
            validation_size=args.validation_size,
            validation_freq=args.validation_freq,
            validation_seed=args.validation_seed,
        )
        print(f"Saved {path}", flush=True)
        _, keras = _runtime()
        keras.backend.clear_session()


if __name__ == "__main__":
    main()
