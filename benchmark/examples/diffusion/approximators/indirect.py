from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import bayesflow as bf
import keras

from ..config import MODELS, TrainingConfig, ensure_dirs, get_history_path, get_name, get_path
from ..networks import PosteriorNetwork, SummaryNetwork
from ..simulators import SIMULATORS


models = list(MODELS)


class SavedHistory:
    def __init__(self, history: dict[str, list[float]]):
        self.history = history


def build_adapter() -> bf.adapters.Adapter:
    return (
        bf.adapters.Adapter()
        .convert_dtype("float64", "float32")
        .as_set(["rt", "conditions"])
        .concatenate(["rt", "conditions"], into="summary_variables") #(batch_size, n_trials, 2)
        .concatenate(["alpha", "nu", "tau"], into="inference_variables") #(batch_size, num_parameters)
    )

def get_simulator(model: str):
    return SIMULATORS[model]


def build_workflow(model: str, config: TrainingConfig = TrainingConfig()) -> bf.BasicWorkflow:
    return bf.BasicWorkflow(
        simulator=get_simulator(model),
        adapter=build_adapter(),
        inference_network=PosteriorNetwork(),
        summary_network=SummaryNetwork(summary_dim=config.summary_dim_for(model),
                                       base_distribution=config.summary_base_distribution),
        standardize="all",
    )

def load_approximator(model: str, config: TrainingConfig = TrainingConfig(), approximation: str = "NPE"):
    name = get_name(model, approximation, config.summary_label)
    path = get_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Trained approximator not found: {path}")
    return keras.saving.load_model(path)


def save_history(history, model: str, config: TrainingConfig = TrainingConfig(), approximation: str = "NPE"):
    name = get_name(model, approximation, config.summary_label)
    path = get_history_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }
    with path.open("w") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_history(model: str, config: TrainingConfig = TrainingConfig(), approximation: str = "NPE"):
    name = get_name(model, approximation, config.summary_label)
    path = get_history_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with path.open("r") as f:
        return SavedHistory(json.load(f))


def train_approximator(
    model: str,
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
):
    ensure_dirs()
    name = get_name(model, "NPE", config.summary_label)
    path = get_path(name)
    if path.exists() and not overwrite:
        history = load_history(model, config=config) if get_history_path(name).exists() else None
        return load_approximator(model, config=config), history

    workflow = build_workflow(model, config=config)

    history = workflow.fit_online(
        epochs=config.epochs,
        batch_size=config.batch_size,
        num_batches_per_epoch=config.num_batches,
    )
    workflow.approximator.save(path)
    save_history(history, model, config=config)
    return workflow.approximator, history


def train_approximators(
    config: TrainingConfig = TrainingConfig(),
    overwrite: bool = False,
):
    return {
        model: train_approximator(model, config=config, overwrite=overwrite)
        for model in models
    }


def get_approximators(load: bool = True, config: TrainingConfig = TrainingConfig()):
    output = {}
    for model in models:
        if load:
            approximator = load_approximator(model, config=config)
            workflow = None
        else:
            workflow = build_workflow(model, config=config)
            approximator = workflow.approximator
        output[model] = {
            "workflow": workflow,
            "approximator": approximator,
            "config": config,
        }
    return output


if __name__ == "__main__":
    default_config = TrainingConfig()
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=default_config.epochs)
    parser.add_argument("--batch-size", type=int, default=default_config.batch_size)
    parser.add_argument("--num-batches", type=int, default=default_config.num_batches)
    parser.add_argument("--summary-dim", type=int, default=default_config.summary_dim)
    parser.add_argument("--summary-multiplier", type=int, default=default_config.summary_multiplier)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-summary-mmd", action="store_true")
    parser.add_argument("--run-suffix", type=str, default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        summary_dim=args.summary_dim,
        summary_multiplier=args.summary_multiplier,
        summary_base_distribution=(
            None if args.no_summary_mmd else "normal"
        ),
        run_suffix=args.run_suffix,
    )
    train_approximators(config=config, overwrite=args.overwrite)


# cd /Users/yimingzang/Documents/Project/benchmark2

# KERAS_BACKEND=tensorflow \
# MPLCONFIGDIR=/private/tmp/matplotlib \
# /opt/anaconda3/envs/benchmark2/bin/python \
# -m benchmark.examples.diffusion.approximators.indirect \
# --summary-multiplier 4 \
# --epochs 100 \
# --batch-size 64 \
# --num-batches 128 \
# --no-summary-mmd \
# --run-suffix noMMD