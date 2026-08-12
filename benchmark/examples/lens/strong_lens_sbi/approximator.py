import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import bayesflow as bf

from .config import CHECKPOINT_DIR


def build_workflow(checkpoint_dir: Path = CHECKPOINT_DIR) -> bf.BasicWorkflow:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_network = bf.networks.ConvolutionalNetwork(summary_dim=24)
    inference_network = bf.networks.CouplingFlow()
    return bf.BasicWorkflow(
        simulator=None,
        inference_variables=["theta"],
        summary_variables=["image"],
        inference_network=inference_network,
        summary_network=summary_network,
        checkpoint_filepath=str(checkpoint_dir),
        checkpoint_name="strong_lens_npe",
        save_best_only=True,
    )
