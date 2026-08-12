from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parameter_title(theta: np.ndarray) -> str:
    theta_E, e1, e2, gamma1, gamma2, x_s, y_s, R_s = theta
    return (
        rf"$\theta_E={theta_E:.2f}$, $e=({e1:.2f},{e2:.2f})$"
        "\n"
        rf"$\gamma=({gamma1:.2f},{gamma2:.2f})$, "
        rf"$s=({x_s:.2f},{y_s:.2f})$, $R_s={R_s:.2f}$"
    )


def plot_simulation_grid(
    theta: np.ndarray,
    images: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for axis, parameters, image in zip(axes.flat, theta, images):
        axis.imshow(image[..., 0], origin="lower", cmap="magma")
        axis.set_title(parameter_title(parameters), fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_noiseless_comparison(
    theta: np.ndarray,
    noiseless: np.ndarray,
    noisy: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, len(theta), figsize=(11, 7))
    for column, parameters in enumerate(theta):
        axes[0, column].imshow(noiseless[column], origin="lower", cmap="magma")
        axes[0, column].set_title(parameter_title(parameters), fontsize=8)
        axes[1, column].imshow(noisy[column, ..., 0], origin="lower", cmap="magma")
        axes[0, column].set_ylabel("Noiseless")
        axes[1, column].set_ylabel("Noisy")
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_training_loss(history: Mapping[str, list], output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(history["loss"], marker="o", markersize=3, label="Training")
    if "val_loss" in history:
        axis.plot(history["val_loss"], marker="o", markersize=3, label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
