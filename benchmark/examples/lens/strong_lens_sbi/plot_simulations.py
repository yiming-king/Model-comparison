import argparse
from pathlib import Path

import numpy as np

from .config import DATA_DIR, FIGURE_DIR
from .plotting import plot_noiseless_comparison, plot_simulation_grid
from .simulator import StrongLensSimulator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "test.npz")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    with np.load(args.dataset) as archive:
        theta = archive["theta"]
        images = archive["image"]

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(theta), size=9, replace=False)
    plot_simulation_grid(
        theta[indices],
        images[indices],
        FIGURE_DIR / "simulations.png",
    )

    comparison_indices = indices[:3]
    simulator = StrongLensSimulator()
    noiseless = np.stack(
        [simulator.simulate_noiseless(parameters) for parameters in theta[comparison_indices]]
    )
    plot_noiseless_comparison(
        theta[comparison_indices],
        noiseless,
        images[comparison_indices],
        FIGURE_DIR / "noiseless_vs_noisy.png",
    )
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
