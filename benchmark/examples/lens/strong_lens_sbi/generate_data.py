import argparse
import time
from pathlib import Path

import numpy as np

from .config import (
    DATA_DIR,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    TEST_SEED,
    TRAIN_SEED,
    VALIDATION_SEED,
)
from .simulator import StrongLensSimulator


def save_dataset(
    simulator: StrongLensSimulator,
    name: str,
    num_samples: int,
    seed: int,
    output_dir: Path,
) -> None:
    start = time.perf_counter()
    dataset = simulator.sample_dataset(num_samples, seed)
    np.savez_compressed(output_dir / f"{name}.npz", **dataset)
    elapsed = time.perf_counter() - start
    print(
        f"{name}: theta {dataset['theta'].shape}, image {dataset['image'].shape}, "
        f"{elapsed:.1f} s"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-validation", type=int, default=N_VALIDATION)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    simulator = StrongLensSimulator()
    save_dataset(simulator, "train", args.n_train, TRAIN_SEED, args.output_dir)
    save_dataset(
        simulator,
        "validation",
        args.n_validation,
        VALIDATION_SEED,
        args.output_dir,
    )
    save_dataset(simulator, "test", args.n_test, TEST_SEED, args.output_dir)


if __name__ == "__main__":
    main()
