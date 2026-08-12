import argparse
from pathlib import Path

import numpy as np

from .approximator import build_workflow
from .config import DATA_DIR, TRAINING_DIR
from .plotting import plot_training_loss


def load_dataset(path: Path, limit: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        data = {"theta": archive["theta"], "image": archive["image"]}
    if limit is not None:
        data = {name: values[:limit] for name, values in data.items()}
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--posterior-samples", type=int, default=1000)
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--max-validation", type=int)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    train_data = load_dataset(args.data_dir / "train.npz", args.max_train)
    validation_data = load_dataset(
        args.data_dir / "validation.npz",
        args.max_validation,
    )
    test_data = load_dataset(args.data_dir / "test.npz")

    workflow = build_workflow()
    history = workflow.fit_offline(
        data=train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    workflow.approximator.save(TRAINING_DIR / "strong_lens_npe.keras")
    plot_training_loss(history.history, TRAINING_DIR / "loss.png")

    samples = workflow.sample(
        num_samples=args.posterior_samples,
        conditions={"image": test_data["image"][:1]},
        seed=4,
    )
    posterior = np.asarray(samples["theta"])
    if posterior.ndim == 3:
        posterior = posterior[0]
    true_theta = test_data["theta"][0]
    np.savez_compressed(
        TRAINING_DIR / "posterior_smoke_test.npz",
        samples=posterior,
        true_theta=true_theta,
    )

    print("True theta:", np.array2string(true_theta, precision=4))
    print("Posterior sample shape:", posterior.shape)
    print("Posterior mean:", np.array2string(posterior.mean(axis=0), precision=4))


if __name__ == "__main__":
    main()
