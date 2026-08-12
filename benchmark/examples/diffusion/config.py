from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")


BASE_DIR = Path(__file__).resolve().parent
APPROXIMATOR_DIR = BASE_DIR / "approximators" / "trained" / "indirect"
HISTORY_DIR = BASE_DIR / "approximators" / "history" / "indirect"
LOG_DIR = BASE_DIR / "approximators" / "logs" / "indirect"
RESULT_DIR = BASE_DIR / "results"
FIGURE_DIR = RESULT_DIR / "plots"

MODELS = ("m0", "m1", "m2", "m3")
MODEL_TITLES = {
    "m0": "M0: common threshold",
    "m1": "M1: threshold by condition",
    "m2": "M2: threshold by accumulator",
    "m3": "M3: condition x accumulator",
}
N_ALPHA = {"m0": 1, "m1": 2, "m2": 2, "m3": 4}
PARAM_DIMS = {model: n_alpha + 3 for model, n_alpha in N_ALPHA.items()}


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 64
    num_batches: int = 128

    summary_dim: int = 30
    summary_multiplier: int | None = None
    summary_base_distribution: str | None = "normal"
    run_suffix: str | None = None

    def summary_dim_for(self, model: str) -> int:
        if self.summary_multiplier is None:
            return self.summary_dim
        return PARAM_DIMS[model] * self.summary_multiplier

    @property
    def summary_label(self) -> str:
        if self.summary_multiplier is not None:
            label = f"S{self.summary_multiplier}D"
        else:
            label = f"S{self.summary_dim}"
        if self.run_suffix:
            label = f"{label}_{self.run_suffix}"
        return label


def get_name(model: str, approximation: str = "NPE", summary_label: str = "S30") -> str:
    return f"{model.upper()}_{approximation}_{summary_label}"


def get_path(name: str) -> Path:
    return APPROXIMATOR_DIR / f"{name}.keras"


def get_history_path(name: str) -> Path:
    return HISTORY_DIR / f"{name}_history.json"


def ensure_dirs() -> None:
    for path in (APPROXIMATOR_DIR, HISTORY_DIR, LOG_DIR, RESULT_DIR, FIGURE_DIR):
        path.mkdir(parents=True, exist_ok=True)
