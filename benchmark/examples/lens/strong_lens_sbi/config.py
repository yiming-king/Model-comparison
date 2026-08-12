from dataclasses import dataclass
from pathlib import Path

import numpy as np


PARAMETER_NAMES = (
    "theta_E",
    "e1",
    "e2",
    "gamma1",
    "gamma2",
    "x_s",
    "y_s",
    "R_s",
)

PRIOR_LOW = np.array(
    [0.7, -0.2, -0.2, -0.08, -0.08, -0.25, -0.25, 0.05],
    dtype=np.float32,
)
PRIOR_HIGH = np.array(
    [1.3, 0.2, 0.2, 0.08, 0.08, 0.25, 0.25, 0.25],
    dtype=np.float32,
)

TRAIN_SEED = 1
VALIDATION_SEED = 2
TEST_SEED = 3

N_TRAIN = 2000
N_VALIDATION = 400
N_TEST = 200

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"
FIGURE_DIR = PACKAGE_DIR / "outputs" / "figures"
TRAINING_DIR = PACKAGE_DIR / "outputs" / "training"
CHECKPOINT_DIR = PACKAGE_DIR / "checkpoints"


@dataclass(frozen=True)
class LensConfig:
    num_pix: int = 64
    pixel_scale: float = 0.05
    psf_fwhm: float = 0.08
    exposure_time: float = 1000.0
    background_rms: float = 0.03
    source_amp: float = 0.20
    source_n_sersic: float = 1.0
    source_e1: float = 0.10
    source_e2: float = -0.05


DEFAULT_CONFIG = LensConfig()
