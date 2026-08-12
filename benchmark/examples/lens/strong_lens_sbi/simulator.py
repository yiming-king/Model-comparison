import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Util import simulation_util

from .config import DEFAULT_CONFIG, PRIOR_HIGH, PRIOR_LOW, LensConfig


class StrongLensSimulator:
    def __init__(self, config: LensConfig = DEFAULT_CONFIG):
        self.config = config
        kwargs_data = simulation_util.data_configure_simple(
            num_pix=config.num_pix,
            delta_pix=config.pixel_scale,
            exposure_time=config.exposure_time,
            background_rms=config.background_rms,
        )
        self.data = ImageData(**kwargs_data)
        self.psf = PSF(
            psf_type="GAUSSIAN",
            fwhm=config.psf_fwhm,
            pixel_size=config.pixel_scale,
        )
        self.lens_model = LensModel(lens_model_list=["SIE", "SHEAR"])
        self.source_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
        self.image_model = ImageModel(
            data_class=self.data,
            psf_class=self.psf,
            lens_model_class=self.lens_model,
            source_model_class=self.source_model,
        )

    def sample_prior(
        self,
        num_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(num_samples, len(PRIOR_LOW))).astype(
            np.float32
        )
    # defult 8 parameters: theta_E, e1, e2, gamma1, gamma2, x_s, y_s, R_s
    def theta_to_kwargs(self, theta: np.ndarray) -> tuple[list[dict], list[dict]]:
        theta_E, e1, e2, gamma1, gamma2, x_s, y_s, R_s = np.asarray(theta)
        kwargs_lens = [
            {
                "theta_E": float(theta_E),
                "e1": float(e1),
                "e2": float(e2),
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
                "ra_0": 0.0,
                "dec_0": 0.0,
            },
        ]
        kwargs_source = [
            {
                "amp": self.config.source_amp,
                "R_sersic": float(R_s),
                "n_sersic": self.config.source_n_sersic,
                "e1": self.config.source_e1,
                "e2": self.config.source_e2,
                "center_x": float(x_s),
                "center_y": float(y_s),
            }
        ]
        return kwargs_lens, kwargs_source

    def simulate_noiseless(self, theta: np.ndarray) -> np.ndarray:
        "lensed background source image without noise"
        kwargs_lens, kwargs_source = self.theta_to_kwargs(theta)
        image = self.image_model.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            lens_light_add=False,
            point_source_add=False,
        )
        return np.asarray(image, dtype=np.float32)

    def add_noise(
        self,
        image: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        background = rng.normal(0.0, self.config.background_rms, image.shape)
        photon_sigma = np.sqrt(np.abs(image) / self.config.exposure_time)
        photon = rng.normal(0.0, photon_sigma)
        return np.asarray(image + background + photon, dtype=np.float32)

    def simulate(
        self,
        theta: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return self.add_noise(self.simulate_noiseless(theta), rng)

    def sample_dataset(self, num_samples: int, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        theta = self.sample_prior(num_samples, rng)
        images = np.empty(
            (num_samples, self.config.num_pix, self.config.num_pix, 1),
            dtype=np.float32,
        )
        for index, parameters in enumerate(theta):
            images[index, ..., 0] = self.simulate(parameters, rng)
        return {"theta": theta, "image": images} # theta(N, len(parameters)), image(N, num_pix, num_pix, 1)