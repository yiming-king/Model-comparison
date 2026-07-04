import keras
from bayesflow.utils.serialization import serializable, serialize, deserialize
from bayesflow.utils import concatenate_valid_shapes, concatenate_valid
from bayesflow.types import Tensor
from bayesflow.networks import InferenceNetwork

@serializable("bayesflow.networks")
class ReshapeData(InferenceNetwork):
    """
    This class reshapes the data inputs of samples.shape = (batch_size, num_trials, 2)
    to samples.shape = (batch_size, num_trials) + conditions.shape = (batch_size, num_trials),
    conditioning the observed response times (samples) on the experimental condition (conditions) that are otherwise
    treated (by the posterior network) as 'data'.
    """
    def __init__(self, distribution, **kwargs):
        super().__init__(**kwargs)
        self.distribution = distribution

    def get_config(self):
        base_config = super().get_config()
        config = {
            "distribution": self.distribution
        }

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def build(self, xz_shape, conditions_shape=None):
        if not self.distribution.built:
            xz_shape = (xz_shape[0], xz_shape[1])
            self.distribution.build(
                xz_shape,
                concatenate_valid_shapes((conditions_shape, xz_shape), axis=-1)
            )

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        conditions = concatenate_valid(
            tensors=(conditions, samples[:,:,1]),
            axis=-1
        )
        samples = samples[:,:,0]

        return self.distribution.log_prob(samples, conditions, **kwargs)

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        conditions = concatenate_valid(
            tensors=(conditions, x[:,:,1]),
            axis=-1
        )
        samples = x[:,:,0]

        return self.distribution.compute_metrics(samples, conditions, stage=stage)