"""Read historical Gaussian NPEs without changing their trained attention layers.

BayesFlow 2.0.14 changed PMA residuals and attention projections. These small
inference-only layers preserve the 2.0.12 computation; unchanged layers are reused.
"""

from io import BytesIO
from zipfile import ZipFile

import h5py
import keras
import bayesflow as bf
from bayesflow.networks.summary.summary_network import SummaryNetwork
from bayesflow.networks.summary.deep_set.equivariant_layer import EquivariantLayer
from bayesflow.networks.summary.transformers.attention.multihead_attention import (
    MultiHeadAttention,
)
from bayesflow.networks.helpers import FFN


class LegacyAttention(MultiHeadAttention):
    def __init__(self, **kwargs):
        super().__init__(gate_attention=False, gate_ffn=False, **kwargs)
        self.input_projector = keras.layers.Dense(
            self.embed_dim,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
        )

    def build(self, x_shape, y_shape):
        if self.built:
            return
        self.input_projector.build(x_shape)
        projected = self.input_projector.compute_output_shape(x_shape)
        if self.ln_attn is not None:
            self.ln_attn.build(projected)
        if self.ln_kv is not None:
            self.ln_kv.build(y_shape)
        self.attention.build(projected, y_shape, y_shape)
        if self.ln_ffn is not None:
            self.ln_ffn.build(projected)
        self.feedforward.build(projected)


class LegacyPooling(keras.Layer):
    def __init__(self, num_seeds, embed_dim, seed_dim, **kwargs):
        super().__init__()
        self.num_seeds, self.embed_dim, self.seed_dim = num_seeds, embed_dim, seed_dim
        self.mab = LegacyAttention(embed_dim=embed_dim, **kwargs)
        self.seed_vector = self.add_weight(
            shape=(num_seeds, seed_dim or embed_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.feedforward = FFN(
            embed_dim=embed_dim,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in ("num_heads", "layer_norm")
            },
        )

    def build(self, input_shape):
        if self.built:
            return
        self.feedforward.build(input_shape)
        transformed = self.feedforward.compute_output_shape(input_shape)
        self.mab.build(
            (input_shape[0], self.num_seeds, self.seed_dim or self.embed_dim),
            transformed,
        )

    def call(self, x, training=False, **kwargs):
        transformed = self.feedforward(x, training=training)
        seeds = keras.ops.tile(
            keras.ops.expand_dims(self.seed_vector, 0), [keras.ops.shape(x)[0], 1, 1]
        )
        summaries = self.mab(seeds, transformed, training=training)
        return keras.ops.reshape(summaries, (keras.ops.shape(x)[0], -1))


class DeepSet(bf.networks.DeepSet):
    def __init__(
        self,
        summary_dim=16,
        embed_dim=64,
        depth=2,
        mlp_widths=(64,),
        inner_pooling="mean",
        num_heads=4,
        num_seeds=4,
        seed_dim=None,
        expansion_factor=4.0,
        glu_variant="swiglu",
        use_bias=False,
        layer_norm=True,
        activation="silu",
        kernel_initializer="he_normal",
        dropout=0.05,
        **kwargs,
    ):
        SummaryNetwork.__init__(self, **kwargs)
        self.summary_dim = summary_dim
        self.equivariant_modules = [
            EquivariantLayer(
                embed_dim=embed_dim,
                mlp_widths=mlp_widths,
                pooling=inner_pooling,
                activation=activation,
                kernel_initializer=kernel_initializer,
                dropout=dropout,
                layer_norm=layer_norm,
            )
            for _ in range(depth)
        ]
        self.pooling_by_attention = LegacyPooling(
            num_seeds=num_seeds,
            embed_dim=mlp_widths[-1],
            seed_dim=seed_dim,
            num_heads=num_heads,
            dropout=dropout,
            expansion_factor=expansion_factor,
            glu_variant=glu_variant,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.output_projector = keras.layers.Dense(summary_dim)


def load_checkpoint(path):
    """Load old or current DeepSet archives for sequential Gaussian inference."""
    with ZipFile(path) as archive:
        with h5py.File(BytesIO(archive.read("model.weights.h5")), "r") as weights:
            legacy = len(weights["layers/deep_set/pooling_by_attention/vars"]) == 1
    if not legacy:
        return keras.models.load_model(path, compile=False)

    # Keras nested deserialization gives the global registry precedence over scope.
    key = "bayesflow.networks>DeepSet"
    registry = keras.saving.get_custom_objects()
    previous = registry[key]
    try:
        registry[key] = DeepSet
        with keras.saving.custom_object_scope({key: DeepSet}):
            return keras.models.load_model(path, compile=False)
    finally:
        registry[key] = previous
