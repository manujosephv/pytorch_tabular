# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""DenoisingAutoEncoder Config"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pytorch_tabular.config import SSLModelConfig


@dataclass
class DenoisingAutoEncoderConfig(SSLModelConfig):
    """DeNoising AutoEncoder configuration
    Args:
        noise_strategy (str): Defines what kind of noise we are introducing to samples. `swap` - Swap noise
                is when we replace values of a feature with random permutations of the same feature. `zero` - Zero
                noise is when we replace values of a feature with zeros. Defaults to swap. Choices are:
                [`swap`,`zero`].

        noise_probabilities (Dict[str, float]): Dict of individual probabilities to corrupt the input
                features with swap/zero noise. Key should be the feature name and if any feature is missing, the
                default_noise_probability is used. Default is an empty dict()

        default_noise_probability (float): Default probability to corrupt the input features with swap/zero
                noise. For features for which noise_probabilities does not define a probability. Default is 0.8

        loss_type_weights (Union[List[float], NoneType]): Weights to be used for the loss function in the
                order [binary, categorical, numerical]. If None, will use the default weights using a formula. eg.
                for binary, default weight will be n_binary/n_features. Defaults to None

        mask_loss_weight (float): Weight to be used for the loss function for the masked features. Defaults
                to 1.0

        max_onehot_cardinality (int): Maximum cardinality of one-hot encoded categorical features. Any
                categorical feature with cardinality>max_onehot_cardinality will be embedded in a learned
                embedding space and others will be converted to a one hot representation. If set to 0, will use
                the embedding strategy for all categorical feature. Default is 4

        embedding_dropout (float): probability of an embedding element to be zeroed.

        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it
                through a BatchNorm layer
    """

    noise_strategy: str = field(
        default="swap",
        metadata={
            "help": "Defines what kind of noise we are introducing to samples. `swap` - Swap noise is when we replace values of a feature with random permutations of the same feature. `zero` - Zero noise is when we replace values of a feature with zeros. Defaults to swap",
            "choices": ["swap", "zero"],
        },
    )
    # Union not supported currently Union[float, Dict[str, float]]
    noise_probabilities: Dict[str, float] = field(
        default_factory=lambda: dict(),
        metadata={
            "help": "Dict of individual probabilities to corrupt the input features with swap/zero noise. Key should be the feature name and if any feature is missing, the default_noise_probability is used. Default is an empty dict()"
        },
    )
    default_noise_probability: float = field(
        default=0.8,
        metadata={
            "help": "Default probability to corrupt the input features with swap/zero noise. For features for which noise_probabilities does not define a probability. Default is 0.8"
        },
    )
    loss_type_weights: Optional[List[float]] = field(
        default=None,
        metadata={
            "help": "Weights to be used for the loss function in the order [binary, categorical, numerical]. If None, will use the default weights using a formula. eg. for binary, default weight will be n_binary/n_features. Defaults to None"
        },
    )
    mask_loss_weight: float = field(
        default=2.0,
        metadata={
            "help": "Weight to be used for the loss function for the masked features. Defaults to 1.0"
        },
    )
    max_onehot_cardinality: int = field(
        default=4,
        metadata={
            "help": "Maximum cardinality of one-hot encoded categorical features. Any categorical feature with cardinality>max_onehot_cardinality will be embedded in a learned embedding space and others will be converted to a one hot representation. If set to 0, will use the embedding strategy for all categorical feature. Default is 4"
        },
    )

    _module_src: str = field(default="ssl_models.dae")
    _model_name: str = field(default="DenoisingAutoEncoderModel")
    _config_name: str = field(default="DenoisingAutoEncoderConfig")

    def __post_init__(self):
        assert hasattr(
            self.encoder_config, "_backbone_name"
        ), "encoder_config should have a _backbone_name attribute"
        if self.decoder_config is not None:
            assert hasattr(
                self.decoder_config, "_backbone_name"
            ), "decoder_config should have a _backbone_name attribute"
        super().__post_init__()


# if __name__ == "__main__":
#     from pytorch_tabular.utils import generate_doc_dataclass
#     print(generate_doc_dataclass(DenoisingAutoEncoderConfig))
