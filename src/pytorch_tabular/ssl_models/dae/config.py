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

        loss_type_weights (Optional[List[float]]): Weights to be used for the loss function in the order
                [binary, categorical, numerical]. If None, will use the default weights using a formula. eg. for
                binary, default weight will be n_binary/n_features. Defaults to None

        mask_loss_weight (float): Weight to be used for the loss function for the masked features. Defaults
                to 1.0

        max_onehot_cardinality (int): Maximum cardinality of one-hot encoded categorical features. Any
                categorical feature with cardinality>max_onehot_cardinality will be embedded in a learned
                embedding space and others will be converted to a one hot representation. If set to 0, will use
                the embedding strategy for all categorical feature. Default is 4


        encoder_config (Optional[pytorch_tabular.config.config.ModelConfig]): The config of the encoder to
                be used for the model. Should be one of the model configs defined in PyTorch Tabular

        decoder_config (Optional[pytorch_tabular.config.config.ModelConfig]): The config of decoder to be
                used for the model. Should be one of the model configs defined in PyTorch Tabular. Defaults to
                nn.Identity

        embedding_dims (Optional[List]): The dimensions of the embedding for each categorical column as a
                list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of
                the categorical column using the rule min(50, (x + 1) // 2)

        embedding_dropout (float): Dropout to be applied to the Categorical Embedding. Defaults to 0.1

        batch_norm_continuous_input (bool): If True, we will normalize the continuous layer by passing it
                through a BatchNorm layer. DEPRECATED - Use head and head_config instead

        learning_rate (float): The learning rate of the model. Defaults to 1e-3

        seed (int): The seed for reproducibility. Defaults to 42
    """

    noise_strategy: str = field(
        default="swap",
        metadata={
            "help": "Defines what kind of noise we are introducing to samples. `swap` - Swap noise is when we replace values of a feature with random permutations of the same feature. `zero` - Zero noise is when we replace values of a feature with zeros. Defaults to swap",
            "choices": ["swap", "zero"],
        },
    )
    # Union not supported by omegaconf. Currently Union[float, Dict[str, float]]
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
        metadata={"help": "Weight to be used for the loss function for the masked features. Defaults to 1.0"},
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
        assert hasattr(self.encoder_config, "_backbone_name"), "encoder_config should have a _backbone_name attribute"
        if self.decoder_config is not None:
            assert hasattr(
                self.decoder_config, "_backbone_name"
            ), "decoder_config should have a _backbone_name attribute"
        super().__post_init__()


# if __name__ == "__main__":
#     from pytorch_tabular.utils import generate_doc_dataclass
#     print(generate_doc_dataclass(DenoisingAutoEncoderConfig))
