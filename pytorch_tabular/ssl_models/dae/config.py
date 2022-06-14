# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""DenoisingAutoEncoder Config"""
from dataclasses import dataclass, field
from typing import Dict, Union

from pytorch_tabular.config import SSLModelConfig


@dataclass
class DenoisingAutoEncoderConfig(SSLModelConfig):
    """DeNoising AutoEncoder configuration
    Args:

        embedding_dims (Union[List[int], NoneType]): The dimensions of the embedding for
            each categorical column as a list of tuples (cardinality, embedding_dim).
            If left empty, will infer using the cardinality of the categorical column using
            the rule min(50, (x + 1) // 2)
        learning_rate (float): The learning rate of the model

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    noise_strategy: str = field(
        default="swap",
        metadata={"help": "Defines what kind of noise we are introducing to samples. `swap` - Swap noise is when we replace values of a feature with random permutations of the same feature. `zero` - Zero noise is when we replace values of a feature with zeros. Defaults to swap",
        "choices": ["swap", "zero"]},
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
    max_onehot_cardinality: int = field(
        default=4,
        metadata={
            "help": "Maximum cardinality of one-hot encoded categorical features. Any categorical feature with cardinality>max_onehot_cardinality will be embedded in a learned embedding space and others will be converted to a one hot representation. If set to 0, will use the embedding strategy for all categorical feature. Default is 4"
        }
    )
    
    _module_src: str = field(default="dae")
    _model_name: str = field(default="DenoisingAutoEncoderModel")
    _config_name: str = field(default="DenoisingAutoEncoderConfig")

    def __post_init__(self):
        assert hasattr(self.encoder_config, "_backbone_name"), "encoder_config should have a _backbone_name attribute"
        assert hasattr(self.decoder_config, "_backbone_name"), "decoder_config should have a _backbone_name attribute"
        super().__post_init__()

# cls = TabTransformerConfig
# desc = "Configuration for Data."
# doc_str = f"{desc}\nArgs:"
# for key in cls.__dataclass_fields__.keys():
#     atr = cls.__dataclass_fields__[key]
#     if atr.init:
#         type = str(atr.type).replace("<class '","").replace("'>","").replace("typing.","")
#         help_str = atr.metadata.get("help","")
#         if "choices" in atr.metadata.keys():
#             help_str += f'. Choices are: [{",".join(["`"+str(ch)+"`" for ch in atr.metadata["choices"]])}].'
#         # help_str += f'. Defaults to {atr.default}'
#         doc_str+=f'\n\t\t{key} ({type}): {help_str}'

# print(doc_str)
