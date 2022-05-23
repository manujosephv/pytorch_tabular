from dataclasses import dataclass, field
# from typing import Any, Dict, Iterable, List, Optional


@dataclass
class LinearHeadConfig:
    """A model class for Linear Head configuration; serves as a template and documentation. The models take a dictionary as input, but if there are keys which are not present in this model calss, it'll throw an exception.
    Args:
        layers (str): Hyphen-separated number of layers and units in the classification head. eg. 32-64-32.
        batch_norm_continuous_input (bool): If True, we will normalize the contiinuous layer by passing it through a BatchNorm layer
        activation (str): The activation type in the classification head.
            The default activation in PyTorch like ReLU, TanH, LeakyReLU, etc.
            https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
        dropout (float): probability of an classification element to be zeroed.
        use_batch_norm (bool): Flag to include a BatchNorm layer after each Linear Layer+DropOut
        initialization (str): Initialization scheme for the linear layers. Choices are: `kaiming` `xavier` `random`

    Raises:
        NotImplementedError: Raises an error if task is not in ['regression','classification']
    """

    layers: str = field(
        default="32",
        metadata={
            "help": "Hyphen-separated number of layers and units in the classification/regression head. eg. 32-64-32."
        },
    )
    activation: str = field(
        default="ReLU",
        metadata={
            "help": "The activation type in the classification head. The default activaion in PyTorch like ReLU, TanH, LeakyReLU, etc. https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity"
        },
    )
    dropout: float = field(
        default=0.5,
        metadata={"help": "probability of an classification element to be zeroed."},
    )
    use_batch_norm: bool = field(
        default=False,
        metadata={
            "help": "Flag to include a BatchNorm layer after each Linear Layer+DropOut"
        },
    )
    initialization: str = field(
        default="kaiming",
        metadata={
            "help": "Initialization scheme for the linear layers",
            "choices": ["kaiming", "xavier", "random"],
        },
    )
