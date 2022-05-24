from torch import nn

from pytorch_tabular.utils import _initialize_layers, _linear_dropout_bn
from pytorch_tabular.models.common.heads import config as head_config


def config_link(r):
    """
    This is a helper function decorator to link the config to the head.
    """

    def wrapper(f):
        f.config_template = r
        return f

    return wrapper


class Head(nn.Module):
    def __init__(self, layers, config_template, **kwargs):
        super().__init__()
        self.layers = layers
        self._config_template = config_template

    def forward(self, x):
        return self.layers(x)


class LinearHead(Head):
    _config_template = head_config.LinearHeadConfig

    def __init__(self, in_units: int, output_dim: int, config, **kwargs):
        # Linear Layers
        _layers = []
        _curr_units = in_units
        for units in config.layers.split("-"):
            try:
                int(units)
            except ValueError:
                if units == "":
                    continue
                else:
                    raise ValueError(f"Invalid units {units} in layers {config.layers}")
            _layers.extend(
                _linear_dropout_bn(
                    config.activation,
                    config.initialization,
                    config.use_batch_norm,
                    _curr_units,
                    int(units),
                    config.dropout,
                )
            )
            _curr_units = int(units)
        # Appending Final Output
        _layers.append(nn.Linear(_curr_units, output_dim))
        linear_layers = nn.Sequential(*_layers)
        _initialize_layers(
            config.activation, config.initialization, linear_layers
        )
        super().__init__(
            layers=linear_layers,
            config_template=head_config.LinearHeadConfig,
        )


# @config_link(head_config.LinearHeadConfig)
# def linear_head(in_units: int, config: DictConfig):
#     # Linear Layers
#     _layers = []
#     _curr_units = in_units
#     for units in config.layers.split("-"):
#         _layers.extend(
#             _linear_dropout_bn(
#                 config.activation,
#                 config.initialization,
#                 config.use_batch_norm,
#                 _curr_units,
#                 int(units),
#                 config.dropout,
#             )
#         )
#         _curr_units = int(units)
#     linear_layers = nn.Sequential(*_layers)
#     return Head(
#         layers=linear_layers,
#         output_dim=_curr_units,
#         config_template=head_config.LinearHeadConfig,
#     )
