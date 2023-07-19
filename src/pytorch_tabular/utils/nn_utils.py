import torch
import torch.nn as nn

from .logger import get_logger

logger = get_logger(__name__)


def _initialize_layers(activation, initialization, layers):
    if type(layers) == nn.Sequential:
        for layer in layers:
            if hasattr(layer, "weight"):
                _initialize_layers(activation, initialization, layer)
    else:
        if activation == "ReLU":
            nonlinearity = "relu"
        elif activation == "LeakyReLU":
            nonlinearity = "leaky_relu"
        else:
            if initialization == "kaiming":
                logger.warning("Kaiming initialization is only recommended for ReLU and LeakyReLU.")
                nonlinearity = "leaky_relu"
            else:
                nonlinearity = "relu"

        if initialization == "kaiming":
            nn.init.kaiming_normal_(layers.weight, nonlinearity=nonlinearity)
        elif initialization == "xavier":
            nn.init.xavier_normal_(
                layers.weight,
                gain=nn.init.calculate_gain(nonlinearity) if activation in ["ReLU", "LeakyReLU"] else 1,
            )
        elif initialization == "random":
            nn.init.normal_(layers.weight)


def _linear_dropout_bn(activation, initialization, use_batch_norm, in_units, out_units, dropout):
    if isinstance(activation, str):
        _activation = getattr(nn, activation)
    else:
        _activation = activation
    layers = []
    if use_batch_norm:
        layers.append(nn.BatchNorm1d(num_features=in_units))
    linear = nn.Linear(in_units, out_units)
    _initialize_layers(activation, initialization, linear)
    layers.extend([linear, _activation()])
    if dropout != 0:
        layers.append(nn.Dropout(dropout))
    return layers


def reset_all_weights(model: nn.Module) -> None:
    """Resets all parameters in a network.

    Args:
        model: The model to reset the parameters of.

    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def to_one_hot(y, depth=None):
    r"""Takes integer with n dims and converts it to 1-hot representation with n + 1 dims.

    The n+1'st dimension will have zeros everywhere but at y'th index, where it will be equal to 1.
    Args:
        y: input integer (IntTensor, LongTensor or Variable) of any shape
        depth (int):  the size of the one hot dimension
    """
    y_flat = y.to(torch.int64).view(-1, 1)
    depth = depth if depth is not None else int(torch.max(y_flat)) + 1
    y_one_hot = torch.zeros(y_flat.size()[0], depth, device=y.device).scatter_(1, y_flat, 1)
    y_one_hot = y_one_hot.view(*(tuple(y.shape) + (-1,)))
    return y_one_hot


def _make_ix_like(input, dim=0):
    d = input.size(dim)
    rho = torch.arange(1, d + 1, device=input.device, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _initialize_kaiming(x, initialization, d_sqrt_inv):
    if initialization == "kaiming_uniform":
        nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
    elif initialization == "kaiming_normal":
        nn.init.normal_(x, std=d_sqrt_inv)
    elif initialization is None:
        pass
    else:
        raise NotImplementedError("initialization should be either of `kaiming_normal`, `kaiming_uniform`, `None`")
