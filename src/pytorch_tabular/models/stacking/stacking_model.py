import inspect

import torch
import torch.nn as nn
from omegaconf import DictConfig

import pytorch_tabular.models as models
from pytorch_tabular.models import BaseModel
from pytorch_tabular.models.common.heads import blocks
from pytorch_tabular.models.gate import GatedAdditiveTreesBackbone
from pytorch_tabular.models.node import NODEBackbone


def instantiate_backbone(hparams, backbone_name):
    backbone_class = getattr(getattr(models, hparams._module_src.split(".")[-1]), backbone_name)
    class_args = list(inspect.signature(backbone_class).parameters.keys())
    if "config" in class_args:
        return backbone_class(config=hparams)
    else:
        return backbone_class(
            **{
                arg: getattr(hparams, arg) if arg != "block_activation" else getattr(nn, getattr(hparams, arg))()
                for arg in class_args
            }
        )


class StackingEmbeddingLayer(nn.Module):
    def __init__(self, embedding_layers: nn.ModuleList):
        super().__init__()
        self.embedding_layers = embedding_layers

    def forward(self, x):
        outputs = []
        for embedding_layer in self.embedding_layers:
            em_output = embedding_layer(x)
            outputs.append(em_output)
        return outputs


class StackingBackbone(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.hparams = config
        self._build_network()

    def _build_network(self):
        self._backbones = nn.ModuleList()
        self._heads = nn.ModuleList()
        self._backbone_output_dims = []
        assert len(self.hparams.model_configs) > 0, "Stacking requires more than 0 model"
        for model_i in range(len(self.hparams.model_configs)):
            # move necessary params to each model config
            self.hparams.model_configs[model_i].embedded_cat_dim = self.hparams.embedded_cat_dim
            self.hparams.model_configs[model_i].continuous_dim = self.hparams.continuous_dim
            self.hparams.model_configs[model_i].n_continuous_features = self.hparams.continuous_dim

            self.hparams.model_configs[model_i].embedding_dims = self.hparams.embedding_dims
            self.hparams.model_configs[model_i].categorical_cardinality = self.hparams.categorical_cardinality
            self.hparams.model_configs[model_i].categorical_dim = self.hparams.categorical_dim
            self.hparams.model_configs[model_i].cat_embedding_dims = self.hparams.embedding_dims

            # if output_dim is not set, set it to 128
            if getattr(self.hparams.model_configs[model_i], "output_dim", None) is None:
                self.hparams.model_configs[model_i].output_dim = 128

            # if inferred_config is not set, set it to None.
            if getattr(self.hparams, "inferred_config", None) is not None:
                self.hparams.model_configs[model_i].inferred_config = self.hparams.inferred_config

            # instantiate backbone
            _backbone = instantiate_backbone(
                self.hparams.model_configs[model_i], self.hparams.model_configs[model_i]._backbone_name
            )
            # set continuous_dim
            _backbone.continuous_dim = self.hparams.continuous_dim
            # if output_dim is not set, set it to the output_dim in model_config
            if getattr(_backbone, "output_dim", None) is None:
                setattr(
                    _backbone,
                    "output_dim",
                    self.hparams.model_configs[model_i].output_dim,
                )
            self._backbones.append(_backbone)
            self._backbone_output_dims.append(_backbone.output_dim)

        self.output_dim = sum(self._backbone_output_dims)

    def _build_embedding_layer(self):
        assert getattr(self, "_backbones", None) is not None, "Backbones are not built"
        embedding_layers = nn.ModuleList()
        for backbone in self._backbones:
            if getattr(backbone, "_build_embedding_layer", None) is None:
                embedding_layers.append(nn.Identity())
            else:
                embedding_layers.append(backbone._build_embedding_layer())
        return StackingEmbeddingLayer(embedding_layers)

    def forward(self, x_list):
        outputs = []
        for i, backbone in enumerate(self._backbones):
            bb_output = backbone(x_list[i])
            if len(bb_output.shape) == 3 and isinstance(backbone, GatedAdditiveTreesBackbone):
                bb_output = bb_output.mean(dim=-1)
            elif len(bb_output.shape) == 3 and isinstance(backbone, NODEBackbone):
                bb_output = bb_output.mean(dim=1)
            outputs.append(bb_output)
        x = torch.cat(outputs, dim=1)
        return x


class StackingModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self._backbone = StackingBackbone(self.hparams)
        self._embedding_layer = self._backbone._build_embedding_layer()
        self.output_dim = self._backbone.output_dim
        self._head = self._get_head_from_config()

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.hparams.head)
        return _head_callable(
            in_units=self.output_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.hparams.head_config),
        )

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self._embedding_layer

    @property
    def head(self):
        return self._head
