from dataclasses import dataclass, field

from pytorch_tabular.config import ModelConfig


@dataclass
class StackingModelConfig(ModelConfig):
    """StackingModelConfig is a configuration class for the StackingModel. It is used to stack multiple models
    together. Now, CategoryEmbeddingModel, TabNetModel, FTTransformerModel, GatedAdditiveTreeEnsembleModel, DANetModel,
    AutoIntModel, GANDALFModel, NodeModel are supported.

    Args:
        model_configs (list[ModelConfig]): List of model configs to stack.

    """

    model_configs: list = field(default_factory=list, metadata={"help": "List of model configs to stack"})
    _module_src: str = field(default="models.stacking")
    _model_name: str = field(default="StackingModel")
    _backbone_name: str = field(default="StackingBackbone")
    _config_name: str = field(default="StackingConfig")


# if __name__ == "__main__":
#    from pytorch_tabular.utils import generate_doc_dataclass
#    print(generate_doc_dataclass(StackingModelConfig))
