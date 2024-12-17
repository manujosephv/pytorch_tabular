import numpy as np
import pytest
import torch
from sklearn.preprocessing import PowerTransformer

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.autoint import AutoIntConfig
from pytorch_tabular.models.category_embedding import CategoryEmbeddingModelConfig
from pytorch_tabular.models.danet import DANetConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig
from pytorch_tabular.models.gandalf import GANDALFConfig
from pytorch_tabular.models.gate import GatedAdditiveTreeEnsembleConfig
from pytorch_tabular.models.node import NodeConfig
from pytorch_tabular.models.stacking import StackingModelConfig
from pytorch_tabular.models.tabnet import TabNetModelConfig


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


def get_model_configs(task):
    all_model_configs = [
        lambda task: CategoryEmbeddingModelConfig(
            task=task,
        ),
        lambda task: TabNetModelConfig(
            task=task,
        ),
        lambda task: FTTransformerConfig(
            task=task,
        ),
        lambda task: GatedAdditiveTreeEnsembleConfig(
            task=task,
        ),
        lambda task: DANetConfig(
            task=task,
        ),
        lambda task: AutoIntConfig(
            task=task,
        ),
        lambda task: GANDALFConfig(
            task=task,
        ),
        lambda task: NodeConfig(
            task=task,
        ),
    ]
    return [model_config(task) for model_config in all_model_configs]


@pytest.mark.parametrize("multi_target", [True, False])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        [],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None, "yeo-johnson"])
@pytest.mark.parametrize("normalize_continuous_features", [True, False])
@pytest.mark.parametrize("target_range", [True, False])
@pytest.mark.parametrize(
    "target_transform",
    [None, PowerTransformer(), (lambda x: np.power(x, 2), lambda x: np.sqrt(x))],
)
@pytest.mark.parametrize("virtual_bz", [None, 32])
# @pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
# @pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad])
@pytest.mark.parametrize(
    "custom_args", [(None, None, None, None), ([fake_metric], [False], torch.nn.L1Loss(), torch.optim.Adagrad)]
)
@pytest.mark.parametrize("custom_head_config", [None, "", "32", "32-32"])
@pytest.mark.parametrize("model_configs", [get_model_configs("regression")])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
    target_transform,
    virtual_bz,
    # custom_metrics,
    # custom_loss,
    # custom_optimizer,
    custom_args,
    custom_head_config,
    model_configs,
):
    (train, test, target) = regression_data
    (custom_metrics, custom_metrics_prob_input, custom_loss, custom_optimizer) = custom_args
    if len(continuous_cols) + len(categorical_cols) == 0:
        return

    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
    )
    model_config_params = {"task": "regression", "virtual_batch_size": virtual_bz}

    if target_range:
        _target_range = []
        for target in data_config.target:
            _target_range.append(
                (
                    float(train[target].min()),
                    float(train[target].max()),
                )
            )
        model_config_params["target_range"] = _target_range
    if custom_head_config is not None:
        model_config_params["head"] = "LinearHead"
        model_config_params["head_config"] = {"layers": custom_head_config}

    model_config_params["model_configs"] = model_configs
    model_config = StackingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(
        train=train,
        metrics=custom_metrics,
        metrics_prob_inputs=custom_metrics_prob_input,
        target_transform=target_transform,
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )

    result = tabular_model.evaluate(test)
    # print(result[0]["valid_loss"])
    if custom_metrics is None:
        assert "test_mean_squared_error" in result[0].keys()
    else:
        assert "test_fake_metric" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
        [],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("model_configs", [get_model_configs("classification")])
def test_classification(
    classification_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    model_configs,
):
    (train, test, target) = classification_data
    if len(continuous_cols) + len(categorical_cols) == 0:
        return

    data_config = DataConfig(
        target=target + ["feature_53"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
    )
    model_config_params = {"task": "classification"}

    model_config_params["model_configs"] = model_configs
    model_config = StackingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    result = tabular_model.evaluate(test)
    # print(result[0]["valid_loss"])
    assert "test_accuracy" in result[0].keys()
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
