#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import pytest

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import AutoIntConfig


@pytest.mark.parametrize("multi_target", [True, False])
@pytest.mark.parametrize(
    "continuous_cols", [["AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]]
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("target_range", [True, False])
@pytest.mark.parametrize("deep_layers", [True, False])
@pytest.mark.parametrize("batch_norm_continuous_input", [True, False])
@pytest.mark.parametrize("attention_pooling", [True, False])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
    deep_layers,
    batch_norm_continuous_input,
    attention_pooling,
):
    (train, test, target) = regression_data
    if len(continuous_cols) + len(categorical_cols) == 0:
        assert True
    else:
        data_config = DataConfig(
            target=target + ["MedInc"] if multi_target else target,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            continuous_feature_transform=continuous_feature_transform,
            normalize_continuous_features=normalize_continuous_features,
        )
        model_config_params = dict(task="regression")
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
        model_config_params["deep_layers"] = deep_layers
        model_config_params["batch_norm_continuous_input"] = batch_norm_continuous_input
        model_config_params["attention_pooling"] = attention_pooling
        model_config = AutoIntConfig(**model_config_params)
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
        tabular_model.fit(train=train, test=test)

        result = tabular_model.evaluate(test)
        # print(result[0]["valid_loss"])
        assert "test_mean_squared_error" in result[0].keys()
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("deep_layers", [True, False])
@pytest.mark.parametrize("batch_norm_continuous_input", [True, False])
def test_classification(
    classification_data,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    deep_layers,
    batch_norm_continuous_input,
):
    (train, test, target) = classification_data
    if len(continuous_cols) + len(categorical_cols) == 0:
        assert True
    else:
        data_config = DataConfig(
            target=target,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            continuous_feature_transform=continuous_feature_transform,
            normalize_continuous_features=normalize_continuous_features,
        )
        model_config_params = dict(task="classification")
        model_config_params["deep_layers"] = deep_layers
        model_config_params["batch_norm_continuous_input"] = batch_norm_continuous_input
        model_config = AutoIntConfig(**model_config_params)
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
        tabular_model.fit(train=train, test=test)

        result = tabular_model.evaluate(test)
        # print(result[0]["valid_loss"])
        assert "test_accuracy" in result[0].keys()
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


# @pytest.mark.parametrize(
#     "continuous_cols",
#     [
#         [
#             "AveRooms",
#             "AveBedrms",
#             "Population",
#             "AveOccup",
#             "Latitude",
#             "Longitude",
#         ],
#     ],
# )
# @pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
# @pytest.mark.parametrize("continuous_feature_transform", [None])
# @pytest.mark.parametrize("normalize_continuous_features", [True])
# @pytest.mark.parametrize("deep_layers", [True])
# @pytest.mark.parametrize("batch_norm_continuous_input", [True])
# @pytest.mark.parametrize("attention_pooling", [True])
# @pytest.mark.parametrize("ssl_task", ["Denoising", "Contrastive"])
# @pytest.mark.parametrize("aug_task", ["cutmix", "mixup"])
# def test_ssl(
#     regression_data,
#     continuous_cols,
#     categorical_cols,
#     continuous_feature_transform,
#     normalize_continuous_features,
#     deep_layers,
#     batch_norm_continuous_input,
#     attention_pooling,
#     ssl_task,
#     aug_task,
# ):
#     (train, test, target) = regression_data
#     if len(continuous_cols) + len(categorical_cols) == 0:
#         assert True
#     else:
#         data_config = DataConfig(
#             target=target,
#             continuous_cols=continuous_cols,
#             categorical_cols=categorical_cols,
#             continuous_feature_transform=continuous_feature_transform,
#             normalize_continuous_features=normalize_continuous_features,
#         )
#         model_config_params = dict(task="ssl", ssl_task=ssl_task, aug_task=aug_task)
#         model_config_params["deep_layers"] = deep_layers
#         model_config_params["batch_norm_continuous_input"] = batch_norm_continuous_input
#         model_config_params["attention_pooling"] = attention_pooling
#         model_config = AutoIntConfig(**model_config_params)
#         trainer_config = TrainerConfig(
#             max_epochs=3,
#             checkpoints=None,
#             early_stopping=None,
#             fast_dev_run=True,
#         )
#         optimizer_config = OptimizerConfig()

#         tabular_model = TabularModel(
#             data_config=data_config,
#             model_config=model_config,
#             optimizer_config=optimizer_config,
#             trainer_config=trainer_config,
#         )
#         tabular_model.fit(train=train, test=test)

#         result = tabular_model.evaluate(test)
#         assert "test_mean_squared_error" in result[0].keys()
