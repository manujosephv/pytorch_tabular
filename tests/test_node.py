#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""
import pytest

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import NodeConfig
from pytorch_tabular import TabularModel


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
@pytest.mark.parametrize("embed_categorical", [True, False])
@pytest.mark.parametrize("continuous_feature_transform", [None, "yeo-johnson"])
@pytest.mark.parametrize("normalize_continuous_features", [True, False])
@pytest.mark.parametrize("target_range", [True, False])
def test_regression(
    regression_data,
    multi_target,
    embed_categorical,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range
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
        model_config_params = dict(task="regression", depth=2, embed_categorical=embed_categorical)
        if target_range:
            model_config_params["target_range"] = [
                train[target].min().item(),
                train[target].max().item(),
            ]
        model_config = NodeConfig(**model_config_params)
        trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
        optimizer_config = OptimizerConfig()

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train, test=test)

        result = tabular_model.evaluate(test)
        if multi_target:
            assert result[0]["valid_loss"] < 30
        else:
            assert result[0]["valid_loss"] < 8
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
        [],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None, "yeo-johnson"])
@pytest.mark.parametrize("embed_categorical", [True, False])
@pytest.mark.parametrize("normalize_continuous_features", [True, False])
def test_classification(
    classification_data,
    continuous_cols,
    categorical_cols,
    embed_categorical,
    continuous_feature_transform,
    normalize_continuous_features,
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
        model_config_params = dict(task="classification", depth=2, embed_categorical=embed_categorical)
        model_config = NodeConfig(**model_config_params)
        trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
        optimizer_config = OptimizerConfig()

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train, test=test)

        result = tabular_model.evaluate(test)
        assert result[0]["valid_loss"] < 2.5
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


# test_regression(
#     multi_target=False,
#     continuous_cols=[
#         "AveRooms",
#         "AveBedrms",
#         "Population",
#         "AveOccup",
#         "Latitude",
#         "Longitude",
#     ],
#     categorical_cols=["HouseAgeBin"],
#     continuous_feature_transform="yeo-johnson",
#     normalize_continuous_features=True,
#     target_range=True,
# )

# classification_data()
