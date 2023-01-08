#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import pytest

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import MDNConfig


@pytest.mark.parametrize("multi_target", [False])
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
        ]
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("variant", ["CategoryEmbeddingModelConfig", "TabTransformerConfig", "FTTransformerConfig"])
@pytest.mark.parametrize("num_gaussian", [1, 2])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    variant,
    num_gaussian,
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
        mdn_config = dict(num_gaussian=num_gaussian)
        model_config_params["head_config"] = mdn_config
        model_config_params["backbone_config_class"] = variant
        model_config_params["backbone_config_params"] = dict(task="backbone")

        model_config = MDNConfig(**model_config_params)
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
        [],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("num_gaussian", [1, 2])
def test_classification(
    classification_data,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    num_gaussian,
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
        mdn_config = dict(num_gaussian=num_gaussian)
        model_config_params["head_config"] = mdn_config
        model_config_params["backbone_config_class"] = "CategoryEmbeddingMDNConfig"
        model_config_params["backbone_config_params"] = dict(task="backbone")

        model_config = MDNConfig(**model_config_params)
        trainer_config = TrainerConfig(
            max_epochs=3,
            checkpoints=None,
            early_stopping=None,
            accelerator="cpu",
            fast_dev_run=True,
        )
        optimizer_config = OptimizerConfig()
        with pytest.raises(AssertionError):
            tabular_model = TabularModel(
                data_config=data_config,
                model_config=model_config,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
            )
            tabular_model.fit(train=train, test=test)
