#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import PowerTransformer

from pytorch_tabular import TabularModel
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


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
# @pytest.mark.parametrize("custom_metrics", [None, [fake_metric]])
# @pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
# @pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad])
@pytest.mark.parametrize("custom_args", [(None, None, None), ([fake_metric], torch.nn.L1Loss(), torch.optim.Adagrad)])
@pytest.mark.parametrize("custom_head_config", [None, "", "32", "32-32"])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
    target_transform,
    # custom_metrics,
    # custom_loss,
    # custom_optimizer,
    custom_args,
    custom_head_config,
):
    (train, test, target) = regression_data
    (custom_metrics, custom_loss, custom_optimizer) = custom_args
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
        if custom_head_config is not None:
            model_config_params["head"] = "LinearHead"
            model_config_params["head_config"] = dict(layers=custom_head_config)
        model_config = CategoryEmbeddingModelConfig(**model_config_params)
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
            test=test,
            metrics=custom_metrics,
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
def test_classification(
    classification_data,
    continuous_cols,
    categorical_cols,
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
        model_config_params = dict(task="classification")
        model_config = CategoryEmbeddingModelConfig(**model_config_params)
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


def test_embedding_transformer(regression_data):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=[
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        categorical_cols=["HouseAgeBin"],
    )
    model_config_params = dict(task="regression")
    model_config = CategoryEmbeddingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=1,
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

    transformer = CategoricalEmbeddingTransformer(tabular_model)
    train_transform = transformer.fit_transform(train)
    embed_cols = [col for col in train_transform.columns if "HouseAgeBin_embed_dim" in col]
    assert len(train["HouseAgeBin"].unique()) + 1 == len(transformer._mapping["HouseAgeBin"].keys())
    assert all([val.shape[0] == len(embed_cols) for val in transformer._mapping["HouseAgeBin"].values()])
