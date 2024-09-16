#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_covtype

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models.node import NodeConfig
from pytorch_tabular.tabular_model import TabularModel


def regression_data():
    dataset = fetch_california_housing(data_home="data", as_frame=True)
    df = dataset.frame.sample(5000)
    df["HouseAgeBin"] = pd.qcut(df["HouseAge"], q=4)
    df["HouseAgeBin"] = "age_" + df.HouseAgeBin.cat.codes.astype(str)
    test_idx = df.sample(int(0.2 * len(df)), random_state=42).index
    test = df[df.index.isin(test_idx)]
    train = df[~df.index.isin(test_idx)]
    return (train, test, dataset.target_names)


def classification_data():
    dataset = fetch_covtype(data_home="data")
    data = np.hstack([dataset.data, dataset.target.reshape(-1, 1)])[:10000, :]
    col_names = [f"feature_{i}" for i in range(data.shape[-1])]
    col_names[-1] = "target"
    data = pd.DataFrame(data, columns=col_names)
    data["feature_0_cat"] = pd.qcut(data["feature_0"], q=4)
    data["feature_0_cat"] = "feature_0_" + data.feature_0_cat.cat.codes.astype(str)
    test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"])


def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
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
        model_config_params = {"task": "regression", "depth": 2}
        model_config = NodeConfig(**model_config_params)
        # model_config_params = dict(task="regression")
        # model_config = NodeConfig(**model_config_params)

        trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
        optimizer_config = OptimizerConfig()

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train)

        result = tabular_model.evaluate(test)
        if multi_target:
            assert result[0]["valid_loss"] < 30
        else:
            assert result[0]["valid_loss"] < 8
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


def test_classification(
    classification_data,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
):
    (train, test, target) = classification_data
    if len(continuous_cols) + len(categorical_cols) == 0:
        return
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
    )
    model_config_params = {"task": "classification", "depth": 2}
    model_config = NodeConfig(**model_config_params)
    trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train)

    result = tabular_model.evaluate(test)
    assert result[0]["valid_loss"] < 2.5
    pred_df = tabular_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]


test_regression(
    regression_data(),
    multi_target=False,
    continuous_cols=[
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
    categorical_cols=["HouseAgeBin"],
    continuous_feature_transform=None,
    normalize_continuous_features=True,
    # target_range=True,
)

# classification_data()
