#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import pytest

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import MixtureDensityHeadConfig, CategoryEmbeddingMDNConfig, NODEMDNConfig, AutoIntMDNConfig
from pytorch_tabular import TabularModel
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer


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
@pytest.mark.parametrize("variant", [CategoryEmbeddingMDNConfig, NODEMDNConfig, AutoIntMDNConfig])
@pytest.mark.parametrize("num_gaussian", [1, 2])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    variant,
    num_gaussian
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
        mdn_config = MixtureDensityHeadConfig(num_gaussian=num_gaussian)
        model_config_params['mdn_config'] = mdn_config
        model_config = variant(**model_config_params)
        trainer_config = TrainerConfig(
            max_epochs=3, checkpoints=None, early_stopping=None, gpus=0, fast_dev_run=True
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
        assert "valid_loss" in result[0].keys()
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
    num_gaussian
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
        mdn_config = MixtureDensityHeadConfig(num_gaussian=num_gaussian)
        model_config_params['mdn_config'] = mdn_config
        model_config = CategoryEmbeddingMDNConfig(**model_config_params)
        trainer_config = TrainerConfig(
            max_epochs=3, checkpoints=None, early_stopping=None, gpus=0, fast_dev_run=True
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

# import numpy as np
# import pandas as pd
# from sklearn.datasets import fetch_california_housing, fetch_covtype

 
# def regression_data():
#     dataset = fetch_california_housing(data_home="data", as_frame=True)
#     df = dataset.frame.sample(5000)
#     df["HouseAgeBin"] = pd.qcut(df["HouseAge"], q=4)
#     df["HouseAgeBin"] = "age_" + df.HouseAgeBin.cat.codes.astype(str)
#     test_idx = df.sample(int(0.2 * len(df)), random_state=42).index
#     test = df[df.index.isin(test_idx)]
#     train = df[~df.index.isin(test_idx)]
#     return (train, test, dataset.target_names)


# def classification_data():
#     dataset = fetch_covtype(data_home="data")
#     data = np.hstack([dataset.data, dataset.target.reshape(-1, 1)])[:10000, :]
#     col_names = [f"feature_{i}" for i in range(data.shape[-1])]
#     col_names[-1] = "target"
#     data = pd.DataFrame(data, columns=col_names)
#     data["feature_0_cat"] = pd.qcut(data["feature_0"], q=4)
#     data["feature_0_cat"] = "feature_0_" + data.feature_0_cat.cat.codes.astype(str)
#     test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
#     test = data[data.index.isin(test_idx)]
#     train = data[~data.index.isin(test_idx)]
#     return (train, test, ["target"])


# test_regression(
#     regression_data(),
#     multi_target=True,
#     continuous_cols=[
#         "AveRooms",
#         "AveBedrms",
#         "Population",
#         "AveOccup",
#         "Latitude",
#         "Longitude",
#     ],
#     categorical_cols=[],
#     continuous_feature_transform="yeo-johnson",
#     normalize_continuous_features=False,
#     target_range=True,
# )
# test_embedding_transformer(regression_data())

# classification_data()
