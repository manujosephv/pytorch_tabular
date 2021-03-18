#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import pytest
import torch

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models import (
    AutoIntConfig,
    CategoryEmbeddingModelConfig,
    NodeConfig,
    TabNetModelConfig,
)

MODEL_CONFIG_SAVE_TEST = [
    CategoryEmbeddingModelConfig,
    AutoIntConfig,
    TabNetModelConfig,
]

MODEL_CONFIG_FEATURE_EXT_TEST = [
    CategoryEmbeddingModelConfig,
    AutoIntConfig,
]


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


@pytest.mark.parametrize(
    "model_config_class",
    MODEL_CONFIG_SAVE_TEST,
)
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
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("custom_metrics", [None, [fake_metric]])
@pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
@pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad])
def test_save_load(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    custom_metrics,
    custom_loss,
    custom_optimizer,
    tmpdir,
):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    model_config_params = dict(task="regression")
    model_config = model_config_class(**model_config_params)
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
    tabular_model.fit(
        train=train,
        test=test,
        metrics=custom_metrics,
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params=None if custom_optimizer is None else {},
    )

    result_1 = tabular_model.evaluate(test)
    sv_dir = tmpdir.mkdir("save_model")
    tabular_model.save_model(str(sv_dir))
    new_mdl = TabularModel.load_from_checkpoint(str(sv_dir))
    result_2 = new_mdl.evaluate(test)
    assert (
        result_1[0][f"test_{tabular_model.model.hparams.metrics[0]}"]
        == result_2[0][f"test_{new_mdl.model.hparams.metrics[0]}"]
    )


@pytest.mark.parametrize(
    "model_config_class",
    MODEL_CONFIG_FEATURE_EXT_TEST,
)
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
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
def test_feature_extractor(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    model_config_params = dict(task="regression")
    model_config = model_config_class(**model_config_params)
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
    tabular_model.fit(
        train=train,
        test=test,
    )
    dt = DeepFeatureExtractor(tabular_model)
    enc_df = dt.fit_transform(test)
    assert any([col for col in enc_df.columns if "backbone" in col])


# import numpy as np
# import pandas as pd
# from sklearn.datasets import fetch_california_housing, fetch_covtype
# from pathlib import Path

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


# test_save_load(
#     regression_data(),
#     model_config_class=CategoryEmbeddingModelConfig,
#     continuous_cols=[
#         "AveRooms",
#         "AveBedrms",
#         "Population",
#         "AveOccup",
#         "Latitude",
#         "Longitude",
#     ],
#     categorical_cols=[],
#     custom_metrics = None, #[fake_metric],
#     custom_loss = None, custom_optimizer = None,
#     tmpdir = Path("tmp")
# )
# test_embedding_transformer(regression_data())

# classification_data()
