#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""
import numpy as np
import pytest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.tabular_datamodule import TabularDatamodule


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
@pytest.mark.parametrize(
    "target_transform",
    [
        None,
        PowerTransformer(method="yeo-johnson"),
        (lambda x: x**2, lambda x: np.sqrt(x)),
    ],
)
@pytest.mark.parametrize("validation_split", [None, 0.3])
@pytest.mark.parametrize("embedding_dims", [None, [(5, 1)]])
def test_dataloader(
    regression_data,
    validation_split,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_transform,
    embedding_dims,
):
    (train, test, target) = regression_data
    train, valid = train_test_split(train, random_state=42)
    if len(continuous_cols) + len(categorical_cols) == 0:
        assert True
    else:
        data_config = DataConfig(
            target=target + ["MedInc"] if multi_target else target,
            continuous_cols=continuous_cols,
            categorical_cols=categorical_cols,
            continuous_feature_transform=continuous_feature_transform,
            normalize_continuous_features=normalize_continuous_features,
            validation_split=validation_split,
        )
        model_config_params = dict(task="regression", embedding_dims=embedding_dims)
        model_config = CategoryEmbeddingModelConfig(**model_config_params)
        trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
        optimizer_config = OptimizerConfig()

        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        config = tabular_model.config
        datamodule = TabularDatamodule(
            train=train,
            validation=valid,
            config=config,
            test=test,
            target_transform=target_transform,
        )
        datamodule.prepare_data()
        datamodule.setup("fit")
        inferred_config = datamodule.update_config(config)
        if len(categorical_cols) > 0:
            assert inferred_config.categorical_cardinality[0] == 5
            if embedding_dims is None:
                assert inferred_config.embedding_dims[0][-1] == 3
            else:
                assert inferred_config.embedding_dims[0][-1] == embedding_dims[0][-1]
        if normalize_continuous_features and len(continuous_cols) > 0:
            assert round(datamodule.train[config.continuous_cols[0]].mean()) == 0
            assert round(datamodule.train[config.continuous_cols[0]].std()) == 1
            # assert round(datamodule.validation[config.continuous_cols[0]].mean()) == 0
            # assert round(datamodule.validation[config.continuous_cols[0]].std()) == 1
        val_loader = datamodule.val_dataloader()
        _val_loader = datamodule.prepare_inference_dataloader(valid)
        chk_1 = next(iter(val_loader))["continuous"]
        chk_2 = next(iter(_val_loader))["continuous"]
        assert np.not_equal(chk_1, chk_2).sum().item() == 0


@pytest.mark.parametrize(
    "freq",
    ["H", "D", "T", "S"],
)
def test_date_encoding(timeseries_data, freq):
    (train, test, target) = timeseries_data
    train, valid = train_test_split(train, random_state=42)
    data_config = DataConfig(
        target=target + ["Occupancy"],
        continuous_cols=["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"],
        categorical_cols=[],
        date_columns=[("date", freq)],
        encode_date_columns=True,
    )
    model_config_params = dict(task="regression")
    model_config = CategoryEmbeddingModelConfig(**model_config_params)
    trainer_config = TrainerConfig(max_epochs=1, checkpoints=None, early_stopping=None)
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    config = tabular_model.config
    datamodule = TabularDatamodule(
        train=train,
        validation=valid,
        config=config,
        test=test,
    )
    datamodule.prepare_data()
    if freq != "S":
        datamodule.setup("fit")
        config = datamodule.config
        if freq == "H":
            assert "_Hour" in datamodule.train.columns
        elif freq == "D":
            assert "_Dayofyear" in datamodule.train.columns
        elif freq == "T":
            assert "_Minute" in datamodule.train.columns
    elif freq == "S":
        try:
            datamodule.setup("fit")
            assert False
        except RuntimeError:
            assert True
