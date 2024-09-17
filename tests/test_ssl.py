#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import pytest
import torch
from sklearn.model_selection import train_test_split

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.ssl_models.dae import DenoisingAutoEncoderConfig


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


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
        ],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("freeze_backbone", [True, False])
@pytest.mark.parametrize("target_range", [False])
@pytest.mark.parametrize(
    "target_transform",
    [None],
)
@pytest.mark.parametrize(
    "custom_args",
    [(None, None, None, None), ([fake_metric], [False], torch.nn.L1Loss(), torch.optim.Adagrad)],
)
@pytest.mark.parametrize("include_input_features_inference", [False, True])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    freeze_backbone,
    target_range,
    target_transform,
    custom_args,
    include_input_features_inference,
):
    (train, test, target) = regression_data
    (custom_metrics, metrics_prob_input, custom_loss, custom_optimizer) = custom_args
    ssl, finetune = train_test_split(train, random_state=42)
    ssl_train, ssl_val = train_test_split(ssl, random_state=42)
    finetune_train, finetune_val = train_test_split(finetune, random_state=42)
    data_config = DataConfig(
        target=target + ["MedInc"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
        handle_missing_values=False,
        handle_unknown_categories=False,
    )
    encoder_config = CategoryEmbeddingModelConfig(
        task="backbone",
        layers="32-16-8",  # Number of nodes in each layer
        activation="LeakyReLU",  # Activation between each layers
    )
    decoder_config = CategoryEmbeddingModelConfig(
        task="backbone",
        layers="8-16-32",  # Number of nodes in each layer
        activation="LeakyReLU",  # Activation between each layers
    )

    model_config_params = {
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
        "include_input_features_inference": include_input_features_inference,
    }
    model_config = DenoisingAutoEncoderConfig(**model_config_params)
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
    tabular_model.pretrain(train=ssl_train, validation=ssl_val)
    if target_range:
        _target_range = []
        for target in data_config.target:
            _target_range.append(
                (
                    float(train[target].min()),
                    float(train[target].max()),
                )
            )
    else:
        _target_range = None
    finetune_model = tabular_model.create_finetune_model(
        task="regression",
        train=finetune_train,
        validation=finetune_val,
        target_transform=target_transform,
        head="LinearHead",
        head_config={
            "layers": "16-8",
            "activation": "LeakyReLU",
        },
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        target_range=_target_range,
        loss=custom_loss,
        metrics=custom_metrics,
        metrics_params=[{}],
        metrics_prob_input=metrics_prob_input,
        optimizer=custom_optimizer,
    )
    assert torch.equal(
        tabular_model.model.encoder.linear_layers[0].weight,
        finetune_model.model._backbone.encoder.linear_layers[0].weight,
    )
    finetune_model.finetune(
        freeze_backbone=freeze_backbone,
    )
    result = finetune_model.evaluate(test)
    if custom_metrics is None:
        assert "test_mean_squared_error" in result[0].keys()
    else:
        assert "test_fake_metric" in result[0].keys()
    pred_df = finetune_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]


@pytest.mark.parametrize("multi_target", [False, True])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("freeze_backbone", [False])
def test_classification(
    classification_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    freeze_backbone,
):
    (train, test, target) = classification_data
    ssl, finetune = train_test_split(train, random_state=42)
    ssl_train, ssl_val = train_test_split(ssl, random_state=42)
    finetune_train, finetune_val = train_test_split(finetune, random_state=42)
    data_config = DataConfig(
        target=target + ["feature_53"] if multi_target else target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        continuous_feature_transform=continuous_feature_transform,
        normalize_continuous_features=normalize_continuous_features,
        handle_missing_values=False,
        handle_unknown_categories=False,
    )
    encoder_config = CategoryEmbeddingModelConfig(
        task="backbone",
        layers="32-16-8",  # Number of nodes in each layer
        activation="LeakyReLU",  # Activation between each layers
    )
    decoder_config = CategoryEmbeddingModelConfig(
        task="backbone",
        layers="8-16-32",  # Number of nodes in each layer
        activation="LeakyReLU",  # Activation between each layers
    )
    model_config_params = {
        "encoder_config": encoder_config,
        "decoder_config": decoder_config,
    }
    model_config = DenoisingAutoEncoderConfig(**model_config_params)
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
    tabular_model.pretrain(train=ssl_train, validation=ssl_val)
    finetune_model = tabular_model.create_finetune_model(
        task="classification",
        train=finetune_train,
        validation=finetune_val,
        head="LinearHead",
        head_config={
            "layers": "16-8",
            "activation": "LeakyReLU",
        },
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
    )
    assert torch.equal(
        tabular_model.model.encoder.linear_layers[0].weight,
        finetune_model.model._backbone.encoder.linear_layers[0].weight,
    )
    finetune_model.finetune(
        freeze_backbone=freeze_backbone,
    )
    result = finetune_model.evaluate(test)
    assert "test_accuracy" in result[0].keys()
    pred_df = finetune_model.predict(test)
    assert pred_df.shape[0] == test.shape[0]
