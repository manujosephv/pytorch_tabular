#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""
import pytest
import torch
import os

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models import (
    AutoIntConfig,
    CategoryEmbeddingModelConfig,
    NodeConfig,
    TabNetModelConfig,
    TabTransformerConfig,
)

MODEL_CONFIG_SAVE_TEST = [
    (CategoryEmbeddingModelConfig, dict(layers="10-20")),
    (
        AutoIntConfig,
        dict(
            num_heads=1,
            num_attn_blocks=1,
        ),
    ),
    (NodeConfig, dict(num_trees=100, depth=2)),
    (TabNetModelConfig, dict(n_a=2, n_d=2)),
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
    model_config_class, model_config_params = model_config_class
    model_config_params["task"] = "regression"
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        gpus=None,
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
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )

    result_1 = tabular_model.evaluate(test)
    # sv_dir = tmpdir/"save_model"
    # sv_dir.mkdir(exist_ok=True, parents=True)
    sv_dir = tmpdir.mkdir("saved_model")
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
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        gpus=None,
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
    )
    dt = DeepFeatureExtractor(tabular_model)
    enc_df = dt.fit_transform(test)
    assert any([col for col in enc_df.columns if "backbone" in col])


MODEL_CONFIG_SAVE_TEST_SSL = [
    (TabTransformerConfig, dict(input_embed_dim=8, num_attn_blocks=1, num_heads=2)),
]


@pytest.mark.parametrize(
    "model_config_class",
    MODEL_CONFIG_SAVE_TEST_SSL,
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
@pytest.mark.parametrize("custom_metrics", [None])
@pytest.mark.parametrize("custom_loss", [None])
@pytest.mark.parametrize("custom_optimizer", [None])
def test_pretrained_backbone(
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

    model_config_class, model_config_params = model_config_class
    model_config_params["task"] = "ssl"
    model_config_params["ssl_task"] = "Denoising"
    model_config_params["aug_task"] = "cutmix"
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        gpus=None,
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
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )
    result_1 = tabular_model.evaluate(test)
    with pytest.raises(AssertionError):
        tabular_model.predict(test)
    assert "test_mean_squared_error" in result_1[0].keys()
    sv_dir = tmpdir.mkdir("saved_model")
    tabular_model.save_model(str(sv_dir))
    old_mdl = TabularModel.load_from_checkpoint(str(sv_dir))
    model_config_params["task"] = "regression"
    model_config_params["ssl_task"] = None
    model_config_params["aug_task"] = None
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        gpus=None,
        fast_dev_run=True,
    )
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
        optimizer_params={},
        trained_backbone=old_mdl.model.backbone,
    )
    result_2 = tabular_model.evaluate(test)
    assert "test_mean_squared_error" in result_2[0].keys()


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
@pytest.mark.parametrize("save_type", ["pytorch", "onnx"])
def test_save_for_inference(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    custom_metrics,
    custom_loss,
    custom_optimizer,
    save_type,
    tmpdir,
):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
    )
    model_config_class, model_config_params = model_config_class
    model_config_params["task"] = "regression"
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        gpus=None,
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
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )
    sv_dir = tmpdir.mkdir("saved_model")

    tabular_model.save_model_for_inference(
        sv_dir / "model.pt" if type == "pytorch" else sv_dir / "model.onnx",
        kind=save_type,
    )
    assert os.path.exists(
        sv_dir / "model.pt" if type == "pytorch" else sv_dir / "model.onnx"
    )
