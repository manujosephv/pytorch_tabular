#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""

import copy
import os

import numpy as np
import pytest
import torch
from scipy.stats import uniform
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold

from pytorch_tabular import TabularModel, TabularModelTuner, model_sweep
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.config.config import SSLModelConfig
from pytorch_tabular.feature_extractor import DeepFeatureExtractor
from pytorch_tabular.models import (
    AutoIntConfig,
    CategoryEmbeddingModelConfig,
    FTTransformerConfig,
    GANDALFConfig,
    GatedAdditiveTreeEnsembleConfig,
    NodeConfig,
    TabNetModelConfig,
)
from pytorch_tabular.ssl_models import DenoisingAutoEncoderConfig

# import os


MODEL_CONFIG_SAVE_TEST = [
    (CategoryEmbeddingModelConfig, {"layers": "10-20"}),
    (GANDALFConfig, {}),
    (NodeConfig, {"num_trees": 100, "depth": 2}),
    (TabNetModelConfig, {"n_a": 2, "n_d": 2}),
]

MODEL_CONFIG_SAVE_ONNX_TEST = [
    (CategoryEmbeddingModelConfig, {"layers": "10-20"}),
    (
        AutoIntConfig,
        {
            "num_heads": 1,
            "num_attn_blocks": 1,
        },
    ),
]
MODEL_CONFIG_FEATURE_EXT_TEST = [
    CategoryEmbeddingModelConfig,
    AutoIntConfig,
    DenoisingAutoEncoderConfig,
]

MODEL_CONFIG_FEATURE_IMP_TEST = [
    (FTTransformerConfig, {"num_heads": 1, "num_attn_blocks": 1}),
    (GANDALFConfig, {}),
    (
        GatedAdditiveTreeEnsembleConfig,
        {"num_trees": 1, "tree_depth": 2, "gflu_stages": 1},
    ),
]

MODEL_CONFIG_CAPTUM_TEST = [
    (FTTransformerConfig, {"num_heads": 1, "num_attn_blocks": 1}),
    (GANDALFConfig, {}),
    (TabNetModelConfig, {}),
]

MODEL_CONFIG_MODEL_SWEEP_TEST = [
    (FTTransformerConfig, {"num_heads": 1, "num_attn_blocks": 1}),
    (GANDALFConfig, {}),
    (TabNetModelConfig, {}),
]

DATASET_CONTINUOUS_COLUMNS = (
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
)


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_SAVE_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("custom_metrics", [None, [fake_metric]])
@pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
@pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad, "SGD", "torch_optimizer.AdaBound"])
@pytest.mark.parametrize("cache_data", ["memory", "disk"])
@pytest.mark.parametrize("inference_only", [True, False])
def test_save_load(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    custom_metrics,
    custom_loss,
    custom_optimizer,
    cache_data,
    inference_only,
    tmp_path_factory,
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
    if cache_data and cache_data == "disk":
        cache_data = str(tmp_path_factory.mktemp("cache"))
    tabular_model.fit(
        train=train,
        metrics=custom_metrics,
        metrics_prob_inputs=None if custom_metrics is None else [False],
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
        cache_data=cache_data,
    )

    result_1 = tabular_model.evaluate(test)
    # sv_dir = tmpdir/"save_model"
    # sv_dir.mkdir(exist_ok=True, parents=True)
    sv_dir = tmp_path_factory.mktemp("saved_model")
    tabular_model.save_model(str(sv_dir), inference_only=inference_only)
    new_mdl = TabularModel.load_model(str(sv_dir))
    result_2 = new_mdl.evaluate(test)
    assert (
        result_1[0][f"test_{tabular_model.model.hparams.metrics[0]}"]
        == result_2[0][f"test_{new_mdl.model.hparams.metrics[0]}"]
    )


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_FEATURE_EXT_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
def test_feature_extractor(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
):
    (train, test, target) = regression_data
    is_ssl = issubclass(model_config_class, SSLModelConfig)
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=False if is_ssl else True,
        handle_unknown_categories=False if is_ssl else True,
    )
    model_config_params = {}
    if not is_ssl:
        model_config_params["task"] = "regression"
    else:
        encoder_config = CategoryEmbeddingModelConfig(
            task="backbone",
            layers="4096-2048-512",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
        )
        decoder_config = CategoryEmbeddingModelConfig(
            task="backbone",
            layers="512-2048-4096",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
        )
        model_config_params["encoder_config"] = encoder_config
        model_config_params["decoder_config"] = decoder_config
    model_config = model_config_class(**model_config_params)
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
    if is_ssl:
        tabular_model.pretrain(
            train=train,
            validation=test,
        )
    else:
        tabular_model.fit(
            train=train,
            validation=test,
        )
    dt = DeepFeatureExtractor(tabular_model)
    enc_df = dt.fit_transform(test)
    assert any(col for col in enc_df.columns if "backbone" in col)


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_FEATURE_IMP_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
def test_feature_importance(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
):
    (train, test, target) = regression_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    model_config = model_config_class(**model_config_params)
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
        validation=test,
    )
    feat_imp = tabular_model.feature_importance()
    assert len(feat_imp) == len(continuous_cols + categorical_cols)


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_SAVE_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("custom_metrics", [None, [fake_metric]])
@pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
@pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad, "SGD", "torch_optimizer.AdaBound"])
def test_save_load_statedict(
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
        metrics=custom_metrics,
        metrics_prob_inputs=None if custom_metrics is None else [False],
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )

    result_1 = tabular_model.evaluate(test)
    # sv_dir = tmpdir/"save_model"
    # sv_dir.mkdir(exist_ok=True, parents=True)
    sv_dir = tmpdir.mkdir("saved_model")
    tabular_model.save_weights(str(sv_dir / "weights.pt"))
    new_mdl = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        model_state_dict_path=str(sv_dir / "weights.pt"),
    )
    datamodule = new_mdl.prepare_dataloader(train)
    model = new_mdl.prepare_model(
        datamodule,
        metrics=custom_metrics,
        metrics_prob_inputs=None if custom_metrics is None else [False],
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )
    new_mdl._prepare_for_training(model, datamodule)
    result_2 = new_mdl.evaluate(test)
    assert (
        result_1[0][f"test_{tabular_model.model.hparams.metrics[0]}"]
        == result_2[0][f"test_{new_mdl.model.hparams.metrics[0]}"]
    )


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
@pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad, "SGD", "torch_optimizer.AdaBound"])
@pytest.mark.parametrize("save_type", ["pytorch"])  # "onnx"
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
        metrics=custom_metrics,
        metrics_prob_inputs=None if custom_metrics is None else [False],
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )
    sv_dir = tmpdir.mkdir("saved_model")

    model_name = "model.pt" if save_type == "pytorch" else "model.onnx"
    tabular_model.save_model_for_inference(
        sv_dir / model_name,
        kind=save_type,
    )
    assert os.path.exists(sv_dir / model_name)


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_FEATURE_EXT_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
def test_model_reset(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
):
    (train, test, target) = regression_data
    is_ssl = issubclass(model_config_class, SSLModelConfig)
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=False if is_ssl else True,
        handle_unknown_categories=False if is_ssl else True,
    )
    model_config_params = {}
    if not is_ssl:
        model_config_params["task"] = "regression"
    else:
        encoder_config = CategoryEmbeddingModelConfig(
            task="backbone",
            layers="4096-2048-512",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
        )
        decoder_config = CategoryEmbeddingModelConfig(
            task="backbone",
            layers="512-2048-4096",  # Number of nodes in each layer
            activation="LeakyReLU",  # Activation between each layers
        )
        model_config_params["encoder_config"] = encoder_config
        model_config_params["decoder_config"] = decoder_config
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=2,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=False,
    )
    optimizer_config = OptimizerConfig()

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    if is_ssl:
        tabular_model.pretrain(
            train=train,
            validation=test,
        )
    else:
        tabular_model.fit(
            train=train,
            validation=test,
        )
    result_1 = tabular_model.evaluate(test)
    tabular_model.model.reset_weights()
    result_2 = tabular_model.evaluate(test)
    assert result_1[0]["test_loss"] != result_2[0]["test_loss"]


def _test_captum(
    model_config_class,
    model_config_params,
    data_config,
    train,
    test,
    attr_method,
    single_row,
    baselines,
):
    model_config = model_config_class(**model_config_params)
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
        validation=test,
    )
    if single_row:
        test = test.head(1)
    else:
        test = test.head(10)

    is_full_baselines = attr_method in ["GradientShap", "DeepLiftShap"]
    is_not_supported = tabular_model.model._get_name() in [
        "TabNetModel",
        "MDNModel",
        "TabTransformerModel",
    ]
    if is_full_baselines and (baselines is None or isinstance(baselines, (float, int))):
        with pytest.raises(ValueError):
            exp = tabular_model.explain(test, method=attr_method, baselines=baselines)
        return
    elif is_not_supported:
        with pytest.raises((NotImplementedError, AssertionError)):
            exp = tabular_model.explain(test, method=attr_method, baselines=baselines)
        return
    elif attr_method in ["FeaturePermutation", "FeatureAblation"] and single_row:
        with pytest.raises(AssertionError):
            exp = tabular_model.explain(test, method=attr_method, baselines=baselines)
        return
    else:
        exp = tabular_model.explain(test, method=attr_method, baselines=baselines)
    assert exp.shape[1] == tabular_model.model.hparams.continuous_dim + tabular_model.model.hparams.categorical_dim


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_CAPTUM_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
@pytest.mark.parametrize(
    "attr_method",
    [
        "GradientShap",
        "IntegratedGradients",
        "DeepLift",
        "DeepLiftShap",
        "InputXGradient",
        "FeaturePermutation",
        "FeatureAblation",
        "KernelShap",
    ],
)
@pytest.mark.parametrize("single_row", [True, False])
@pytest.mark.parametrize("baselines", ["b|100", None, 0])
def test_captum_integration_regression(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    attr_method,
    single_row,
    baselines,
):
    (train, test, target) = regression_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    _test_captum(
        model_config_class,
        model_config_params,
        data_config,
        train,
        test,
        attr_method,
        single_row,
        baselines,
    )


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_CAPTUM_TEST)
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize(
    "attr_method",
    [
        "GradientShap",
        "IntegratedGradients",
        "DeepLift",
        "DeepLiftShap",
        "InputXGradient",
        "FeaturePermutation",
        "FeatureAblation",
        "KernelShap",
    ],
)
@pytest.mark.parametrize("single_row", [False])
@pytest.mark.parametrize("baselines", ["b|100"])
def test_captum_integration_classification(
    classification_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    attr_method,
    single_row,
    baselines,
):
    (train, test, target) = classification_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "classification"
    _test_captum(
        model_config_class,
        model_config_params,
        data_config,
        train,
        test,
        attr_method,
        single_row,
        baselines,
    )


def _run_cv(
    model_config_class,
    model_config_params,
    data_config,
    train,
    metric,
    cv,
    return_oof,
    reset_datamodule,
):
    model_config = model_config_class(**model_config_params)
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
    cv_scores, oof_predictions = tabular_model.cross_validate(
        cv,
        train,
        metric=metric,
        return_oof=return_oof,
        reset_datamodule=reset_datamodule,
    )
    return cv_scores, oof_predictions


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("cv", [5, KFold(n_splits=3, shuffle=True, random_state=42)])
@pytest.mark.parametrize(
    "metric",
    [
        "loss",
        None,
        lambda y_true, y_pred: r2_score(y_true, y_pred["MedHouseVal_prediction"].values),
    ],
)
@pytest.mark.parametrize("return_oof", [True])
@pytest.mark.parametrize("reset_datamodule", [True, False])
def test_cross_validate_regression(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    cv,
    metric,
    return_oof,
    reset_datamodule,
):
    (train, test, target) = regression_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    if cv is None:
        cv_splits = 5
    elif isinstance(cv, int):
        cv_splits = cv
    else:
        cv_splits = cv.n_splits
    cv_scores, oof_predictions = _run_cv(
        model_config_class,
        model_config_params,
        data_config,
        train,
        metric,
        cv,
        return_oof,
        reset_datamodule,
    )
    assert len(cv_scores) == cv_splits
    if return_oof:
        assert len(oof_predictions) == cv_splits


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize("cv", [None])
@pytest.mark.parametrize(
    "metric",
    [
        "accuracy",
        None,
        lambda y_true, y_pred: accuracy_score(y_true, y_pred["target_prediction"].values),
    ],
)
@pytest.mark.parametrize("return_oof", [True])
@pytest.mark.parametrize("reset_datamodule", [False])
def test_cross_validate_classification(
    classification_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    cv,
    metric,
    return_oof,
    reset_datamodule,
):
    (train, test, target) = classification_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "classification"
    if cv is None:
        cv_splits = 5
    elif isinstance(cv, int):
        cv_splits = cv
    else:
        cv_splits = cv.n_splits
    cv_scores, oof_predictions = _run_cv(
        model_config_class,
        model_config_params,
        data_config,
        train,
        metric,
        cv,
        return_oof,
        reset_datamodule,
    )
    assert len(cv_scores) == cv_splits
    if return_oof:
        assert len(oof_predictions) == cv_splits


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("cv", [None, "validation", 5])
@pytest.mark.parametrize(
    "metric",
    [
        "loss",
        lambda y_true, y_pred: r2_score(y_true, y_pred["MedHouseVal_prediction"].values),
    ],
)
@pytest.mark.parametrize("strategy", ["grid_search", "random_search"])
def test_tuner(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    cv,
    metric,
    strategy,
):
    (train, test, target) = regression_data
    if cv == "validation":
        # To test flow with no CV and no Validation data
        test = None
        cv = None
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    model_config = model_config_class(**model_config_params)
    trainer_config = TrainerConfig(
        max_epochs=1,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()
    tuner = TabularModelTuner(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    if strategy == "grid_search":
        search_space = {
            "model_config__layers": ["8-4", "16-8"],
            "model_config.head_config__dropout": [0.1, 0.2],
            "optimizer_config__optimizer": ["RAdam", "AdamW"],
        }
    else:
        search_space = {
            "model_config__layers": ["8-4", "16-8"],
            "model_config.head_config__dropout": uniform(0, 0.5),
            "optimizer_config__optimizer": ["RAdam", "AdamW"],
        }
    result = tuner.tune(
        train=train,
        validation=test,
        search_space=search_space,
        strategy=strategy,
        n_trials=2,
        cv=cv,
        metric=metric,
        mode="min",
        progress_bar=False,
    )
    if strategy == "grid_search":
        assert len(result.trials_df) == 8
    else:
        assert len(result.trials_df) == 2
    metric_str = metric.__name__ if callable(metric) else metric
    assert result.best_score in result.trials_df[metric_str].values.tolist()


def _run_bagging(
    model_config_class,
    model_config_params,
    data_config,
    train,
    test,
    cv,
    aggregate,
):
    model_config = model_config_class(**model_config_params)
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

    pred_df = tabular_model.bagging_predict(
        cv=cv,
        train=train,
        test=test,
        aggregate=aggregate,
    )
    return pred_df


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize("cv", [2])
@pytest.mark.parametrize(
    "aggregate",
    ["mean", "median", "min", "max", "hard_voting", lambda x: np.median(x, axis=0)],
)
def test_bagging_classification(
    classification_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    cv,
    aggregate,
):
    (train, test, target) = classification_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "classification"
    pred_df = _run_bagging(
        model_config_class,
        model_config_params,
        data_config,
        train,
        test,
        cv,
        aggregate,
    )
    assert len(pred_df) == len(test)
    assert len(set(pred_df["prediction"].values.tolist()) - set(test[target[0]].values.tolist())) == 0


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("cv", [2, KFold(n_splits=3, shuffle=True, random_state=42)])
@pytest.mark.parametrize(
    "aggregate",
    ["mean", "median", "min", "max", "hard_voting", lambda x: np.median(x, axis=0)],
)
def test_bagging_regression(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    cv,
    aggregate,
):
    (train, test, target) = regression_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    if aggregate == "hard_voting":
        with pytest.raises(AssertionError):
            pred_df = _run_bagging(
                model_config_class,
                model_config_params,
                data_config,
                train,
                test,
                cv,
                aggregate,
            )
        return
    else:
        pred_df = _run_bagging(
            model_config_class,
            model_config_params,
            data_config,
            train,
            test,
            cv,
            aggregate,
        )
        assert len(pred_df) == len(test)


def _run_tta(
    model_config_class,
    model_config_params,
    data_config,
    train,
    test,
    aggregate,
):
    model_config = model_config_class(**model_config_params)
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
    tabular_model.fit(train)
    pred_df = tabular_model.predict(test, test_time_augmentation=True, num_tta=2, aggregate_tta=aggregate)
    return pred_df


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize(
    "aggregate",
    ["mean", "median", "min", "max", "hard_voting", lambda x: np.median(x, axis=0)],
)
def test_tta_classification(
    classification_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    aggregate,
):
    (train, test, target) = classification_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "classification"
    pred_df = _run_tta(
        model_config_class,
        model_config_params,
        data_config,
        train,
        test,
        aggregate,
    )
    assert len(pred_df) == len(test)
    assert len(set(pred_df["prediction"].values.tolist()) - set(test[target[0]].values.tolist())) == 0


@pytest.mark.parametrize("model_config_class", [(CategoryEmbeddingModelConfig, {"layers": "10-20"})])
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize(
    "aggregate",
    ["mean", "median", "min", "max", "hard_voting", lambda x: np.median(x, axis=0)],
)
def test_tta_regression(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    aggregate,
):
    (train, test, target) = regression_data
    model_config_class, model_config_params = model_config_class
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )

    model_config_params["task"] = "regression"
    if aggregate == "hard_voting":
        with pytest.raises(AssertionError):
            pred_df = _run_tta(
                model_config_class,
                model_config_params,
                data_config,
                train,
                test,
                aggregate,
            )
        return
    else:
        pred_df = _run_tta(
            model_config_class,
            model_config_params,
            data_config,
            train,
            test,
            aggregate,
        )
        assert len(pred_df) == len(test)
    pred_df = _run_tta(
        model_config_class,
        model_config_params,
        data_config,
        train,
        test,
        aggregate,
    )
    assert len(pred_df) == len(test)


def _run_model_compare(
    task, model_list, data_config, trainer_config, optimizer_config, train, test, metric, rank_metric
):
    model_list = copy.deepcopy(model_list)
    if isinstance(model_list, list):
        model_list = [mdl(task=task, **params) for mdl, params in model_list]

    return model_sweep(
        task=task,
        train=train,
        test=test,
        data_config=data_config,
        trainer_config=trainer_config,
        optimizer_config=optimizer_config,
        model_list=model_list,
        metrics=metric[0],
        metrics_params=metric[1],
        metrics_prob_input=metric[2],
        rank_metric=rank_metric,
    )


@pytest.mark.parametrize("model_list", ["lite", MODEL_CONFIG_MODEL_SWEEP_TEST])
@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
@pytest.mark.parametrize(
    "metric",
    [
        (None, None, None),
        (["accuracy"], [{}], [False]),
        (["accuracy", "f1_score"], [{}, {"average": "macro"}], [False, True]),
    ],
)
@pytest.mark.parametrize("rank_metric", [("accuracy", "higher_is_better"), ("loss", "lower_is_better")])
def test_model_compare_classification(
    classification_data, model_list, continuous_cols, categorical_cols, metric, rank_metric
):
    (train, test, target) = classification_data
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )
    trainer_config = TrainerConfig(
        max_epochs=1, checkpoints=None, early_stopping=None, accelerator="cpu", fast_dev_run=True
    )
    optimizer_config = OptimizerConfig()
    comp_df, best_model = _run_model_compare(
        "classification", model_list, data_config, trainer_config, optimizer_config, train, test, metric, rank_metric
    )
    if model_list == "lite":
        assert len(comp_df) == 3
    else:
        assert len(comp_df) == len(model_list)
    # best_score = comp_df[f"test_{rank_metric[0]}"].values.tolist()[0]
    # # there may be multiple models with the same score
    # best_models = comp_df.loc[comp_df[f"test_{rank_metric[0]}"] == best_score, "model"].values.tolist()
    # assert best_model.model._get_name() in best_models


@pytest.mark.parametrize("model_list", ["lite", MODEL_CONFIG_MODEL_SWEEP_TEST])
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize(
    "metric",
    [
        (["mean_squared_error"], [{}], [False]),
    ],
)
@pytest.mark.parametrize("rank_metric", [("mean_squared_error", "lower_is_better"), ("loss", "lower_is_better")])
def test_model_compare_regression(regression_data, model_list, continuous_cols, categorical_cols, metric, rank_metric):
    (train, test, target) = regression_data
    data_config = DataConfig(
        target=target,
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        handle_missing_values=True,
        handle_unknown_categories=True,
    )
    trainer_config = TrainerConfig(
        max_epochs=3,
        checkpoints=None,
        early_stopping=None,
        accelerator="cpu",
        fast_dev_run=True,
    )
    optimizer_config = OptimizerConfig()
    comp_df, best_model = _run_model_compare(
        "regression", model_list, data_config, trainer_config, optimizer_config, train, test, metric, rank_metric
    )
    if model_list == "lite":
        assert len(comp_df) == 3
    else:
        assert len(comp_df) == len(model_list)
    # best_score = comp_df[f"test_{rank_metric[0]}"].values.tolist()[0]
    # # there may be multiple models with the same score
    # best_models = comp_df.loc[comp_df[f"test_{rank_metric[0]}"] == best_score, "model"].values.tolist()
    # assert best_model.model._get_name() in best_models


@pytest.mark.parametrize("model_config_class", MODEL_CONFIG_SAVE_TEST)
@pytest.mark.parametrize("continuous_cols", [list(DATASET_CONTINUOUS_COLUMNS)])
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"]])
@pytest.mark.parametrize("custom_metrics", [None, [fake_metric]])
@pytest.mark.parametrize("custom_loss", [None, torch.nn.L1Loss()])
@pytest.mark.parametrize("custom_optimizer", [None, torch.optim.Adagrad, "SGD", "torch_optimizer.AdaBound"])
def test_str_repr(
    regression_data,
    model_config_class,
    continuous_cols,
    categorical_cols,
    custom_metrics,
    custom_loss,
    custom_optimizer,
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
    assert "Not Initialized" in str(tabular_model)
    assert "Not Initialized" in repr(tabular_model)
    assert "Model Summary" not in tabular_model._repr_html_()
    assert "Model Config" in tabular_model._repr_html_()
    assert "config" in tabular_model.__repr__()
    assert "config" not in str(tabular_model)
    tabular_model.fit(
        train=train,
        metrics=custom_metrics,
        metrics_prob_inputs=None if custom_metrics is None else [False],
        loss=custom_loss,
        optimizer=custom_optimizer,
        optimizer_params={},
    )
    assert model_config_class._model_name in str(tabular_model)
    assert model_config_class._model_name in repr(tabular_model)
    assert "Model Summary" in tabular_model._repr_html_()
    assert "Model Config" in tabular_model._repr_html_()
    assert "config" in tabular_model.__repr__()
    assert model_config_class._model_name in tabular_model._repr_html_()
