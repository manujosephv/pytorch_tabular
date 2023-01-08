#!/usr/bin/env python
"""Tests for `pytorch_tabular` package."""
import pytest

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import GatedAdditiveTreeEnsembleConfig

# from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer


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
    ],
)
@pytest.mark.parametrize("categorical_cols", [["HouseAgeBin"], []])
@pytest.mark.parametrize("continuous_feature_transform", [None])
@pytest.mark.parametrize("normalize_continuous_features", [True])
@pytest.mark.parametrize("target_range", [True, False])
def test_regression(
    regression_data,
    multi_target,
    continuous_cols,
    categorical_cols,
    continuous_feature_transform,
    normalize_continuous_features,
    target_range,
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
        model_config_params = dict(
            task="regression",
            gflu_stages=1,
            tree_depth=1,
            num_trees=2,
        )
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
        model_config = GatedAdditiveTreeEnsembleConfig(**model_config_params)
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

        result = tabular_model.evaluate(test)
        assert "test_mean_squared_error" in result[0].keys()
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


@pytest.mark.parametrize(
    "continuous_cols",
    [
        [f"feature_{i}" for i in range(54)],
    ],
)
@pytest.mark.parametrize("categorical_cols", [["feature_0_cat"]])
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
        model_config_params = dict(
            task="classification",
            gflu_stages=1,
            tree_depth=1,
            num_trees=2,
        )
        model_config = GatedAdditiveTreeEnsembleConfig(**model_config_params)
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

        result = tabular_model.evaluate(test)
        assert "test_accuracy" in result[0].keys()
        pred_df = tabular_model.predict(test)
        assert pred_df.shape[0] == test.shape[0]


# def test_embedding_transformer(regression_data):
#     (train, test, target) = regression_data
#     data_config = DataConfig(
#         target=target,
#         continuous_cols=[
#             "AveRooms",
#             "AveBedrms",
#             "Population",
#             "AveOccup",
#             "Latitude",
#             "Longitude",
#         ],
#         categorical_cols=["HouseAgeBin"],
#     )
#     model_config_params = dict(
#         task="regression",
#         input_embed_dim=8,
#         num_attn_blocks=1,
#         num_heads=2,
#     )
#     model_config = FTTransformerConfig(**model_config_params)
#     trainer_config = TrainerConfig(
#         max_epochs=1,
#         checkpoints=None,
#         early_stopping=None,
#         accelerator="cpu",
#         fast_dev_run=True,
#     )
#     optimizer_config = OptimizerConfig()

#     tabular_model = TabularModel(
#         data_config=data_config,
#         model_config=model_config,
#         optimizer_config=optimizer_config,
#         trainer_config=trainer_config,
#     )
#     tabular_model.fit(train=train, test=test)

#     transformer = CategoricalEmbeddingTransformer(tabular_model)
#     train_transform = transformer.fit_transform(train)
#     embed_cols = [
#         col for col in train_transform.columns if "HouseAgeBin_embed_dim" in col
#     ]
#     assert len(train["HouseAgeBin"].unique()) + 1 == len(
#         transformer._mapping["HouseAgeBin"].keys()
#     )
#     assert all(
#         [
#             val.shape[0] == len(embed_cols)
#             for val in transformer._mapping["HouseAgeBin"].values()
#         ]
#     )
