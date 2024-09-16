from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.datasets import fetch_california_housing

from pytorch_tabular.config import DataConfig, ModelConfig, OptimizerConfig, TrainerConfig

# from pytorch_tabular.models.deep_gmm import (
#     DeepGaussianMixtureModelConfig,
# )
from pytorch_tabular.models.node import NODEBackbone

# from pytorch_tabular.models.node import utils as utils
from pytorch_tabular.tabular_model import TabularModel


@dataclass
class MultiStageModelConfig(ModelConfig):
    num_layers: int = field(
        default=1,
        metadata={"help": "Number of Oblivious Decision Tree Layers in the Dense Architecture"},
    )
    num_trees: int = field(
        default=2048,
        metadata={"help": "Number of Oblivious Decision Trees in each layer"},
    )
    additional_tree_output_dim: int = field(
        default=3,
        metadata={
            "help": "The additional output dimensions which is only used to pass through different layers"
            " of the architectures. Only the first output_dim outputs will be used for prediction"
        },
    )
    depth: int = field(
        default=6,
        metadata={"help": "The depth of the individual Oblivious Decision Trees"},
    )
    choice_function: str = field(
        default="entmax15",
        metadata={
            "help": "Generates a sparse probability distribution to be used as feature weights"
            " (aka, soft feature selection)",
            "choices": ["entmax15", "sparsemax"],
        },
    )
    bin_function: str = field(
        default="entmoid15",
        metadata={
            "help": "Generates a sparse probability distribution to be used as tree leaf weights",
            "choices": ["entmoid15", "sparsemoid"],
        },
    )
    max_features: Optional[int] = field(
        default=None,
        metadata={
            "help": "If not None, sets a max limit on the number of features to be carried forward"
            " from layer to layer in the Dense Architecture"
        },
    )
    input_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout to be applied to the inputs between layers of the Dense Architecture"},
    )
    initialize_response: str = field(
        default="normal",
        metadata={
            "help": "Initializing the response variable in the Oblivious Decision Trees."
            " By default, it is a standard normal distribution",
            "choices": ["normal", "uniform"],
        },
    )
    initialize_selection_logits: str = field(
        default="uniform",
        metadata={
            "help": "Initializing the feature selector. By default is a uniform distribution across the features",
            "choices": ["uniform", "normal"],
        },
    )
    threshold_init_beta: float = field(
        default=1.0,
        metadata={
            "help": """
                Used in the Data-aware initialization of thresholds where the threshold is initialized randomly
                (with a beta distribution) to feature values in the first batch.
                It initializes threshold to a q-th quantile of data points.
                where q ~ Beta(:threshold_init_beta:, :threshold_init_beta:)
                If this param is set to 1, initial thresholds will have the same distribution as data points
                If greater than 1 (e.g. 10), thresholds will be closer to median data value
                If less than 1 (e.g. 0.1), thresholds will approach min/max data values.
            """
        },
    )
    threshold_init_cutoff: float = field(
        default=1.0,
        metadata={
            "help": """
                Used in the Data-aware initialization of scales(used in the scaling ODTs).
                It is initialized in such a way that all the samples in the first batch belong to the linear
                region of the entmoid/sparsemoid(bin-selectors) and thereby have non-zero gradients
                Threshold log-temperatures initializer, in (0, inf)
                By default(1.0), log-temperatures are initialized in such a way that all bin selectors
                end up in the linear region of sparse-sigmoid. The temperatures are then scaled by this parameter.
                Setting this value > 1.0 will result in some margin between data points and sparse-sigmoid cutoff value
                Setting this value < 1.0 will cause (1 - value) part of data points to end up in flat sparse-sigmoid
                region. For instance, threshold_init_cutoff = 0.9 will set 10% points equal to 0.0 or 1.0
                Setting this value > 1.0 will result in a margin between data points and sparse-sigmoid cutoff value
                All points will be between (0.5 - 0.5 / threshold_init_cutoff) and (0.5 + 0.5 / threshold_init_cutoff)
            """
        },
    )
    embedding_dims: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The dimensions of the embedding for each categorical column as a list of tuples"
            " (cardinality, embedding_dim). If left empty, will infer using the cardinality"
            " of the categorical column using the rule min(50, (x + 1) // 2)"
        },
    )
    embedding_dropout: float = field(
        default=0.0,
        metadata={"help": "probability of an embedding element to be zeroed."},
    )


from pytorch_tabular.models import BaseModel  # noqa: E402


class MultiStageModel(BaseModel):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self.embedding_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in self.hparams.embedding_dims])
        if self.hparams.embedding_dropout != 0 and self.hparams.embedded_cat_dim != 0:
            self.embedding_dropout = nn.Dropout(self.hparams.embedding_dropout)
        self.hparams.node_input_dim = self.hparams.continuous_dim + self.hparams.embedded_cat_dim
        self.backbone = NODEBackbone(self.hparams)
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification

        def subset_clf(x):
            return x[..., :2].mean(dim=-2)

        def subset_rg(x):
            return x[..., 2:4].mean(dim=-2)

        # self.clf_out = utils.Lambda(subset_clf)
        # self.rg_out = utils.Lambda(subset_rg)
        self.classification_loss = nn.CrossEntropyLoss()

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.hparams.embedded_cat_dim != 0:
            x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [embedding_layer(categorical_data[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.hparams.embedded_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        if self.hparams.embedding_dropout != 0 and self.hparams.embedded_cat_dim != 0:
            x = self.embedding_dropout(x)
        x = self.backbone(x)
        clf_logits = self.clf_out(x)
        clf_prob = nn.functional.gumbel_softmax(clf_logits, tau=1, dim=-1)

        rg_out = self.rg_out(x)

        y_hat = torch.sum(clf_prob * rg_out, dim=-1)
        if (self.hparams.task == "regression") and (self.hparams.target_range is not None):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return {"logits": y_hat, "clf_logits": clf_logits}

    def training_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        loss = self.calculate_loss(y, ret_value["clf_logits"], ret_value["logits"], tag="train")
        self.calculate_metrics(y, ret_value["logits"], tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        self.calculate_loss(y, ret_value["clf_logits"], ret_value["logits"], tag="valid")
        self.calculate_metrics(y, ret_value["logits"], tag="valid")
        return ret_value["logits"], y

    def test_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        self.calculate_loss(y, ret_value["clf_logits"], ret_value["logits"], tag="test")
        self.calculate_metrics(y, ret_value["logits"], tag="test")
        return ret_value["logits"], y

    def calculate_loss(self, y, classification_logits, y_hat, tag):
        cl_loss = self.classification_loss(classification_logits.squeeze(), y[:, 0].squeeze().long())
        rg_loss = self.loss(y_hat, y[:, 1])
        self.log(
            f"{tag}_classification_loss",
            cl_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        self.log(
            f"{tag}_regression_loss",
            cl_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=False,
        )
        computed_loss = cl_loss + rg_loss
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return computed_loss

    # Escaping metric calculation for cause default calculation would fail and not make sense
    # for this type of combined classification and regression task
    def calculate_metrics(self, y, y_hat, tag):
        pass


dataset = fetch_california_housing(data_home="data", as_frame=True)
dataset.frame["HouseAgeBin"] = pd.qcut(dataset.frame["HouseAge"], q=4)
dataset.frame.HouseAgeBin = "age_" + dataset.frame.HouseAgeBin.cat.codes.astype(str)

test_idx = dataset.frame.sample(int(0.2 * len(dataset.frame)), random_state=42).index
test = dataset.frame[dataset.frame.index.isin(test_idx)]
train = dataset.frame[~dataset.frame.index.isin(test_idx)]


epochs = 15
batch_size = 128
steps_per_epoch = int((len(train) // batch_size) * 0.9)
data_config = DataConfig(
    target=["HouseAgeBin"] + dataset.target_names,
    continuous_cols=[
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
    # continuous_cols=[],
    categorical_cols=["HouseAgeBin"],
    continuous_feature_transform="quantile_uniform",  # "yeo-johnson",
    normalize_continuous_features=True,
)
trainer_config = TrainerConfig(
    auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=batch_size,
    max_epochs=epochs,
    early_stopping_patience=5,
    checkpoints=None,
    #         fast_dev_run=True,
)
optimizer_config = OptimizerConfig(
    lr_scheduler="OneCycleLR",
    lr_scheduler_params={"max_lr": 0.005, "epochs": epochs, "steps_per_epoch": steps_per_epoch},
)
#     optimizer_config = OptimizerConfig(lr_scheduler="ReduceLROnPlateau", lr_scheduler_params={"patience":3})
model_config = MultiStageModelConfig(
    task="regression",
    num_layers=1,  # Number of Dense Layers
    num_trees=2048,  # Number of Trees in each layer
    depth=6,  # Depth of each Tree
    learning_rate=0.02,
    additional_tree_output_dim=25,
)
# model_config.validate()
# model_config = NodeConfig(task="regression", depth=2)
# trainer_config = TrainerConfig(checkpoints=None, max_epochs=5, profiler=None)
# experiment_config = ExperimentConfig(
#     project_name="DeepGMM_test",
#     run_name="wand_debug",
#     log_target="wandb",
#     exp_watch="gradients",
#     log_logits=True
# )
# optimizer_config = OptimizerConfig()

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
    model_callable=MultiStageModel,
)
tabular_model.fit(train=train)

result = tabular_model.evaluate(test)
# print(result)
# # print(result[0]['train_loss'])
pred_df = tabular_model.predict(test, quantiles=[0.25])
print(pred_df.head())
# pred_df.to_csv("output/temp2.csv")
