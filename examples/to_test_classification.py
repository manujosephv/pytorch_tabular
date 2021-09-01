from pytorch_tabular.models.tab_transformer.config import TabTransformerConfig
from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
import torch
import numpy as np
from torch.functional import norm
# torch.manual_seed(0)
# np.random.seed(0)
# torch.set_deterministic(True)

from sklearn.datasets import fetch_covtype

# from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.models.category_embedding.config import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from pathlib import Path
# import wget
from pytorch_tabular.utils import get_balanced_sampler, get_class_weighted_cross_entropy


BASE_DIR = Path.home().joinpath('data')
datafile = BASE_DIR.joinpath('covtype.data.gz')
datafile.parent.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
if not datafile.exists():
    wget.download(url, datafile.as_posix())

target_name = ["Covertype"]

cat_col_names = [
    "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
    "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4",
    "Soil_Type5", "Soil_Type6", "Soil_Type7", "Soil_Type8", "Soil_Type9",
    "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13", "Soil_Type14",
    "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19",
    "Soil_Type20", "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24",
    "Soil_Type25", "Soil_Type26", "Soil_Type27", "Soil_Type28", "Soil_Type29",
    "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
    "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39",
    "Soil_Type40"
]

num_col_names = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points"
]

feature_columns = (
    num_col_names + cat_col_names + target_name)

df = pd.read_csv(datafile, header=None, names=feature_columns)
# cat_col_names = []

# num_col_names = [
#     "Elevation", "Aspect"
# ]
# feature_columns = (
#     num_col_names + cat_col_names + target_name)
# df = df.loc[:,feature_columns]
df.head()
train, test = train_test_split(df, random_state=42)
train, val = train_test_split(train, random_state=42)
num_classes = len(set(train[target_name].values.ravel()))

data_config = DataConfig(
    target=target_name,
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform=None,#"quantile_normal",
    normalize_continuous_features=False,
)
# model_config = CategoryEmbeddingModelConfig(task="classification", metrics=["f1","accuracy"], metrics_params=[{"num_classes":num_classes},{}])
# model_config = NodeConfig(
#     task="classification",
#     depth=4,
#     num_trees=1024,
#     input_dropout=0.0,
#     metrics=["f1", "accuracy"],
#     metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
# )
model_config = TabTransformerConfig(
    task="classification",
    metrics=["f1", "accuracy"],
    share_embedding = True,
    share_embedding_strategy="add",
    shared_embedding_fraction=0.25,
    metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
)
# model_config = FTTransformerConfig(
#     task="classification",
#     metrics=["f1", "accuracy"],
#     # embedding_initialization=None,
#     embedding_bias=True,
#     share_embedding = True,
#     share_embedding_strategy="fraction",
#     shared_embedding_fraction=0.25,
#     metrics_params=[{"num_classes": num_classes, "average": "macro"}, {}],
# )
trainer_config = TrainerConfig(gpus=-1, auto_select_gpus=True, fast_dev_run=True, max_epochs=5, batch_size=512)
experiment_config = ExperimentConfig(project_name="PyTorch Tabular Example", 
                                     run_name="node_forest_cov", 
                                     exp_watch="gradients", 
                                     log_target="wandb", 
                                     log_logits=True)
optimizer_config = OptimizerConfig()

# tabular_model = TabularModel(
#     data_config="examples/data_config.yml",
#     model_config="examples/model_config.yml",
#     optimizer_config="examples/optimizer_config.yml",
#     trainer_config="examples/trainer_config.yml",
#     # experiment_config=experiment_config,
# )
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
)
sampler = get_balanced_sampler(train[target_name].values.ravel())
# cust_loss = get_class_weighted_cross_entropy(train[target_name].values.ravel())
tabular_model.fit(
    train=train, 
    validation=val, 
    # loss=cust_loss,
    train_sampler=sampler)

from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer
transformer = CategoricalEmbeddingTransformer(tabular_model)
train_transform = transformer.fit_transform(train)
# test_transform = transformer.transform(test)
# ft = tabular_model.model.feature_importance()
# result = tabular_model.evaluate(test)
# print(result)
# test.drop(columns=ta6rget_name, inplace=True)
# pred_df = tabular_model.predict(test)
# print(pred_df.head())
# pred_df.to_csv("output/temp2.csv")
# tabular_model.save_model("test_save")
# new_model = TabularModel.load_from_checkpoint("test_save")
# result = new_model.evaluate(test)
# print(result)