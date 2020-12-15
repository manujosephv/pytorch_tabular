from pytorch_tabular.models.node.config import NodeConfig
from sklearn.datasets import fetch_california_housing
from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
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


dataset = fetch_california_housing(data_home="data", as_frame=True)
dataset.frame["HouseAgeBin"] = pd.qcut(dataset.frame["HouseAge"], q=4)
dataset.frame.HouseAgeBin = "age_" + dataset.frame.HouseAgeBin.cat.codes.astype(str)

test_idx = dataset.frame.sample(int(0.2 * len(dataset.frame)), random_state=42).index
test = dataset.frame[dataset.frame.index.isin(test_idx)]
train = dataset.frame[~dataset.frame.index.isin(test_idx)]

data_config = DataConfig(
    target=dataset.target_names+ ["MedInc"],
    # continuous_cols=[
    #     "AveRooms",
    #     "AveBedrms",
    #     "Population",
    #     "AveOccup",
    #     "Latitude",
    #     "Longitude",
    # ],
    continuous_cols=[],
    categorical_cols=["HouseAgeBin"],
    continuous_feature_transform=None,#"yeo-johnson",
    normalize_continuous_features=True
)
# model_config = CategoryEmbeddingModelConfig(
#     task="regression",
#     target_range=[
#         dataset.frame[dataset.target_names].min().item(),
#         dataset.frame[dataset.target_names].max().item(),
#     ],
# )
model_config = NodeConfig(task="regression", depth=2, embed_categorical=False)
trainer_config = TrainerConfig(checkpoints=None, max_epochs=1)
# experiment_config = ExperimentConfig(project_name="Tabular_test")
optimizer_config = OptimizerConfig()

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
)
tabular_model.fit(train=train, test=test)

result = tabular_model.evaluate(test)
print(result)
print(result[0]['train_loss'])
pred_df = tabular_model.predict(test)
pred_df.to_csv("output/temp2.csv")