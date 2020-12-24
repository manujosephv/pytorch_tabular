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
from pytorch_tabular.models.category_embedding.config import CategoryEmbeddingModelConfig
from pytorch_tabular.models.category_embedding.category_embedding_model import CategoryEmbeddingModel
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
import numpy as np
from sklearn.preprocessing import PowerTransformer


dataset = fetch_covtype(data_home="data")
data = np.hstack([dataset.data, dataset.target.reshape(-1,1)])[:5000,:]
col_names = [f"feature_{i}" for i in range(data.shape[-1])]
col_names[-1] = "target"
data = pd.DataFrame(data, columns=col_names)
test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
test = data[data.index.isin(test_idx)]
train = data[~data.index.isin(test_idx)]

data_config = DataConfig(
        target=['target'],
        continuous_cols=col_names[:-1],
        categorical_cols=[],
        continuous_feature_transform="yeo-johnson"
    )
# model_config = CategoryEmbeddingModelConfig(task="classification")
model_config = NodeConfig(task="classification", depth=2)
trainer_config = TrainerConfig()
# experiment_config = ExperimentConfig(project_name="Tabular_test")
optimizer_config = OptimizerConfig()

tabular_model = TabularModel(
    data_config="examples/data_config.yml",
    model_config="examples/model_config.yml",
    optimizer_config="examples/optimizer_config.yml",
    trainer_config="examples/trainer_config.yml",
    # experiment_config=experiment_config,
)
tabular_model.fit(train=train, test=test, target_transform=PowerTransformer(method="yeo-johnson"))

result = tabular_model.evaluate(test)
print(result)
pred_df = tabular_model.predict(test)
pred_df.to_csv("output/temp2.csv")
# tabular_model.save_model("test_save")
# new_model = TabularModel.load_from_checkpoint("test_save")
# result = new_model.evaluate(test)