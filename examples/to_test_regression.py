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
from pytorch_tabular.models import AutoIntModel, AutoIntConfig

from pytorch_tabular.models.mixture_density import (
    CategoryEmbeddingMDNConfig,
    MixtureDensityHeadConfig,
    NODEMDNConfig,
)

# from pytorch_tabular.models.deep_gmm import (
#     DeepGaussianMixtureModelConfig,
# )
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
from sklearn.preprocessing import PowerTransformer
import torch

dataset = fetch_california_housing(data_home="data", as_frame=True)
dataset.frame["HouseAgeBin"] = pd.qcut(dataset.frame["HouseAge"], q=4)
dataset.frame.HouseAgeBin = "age_" + dataset.frame.HouseAgeBin.cat.codes.astype(str)
dataset.frame["AveRoomsBin"] = pd.qcut(dataset.frame["AveRooms"], q=3)
dataset.frame.AveRoomsBin = "av_rm_" + dataset.frame.AveRoomsBin.cat.codes.astype(str)

test_idx = dataset.frame.sample(int(0.2 * len(dataset.frame)), random_state=42).index
test = dataset.frame[dataset.frame.index.isin(test_idx)]
train = dataset.frame[~dataset.frame.index.isin(test_idx)]

data_config = DataConfig(
    target=dataset.target_names,
    continuous_cols=[
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
    # continuous_cols=[],
    categorical_cols=["HouseAgeBin", "AveRoomsBin"],
    continuous_feature_transform=None,  # "yeo-johnson",
    normalize_continuous_features=True,
)

# mdn_config = MixtureDensityHeadConfig(num_gaussian=2)
# model_config = NODEMDNConfig(
#     task="regression",
#     # initialization="blah",
#     mdn_config = mdn_config
# )
# # model_config.validate()
model_config = CategoryEmbeddingModelConfig(task="regression")
# model_config = AutoIntConfig(
#     task="regression",
#     deep_layers=True,
#     embedding_dropout=0.2,
#     batch_norm_continuous_input=True,
#     attention_pooling=True,
# )
trainer_config = TrainerConfig(
    checkpoints=None,
    max_epochs=2,
    gpus=1,
    profiler=None,
    fast_dev_run=False,
    auto_lr_find=True,
)
# experiment_config = ExperimentConfig(
#     project_name="DeepGMM_test",
#     run_name="wand_debug",
#     log_target="wandb",
#     exp_watch="gradients",
#     log_logits=True
# )
optimizer_config = OptimizerConfig()


def fake_metric(y_hat, y):
    return (y_hat - y).mean()


from sklearn.preprocessing import PowerTransformer

tr = PowerTransformer()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
)
tabular_model.fit(
    train=train,
    test=test,
    metrics=[fake_metric],
    target_transform=tr,
    loss=torch.nn.L1Loss(),
    optimizer=torch.optim.Adagrad,
    optimizer_params={},
)

from pytorch_tabular.feature_extractor import DeepFeatureExtractor

# dt = DeepFeatureExtractor(tabular_model)
# enc_df = dt.fit_transform(test)
# print(enc_df.head())
# tabular_model.save_model("examples/sample")
result = tabular_model.evaluate(test)
print(result)
# # # print(result[0]['train_loss'])
# new_mdl = TabularModel.load_from_checkpoint("examples/sample")
# # TODO test none no test loader
# result = new_mdl.evaluate(test)
# print(result)
# tabular_model.fit(
#     train=train, test=test, metrics=[fake_metric], target_transform=tr, max_epochs=2
# )
# pred_df = tabular_model.predict(test, quantiles=[0.25], ret_logits=True)
# print(pred_df.head())

# pred_df.to_csv("output/temp2.csv")
