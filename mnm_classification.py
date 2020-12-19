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
import numpy as np


X_train = pd.read_csv("data/mnm/dust_primer_cb_train.csv")
y_train = pd.read_csv("data/mnm/y_train_primer.csv").loc[:, "dust"]
orig_train = X_train.join(y_train)
target = ["dust"]
categorical_cols = [
    "PLATFORM",
    "PRIMAR",
    "PRIMAR_Color_Intensity",
    "PRIMAR_Color_M_S",
    "Prev_PLATFORM",
    "Prev_PRIMAR",
    "Prev_PRIMAR_Color_Intensity",
    "Prev_PRIMAR_Color_M_S",
    "Prev_TOPCOAT",
    "Prev_TOPCOAT_Color_Intensity",
    "Prev_TOPCOAT_Color_M_S",
    "TOPCOAT",
    "TOPCOAT_Color_Intensity",
    "TOPCOAT_Color_M_S",
    "bc_hum_threshold",
    "bc_temp_threshold",
    "cc_hum_threshold",
    "cc_temp_threshold",
    "primer_hum_threshold",
    "primer_temp_threshold",
]
numerical_cols = [
    "ER_day",
    "TOPCOAT_last_used",
    "T_diff_prev",
    "amb_hum_last_recorded",
    "amb_temp_last_recorded",
    "base_coat_hum_last_recorded",
    "base_coat_temp_last_recorded",
    "bc_intvl_heat_index_mean",
    "bc_intvl_wind_chill_index_mean",
    "cc_intvl_heat_index_mean",
    "cc_intvl_wind_chill_index_mean",
    "clear_coat_hum_last_recorded",
    "clear_coat_temp_last_recorded",
    "dewpoint_amb",
    "dewpoint_bc",
    "dewpoint_cc",
    "first_run_of_day_f",
    "hour_of_run",
    "lunch_flag",
    "month_of_run",
    "num_vehicles_in_bc",
    "num_vehicles_in_cc",
    "paint_pressure_last_record",
    "paint_temp_last_record",
    "primer_heatup_zone1_max",
    "primer_heatup_zone1_mean",
    "primer_heatup_zone1_min",
    "primer_heatup_zone2_max",
    "primer_heatup_zone2_mean",
    "primer_heatup_zone2_min",
    "primer_holding_zone1_max",
    "primer_holding_zone1_mean",
    "primer_holding_zone1_min",
    "primer_holding_zone2_max",
    "primer_holding_zone2_mean",
    "primer_holding_zone2_min",
    "primer_intvl_amb_hum_max",
    "primer_intvl_amb_hum_mean",
    "primer_intvl_amb_hum_min",
    "primer_intvl_amb_temp_max",
    "primer_intvl_amb_temp_mean",
    "primer_intvl_amb_temp_min",
    "primer_intvl_heat_index_mean",
    "primer_intvl_primer_hum_max",
    "primer_intvl_primer_hum_mean",
    "primer_intvl_primer_hum_min",
    "primer_intvl_primer_temp_max",
    "primer_intvl_primer_temp_mean",
    "primer_intvl_primer_temp_min",
    "primer_intvl_wind_chill_index_mean",
    "primer_rh_change",
    "primer_temp_change",
    "pt_primer_lag",
    "pt_tc_lag",
    "run_number_of_body",
    "shift_change_flag",
    "tc_primerexit_lag",
    "viscosity_day",
    "weekday_of_run",
    "wind_speed",
]
date_columns = []
train = orig_train[numerical_cols + categorical_cols + date_columns + [target]].copy()

data_config = DataConfig(
    target=target,
    continuous_cols=numerical_cols,
    categorical_cols=categorical_cols,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True,
)
# model_config = CategoryEmbeddingModelConfig(task="classification")
model_config = NodeConfig(
    task="classification",
    num_layers=2,
    num_trees=512,
    learning_rate=1e-1,
    embed_categorical=True,
    metrics=["MeanSquaredLogError"],
)
trainer_config = TrainerConfig()
experiment_config = ExperimentConfig(project_name="Tabular_test")
optimizer_config = OptimizerConfig()

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
)
tabular_model.fit(train=train)

result = tabular_model.evaluate(train)
print(result)
pred_df = tabular_model.predict(train)
pred_df.to_csv("output/mnm.csv")