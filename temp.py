from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import CategoryEmbeddingModel, TabularModel
import pytorch_lightning as pl

df = pd.read_csv(r"data\temp.csv")
df["intro_period_qtr"] = df.intro_period.str.split("-", expand=True).iloc[:, 1]
for col in ["intro_date_usa", "phaseout_date_usa"]:
    df[col] = pd.to_datetime(df[col])
df["year"] = df["intro_date_usa"].dt.year
test_size = int(0.2 * (df[df["year"] == 2020].shape[0]))
test_idx = df[df.year == 2020].sample(test_size, random_state=42).index
train = df[~df.index.isin(test_idx)].copy()
test = df[df.index.isin(test_idx)].copy()

data_config = DataConfig(
    target=["block_0", "block_1", "block_2"],
    continuous_cols=[
        "retail_price_usa",
        "phaseout_days_1",
        "phaseout_days_2",
        "phaseout_days_3",
    ],
    categorical_cols=[
        "usa_stocking_policy",
        "usa_merch_class",
        "primary_color",
        "secondary_color",
        "gender",
        "gbu",
        "line_plan_business",
        "model_series",
        "style_state",
        "closure_type",
        "segment",
        "smu_type",
        "mdl",
        "intro_period_qtr",
    ],
    date_cols=[('intro_date_usa','D'),('phaseout_date_usa',"D")]
)

model_config = ModelConfig(task="regression", layers="64-32", activation="SELU")
trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=2048,
    max_epochs=1000,
    # track_grad_norm=2,
    gradient_clip_val=10
)
experiment_config = ExperimentConfig(project_name="Tabular_test", log_logits=True)
optimizer_config = OptimizerConfig()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
)
tabular_model.fit(train=train, test=test)

result = tabular_model.evaluate(test)
print(result)
pred_df = tabular_model.predict(test)
pred_df.to_csv("temp.csv")
