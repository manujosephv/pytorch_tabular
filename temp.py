from pytorch_tabular.models.tabnet.config import TabNetModelConfig
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    OptimizerConfig,
    TrainerConfig,
)
import pandas as pd
from pytorch_tabular.tabular_model import TabularModel

def read_style_df(n_seasons=4, periods=6):
    df = pd.read_csv(f"data\\style_df_{n_seasons}_seasons_{periods}_periods.csv")
    for col in ['intro_date_usa','phaseout_date_usa']:
        df[col] = pd.to_datetime(df[col])
    df['year'] = df['intro_date_usa'].dt.year
    return df

style_df = read_style_df(n_seasons=4, periods=6)
style_df = style_df.dropna()
test_size = int(0.2*(style_df[style_df['year']==2020].shape[0]))
test_idx = style_df[style_df.year==2020].sample(test_size, random_state=42).index
train = style_df[~style_df.index.isin(test_idx)].copy()
test = style_df[style_df.index.isin(test_idx)].copy()

data_config = DataConfig(
    target=["block_0"],
    continuous_cols=[
        "retail_price_usa",
        "phaseout_days_1"
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
        "mdl",
        "intro_period_qtr",
    ],
    normalize_continuous_features=True,
    continuous_feature_transform="quantile_normal",
    date_cols=[('intro_date_usa','D')],
    encode_date_cols=True
)
# TODO Data_cols and encode_date_col causing an error
# model_config = CategoryEmbeddingModelConfig(task="regression", layers="64-32-16", activation="SELU")
model_config = NodeConfig(task="regression", learning_rate=1, embed_categorical=True)
trainer_config = TrainerConfig(
    # auto_lr_find=True,
    # auto_scale_batch_size=True,
    batch_size=1024,
    max_epochs=1000,
    gpus=1,
    # track_grad_norm=2,
    gradient_clip_val=10
)
# experiment_config = ExperimentConfig(project_name="Tabular_test", log_logits=True)
optimizer_config = OptimizerConfig()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train, test=test)

result = tabular_model.evaluate(test)
print(result)
pred_df = tabular_model.predict(test)
pred_df.to_csv("temp_2.csv")
