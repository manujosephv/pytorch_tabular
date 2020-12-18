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
import joblib


def read_style_df(n_seasons=4, periods=6):
    df = pd.read_csv(f"data\\style_df_{n_seasons}_seasons_{periods}_periods.csv")
    for col in [
        "intro_date_usa",
        "phaseout_date_usa",
        "intro_date_w_margin",
        "phaseout_date_w_margin",
    ]:
        df[col] = pd.to_datetime(df[col])
    df["year"] = df["intro_date_usa"].dt.year
    return df


def split_test_train(df, pct=0.2):
    test_size = int(pct * (df[df["year"] == 2020].shape[0]))
    test_idx = df[df.year == 2020].sample(test_size, random_state=42).index
    train = df[~df.index.isin(test_idx)].copy()
    test = df[df.index.isin(test_idx)].copy()
    return train, test


style_df = read_style_df(n_seasons=4, periods=6)
style_df = style_df.dropna()
style_df["days_since_intro"] = (
    pd.datetime.now() - style_df["intro_date_w_margin"]
).apply(lambda x: x.days)
style_df["years_since_intro"] = style_df["days_since_intro"] // (365.25)
style_df["mdl"] = style_df["mdl"].apply(lambda x: x[:-2])
style_df = style_df.merge(
    style_df.groupby("mdl")[["block_0", "retail_price_usa"]].mean().reset_index(),
    left_on="model_predecessor",
    right_on="mdl",
    suffixes=("", "_predecessor"),
    how="left",
)
style_df["line_sequence"] = (
    style_df["gender_model_plan_sequence"].str.split("-", expand=True).iloc[:, 1]
)
nb_color_map = joblib.load("data\color_map.dict")
style_df[["primary_r", "primary_g", "primary_b"]] = (
    style_df["primary_color"].map(nb_color_map).apply(pd.Series)
)
style_df[["secondary_r", "secondary_g", "secondary_b"]] = (
    style_df["secondary_color"].map(nb_color_map).apply(pd.Series)
)
gender_mapping = {
    "Grade Boys": "Boys",
    "Grade Girls": "Girls",
    "Infant Girls": "Infants",
    "Infant Boys": "Infants",
    "Boys": "Boys",
    "Mens": "Mens",
    "Pre Girls": "Girls",
    "Pre Boys": "Boys",
    "Womens": "Womens",
    "Girls": "Girls",
}
style_df["mapped_gender"] = style_df.gender.map(gender_mapping)
style_df["intro_date_month"] = style_df.intro_date_usa.dt.month
style_df["block_0_predecessor"] = style_df["block_0_predecessor"].fillna(
    style_df["block_0_predecessor"].mean()
)
style_df["retail_price_usa_predecessor"] = style_df[
    "retail_price_usa_predecessor"
].fillna(style_df["retail_price_usa_predecessor"].mean())

# mean_target = style_df["block_0"].mean()
# style_df["block_0"] = style_df["block_0"] / mean_target
train, test = split_test_train(style_df)

data_config = DataConfig(
    target=["block_0"],
    continuous_cols=[
        "retail_price_usa",
        "phaseout_days_0",
        "days_since_intro",
        "years_since_intro",
        "block_0_predecessor",
        "retail_price_usa_predecessor",
        "primary_r",
        "primary_g",
        "primary_b",
        "secondary_r",
        "secondary_g",
        "secondary_b",
        "length_of_life",
    ],
    categorical_cols=[
        "usa_stocking_policy",
        "usa_merch_class",
        "gbu",
        "line_plan_business",
        "model_series",
        "gender_model_number",
        "style_state",
        "closure_type",
        "segment",
        "mdl",
        "intro_period_qtr",
        "line_sequence",
        "mapped_gender",
        "npi_property",
        "map",
        "intro_date_month",
    ],
    normalize_continuous_features=True,
    continuous_feature_transform="quantile_normal",
    # date_cols=[('intro_date_usa','M')],
    # encode_date_cols=True
)
# model_config = CategoryEmbeddingModelConfig(task="regression", layers="64-32-16", activation="SELU")
model_config = NodeConfig(
    task="regression",
    num_layers=2,
    num_trees=1024,
    learning_rate=1,
    embed_categorical=True,
    metrics=["MeanSquaredLogError"],
    target_range=(train['block_0'].min().item(), train['block_0'].max().item())
)
# model_config = TabNetModelConfig(task="regression",metrics=["MeanSquaredLogError"])
trainer_config = TrainerConfig(
    auto_lr_find=True,
    # auto_scale_batch_size=True,
    batch_size=128,
    max_epochs=1000,
    gpus=1,
    # track_grad_norm=2,
    gradient_clip_val=10,
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
# pred_df["block_0_prediction"] = pred_df["block_0_prediction"] * mean_target
# pred_df["block_0"] = pred_df["block_0"] * mean_target
pred_df.to_csv("output/temp_3.csv")
