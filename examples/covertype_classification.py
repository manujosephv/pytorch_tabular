from pathlib import Path

import pandas as pd
import wget
from sklearn.model_selection import train_test_split

from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig
from pytorch_tabular.tabular_model import TabularModel

BASE_DIR = Path.home().joinpath("data")
datafile = BASE_DIR.joinpath("covtype.data.gz")
datafile.parent.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
if not datafile.exists():
    wget.download(url, datafile.as_posix())

target_name = ["Covertype"]

cat_col_names = [
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40",
]

num_col_names = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

feature_columns = num_col_names + cat_col_names + target_name

df = pd.read_csv(datafile, header=None, names=feature_columns)
train, test = train_test_split(df, random_state=42)
train, val = train_test_split(train, random_state=42)
num_classes = len(set(train[target_name].values.ravel()))

data_config = DataConfig(
    target=target_name,
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform=None,  # "quantile_normal",
    normalize_continuous_features=True,
)
head_config = LinearHeadConfig(
    layers="",
    dropout=0.1,
    initialization="kaiming",  # No additional layer in head, just a mapping layer to output_dim
).__dict__  # Convert to dict to pass to the model config (OmegaConf doesn't accept objects)
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    metrics=["f1_score", "accuracy"],
    metrics_params=[{"num_classes": num_classes}, {}],
    metrics_prob_input=[True, False],
)
trainer_config = TrainerConfig(auto_lr_find=True, fast_dev_run=False, max_epochs=5, batch_size=512)
optimizer_config = OptimizerConfig()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(
    train=train,
    validation=val,
)

pred_df = tabular_model.predict(test)
print(pred_df.head())
tabular_model.save_model("examples/test_save")
