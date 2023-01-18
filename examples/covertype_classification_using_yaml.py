from pathlib import Path

import pandas as pd
import wget
from sklearn.model_selection import train_test_split

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

gate_lite = TabularModel(
    data_config="examples/yaml_config/data_config.yml",
    model_config="examples/yaml_config/gate_lite_model_config.yml",
    optimizer_config="examples/yaml_config/optimizer_config.yml",
    trainer_config="examples/yaml_config/trainer_config.yml",
)

datamodule = gate_lite.prepare_dataloader(train=train, validation=val, seed=42)
model = gate_lite.prepare_model(datamodule)

gate_lite.train(model, datamodule)

pred_df = gate_lite.predict(test, include_input_features=False)
pred_df["Model"] = "Gate Lite"
print(pred_df.head())

gate_full = TabularModel(
    data_config="examples/yaml_config/data_config.yml",
    model_config="examples/yaml_config/gate_full_model_config.yml",
    optimizer_config="examples/yaml_config/optimizer_config.yml",
    trainer_config="examples/yaml_config/trainer_config.yml",
)
gate_full_model = gate_full.prepare_model(datamodule)
gate_full.train(gate_full_model, datamodule)


pred_df_ = gate_lite.predict(test, include_input_features=False)
pred_df_["Model"] = "Gate Full"
pred_df = pd.concat([pred_df, pred_df_])
print(pred_df_.head())
del pred_df_
