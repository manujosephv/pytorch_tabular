import random
from pathlib import Path

import numpy as np
import pandas as pd

# os.chdir("..")
from sklearn.datasets import make_classification

# import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def make_mixed_classification(n_samples, n_features, n_categories):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42, n_informative=5)
    cat_cols = random.choices(list(range(X.shape[-1])), k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    card_l = [2, 3, 5, 5]
    # for col in cat_cols:
    #     X[:, col] = pd.qcut(X[:, col], q=4).codes.astype(int)
    for card, col in zip(card_l, cat_cols):
        X[:, col] = pd.qcut(X[:, col], q=card).codes.astype(int)
    col_names = []
    num_col_names = []
    cat_col_names = []
    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y, name=target_name)
    data = X.join(y)
    return data, cat_col_names, num_col_names


def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred, average="macro" if len(np.unique(y_true)) > 2 else "binary")
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")


# data, cat_col_names, num_col_names = make_mixed_classification(
#     n_samples=500000, n_features=20, n_categories=4
# )
BASE_DIR = Path.home().joinpath("data")
datafile = BASE_DIR.joinpath("covtype.data.gz")
datafile.parent.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
if not datafile.exists():
    import wget

    wget.download(url, datafile.as_posix())

target_name = "Covertype"

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

feature_columns = num_col_names + cat_col_names + [target_name]

data = pd.read_csv(datafile, header=None, names=feature_columns)
data.dropna(inplace=True)

ssl, finetune = train_test_split(data, random_state=42)
ssl_train, ssl_val = train_test_split(ssl, random_state=42)
finetune_train, finetune_test = train_test_split(finetune, random_state=42)
finetune_train, finetune_val = train_test_split(finetune_train, random_state=42)

from pytorch_tabular import TabularModel  # noqa: E402
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig  # noqa: E402
from pytorch_tabular.models import CategoryEmbeddingModelConfig  # noqa: E402
from pytorch_tabular.ssl_models.dae import DenoisingAutoEncoderConfig  # noqa: E402

max_epochs = 1
batch_size = 1024
lr = 1e-3

data_config = DataConfig(
    # target should always be a list.
    target=[target_name],
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True,
    handle_missing_values=False,
    handle_unknown_categories=False,
)
trainer_config = TrainerConfig(
    auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=batch_size,
    max_epochs=max_epochs,
    fast_dev_run=True,
)
optimizer_config = OptimizerConfig()
encoder_config = CategoryEmbeddingModelConfig(
    task="backbone",
    layers="4096-2048-512",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
)

decoder_config = CategoryEmbeddingModelConfig(
    task="backbone",
    layers="512-2048-4096",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
)
# encoder_config = TabTransformerConfig(
#     task="backbone",
# )
dae_config = DenoisingAutoEncoderConfig(encoder_config=encoder_config, decoder_config=decoder_config, learning_rate=lr)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=dae_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

# tabular_model.fit(train=train, validation=val)
tabular_model.pretrain(train=ssl_train, validation=ssl_val)

ft_trainer_config = TrainerConfig(
    auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=batch_size,
    max_epochs=max_epochs,
    fast_dev_run=False,
)
ft_optimizer_config = OptimizerConfig(optimizer="SGD")
finetune_model = tabular_model.create_finetune_model(
    task="classification",
    # target=[
    #     target_name
    # ],
    head="LinearHead",
    head_config={
        "layers": "64-32-16",
        "activation": "LeakyReLU",
    },
    trainer_config=ft_trainer_config,
    optimizer_config=ft_optimizer_config,
)
# decoder=nn.Identity(),
finetune_model.finetune(train=finetune_train, validation=finetune_val, freeze_backbone=True)

tgt = finetune_test[target_name]
finetune_test.drop(columns=[target_name], inplace=True)
pred_df = finetune_model.predict(finetune_test)
print_metrics(tgt, pred_df["prediction"], "Test")


model_config = CategoryEmbeddingModelConfig(
    task="classification",
    head="LinearHead",
    head_config={
        "layers": "512-128-64-32-16",
        "activation": "LeakyReLU",
    },
    learning_rate=lr,
)

trainer_config = TrainerConfig(
    auto_select_gpus=True,
    fast_dev_run=True,
    max_epochs=max_epochs,
    batch_size=batch_size,
)
optimizer_config = OptimizerConfig()
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=finetune_train, validation=finetune_val)
pred_df = tabular_model.predict(finetune_test)
print_metrics(tgt, pred_df["prediction"], "Test")
