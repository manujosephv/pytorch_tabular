import os
# os.chdir("..")
from sklearn.datasets import fetch_covtype, make_classification
from sklearn.model_selection import train_test_split
import random
import numpy as np
import pandas as pd
# import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score

def make_mixed_classification(n_samples, n_features, n_categories):
    X,y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42, n_informative=5)
    cat_cols = random.choices(list(range(X.shape[-1])),k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:,col] = pd.qcut(X[:,col], q=4).codes.astype(int)
    col_names = [] 
    num_col_names=[]
    cat_col_names=[]
    for i in range(X.shape[-1]):
        if i in cat_cols:
            col_names.append(f"cat_col_{i}")
            cat_col_names.append(f"cat_col_{i}")
        if i in num_cols:
            col_names.append(f"num_col_{i}")
            num_col_names.append(f"num_col_{i}")
    X = pd.DataFrame(X, columns=col_names)
    y = pd.Series(y, name="target")
    data = X.join(y)
    return data, cat_col_names, num_col_names

def print_metrics(y_true, y_pred, tag):
    if isinstance(y_true, pd.DataFrame) or isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.DataFrame) or isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    if y_true.ndim>1:
        y_true=y_true.ravel()
    if y_pred.ndim>1:
        y_pred=y_pred.ravel()
    val_acc = accuracy_score(y_true, y_pred)
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")

data, cat_col_names, num_col_names = make_mixed_classification(n_samples=10000, n_features=20, n_categories=4)
train, test = train_test_split(data, random_state=42)
train, val = train_test_split(train, random_state=42)

from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, NodeConfig, TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer

data_config = DataConfig(
    target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=10,
    gpus=1, #index of the GPU to use. 0, means CPU
    fast_dev_run=True
)
optimizer_config = OptimizerConfig()
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="4096-4096-512",  # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    learning_rate = 1e-3,
    metrics=["accuracy"]
)
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train, validation=val)
test.drop(columns=['target'], inplace=True)
pred_df = tabular_model.predict(test)

# tabular_model.fit(train=train, validation=val)
# tabular_model.fit(train=train, validation=val, max_epochs=5)
# tabular_model.fit(train=train, validation=val, max_epochs=5, reset=True)