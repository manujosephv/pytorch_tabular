import random

import pandas as pd

# os.chdir("..")
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def make_mixed_classification(n_samples, n_features, n_categories):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42, n_informative=5)
    cat_cols = random.choices(list(range(X.shape[-1])), k=n_categories)
    num_cols = [i for i in range(X.shape[-1]) if i not in cat_cols]
    for col in cat_cols:
        X[:, col] = pd.qcut(X[:, col], q=4).codes.astype(int)
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
    y = pd.Series(y, name="target")
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
    val_f1 = f1_score(y_true, y_pred)
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")


data, cat_col_names, num_col_names = make_mixed_classification(n_samples=10000, n_features=20, n_categories=4)
train, test = train_test_split(data, random_state=42)
train, val = train_test_split(train, random_state=42)

from pytorch_tabular import TabularModel  # noqa: E402
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig  # noqa: E402
from pytorch_tabular.models import FTTransformerConfig as model_config  # noqa: E402

data_config = DataConfig(
    # target should always be a list. Multi-targets are only supported for regression.
    # Multi-Task Classification is not implemented
    # target=["target"],
    target=["num_col_2"],
    # continuous_cols=num_col_names,
    continuous_cols=[c for c in num_col_names if c != "num_col_2"],
    categorical_cols=cat_col_names,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True,
)
trainer_config = TrainerConfig(
    auto_lr_find=False,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=32,
    max_epochs=2,
    fast_dev_run=True,
    # profiler="simple",
    early_stopping=None,
    checkpoints=None,
    # trainer_kwargs={"limit_train_batches": 10},
)
optimizer_config = OptimizerConfig()
model_config = model_config(
    # num_layers=1,
    # num_trees=2,
    # depth=2,
    input_embed_dim=4,
    num_heads=1,
    num_attn_blocks=1,
    # gflu_stages=2,
    # tree_depth=2,
    # num_trees=5,
    # task="classification",
    task="regression",
    learning_rate=1e-3,
    # metrics=["auroc"],
    metrics_prob_input=[True],
)
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)

tabular_model.fit(train=train, validation=val)
# tabular_model.save_model("saved_models/gandalftree")
# tabular_model = TabularModel.load_model("saved_models/gandalftree")
# calculate runtime
import time  # noqa: E402

print("Test Shape: ", test.shape)
for method in [
    "GradientShap",
    "IntegratedGradients",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "FeaturePermutation",
    "FeatureAblation",
    "KernelShap",
]:
    start = time.time()
    # try:
    exp = tabular_model.explain(test.head(1), method=method, baselines="b|100")
    assert exp.shape[1] == tabular_model.model.hparams.continuous_dim + tabular_model.model.hparams.categorical_dim
    # except AssertionError:
    #     pass
    end = time.time()
    print(f"{method} Time: ", end - start)

    # print(exp)
# test.drop(columns=["target"], inplace=True)
# pred_df = tabular_model.predict(test)
# pred_df = tabular_model.predict(test, device="cpu")
# pred_df = tabular_model.predict(test, device="cuda")
# import torch

# pred_df = tabular_model.predict(test, device=torch.device("cuda"))
# tabular_model.fit(train=train, validation=val)
# tabular_model.fit(train=train, validation=val, max_epochs=5)
# tabular_model.fit(train=train, validation=val, max_epochs=5, reset=True)
[
    "GradientShap",
    "IntegratedGradients",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "FeaturePermutation",
    "FeatureAblation",
    "KernelShap",
]
# TODO Check FeatureAblation, KernelShap - Error

# t = torch.rand(128,200)
# a = t.numpy()

# start = time.time()
# t.median(dim=-1)
# end = time.time()
# print("torch median", end - start)

# start = time.time()
# t.quantile(torch.rand(128), dim=-1)
# end = time.time()
# print("torch quant ", end - start)

# start = time.time()
# np.median(t.numpy(), axis=-1)
# end = time.time()
# print("numpy median", end - start)

# start = time.time()
# np.quantile(t.numpy(), np.random.rand(128), axis=-1)
# end = time.time()
# print("numpy quant ", end - start)

# start = time.time()
# st = torch.sort(t, dim=-1)
# end = time.time()
# print("torch sort", end - start)

# start = time.time()
# st = np.sort(t.numpy(), axis=-1)
# end = time.time()
# print("numpy sort", end - start)
