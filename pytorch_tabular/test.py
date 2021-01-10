# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import os
# os.chdir("..")
from sklearn.datasets import fetch_covtype
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import category_encoders as ce
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# # Utility Functions

# %%
def load_classification_data():
    dataset = fetch_covtype(data_home="data")
    data = np.hstack([dataset.data, dataset.target.reshape(-1, 1)])
    col_names = [f"feature_{i}" for i in range(data.shape[-1])]
    col_names[-1] = "target"
    data = pd.DataFrame(data, columns=col_names)
    data["feature_0_cat"] = pd.qcut(data["feature_0"], q=4)
    data["feature_0_cat"] = "feature_0_" + data.feature_0_cat.cat.codes.astype(str)
    test_idx = data.sample(int(0.2 * len(data)), random_state=42).index
    test = data[data.index.isin(test_idx)]
    train = data[~data.index.isin(test_idx)]
    return (train, test, ["target"])

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
    val_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{tag} Acc: {val_acc} | {tag} F1: {val_f1}")

# %% [markdown]
# # Load Forest Cover Data

# %%
train, test, target_col = load_classification_data()
train, val = train_test_split(train, random_state=42)


# %%
cat_col_names = ["feature_0_cat"]
num_col_names = [col for col in train.columns if col not in cat_col_names+target_col]


# %%
encoder = ce.OneHotEncoder(cols=cat_col_names)
train_transform = encoder.fit_transform(train)
val_transform = encoder.transform(val)
test_transform = encoder.transform(test)

# %% [markdown]
# ## Baseline
# 
# Let's use the default LightGBM model as a baseline.

# %%
# clf = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
# clf.fit(train_transform.drop(columns=target_col), train_transform[target_col].values.ravel())
# val_pred = clf.predict(val_transform.drop(columns=target_col))
# print_metrics(val_transform[target_col], val_pred, "Validation")
# test_pred = clf.predict(test_transform.drop(columns='target'))
# print_metrics(test_transform[target_col], test_pred, "Holdout")

# %% [markdown]
# ## CategoryEmbedding Model

# %%
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig, NodeConfig, TabNetModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer


# %%
data_config = DataConfig(
    target=target_col, #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
    continuous_feature_transform="quantile_normal",
    normalize_continuous_features=True
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=1000,
    gpus=1, #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="4096-4096-512",  # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    learning_rate = 1e-3,
    metrics=["accuracy", "f1"],
    metrics_params=[{},{"average":"micro"}]
)
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)


# %%
tabular_model.fit(train=train, test=test)


# %%
result = tabular_model.evaluate(test)
print(result)

# %% [markdown]
# To get the prediction as a dataframe, we can use the `predict` method. This will add predictions to the same dataframe that was passed in. For classification problems, we get both the probabilities and the final prediction taking 0.5 as the threshold

# %%
pred_df = tabular_model.predict(test)
pred_df.head()


# %%
print_metrics(test['target'], pred_df["prediction"], tag="Holdout")

# %% [markdown]
# ## Extract the Learned Embedding
# 
# For the models that support (CategoryEmbeddingModel and CategoryEmbeddingNODE), we can extract the learned embeddings into a sci-kit learn style Transformer. You can use this in your Sci-kit Learn pipelines and workflows as a drop in replacement.

# %%
transformer = CategoricalEmbeddingTransformer(tabular_model)
train_transform = transformer.fit_transform(train)
clf = lgb.LGBMClassifier(random_state=42)
clf.fit(train_transform.drop(columns='target'), train_transform['target'])


# %%
val_transform = transformer.transform(val)
val_pred = clf.predict(val_transform.drop(columns=target_col))
print_metrics(val_transform[target_col], val_pred, "Validation")
test_transform = transformer.transform(test)
test_pred = clf.predict(test_transform.drop(columns=target_col))
print_metrics(test_transform[target_col], test_pred, "Holdout")

# %% [markdown]
# 
# |Split|One-Hot Encoding|Neural Embedding|
# |--|--|--|
# |Validation Accuracy|85.28%|**85.61%**|
# |Validation F1|82.55%|**82.60%**|
# |Holdout Accuracy|85.17%|**85.55%**|
# |Holdout F1|81.75%|**82.33%**|
# 

