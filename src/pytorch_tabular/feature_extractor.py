# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
from collections import defaultdict

import pandas as pd
from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin

from pytorch_tabular.models import NODEModel, TabNetModel
from pytorch_tabular.models.mixture_density import MDNModel

try:
    import cPickle as pickle
except ImportError:
    import pickle

import torch


class DeepFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, tabular_model, extract_keys=["backbone_features"], drop_original=True):
        """Initializes the Transformer and extracts the neural features.

        Args:
            tabular_model (TabularModel): The trained TabularModel object
            extract_keys (list, optional): The keys of the features to extract. Defaults to ["backbone_features"].
            drop_original (bool, optional): Whether to drop the original columns. Defaults to True.

        """
        assert not (
            isinstance(tabular_model.model, NODEModel)
            or isinstance(tabular_model.model, TabNetModel)
            or isinstance(tabular_model.model, MDNModel)
        ), "FeatureExtractor doesn't work for Mixture Density Networks, NODE Model, & Tabnet Model"
        self.tabular_model = tabular_model
        self.extract_keys = extract_keys
        self.drop_original = drop_original

    def fit(self, X, y=None):
        """Just for compatibility.

        Does not do anything

        """
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transforms the categorical columns specified to the trained neural features from the model.

        Args:
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
            y ([type], optional): Only for compatibility. Not used. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            pd.DataFrame: The encoded dataframe

        """

        X_encoded = X.copy(deep=True)
        orig_features = X_encoded.columns
        self.tabular_model.model.eval()
        inference_dataloader = self.tabular_model.datamodule.prepare_inference_dataloader(X_encoded)
        logits_predictions = defaultdict(list)
        for batch in track(inference_dataloader, description="Generating Features..."):
            for k, v in batch.items():
                if isinstance(v, list) and (len(v) == 0):
                    # Skipping empty list
                    continue
                batch[k] = v.to(self.tabular_model.model.device)
            if self.tabular_model.config.task == "ssl":
                ret_value = {"backbone_features": self.tabular_model.model.predict(batch, ret_model_output=True)}
            else:
                _, ret_value = self.tabular_model.model.predict(batch, ret_model_output=True)
            for k in self.extract_keys:
                if k in ret_value.keys():
                    logits_predictions[k].append(ret_value[k].detach().cpu())

        logits_dfs = []
        for k, v in logits_predictions.items():
            v = torch.cat(v, dim=0).numpy()
            if v.ndim == 1:
                v = v.reshape(-1, 1)
            if v.shape[-1] > 1:
                temp_df = pd.DataFrame({f"{k}_{i}": v[:, i] for i in range(v.shape[-1])})
            else:
                temp_df = pd.DataFrame({f"{k}": v[:, 0]})

            # Append the temp DataFrame to the list
            logits_dfs.append(temp_df)

        preds = pd.concat(logits_dfs, axis=1)
        X_encoded = pd.concat([X_encoded, preds], axis=1)

        if self.drop_original:
            X_encoded.drop(columns=orig_features, inplace=True)
        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Encode given columns of X based on the learned features.

        Args:
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
            y ([type], optional): Only for compatibility. Not used. Defaults to None.

        Returns:
            pd.DataFrame: The encoded dataframe

        """
        self.fit(X, y)
        return self.transform(X)

    def save_as_object_file(self, path):
        """Saves the feature extractor as a pickle file.

        Args:
            path (str): The path to save the file

        """
        if not self._mapping:
            raise ValueError("`fit` method must be called before `save_as_object_file`.")
        pickle.dump(self.__dict__, open(path, "wb"))

    def load_from_object_file(self, path):
        """Loads the feature extractor from a pickle file.

        Args:
            path (str): The path to load the file from

        """
        for k, v in pickle.load(open(path, "rb")).items():
            setattr(self, k, v)
