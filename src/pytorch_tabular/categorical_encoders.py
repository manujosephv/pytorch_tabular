# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Modified https://github.com/tcassou/mlencoders/blob/master/mlencoders/base_encoder.py to suit NN encoding
"""Category Encoders"""
from __future__ import absolute_import, division, print_function, unicode_literals

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pandas as pd
from rich.progress import track
from sklearn.base import BaseEstimator, TransformerMixin

from pytorch_tabular.utils import get_logger

logger = get_logger(__name__)
NAN_CATEGORY = 0


class BaseEncoder(object):
    def __init__(self, cols, handle_unseen, min_samples, imputed, handle_missing):
        self.cols = cols
        self.handle_unseen = handle_unseen
        self.handle_missing = handle_missing
        self.min_samples = max(1, min_samples)
        # In case of unseen value or not enough data to learn the mapping, we use this value for imputation
        self._imputed = imputed
        # dict {str: pandas.DataFrame} column name --> mapping from category (index of df) to value (column of df)
        self._mapping = {}

    def transform(self, X):
        """Transform categorical data based on mapping learnt at fitting time.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        if not self._mapping:
            raise ValueError("`fit` method must be called before `transform`.")
        assert all(c in X.columns for c in self.cols)
        if self.handle_missing == "error":
            assert (
                not X[self.cols].isnull().any().any()
            ), "`handle_missing` = `error` and missing values found in columns to encode."
        X_encoded = X.copy(deep=True)
        for col, mapping in self._mapping.items():
            X_encoded[col] = X_encoded[col].fillna(NAN_CATEGORY).map(mapping["value"])

            if self.handle_unseen == "impute":
                X_encoded[col].fillna(self._imputed, inplace=True)
            elif self.handle_unseen == "error":
                if np.unique(X_encoded[col]).shape[0] > mapping.shape[0]:
                    raise ValueError("Unseen categories found in `{}` column.".format(col))

        return X_encoded

    def fit_transform(self, X, y=None):
        """Encode given columns of X according to y, and transform X based on the learnt mapping.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :param pandas.Series y: pandas Series of target values, shape (n_samples,).
            Required only for encoders that need it: TargetEncoder, WeightOfEvidenceEncoder
        :return: encoded DataFrame of shape (n_samples, n_features), initial categorical columns are dropped, and
            replaced with encoded columns. DataFrame passed in argument is unchanged.
        :rtype: pandas.DataFrame
        """
        self.fit(X, y)
        return self.transform(X)

    def _input_check(self, name, value, options):
        if value not in options:
            raise ValueError("Wrong input: {} parameter must be in {}".format(name, options))

    def _before_fit_check(self, X, y):
        # Checking columns to encode
        if self.cols is None:
            self.cols = X.columns
        else:
            assert all(c in X.columns for c in self.cols)
        # Checking input, depending on encoder type
        assert self.__class__.__name__ == "OrdinalEncoder" or y is not None
        if y is not None:
            assert X.shape[0] == y.shape[0]

    def save_as_object_file(self, path):
        if not self._mapping:
            raise ValueError("`fit` method must be called before `save_as_object_file`.")
        pickle.dump(self.__dict__, open(path, "wb"))

    def load_from_object_file(self, path):
        for k, v in pickle.load(open(path, "rb")).items():
            setattr(self, k, v)


class OrdinalEncoder(BaseEncoder):
    """
    Target Encoder for categorical features.
    """

    def __init__(self, cols=None, handle_unseen="impute", handle_missing="impute"):
        """Instantiation
        :param [str] cols: list of columns to encode, or None (then all dataset columns will be encoded at fitting time)
        :param str handle_unseen:
            'error'  - raise an error if a category unseen at fitting time is found
            'ignore' - skip unseen categories
            'impute' - impute new categories to a predefined value, which is same as NAN_CATEGORY
        :return: None
        """
        self._input_check("handle_unseen", handle_unseen, ["error", "ignore", "impute"])
        self._input_check("handle_missing", handle_missing, ["error", "impute"])
        super(OrdinalEncoder, self).__init__(cols, handle_unseen, 1, NAN_CATEGORY, handle_missing)

    def fit(self, X, y=None):
        """Label Encode given columns of X.
        :param pandas.DataFrame X: DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
        :return: None
        """
        self._before_fit_check(X, y)
        if self.handle_missing == "error":
            assert (
                not X[self.cols].isnull().any().any()
            ), "`handle_missing` = `error` and missing values found in columns to encode."
        for col in self.cols:
            map = (
                pd.Series(pd.unique(X[col].fillna(NAN_CATEGORY)), name=col)
                .reset_index()
                .rename(columns={"index": "value"})
            )
            map["value"] += 1
            self._mapping[col] = map.set_index(col)


class CategoricalEmbeddingTransformer(BaseEstimator, TransformerMixin):

    NAN_CATEGORY = 0

    def __init__(self, tabular_model):
        """Initializes the Transformer and extracts the neural embeddings

        Args:
            tabular_model (TabularModel): The trained TabularModel object
        """
        self._categorical_encoder = tabular_model.datamodule.categorical_encoder
        self.cols = tabular_model.model.hparams.categorical_cols
        # dict {str: np.ndarray} column name --> mapping from category (index of df) to value (column of df)
        self._mapping = {}

        self._extract_embedding(tabular_model.model)

    def _extract_embedding(self, model):
        try:
            embedding_layer = model.extract_embedding()
        except ValueError as e:
            logger.error(
                f"Extracting embedding layer from model received this error: {e}. Some models do not support this feature."
            )
            embedding_layer = None
        if embedding_layer is not None:
            for i, col in enumerate(self.cols):
                self._mapping[col] = {}
                embedding = embedding_layer[i]
                self._mapping[col][self.NAN_CATEGORY] = embedding.weight[0, :].detach().cpu().numpy().ravel()
                for key in self._categorical_encoder._mapping[col].index:
                    self._mapping[col][key] = (
                        embedding.weight[self._categorical_encoder._mapping[col].loc[key], :]
                        .detach()
                        .cpu()
                        .numpy()
                        .ravel()
                    )
        else:
            raise ValueError("Passed model doesn't support this feature.")

    def fit(self, X, y=None):
        """Just for compatibility. Does not do anything"""
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transforms the categorical columns specified to the trained neural embedding from the model

        Args:
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
            y ([type], optional): Only for compatibility. Not used. Defaults to None.

        Raises:
            ValueError: [description]

        Returns:
            pd.DataFrame: The encoded dataframe
        """
        if not self._mapping:
            raise ValueError(
                "Passed model should either have an attribute `embeddng_layers` or a method `extract_embedding` defined for `transform`."
            )
        assert all(c in X.columns for c in self.cols)

        X_encoded = X.copy(deep=True)
        for col, mapping in track(
            self._mapping.items(),
            description="Encoding the data...",
            total=len(self._mapping.values()),
        ):
            for dim in range(mapping[self.NAN_CATEGORY].shape[0]):
                X_encoded.loc[:, f"{col}_embed_dim_{dim}"] = (
                    X_encoded[col].fillna(self.NAN_CATEGORY).map({k: v[dim] for k, v in mapping.items()})
                )
                # Filling unseen categories also with NAN Embedding
                X_encoded[f"{col}_embed_dim_{dim}"].fillna(mapping[self.NAN_CATEGORY][dim], inplace=True)
        X_encoded.drop(columns=self.cols, inplace=True)
        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Encode given columns of X based on the learned embedding.

        Args:
            X (pd.DataFrame): DataFrame of features, shape (n_samples, n_features). Must contain columns to encode.
            y ([type], optional): Only for compatibility. Not used. Defaults to None.

        Returns:
            pd.DataFrame: The encoded dataframe
        """
        self.fit(X, y)
        return self.transform(X)

    def save_as_object_file(self, path):
        if not self._mapping:
            raise ValueError("`fit` method must be called before `save_as_object_file`.")
        pickle.dump(self.__dict__, open(path, "wb"))

    def load_from_object_file(self, path):
        for k, v in pickle.load(open(path, "rb")).items():
            setattr(self, k, v)
