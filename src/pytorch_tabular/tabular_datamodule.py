# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Data Module."""

import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pandas import DataFrame, DatetimeTZDtype, to_datetime
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.base import TransformerMixin, copy
from sklearn.preprocessing import (
    FunctionTransformer,
    LabelEncoder,
    PowerTransformer,
    QuantileTransformer,
    StandardScaler,
)
from torch.utils.data import DataLoader, Dataset

from pytorch_tabular.config import InferredConfig
from pytorch_tabular.utils import get_logger

from .categorical_encoders import OrdinalEncoder

logger = get_logger(__name__)


class TabularDataset(Dataset):
    def __init__(
        self,
        data: DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        target: List[str] = None,
    ):
        """Dataset to Load Tabular Data.

        Args:
            data (DataFrame): Pandas DataFrame to load during training
            task (str):
                Whether it is a classification or regression task. If classification, it returns a LongTensor as target
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
                These columns must be ordinal encoded beforehand. Defaults to None.
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.

        """

        self.task = task
        self.n = data.shape[0]
        self.target = target
        self.index = data.index
        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)  # .astype(np.int64)
        else:
            self.y = np.zeros((self.n, 1))  # .astype(np.int64)

        if task == "classification":
            self.y = self.y.astype(np.int64)
        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values

        if self.categorical_cols:
            self.categorical_X = data[categorical_cols]
            self.categorical_X = self.categorical_X.astype(np.int64).values

    @property
    def data(self):
        """Returns the data as a pandas dataframe."""
        if self.continuous_cols and self.categorical_cols:
            data = pd.DataFrame(
                np.concatenate([self.categorical_X, self.continuous_X], axis=1),
                columns=self.categorical_cols + self.continuous_cols,
                index=self.index,
            )
        elif self.continuous_cols:
            data = pd.DataFrame(self.continuous_X, columns=self.continuous_cols, index=self.index)
        elif self.categorical_cols:
            data = pd.DataFrame(self.categorical_X, columns=self.categorical_cols, index=self.index)
        else:
            data = pd.DataFrame()
        for i, t in enumerate(self.target):
            data[t] = self.y[:, i]
        return data

    def __len__(self):
        """Denotes the total number of samples."""
        return self.n

    def __getitem__(self, idx):
        """Generates one sample of data."""
        return {
            "target": self.y[idx],
            "continuous": (self.continuous_X[idx] if self.continuous_cols else torch.Tensor()),
            "categorical": (self.categorical_X[idx] if self.categorical_cols else torch.Tensor()),
        }


class TabularDatamodule(pl.LightningDataModule):
    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": QuantileTransformer,
            "params": {"output_distribution": "uniform", "random_state": None},
        },
        "quantile_normal": {
            "callable": QuantileTransformer,
            "params": {"output_distribution": "normal", "random_state": None},
        },
        "box-cox": {
            "callable": PowerTransformer,
            "params": {"method": "box-cox", "standardize": False},
        },
        "yeo-johnson": {
            "callable": PowerTransformer,
            "params": {"method": "yeo-johnson", "standardize": False},
        },
    }

    class CACHE_MODES(Enum):
        MEMORY = "memory"
        DISK = "disk"
        INFERENCE = "inference"

    def __init__(
        self,
        train: DataFrame,
        config: DictConfig,
        validation: DataFrame = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = 42,
        cache_data: str = "memory",
        copy_data: bool = True,
        verbose: bool = True,
    ):
        """The Pytorch Lightning Datamodule for Tabular Data.

        Args:
            train (DataFrame): The Training Dataframe

            config (DictConfig): Merged configuration object from ModelConfig, DataConfig,
                TrainerConfig, OptimizerConfig & ExperimentConfig

            validation (DataFrame, optional): Validation Dataframe.
                If left empty, we use the validation split from DataConfig to split a random sample as validation.
                Defaults to None.

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional):
                If provided, applies the transform to the target before modelling and inverse the transform during
                prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
                a tuple of callables (transform_func, inverse_transform_func)
                Defaults to None.

            train_sampler (Optional[torch.utils.data.Sampler], optional):
                If provided, the sampler will be used to sample the train data. Defaults to None.

            seed (Optional[int], optional): Seed to use for reproducible dataloaders. Defaults to 42.

            cache_data (str): Decides how to cache the data in the dataloader. If set to
                "memory", will cache in memory. If set to a valid path, will cache in that path. Defaults to "memory".

            copy_data (bool): If True, will copy the dataframes before preprocessing. Defaults to True.

            verbose (bool): Sets the verbosity of the databodule logging

        """
        super().__init__()
        self.train = train.copy() if copy_data else train
        if validation is not None:
            self.validation = validation.copy() if copy_data else validation
        else:
            self.validation = None
        self._set_target_transform(target_transform)
        self.target = config.target or []
        self.batch_size = config.batch_size
        self.train_sampler = train_sampler
        self.config = config
        self.seed = seed
        self.verbose = verbose
        self._fitted = False
        self._setup_cache(cache_data)
        self._inferred_config = self._update_config(config)

    @property
    def categorical_encoder(self):
        """Returns the categorical encoder."""
        return getattr(self, "_categorical_encoder", None)

    @categorical_encoder.setter
    def categorical_encoder(self, value):
        self._categorical_encoder = value

    @property
    def continuous_transform(self):
        """Returns the continuous transform."""
        return getattr(self, "_continuous_transform", None)

    @continuous_transform.setter
    def continuous_transform(self, value):
        self._continuous_transform = value

    @property
    def scaler(self):
        """Returns the scaler."""
        return getattr(self, "_scaler", None)

    @scaler.setter
    def scaler(self, value):
        self._scaler = value

    @property
    def label_encoder(self):
        """Returns the label encoder."""
        return getattr(self, "_label_encoder", None)

    @label_encoder.setter
    def label_encoder(self, value):
        self._label_encoder = value

    @property
    def target_transforms(self):
        """Returns the target transforms."""
        if self.do_target_transform:
            return self._target_transforms
        else:
            return None

    @target_transforms.setter
    def target_transforms(self, value):
        self._target_transforms = value

    def _setup_cache(self, cache_data: Union[str, bool]) -> None:
        cache_data = cache_data.lower()
        if cache_data == self.CACHE_MODES.MEMORY.value:
            self.cache_mode = self.CACHE_MODES.MEMORY
        elif isinstance(cache_data, str):
            self.cache_mode = self.CACHE_MODES.DISK
            self.cache_dir = Path(cache_data)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            logger.warning(f"{cache_data} is not a valid path. Caching in memory")
            self.cache_mode = self.CACHE_MODES.MEMORY

    def _set_target_transform(self, target_transform: Union[TransformerMixin, Tuple]) -> None:
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                target_transform = FunctionTransformer(func=target_transform[0], inverse_func=target_transform[1])
            self.do_target_transform = True
        else:
            self.do_target_transform = False
        self.target_transform_template = target_transform

    def _update_config(self, config) -> InferredConfig:
        """Calculates and updates a few key information to the config object.

        Args:
            config (DictConfig): The config object

        Returns:
            InferredConfig: The updated config object

        """
        categorical_dim = len(config.categorical_cols)
        continuous_dim = len(config.continuous_cols)
        # self._output_dim_clf = len(np.unique(self.train_dataset.y)) if config.target else None
        # self._output_dim_reg = len(config.target) if config.target else None
        if config.task == "regression":
            # self._output_dim_reg = len(config.target) if config.target else None if self.train is not None:
            output_dim = len(config.target) if config.target else None
            output_cardinality = None
        elif config.task == "classification":
            # self._output_dim_clf = len(np.unique(self.train_dataset.y)) if config.target else None
            if self.train is not None:
                output_cardinality = (
                    self.train[config.target].fillna("NA").nunique().tolist() if config.target else None
                )
                output_dim = sum(output_cardinality)
            else:
                output_cardinality = (
                    self.train_dataset.data[config.target].fillna("NA").nunique().tolist() if config.target else None
                )
                output_dim = sum(output_cardinality)
        elif config.task == "ssl":
            output_cardinality = None
            output_dim = None
        else:
            raise ValueError(f"{config.task} is an unsupported task.")
        if self.train is not None:
            category_cols = self.train[config.categorical_cols].select_dtypes(include="category").columns
            self.train[category_cols] = self.train[category_cols].astype("object")
            categorical_cardinality = [
                int(x) + 1 for x in list(self.train[config.categorical_cols].fillna("NA").nunique().values)
            ]
        else:
            category_cols = self.train_dataset.data[config.categorical_cols].select_dtypes(include="category").columns
            self.train_dataset.data[category_cols] = self.train_dataset.data[category_cols].astype("object")
            categorical_cardinality = [
                int(x) + 1 for x in list(self.train_dataset.data[config.categorical_cols].nunique().values)
            ]
        if getattr(config, "embedding_dims", None) is not None:
            embedding_dims = config.embedding_dims
        else:
            embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_cardinality]
        return InferredConfig(
            categorical_dim=categorical_dim,
            continuous_dim=continuous_dim,
            output_dim=output_dim,
            output_cardinality=output_cardinality,
            categorical_cardinality=categorical_cardinality,
            embedding_dims=embedding_dims,
        )

    def update_config(self, config) -> InferredConfig:
        """Calculates and updates a few key information to the config object. Logic happens in _update_config. This is
        just a wrapper to make it accessible from outside and not break current apis.

        Args:
            config (DictConfig): The config object

        Returns:
            InferredConfig: The updated config object

        """
        if self.cache_mode is self.CACHE_MODES.INFERENCE:
            warnings.warn("Cannot update config in inference mode. Returning the cached config")
            return self._inferred_config
        else:
            return self._update_config(config)

    def _encode_date_columns(self, data: DataFrame) -> DataFrame:
        added_features = []
        for field_name, freq, format in self.config.date_columns:
            data = self.make_date(data, field_name, format)
            data, _new_feats = self.add_datepart(data, field_name, frequency=freq, prefix=None, drop=True)
            added_features += _new_feats
        return data, added_features

    def _encode_categorical_columns(self, data: DataFrame, stage: str) -> DataFrame:
        if stage != "fit":
            # Inference
            return self.categorical_encoder.transform(data)
        # Fit
        logger.debug("Encoding Categorical Columns using OrdinalEncoder")
        self.categorical_encoder = OrdinalEncoder(
            cols=self.config.categorical_cols,
            handle_unseen=("impute" if self.config.handle_unknown_categories else "error"),
            handle_missing="impute" if self.config.handle_missing_values else "error",
        )
        data = self.categorical_encoder.fit_transform(data)
        return data

    def _transform_continuous_columns(self, data: DataFrame, stage: str) -> DataFrame:
        if stage == "fit":
            transform = self.CONTINUOUS_TRANSFORMS[self.config.continuous_feature_transform]
            if "random_state" in transform["params"] and self.seed is not None:
                transform["params"]["random_state"] = self.seed
            self.continuous_transform = transform["callable"](**transform["params"])
            # can be accessed through property continuous_transform
            data.loc[:, self.config.continuous_cols] = self.continuous_transform.fit_transform(
                data.loc[:, self.config.continuous_cols]
            )
        else:
            data.loc[:, self.config.continuous_cols] = self.continuous_transform.transform(
                data.loc[:, self.config.continuous_cols]
            )
        return data

    def _normalize_continuous_columns(self, data: DataFrame, stage: str) -> DataFrame:
        if stage == "fit":
            self.scaler = StandardScaler()
            data.loc[:, self.config.continuous_cols] = self.scaler.fit_transform(
                data.loc[:, self.config.continuous_cols]
            )
        else:
            data.loc[:, self.config.continuous_cols] = self.scaler.transform(data.loc[:, self.config.continuous_cols])
        return data

    def _label_encode_target(self, data: DataFrame, stage: str) -> DataFrame:
        if self.config.task != "classification":
            return data
        if stage == "fit" or self.label_encoder is None:
            self.label_encoder = [None] * len(self.config.target)
            for i in range(len(self.config.target)):
                self.label_encoder[i] = LabelEncoder()
                data[self.config.target[i]] = self.label_encoder[i].fit_transform(data[self.config.target[i]])
        else:
            for i in range(len(self.config.target)):
                if self.config.target[i] in data.columns:
                    data[self.config.target[i]] = self.label_encoder[i].transform(data[self.config.target[i]])
        return data

    def _target_transform(self, data: DataFrame, stage: str) -> DataFrame:
        if self.config.task != "regression":
            return data
        # target transform only for regression
        if not all(col in data.columns for col in self.config.target):
            return data
        if self.do_target_transform:
            if stage == "fit" or self.target_transforms is None:
                target_transforms = []
                for col in self.config.target:
                    _target_transform = copy.deepcopy(self.target_transform_template)
                    data[col] = _target_transform.fit_transform(data[col].values.reshape(-1, 1))
                    target_transforms.append(_target_transform)
                self.target_transforms = target_transforms
            else:
                for col, _target_transform in zip(self.config.target, self.target_transforms):
                    data[col] = _target_transform.transform(data[col].values.reshape(-1, 1))
        return data

    def preprocess_data(self, data: DataFrame, stage: str = "inference") -> Tuple[DataFrame, list]:
        """The preprocessing, like Categorical Encoding, Normalization, etc. which any dataframe should undergo before
        feeding into the dataloder.

        Args:
            data (DataFrame): A dataframe with the features and target
            stage (str, optional): Internal parameter. Used to distinguisj between fit and inference.
                Defaults to "inference".

        Returns:
            Returns the processed dataframe and the added features(list) as a tuple

        """
        added_features = None
        if self.config.encode_date_columns:
            data, added_features = self._encode_date_columns(data)
        # The only features that are added are the date features extracted
        # from the date which are categorical in nature
        if (added_features is not None) and (stage == "fit"):
            logger.debug(f"Added {added_features} features after encoding the date_columns")
            self.config.categorical_cols += added_features
            # Update the categorical dimension in config
            self.config.categorical_dim = (
                len(self.config.categorical_cols) if self.config.categorical_cols is not None else 0
            )
            self._inferred_config = self._update_config(self.config)
        # Encoding Categorical Columns
        if len(self.config.categorical_cols) > 0:
            data = self._encode_categorical_columns(data, stage)

        # Transforming Continuous Columns
        if (self.config.continuous_feature_transform is not None) and (len(self.config.continuous_cols) > 0):
            data = self._transform_continuous_columns(data, stage)
        # Normalizing Continuous Columns
        if (self.config.normalize_continuous_features) and (len(self.config.continuous_cols) > 0):
            data = self._normalize_continuous_columns(data, stage)
        # Converting target labels to a 0 indexed label
        data = self._label_encode_target(data, stage)
        # Target Transforms
        data = self._target_transform(data, stage)
        return data, added_features

    def _cache_dataset(self):
        train_dataset = TabularDataset(
            task=self.config.task,
            data=self.train,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=self.target,
        )
        self.train = None

        validation_dataset = TabularDataset(
            task=self.config.task,
            data=self.validation,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=self.target,
        )
        self.validation = None

        if self.cache_mode is self.CACHE_MODES.DISK:
            torch.save(train_dataset, self.cache_dir / "train_dataset", pickle_protocol=self.config.pickle_protocol)
            torch.save(
                validation_dataset, self.cache_dir / "validation_dataset", pickle_protocol=self.config.pickle_protocol
            )
        elif self.cache_mode is self.CACHE_MODES.MEMORY:
            self.train_dataset = train_dataset
            self.validation_dataset = validation_dataset
        elif self.cache_mode is self.CACHE_MODES.INFERENCE:
            self.train_dataset = None
            self.validation_dataset = None
        else:
            raise ValueError(f"{self.cache_mode} is not a valid cache mode")

    def split_train_val(self, train):
        logger.debug(
            "No validation data provided." f" Using {self.config.validation_split*100}% of train data as validation"
        )
        val_idx = train.sample(
            int(self.config.validation_split * len(train)),
            random_state=self.seed,
        ).index
        validation = train[train.index.isin(val_idx)]
        train = train[~train.index.isin(val_idx)]
        return train, validation

    def setup(self, stage: Optional[str] = None) -> None:
        """Data Operations you want to perform on all GPUs, like train-test split, transformations, etc. This is called
        before accessing the dataloaders.

        Args:
            stage (Optional[str], optional):
                Internal parameter to distinguish between fit and inference. Defaults to None.

        """
        if not (stage is None or stage == "fit" or stage == "ssl_finetune"):
            return
        if self.verbose:
            logger.info(f"Setting up the datamodule for {self.config.task} task")
        is_ssl = stage == "ssl_finetune"
        if self.validation is None:
            self.train, self.validation = self.split_train_val(self.train)
        else:
            self.validation = self.validation.copy()
        # Preprocessing Train, Validation
        self.train, _ = self.preprocess_data(self.train, stage="inference" if is_ssl else "fit")
        self.validation, _ = self.preprocess_data(self.validation, stage="inference")
        self._fitted = True
        self._cache_dataset()

    def inference_only_copy(self):
        """Creates a copy of the datamodule with the train and validation datasets removed. This is useful for
        inference only scenarios where we don't want to save the train and validation datasets.

        Returns:
            TabularDatamodule: A copy of the datamodule with the train and validation datasets removed.

        """
        if not self._fitted:
            raise RuntimeError("Can create an inference only copy only after model is fitted")
        dm_inference = copy.copy(self)
        dm_inference.train_dataset = None
        dm_inference.validation_dataset = None
        dm_inference.cache_mode = self.CACHE_MODES.INFERENCE
        return dm_inference

    # adapted from gluonts
    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """Returns a list of time features that will be appropriate for the given frequency string.

        Args:
            freq_str (str): Frequency string of the form `[multiple][granularity]` such as "12H", "5min", "1D" etc.

        Returns:
            List of added features

        """

        features_by_offsets = {
            offsets.YearBegin: [],
            offsets.YearEnd: [],
            offsets.MonthBegin: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.MonthEnd: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
            ],
            offsets.Week: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "Week",
            ],
            offsets.Day: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "WeekDay",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.BusinessDay: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "WeekDay",
                "Dayofweek",
                "Dayofyear",
            ],
            offsets.Hour: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "WeekDay",
                "Dayofweek",
                "Dayofyear",
                "Hour",
            ],
            offsets.Minute: [
                "Month",
                "Quarter",
                "Is_quarter_end",
                "Is_quarter_start",
                "Is_year_end",
                "Is_year_start",
                "Is_month_start",
                "WeekDay",
                "Dayofweek",
                "Dayofyear",
                "Hour",
                "Minute",
            ],
        }

        offset = to_offset(freq_str)

        for offset_type, feature in features_by_offsets.items():
            if isinstance(offset, offset_type):
                return feature

        supported_freq_msg = f"""
        Unsupported frequency {freq_str}

        The following frequencies are supported:

            Y, YS   - yearly
                alias: A
            M, MS   - monthly
            W   - weekly
            D   - daily
            B   - business days
            H   - hourly
            T   - minutely
                alias: min
        """
        raise RuntimeError(supported_freq_msg)

    # adapted from fastai
    @classmethod
    def make_date(cls, df: DataFrame, date_field: str, date_format: str = "ISO8601") -> DataFrame:
        """Make sure `df[date_field]` is of the right date type.

        Args:
            df (DataFrame): Dataframe

            date_field (str): Date field name

        Returns:
            Dataframe with date field converted to datetime

        """
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = to_datetime(df[date_field], format=date_format)
        return df

    # adapted from fastai
    @classmethod
    def add_datepart(
        cls,
        df: DataFrame,
        field_name: str,
        frequency: str,
        prefix: str = None,
        drop: bool = True,
    ) -> Tuple[DataFrame, List[str]]:
        """Helper function that adds columns relevant to a date in the column `field_name` of `df`.

        Args:
            df (DataFrame): Dataframe

            field_name (str): Date field name

            frequency (str): Frequency string of the form `[multiple][granularity]` such as "12H", "5min", "1D" etc.

            prefix (str, optional): Prefix to add to the new columns. Defaults to None.

            drop (bool, optional): Drop the original column. Defaults to True.

        Returns:
            Dataframe with added columns and list of added columns

        """
        field = df[field_name]
        prefix = (re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix) + "_"
        attr = cls.time_features_from_frequency_str(frequency)
        added_features = []
        for n in attr:
            if n == "Week":
                continue
            df[prefix + n] = getattr(field.dt, n.lower())
            added_features.append(prefix + n)
        # Pandas removed `dt.week` in v1.1.10
        if "Week" in attr:
            week = field.dt.isocalendar().week if hasattr(field.dt, "isocalendar") else field.dt.week
            df.insert(3, prefix + "Week", week)
            added_features.append(prefix + "Week")
        # TODO Not adding Elapsed by default. Need to route it through config
        # mask = ~field.isna()
        # df[prefix + "Elapsed"] = np.where(
        #     mask, field.values.astype(np.int64) // 10 ** 9, None
        # )
        # added_features.append(prefix + "Elapsed")
        if drop:
            df.drop(field_name, axis=1, inplace=True)

        # Removing features woth zero variations
        # for col in added_features:
        #     if len(df[col].unique()) == 1:
        #         df.drop(columns=col, inplace=True)
        #         added_features.remove(col)
        return df, added_features

    def _load_dataset_from_cache(self, tag: str = "train"):
        if self.cache_mode is self.CACHE_MODES.MEMORY:
            try:
                dataset = getattr(self, f"_{tag}_dataset")
            except AttributeError:
                raise AttributeError(
                    f"{tag}_dataset not found in memory. Please provide the data for" f" {tag} dataloader"
                )
        elif self.cache_mode is self.CACHE_MODES.DISK:
            try:
                dataset = torch.load(self.cache_dir / f"{tag}_dataset", weights_only=False)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"{tag}_dataset not found in {self.cache_dir}. Please provide the" f" data for {tag} dataloader"
                )
        elif self.cache_mode is self.CACHE_MODES.INFERENCE:
            raise RuntimeError("Cannot load dataset in inference mode. Use" " `prepare_inference_dataloader` instead")
        else:
            raise ValueError(f"{self.cache_mode} is not a valid cache mode")
        return dataset

    @property
    def train_dataset(self) -> TabularDataset:
        """Returns the train dataset.

        Returns:
            TabularDataset: The train dataset

        """
        return self._load_dataset_from_cache("train")

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def validation_dataset(self) -> TabularDataset:
        """Returns the validation dataset.

        Returns:
            TabularDataset: The validation dataset

        """
        return self._load_dataset_from_cache("validation")

    @validation_dataset.setter
    def validation_dataset(self, value):
        self._validation_dataset = value

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the train set.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: Train dataloader

        """
        return DataLoader(
            self.train_dataset,
            batch_size or self.batch_size,
            shuffle=True if self.train_sampler is None else False,
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            pin_memory=self.config.pin_memory,
            **self.config.dataloader_kwargs,
        )

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the validation set.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: Validation dataloader

        """
        return DataLoader(
            self.validation_dataset,
            batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            **self.config.dataloader_kwargs,
        )

    def _prepare_inference_data(self, df: DataFrame) -> DataFrame:
        """Prepare data for inference."""
        # TODO Is the target encoding necessary?
        if len(set(self.target) - set(df.columns)) > 0:
            if self.config.task == "classification":
                for i in range(len(self.target)):
                    df.loc[:, self.target[i]] = np.array([self.label_encoder[i].classes_[0]] * len(df)).reshape(-1, 1)
            else:
                df.loc[:, self.target] = np.zeros((len(df), len(self.target)))
        df, _ = self.preprocess_data(df, stage="inference")
        return df

    def prepare_inference_dataloader(
        self, df: DataFrame, batch_size: Optional[int] = None, copy_df: bool = True
    ) -> DataLoader:
        """Function that prepares and loads the new data.

        Args:
            df (DataFrame): Dataframe with the features and target
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.
            copy_df (bool, optional): Whether to copy the dataframe before processing or not. Defaults to False.
        Returns:
            DataLoader: The dataloader for the passed in dataframe

        """
        if copy_df:
            df = df.copy()
        df = self._prepare_inference_data(df)
        dataset = TabularDataset(
            task=self.config.task,
            data=df,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            target=(self.target if all(col in df.columns for col in self.target) else None),
        )
        return DataLoader(
            dataset,
            batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            **self.config.dataloader_kwargs,
        )

    def save_dataloader(self, path: Union[str, Path]) -> None:
        """Saves the dataloader to a path.

        Args:
            path (Union[str, Path]): Path to save the dataloader

        """
        if isinstance(path, str):
            path = Path(path)
        joblib.dump(self, path)

    @classmethod
    def load_datamodule(cls, path: Union[str, Path]):
        """Loads a datamodule from a path.

        Args:
            path (Union[str, Path]): Path to the datamodule

        Returns:
            TabularDatamodule (TabularDatamodule): The datamodule loaded from the path

        """
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist.")
        datamodule = joblib.load(path)
        return datamodule

    def copy(
        self,
        train: DataFrame,
        validation: DataFrame = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = None,
        cache_data: str = None,
        copy_data: bool = None,
        verbose: bool = None,
        call_setup: bool = True,
        config_override: Optional[Dict] = {},
    ):
        if config_override is not None:
            for k, v in config_override.items():
                setattr(self.config, k, v)
        dm = TabularDatamodule(
            train=train,
            config=self.config,
            validation=validation,
            target_transform=target_transform or self.target_transforms,
            train_sampler=train_sampler or self.train_sampler,
            seed=seed or self.seed,
            cache_data=cache_data or self.cache_mode.value,
            copy_data=copy_data or True,
            verbose=verbose or self.verbose,
        )
        dm.categorical_encoder = self.categorical_encoder
        dm.continuous_transform = self.continuous_transform
        dm.scaler = self.scaler
        dm.label_encoder = self.label_encoder
        dm.target_transforms = self.target_transforms
        dm.setup(stage="ssl_finetune") if call_setup else None
        return dm
