# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Data Module"""
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import category_encoders as ce
import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from sklearn.base import copy, TransformerMixin
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


class TabularDatamodule(pl.LightningDataModule):

    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="uniform", random_state=None),
        },
        "quantile_normal": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="normal", random_state=None),
        },
        "box-cox": {
            "callable": PowerTransformer,
            "params": dict(method="box-cox", standardize=False),
        },
        "yeo-johnson": {
            "callable": PowerTransformer,
            "params": dict(method="yeo-johnson", standardize=False),
        },
    }

    def __init__(
        self,
        train: pd.DataFrame,
        config: DictConfig,
        validation: pd.DataFrame = None,
        test: pd.DataFrame = None,
        target_transform: Optional[Union[TransformerMixin, Tuple]] = None,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        seed: Optional[int] = 42,
    ):
        """The Pytorch Lightning Datamodule for Tabular Data

        Args:
            train (pd.DataFrame): The Training Dataframe

            config (DictConfig): Merged configuration object from ModelConfig, DataConfig,
                TrainerConfig, OptimizerConfig & ExperimentConfig

            validation (pd.DataFrame, optional): Validation Dataframe.
                If left empty, we use the validation split from DataConfig to split a random sample as validation.
                Defaults to None.

            test (pd.DataFrame, optional): Holdout DataFrame to check final performance on.
                Defaults to None.

            target_transform (Optional[Union[TransformerMixin, Tuple(Callable)]], optional): If provided, applies the transform to the target before modelling
                and inverse the transform during prediction. The parameter can either be a sklearn Transformer which has an inverse_transform method, or
                a tuple of callables (transform_func, inverse_transform_func)
        """
        super().__init__()
        self.train = train.copy()
        self.validation = validation
        self._set_target_transform(target_transform)
        self.test = test if test is None else test.copy()
        self.target = config.target
        self.batch_size = config.batch_size
        self.train_sampler = train_sampler
        self.config = config
        self.seed = seed
        self._fitted = False

    def _set_target_transform(self, target_transform: Union[TransformerMixin, Tuple]) -> None:
        if target_transform is not None:
            if isinstance(target_transform, Iterable):
                target_transform = FunctionTransformer(func=target_transform[0], inverse_func=target_transform[1])
            self.do_target_transform = True
        else:
            self.do_target_transform = False
        self.target_transform_template = target_transform

    def update_config(self, config) -> InferredConfig:
        """Calculates and updates a few key information to the config object

        Args:
            config (DictConfig): The config object

        Returns:
            InferredConfig: The updated config object
        """
        categorical_dim = len(config.categorical_cols)
        continuous_dim = len(config.continuous_cols)
        if config.task == "regression":
            output_dim = len(config.target)
        elif config.task == "classification":
            output_dim = len(self.train[config.target[0]].unique())
        else:
            output_dim = None
        categorical_cardinality = None
        embedding_dims = None
        if not self.do_leave_one_out_encoder():
            categorical_cardinality = [
                int(self.train[col].fillna("NA").nunique()) + 1 for col in config.categorical_cols
            ]
            embedding_dims = [(x, min(50, (x + 1) // 2)) for x in categorical_cardinality]
            if hasattr(config, "embedding_dims"):
                if config.embedding_dims is not None:
                    embedding_dims = config.embedding_dims
        return InferredConfig(
            categorical_dim=categorical_dim,
            continuous_dim=continuous_dim,
            output_dim=output_dim,
            categorical_cardinality=categorical_cardinality,
            embedding_dims=embedding_dims,
        )

    def do_leave_one_out_encoder(self) -> bool:
        """Checks the special condition for NODE where we use a LeaveOneOutEncoder to encode categorical columns
        DEPRECATED: Automatically encoding categorical columns using LeaveOneOutEncoder is deprecated.

        Returns:
            bool: True if LeaveOneOutEncoder is used
        """
        if hasattr(self.config, "_model_name"):
            return (self.config._model_name == "NODEModel") and (not self.config.embed_categorical)
        else:
            return False

    def _encode_date_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        added_features = []
        for field_name, freq in self.config.date_columns:
            data = self.make_date(data, field_name)
            data, _new_feats = self.add_datepart(data, field_name, frequency=freq, prefix=None, drop=True)
            added_features += _new_feats
        return data, added_features

    def _encode_categorical_columns(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        if stage == "fit":
            if self.do_leave_one_out_encoder():
                logger.debug("Encoding Categorical Columns using LeavOneOutEncoder")
                self.categorical_encoder = ce.LeaveOneOutEncoder(
                    cols=self.config.categorical_cols, random_state=self.seed
                )
                # Multi-Target Regression uses the first target to encode the categorical columns
                if len(self.config.target) > 1:
                    logger.warning(
                        f"Multi-Target Regression: using the first target({self.config.target[0]}) to encode the categorical columns"
                    )
                data = self.categorical_encoder.fit_transform(data, data[self.config.target[0]])
            else:
                logger.debug("Encoding Categorical Columns using OrdinalEncoder")
                self.categorical_encoder = OrdinalEncoder(
                    cols=self.config.categorical_cols,
                    handle_unseen="impute" if self.config.handle_unknown_categories else "error",
                    handle_missing="impute" if self.config.handle_missing_values else "error",
                )
                data = self.categorical_encoder.fit_transform(data)
        else:
            data = self.categorical_encoder.transform(data)
        return data

    def _transform_continuous_columns(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        if stage == "fit":
            transform = self.CONTINUOUS_TRANSFORMS[self.config.continuous_feature_transform]
            if "random_state" in transform["params"] and self.seed is not None:
                transform["params"]["random_state"] = self.seed
            self.continuous_transform = transform["callable"](**transform["params"])
            # TODO implement quantile noise
            data.loc[:, self.config.continuous_cols] = self.continuous_transform.fit_transform(
                data.loc[:, self.config.continuous_cols]
            )
        else:
            data.loc[:, self.config.continuous_cols] = self.continuous_transform.transform(
                data.loc[:, self.config.continuous_cols]
            )
        return data

    def _normalize_continuous_columns(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        if stage == "fit":
            self.scaler = StandardScaler()
            data.loc[:, self.config.continuous_cols] = self.scaler.fit_transform(
                data.loc[:, self.config.continuous_cols]
            )
        else:
            data.loc[:, self.config.continuous_cols] = self.scaler.transform(data.loc[:, self.config.continuous_cols])
        return data

    def _label_encode_target(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        if self.config.task == "classification":
            if stage == "fit":
                self.label_encoder = LabelEncoder()
                data[self.config.target[0]] = self.label_encoder.fit_transform(data[self.config.target[0]])
            else:
                if self.config.target[0] in data.columns:
                    data[self.config.target[0]] = self.label_encoder.transform(data[self.config.target[0]])
        return data

    def _target_transform(self, data: pd.DataFrame, stage: str) -> pd.DataFrame:
        if self.config.task == "regression":
            # target transform only for regression
            if all([col in data.columns for col in self.config.target]):
                if self.do_target_transform:
                    if stage == "fit":
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

    def preprocess_data(self, data: pd.DataFrame, stage: str = "inference") -> Tuple[pd.DataFrame, list]:
        """The preprocessing, like Categorical Encoding, Normalization, etc. which any dataframe should undergo before feeding into the dataloder

        Args:
            data (pd.DataFrame): A dataframe with the features and target
            stage (str, optional): Internal parameter. Used to distinguisj between fit and inference. Defaults to "inference".

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

    def setup(self, stage: Optional[str] = None) -> None:
        """Data Operations you want to perform on all GPUs, like train-test split, transformations, etc.
        This is called before accessing the dataloaders

        Args:
            stage (Optional[str], optional): Internal parameter to distinguish between fit and inference. Defaults to None.
        """
        if stage == "fit" or stage is None:
            logger.info(f"Setting up the datamodule for {self.config.task} task")
            if self.validation is None:
                logger.debug(
                    f"No validation data provided. Using {self.config.validation_split*100}% of train data as validation"
                )
                val_idx = self.train.sample(
                    int(self.config.validation_split * len(self.train)),
                    random_state=self.seed,
                ).index
                self.validation = self.train[self.train.index.isin(val_idx)]
                self.train = self.train[~self.train.index.isin(val_idx)]
            else:
                self.validation = self.validation.copy()
            # Preprocessing Train, Validation
            self.train, _ = self.preprocess_data(self.train, stage="fit")
            self.validation, _ = self.preprocess_data(self.validation, stage="inference")
            if self.test is not None:
                self.test, _ = self.preprocess_data(self.test, stage="inference")
            self._fitted = True

    # adapted from gluonts
    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """
        Returns a list of time features that will be appropriate for the given frequency string.

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
                "Week" "Day",
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
                "Week" "Day",
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
                "Week" "Day",
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
                "Week" "Day",
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
    def make_date(cls, df: pd.DataFrame, date_field: str) -> pd.DataFrame:
        """Make sure `df[date_field]` is of the right date type.

        Args:
            df (pd.DataFrame): Dataframe

            date_field (str): Date field name

        Returns:
            Dataframe with date field converted to datetime
        """
        field_dtype = df[date_field].dtype
        if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            field_dtype = np.datetime64
        if not np.issubdtype(field_dtype, np.datetime64):
            df[date_field] = pd.to_datetime(df[date_field], infer_datetime_format=True)
        return df

    # adapted from fastai
    @classmethod
    def add_datepart(
        cls,
        df: pd.DataFrame,
        field_name: str,
        frequency: str,
        prefix: str = None,
        drop: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Helper function that adds columns relevant to a date in the column `field_name` of `df`.

        Args:
            df (pd.DataFrame): Dataframe

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

    def train_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the train set.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: Train dataloader
        """
        dataset = TabularDataset(
            task=self.config.task,
            data=self.train,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=True if self.train_sampler is None else False,
            num_workers=self.config.num_workers,
            sampler=self.train_sampler,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the validation set.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: Validation dataloader
        """
        dataset = TabularDataset(
            task=self.config.task,
            data=self.validation,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def test_dataloader(self, batch_size: Optional[int] = None) -> DataLoader:
        """Function that loads the validation set.

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: Test dataloader
        """
        if self.test is not None:
            dataset = TabularDataset(
                task=self.config.task,
                data=self.test,
                categorical_cols=self.config.categorical_cols,
                continuous_cols=self.config.continuous_cols,
                embed_categorical=(not self.do_leave_one_out_encoder()),
                target=self.target,
            )
            return DataLoader(
                dataset,
                batch_size if batch_size is not None else self.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )

    def _prepare_inference_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for inference."""
        # TODO Is the target encoding necessary?
        if len(set(self.target) - set(df.columns)) > 0:
            if self.config.task == "classification":
                df.loc[:, self.target] = np.array([self.label_encoder.classes_[0]] * len(df)).reshape(-1, 1)
            else:
                df.loc[:, self.target] = np.zeros((len(df), len(self.target)))
        df, _ = self.preprocess_data(df, stage="inference")
        return df

    def prepare_inference_dataloader(self, df: pd.DataFrame, batch_size: Optional[int] = None) -> DataLoader:
        """Function that prepares and loads the new data.

        Args:
            df (pd.DataFrame): Dataframe with the features and target
            batch_size (Optional[int], optional): Batch size. Defaults to `self.batch_size`.

        Returns:
            DataLoader: The dataloader for the passed in dataframe
        """
        df = df.copy()
        df = self._prepare_inference_data(df)
        dataset = TabularDataset(
            task=self.config.task,
            data=df,
            categorical_cols=self.config.categorical_cols,
            continuous_cols=self.config.continuous_cols,
            embed_categorical=(not self.do_leave_one_out_encoder()),
            target=self.target if all([col in df.columns for col in self.target]) else None,
        )
        return DataLoader(
            dataset,
            batch_size if batch_size is not None else self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
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


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        task: str,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        embed_categorical: bool = True,
        target: List[str] = None,
    ):
        """Dataset to Load Tabular Data

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training
            task (str): Whether it is a classification or regression task. If classification, it returns a LongTensor as target
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
            These columns must be ordinal encoded beforehand. Defaults to None.
            embed_categorical (bool): Flag to tell the dataset whether to convert categorical columns to LongTensor or retain as float.
            If we are going to embed categorical cols with an embedding layer, we need to convert the columns to LongTensor
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.
        """

        self.task = task
        self.n = data.shape[0]

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
            if embed_categorical:
                self.categorical_X = self.categorical_X.astype(np.int64).values
            else:
                self.categorical_X = self.categorical_X.astype(np.float32).values

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return {
            "target": self.y[idx],
            "continuous": self.continuous_X[idx] if self.continuous_cols else torch.Tensor(),
            "categorical": self.categorical_X[idx] if self.categorical_cols else torch.Tensor(),
        }
