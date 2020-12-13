from typing import Optional, List
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from .category_encoders import OrdinalEncoder
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
import re
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


class TabularDatamodule(pl.LightningDataModule):

    CONTINUOUS_TRANSFORMS = {
        "quantile_uniform": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="uniform", random_state=42),
        },
        "quantile_normal": {
            "callable": QuantileTransformer,
            "params": dict(output_distribution="normal", random_state=42),
        },
        "box-cox": {"callable": PowerTransformer, "params": dict(method="box-cox")},
        "yeo-johnson": {
            "callable": PowerTransformer,
            "params": dict(method="yeo-johnson"),
        },
    }

    def __init__(
        self,
        train: pd.DataFrame,
        config: DictConfig,
        validation: pd.DataFrame = None,
        test: pd.DataFrame = None,
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
        """
        super().__init__()
        self.train = train.copy()
        self.validation = validation
        # if validation is None:
        #     val_idx = train.sample(int(config.validation_split * len(train))).index
        #     self.validation = self.train[self.train.index.isin(val_idx)].copy()
        #     self.train = self.train[~self.train.index.isin(val_idx)]
        # else:
        self.validation = validation
        self.test = test if test is None else test.copy()
        self.categorical_cols = config.categorical_cols
        self.continuous_cols = config.continuous_cols
        self.target = config.target
        self.batch_size = config.batch_size
        self.config = config
        self.update_config()

    def update_config(self):
        self.config.categorical_cardinality = [
            int(self.train[col].fillna("NA").nunique()) + 1
            for col in self.config.categorical_cols
        ]
        if self.config.embedding_dims is None:
            self.config.embedding_dims = [
                (x, min(50, (x + 1) // 2)) for x in self.config.categorical_cardinality
            ]

    def preprocess_data(self, data, stage="inference"):
        added_features = None
        if self.config.encode_date_cols:
            for field_name, freq in self.config.date_cols:
                data = self.make_date(data, field_name)
                data, added_features = self.add_datepart(
                    data, field_name, frequency=freq, prefix=None, drop=True
                )
        # Encoding Categorical Columns
        if len(self.config.categorical_cols) > 0:
            if stage == "fit":
                self.categorical_encoder = OrdinalEncoder(
                    cols=self.config.categorical_cols
                )
                data = self.categorical_encoder.fit_transform(data)
            else:
                data = self.categorical_encoder.transform(data)
        if self.config.continuous_feature_transform is not None:
            if stage == "fit":
                transform = self.CONTINUOUS_TRANSFORMS[
                    self.config.continuous_feature_transform
                ]
                self.continuous_transform = transform["callable"](**transform["params"])
                data.loc[
                    :, self.config.continuous_cols
                ] = self.continuous_transform.fit_transform(
                    data.loc[:, self.config.continuous_cols]
                )
            else:
                data.loc[
                    :, self.config.continuous_cols
                ] = self.continuous_transform.transform(
                    data.loc[:, self.config.continuous_cols]
                )
        return data, added_features

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self.validation is None:
                val_idx = self.train.sample(
                    int(self.config.validation_split * len(self.train)), random_state=42
                ).index
                self.validation = self.train[self.train.index.isin(val_idx)]
                self.train = self.train[~self.train.index.isin(val_idx)]
            # Encoding Date Variables
            self.train, added_features = self.preprocess_data(self.train, stage="fit")
            if added_features is not None:
                self.config.categorical_cols += added_features
            self.validation, _ = self.preprocess_data(
                self.validation, stage="inference"
            )
            if self.test is not None:
                self.test, _ = self.preprocess_data(self.test, stage="inference")
            # Calculating the categorical dims and embedding dims
            self.update_config()

    # adapted from gluonts
    @classmethod
    def time_features_from_frequency_str(cls, freq_str: str) -> List[str]:
        """
        Returns a list of time features that will be appropriate for the given frequency string.

        Parameters
        ----------

        freq_str
            Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

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
    def make_date(cls, df: pd.DataFrame, date_field: str):
        "Make sure `df[date_field]` is of the right date type."
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
    ):
        "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
        field = df[field_name]
        prefix = (
            re.sub("[Dd]ate$", "", field_name) if prefix is None else prefix
        ) + "_"
        attr = cls.time_features_from_frequency_str(frequency)
        added_features = []
        for n in attr:
            if n == "Week":
                continue
            df[prefix + n] = getattr(field.dt, n.lower())
            added_features.append(prefix + n)
        # Pandas removed `dt.week` in v1.1.10
        if "Week" in attr:
            week = (
                field.dt.isocalendar().week
                if hasattr(field.dt, "isocalendar")
                else field.dt.week
            )
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
        return df, added_features

    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        dataset = TabularDataset(
            data=self.train,
            categorical_cols=self.categorical_cols,
            continuous_cols=self.continuous_cols,
            target=self.target,
        )
        return DataLoader(
            dataset, self.batch_size, shuffle=True, num_workers=self.config.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        dataset = TabularDataset(
            data=self.validation,
            categorical_cols=self.categorical_cols,
            continuous_cols=self.continuous_cols,
            target=self.target,
        )
        return DataLoader(
            dataset, self.batch_size, shuffle=False, num_workers=self.config.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        if self.test is not None:
            dataset = TabularDataset(
                data=self.test,
                categorical_cols=self.categorical_cols,
                continuous_cols=self.continuous_cols,
                target=self.target,
            )
            return DataLoader(
                dataset,
                self.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
            )

    def prepare_inference_dataloader(self, df) -> DataLoader:
        """ Function that prepares and loads the new data. """
        df = df.copy()
        if len(set(self.target) - set(df.columns)) < 0:
            df.loc[:, self.target] = np.zeros((len(df), len(self.target)))
        df, _ = self.preprocess_data(df, stage="inference")

        dataset = TabularDataset(
            data=df,
            categorical_cols=self.categorical_cols,
            continuous_cols=self.continuous_cols,
            target=self.target,
        )
        return DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )


class TabularDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        continuous_cols: List[str] = None,
        categorical_cols: List[str] = None,
        target: List[str] = None,
    ):
        """Dataset to Load Tabular Data

        Args:
            data (pd.DataFrame): Pandas DataFrame to load during training
            continuous_cols (List[str], optional): A list of names of continuous columns. Defaults to None.
            categorical_cols (List[str], optional): A list of names of categorical columns.
            These columns must be ordinal encoded beforehand. Defaults to None.
            target (List[str], optional): A list of strings with target column name(s). Defaults to None.
        """

        self.n = data.shape[0]

        if target:
            self.y = data[target].astype(np.float32).values
            if isinstance(target, str):
                self.y = self.y.reshape(-1, 1)
        else:
            self.y = np.zeros((self.n, 1))

        self.categorical_cols = categorical_cols if categorical_cols else []
        self.continuous_cols = continuous_cols if continuous_cols else []

        if self.continuous_cols:
            self.continuous_X = data[self.continuous_cols].astype(np.float32).values
        else:
            # Adding a dummy Continous column
            # TODO check if needed
            self.continuous_X = np.zeros((self.n, 1))

        if self.categorical_cols:
            self.categorical_X = data[categorical_cols].astype(np.int64).values
        else:
            self.categorical_X = np.zeros((self.n, 1))

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
            "continuous": self.continuous_X[idx],
            "categorical": self.categorical_X[idx],
        }
