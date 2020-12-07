from typing import Optional, List
import numpy as np
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
from category_encoders import OrdinalEncoder


class TabularDatamodule(pl.LightningDataModule):
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
        self.train = train
        self.validation = validation
        if validation is None:
            val_idx = train.sample(int(config.validation_split * len(train))).index
            self.validation = self.train[self.train.index.isin(val_idx)]
            self.train = self.train[~self.train.index.isin(val_idx)]
        else:
            self.validation = validation
        self.test = test
        self.categorical_cols = config.categorical_cols
        self.continuous_cols = config.continuous_cols
        self.target = config.target
        self.batch_size = config.batch_size
        self.config = config
        self.update_config()

    def update_config(self):
        # self.config.categorical_dim = (
        #     len(self.categorical_cols) if self.config.categorical_cols is not None else 0
        # )
        # self.config.output_dim = len(self.config.target)
        # self.config.continuous_dim = (
        #     len(self.config.continuous_cols)
        #     if len(self.config.continuous_cols) > 0
        #     else (len(self.train.columns) - self._output_dim - self._categorical_dim)
        # )
        self.config.categorical_cardinality = [
            int(self.train[col].nunique()) + 1 for col in self.config.categorical_cols
        ]
        if self.config.embedding_dims is None:
            self.config.embedding_dims = [
                (x, min(50, (x + 1) // 2)) for x in self.config.categorical_cardinality
            ]

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self.validation is None:
                val_idx = self.train.sample(
                    int(self.config.validation_split * len(self.train)), random_state=42
                ).index
                self.validation = self.train[self.train.index.isin(val_idx)]
                self.train = self.train[~self.train.index.isin(val_idx)]
            # Encoding Categorical Columns
            if len(self.config.categorical_cols) > 0:
                self.categorical_encoder = OrdinalEncoder(
                    cols=self.config.categorical_cols
                )
                self.train = self.categorical_encoder.fit_transform(self.train)
                # casting missing to 0
                for _map in self.categorical_encoder.category_mapping:
                    _map["mapping"][np.nan] = 0
                self.validation = self.categorical_encoder.transform(self.validation)
                # Replacing new with 0
                self.validation.loc[:, self.config.categorical_cols].replace(
                    -1, 0, inplace=True
                )
                if self.test is not None:
                    self.test = self.categorical_encoder.transform(self.test)
                    # Replacing new with 0
                    self.test.loc[:, self.config.categorical_cols].replace(
                        -1, 0, inplace=True
                    )
            # TODO Add date encoding for date_cols and add to categorical columns

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
