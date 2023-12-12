# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model."""
from contextlib import nullcontext
import copy
import inspect
import os
import warnings
from collections import defaultdict, namedtuple
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pandas import DataFrame
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import (
    GradientAccumulationScheduler,
)
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.utilities.model_summary import summarize
from sklearn.base import TransformerMixin
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn

from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.config.config import InferredConfig
from pytorch_tabular.models.base_model import BaseModel, _CaptumModel, _GenericModel
from pytorch_tabular.models.common.layers.embeddings import (
    Embedding1dLayer,
    Embedding2dLayer,
    PreEncoded1dLayer,
)

from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.utils import get_logger, getattr_nested, pl_load
from sklearn.model_selection import ParameterGrid, ParameterSampler
from rich.progress import Progress
from copy import deepcopy

try:
    import captum.attr

    CAPTUM_INSTALLED = True
except ImportError:
    CAPTUM_INSTALLED = False

logger = get_logger(__name__)


class TabularModelTuner:
    """Tabular Model Tuner.

    This class is used to tune the hyperparameters of a TabularModel, given the search space,
     strategy and metric to optimize.
    """

    ALLOWABLE_STRATEGIES = ["grid_search", "random_search"]
    OUTPUT = namedtuple("OUTPUT", ["trials_df", "best_params", "best_score"])
    def __init__(
        self,
        data_config: Optional[Union[DataConfig, str]] = None,
        model_config: Optional[Union[ModelConfig, str]] = None,
        optimizer_config: Optional[Union[OptimizerConfig, str]] = None,
        trainer_config: Optional[Union[TrainerConfig, str]] = None,
        model_callable: Optional[Callable] = None,
        model_state_dict_path: Optional[Union[str, Path]] = None,
    ):
        if trainer_config.profiler is not None:
            warnings.warn(
                "Profiler is not supported in tuner. Set profiler=None to disable this warning."
            )
            trainer_config.profiler = None
        if trainer_config.fast_dev_run:
            warnings.warn(
                "fast_dev_run is turned on. Tuning results won't be accurate."
            )
        if trainer_config.progress_bar != "none":
            warnings.warn(
                "Turning off progress bar. Set progress_bar='none' to disable this warning."
            )
        trainer_config.trainer_kwargs.update(
            {"enable_model_summary": False}
        )
        self.data_config = data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.trainer_config = trainer_config
        self.tabular_model_init_kwargs = dict(
            model_callable=model_callable, model_state_dict_path=model_state_dict_path
        )

    def _check_assign_config(self, config, param, value):
        if isinstance(config, DictConfig):
            if param in config:
                config[param] = value
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")
        elif isinstance(config, ModelConfig):
            if hasattr(config, param):
                config.param = value
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")

    def _update_configs(
        self,
        trainer_config: TrainerConfig,
        optimizer_config: OptimizerConfig,
        model_config: ModelConfig,
        params: Dict,
    ):
        """Update the configs with the new parameters."""
        # update configs with the new parameters
        for k, v in params.items():
            root, param = k.split("__")
            if root.startswith("trainer_config"):
                self._check_assign_config(trainer_config, param, v)
            elif root.startswith("optimizer_config"):
                self._check_assign_config(optimizer_config, param, v)
            elif root.startswith("model_config.head_config"):
                param = param.replace("model_config.head_config.", "")
                self._check_assign_config(model_config.head_config, param, v)
            elif root.startswith("model_config") and "head_config" not in root:
                self._check_assign_config(model_config, param, v)
            else:
                raise ValueError(f"{k} is not in the proper format. Use __ to separate the "
                                "root and param. for eg. `training_config__batch_size` should be "
                                "used to update the batch_size parameter in the training_config")
        return trainer_config, optimizer_config, model_config
        

    def tune(
        self,
        train: DataFrame,
        search_space: Dict,
        metric: Union[str, Callable],
        mode: str,
        strategy: str,
        validation: Optional[DataFrame] = None,
        n_trials: Optional[int] = None,
        cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
        cv_agg_func: Optional[Callable] = np.mean,
        cv_kwargs: Optional[Dict] = {},
        verbose: bool = False,
        progress_bar: bool = True,
        random_state: Optional[int] = 42,
        **kwargs,
    ):
        """Tune the hyperparameters of the TabularModel.
        Args:
        train (DataFrame): Training data

        validation (DataFrame, optional): Validation data. Defaults to None.

        search_space (Dict): A dictionary of the form {param_name: [values to try]} 
            for grid search or {param_name: distribution} for random search

        metric (Union[str, Callable]): The metric to be used for evaluation.
            If str is provided, will use that metric from the defined ones. 
            If callable is provided, will use that function as the metric. 
            We expect callable to be of the form `metric(y_true, y_pred)`. For classification
            problems, The `y_pred` is a dataframe with the probabilities for each class
            (<class>_probability) and a final prediction(prediction). And for Regression, it is a
            dataframe with a final prediction (<target>_prediction).
            Defaults to None.

        mode (str): One of ['max', 'min']. Whether to maximize or minimize the metric.

        strategy (str): One of ['grid_search', 'random_search']. The strategy to use for tuning.

        n_trials (int, optional): Number of trials to run. Only used for random search.
            Defaults to None.

        cv (Optional[Union[int, Iterable, BaseCrossValidator]]): Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to not use any cross validation. We will just use the validation data
            - integer, to specify the number of folds in a (Stratified)KFold,
            - An iterable yielding (train, test) splits as arrays of indices.
            - A scikit-learn CV splitter.
            Defaults to None.

        cv_agg_func (Optional[Callable], optional): Function to aggregate the cross validation scores.
            Defaults to np.mean.

        cv_kwargs (Optional[Dict], optional): Additional keyword arguments to be passed to the cross validation
            method. Defaults to {}.

        verbose (bool, optional): Whether to print the results of each trial. Defaults to False.

        progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

        random_state (Optional[int], optional): Random state to be used for random search. Defaults to 42.

        **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

        Returns:
            OUTPUT: A named tuple with the following attributes:
                trials_df (DataFrame): A dataframe with the results of each trial
                best_params (Dict): The best parameters found
                best_score (float): The best score found
        """
        assert (
            strategy in self.ALLOWABLE_STRATEGIES
        ), f"tuner must be one of {self.ALLOWABLE_STRATEGIES}"
        assert mode in ["max", "min"], "mode must be one of ['max', 'min']"
        assert metric is not None, "metric must be specified"
        assert (
            isinstance(search_space, dict) and len(search_space) > 0
        ), "search_space must be a non-empty dict"
        if cv is not None and validation is not None:
            warnings.warn(
                "Both validation and cv are provided. Ignoring validation and using cv"
            )
            validation = None
            
        
        if strategy == "grid_search":
            assert all(isinstance(v, list) for v in search_space.values()), (
                "For grid search, all values in search_space must be a list "
                "of values to try"
            )
            iterator = ParameterGrid(search_space)
            if n_trials is not None:
                warnings.warn(
                    "n_trials is ignored for grid search to do a complete sweep of"
                    " the grid"
                )
            n_trials = sum(1 for _ in iterator)
        elif strategy == "random_search":
            assert (
                n_trials is not None
            ), "n_trials must be specified for random search"
            iterator = ParameterSampler(
                search_space, n_iter=n_trials, random_state=random_state
            )
        else:
            raise NotImplementedError(f"{strategy} is not implemented yet.")
        if progress_bar:
            ctx_mgr = Progress()
        else:
            ctx_mgr = nullcontext()

        temp_tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=self.model_config,
            optimizer_config=self.optimizer_config,
            trainer_config=self.trainer_config,
            **self.tabular_model_init_kwargs,
        )
        prep_dl_kwargs, prep_model_kwargs, train_kwargs = temp_tabular_model._split_kwargs(kwargs)
        if "seed" not in prep_dl_kwargs:
            prep_dl_kwargs["seed"] = random_state
        datamodule = temp_tabular_model.prepare_dataloader(
            train=train, validation=validation, **prep_dl_kwargs
        )
        if isinstance(metric, str):
            # metric = metric_to_pt_metric(metric)
            is_callable_metric = False
        elif callable(metric):
            is_callable_metric = True
        del temp_tabular_model
        trials = []

        with ctx_mgr as progress:
            if progress:
                task = progress.add_task(
                    f"[green]{strategy.replace('_',' ').title()}...", total=n_trials
                )
            for i, params in enumerate(iterator):
                # Copying the configs as a base
                # Make sure all default parameters that you want to be set for all
                # trials are in the original configs
                trainer_config_t = deepcopy(self.trainer_config)
                optimizer_config_t = deepcopy(self.optimizer_config)
                model_config_t = deepcopy(self.model_config)

                trainer_config_t, optimizer_config_t, model_config_t = (
                    self._update_configs(
                        trainer_config_t, optimizer_config_t, model_config_t, params
                    )
                )
                # Initialize Tabular model using the new config
                tabular_model_t = TabularModel(
                    data_config=self.data_config,
                    model_config=model_config_t,
                    optimizer_config=optimizer_config_t,
                    trainer_config=trainer_config_t,
                    **self.tabular_model_init_kwargs,
                )
                if cv is not None:
                    cv_scores, _ = tabular_model_t.cross_validate(
                        cv=cv,
                        train=train,
                        metric=metric,
                        **cv_kwargs,
                    )
                    params.update({metric.__name__ if is_callable_metric else metric:cv_agg_func(cv_scores)})
                else:
                    
                    model = tabular_model_t.prepare_model(
                        datamodule=datamodule,
                        **prep_model_kwargs,
                    )
                    tabular_model_t.train(model=model, datamodule=datamodule, **train_kwargs)
                    if is_callable_metric:
                        preds = tabular_model_t.predict(validation, include_input_features=False)
                        params.update({metric.__name__:metric(validation[tabular_model_t.config.target], preds)})
                    else:
                        result = tabular_model_t.evaluate(validation, verbose=False)
                        params.update({k.replace("test_", ""):v for k,v in result[0].items()})
                params.update({"trial_id": i})
                trials.append(params)
                if verbose:
                    logger.info(f"Trial {i+1}/{n_trials}: {params} | Score: {params[metric]}")
                if progress:
                    progress.update(task, advance=1)
        trials_df = pd.DataFrame(trials)
        trials = trials_df.pop("trial_id")
        if mode == "max":
            best_idx = trials_df[metric].idxmax()
        elif mode == "min":
            best_idx = trials_df[metric].idxmin()
        else:
            raise NotImplementedError(f"{mode} is not implemented yet.")
        best_params = trials_df.iloc[best_idx].to_dict()
        best_score = best_params.pop(metric)
        trials_df.insert(0, "trial_id", trials)
        return self.OUTPUT(trials_df, best_params, best_score)
