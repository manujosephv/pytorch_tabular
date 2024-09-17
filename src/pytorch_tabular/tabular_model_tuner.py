# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Tabular Model."""

import warnings
from collections import namedtuple
from copy import deepcopy
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from omegaconf.dictconfig import DictConfig
from pandas import DataFrame
from rich.progress import Progress
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, ParameterSampler

from pytorch_tabular.config import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.tabular_model import TabularModel
from pytorch_tabular.utils import OOMException, OutOfMemoryHandler, get_logger, suppress_lightning_logs

logger = get_logger(__name__)


class TabularModelTuner:
    """Tabular Model Tuner.

    This class is used to tune the hyperparameters of a TabularModel, given the search space,  strategy and metric to
    optimize.

    """

    ALLOWABLE_STRATEGIES = ["grid_search", "random_search"]
    OUTPUT = namedtuple("OUTPUT", ["trials_df", "best_params", "best_score", "best_model"])

    def __init__(
        self,
        data_config: Optional[Union[DataConfig, str]] = None,
        model_config: Optional[Union[ModelConfig, str]] = None,
        optimizer_config: Optional[Union[OptimizerConfig, str]] = None,
        trainer_config: Optional[Union[TrainerConfig, List[TrainerConfig]]] = None,
        model_callable: Optional[Callable] = None,
        model_state_dict_path: Optional[Union[str, Path]] = None,
        suppress_lightning_logger: bool = True,
        **kwargs,
    ):
        """Tabular Model Tuner helps you tune the hyperparameters of a TabularModel.

        Args:
            data_config (Optional[Union[DataConfig, str]], optional): The DataConfig for the TabularModel.
                If str is passed, will initialize the DataConfig using the yaml file in that path.
                Defaults to None.

            model_config (Optional[Union[ModelConfig, List[TrainerConfig]], optional): The ModelConfig for the
                TabularModel. If str is passed, will initialize the ModelConfig using the yaml file in that path.
                Defaults to None.

            optimizer_config (Optional[Union[OptimizerConfig, str]], optional): The OptimizerConfig for the
                TabularModel. If str is passed, will initialize the OptimizerConfig using the yaml file in
                that path. Defaults to None.

            trainer_config (Optional[Union[TrainerConfig, str]], optional): The TrainerConfig for the TabularModel.
                If str is passed, will initialize the TrainerConfig using the yaml file in that path.
                Defaults to None.

            model_callable (Optional[Callable], optional): A callable that returns a PyTorch Tabular Model.
                If provided, will ignore the model_config and use this callable to initialize the model.
                Defaults to None.

            model_state_dict_path (Optional[Union[str, Path]], optional): Path to the state dict of the model.

                If provided, will ignore the model_config and use this state dict to initialize the model.
                Defaults to None.

            suppress_lightning_logger (bool, optional): Whether to suppress the lightning logger. Defaults to True.

            **kwargs: Additional keyword arguments to be passed to the TabularModel init.

        """
        if not isinstance(model_config, list):
            model_config = [model_config]

        if trainer_config.profiler is not None:
            warnings.warn(
                "Profiler is not supported in tuner. Set profiler=None in TrainerConfig to disable this warning."
            )
            trainer_config.profiler = None
        if trainer_config.fast_dev_run:
            warnings.warn("fast_dev_run is turned on. Tuning results won't be accurate.")
        if trainer_config.progress_bar != "none":
            # If config and tuner have progress bar enabled, it will result in a bug within the library (rich.progress)
            trainer_config.progress_bar = "none"
            warnings.warn("Turning off progress bar. Set progress_bar='none' in TrainerConfig to disable this warning.")
        trainer_config.trainer_kwargs.update({"enable_model_summary": False})
        self.data_config = data_config
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self.trainer_config = trainer_config
        self.suppress_lightning_logger = suppress_lightning_logger
        self.tabular_model_init_kwargs = {
            "model_callable": model_callable,
            "model_state_dict_path": model_state_dict_path,
            **kwargs,
        }

    def _check_assign_config(self, config, param, value):
        if isinstance(config, DictConfig):
            if param in config:
                config[param] = value
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")
        elif isinstance(config, (ModelConfig, OptimizerConfig)):
            if hasattr(config, param):
                setattr(config, param, value)
            else:
                raise ValueError(f"{param} is not a valid parameter for {str(config)}")

    def _update_configs(
        self,
        optimizer_config: OptimizerConfig,
        model_config: ModelConfig,
        params: Dict,
    ):
        """Update the configs with the new parameters."""
        # update configs with the new parameters
        for k, v in params.items():
            if k == "model":
                continue

            root, param = k.split("__")
            if root.startswith("trainer_config"):
                raise ValueError(
                    "The trainer_config is not supported by tuner. Please remove it from tuner parameters!"
                )
            elif root.startswith("optimizer_config"):
                self._check_assign_config(optimizer_config, param, v)
            elif root.startswith("model_config.head_config"):
                param = param.replace("model_config.head_config.", "")
                self._check_assign_config(model_config.head_config, param, v)
            elif root.startswith("model_config") and "head_config" not in root:
                self._check_assign_config(model_config, param, v)
            else:
                raise ValueError(
                    f"{k} is not in the proper format. Use __ to separate the "
                    "root and param. for eg. `optimizer_config__optimizer` should be "
                    "used to update the optimizer parameter in the optimizer_config"
                )
        return optimizer_config, model_config

    def tune(
        self,
        train: DataFrame,
        search_space: Union[Dict, List[Dict]],
        metric: Union[str, Callable],
        mode: str,
        strategy: str,
        validation: Optional[DataFrame] = None,
        n_trials: Optional[int] = None,
        cv: Optional[Union[int, Iterable, BaseCrossValidator]] = None,
        cv_agg_func: Optional[Callable] = np.mean,
        cv_kwargs: Optional[Dict] = {},
        return_best_model: bool = True,
        verbose: bool = False,
        progress_bar: bool = True,
        random_state: Optional[int] = 42,
        ignore_oom: bool = True,
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

            return_best_model (bool, optional): If True, will return the best model. Defaults to True.

            verbose (bool, optional): Whether to print the results of each trial. Defaults to False.

            progress_bar (bool, optional): Whether to show a progress bar. Defaults to True.

            random_state (Optional[int], optional): Random state to be used for random search. Defaults to 42.

            ignore_oom (bool, optional): Whether to ignore out of memory errors. Defaults to True.

            **kwargs: Additional keyword arguments to be passed to the TabularModel fit.

        Returns:
            OUTPUT: A named tuple with the following attributes:
                trials_df (DataFrame): A dataframe with the results of each trial
                best_params (Dict): The best parameters found
                best_score (float): The best score found
                best_model (TabularModel or None): If return_best_model is True, return best_model otherwise return None

        """
        assert strategy in self.ALLOWABLE_STRATEGIES, f"tuner must be one of {self.ALLOWABLE_STRATEGIES}"
        assert mode in ["max", "min"], "mode must be one of ['max', 'min']"
        assert metric is not None, "metric must be specified"
        assert (isinstance(search_space, dict) or (isinstance(search_space, list))) and len(
            search_space
        ) > 0, "search_space must be a non-empty dict"
        if self.suppress_lightning_logger:
            suppress_lightning_logs()
        if cv is not None and validation is not None:
            warnings.warn(
                "Both validation and cv are provided. Ignoring validation and using cv. Use "
                "`validation=None` to turn off this warning."
            )
            validation = None

        if not isinstance(search_space, list):
            search_space = [search_space]

        assert len(self.model_config) == len(search_space), "model_config and search_space must have the same length"

        verbose_tabular_model = self.tabular_model_init_kwargs.pop("verbose", False)

        with Progress() as progress:
            model_config_iterator = range(len(self.model_config))
            if progress_bar:
                model_config_iterator = progress.track(
                    model_config_iterator, description="[green]Running models config..."
                )

            datamodule = None
            trials = []
            best_model = None
            best_score = 0.0
            for idx in model_config_iterator:
                search_space_temp = {
                    **{"model": [f"{idx}-{self.model_config[idx].__class__.__name__}"]},
                    **search_space[idx],
                }

                if strategy == "grid_search":
                    assert all(
                        isinstance(v, list) for v in search_space_temp.values()
                    ), "For grid search, all values in search_space must be a list of values to try"
                    search_space_iterator = list(ParameterGrid(search_space_temp))
                    if n_trials is not None:
                        warnings.warn(
                            "n_trials is ignored for grid search to do a complete sweep of"
                            " the grid. Set n_trials=None to turn off this warning."
                        )
                    n_trials = sum(1 for _ in search_space_iterator)
                elif strategy == "random_search":
                    assert n_trials is not None, "n_trials must be specified for random search"
                    search_space_iterator = list(
                        ParameterSampler(search_space_temp, n_iter=n_trials, random_state=random_state)
                    )
                else:
                    raise NotImplementedError(f"{strategy} is not implemented yet.")

                # Sort by trainer_config to recreate the datamodule when necessary
                trainer_configs = [key for key in search_space_iterator if "trainer_config" in key]
                for key in trainer_configs:
                    search_space_iterator = sorted(
                        search_space_iterator, key=lambda search_space_iterator: search_space_iterator[key]
                    )

                if progress_bar:
                    search_space_iterator = progress.track(
                        search_space_iterator,
                        description=f"[blue]Training {idx}-{self.model_config[idx].__class__.__name__}...",
                    )

                if isinstance(metric, str):
                    is_callable_metric = False
                    metric_str = metric
                elif callable(metric):
                    is_callable_metric = True
                    metric_str = metric.__name__

                for i, params in enumerate(search_space_iterator):
                    # Copying the configs as a base
                    # Make sure all default parameters that you want to be set for all
                    # trials are in the original configs
                    trainer_config_t = deepcopy(self.trainer_config)
                    optimizer_config_t = deepcopy(self.optimizer_config)
                    model_config_t = deepcopy(self.model_config[idx])

                    optimizer_config_t, model_config_t = self._update_configs(
                        optimizer_config_t, model_config_t, params
                    )
                    # Initialize Tabular model using the new config
                    tabular_model_t = TabularModel(
                        data_config=self.data_config,
                        model_config=model_config_t,
                        optimizer_config=optimizer_config_t,
                        trainer_config=trainer_config_t,
                        verbose=verbose_tabular_model,
                        **self.tabular_model_init_kwargs,
                    )

                    # Create datamodule
                    if not datamodule:
                        prep_dl_kwargs, prep_model_kwargs, train_kwargs = tabular_model_t._split_kwargs(kwargs)
                        if "seed" not in prep_dl_kwargs:
                            prep_dl_kwargs["seed"] = random_state
                        datamodule = tabular_model_t.prepare_dataloader(
                            train=train, validation=validation, **prep_dl_kwargs
                        )
                        validation = validation if validation is not None else datamodule.validation_dataset.data

                    if cv is not None:
                        cv_verbose = cv_kwargs.pop("verbose", False)
                        cv_kwargs.pop("handle_oom", None)
                        with OutOfMemoryHandler(handle_oom=True) as handler:
                            cv_scores, _ = tabular_model_t.cross_validate(
                                cv=cv,
                                train=train,
                                metric=metric,
                                verbose=cv_verbose,
                                handle_oom=False,
                                **cv_kwargs,
                            )
                        if handler.oom_triggered:
                            if not ignore_oom:
                                raise OOMException(
                                    "Out of memory error occurred during cross validation. "
                                    "Set ignore_oom=True to ignore this error."
                                )
                            else:
                                params.update({metric_str: (np.inf if mode == "min" else -np.inf)})
                                params.update({"model": f"{params['model']} (OOM)"})
                        else:
                            params.update({metric_str: cv_agg_func(cv_scores)})
                    else:
                        model = tabular_model_t.prepare_model(
                            datamodule=datamodule,
                            **prep_model_kwargs,
                        )
                        train_kwargs.pop("handle_oom", None)
                        with OutOfMemoryHandler(handle_oom=True) as handler:
                            tabular_model_t.train(model=model, datamodule=datamodule, handle_oom=False, **train_kwargs)
                        if handler.oom_triggered:
                            if not ignore_oom:
                                raise OOMException(
                                    "Out of memory error occurred during training. "
                                    "Set ignore_oom=True to ignore this error."
                                )
                            else:
                                params.update({metric_str: (np.inf if mode == "min" else -np.inf)})
                                params.update({"model": f"{params['model']} (OOM)"})
                        else:
                            if is_callable_metric:
                                preds = tabular_model_t.predict(validation, include_input_features=False)
                                params.update({metric_str: metric(validation[tabular_model_t.config.target], preds)})
                            else:
                                result = tabular_model_t.evaluate(validation, verbose=False)
                                params.update({k.replace("test_", ""): v for k, v in result[0].items()})

                            if return_best_model:
                                # Removing the datamodule from the model to save memory
                                tabular_model_t.datamodule = None
                                if best_model is None:
                                    best_model = deepcopy(tabular_model_t)
                                    best_score = params[metric_str]
                                else:
                                    if mode == "min":
                                        if params[metric_str] < best_score:
                                            best_model = deepcopy(tabular_model_t)
                                            best_score = params[metric_str]
                                    elif mode == "max":
                                        if params[metric_str] > best_score:
                                            best_model = deepcopy(tabular_model_t)
                                            best_score = params[metric_str]

                    params.update({"trial_id": i})
                    trials.append(params)
                    if verbose:
                        logger.info(f"Trial {i+1}/{n_trials}: {params} | Score: {params[metric]}")

        trials_df = pd.DataFrame(trials)
        trials = trials_df.pop("trial_id")
        if mode == "max":
            best_idx = trials_df[metric_str].idxmax()
        elif mode == "min":
            best_idx = trials_df[metric_str].idxmin()
        else:
            raise NotImplementedError(f"{mode} is not implemented yet.")
        best_params = trials_df.iloc[best_idx].to_dict()
        best_score = best_params.pop(metric_str)
        trials_df.insert(0, "trial_id", trials)

        if verbose:
            logger.info("Model Tuner Finished")
            logger.info(f"Best Model: {best_params['model']} - Best Score ({metric_str}): {best_score}")

        if return_best_model and best_model is not None:
            best_model.datamodule = datamodule

            return self.OUTPUT(trials_df, best_params, best_score, best_model)
        else:
            return self.OUTPUT(trials_df, best_params, best_score, None)
