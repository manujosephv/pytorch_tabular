"""Top-level package for Pytorch Tabular."""

__author__ = """Manu Joseph"""
__email__ = "manujosephv@gmail.com"
__version__ = "1.0.2"

import warnings

import numpy as np
from . import models, ssl_models
from .categorical_encoders import CategoricalEmbeddingTransformer
from .feature_extractor import DeepFeatureExtractor
from .tabular_datamodule import TabularDatamodule
from .tabular_model import TabularModel
from .tabular_model_tuner import TabularModelTuner
# from .model_comparator import compare_models
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
import time
from typing import Callable, List, Optional, Tuple, Union
from .utils import OOMException, OutOfMemoryHandler
import pandas as pd

__all__ = [
    "TabularModel",
    "TabularModelTuner",
    "TabularDatamodule",
    "models",
    "ssl_models",
    "CategoricalEmbeddingTransformer",
    "DeepFeatureExtractor",
    "utils",
    "compare_models",
    "available_models",
    "available_ssl_models",
]

# fix Sphinx issues, see https://bit.ly/2K2eptM
for item in __all__:
    if hasattr(item, "__module__"):
        setattr(item, "__module__", __name__)


def available_models():
    return [cl for cl in dir(models) if "config" in cl.lower()]


def available_ssl_models():
    return [cl for cl in dir(ssl_models) if "config" in cl.lower()]


MODEL_PRESETS = {
    "lite": [
        "CategoryEmbeddingModelConfig",
        "DANetConfig",
        "GANDALFConfig",
        "TabNetModelConfig",
    ],
    "full": [m for m in available_models() if m not in ['MDNConfig', "NodeConfig"]],
    "high_memory": [m for m in available_models() if m not in ['MDNConfig']],
}


def _validate_args(
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: Union[DataConfig, str],
    optimizer_config: Union[OptimizerConfig, str],
    trainer_config: Union[TrainerConfig, str],
    model_list: Union[str, List[Union[ModelConfig, str]]] = "fast",
    metrics: Optional[List[Union[str, Callable]]] = None,
    metrics_params: Optional[List[dict]] = None,
    metrics_prob_input: Optional[List[bool]] = None,
    validation: Optional[pd.DataFrame] = None,
    experiment_config: Optional[Union[ExperimentConfig, str]] = None,
    common_model_args: Optional[dict] = {},
    rank_metric: Optional[str] = "loss",
):
    assert task in [
        "classification",
        "regression",
    ], f"task must be one of ['classification', 'regression'], but got {task}"
    assert isinstance(
        train, pd.DataFrame
    ), f"train must be a pandas DataFrame, but got {type(train)}"
    assert isinstance(
        test, pd.DataFrame
    ), f"test must be a pandas DataFrame, but got {type(test)}"
    assert model_list is not None, "models cannot be None"
    assert isinstance(
        model_list, (str, list)
    ), f"models must be a string or list of strings, but got {type(model_list)}"
    if isinstance(model_list, str):
        assert model_list in MODEL_PRESETS.keys(), (
            f"models must be one of {MODEL_PRESETS.keys()}, "
            f"but got {model_list}"
        )
    else:  # isinstance(models, list):
        assert all(
            [isinstance(m, (str, ModelConfig)) for m in model_list]
        ), f"models must be a list of strings or ModelConfigs, but got {model_list}"
    if metrics is not None:
        assert isinstance(
            metrics, list
        ), f"metrics must be a list of strings or callables, but got {type(metrics)}"
        assert all(
            [isinstance(m, (str, Callable)) for m in metrics]
        ), f"metrics must be a list of strings or callables, but got {metrics}"
        assert (
            metrics_params is not None
        ), "metric_params cannot be None when metrics is not None"
        assert (
            metrics_prob_input is not None
        ), "metrics_prob_inputs cannot be None when metrics is not None"
        assert isinstance(
            metrics_params, list
        ), f"metric_params must be a list of dicts, but got {type(metrics_params)}"
        assert isinstance(metrics_prob_input, list), (
            "metrics_prob_inputs must be a list of bools, but got"
            f" {type(metrics_prob_input)}"
        )
        assert len(metrics) == len(metrics_params), (
            "metrics and metric_params must be of the same length, but got"
            f" {len(metrics)} and {len(metrics_params)}"
        )
        assert len(metrics) == len(metrics_prob_input), (
            "metrics and metrics_prob_inputs must be of the same length, but got"
            f" {len(metrics)} and {len(metrics_prob_input)}"
        )
        assert all(
            [isinstance(m, dict) for m in metrics_params]
        ), f"metric_params must be a list of dicts, but got {metrics_params}"
    if common_model_args is not None:
        # all args should be members of ModelConfig
        assert all([k in ModelConfig.__dataclass_fields__.keys() for k in common_model_args.keys()]), (
            "common_model_args must be a subset of ModelConfig, but got"
            f" {common_model_args.keys()}"
        )
    if rank_metric[0] not in ["loss", "accuracy", "mean_squared_error"]:
        assert (
            rank_metric[0] in metrics
        ), f"rank_metric must be one of {metrics}, but got {rank_metric}"
    assert rank_metric[1] in [
        "lower_is_better",
        "higher_is_better",
    ], (
        "rank_metric[1] must be one of ['lower_is_better', 'higher_is_better'], but"
        f" got {rank_metric[1]}"
    )


def compare_models(
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: Union[DataConfig, str],
    optimizer_config: Union[OptimizerConfig, str],
    trainer_config: Union[TrainerConfig, str],
    model_list: Union[str, List[Union[ModelConfig, str]]] = "lite",
    metrics: Optional[List[Union[str, Callable]]] = None,
    metrics_params: Optional[List[dict]] = None,
    metrics_prob_input: Optional[List[bool]] = None,
    validation: Optional[pd.DataFrame] = None,
    experiment_config: Optional[Union[ExperimentConfig, str]] = None,
    common_model_args: Optional[dict] = {},
    rank_metric: Optional[Tuple[str, str]] = ("loss", "lower_is_better"),
    return_best_model: bool = True,
    seed: int = 42,
    ignore_oom: bool = True,
):
    """Compare multiple models on the same dataset.

    Args:
        task (str): The type of prediction task. Either 'classification' or 'regression'

        train (pd.DataFrame): The training data

        test (pd.DataFrame): The test data on which performance is evaluated

        data_config (Union[DataConfig, str]): DataConfig object or path to the yaml file.

        optimizer_config (Union[OptimizerConfig, str]): OptimizerConfig object or path to the yaml file.

        trainer_config (Union[TrainerConfig, str]): TrainerConfig object or path to the yaml file.

        models (Union[str, List[Union[ModelConfig, str]]], optional): The list of models to compare. This can be one of
                the presets defined in ``model_comprator.MODEL_PRESETS`` or a list of ``ModelConfig`` objects.
                Defaults to "fast".

        metrics (Optional[List[str]]): the list of metrics you need to track during training. The metrics
                should be one of the functional metrics implemented in ``torchmetrics``. By default, it is
                accuracy if classification and mean_squared_error for regression

        metrics_prob_input (Optional[bool]): Is a mandatory parameter for classification metrics defined in
                the config. This defines whether the input to the metric function is the probability or the class.
                Length should be same as the number of metrics. Defaults to None.

        metrics_params (Optional[List]): The parameters to be passed to the metrics function. `task` is forced to
                be `multiclass` because the multiclass version can handle binary as well and for simplicity we are
                only using `multiclass`.

        validation (Optional[DataFrame], optional):
                If provided, will use this dataframe as the validation while training.
                Used in Early Stopping and Logging. If left empty, will use 20% of Train data as validation.
                Defaults to None.

        experiment_config (Optional[Union[ExperimentConfig, str]], optional): ExperimentConfig object or path to the yaml file.

        common_model_args (Optional[dict], optional): The model argument which are common to all models. The list of params can
            be found in ``ModelConfig``. If not provided, will use defaults. Defaults to {}.

        rank_metric (Optional[Tuple[str, str]], optional): The metric to use for ranking the models. The first element of the tuple
            is the metric name and the second element is the direction. Defaults to ('loss', "lower_is_better").

        return_best_model (bool, optional): If True, will return the best model. Defaults to True.

        seed (int, optional): The seed for reproducibility. Defaults to 42.

        ignore_oom (bool, optional): If True, will ignore the Out of Memory error and continue with the next model.
    """
    _validate_args(
        task=task,
        train=train,
        test=test,
        data_config=data_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        model_list=model_list,
        metrics=metrics,
        metrics_params=metrics_params,
        metrics_prob_input=metrics_prob_input,
        validation=validation,
        experiment_config=experiment_config,
        common_model_args=common_model_args,
        rank_metric=rank_metric,
    )
    if model_list in ["full", "high_memory"]:
        warnings.warn(
            "The full model list is quite large and uses a lot of memory. "
            "Consider using `lite` or define configs yourselves for a faster run"
        )
    _model_args = ["metrics", "metrics_params", "metrics_prob_input"]
    # Replacing the common model args with the ones passed in the function
    for arg in _model_args:
        if locals()[arg] is not None:
            common_model_args[arg] = locals()[arg]
    if isinstance(model_list, str):
        model_list = MODEL_PRESETS[model_list]
        model_list = [
            (
                getattr(models, m)(task=task, **common_model_args)
                if isinstance(m, str)
                else m
            )
            for m in model_list
        ]
    else:
        if len(common_model_args) > 0:
            warnings.warn(
                "common_model_args are ignored when model_list is not a string"
            )
    def _init_tabular_model(m):
        return TabularModel(
            data_config=data_config,
            model_config=m,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            experiment_config=experiment_config,
            verbose=False
        )

    datamodule = _init_tabular_model(model_list[0]).prepare_dataloader(
        train=train, validation=validation, seed=seed
    )
    results = []
    best_model = None
    is_lower_better = rank_metric[1] == "lower_is_better"
    best_score = 1e9 if is_lower_better else -1e9
    for tabular_model in model_list:
        name = tabular_model._model_name
        params = tabular_model.__dict__
        start_time = time.time()
        tabular_model = _init_tabular_model(tabular_model)
        model = tabular_model.prepare_model(datamodule)
        with OutOfMemoryHandler(handle_oom=True) as handler:
            tabular_model.train(model, datamodule, handle_oom=False)
        res_dict = {
            "model": name,
        }
        if handler.oom_triggered:
            if not ignore_oom:
                raise OOMException(
                    "Out of memory error occurred during cross validation. "
                    "Set ignore_oom=True to ignore this error."
                )
            else:
                res_dict.update(
                    {
                        f"test_{rank_metric[0]}": np.inf if is_lower_better else -np.inf,
                        "time_taken": "NA",
                    }
                )
                res_dict['model'] = name + " (OOM)"
        else:
            res_dict.update(
                tabular_model.evaluate(test=test, verbose=False)[0]
            )
            res_dict["time_taken"] = time.time() - start_time
        res_dict["params"] = params
        results.append(res_dict)
        if best_model is None:
            best_model = tabular_model
        else:
            if is_lower_better:
                if (
                    res_dict[f"test_{rank_metric[0]}"]
                    < best_score
                ):
                    best_model = tabular_model
                    best_score = res_dict[f"test_{rank_metric[0]}"]
            else:
                if (
                    res_dict[f"test_{rank_metric[0]}"]
                    > best_score
                ):
                    best_model = tabular_model
                    best_score = res_dict[f"test_{rank_metric[0]}"]
    results = pd.DataFrame(results).sort_values(
        by=f"test_{rank_metric[0]}", ascending=is_lower_better
    )
    if return_best_model:
        return results, best_model
    else:
        return results
