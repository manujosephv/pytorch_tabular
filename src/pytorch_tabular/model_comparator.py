import time
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd

from pytorch_tabular import TabularModel
from pytorch_tabular import models as pt_models
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular import available_models

MODEL_PRESETS = {
    "lite": [
        "CategoryEmbeddingModelConfig",
        "DANetConfig",
        "GANDALFConfig",
        "TabNetModelConfig",
    ],
    "full": available_models(),
}


def _validate_args(
    task: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_config: Union[DataConfig, str],
    optimizer_config: Union[OptimizerConfig, str],
    trainer_config: Union[TrainerConfig, str],
    models: Union[str, List[Union[ModelConfig, str]]] = "fast",
    metrics: Optional[List[Union[str, Callable]]] = None,
    metric_params: Optional[List[dict]] = None,
    metrics_prob_inputs: Optional[List[bool]] = None,
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
    assert models is not None, "models cannot be None"
    assert isinstance(
        models, (str, list)
    ), f"models must be a string or list of strings, but got {type(models)}"
    if isinstance(models, str):
        assert models in MODEL_PRESETS.keys(), (
            f"models must be one of {MODEL_PRESETS.keys()}, "
            f"but got {models}"
        )
    else:  # isinstance(models, list):
        assert all(
            [isinstance(m, (str, ModelConfig)) for m in models]
        ), f"models must be a list of strings or ModelConfigs, but got {models}"
    if metrics is not None:
        assert isinstance(
            metrics, list
        ), f"metrics must be a list of strings or callables, but got {type(metrics)}"
        assert all(
            [isinstance(m, (str, Callable)) for m in metrics]
        ), f"metrics must be a list of strings or callables, but got {metrics}"
        assert (
            metric_params is not None
        ), "metric_params cannot be None when metrics is not None"
        assert (
            metrics_prob_inputs is not None
        ), "metrics_prob_inputs cannot be None when metrics is not None"
        assert isinstance(
            metric_params, list
        ), f"metric_params must be a list of dicts, but got {type(metric_params)}"
        assert isinstance(metrics_prob_inputs, list), (
            "metrics_prob_inputs must be a list of bools, but got"
            f" {type(metrics_prob_inputs)}"
        )
        assert len(metrics) == len(metric_params), (
            "metrics and metric_params must be of the same length, but got"
            f" {len(metrics)} and {len(metric_params)}"
        )
        assert len(metrics) == len(metrics_prob_inputs), (
            "metrics and metrics_prob_inputs must be of the same length, but got"
            f" {len(metrics)} and {len(metrics_prob_inputs)}"
        )
        assert all(
            [isinstance(m, dict) for m in metric_params]
        ), f"metric_params must be a list of dicts, but got {metric_params}"
    if common_model_args is not None:
        # all args should be members of ModelConfig
        assert all([hasattr(ModelConfig, k) for k in common_model_args.keys()]), (
            "common_model_args must be a subset of ModelConfig, but got"
            f" {common_model_args.keys()}"
        )
    if rank_metric[0] != "loss":
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
    models: Union[str, List[Union[ModelConfig, str]]] = "fast",
    metrics: Optional[List[Union[str, Callable]]] = None,
    metric_params: Optional[List[dict]] = None,
    metrics_prob_inputs: Optional[List[bool]] = None,
    validation: Optional[pd.DataFrame] = None,
    experiment_config: Optional[Union[ExperimentConfig, str]] = None,
    common_model_args: Optional[dict] = {},
    rank_metric: Optional[Tuple[str, str]] = ("loss", "lower_is_better"),
    return_best_model: bool = True,
    seed: int = 42,
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
    """
    _validate_args(
        task=task,
        train=train,
        test=test,
        data_config=data_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
        models=models,
        metrics=metrics,
        metric_params=metric_params,
        metrics_prob_inputs=metrics_prob_inputs,
        validation=validation,
        experiment_config=experiment_config,
        common_model_args=common_model_args,
        rank_metric=rank_metric,
    )
    _model_args = ["metrics", "metric_params", "metrics_prob_inputs"]
    # Replacing the common model args with the ones passed in the function
    for arg in _model_args:
        if locals()[arg] is not None:
            common_model_args[arg] = locals()[arg]
    if isinstance(models, str):
        models = MODEL_PRESETS[models]
        models = [
            (
                getattr(pt_models, m)(task=task, **common_model_args)
                if isinstance(m, str)
                else m
            )
            for m in models
        ]

        def _init_tabular_model(m):
            return TabularModel(
                data_config=data_config,
                model_config=m,
                optimizer_config=optimizer_config,
                trainer_config=trainer_config,
                experiment_config=experiment_config,
            )

        datamodule = _init_tabular_model(models[0]).prepare_dataloader(
            train=train, validation=validation, seed=seed
        )
        results = []
        best_model = None
        is_lower_better = rank_metric[1] == "lower_is_better"
        for tabular_model in models:
            start_time = time.time()
            tabular_model = _init_tabular_model(tabular_model)
            model = tabular_model.prepare_model(datamodule)
            tabular_model.train(model, datamodule)
            res_dict = {
                "Sl. No.": len(results) + 1,
                "model": tabular_model.model_config._model_name,
            }
            res_dict.update(
                tabular_model.evaluate(test_loader=datamodule.test_loader())[0]
            )
            res_dict["time_taken"] = time.time() - start_time
            res_dict.update(tabular_model.model_config.__dict__)
            results.append(res_dict)
            if best_model is None:
                best_model = tabular_model
            else:
                if is_lower_better:
                    if (
                        res_dict[f"test_{rank_metric[0]}"]
                        < results[best_model][rank_metric[0]]
                    ):
                        best_model = tabular_model
                else:
                    if (
                        res_dict[f"test_{rank_metric[0]}"]
                        > results[best_model][rank_metric[0]]
                    ):
                        best_model = tabular_model
        results = pd.DataFrame(results).sort_values(
            by=rank_metric[0], ascending=is_lower_better
        )
        if return_best_model:
            return results, best_model
        else:
            return results
