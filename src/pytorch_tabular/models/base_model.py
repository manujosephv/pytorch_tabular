# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Base Model."""

import importlib
import warnings
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from omegaconf import DictConfig, OmegaConf
from pandas import DataFrame
from torch import Tensor
from torch.optim import Optimizer

from pytorch_tabular.models.common.heads import blocks
from pytorch_tabular.models.common.layers import PreEncoded1dLayer
from pytorch_tabular.utils import get_logger, reset_all_weights

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:
    WANDB_INSTALLED = False

try:
    import plotly.graph_objects as go

    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False


logger = get_logger(__name__)


def safe_merge_config(config: DictConfig, inferred_config: DictConfig) -> DictConfig:
    """Merge two configurations.

    Args:
        config: The base configuration.
        inferred_config: The custom configuration.

    Returns:
        The merged configuration.

    """
    # using base config values if exist
    inferred_config.embedding_dims = config.get("embedding_dims") or inferred_config.embedding_dims
    merged_config = OmegaConf.merge(OmegaConf.to_container(config), OmegaConf.to_container(inferred_config))
    return merged_config


def _create_optimizer(optimizer: Union[str, Callable]) -> Type[Optimizer]:
    """Instantiate Optimizer."""
    if callable(optimizer):
        return optimizer
    if "." not in optimizer:
        return getattr(torch.optim, optimizer)
    py_path, cls_name = optimizer.rsplit(".", 1)
    module = importlib.import_module(py_path)
    return getattr(module, cls_name)


class BaseModel(pl.LightningModule, metaclass=ABCMeta):
    def __init__(
        self,
        config: DictConfig,
        custom_loss: Optional[torch.nn.Module] = None,
        custom_metrics: Optional[List[Callable]] = None,
        custom_metrics_prob_inputs: Optional[List[bool]] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs,
    ):
        """Base Model for PyTorch Tabular.

        Args:
            config (DictConfig): The configuration for the model.
            custom_loss (Optional[torch.nn.Module], optional): A custom loss function. Defaults to None.
            custom_metrics (Optional[List[Callable]], optional): A list of custom metrics. Defaults to None.
            custom_metrics_prob_inputs (Optional[List[bool]], optional): A list of boolean values indicating whether the
                metric requires probability inputs. Defaults to None.
            custom_optimizer (Optional[torch.optim.Optimizer], optional):
                A custom optimizer as callable or string to be imported. Defaults to None.
            custom_optimizer_params (Dict, optional): A dictionary of custom optimizer parameters. Defaults to {}.
            kwargs (Dict, optional): Additional keyword arguments.

        """
        super().__init__()
        assert "inferred_config" in kwargs, "inferred_config not found in initialization arguments"
        inferred_config = kwargs["inferred_config"]
        # Merging the config and inferred config
        config = safe_merge_config(config, inferred_config)
        self.custom_loss = custom_loss
        self.custom_metrics = custom_metrics
        self.custom_metrics_prob_inputs = custom_metrics_prob_inputs
        self.custom_optimizer = custom_optimizer
        self.custom_optimizer_params = custom_optimizer_params
        self.kwargs = kwargs
        # Updating config with custom parameters for experiment tracking
        if self.custom_loss is not None:
            config.loss = str(self.custom_loss)
        if self.custom_metrics is not None:
            # Adding metrics to config for hparams logging and tracking
            config.metrics = []
            config.metrics_params = []
            for metric in self.custom_metrics:
                if isinstance(metric, partial):
                    # extracting func names from partial functions
                    config.metrics.append(metric.func.__name__)
                    config.metrics_params.append(metric.keywords)
                else:
                    config.metrics.append(metric.__name__)
                    config.metrics_params.append(vars(metric))
            if config.task == "classification":
                config.metrics_prob_input = self.custom_metrics_prob_inputs
                for i, mp in enumerate(config.metrics_params):
                    mp.sub_params_list = []
                    for j, num_classes in enumerate(inferred_config.output_cardinality):
                        config.metrics_params[i].sub_params_list.append(
                            OmegaConf.create(
                                {
                                    "task": mp.get("task", "multiclass"),
                                    "num_classes": mp.get("num_classes", num_classes),
                                }
                            )
                        )

        # Updating default metrics in config
        elif config.task == "classification":
            # Adding metric_params to config for classification task
            for i, mp in enumerate(config.metrics_params):
                mp.sub_params_list = []
                for j, num_classes in enumerate(inferred_config.output_cardinality):
                    # config.metrics_params[i][j]["task"] = mp.get("task", "multiclass")
                    # config.metrics_params[i][j]["num_classes"] = mp.get("num_classes", num_classes)

                    config.metrics_params[i].sub_params_list.append(
                        OmegaConf.create(
                            {"task": mp.get("task", "multiclass"), "num_classes": mp.get("num_classes", num_classes)}
                        )
                    )

                    if config.metrics[i] in (
                        "accuracy",
                        "precision",
                        "recall",
                        "precision_recall",
                        "specificity",
                        "f1_score",
                        "fbeta_score",
                    ):
                        config.metrics_params[i].sub_params_list[j]["top_k"] = mp.get("top_k", 1)

        if self.custom_optimizer is not None:
            config.optimizer = str(self.custom_optimizer.__class__.__name__)
        if len(self.custom_optimizer_params) > 0:
            config.optimizer_params = self.custom_optimizer_params
        self.save_hyperparameters(config)
        # The concatenated output dim of the embedding layer
        self._build_network()
        self._setup_loss()
        self._setup_metrics()
        self._check_and_verify()
        self.do_log_logits = (
            hasattr(self.hparams, "log_logits") and self.hparams.log_logits and self.hparams.log_target == "wandb"
        )
        if self.do_log_logits:
            self._val_logits = []
        if not WANDB_INSTALLED and self.do_log_logits:
            self.do_log_logits = False
            warnings.warn(
                "Wandb is not installed. Please install wandb to log logits. "
                "You can install wandb using pip install wandb or install PyTorch Tabular"
                " using pip install pytorch-tabular[extra]"
            )
        if not PLOTLY_INSTALLED and self.do_log_logits:
            self.do_log_logits = False
            warnings.warn(
                "Plotly is not installed. Please install plotly to log logits. "
                "You can install plotly using pip install plotly or install PyTorch Tabular"
                " using pip install pytorch-tabular[extra]"
            )

    @abstractmethod
    def _build_network(self):
        pass

    @property
    def backbone(self):
        raise NotImplementedError("backbone property needs to be implemented by inheriting classes")

    @property
    def embedding_layer(self):
        raise NotImplementedError("embedding_layer property needs to be implemented by inheriting classes")

    @property
    def head(self):
        raise NotImplementedError("head property needs to be implemented by inheriting classes")

    def _check_and_verify(self):
        assert hasattr(self, "backbone"), "Model has no attribute called `backbone`"
        assert hasattr(self.backbone, "output_dim"), "Backbone needs to have attribute `output_dim`"
        assert hasattr(self, "head"), "Model has no attribute called `head`"

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.hparams.head)
        return _head_callable(
            in_units=self.backbone.output_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.hparams.head_config),
        )  # output_dim auto-calculated from other configs

    def _setup_loss(self):
        if self.custom_loss is None:
            try:
                self.loss = getattr(nn, self.hparams.loss)()
            except AttributeError as e:
                logger.error(f"{self.hparams.loss} is not a valid loss defined in the torch.nn module")
                raise e
        else:
            self.loss = self.custom_loss

    def _setup_metrics(self):
        if self.custom_metrics is None:
            self.metrics = []
            task_module = torchmetrics.functional
            for metric in self.hparams.metrics:
                try:
                    self.metrics.append(getattr(task_module, metric))
                except AttributeError as e:
                    logger.error(
                        f"{metric} is not a valid functional metric defined in the torchmetrics.functional module"
                    )
                    raise e
        else:
            self.metrics = self.custom_metrics

    def calculate_loss(self, output: Dict, y: torch.Tensor, tag: str, sync_dist: bool = False) -> torch.Tensor:
        """Calculates the loss for the model.

        Args:
            output (Dict): The output dictionary from the model
            y (torch.Tensor): The target tensor
            tag (str): The tag to use for logging
            sync_dist (bool): enable distributed sync of logs

        Returns:
            torch.Tensor: The loss value

        """
        y_hat = output["logits"]
        reg_terms = [k for k, v in output.items() if "regularization" in k]
        reg_loss = 0
        for t in reg_terms:
            # Log only if non-zero
            if output[t] != 0:
                reg_loss += output[t]
                self.log(
                    f"{tag}_{t}_loss",
                    output[t],
                    on_epoch=True,
                    on_step=False,
                    logger=True,
                    prog_bar=False,
                    sync_dist=sync_dist,
                )
        if self.hparams.task == "regression":
            computed_loss = reg_loss
            for i in range(self.hparams.output_dim):
                _loss = self.loss(y_hat[:, i], y[:, i])
                computed_loss += _loss
                if self.hparams.output_dim > 1:
                    self.log(
                        f"{tag}_loss_{i}",
                        _loss,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False,
                        sync_dist=sync_dist,
                    )
        else:
            # TODO loss fails with batch size of 1?
            computed_loss = reg_loss
            start_index = 0
            for i in range(len(self.hparams.output_cardinality)):
                end_index = start_index + self.hparams.output_cardinality[i]
                _loss = self.loss(y_hat[:, start_index:end_index], y[:, i])
                computed_loss += _loss
                if self.hparams.output_dim > 1:
                    self.log(
                        f"{tag}_loss_{i}",
                        _loss,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False,
                        sync_dist=sync_dist,
                    )
                start_index = end_index
        self.log(
            f"{tag}_loss",
            computed_loss,
            on_epoch=(tag in ["valid", "test"]),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        return computed_loss

    def calculate_metrics(
        self, y: torch.Tensor, y_hat: torch.Tensor, tag: str, sync_dist: bool = False
    ) -> List[torch.Tensor]:
        """Calculates the metrics for the model.

        Args:
            y (torch.Tensor): The target tensor

            y_hat (torch.Tensor): The predicted tensor

            tag (str): The tag to use for logging

            sync_dist (bool): enable distributed sync of logs

        Returns:
            List[torch.Tensor]: The list of metric values

        """
        metrics = []
        for metric, metric_str, prob_inp, metric_params in zip(
            self.metrics,
            self.hparams.metrics,
            self.hparams.metrics_prob_input,
            self.hparams.metrics_params,
        ):
            if self.hparams.task == "regression":
                _metrics = []
                for i in range(self.hparams.output_dim):
                    name = metric.func.__name__ if isinstance(metric, partial) else metric.__name__
                    if name == torchmetrics.functional.mean_squared_log_error.__name__:
                        # MSLE should only be used in strictly positive targets. It is undefined otherwise
                        _metric = metric(
                            torch.clamp(y_hat[:, i], min=0),
                            torch.clamp(y[:, i], min=0),
                            **metric_params,
                        )
                    else:
                        _metric = metric(y_hat[:, i], y[:, i], **metric_params)
                    if self.hparams.output_dim > 1:
                        self.log(
                            f"{tag}_{metric_str}_{i}",
                            _metric,
                            on_epoch=True,
                            on_step=False,
                            logger=True,
                            prog_bar=False,
                            sync_dist=sync_dist,
                        )
                    _metrics.append(_metric)
                avg_metric = torch.stack(_metrics, dim=0).sum()
            else:
                _metrics = []
                start_index = 0
                for i, cardinality in enumerate(self.hparams.output_cardinality):
                    end_index = start_index + cardinality
                    y_hat_i = nn.Softmax(dim=-1)(y_hat[:, start_index:end_index].squeeze())
                    if prob_inp:
                        _metric = metric(y_hat_i, y[:, i : i + 1].squeeze(), **metric_params.sub_params_list[i])
                    else:
                        _metric = metric(
                            torch.argmax(y_hat_i, dim=-1), y[:, i : i + 1].squeeze(), **metric_params.sub_params_list[i]
                        )
                    if len(self.hparams.output_cardinality) > 1:
                        self.log(
                            f"{tag}_{metric_str}_{i}",
                            _metric,
                            on_epoch=True,
                            on_step=False,
                            logger=True,
                            prog_bar=False,
                            sync_dist=sync_dist,
                        )
                    _metrics.append(_metric)
                    start_index = end_index
                avg_metric = torch.stack(_metrics, dim=0).sum()
            metrics.append(avg_metric)
            self.log(
                f"{tag}_{metric_str}",
                avg_metric,
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=True,
                sync_dist=sync_dist,
            )
        return metrics

    def data_aware_initialization(self, datamodule):
        """Performs data-aware initialization of the model when defined."""
        pass

    def compute_backbone(self, x: Dict) -> torch.Tensor:
        # Returns output
        x = self.backbone(x)
        return x

    def embed_input(self, x: Dict) -> torch.Tensor:
        return self.embedding_layer(x)

    def apply_output_sigmoid_scaling(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Applies sigmoid scaling to the output of the model if the task is regression and the target range is
        defined.

        Args:
            y_hat (torch.Tensor): The output of the model

        Returns:
            torch.Tensor: The output of the model with sigmoid scaling applied

        """
        if (self.hparams.task == "regression") and (self.hparams.target_range is not None):
            for i in range(self.hparams.output_dim):
                y_min, y_max = self.hparams.target_range[i]
                y_hat[:, i] = y_min + nn.Sigmoid()(y_hat[:, i]) * (y_max - y_min)
        return y_hat

    def pack_output(self, y_hat: torch.Tensor, backbone_features: torch.tensor) -> Dict[str, Any]:
        """Packs the output of the model.

        Args:
            y_hat (torch.Tensor): The output of the model

            backbone_features (torch.tensor): The backbone features

        Returns:
            The packed output of the model

        """
        # if self.head is the Identity function it means that we cannot extract backbone features,
        # because the model cannot be divide in backbone and head (i.e. TabNet)
        if type(self.head) is nn.Identity:
            return {"logits": y_hat}
        return {"logits": y_hat, "backbone_features": backbone_features}

    def compute_head(self, backbone_features: Tensor) -> Dict[str, Any]:
        """Computes the head of the model.

        Args:
            backbone_features (Tensor): The backbone features

        Returns:
            The output of the model

        """
        y_hat = self.head(backbone_features)
        y_hat = self.apply_output_sigmoid_scaling(y_hat)
        return self.pack_output(y_hat, backbone_features)

    def forward(self, x: Dict) -> Dict[str, Any]:
        """The forward pass of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys

        """
        x = self.embed_input(x)
        x = self.compute_backbone(x)
        return self.compute_head(x)

    def predict(self, x: Dict, ret_model_output: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """Predicts the output of the model.

        Args:
            x (Dict): The input of the model with 'continuous' and 'categorical' keys

            ret_model_output (bool): If True, the method returns the output of the model

        Returns:
            The output of the model

        """
        assert self.hparams.task != "ssl", "It's not allowed to use the method predict in case of ssl task"
        ret_value = self.forward(x)
        if ret_model_output:
            return ret_value.get("logits"), ret_value
        return ret_value.get("logits")

    def forward_pass(self, batch):
        return self(batch), None

    def extract_embedding(self):
        """Extracts the embedding of the model.

        This is used in `CategoricalEmbeddingTransformer`

        """
        if self.hparams.categorical_dim > 0:
            if not isinstance(self.embedding_layer, PreEncoded1dLayer):
                return self.embedding_layer.cat_embedding_layers
            else:
                raise ValueError(
                    "Cannot extract embedding for PreEncoded1dLayer. Please use a different embedding layer."
                )
        else:
            raise ValueError(
                "Model has been trained with no categorical feature and therefore can't be used"
                " as a Categorical Encoder"
            )

    def training_step(self, batch, batch_idx):
        output, y = self.forward_pass(batch)
        # y is not None for SSL task.Rest of the tasks target is
        # fetched from the batch
        y = batch["target"] if y is None else y
        y_hat = output["logits"]
        loss = self.calculate_loss(output, y, tag="train")
        self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            output, y = self.forward_pass(batch)
            # y is not None for SSL task.Rest of the tasks target is
            # fetched from the batch
            y = batch["target"] if y is None else y
            y_hat = output["logits"]
            self.calculate_loss(output, y, tag="valid", sync_dist=True)
            self.calculate_metrics(y, y_hat, tag="valid", sync_dist=True)
        return y_hat, y

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            output, y = self.forward_pass(batch)
            # y is not None for SSL task. Rest of the tasks target is
            # fetched from the batch
            y = batch["target"] if y is None else y
            y_hat = output["logits"]
            self.calculate_loss(output, y, tag="test", sync_dist=True)
            self.calculate_metrics(y, y_hat, tag="test", sync_dist=True)
        return y_hat, y

    def configure_optimizers(self):
        if self.custom_optimizer is None:
            # Loading from the config
            try:
                self._optimizer = _create_optimizer(self.hparams.optimizer)
                opt = self._optimizer(
                    self.parameters(),
                    lr=self.hparams.learning_rate,
                    **self.hparams.optimizer_params,
                )
            except AttributeError as e:
                logger.error(f"{self.hparams.optimizer} is not a valid optimizer defined in the torch.optim module")
                raise e
        else:
            # Loading from custom fit arguments
            self._optimizer = _create_optimizer(self.custom_optimizer)

            opt = self._optimizer(
                self.parameters(),
                lr=self.hparams.learning_rate,
                **self.custom_optimizer_params,
            )
        if self.hparams.lr_scheduler is not None:
            try:
                self._lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)
            except AttributeError as e:
                logger.error(
                    f"{self.hparams.lr_scheduler} is not a valid learning rate sheduler defined"
                    f" in the torch.optim.lr_scheduler module"
                )
                raise e
            if isinstance(self._lr_scheduler, torch.optim.lr_scheduler._LRScheduler):
                return {
                    "optimizer": opt,
                    "lr_scheduler": self._lr_scheduler(opt, **self.hparams.lr_scheduler_params),
                }
            return {
                "optimizer": opt,
                "lr_scheduler": self._lr_scheduler(opt, **self.hparams.lr_scheduler_params),
                "monitor": self.hparams.lr_scheduler_monitor_metric,
            }
        else:
            return opt

    def create_plotly_histogram(self, arr, name, bin_dict=None):
        fig = go.Figure()
        for i in range(arr.shape[-1]):
            fig.add_trace(
                go.Histogram(
                    x=arr[:, i],
                    histnorm="probability",
                    name=f"{name}_{i}",
                    xbins=bin_dict,  # dict(start=0.0, end=1.0, size=0.1),  # bins used for histogram
                )
            )
        # Overlay both histograms
        fig.update_layout(
            barmode="overlay",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        return fig

    def on_validation_batch_end(self, outputs, batch, batch_idx: int) -> None:
        if self.do_log_logits:
            self._val_logits.append(outputs[0][0])
        super().on_validation_batch_end(outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        if self.do_log_logits:
            logits = torch.cat(self._val_logits).detach().cpu()
            self._val_logits = []
            fig = self.create_plotly_histogram(logits, "logits")
            wandb.log(
                {"valid_logits": wandb.Plotly(fig), "global_step": self.global_step},
                commit=False,
            )
        super().on_validation_epoch_end()

    def reset_weights(self):
        reset_all_weights(self.backbone)
        reset_all_weights(self.head)
        reset_all_weights(self.embedding_layer)

    def feature_importance(self) -> DataFrame:
        """Returns a dataframe with feature importance for the model."""
        if hasattr(self.backbone, "feature_importance_"):
            imp = self.backbone.feature_importance_
            n_feat = len(self.hparams.categorical_cols + self.hparams.continuous_cols)
            if self.hparams.categorical_dim > 0:
                if imp.shape[0] != n_feat:
                    # Combining Cat Embedded Dimensions to a single one by averaging
                    wt = []
                    norm = []
                    ft_idx = 0
                    for _, embd_dim in self.hparams.embedding_dims:
                        wt.extend([ft_idx] * embd_dim)
                        norm.append(embd_dim)
                        ft_idx += 1
                    for _ in self.hparams.continuous_cols:
                        wt.extend([ft_idx])
                        norm.append(1)
                        ft_idx += 1
                    imp = np.bincount(wt, weights=imp) / np.array(norm)
                else:
                    # For models like FTTransformer, we dont need to do anything
                    # It takes categorical and continuous as individual 2-D features
                    pass
            importance_df = DataFrame(
                {
                    "Features": self.hparams.categorical_cols + self.hparams.continuous_cols,
                    "importance": imp,
                }
            )
            return importance_df
        else:
            raise ValueError("Feature Importance unavailable for this model.")


class _GenericModel(BaseModel):
    def __init__(
        self,
        backbone: nn.Module,
        head: str,
        head_config: Dict,
        config: DictConfig,
        custom_loss: Optional[torch.nn.Module] = None,
        custom_metrics: Optional[List[Callable]] = None,
        custom_metrics_prob_inputs: Optional[List[bool]] = None,
        custom_optimizer: Optional[torch.optim.Optimizer] = None,
        custom_optimizer_params: Dict = {},
        **kwargs,
    ):
        assert hasattr(config, "loss") or custom_loss is not None, "Loss function not defined in the config"
        super().__init__(
            config,
            custom_loss,
            custom_metrics,
            custom_metrics_prob_inputs,
            custom_optimizer,
            custom_optimizer_params,
            head=head,
            head_config=head_config,
            backbone=backbone,
            **kwargs,
        )
        # self._backbone = backbone
        # self.head = head
        # self.head_config = head_config
        # backbone.mode = "fine_tune"
        # self._head = self._get_head_from_config()

    def _get_head_from_config(self):
        _head_callable = getattr(blocks, self.kwargs.get("head"))
        return _head_callable(
            in_units=self.backbone.output_dim,
            output_dim=self.hparams.output_dim,
            config=_head_callable._config_template(**self.kwargs.get("head_config")),
        )  # output_dim auto-calculated from other configs

    @property
    def backbone(self):
        return self._backbone

    @property
    def embedding_layer(self):
        return self.backbone.embedding_layer

    @property
    def head(self):
        return self._head

    def _build_network(self):
        # Leaving it blank because of some parameter passing issues
        # all components are initialized in the init function
        self._backbone = self.kwargs.get("backbone")
        self._head = self._get_head_from_config()


class _CaptumModel(nn.Module):
    def __init__(self, model: BaseModel):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor):
        x = self.model.compute_backbone(x)
        return self.model.compute_head(x)["logits"]
