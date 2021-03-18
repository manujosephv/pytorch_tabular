# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
"""Mixture Density Models"""
import logging
import math
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.autograd import Variable
from torch.distributions import Categorical

from pytorch_tabular.models.autoint import AutoIntBackbone
from pytorch_tabular.models.category_embedding import FeedForwardBackbone
from pytorch_tabular.models.node import NODEBackbone
from pytorch_tabular.models.node import utils as utils

from ..base_model import BaseModel

try:
    import wandb

    WANDB_INSTALLED = True
except ImportError:
    WANDB_INSTALLED = False
logger = logging.getLogger(__name__)

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)
LOG2PI = math.log(2 * math.pi)


class MixtureDensityHead(nn.Module):
    def __init__(self, config: DictConfig, **kwargs):
        self.hparams = config
        super().__init__()
        self._build_network()

    def _build_network(self):
        self.pi = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.pi.weight)
        self.sigma = nn.Linear(
            self.hparams.input_dim,
            self.hparams.num_gaussian,
            bias=self.hparams.sigma_bias_flag,
        )
        self.mu = nn.Linear(self.hparams.input_dim, self.hparams.num_gaussian)
        nn.init.normal_(self.mu.weight)
        if self.hparams.mu_bias_init is not None:
            for i, bias in enumerate(self.hparams.mu_bias_init):
                nn.init.constant_(self.mu.bias[i], bias)

    def forward(self, x):
        pi = self.pi(x)
        sigma = self.sigma(x)
        # Applying modified ELU activation
        sigma = nn.ELU()(sigma) + 1 + 1e-15
        mu = self.mu(x)
        return pi, sigma, mu

    def gaussian_probability(self, sigma, mu, target, log=False):
        """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

        Arguments:
            sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
                size, G is the number of Gaussians, and O is the number of
                dimensions per Gaussian.
            mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
                number of Gaussians, and O is the number of dimensions per Gaussian.
            target (BxI): A batch of target. B is the batch size and I is the number of
                input dimensions.
        Returns:
            probabilities (BxG): The probability of each point in the probability
                of the distribution in the corresponding sigma/mu index.
        """
        target = target.expand_as(sigma)
        if log:
            ret = (
                -torch.log(sigma)
                - 0.5 * LOG2PI
                - 0.5 * torch.pow((target - mu) / sigma, 2)
            )
        else:
            ret = (ONEOVERSQRT2PI / sigma) * torch.exp(
                -0.5 * ((target - mu) / sigma) ** 2
            )
        return ret  # torch.prod(ret, 2)

    def log_prob(self, pi, sigma, mu, y):
        log_component_prob = self.gaussian_probability(sigma, mu, y, log=True)
        log_mix_prob = torch.log(
            nn.functional.gumbel_softmax(
                pi, tau=self.hparams.softmax_temperature, dim=-1
            )
            + 1e-15
        )
        return torch.logsumexp(log_component_prob + log_mix_prob, dim=-1)

    def sample(self, pi, sigma, mu):
        """Draw samples from a MoG."""
        categorical = Categorical(pi)
        pis = categorical.sample().unsqueeze(1)
        sample = Variable(sigma.data.new(sigma.size(0), 1).normal_())
        # Gathering from the n Gaussian Distribution based on sampled indices
        sample = sample * sigma.gather(1, pis) + mu.gather(1, pis)
        return sample

    def generate_samples(self, pi, sigma, mu, n_samples=None):
        if n_samples is None:
            n_samples = self.hparams.n_samples
        samples = []
        softmax_pi = nn.functional.gumbel_softmax(
            pi, tau=self.hparams.softmax_temperature, dim=-1
        )
        assert (
            softmax_pi < 0
        ).sum().item() == 0, "pi parameter should not have negative"
        for _ in range(n_samples):
            samples.append(self.sample(softmax_pi, sigma, mu))
        samples = torch.cat(samples, dim=1)
        return samples

    def generate_point_predictions(self, pi, sigma, mu, n_samples=None):
        # Sample using n_samples and take average
        samples = self.generate_samples(pi, sigma, mu, n_samples)
        if self.hparams.central_tendency == "mean":
            y_hat = torch.mean(samples, dim=-1)
        elif self.hparams.central_tendency == "median":
            y_hat = torch.median(samples, dim=-1).values
        return y_hat


class BaseMDN(BaseModel, metaclass=ABCMeta):
    def __init__(self, config: DictConfig, **kwargs):
        assert config.task == "regression", "MDN is only implemented for Regression"
        assert config.output_dim == 1, "MDN is not implemented for multi-targets"
        if config.target_range is not None:
            logger.warning("MDN does not use target range. Ignoring it.")
        super().__init__(config, **kwargs)

    @abstractmethod
    def unpack_input(self, x: Dict):
        pass

    def forward(self, x: Dict):
        x = self.unpack_input(x)
        x = self.backbone(x)
        pi, sigma, mu = self.mdn(x)
        return {"pi": pi, "sigma": sigma, "mu": mu, "backbone_features": x}

    def predict(self, x: Dict):
        ret_value = self.forward(x)
        return self.mdn.generate_point_predictions(
            ret_value["pi"], ret_value["sigma"], ret_value["mu"]
        )

    def sample(self, x: Dict, n_samples: Optional[int] = None, ret_model_output=False):
        ret_value = self.forward(x)
        samples = self.mdn.generate_samples(
            ret_value["pi"], ret_value["sigma"], ret_value["mu"], n_samples
        )
        if ret_model_output:
            return samples, ret_value
        else:
            return samples

    def calculate_loss(self, y, pi, sigma, mu, tag="train"):
        # NLL Loss
        log_prob = self.mdn.log_prob(pi, sigma, mu, y)
        loss = torch.mean(-log_prob)
        if self.hparams.mdn_config.weight_regularization is not None:
            sigma_l1_reg = 0
            pi_l1_reg = 0
            mu_l1_reg = 0
            if self.hparams.mdn_config.lambda_sigma > 0:
                # Weight Regularization Sigma
                sigma_params = torch.cat(
                    [x.view(-1) for x in self.mdn.sigma.parameters()]
                )
                sigma_l1_reg = self.hparams.mdn_config.lambda_sigma * torch.norm(
                    sigma_params, self.hparams.mdn_config.weight_regularization
                )
            if self.hparams.mdn_config.lambda_pi > 0:
                pi_params = torch.cat([x.view(-1) for x in self.mdn.pi.parameters()])
                pi_l1_reg = self.hparams.mdn_config.lambda_sigma * torch.norm(
                    pi_params, self.hparams.mdn_config.weight_regularization
                )
            if self.hparams.mdn_config.lambda_mu > 0:
                mu_params = torch.cat([x.view(-1) for x in self.mdn.mu.parameters()])
                mu_l1_reg = self.hparams.mdn_config.lambda_mu * torch.norm(
                    mu_params, self.hparams.mdn_config.weight_regularization
                )

            loss = loss + sigma_l1_reg + pi_l1_reg + mu_l1_reg
        self.log(
            f"{tag}_loss",
            loss,
            on_epoch=(tag == "valid"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        loss = self.calculate_loss(
            y, ret_value["pi"], ret_value["sigma"], ret_value["mu"], tag="train"
        )
        if self.hparams.mdn_config.speedup_training:
            pass
        else:
            y_hat = self.mdn.generate_point_predictions(
                ret_value["pi"], ret_value["sigma"], ret_value["mu"]
            )
            _ = self.calculate_metrics(y, y_hat, tag="train")
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        _ = self.calculate_loss(
            y, ret_value["pi"], ret_value["sigma"], ret_value["mu"], tag="valid"
        )
        y_hat = self.mdn.generate_point_predictions(
            ret_value["pi"], ret_value["sigma"], ret_value["mu"]
        )
        _ = self.calculate_metrics(y, y_hat, tag="valid")
        return y_hat, y, ret_value

    def test_step(self, batch, batch_idx):
        y = batch["target"]
        ret_value = self(batch)
        _ = self.calculate_loss(
            y, ret_value["pi"], ret_value["sigma"], ret_value["mu"], tag="test"
        )
        y_hat = self.mdn.generate_point_predictions(
            ret_value["pi"], ret_value["sigma"], ret_value["mu"]
        )
        _ = self.calculate_metrics(y, y_hat, tag="test")
        return y_hat, y

    def validation_epoch_end(self, outputs) -> None:
        do_log_logits = (
            self.hparams.log_logits
            and self.hparams.log_target == "wandb"
            and WANDB_INSTALLED
        )
        pi = [
            nn.functional.gumbel_softmax(
                output[2]["pi"], tau=self.hparams.mdn_config.softmax_temperature, dim=-1
            )
            for output in outputs
        ]
        pi = torch.cat(pi).detach().cpu()
        for i in range(self.hparams.mdn_config.num_gaussian):
            self.log(
                f"mean_pi_{i}",
                pi[:, i].mean(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
            )

        mu = [output[2]["mu"] for output in outputs]
        mu = torch.cat(mu).detach().cpu()
        for i in range(self.hparams.mdn_config.num_gaussian):
            self.log(
                f"mean_mu_{i}",
                mu[:, i].mean(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
            )

        sigma = [output[2]["sigma"] for output in outputs]
        sigma = torch.cat(sigma).detach().cpu()
        for i in range(self.hparams.mdn_config.num_gaussian):
            self.log(
                f"mean_sigma_{i}",
                sigma[:, i].mean(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
            )

        if do_log_logits:
            logits = [output[0] for output in outputs]
            logits = torch.cat(logits).detach().cpu()
            fig = self.create_plotly_histogram(logits.unsqueeze(1), "logits")
            wandb.log(
                {
                    "valid_logits": fig,
                    "global_step": self.global_step,
                },
                commit=False,
            )
            if self.hparams.mdn_config.log_debug_plot:
                fig = self.create_plotly_histogram(
                    pi, "pi", bin_dict=dict(start=0.0, end=1.0, size=0.1)
                )
                wandb.log(
                    {
                        "valid_pi": fig,
                        "global_step": self.global_step,
                    },
                    commit=False,
                )

                fig = self.create_plotly_histogram(mu, "mu")
                wandb.log(
                    {
                        "valid_mu": fig,
                        "global_step": self.global_step,
                    },
                    commit=False,
                )

                fig = self.create_plotly_histogram(sigma, "sigma")
                wandb.log(
                    {
                        "valid_sigma": fig,
                        "global_step": self.global_step,
                    },
                    commit=False,
                )


class CategoryEmbeddingMDN(BaseMDN):
    def __init__(self, config: DictConfig, **kwargs):
        self.embedding_cat_dim = sum([y for x, y in config.embedding_dims])
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Embedding layers
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(x, y) for x, y in self.hparams.embedding_dims]
        )
        # Continuous Layers
        if self.hparams.batch_norm_continuous_input:
            self.normalizing_batch_norm = nn.BatchNorm1d(self.hparams.continuous_dim)
        # Backbone
        self.backbone = FeedForwardBackbone(self.hparams)
        # Adding the last layer
        self.hparams.mdn_config.input_dim = self.backbone.output_dim
        self.mdn = MixtureDensityHead(self.hparams.mdn_config)

    def unpack_input(self, x: Dict):
        continuous_data, categorical_data = x["continuous"], x["categorical"]
        if self.embedding_cat_dim != 0:
            x = []
            # for i, embedding_layer in enumerate(self.embedding_layers):
            #     x.append(embedding_layer(categorical_data[:, i]))
            x = [
                embedding_layer(categorical_data[:, i])
                for i, embedding_layer in enumerate(self.embedding_layers)
            ]
            x = torch.cat(x, 1)

        if self.hparams.continuous_dim != 0:
            if self.hparams.batch_norm_continuous_input:
                continuous_data = self.normalizing_batch_norm(continuous_data)

            if self.embedding_cat_dim != 0:
                x = torch.cat([x, continuous_data], 1)
            else:
                x = continuous_data
        return x


class NODEMDN(BaseMDN):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        self.hparams.node_input_dim = (
            self.hparams.continuous_dim + self.hparams.categorical_dim
        )
        backbone = NODEBackbone(self.hparams)
        # average first n channels of every tree, where n is the number of output targets for regression
        # and number of classes for classification

        def subset(x):
            return x[..., :].mean(dim=-2)

        output_response = utils.Lambda(subset)
        self.backbone = nn.Sequential(backbone, output_response)
        # Adding the last layer
        self.hparams.mdn_config.input_dim = backbone.output_dim
        self.mdn = MixtureDensityHead(self.hparams.mdn_config)

    def unpack_input(self, x: Dict):
        # unpacking into a tuple
        x = x["categorical"], x["continuous"]
        # eliminating None in case there is no categorical or continuous columns
        x = (item for item in x if len(item) > 0)
        x = torch.cat(tuple(x), dim=1)
        return x


class AutoIntMDN(BaseMDN):
    def __init__(self, config: DictConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _build_network(self):
        # Backbone
        self.backbone = AutoIntBackbone(self.hparams)
        # Adding the last layer
        self.hparams.mdn_config.input_dim = self.backbone.output_dim
        self.mdn = MixtureDensityHead(self.hparams.mdn_config)

    def unpack_input(self, x: Dict):
        # Returning the dict because autoInt backbone expects the dict output
        return x
