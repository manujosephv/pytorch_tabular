from pytorch_tabular.models.tabnet.config import TabNetModelConfig
from sklearn.datasets import fetch_covtype

# from torch.utils import data
from pytorch_tabular.config import (
    DataConfig,
    ExperimentConfig,
    ExperimentRunManager,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
from pytorch_tabular.models.node.config import NodeConfig
from pytorch_tabular.models.category_embedding.config import (
    CategoryEmbeddingModelConfig,
)
from pytorch_tabular.models.category_embedding.category_embedding_model import (
    CategoryEmbeddingModel,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl
import numpy as np
from sklearn.model_selection import train_test_split


X_train = pd.read_csv("data/mnm/dust_primer_cb_train.csv")
y_train = pd.read_csv("data/mnm/y_train_primer.csv").loc[:, "dust"]
orig_train = X_train.join(y_train)
target = ["dust"]
categorical_cols = [
    "PLATFORM",
    "PRIMAR",
    "PRIMAR_Color_Intensity",
    "PRIMAR_Color_M_S",
    "Prev_PLATFORM",
    "Prev_PRIMAR",
    "Prev_PRIMAR_Color_Intensity",
    "Prev_PRIMAR_Color_M_S",
    "Prev_TOPCOAT",
    "Prev_TOPCOAT_Color_Intensity",
    "Prev_TOPCOAT_Color_M_S",
    "TOPCOAT",
    "TOPCOAT_Color_Intensity",
    "TOPCOAT_Color_M_S",
    "bc_hum_threshold",
    "bc_temp_threshold",
    "cc_hum_threshold",
    "cc_temp_threshold",
    "primer_hum_threshold",
    "primer_temp_threshold",
]
numerical_cols = [
    "ER_day",
    "TOPCOAT_last_used",
    "T_diff_prev",
    "amb_hum_last_recorded",
    "amb_temp_last_recorded",
    "base_coat_hum_last_recorded",
    "base_coat_temp_last_recorded",
    "bc_intvl_heat_index_mean",
    "bc_intvl_wind_chill_index_mean",
    "cc_intvl_heat_index_mean",
    "cc_intvl_wind_chill_index_mean",
    "clear_coat_hum_last_recorded",
    "clear_coat_temp_last_recorded",
    "dewpoint_amb",
    "dewpoint_bc",
    "dewpoint_cc",
    "first_run_of_day_f",
    "hour_of_run",
    "lunch_flag",
    "month_of_run",
    "num_vehicles_in_bc",
    "num_vehicles_in_cc",
    "paint_pressure_last_record",
    "paint_temp_last_record",
    "primer_heatup_zone1_max",
    "primer_heatup_zone1_mean",
    "primer_heatup_zone1_min",
    "primer_heatup_zone2_max",
    "primer_heatup_zone2_mean",
    "primer_heatup_zone2_min",
    "primer_holding_zone1_max",
    "primer_holding_zone1_mean",
    "primer_holding_zone1_min",
    "primer_holding_zone2_max",
    "primer_holding_zone2_mean",
    "primer_holding_zone2_min",
    "primer_intvl_amb_hum_max",
    "primer_intvl_amb_hum_mean",
    "primer_intvl_amb_hum_min",
    "primer_intvl_amb_temp_max",
    "primer_intvl_amb_temp_mean",
    "primer_intvl_amb_temp_min",
    "primer_intvl_heat_index_mean",
    "primer_intvl_primer_hum_max",
    "primer_intvl_primer_hum_mean",
    "primer_intvl_primer_hum_min",
    "primer_intvl_primer_temp_max",
    "primer_intvl_primer_temp_mean",
    "primer_intvl_primer_temp_min",
    "primer_intvl_wind_chill_index_mean",
    "primer_rh_change",
    "primer_temp_change",
    "pt_primer_lag",
    "pt_tc_lag",
    "run_number_of_body",
    "shift_change_flag",
    "tc_primerexit_lag",
    "viscosity_day",
    "weekday_of_run",
    "wind_speed",
]
date_columns = []
train = orig_train[numerical_cols + categorical_cols + date_columns + target].copy()
train, val = train_test_split(train, test_size=0.2)
aug_train = pd.read_csv("data/mnm/ctgan_synthetic.csv")[numerical_cols + categorical_cols + date_columns + target]
aug_train['dust'] = aug_train['dust'].astype(int)
train = pd.concat([train, aug_train], sort=False)

data_config = DataConfig(
    target=target,
    continuous_cols=numerical_cols,
    categorical_cols=categorical_cols,
    # continuous_feature_transform="quantile_uniform",
    normalize_continuous_features=True,
)
# model_config = CategoryEmbeddingModelConfig(task="classification")
model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="2048-1024-2048",
    activation="SELU",
    metrics=["auroc"],
)
# model_config = NodeConfig(
#     task="classification",
#     num_layers=2,
#     num_trees=1024,
#     depth=6,
#     learning_rate=1e-1,
#     embed_categorical=True,
#     metrics=["auroc"],
# )
# model_config = TabNetModelConfig(
#     task="classification",
#     metrics=["auroc"],
# )
trainer_config = TrainerConfig(
    auto_lr_find=True,
    batch_size=1024,
    max_epochs=1000,
    gpus=1,
    early_stopping_patience=10,
    checkpoints_save_top_k=1,
    # track_grad_norm=2,
    # gradient_clip_val=10,
)
# experiment_config = ExperimentConfig(project_name="Tabular_test")
optimizer_config = OptimizerConfig(optimizer_params={"weight_decay": 1e-2})

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    # experiment_config=experiment_config,
)

import torch_optimizer as custom_optim
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean',**kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6 # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)
        
        
        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss

class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=[0.25,0.75], gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float,int)):
            assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1-self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous() # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1)) # [N,d1*d2..,C]-> [N*d1*d2..,C]
        logit = nn.Softmax(dim=-1)(logit)
        target = target.view(-1, 1) # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0,target.view(-1))
            logpt = alpha_class*logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

# from sklearn.metrics import roc_auc_score

# def auc_roc(y_hat, y):


tabular_model.fit(
    train=train,
    valid=val,
    loss=FocalLoss_Ori(num_class=2),
    # metrics=[lambda y_hat, y: roc_auc_score(y_hat.detach().cpu().numpy(), y.detach().cpu().numpy(), multi_class='ovr')],
    optimizer=custom_optim.QHAdam,
    optimizer_params={"nus": (0.7, 1.0), "betas": (0.95, 0.998)},
)

result = tabular_model.evaluate(val)
print(result)
pred_df = tabular_model.predict(val)
pred_df.to_csv("output/mnm.csv")