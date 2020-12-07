from sklearn.datasets import fetch_california_housing
from config.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    TrainerConfig,
)
import pandas as pd
from omegaconf import OmegaConf
from pytorch_tabular.tabular_datamodule import TabularDatamodule
from pytorch_tabular.tabular_model import TabularModel
import pytorch_lightning as pl

dataset = fetch_california_housing(data_home="data", as_frame=True)
dataset.frame["HouseAgeBin"] = pd.qcut(dataset.frame["HouseAge"], q=4)
dataset.frame.HouseAgeBin = "age_" + dataset.frame.HouseAgeBin.cat.codes.astype(str)

test_idx = dataset.frame.sample(int(0.2 * len(dataset.frame)), random_state=42).index
test = dataset.frame[dataset.frame.index.isin(test_idx)]
train = dataset.frame[~dataset.frame.index.isin(test_idx)]

data_config = OmegaConf.structured(
    DataConfig(
        target=dataset.target_names+["MedInc"],
        continuous_cols=[
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ],
        categorical_cols=["HouseAgeBin"],
    )
)
model_config = OmegaConf.structured(ModelConfig(task="regression"))
trainer_config = OmegaConf.structured(TrainerConfig())
experiment_config = OmegaConf.structured(ExperimentConfig(project_name="Tabular_test"))
optimizer_config = OmegaConf.structured(OptimizerConfig())
# print(OmegaConf.to_yaml(data_config))
# print(OmegaConf.to_yaml(model_config))
# print(OmegaConf.to_yaml(trainer_config))
# print(OmegaConf.to_yaml(experiment_config))
# print(OmegaConf.to_yaml(optimizer_config))
# print("")
config = OmegaConf.merge(
    OmegaConf.to_container(data_config),
    OmegaConf.to_container(model_config),
    OmegaConf.to_container(trainer_config),
    OmegaConf.to_container(experiment_config),
    OmegaConf.to_container(optimizer_config),
)
print(OmegaConf.to_yaml(config))

datamodule = TabularDatamodule(train=train, config=config, test=test)
# config.categorical_dim = datamodule._categorical_dim
# config.continuous_dim = datamodule._continuous_dim
# config.output_dim = datamodule._output_dim
config = datamodule.config
model = TabularModel(config)


config.checkpoint_callback = True if config.checkpoints else False
name = config.run_name if config.run_name else f"{config.task}"
if name in config.exp_version_manager.keys():
    uid = config.exp_version_manager[name] + 1
else:
    uid = 1
# TODO Need to chage the versioning mechanism
# config.update_versions(name, uid)
if config.log_target == "tensorboard":
    logger = pl.loggers.TensorBoardLogger(
        name=name, save_dir="lightning_logs", version=uid
    )
elif config.log_target == "wandb":
    logger = pl.loggers.WandbLogger(
        name=f"{name}_{uid}", project=config.project_name, offline=False
    )
    logger.watch(model, log=config.exp_watch, log_freq=config.exp_log_freq)
else:
    raise NotImplementedError(
        f"{config.log_target} is not implemented. Try one of [wandb, tensorboard]"
    )
# config.logger = logger
callbacks = []
if config.early_stopping is not None:
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor=config.early_stopping,
        min_delta=config.early_stopping_min_delta,
        patience=config.early_stopping_patience,
        verbose=False,
        mode=config.early_stopping_mode,
    )
    callbacks.append(early_stop_callback)
if config.checkpoints:
    ckpt_name = f"{config.task_type}-{uid}"
    ckpt_name = ckpt_name.replace(" ", "_") + "_{epoch}-{valid_loss:.2f}"
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor=config.checkpoints,
        dirpath=config.checkpoints_path,
        filename=ckpt_name,
        save_top_k=config.checkpoints_save_top_k,
        mode=config.checkpoints_mode,
    )
    callbacks.append(model_checkpoint)
# else:
#     trainer.checkpoint_callback = False
# config.callbacks = callbacks
# ------------
# training
# ------------
# TODO Debug
datamodule.prepare_data()
# splits/transforms
datamodule.setup("fit")
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()

trainer_args = vars(pl.Trainer()).keys()
trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    **{k: v for k, v in vars(config).items() if k in trainer_args},
)

trainer.tune(model, train_loader, val_loader)
trainer.fit(model, train_loader, val_loader)
