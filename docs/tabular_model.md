After defining all the configs, we need to put it all together and this is where `TabularModel` comes in. `TabularModel` is the core work horse, which orchestrates and sets everything up.

`TabularModel` parses the configs and:

1. initializes the model
1. sets up the experiment tracking framework
1. initializes and sets up the `TabularDatamodule` which handles all the data transformations and preparation of the DataLoaders
1. sets up the callbacks and the Pytorch Lightning Trainer
1. enables you to train, save, load, and predict

## Initializing Tabular Model

### Basic Usage:

- `data_config`: DataConfig: DataConfig object or path to the yaml file.
- `model_config`: ModelConfig: A subclass of ModelConfig or path to the yaml file. Determines which model to run from the type of config.
- `optimizer_config`: OptimizerConfig: OptimizerConfig object or path to the yaml file.
- `trainer_config`: TrainerConfig: TrainerConfig object or path to the yaml file.
- `experiment_config`: ExperimentConfig: ExperimentConfig object or path to the yaml file.

#### Usage Example

```python
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config,
)
```

### Advanced Usage

- `config`: DictConfig: Another way of initializing `TabularModel` is with an `Dictconfig` from `omegaconf`. Although not recommended, you can create a normal dictionary with all the parameters dumped into it and create a `DictConfig` from `omegaconf` and pass it here. The downside is that you'll be skipping all the validation(both type validation and logical validations). This is primarily used internally to load a saved model from a checkpoint.
- `model_callable`: Optional\[Callable\]:  Usually, the model callable and parameters are inferred from the ModelConfig. But in special cases, like when working with a custom model, you can pass the class(not the initialized object) to this parameter and override the config based initialization.

# Training API (Supervised Learning)

There are two APIs for training or 'fit'-ing a model.

1. High-level API
1. Low-level API

The low-level API is more flexible and allows you to customize the training loop. The high-level API is easier to use and is recommended for most use cases.

## High-Level API

::: pytorch_tabular.TabularModel.fit
    options:
        show_root_heading: yes
        heading_level: 4

## Low-Level API

The low-level API is more flexible and allows you to write more complicated logic like cross validation, ensembling, etc. The low-level API is more verbose and requires you to write more code, but it comes with more control to the user.

The `fit` method is split into three sub-methods:

1. `prepare_dataloader`

1. `prepare_model`

1. `train`

### prepare_dataloader

This method is responsible for setting up the `TabularDataModule` and returns the object. You can save this object using `save_dataloader` and load it later using `load_datamodule` to skip the data preparation step. This is useful when you are doing cross validation or ensembling.   

::: pytorch_tabular.TabularModel.prepare_dataloader
    options:
        show_root_heading: yes
        heading_level: 4

### prepare_model

This method is responsible for setting up and initializing the model and takes in the prepared datamodule as an input. It returns the model instance.    

::: pytorch_tabular.TabularModel.prepare_model
    options:
        show_root_heading: yes
        heading_level: 4

### train

This method is responsible for training the model and takes in the prepared datamodule and model as an input. It returns the trained model instance.    

::: pytorch_tabular.TabularModel.train
    options:
        show_root_heading: yes
        heading_level: 4

# Training API (Self-Supervised Learning)

For self-supervised learning, there is a different API because the process is different.

1. [pytorch_tabular.TabularModel.pretrain][]: This method is responsible for pretraining the model. It takes in the the input dataframes, and other parameters to pre-train on the provided data.
1. [pytorch_tabular.TabularModel.create_finetune_model][]: If we want to use the pretrained model for finetuning, we need to create a new model with the pretrained weights. This method is responsible for creating a finetune model. It takes in the pre-trained model and returns a finetune model. The returned object is a separate instance of `TabularModel` and can be used to finetune the model.
1. [pytorch_tabular.TabularModel.finetune][]: This method is responsible for finetuning the model and can only be used with a model which is created through `create_finetune_model`. It takes in the the input dataframes, and other parameters to finetune on the provided data.

!!! note

    The dataframes passed to `pretrain` need not have the target column. But even if you defined the target column in `DataConfig`, it will be ignored. But the dataframes passed to `finetune` must have the target column.

::: pytorch_tabular.TabularModel.pretrain
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.create_finetune_model
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.finetune
    options:
        show_root_heading: yes
        heading_level: 4
# Model Evaluation

::: pytorch_tabular.TabularModel.predict
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.evaluate
    options:
        show_root_heading: yes
        heading_level: 4

# Artifact Saving and Loading

## Saving the Model, Datamodule, and Configs

::: pytorch_tabular.TabularModel.save_config
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.save_datamodule
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.save_model
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.save_model_for_inference
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.save_weights
    options:
        show_root_heading: yes
        heading_level: 4

## Loading the Model and Datamodule

::: pytorch_tabular.TabularModel.load_best_model
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.load_from_checkpoint
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.load_model
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.load_weights
    options:
        show_root_heading: yes
        heading_level: 4

# Other Functions

::: pytorch_tabular.TabularModel.find_learning_rate
    options:
        show_root_heading: yes
        heading_level: 4
::: pytorch_tabular.TabularModel.summary
    options:
        show_root_heading: yes
        heading_level: 4
