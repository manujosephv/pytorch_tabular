After defining all the configs, we need to put it all together and this is where `TabularModel` comes in. `TabularModel` is the core work horse, which orchestrates and sets everything up.

`TabularModel` parses the configs and:

1. initializes the model
2. sets up the experiment tracking framework
3. initializes and sets up the `TabularDatamodule` which handles all the data transformations and preparation of the DataLoaders
4. sets up the callbacks and the Pytorch Lightning Trainer
5. enables you to train, save, load, and predict

## Initializing Tabular Model

### Basic Usage:

-   `data_config`: DataConfig: DataConfig object or path to the yaml file.
-   `model_config`: ModelConfig: A subclass of ModelConfig or path to the yaml file. Determines which model to run from the type of config.
-   `optimizer_config`: OptimizerConfig: OptimizerConfig object or path to the yaml file.
-   `trainer_config`: TrainerConfig: TrainerConfig object or path to the yaml file.
-   `experiment_config`: ExperimentConfig: ExperimentConfig object or path to the yaml file.

#### Usage Example

```python
tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
    experiment_config=experiment_config
)
```

### Advanced Usage

-   `config`: DictConfig: Another way of initializing `TabularModel` is with an `Dictconfig` from `omegaconf`. Although not recommended, you can create a normal dictionary with all the parameters dumped into it and create a `DictConfig` from `omegaconf` and pass it here. The downside is that you'll be skipping all the validation(both type validation and logical validations). This is primarily used internally to load a saved model from a checkpoint.
-   `model_callable`: Optional[Callable]:  Usually, the model callable and parameters are inferred from the ModelConfig. But in special cases, like when working with a custom model, you can pass the class(not the initialized object) to this parameter and override the config based initialization.

## Functions



::: pytorch_tabular.TabularModel.fit
::: pytorch_tabular.TabularModel.evaluate
::: pytorch_tabular.TabularModel.predict
::: pytorch_tabular.TabularModel.save_model
::: pytorch_tabular.TabularModel.load_from_checkpoint
::: pytorch_tabular.TabularModel.find_learning_rate