Training a Deep Learning model can get arbritarily complex. PyTorch Tabular, by inheriting PyTorch Lightning, offloads the whole workload onto the underlying PyTorch Lightning Framework. It has been made to make training your models easy as a breeze and at the same time give you the flexibility to make the training process your own.

The trainer in PyTorch Tabular have inherited all the features of the Pytorch Lightning trainer, either directly or indirectly. 

## Basic Usage

The parameters that you would set most frequently are:

- `batch_size`: int: Number of samples in each batch of training. Defaults to `64`
- `max_epochs`: int: Maximum number of epochs to be run. The maximum is in case of Early Stopping where this becomes the maximum and without Early Stopping, this is the number of epochs that will be run Defaults to `10`
- `devices`: (Optional[int]): Number of devices to train on (int). -1 uses all available devices. By default uses all available devices (-1)

- `accelerator`: Optional\[str\]: The accelerator to use for training. Can be one of 'cpu','gpu','tpu','ipu','auto'. Defaults to 'auto'.
- `load_best`: int: Flag to load the best model saved during training. This will be ignored if checkpoint saving is turned off. Defaults to True

### Usage Example

```python
trainer_config = TrainerConfig(batch_size=64, max_epochs=10, accelerator="auto")
```

PyTorch Tabular uses Early Stopping by default and monitors `valid_loss` to stop training. Checkpoint saving is also turned on by default, which monitors `valid_loss` and saved the best model in a folder `saved_models`. All of these are configurable as we will see in the next section.

## Advanced Usage

### Early Stopping and Checkpoint Saving

Early Stopping is turned on by default. But you can turn it off by setting `early_stopping` to `None`. On the other hand, if you want to monitor some other metric, you just need to give that metric name in the `early_stopping` parameter. Few other paramters that controls early stopping are:

- `early_stopping_min_delta`: float: The minimum delta in the loss/metric which qualifies as an improvement in early stopping. Defaults to `0.001`
- `early_stopping_mode`: str: The direction in which the loss/metric should be optimized. Choices are `max` and `min`. Defaults to `min`
- `early_stopping_patience`: int: The number of epochs to wait until there is no further improvements in loss/metric. Defaults to `3`
- `min_epochs`: int: Minimum number of epochs to be run. This many epochs are run regardless of the stopping criteria. Defaults to `1`

Checkpoint Saving is also turned on by default and to turn it off you can set the `checkpoints` parameter to `None`. If you want to monitor some other metric, you just need to give that metric name in the `early_stopping` parameter. Few other paramters that controls checkpoint saving are:

- `checkpoints_path`: str: The path where the saved models will be. Defaults to `saved_models`
- `checkpoints_mode`: str: The direction in which the loss/metric should be optimized. Choices are `max` and `min`. Defaults to `min`
- `checkpoints_save_top_k`: int: The number of best models to save. If you want to save more than one best models, you can set this parameter to >1. Defaults to `1`

!!!note
    Make sure the name of the metric/loss you want to track exactly matches the ones in the logs. Recommended way is to run a model and check the results by evaluating the model. From the resulting dictionary, you can pick up a key to track during training.

### Learning Rate Finder

First proposed in this paper [Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) and the subsequently popularized by fast.ai, is a technique to reach the neighbourhood of optimum learning rate without costly search. PyTorch Tabular let's you find the optimal learning rate(using the method proposed in the paper) and automatically use that for training the network. All this can be turned on with a simple flag `auto_lr_find`

We can also run the learning rate finder as a separate step using [pytorch_tabular.TabularModel.find_learning_rate].

## Controlling the Gradients/Optimization

While training, there can be situations where you need to have a heavier control on the gradient optimization process. For eg. if the gradients are exploding, you might want to clip gradient values before each update. `gradient_clip_val` let's you do that.

Sometimes, you might want to accumulate gradients across multiple batches before you do a backward propoagation(may be because a larger batch size does not fit in your GPU). PyTorch Tabular let's you do this with `accumulate_grad_batches`

## Debugging

Many times, you will need to debug a model and see why it is not performing as it is supposed to. Or even, while developing new models, you will need to debug the model a lot. PyTorch Lightning has a few features for this usecase, which Pytorch Tabular has adopted.

To find out performance bottle necks, we can use:

- `profiler`: Optional\[str\]: To profile individual steps during training and assist in identifying bottlenecks. Choices are: `None` `simple` `advanced`. Defaults to `None`

To check if the whole setup runs without errors, we can use:

- `fast_dev_run`: Optional\[str\]: Quick Debug Run of Val. Defaults to `False`

If the model is not learning properly:

- `overfit_batches`: float: Uses this much data of the training set. If nonzero, will use the same training set for validation and testing. If the training dataloaders have shuffle=True, Lightning will automatically disable it. Useful for quickly debugging or trying to overfit on purpose. Defaults to `0`

- `track_grad_norm`: bool: This is only used if experiment tracking is setup. Track and Log Gradient Norms in the logger. -1 by default means no tracking. 1 for the L1 norm, 2 for L2 norm, etc. Defaults to `False`. If the gradient norm falls to zero quickly, then we have a problem.

## Using the entire PyTorch Lightning Trainer

To unlock the full potential of the PyTorch Lightning Trainer, you can use the `trainer_kwargs` parameter. This will let you pass any parameter that is supported by the PyTorch Lightning Trainer. Full documentation can be found [here](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)


::: pytorch_tabular.config.TrainerConfig
    options:
        show_root_heading: yes
