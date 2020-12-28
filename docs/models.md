Choosing which model to use and what parameters to set in those models is specific to a particular dataset. In PyTorch Tabular, you can choose the model and its parameters using the model specific config classes.

## Basic Usage (Common for all ModelConfigs)

While there are separate config classes for each model, all of them share a few core parameters in a `ModelConfig` class.

-   `task`: str: This defines whether we are running a `regression` or `classification` model

### Usage Example
```python
model_config = <ModelSpecificConfig>(task="classification)
```
That's it, Thats the most basic necessity. All the rest is intelligently inferred or set to intelligent defaults.
## Advanced Usage:

### Learning Rate, Loss and Metrics
Adam Optimizer and the `learning_rate` of 1e-3 is a default that is set in PyTorch Tabular. It's a rule of thumb that works in most cases and a good starting point which has worked well empirically. If you want to change the learning rate(which is a pretty important hyperparameter), this is where you should. There is also an automatic way to derive a good learning rate which we will talk about in the TrainerConfig. In that case, Pytorch Tabular will ignore the learning rate set through this parameter

Another key component of the model is the `loss`. Pytorch Tabular can use any loss function from standard PyTorch([`torch.nn`](https://pytorch.org/docs/stable/nn.html#loss-functions)) through this config. By default it is set to `MSELoss` for regression and `CrossEntropyLoss` for classification, which works well for those use cases and are the most popular loss functions used. If you want to use something else specficaly, like `L1Loss`, you just need to mention it in the `loss` parameter

```python
loss = "L1Loss
```
PyTorch Tabular also accepts custom loss functions(which are drop in replacements for the standard loss functions) through the `fit` method in the `TabularModel`.

!!! warning 
    **If you do not know what you are doing, leave the loss functions at default values.** Models return the raw logits and it is the responsibility of the Loss function to apply the right activations.

While the Loss functions drive the gradient based optimization, we keep track of the metrics that we care about during training. By default, PyTorch Tabular tracks Accuracy for classification and Mean Squared Error for regression. You can choose any functional metrics (as a list of strings) from PyTorch Lightning through the config([Classification](https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#functional-metrics-classification) and [Regression](https://pytorch-lightning.readthedocs.io/en/latest/metrics.html#functional-metrics-regression)).

Some metrics need some parameters to work the way we expect it to. For eg. the averaging scheme for a multi-class f1 score. Such parameters can be fed in through `metrics_params`, which is a list of dictionaries holding the parameters for the metrics declared in the same order.

```python
metrics = ["accuracy","f1"]
metrics_params = [{},{num_classes:2}]
```
PyTorch Tabular also accepts custom metric functions with a signature `callable(pred:torch.Tensor,target:torch.Tensor)` through the parameters `custom_metric` (the callable) and `custom_metric_params` (a dictionary with parameters to be passed to the metric) in the `fit` method of the `TabularModel`

### Target Range for Regression
For classification problems, the targets are always 0 or 1, once we one-hot the class labels. But for regression, it's a real valued value between (-inf, inf), theoretically. More practically, it usually is between known bounds. Sometimes, it is an extra burden on the model to learn this bounds and `target_range` is a way to take that burden off the model. This technique was popularized by Jeremy Howard in fast.ai and is quite effective in practice.

If we know that the output value of a regression should be between a `min` and `max` value, we can provide those values as a tuple to `target_range`. In case of multiple targets, we set the `target_range` to be a list of tuples, each entry in the list corresponds to the respective entry in the `target` parameter.

For classification problems, this parameter is ignored.
```python
target_range = [(train[target].min()*0.8,train[target].max()*1.2)]
```

## Available Models

Now let's look at the different models available in PyTorch Tabular and their configurations

### CategoryEmbeddingModel

This is the most basic model in the library. The architecture is pretty simple - a Feed Forward Network with the Categorical Features passed through an learnable embedding layer.

All the parameters have intelligent default values. Let's look at few of them:

-   `layers`: str: Hyphen-separated number of layers and units in the classification head. Defaults to `"128-64-32"`
-   `activation`: str: The activation type in the classification head. The default [activations in PyTorch](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) like ReLU, TanH, LeakyReLU, etc. Defaults to `ReLU`
-   `embedding_dims`: list: The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)
-   `initialization`: str: Initialization scheme for the linear layers. Choices are: `kaiming` `xavier` `random`. Defaults to `kaiming`
-   `use_batch_norm`: bool: Flag to include a BatchNorm layer after each Linear Layer+DropOut. Defaults to `False`
-   `dropout`: float: The probability of the element to be zeroed. This applies to all the linear layers. Defaults to `0.5`
-   `embedding_dropout`: float: The probability of the embedding element to be zeroed. This applies to the concatenated embedding layer. Defaults to `0.5`
-   `batch_norm_continuous_input`: bool: If True, we will normalize the continuous layer by passing it through a BatchNorm layer. Defaults to `True`


### NODEModel

[Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) is a model presented in ICLR 2020 and according to the authors have beaten well-tuned Gradient Boosting models on many datasets. It uses a Neural equivalent of Oblivious Trees(the kind of trees Catboost uses) as the basic building blocks of the architecture.

The basic block, or a "layer" looks something like below(from the paper)

![NODE Architecture](imgs/node_arch.png)

And deeper architectures are created by stacking such layers with residual connections to the input(like DenseNet)

![NODE Architecture](imgs/node_dense_arch.png)

All the parameters have beet set to recommended values from the paper. Let's look at few of them:

-   `num_layers`: int: Number of Oblivious Decision Tree Layers in the Dense Architecture. Defaults to `1`
-   `num_trees`: int: Number of Oblivious Decision Trees in each layer. Defaults to `2048`
-   `depth`: int: The depth of the individual Oblivious Decision Trees. Parameters increase exponentially with the increase in depth. Defaults to `6`
-   `choice_function`: str: Generates a sparse probability distribution to be used as feature weights(aka, soft feature selection). Choices are: `entmax15` `sparsemax`. Defaults to `entmax15`
-   `bin_function`: str: Generates a sparse probability distribution to be used as tree leaf weights. Choices are: `entmax15` `sparsemax`. Defaults to `entmax15`
-   `additional_tree_output_dim`: int: The additional output dimensions which is only used to pass through different layers of the architectures. Only the first output_dim outputs will be used for prediction. Defaults to `3`
-   `input_dropout`: float: Dropout which is applied to the input to the different layers in the Dense Architecture. The probability of the element to be zeroed. Defaults to `0.5`
-   `embed_categorical`: bool: Flag to embed categorical columns using an Embedding Layer. If turned off, the categorical columns are encoded using LeaveOneOutEncoder. Defaults to `False`
-   `embedding_dims`: list: The dimensions of the embedding for each categorical column as a list of tuples (cardinality, embedding_dim). If left empty, will infer using the cardinality of the categorical column using the rule min(50, (x + 1) // 2)
-   `embedding_dropout`: float: The probability of the embedding element to be zeroed. This applies to the concatenated embedding layer. Defaults to `0.5`

**For a complete list of parameters refer to the API Docs**

!!! note
    NODE model has a lot of parameters and therefore takes up a lot of memory. Smaller batchsizes(like 64 or 128) makes the model manageable in a smaller GPU(~4GB), but empirically, smaller batch sizes do not really work well.

### TabNet

* [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) is another model coming out of Google Research which uses Sparse Attention in multiple steps of decision making to model the output.

The architecture is as shown below(from the paper)

![TabNet Architecture](imgs/tabnet_architecture.png)

All the parameters have beet set to recommended values from the paper. Let's look at few of them:

-   `n_d`: int: Dimension of the prediction layer (usually between 4 and 64). Defaults to `8`
-   `n_a`: int: Dimension of the attention layer (usually between 4 and 64). Defaults to `8`
-   `n_steps`: int: Number of sucessive steps in the newtork (usually betwenn 3 and 10). Defaults to `3`
-   `n_independent`: int: Number of independent GLU layer in each GLU block. Defaults to `2`
-   `n_shared`: int: Number of independent GLU layer in each GLU block. Defaults to `2`
-   `virtual_batch_size`: int: Batch size for Ghost Batch Normalization. BatchNorm on large batches sometimes does not do very well and therefore Ghost Batch Normalization which does batch normalization in smaller virtual batches is implemented in TabNet. Defaults to `128`

**For a complete list of parameters refer to the API Docs**

## Implementing New Architectures

PyTorch Tabular is very easy to extend and infinitely customizable. All the models that have been implemented in PyTorch Tabular inherits an Abstract Class `BaseModel` which is in fact a PyTorchLightning Model.

It handles all the major functions like decoding the config params and setting up the loss and metrics. It also calculates the Loss and metrics and feeds it back to the PyTorch Lightning Trainer which does the back-propagation.

There are two methods that need to be defined in any class that inherits the Base Model:

1. `_build_network` --> This is where you initialize the components required for your model to work
2. `forward` --> This is the forward pass of the model.

While this is the bare minimum, you can redefine or use any of the Pytorch Lightning standard methods to tweak your model and training to your liking.

In addition to the model, you will also need to define a config. Configs are python dataclasses and should inherit `ModelConfig` and will have all the parameters of the ModelConfig. by default. Any additional parameter should be defined in the dataclass. 


**Key things to note:**

1. All the different parameters in the different configs(like TrainerConfig, OptimizerConfig, etc) are all available in `config` before calling `super()` and in `self.hparams` after.
2. the input batch at the `forward` method is a dictionary with keys `continuous` and `categorical`
3. In the `_build_network` method, save every component that you want access in the `forward` to `self`
4. The `forward` method should just have the forward pass and return the outut of the forward pass. In case of classification, do not apply a Sigmoid or Softmax at the end in the forward pass.
5. There is one deviation from the normal when we create a TabularModel object with the configs. Earlier the model was inferred from the config and initialized autmatically. But here, we have to use the `model_callable` parameter of the TabularModel and pass in the model class(not the initialized object)

Please checkout the Implementing New Architectures [tutorial](tutorials.md) for a working example.