![PyTorch Tabular](docs/imgs/pytorch_tabular_logo.png)
[![pypi](https://img.shields.io/pypi/v/pytorch_tabular.svg)](https://pypi.python.org/pypi/pytorch_tabular)
[![travis](https://img.shields.io/travis/manujosephv/pytorch_tabular.svg)](https://travis-ci.com/manujosephv/pytorch_tabular)
[![documentation status](https://readthedocs.org/projects/pytorch_tabular/badge/?version=latest)](https://pytorch_tabular.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch_tabular)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/manujosephv/pytorch_tabular/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manujosephv/pytorch_tabular/blob/main/docs/tutorials/01-Basic_Usage.ipynb.py)

PyTorch Tabular aims to make Deep Learning with Tabular data easy and accessible to real-world cases and research alike. The core principles behind the design of the library are:
* Low Resistance Useability
* Easy Customization
* Scalable and Easier to Deploy

It has been built on the shoulders of giants like **PyTorch**(obviously), and **PyTorch Lightning**.

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Available Models](#available-models)
- [Usage](#usage)
- [Blog](#blog)
- [References and Citations](#references-and-citations)


## Installation

Although the installation includes PyTorch, the best and recommended way is to first install PyTorch from [here](https://pytorch.org/get-started/locally/), picking up the right CUDA version for your machine.

Once, you have got Pytorch installed, just use:
```
 pip install pytorch_tabular[all]
```

to install the complete library with extra dependencies.

And :
```
 pip install pytorch_tabular
```

for the bare essentials.


The sources for pytorch_tabular can be downloaded from the `Github repo`_.

You can either clone the public repository:

```
git clone git://github.com/manujosephv/pytorch_tabular
```

Once you have a copy of the source, you can install it with:

```
python setup.py install
```

## Documentation

For complete Documentation with tutorials visit []

## Available Models

* FeedForward Network with Category Embedding is a simple FF network, but with and Embedding layers for the categorical columns.
* [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) is a model presented in ICLR 2020 and according to the authors have beaten well-tuned Gradient Boosting models on many datasets.
* [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) is another model coming out of Google Research which uses Sparse Attention in multiple steps of decision making to model the output.

To implement new models, see the [How to implement new models tutorial](https://github.com/manujosephv/pytorch_tabular/blob/main/docs/04-Implementing%20New%20Architectures.ipynb). It covers basic as well as advanced architectures.

## Usage
```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig

data_config = DataConfig(
    target=['target'], #target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True, # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
    gpus=1, #index of the GPU to use. 0, means CPU
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU", # Activation between each layers
    learning_rate = 1e-3
)

tabular_model = TabularModel(
    data_config=data_config,
    model_config=model_config,
    optimizer_config=optimizer_config,
    trainer_config=trainer_config,
)
tabular_model.fit(train=train, validation=val)
result = tabular_model.evaluate(test)
pred_df = tabular_model.predict(test)
tabular_model.save_model("examples/basic")
loaded_model = TabularModel.load_from_checkpoint("examples/basic")
```
## Blog

[PyTorch Tabular – A Framework for Deep Learning for Tabular Data](https://deep-and-shallow.com/2021/01/27/pytorch-tabular-a-framework-for-deep-learning-for-tabular-data/)

## Future Roadmap(Contributions are Welcome)

1. Add GaussRank as Feature Transformation
2. Add ability to use custom activations in CategoryEmbeddingModel
3. Add differential dropouts(layer-wise) in CategoryEmbeddingModel
4. Add Fourier Encoding for cyclic time variables
5. Integrate Optuna Hyperparameter Tuning
6. Add Text and Image Modalities for mixed modal problems
7. Integrate Wide and Deep model
8. Integrate TabTransformer
## References and Citations

[1] Sergei Popov, Stanislav Morozov, Artem Babenko. [*"Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"*](https://arxiv.org/abs/1909.06312). arXiv:1909.06312 [cs.LG] (2019)

[2] Sercan O. Arik, Tomas Pfister;. [*"TabNet: Attentive Interpretable Tabular Learning"*](https://arxiv.org/abs/1908.07442). 	arXiv:1908.07442 (2019).