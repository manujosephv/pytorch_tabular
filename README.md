![PyTorch Tabular](docs/imgs/pytorch_tabular_logo.png)
[![pypi](https://img.shields.io/pypi/v/pytorch_tabular.svg)](https://pypi.python.org/pypi/pytorch_tabular)
[![travis](https://img.shields.io/travis/manujosephv/pytorch_tabular.svg)](https://travis-ci.com/manujosephv/pytorch_tabular)
[![documentation status](https://readthedocs.org/projects/pytorch_tabular/badge/?version=latest)](https://pytorch_tabular.readthedocs.io/en/latest/?badge=latest)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/manujosephv/pytorch_tabular/issues)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Zhenye-Na/DA-RNN/blob/master/src/da_rnn.ipynb.py) -->

PyTorch Tabular aims to make Deep Learning with Tabular data easy and accessible to real-world cases and research alike. The core principles behind the design of the library are:
* Low Resistance Useability
* Easy Customization
* Scalable and Easier to Deploy

It has been build on the shoulders of giants like **PyTorch**(obviously), and **PyTorch Lightning**.

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
 pip install pytorch_tabular
```

to install the library.


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
* [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) is another model coming out of Google Research

To implement new models, see the [How to implement new models tutorial](). It covers basic as well as advanced architectures.

## Usage
```

```

## References and Citations

[1] Sergei Popov, Stanislav Morozov, Artem Babenko. [*"Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data"*](https://arxiv.org/abs/1909.06312). arXiv:1909.06312 [cs.LG] (2019)

[2] Sercan O. Arik, Tomas Pfister;. [*"TabNet: Attentive Interpretable Tabular Learning"*](https://arxiv.org/abs/1908.07442). 	arXiv:1908.07442 (2019).