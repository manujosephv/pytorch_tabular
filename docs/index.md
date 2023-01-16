![PyTorch Tabular](imgs/pytorch_tabular_logo.png)

[![pypi](https://img.shields.io/pypi/v/pytorch_tabular.svg)](https://pypi.python.org/pypi/pytorch_tabular)
[![Testing](https://github.com/manujosephv/pytorch_tabular/actions/workflows/testing.yml/badge.svg?event=push)](https://github.com/manujosephv/pytorch_tabular/actions/workflows/testing.yml)
[![documentation status](https://readthedocs.org/projects/pytorch_tabular/badge/?version=latest)](https://pytorch_tabular.readthedocs.io/en/latest/?badge=latest)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/manujosephv/pytorch_tabular/main.svg)](https://results.pre-commit.ci/latest/github/manujosephv/pytorch_tabular/main)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch_tabular)
[![DOI](https://zenodo.org/badge/321584367.svg)](https://zenodo.org/badge/latestdoi/321584367)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/manujosephv/pytorch_tabular/issues)

PyTorch Tabular aims to make Deep Learning with Tabular data easy and accessible to real-world cases and research alike. The core principles behind the design of the library are:

- **Low Resistance Usability**
- **Easy Customization**
- **Scalable and Easier to Deploy**

It has been built on the shoulders of giants like [**PyTorch**](https://pytorch.org/)(obviously), [**PyTorch Lightning**](https://www.pytorchlightning.ai/), and [pandas](https://pandas.pydata.org/)

## Installation

Although the installation includes PyTorch, the best and recommended way is to first install PyTorch from [here](https://pytorch.org/get-started/locally/), picking up the right CUDA version for your machine. (PyTorch Version >1.3)

Once, you have got Pytorch installed, just use:

```bash
 pip install pytorch_tabular[extra]
```

to install the complete library with extra dependencies(Weights&Biases and Plotly).

And :

```bash
 pip install pytorch_tabular
```

for the bare essentials.

The sources for pytorch_tabular can be downloaded from the `Github repo`.

You can either clone the public repository:

```bash
git clone git://github.com/manujosephv/pytorch_tabular
```

Once you have a copy of the source, you can install it with:

```bash
pip install .
```

or

```bash
python setup.py install
```

## Usage

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
)

data_config = DataConfig(
    target=[
        "target"
    ],  # target should always be a list. Multi-targets are only supported for regression. Multi-Task Classification is not implemented
    continuous_cols=num_col_names,
    categorical_cols=cat_col_names,
)
trainer_config = TrainerConfig(
    auto_lr_find=True,  # Runs the LRFinder to automatically derive a learning rate
    batch_size=1024,
    max_epochs=100,
)
optimizer_config = OptimizerConfig()

model_config = CategoryEmbeddingModelConfig(
    task="classification",
    layers="1024-512-512",  # Number of nodes in each layer
    activation="LeakyReLU",  # Activation between each layers
    learning_rate=1e-3,
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

## Citation

If you use PyTorch Tabular for a scientific publication, we would appreciate citations to the published software and the following paper:

- [arxiv Paper](https://arxiv.org/abs/2104.13638)

```
@misc{joseph2021pytorch,
      title={PyTorch Tabular: A Framework for Deep Learning with Tabular Data},
      author={Manu Joseph},
      year={2021},
      eprint={2104.13638},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

- Zenodo Software Citation

```
@article{manujosephv_2021,
    title={manujosephv/pytorch_tabular: v0.5.0-alpha},
    DOI={10.5281/zenodo.4732773},
    abstractNote={<p>First Alpha Release</p>},
    publisher={Zenodo},
    author={manujosephv},
    year={2021},
    month={May}
}
```
