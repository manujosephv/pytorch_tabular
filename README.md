![PyTorch Tabular](docs/imgs/pytorch_tabular_logo.png)

[![pypi](https://img.shields.io/pypi/v/pytorch_tabular.svg)](https://pypi.python.org/pypi/pytorch_tabular)
[![Testing](https://github.com/manujosephv/pytorch_tabular/actions/workflows/testing.yml/badge.svg?event=push)](https://github.com/manujosephv/pytorch_tabular/actions/workflows/testing.yml)
[![documentation status](https://readthedocs.org/projects/pytorch_tabular/badge/?version=latest)](https://pytorch-tabular.readthedocs.io/en/latest/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/manujosephv/pytorch_tabular/main.svg)](https://results.pre-commit.ci/latest/github/manujosephv/pytorch_tabular/main)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manujosephv/pytorch_tabular/blob/main/docs/tutorials/01-Basic_Usage.ipynb)

![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch_tabular)
[![DOI](https://zenodo.org/badge/321584367.svg)](https://zenodo.org/badge/latestdoi/321584367)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/manujosephv/pytorch_tabular/issues)

PyTorch Tabular aims to make Deep Learning with Tabular data easy and accessible to real-world cases and research alike. The core principles behind the design of the library are:

- Low Resistance Usability
- Easy Customization
- Scalable and Easier to Deploy

It has been built on the shoulders of giants like **PyTorch**(obviously), and **PyTorch Lightning**.

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Available Models](#available-models)
- [Usage](#usage)
- [Blogs](#blogs)
- [Citation](#citation)

## Installation

Although the installation includes PyTorch, the best and recommended way is to first install PyTorch from [here](https://pytorch.org/get-started/locally/), picking up the right CUDA version for your machine.

Once, you have got Pytorch installed, just use:

```bash
pip install -U “pytorch_tabular[extra]”
```

to install the complete library with extra dependencies (Weights&Biases & Plotly).

And :

```bash
pip install -U “pytorch_tabular”
```

for the bare essentials.

The sources for pytorch_tabular can be downloaded from the `Github repo`\_.

You can either clone the public repository:

```bash
git clone git://github.com/manujosephv/pytorch_tabular
```

Once you have a copy of the source, you can install it with:

```bash
cd pytorch_tabular && pip install .[extra]
```

## Documentation

For complete Documentation with tutorials visit [ReadTheDocs](https://pytorch-tabular.readthedocs.io/en/latest/)

## Available Models

- FeedForward Network with Category Embedding is a simple FF network, but with an Embedding layers for the categorical columns.
- [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) is a model presented in ICLR 2020 and according to the authors have beaten well-tuned Gradient Boosting models on many datasets.
- [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) is another model coming out of Google Research which uses Sparse Attention in multiple steps of decision making to model the output.
- [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) is a regression model which uses gaussian components to approximate the target function and  provide a probabilistic prediction out of the box.
- [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) is a model which tries to learn interactions between the features in an automated way and create a better representation and then use this representation in downstream task
- [TabTransformer](https://arxiv.org/abs/2012.06678) is an adaptation of the Transformer model for Tabular Data which creates contextual representations for categorical features.
- FT Transformer from [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)
- [Gated Additive Tree Ensemble](https://arxiv.org/abs/2207.08548v3) is a novel high-performance, parameter and computationally efficient deep learning architecture for tabular data. GATE uses a gating mechanism, inspired from GRU, as a feature representation learning unit with an in-built feature selection mechanism. We combine it with an ensemble of differentiable, non-linear decision trees, re-weighted with simple self-attention to predict our desired output.
- [Gated Adaptive Network for Deep Automated Learning of Features (GANDALF)](https://arxiv.org/abs/2207.08548) is pared-down version of GATE which is more efficient and performing than GATE. GANDALF makes GFLUs the main learning unit, also introducing some speed-ups in the process. With very minimal hyperparameters to tune, this becomes an easy to use and tune model.
- [DANETs: Deep Abstract Networks for Tabular Data Classification and Regression](https://arxiv.org/pdf/2112.02962v4.pdf) is a novel and flexible neural component for tabular data, called Abstract Layer (AbstLay), which learns to explicitly group correlative input features and generate higher-level features for semantics abstraction.  A special basic block is built using AbstLays, and we construct a family of Deep Abstract Networks (DANets) for tabular data classification and regression by stacking such blocks.

**Semi-Supervised Learning**

- [Denoising AutoEncoder](https://www.kaggle.com/code/springmanndaniel/1st-place-turn-your-data-into-daeta) is an autoencoder which learns robust feature representation, to compensate any noise in the dataset.

## Implement Custom Models
To implement new models, see the [How to implement new models tutorial](https://github.com/manujosephv/pytorch_tabular/blob/main/docs/tutorials/04-Implementing%20New%20Architectures.ipynb). It covers basic as well as advanced architectures.

## Usage

```python
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import (
    DataConfig,
    OptimizerConfig,
    TrainerConfig,
    ExperimentConfig,
)

data_config = DataConfig(
    target=[
        "target"
    ],  # target should always be a list.
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
loaded_model = TabularModel.load_model("examples/basic")
```

## Blogs

- [PyTorch Tabular – A Framework for Deep Learning for Tabular Data](https://deep-and-shallow.com/2021/01/27/pytorch-tabular-a-framework-for-deep-learning-for-tabular-data/)
- [Neural Oblivious Decision Ensembles(NODE) – A State-of-the-Art Deep Learning Algorithm for Tabular Data](https://deep-and-shallow.com/2021/02/25/neural-oblivious-decision-ensemblesnode-a-state-of-the-art-deep-learning-algorithm-for-tabular-data/)
- [Mixture Density Networks: Probabilistic Regression for Uncertainty Estimation](https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/)

## Future Roadmap(Contributions are Welcome)

1. Integrate Optuna Hyperparameter Tuning
1. Migrate Datamodule to Polars or NVTabular for faster data loading and to handle larger than RAM datasets.
1. Add GaussRank as Feature Transformation
1. Have a scikit-learn compatible API
1. Enable support for multi-label classification
1. Keep adding more architectures

## Contributors

<!-- readme: contributors -start -->
<table>
	<tbody>
		<tr>
            <td align="center">
                <a href="https://github.com/manujosephv">
                    <img src="https://avatars.githubusercontent.com/u/10508493?v=4" width="100;" alt="manujosephv"/>
                    <br />
                    <sub><b>Manu Joseph</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Borda">
                    <img src="https://avatars.githubusercontent.com/u/6035284?v=4" width="100;" alt="Borda"/>
                    <br />
                    <sub><b>Jirka Borovec</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/wsad1">
                    <img src="https://avatars.githubusercontent.com/u/13963626?v=4" width="100;" alt="wsad1"/>
                    <br />
                    <sub><b>Jinu Sunil</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ProgramadorArtificial">
                    <img src="https://avatars.githubusercontent.com/u/130674366?v=4" width="100;" alt="ProgramadorArtificial"/>
                    <br />
                    <sub><b>Programador Artificial</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/sorenmacbeth">
                    <img src="https://avatars.githubusercontent.com/u/130043?v=4" width="100;" alt="sorenmacbeth"/>
                    <br />
                    <sub><b>Soren Macbeth</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/fonnesbeck">
                    <img src="https://avatars.githubusercontent.com/u/81476?v=4" width="100;" alt="fonnesbeck"/>
                    <br />
                    <sub><b>Chris Fonnesbeck</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/snehilchatterjee">
                    <img src="https://avatars.githubusercontent.com/u/127598707?v=4" width="100;" alt="snehilchatterjee"/>
                    <br />
                    <sub><b>Snehil Chatterjee</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/jxtrbtk">
                    <img src="https://avatars.githubusercontent.com/u/40494970?v=4" width="100;" alt="jxtrbtk"/>
                    <br />
                    <sub><b>Null</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/abhisharsinha">
                    <img src="https://avatars.githubusercontent.com/u/24841841?v=4" width="100;" alt="abhisharsinha"/>
                    <br />
                    <sub><b>Abhishar Sinha</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/ndrsfel">
                    <img src="https://avatars.githubusercontent.com/u/21068727?v=4" width="100;" alt="ndrsfel"/>
                    <br />
                    <sub><b>Andreas</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/charitarthchugh">
                    <img src="https://avatars.githubusercontent.com/u/37895518?v=4" width="100;" alt="charitarthchugh"/>
                    <br />
                    <sub><b>Charitarth Chugh</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/EeyoreLee">
                    <img src="https://avatars.githubusercontent.com/u/49790022?v=4" width="100;" alt="EeyoreLee"/>
                    <br />
                    <sub><b>Earlee</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/JulianRein">
                    <img src="https://avatars.githubusercontent.com/u/35046938?v=4" width="100;" alt="JulianRein"/>
                    <br />
                    <sub><b>Null</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/krshrimali">
                    <img src="https://avatars.githubusercontent.com/u/19997320?v=4" width="100;" alt="krshrimali"/>
                    <br />
                    <sub><b>Kushashwa Ravi Shrimali</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/Actis92">
                    <img src="https://avatars.githubusercontent.com/u/46601193?v=4" width="100;" alt="Actis92"/>
                    <br />
                    <sub><b>Luca Actis Grosso</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/sgbaird">
                    <img src="https://avatars.githubusercontent.com/u/45469701?v=4" width="100;" alt="sgbaird"/>
                    <br />
                    <sub><b>Sterling G. Baird</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/furyhawk">
                    <img src="https://avatars.githubusercontent.com/u/831682?v=4" width="100;" alt="furyhawk"/>
                    <br />
                    <sub><b>Teck Meng</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/yinyunie">
                    <img src="https://avatars.githubusercontent.com/u/25686434?v=4" width="100;" alt="yinyunie"/>
                    <br />
                    <sub><b>Yinyu Nie</b></sub>
                </a>
            </td>
		</tr>
		<tr>
            <td align="center">
                <a href="https://github.com/YonyBresler">
                    <img src="https://avatars.githubusercontent.com/u/24940683?v=4" width="100;" alt="YonyBresler"/>
                    <br />
                    <sub><b>YonyBresler</b></sub>
                </a>
            </td>
            <td align="center">
                <a href="https://github.com/HernandoR">
                    <img src="https://avatars.githubusercontent.com/u/45709656?v=4" width="100;" alt="HernandoR"/>
                    <br />
                    <sub><b>Liu Zhen</b></sub>
                </a>
            </td>
		</tr>
	<tbody>
</table>
<!-- readme: contributors -end -->

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
@software{manu_joseph_2023_7554473,
  author       = {Manu Joseph and
                  Jinu Sunil and
                  Jiri Borovec and
                  Chris Fonnesbeck and
                  jxtrbtk and
                  Andreas and
                  JulianRein and
                  Kushashwa Ravi Shrimali and
                  Luca Actis Grosso and
                  Sterling G. Baird and
                  Yinyu Nie},
  title        = {manujosephv/pytorch\_tabular: v1.0.1},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.1},
  doi          = {10.5281/zenodo.7554473},
  url          = {https://doi.org/10.5281/zenodo.7554473}
}
```
