![PyTorch Tabular](docs/imgs/pytorch_tabular_logo.png)    

[![pypi](https://img.shields.io/pypi/v/pytorch_tabular.svg)](https://pypi.python.org/pypi/pytorch_tabular)
[![travis](https://img.shields.io/travis/manujosephv/pytorch_tabular.svg)](https://travis-ci.com/manujosephv/pytorch_tabular)
[![documentation status](https://readthedocs.org/projects/pytorch_tabular/badge/?version=latest)](https://pytorch_tabular.readthedocs.io/en/latest/?badge=latest)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch_tabular)
[![DOI](https://zenodo.org/badge/321584367.svg)](https://zenodo.org/badge/latestdoi/321584367)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat-square)](https://github.com/manujosephv/pytorch_tabular/issues)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manujosephv/pytorch_tabular/blob/main/docs/tutorials/01-Basic_Usage.ipynb)


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
- [Blogs](#blogs)
- [Citation](#citation)


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

For complete Documentation with tutorials visit [ReadTheDocs](https://pytorch-tabular.readthedocs.io/en/latest/ )

## Available Models

* FeedForward Network with Category Embedding is a simple FF network, but with an Embedding layers for the categorical columns.
* [Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data](https://arxiv.org/abs/1909.06312) is a model presented in ICLR 2020 and according to the authors have beaten well-tuned Gradient Boosting models on many datasets.
* [TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442) is another model coming out of Google Research which uses Sparse Attention in multiple steps of decision making to model the output.
* [Mixture Density Networks](https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) is a regression model which uses gaussian components to approximate the target function and  provide a probabilistic prediction out of the box.
* [AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921) is a model which tries to learn interactions between the features in an automated way and create a better representation and then use this representation in downstream task
* [TabTransformer](https://arxiv.org/abs/2012.06678) is an adaptation of the Transformer model for Tabular Data which creates contextual representations for categorical features.
* FT Transformer from [Revisiting Deep Learning Models for Tabular Data](https://arxiv.org/abs/2106.11959)

To implement new models, see the [How to implement new models tutorial](https://github.com/manujosephv/pytorch_tabular/blob/main/docs/tutorials/04-Implementing%20New%20Architectures.ipynb). It covers basic as well as advanced architectures.

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
## Blogs

- [PyTorch Tabular – A Framework for Deep Learning for Tabular Data](https://deep-and-shallow.com/2021/01/27/pytorch-tabular-a-framework-for-deep-learning-for-tabular-data/)
- [Neural Oblivious Decision Ensembles(NODE) – A State-of-the-Art Deep Learning Algorithm for Tabular Data](https://deep-and-shallow.com/2021/02/25/neural-oblivious-decision-ensemblesnode-a-state-of-the-art-deep-learning-algorithm-for-tabular-data/)
- [Mixture Density Networks: Probabilistic Regression for Uncertainty Estimation](https://deep-and-shallow.com/2021/03/20/mixture-density-networks-probabilistic-regression-for-uncertainty-estimation/)

## Future Roadmap(Contributions are Welcome)

1. Add GaussRank as Feature Transformation
2. Add ability to use custom activations in CategoryEmbeddingModel
3. Add differential dropouts(layer-wise) in CategoryEmbeddingModel
4. Add Fourier Encoding for cyclic time variables
5. Integrate Optuna Hyperparameter Tuning
6. Add Text and Image Modalities for mixed modal problems
7. Add Variable Importance
8. Integrate SHAP for interpretability
        
**DL Models**
     
9. [DNF-Net: A Neural Architecture for Tabular Data](https://www.semanticscholar.org/paper/DNF-Net%3A-A-Neural-Architecture-for-Tabular-Data-Abutbul-Elidan/99c49f3a917815eed2144bfb5d064623ff09ade5)
10. [Attention augmented differentiable forest for tabular data](https://www.semanticscholar.org/paper/Attention-augmented-differentiable-forest-for-data-Chen/57990b40affc5f34f4029dab39bc78e44e7d3b10)
11. [XBNet : An Extremely Boosted Neural Network](https://arxiv.org/abs/2106.05239v2)
## Citation
If you use PyTorch Tabular for a scientific publication, we would appreciate citations to the published software and the following paper:


* [arxiv Paper](https://arxiv.org/abs/2104.13638)
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
* Zenodo Software Citation
```
@article{manujosephv_2021, 
    title={manujosephv/pytorch_tabular: v0.7.0-alpha}, 
    DOI={10.5281/zenodo.5359010}, 
    abstractNote={<p>Added a few more SOTA models - TabTransformer, FTTransformer
        Made improvements in the model save and load capability
        Made installation less restrictive by unfreezing some dependencies.</p>}, 
    publisher={Zenodo}, 
    author={manujosephv}, 
    year={2021}, 
    month={May}
}
```
