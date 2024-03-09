!!! note

    Although the installation includes PyTorch, the best and recommended way is to first install PyTorch from [here](https://pytorch.org/get-started/locally/), picking up the right CUDA version for your machine. (PyTorch Version >1.3)

Once, you have got PyTorch installed and working, just use:

```bash
 pip install "pytorch_tabular[extra]"
```

to install the complete library with extra dependencies:

- Weights&Biases for experiment tracking
- Plotly for some visualization
- Captum for Interpretability

And :

``` bash
 pip install "pytorch_tabular"
```

for the bare essentials.

The sources for `pytorch_tabular` can be downloaded from the Github repo.

You can clone the public repository:

``` bash
git clone git://github.com/manujosephv/pytorch_tabular
```

Once you have a copy of the source, you can install it with:

``` bash
pip install .
```

or

``` bash
python setup.py install
```
