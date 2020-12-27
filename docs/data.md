PyTorch Tabular uses Pandas Dataframes as the container which holds data. As Pandas is the most popular way of handling tabular data, this was an obvious choice. Keeping ease of useability in mind, PyTorch Tabular accepts dataframes as is, i.e. no need to split the data into X and y like in Sci-kit Learn.

Pytorch Tabular handles this using a `DataConfig` object.

## Required Parameters

The bare minimum you need to set in a `DataConfig` object are:
1. `target`: 

::: pytorch_tabular.config.DataConfig
