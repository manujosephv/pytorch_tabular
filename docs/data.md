PyTorch Tabular uses Pandas Dataframes as the container which holds data. As Pandas is the most popular way of handling tabular data, this was an obvious choice. Keeping ease of useability in mind, PyTorch Tabular accepts dataframes as is, i.e. no need to split the data into `X` and `y` like in Sci-kit Learn.

Pytorch Tabular handles this using a `DataConfig` object.

## Basic Usage

- `target`: List\[str\]: A list of strings with the names of the target column(s)
- `continuous_cols`: List\[str\]: Column names of the numeric fields. Defaults to \[\]
- `categorical_cols`: List\[str\]: Column names of the categorical fields to treat differently

### Usage Example

```python
data_config = DataConfig(
    target=["label"],
    continuous_cols=["feature_1", "feature_2"],
    categorical_cols=["cat_feature_1", "cat_feature_2"],
)
```

## Advanced Usage:

### Date Columns

If you have date_columns in the dataframe, mention the column names in `date_columns` parameter and set `encode_date_columns` to `True`. This will extract relevant features like the Month, Week, Quarter etc. and add them to your feature list internally.

`date_columns` is not just a list of column names, but a list of (column name, freq) tuples. The freq is a standard Pandas date frequency tags which denotes the lowest temporal granularity which is relevant for the problem.

For eg., if there is a date column for Launch Date for a Product and they only launch once a month. Then there is no sense in extracting features like week, or day etc. So, we keep the frequency at `M`

```python
date_columns = [("launch_date", "M")]
```

### Feature Transformations

Feature Scaling is an almost essential step to get goog performance from most Machine Learning Algorithms, and Deep Learning is not an exception. `normalize_continuous_features` flag(which is `True` by default) scales the input continuous features using a `StandardScaler`

Sometimes, changing the feature distributions using non-linear transformations helps the machine learning/deep learning algorithms.

PyTorch Tabular offers 4 standard transformations using the `continuous_feature_transform` parameter:

- `yeo-johnson`
- `box-cox`
- `quantile_uniform`
- `quantile_normal`

`yeo-johnson` and `box-cox` are a family of parametric, monotonic transformations that aim to map data from any distribution to as close to a Gaussian distribution as possible in order to stabilize variance and minimize skewness. `box-cox` can only be applied to *strictly positive* data. Sci-kit Learn has a good [write-up](https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution) about them

`quantile_normal` and `quantile_uniform` are monotonic, non-parametric transformations which aims to transfom the features to a normal distribution or a uniform distribution, respectively.By performing a rank transformation, a quantile transform smooths out unusual distributions and is less influenced by outliers than scaling methods. It does, however, distort correlations and distances within and across features.

::: pytorch_tabular.config.DataConfig
    options:
        show_root_heading: yes
