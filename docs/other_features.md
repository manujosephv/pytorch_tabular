Apart from training and using Deep Networks for tabular data, PyTorch Tabular also has some cool features which can help your classical ML/ sci-kit learn pipelines

## Categorical Embeddings

The CategoryEmbedding Model can also be used as a way to encode your categorical columns. instead of using a One-hot encoder or a variant of TargetMean Encoding, you can use a learned embedding to encode your categorical features. And all this can be done using a scikit-learn style Transformer.

### Usage Example

```python
# passing the trained model as an argument
transformer = CategoricalEmbeddingTransformer(tabular_model)
# passing the train dataframe to extract the embeddings and replace categorical features
# defined in the trained tabular_model
train_transformed = transformer.fit_transform(train)
# using the extracted embeddings on new dataframe
val_transformed = transformer.transform(val)
```

::: pytorch_tabular.categorical_encoders.CategoricalEmbeddingTransformer
    options:
        show_root_heading: yes
## Feature Extractor

What if you want to use the features learnt by the Neural Network in your ML model? Pytorch Tabular let's you do that as well, and with ease. Again, a scikit-learn style Transformer does the job for you.

### Usage Example
```python
# passing the trained model as an argument
dt = DeepFeatureExtractor(tabular_model)
# passing the train dataframe to extract the last layer features
# here `fit` is there only for compatibility and does not do anything
enc_df = dt.fit_transform(train)
# using the extracted embeddings on new dataframe
val_transformed = transformer.transform(val)
```

::: pytorch_tabular.feature_extractor.DeepFeatureExtractor
    options:
        show_root_heading: yes
