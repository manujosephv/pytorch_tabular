The explainability features in PyTorch Tabular allow users to interpret and understand the predictions made by a tabular deep learning model. These features provide insights into the model's decision-making process and help identify the most influential features. Some of the explainability features are inbuilt from the models, and a lot of others are based on the [Captum](https://captum.ai/) library.

## Native Feature Importance
One of the features of the GBDT models which everybody loves is the feature importance. It helps us understand which features are the most important for the model. PyTorch Tabular provides a similar feature for some of the models - GANDALF, GATE, and FTTransformers - where the models natively support the extraction of feature importance.
    
``` python
# tabular_model is the trained model of a supported model
tabular_model.feature_importance()
```

## Local Feature Attributions/Explanations
Local feature attributions/explanations help us understand the contribution of each feature towards the prediction for a particular sample. PyTorch Tabular provides this feature for all the models except TabTransformer, Tabnet, and Mixed Density Networks. It is based on the [Captum](https://captum.ai/) library. The library provides a lot of algorithms for computing feature attributions. PyTorch Tabular provides a wrapper around the library to make it easy to use. The following algorithms are supported:

- GradientShap: https://captum.ai/api/gradient_shap.html
- IntegratedGradients: https://captum.ai/api/integrated_gradients.html
- DeepLift: https://captum.ai/api/deep_lift.html
- DeepLiftShap: https://captum.ai/api/deep_lift_shap.html
- InputXGradient: https://captum.ai/api/input_x_gradient.html
- FeaturePermutation: https://captum.ai/api/feature_permutation.html
- FeatureAblation: https://captum.ai/api/feature_ablation.html
- KernelShap: https://captum.ai/api/kernel_shap.html

`PyTorch Tabular` also supports explaining single instances as well as batches of instances. But, larger datasets will take longer to explain. An exception is the `FeaturePermutation` and `FeatureAblation` methods, which is only meaningful for large batches of instances.

Most of these explainability methods require a baseline. This is used to compare the attributions of the input with the attributions of the baseline. The baseline can be a scalar value, a tensor of the same shape as the input, or a special string like "b|10000" which means 10000 samples from the training data. If the baseline is not provided, the default baseline (zero) is used.

``` python
# tabular_model is the trained model of a supported model

# Explain a single instance using the GradientShap method and baseline as 10000 samples from the training data
tabular_model.explain(test.head(1), method="GradientShap", baselines="b|10000")

# Explain a batch of instances using the IntegratedGradients method and baseline as 0
tabular_model.explain(test.head(10), method="IntegratedGradients", baselines=0)
```

Checkout the [Captum documentation](https://captum.ai/docs/algorithms) for more details on the algorithms and the [Explainability Tutorial](tutorials/14-Explainability.ipynb) for example usage.

## API Reference
::: pytorch_tabular.TabularModel.explain
    options:
        show_root_heading: yes
        heading_level: 4
