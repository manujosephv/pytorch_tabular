History
=======

0.0.1 (2021-01-26)
------------------

-   First release on PyPI.

0.2.0 (2021-02-07)
------------------

-   Fixed an issue with torch.clip and torch version
-   Fixed an issue with `gpus` parameter in TrainerConfig, by setting default value to `None` for CPU
-   Added feature to use custom sampler in the training dataloader
-   Updated documentation and added a new tutorial for imbalanced classification

0.3.0 (2021-03-02)
------------------
-   Fixed a bug on inference

0.4.0 (2021-03-18)
------------------
-   Added AutoInt Model
-   Added Mixture Density Networks
-   Refactored the classes to separate backbones from the head of the models
-   Changed the saving and loading model to work for custom parameters that you pass in `fit`

