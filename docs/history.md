# History

Certainly! Here's a release update you can use for the history.md file:

---
## 1.1.0 (2024-01-15)

### New Features and Enhancements
- **Added DANet Model**: Added a new model, DANet, for tabular data.
- **Explainability**: Integrated Captum for explainability
- **Hyperparameter Tuner:** Added Grid and Random Search functionality to search through hyperparameters and return best model.
- **Model Sweep:** Added an easy "Model Sweep" method with which we can sweep a list of models with given data and quickly assess performance.
- **Documentation Enhancements:** Improved documentation to make it more user-friendly and informative
- **Dependency Updates:** Updated various dependencies for improved compatibility and security
- **Graceful Out-of-Memory Handling:** Added graceful out-of-memory handling for tabular models
- **GhostBatchNorm:** Added GhostBatchNorm to the library

### Deprecations
- **Deprecations:** Handled deprecations and updated the library accordingly
- **Entmax Dependency Removed:** Removed dependency on entmax

### Infrastructure and CI/CD
- **Continuous Integration:** Improved CI with new actions and labels
- **Dependency Management:** Updated dependencies and restructured requirements

### API Changes
- [BREAKING CHANGE] **SSL API Change:** Addressed SSL API change, along with documentation and tutorial updates.
- **Model Changes:** Added is_fitted and other markers to the tabular model.
- **Custom Optimizer:** Allow custom optimizer in the model config.

### Contributors
- Thanks to all the contributors who helped shape this release! ([List of Contributors](Link_to_Contributors))

### Upgrading
- Ensure to check the updated documentation for any breaking changes or new features.
- If you are using SSL, please check the updated API and documentation.

## 1.0.2 (2023-05-31)

### New Features:

- Added Feature Importance: The library now includes a new method in TabularModel and BaseModel for enabling feature importance. Feature Importance has been enabled for FTTransformer and GATE models. [Commit: dc2a49e]
### Enhancements:

- Enabled two more parameters in the GATE model. [Commit: 3680413]
- Included metric_prob_input parameter in the library configuration. This update allows for better control over metrics in the models. [Commit: 0612db5]
- Slight improvements to the GATE model, including changes to defaults for better performance. [Commit: c30a6c3]
- Minor bug fixes and improvements, including accelerator options in the configuration and progress bar enhancements. [Commit: f932230, bdd9adb, f932230]
### Dependency Updates:

- Updated dependencies, including docformatter, pyupgrade, and ruff-pre-commit. [Commits: 4aae9a8, b3df4ce, bdd9adb, 55e800c, c6c4679, c01154b, 107cd2f]
### Documentation Updates:

- Updated the library's README.md file. [Commits: db8f3b2, cab6bf1, 669faec, 1e6c400, 3097799, 7fabf6b]
### Other Improvements:

- Various code optimizations, bug fixes, and CI enhancements. [Commits: 5637020, e5171bf, 812b40f]

For more details, you can refer to the respective commits on the library's GitHub repository.

## 1.0.1 (2023-01-20)

- Bugfix for default metric for binary classification




## 1.0.0 (2023-01-18)

- Added a new task - Self Supervised Learning (SSL) and a separate training API for it.
- Added new SOTA model - Gated Additive Tree Ensembles (GATE).
- Added one SSL model - Denoising AutoEncoder.
- Added lots of new tutorials and updated entire documentation.
- Improved code documentation and type hints.
- Separated a Model into separate Embedding, Backbone, and Head.
- Refactored all models to separate Backbone as native PyTorch Model(nn.Module).
- Refactored commonly used modules (layers, activations etc. to a common module).
- Changed MixedDensityNetworks completely (breaking change). Now MDN is a head you can use with any model.
- Enabled a low level api for training model.
- Enabled saving and loading of datamodule.
- Added trainer_kwargs to pass any trainer argument PyTorch Lightning supports.
- Added Early Stopping and Model Checkpoint kwargs to use all the arguments in PyTorch Lightining.
- Enabled prediction using GPUs in predict method.
- Added `reset_model` to reset model weights to random.
- Added many save and load functions including ONNX(experimental).
- Added random seed as a parameter.
- Switched over completely to Rich progressbars from tqdm.
- Fixed class-balancing / mu propagation and set default to 1.0.
- Added PyTorch Profiler for debugging performance issues.
- Fixed bugs with FTTransformer and TabTransformer.
- Updated MixedDensityNetworks fixing a bug with lambda_pi.
- Many CI/CD improvements including complete integration with GitHub Actions.
- Upgraded all dependencies, including PyTorch Lightning, pandas, to latest versions and added dependabot to manage it going forward.
- Added pre-commit to ensure code integrity and standardization.

## 0.7.0 (2021-09-01)

- Implemented TabTransformer and FTTransformer models
- Included capability to save a model using GPU an load in CPU
- Made the temp folder pytorch tabular specific to avoid conflicts with other tmp folders.
- Some bug fixes
- Edited an error out of Advanced Tutorial in docs

## 0.6.0 (2021-06-21)

- Upgraded versions of PyTorch Lightning to 1.3.6
- Changed the way `gpus` parameter is handled to avoid confusion. `None` is CPU, `-1` is all GPUs, `int` is number of GPUs
- Added a few more Trainer Params like `deterministic`, `auto_select_gpus`
- Some bug fixes and changes to docs
- Added `seed_everything` to the fit method to ensure reproducibility
- Refactored data_aware_initialization to be part of the BaseModel. Inherited Models can override the method to implement data aware initialization techniques

## 0.5.0 (2021-03-18)

- Added more documentation
- Added Zenodo citation

## 0.4.0 (2021-03-18)

- Added AutoInt Model
- Added Mixture Density Networks
- Refactored the classes to separate backbones from the head of the models
- Changed the saving and loading model to work for custom parameters that you pass in `fit`

## 0.3.0 (2021-03-02)

- Fixed a bug on inference

## 0.2.0 (2021-02-07)

- Fixed an issue with torch.clip and torch version
- Fixed an issue with `gpus` parameter in TrainerConfig, by setting default value to `None` for CPU
- Added feature to use custom sampler in the training dataloader
- Updated documentation and added a new tutorial for imbalanced classification

## 0.0.1 (2021-01-26)

- First release on PyPI.
