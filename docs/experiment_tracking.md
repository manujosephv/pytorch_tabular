Experiment Tracking is almost an essential part of machine learning. It is critical in upholding reproduceability. PyTorch Tabular embraces this and supports experiment tracking internally. Currently, PyTorch Tabular supports two experiment Tracking Framework:

1. Tensorboard
2. Weights and Biases

Tensorboard logging is barebones. PyTorch Tabular just logs the losses and metrics to tensorboard. W&B tracking is much more feature rich - in addition to tracking losses and metrics, it can also track the gradients of the different layers, logits of your model across epochs, etc.

## Basic Usage

