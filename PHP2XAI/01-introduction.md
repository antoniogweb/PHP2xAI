# 1. Introduction and architecture

## Purpose

PHP2xAI separates declarative model definition from numerical execution. The model is written in PHP, while operations can be executed by either the PHP or C++ runtime.

This separation makes it possible to:

- use PHP for datasets, configuration, and application integration;
- quickly experiment with new architectures;
- export the graph as JSON;
- use C++ for faster training and inference;
- keep the same logical model in both runtimes.

## Main components

### Tensor

`Tensor` is the object used by a model to describe data and operations. While the model is being built, operations are registered in the `GraphContext`.

### GraphContext

The context assigns an identifier to each tensor and records operations. The result is a graph containing:

- tensors;
- operations;
- input tensor;
- target tensor;
- loss tensor;
- list of trainable tensors.

### GraphRuntime

The runtime loads the graph and executes its operations. The PHP runtime is implemented in `Runtime/PHP/Core/GraphRuntime.php`; the C++ runtime is implemented in `Runtime/CPP/Core/`.

### Model

The abstract `Model` class coordinates:

- training graph generation;
- inference model generation;
- training;
- validation;
- weight saving and loading;
- runtime selection.

## Training and inference

A model normally exposes three separate paths:

- `forward($x)`: path used to compute the loss and train;
- `loss($x, $y)`: builds the value to minimize;
- `output($x)`: path used for inference.

This distinction is important for operations such as dropout, which must be active during training and disabled during inference.
