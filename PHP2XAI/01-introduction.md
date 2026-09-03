# 1. Introduction and architecture

## Purpose

PHP2xAI separates declarative model definition from numerical execution. PHP is the frontend used to define a model, construct its graph, and serialize that graph as JSON. The principal deployment target is a fully native C++ training runtime: after JSON generation, the C++ process owns dataset loading, batching, forward propagation, backward propagation, optimizer updates, validation, and weight saving.

The design makes it possible to:

- use PHP for model declaration, graph generation, configuration, and application integration;
- quickly experiment with new architectures;
- export a portable graph and training configuration as JSON;
- train entirely in C++ without calling PHP for individual batches or optimizer steps;
- use SIMD/Eigen acceleration today and leave room for a future CUDA backend;
- invoke native inference from PHP through FFI;
- retain the PHP runtime as a readable implementation and development reference.

## Execution architecture

```text
PHP model definition
        |
        | GraphContext records tensors and operations
        v
model.json / config.json
        |
        +--> C++ training executable
        |      GraphRuntime + Dataset + Optimizer + validation
        |      scalar kernels, Eigen/SIMD kernels, future CUDA backend
        |
        +--> PHP application + FFI
               C++ shared library executes inference from model.json + weights.json
```

The JSON graph is the boundary between frontend and runtime. It contains tensor definitions, operations, shapes, fixed attributes, trainable tensor identifiers, and either a loss or output identifier. A training configuration additionally contains optimizer settings, dataset paths, batch size, epoch count, logging settings, and output paths.

## Fully native C++ training

When the C++ runtime is selected for training, PHP generates the graph and configuration files, then launches the C++ executable with `config.json`. The native `Core` object reads that file and creates its own:

- `GraphRuntime` for forward and backward execution;
- streaming training and validation datasets;
- optimizer, such as Adam or Fixed;
- training loop, validation loop, and weight serialization.

PHP does not perform per-batch numerical work in this mode. It is used before training to describe and export the model; the native process then runs the complete training job from the JSON configuration.

## Native inference through PHP FFI

Inference can remain part of a PHP web or application process without moving numerical execution back into PHP. PHP loads the C++ shared library through FFI, gives it `model.json` and `weights.json`, and calls the native prediction functions with flat input data. The C++ runtime parses the graph, executes it, and returns output values or an integer label to PHP.

The PHP runtime can still be selected for debugging, portability, and reference comparisons. It is not the intended high-throughput training path.

## SIMD, Eigen, and future CUDA

The native runtime keeps its graph and operation contract independent from the acceleration backend. Scalar kernels provide the general reference and stride-aware fallbacks. Common contiguous high-volume paths are being migrated progressively to Eigen, which can use SIMD instructions selected by the compiler and CPU architecture. The same separation is intended to permit a future CUDA backend without changing PHP model code or the serialized graph format.

See [SIMD and Eigen migration](15-simd-and-eigen.md) for the current migration state.

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

The runtime loads the graph and executes its operations. The PHP runtime is implemented in `Runtime/PHP/Core/GraphRuntime.php`; the C++ runtime is implemented in `Runtime/CPP/Core/`. The native runtime is used directly by the C++ training executable and by the C++ shared library loaded through PHP FFI for inference.

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
