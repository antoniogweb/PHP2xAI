# 1. Introduction and architecture

## Purpose

PHP2xAI separates declarative model definition from numerical execution. PHP is the frontend used to define a model, construct its graph, and serialize that graph as JSON. The principal deployment target is a fully native C++ training runtime: after JSON generation, the C++ process owns dataset loading, batching, forward propagation, backward propagation, optimizer updates, validation, and weight saving.

The design makes it possible to:

- use PHP for model declaration, graph generation, configuration, and application integration;
- quickly experiment with new architectures;
- export a portable graph and training configuration as JSON;
- train entirely in C++ without calling PHP for individual batches or optimizer steps;
- use the typed C++ runtime with NAIVE and Eigen CPU kernels;
- keep device-specific execution as a future extension (CUDA is not implemented yet);
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
        |      GraphRuntime + TXT/HDF5 Dataset + Optimizer + validation
        |      typed NAIVE kernels or Eigen CPU overrides
        |
        +--> PHP application + FFI
               C++ shared library executes inference from model.json + weights.json
```

The JSON graph is the boundary between frontend and runtime. It contains tensor definitions, operations, shapes, fixed attributes, trainable tensor identifiers, and either a loss or output identifier. A training configuration additionally contains optimizer settings, dataset paths, batch size, epoch count, logging settings, and output paths.

## Fully native C++ training

When the C++ runtime is selected for training, PHP generates the graph and configuration files, then launches the C++ executable with `config.json`. The native `Core` object reads that file and creates its own:

- `GraphRuntime` for forward and backward execution;
- TXT or HDF5 training and validation datasets;
- optimizer, such as Adam or Fixed;
- training loop, validation loop, and weight serialization.

PHP does not perform per-batch numerical work in this mode. It is used before training to describe and export the model; the native process then runs the complete training job from the JSON configuration.

## Native inference through PHP FFI

Inference can remain part of a PHP web or application process without moving numerical execution back into PHP. PHP loads the C++ shared library through FFI, gives it `model.json` and `weights.json`, and calls the native prediction functions with flat input data. The C++ runtime parses the graph, executes it, and returns output values or an integer label to PHP.

The PHP runtime can still be selected for debugging, portability, and reference comparisons. It is not the intended high-throughput training path.

## Typed runtime and providers

The native C++ runtime stores each tensor in a buffer allocated for its declared dtype. NAIVE kernels dispatch on that dtype and use typed pointers; Eigen overrides selected CPU kernels while inheriting the NAIVE implementation for other paths. The graph selects operation kernels by shape and attributes, while the runtime provider selects the implementation of virtual kernel entry points. CUDA is a future possibility, not a currently available provider.

The public PHP and FFI boundaries still exchange values as PHP numbers and C++ `Scalar` (`float`). On entry, C++ writes those values into the storage type declared by each graph tensor; reads back to PHP/FFI convert values to `Scalar`. Inside the C++ runtime, tensors and kernels support `FLOAT32`, `FLOAT64`, `INT32`, and `INT64`. The FFI boundary has not yet become a zero-conversion, arbitrary-dtype interface.

See [SIMD and Eigen](15-simd-and-eigen.md) for provider selection and the current optimized paths.

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
