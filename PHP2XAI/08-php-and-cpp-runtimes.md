# 8. PHP and C++ runtimes

Both runtimes execute the same serialized graph.

| Function | PHP | C++ |
|---|---:|---:|
| Model definition | yes | no |
| Graph generation | yes | no |
| Training | yes | yes |
| Validation | yes | yes |
| Inference | yes | yes |
| TXT dataset (`StreamFileDataset`) | yes | yes |
| HDF5 dataset (`HDF5Dataset`) | yes | yes |
| Batch `[B,D]` | yes | yes |

With the C++ runtime, the PHP model normally generates:

- `model.json` for the inference graph;
- `config.json` for training, optimizer, and dataset settings;
- `weights.json` for the best weights.

The C++ runtime reads the configuration, loads graph and weights, constructs the datasets, and starts `Core::train()`.

The training configuration contains `dataset_type`, which selects the native dataset implementation: `TXT` creates `StreamFileDataset`, while `HDF5` creates `HDF5Dataset`. PHP derives this value from the training dataset passed to `TrainValidateDataset`; currently training and validation must use the same file format. Older configurations without `dataset_type` are interpreted as `TXT`. HDF5 requires the `php2xai_hdf5.so` FFI library when PHP creates or reads the files, and the HDF5-linked C++ runtime when native training reads them. See [Datasets and batches](06-datasets-and-batches.md) for setup and file details.

The `training` flag defaults to `false`, which means evaluation mode. `Core::train()` sets it to `true`; validation sets it back to `false`.

## Typed C++ tensor storage

The C++ `Tensor` type is defined privately in `Core/runtime.cpp`. It records its dtype, shape, row-major strides, element count, data pointer, and gradient pointer. The pointers refer to buffers owned by RAII storage; allocating a tensor creates elements of the declared C++ type, and destruction releases the buffers. `dataAs<T>()` and `gradAs<T>()` provide typed access inside kernels. Gradients use the tensor's storage dtype as well.

The supported C++ dtypes are `FLOAT32`, `FLOAT64`, `INT32`, and `INT64`. Dtype-aware NAIVE kernel entry points dispatch on the relevant tensor dtype and call a C++ template implementation from the operation's header. Kernel arithmetic and accumulations use `Scalar` (`float`) where specified by the template, then store results using the tensor dtype. Operations that combine indices and values dispatch each input independently; for example, embeddings read an integer ID tensor and a floating-point table.

`setInput()` and `setTarget()` receive `std::vector<Scalar>` and convert values into the graph tensor's dtype. C++ getters similarly convert values back to `Scalar`. PHP's current C++ FFI data interface also uses `float` arrays. Thus dtypes are honored inside graph storage and kernels, but arbitrary typed buffers are not yet passed directly across FFI. Eigen half and bfloat16 are not graph dtypes at this time.

## Forward and backward flow

`GraphRuntime::forward()` walks graph operations in order. An operation method resolves tensor IDs to tensor references and dispatches on its recorded kernel name, for example `ADD_2D_LAST` or `ADD_GENERIC_LAST`. Kernel entry points take tensor references rather than IDs. The backward pass walks operations in reverse and calls their corresponding backward methods; backward is a phase of graph execution, not a graph operation. NAIVE kernel entry points call matching templates such as `ADD_2D_LAST_TEMPLATE<T>`.

Generic NAIVE implementations now cover arbitrary valid axes/layouts for the generic kernels used by add, broadcast matmul, transpose, slice, mean, softmax, layer norm, RMS norm, RoPE, cross entropy, logits cross entropy, and integer-label logits cross entropy. The runtime still validates ranks, axes, shapes, and dtype compatibility at kernel boundaries. `GENERIC` means a general shape path; it does not mean every Eigen provider has a specialized implementation.

## Operation coverage and graph features

The native dispatcher handles the current graph operation set, including KV cache. KV cache is a stateful, two-input/two-output graph operation: `PREFILL` initializes per-layer key/value state, `DECODE` appends the current token and exposes the cached sequence, and `resetKvCache()` clears all layers or one selected layer. RoPE offset advances with cached decode steps. Ordinary `TRAIN` and stateless `INFER` preserve the key/value tensors without persistent cache behavior. Persistent cache modes are for autoregressive inference.

The runtime and kernel arithmetic are dtype-aware, but the C++ optimizer is not yet templated: optimizer parameters and moments use `Scalar` (`float`) and it reads and writes tensor elements through scalar accessors. Full precision-preserving optimizer updates for `FLOAT64`, and future Eigen half or bfloat types, still require a typed optimizer path.

For the status and migration plan of SIMD/Eigen-accelerated C++ kernels, see [SIMD and Eigen migration](15-simd-and-eigen.md).
