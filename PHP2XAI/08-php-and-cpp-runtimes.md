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

Both runtimes implement the same forward and backward behavior for `transpose()`, contiguous `reshape()`, sinusoidal `positionalEncoding()`, `gelu()` with the tanh approximation, `scale()`, `layerNorm()`, and `applyPaddingMask()`. LayerNorm has an optimized contiguous last-axis kernel and a stride-aware generic-axis kernel; both implementations use `epsilon = 1e-5` and accumulate gradients for the input, `gamma`, and `beta`. `applyPaddingMask()` uses a contiguous generic-last-axis path for `[B, ..., L]` scores and a `[B,L]` binary mask, writing negative infinity into masked score positions.

For the status and migration plan of SIMD/Eigen-accelerated C++ kernels, see [SIMD and Eigen migration](15-simd-and-eigen.md).
