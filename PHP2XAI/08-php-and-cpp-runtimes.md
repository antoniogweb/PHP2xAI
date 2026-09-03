# 8. PHP and C++ runtimes

Both runtimes execute the same serialized graph.

| Function | PHP | C++ |
|---|---:|---:|
| Model definition | yes | no |
| Graph generation | yes | no |
| Training | yes | yes |
| Validation | yes | yes |
| Inference | yes | yes |
| Streaming dataset | yes | yes |
| Batch `[B,D]` | yes | yes |

With the C++ runtime, the PHP model normally generates:

- `model.json` for the inference graph;
- `config.json` for training, optimizer, and dataset settings;
- `weights.json` for the best weights.

The C++ runtime reads the configuration, loads graph and weights, constructs the datasets, and starts `Core::train()`.

The `training` flag defaults to `false`, which means evaluation mode. `Core::train()` sets it to `true`; validation sets it back to `false`.

Both runtimes implement the same forward and backward behavior for `transpose()`, contiguous `reshape()`, sinusoidal `positionalEncoding()`, `gelu()` with the tanh approximation, `scale()`, and `layerNorm()`. LayerNorm has an optimized contiguous last-axis kernel and a stride-aware generic-axis kernel; both implementations use `epsilon = 1e-5` and accumulate gradients for the input, `gamma`, and `beta`.
