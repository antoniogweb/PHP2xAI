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
