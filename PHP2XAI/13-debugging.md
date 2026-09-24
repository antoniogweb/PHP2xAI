# 13. Debugging and troubleshooting

## Training does not converge

Check:

- learning rate;
- weight initialization;
- batch shape;
- loss reduction with `mean()`;
- target compatibility with the loss;
- gradient clipping;
- dropout active only during training;
- validation in evaluation mode.

## Dimension errors

Print the shape and number of elements of input and target. `pack()` returns flat vectors, so the shape is determined by the placeholder registered in the graph.

For HDF5, inspect `fieldMetadata('x')`, `fieldMetadata('y')`, and `count()`. The metadata shape describes one sample and excludes the leading sample axis. Check that the number of elements returned by `pack()` equals `batchSize * product(sampleShape)` for each field, and that the field shapes agree with the graph placeholders. The input and target fields must contain the same number of samples.

## HDF5 file cannot be opened

Confirm that the file exists at the path resolved by the running script, that both fields are present, and that no other process holds an incompatible HDF5 file lock. PHP FFI requires the `php2xai_hdf5.so` library for the current platform; native C++ training requires the HDF5-linked runtime built from the C++ sources. When PHP launches C++ training, the framework closes the PHP dataset handles before starting the native process to release their locks. Also check `dataset_type` in the training configuration: it must be `HDF5` for `.h5` paths, and must match the format of both training and validation datasets.

## Noisy validation

Verify that the runtime has `training = false` during `validationLoss()`. If the graph contains dropout and remains in training mode, the loss changes randomly on every pass.

## PHP and C++ produce different results

Use the same batch and compare, in order:

1. packed input;
2. output of every operation;
3. loss;
4. parameter gradients;
5. weights after one step.

Also check that the JSON graph is identical and that operation attributes are loaded by C++.

## Dropout problems

Backward must use the same mask as forward. Reconstructing it from input and output values is incorrect when the input contains zeros, for example after `ReLU()`.

## Final checklist

- The dataset has no empty or malformed lines.
- The configured delimiter is correct.
- `x` and `y` have consistent dimensions.
- The final loss is scalar.
- Only trainable tensors are updated.
- The exported model does not apply dropout during inference.
