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
