# 5. Computational graph

## Construction

When a model performs operations on a `Tensor` associated with a `GraphContext`, each operation is recorded. An operation contains:

```json
{
  "op": "matmul",
  "inputs": [0, 2],
  "output": 5,
  "attributes": {}
}
```

Attributes are fixed operation values, such as the selected kernel, dropout percentage, or the scalar used by `scale()`. They are not trainable Tensors. For example, `layer_norm` records `kernel` and `axes`, while its trainable `gamma` and `beta` are normal tensor inputs. Operations such as `reshape()`, whose output shape is already stored in the output tensor definition, do not need to duplicate that shape in their attributes.

## Training graph

`Model::generateGraph()` registers the input, target, parameters, and loss. The loss tensor is marked as `loss` and the runtime uses it as the starting point for backward propagation.

## Inference graph

`Model::generateModel()` registers the `output()` path and saves the output identifier. This graph does not require a target and is used by `predict()`.

It is recommended that `output()` contain no dropout, even though the runtime can bypass it when `training` is `false`.

## Serialization

The graph contains shapes, types, operations, and initial parameter data. Updated weights are saved separately, so the model and weights can be replaced independently.
