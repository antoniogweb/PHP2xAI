# 4. Tensors and operations

## Creation

```php
$a = Tensor::zeros([4, 8]);
$b = Tensor::random([4, 8]);
$w = Tensor::init([8, 16], 0.05);
```

Shapes are expressed as arrays. Data is stored in row-major order.

## Main operations

```php
$h = $x->matmul($W)->add($b);
$h = $h->ReLU();
$p = $h->softmax();
```

Available operations include:

- `matmul()` for matrix multiplication;
- `add()` for biases and supported broadcasting;
- `sig()`, `ReLU()`, and `gelu()` for activations;
- `scale()` for elementwise multiplication by a fixed scalar;
- `layerNorm()` for feature normalization with trainable scale and bias;
- `transpose()` and `reshape()` for layout changes;
- `positionalEncoding()` for sinusoidal sequence positions;
- `softmax()` for probabilities;
- `CE()`, `CELogits()`, and `CELogitsLabelInt()` for cross entropy;
- `mean()` for reducing batch losses;
- `embeddings()`, `paddingMask()`, and `meanPooling()` for sequences;
- `dropout()` for regularization.

## Batch shapes

The usual batch shape is:

```text
x: [B, D]
y: [B] or [B, C]
```

The model must be built so that every operation supports the leading batch dimension.

## Parameters and gradients

A parameter can be marked as non-trainable:

```php
$tensor->setTrainable(false);
```

This does not necessarily prevent gradients from being computed with respect to the tensor; it prevents the tensor from being included among the parameters updated by the optimizer.


## What a Tensor represents

A `Tensor` is the basic data object used by PHP2xAI. It describes a multidimensional array of numerical values together with its shape, gradient storage, name, and graph context. Tensors can represent inputs, targets, model parameters, intermediate activations, and losses.

The same Tensor API is used while defining a model and while describing the values executed by a runtime. This is what allows a model written in PHP to be exported as a graph and later executed by the C++ runtime. Model code therefore looks like numerical code, while also recording the computation needed for forward and backward passes.

The shape is always expressed as an array of dimensions. `[4, 8]` means four rows with eight values per row. `[2, 3, 4]` means two blocks, each containing a tensor with shape `[3, 4]`. Shape conventions should be documented for every model because most runtime errors are caused by incompatible dimensions rather than by the arithmetic itself.

## Flat storage and row-major order

Internally, Tensor values are stored in a flat one-dimensional array. The shape determines how those values are interpreted. PHP2xAI uses row-major order, so the last dimension varies fastest.

For example, a tensor with shape `[2, 3]`:

```text
[[a, b, c],
 [d, e, f]]
```

is stored as:

```text
[a, b, c, d, e, f]
```

This is important when using `StreamFileDataset::pack()`. The dataset returns a flat vector, but the input placeholder tells the runtime whether that vector represents `[B, D]`, `[B, T, D]`, or another shape. The number of values must always match the product of the dimensions in the declared shape.

## Graph construction and deferred execution

When a Tensor belongs to a `GraphContext`, operations are normally registered instead of being treated as isolated array manipulations. For example:

```php
$hidden = $x->matmul($W)->add($b)->ReLU();
```

creates the chain `matmul → add → ReLU` in the graph. Each operation records its input identifiers, output identifier, and optional fixed attributes. The runtime can later execute this graph in order for forward propagation and in reverse order for backward propagation.

This means that model methods should describe the computation and should not manually update parameters. Parameter updates belong to the optimizer, after gradients have been computed. Keeping these responsibilities separate makes the same graph usable by both the PHP and C++ runtimes.

## Matrix multiplication and dense layers

`matmul()` is the main operation used to implement dense layers:

```php
$logits = $x->matmul($W)->add($b);
```

For a batch, the expected shapes are normally:

```text
x:      [B, D]
W:      [D, H]
b:      [H]
result: [B, H]
```

The inner dimensions must match: the input feature dimension `D` must equal the first dimension of `W`. The result contains one row per sample and one column per output unit. The bias vector is then broadcast over the leading batch dimension.

For a single sample, the corresponding path may use `[D]` and `[D, H]` and produce `[H]`, depending on the operation and runtime path. A model intended for batch training should use one convention consistently. Accidentally mixing sample vectors and batch matrices is a common source of incompatible input errors.

## Transpose and reshape

`transpose()` swaps exactly two axes. By default it swaps the final two dimensions:

```php
$matrixT = $matrix->transpose();          // [-2, -1]
$byHead = $x->transpose([1, 2]);          // [B, L, H, dk] -> [B, H, L, dk]
$forScores = $byHead->transpose();        // [B, H, L, dk] -> [B, H, dk, L]
```

Negative axis indices are accepted. The operation supports dedicated paths for common 2D, 3D, and attention 4D layouts, with a generic fallback for any valid pair of distinct axes. PHP2xAI materializes the transposed values into a new contiguous output tensor; it does not currently create a non-contiguous transpose view.

`reshape()` changes how the same row-major sequence of values is grouped:

```php
$flat = $x->reshape([2, -1]); // [2, 3, 4] becomes [2, 12]
```

The product of the requested dimensions must equal the input element count. One dimension can be `-1`; its value is inferred from the remaining dimensions. The current runtime requires a contiguous input and produces a contiguous output by copying the flat storage. Its backward pass is a linear identity mapping of gradients.

For example, reshape does not transpose data:

```text
[ [1, 2, 3], [4, 5, 6] ]  shape [2, 3]
becomes [ [1, 2], [3, 4], [5, 6] ]  shape [3, 2]
```

## Layer normalization

`layerNorm()` normalizes one axis and then applies a trainable scale `gamma` and bias `beta`:

```php
$gamma = Tensor::createFromData(array_fill(0, $D, 1.0));
$beta = Tensor::zeros([$D]);
$normalized = $x->layerNorm($gamma, $beta); // axis = -1
```

`gamma` and `beta` must both have shape `[D]`, where `D` is the size of the normalized axis. The default is the final axis, so it is the usual Transformer operation for `[B, L, D]` and also works for `[D]`, `[B, D]`, and `[B, H, L, D]`. For example, every token independently normalizes its `D` features in `[B, L, D]`.

For a slice `x` of `D` values, the forward pass uses population variance and `epsilon = 1e-5`:

```text
mean     = (1 / D) * sum_i x_i
variance = (1 / D) * sum_i (x_i - mean)^2
xHat_i   = (x_i - mean) / sqrt(variance + epsilon)
y_i      = gamma_i * xHat_i + beta_i
```

An explicit non-final axis selects the generic path:

```php
// For every fixed [B, :, L, D] slice, normalize over H.
$normalized = $x->layerNorm($gammaH, $betaH, 1); // x: [B, H, L, D]
```

The default `axis = -1` selects `LAYER_NORM_LAST_AXIS`, a contiguous fast path that treats the tensor as `outer = numel / D` independent rows. Any other axis selects `LAYER_NORM_GENERIC`, which iterates slices using shape and strides. Both backward paths accumulate gradients for `X`, `gamma`, and `beta`; `gamma` and `beta` remain ordinary trainable parameter tensors.

## Addition and broadcasting

`add()` is most often used for bias addition:

```php
$hidden = $x->matmul($W)->add($b);
```

If the matrix has shape `[B, H]` and the bias has shape `[H]`, the same bias is applied independently to all `B` rows. Broadcasting does not create `B` trainable copies: there is still only one parameter vector with `H` values.

The runtime supports specific broadcasting patterns rather than every broadcasting rule available in array-oriented languages. When an addition fails, inspect the complete shapes of both tensors and verify that the last feature dimensions match.

## Activations and numerical meaning

Activation functions are generally applied element by element:

```php
$hidden = $hidden->ReLU();
$probabilities = $hidden->softmax();
```

`ReLU()` preserves positive values and maps negative values to zero. It is commonly used in hidden layers. `sig()` maps values to `(0, 1)` and is useful for binary outputs or gates. `softmax()` transforms a group of logits into normalized values whose sum is one along the selected axis.

`gelu()` uses the common tanh approximation of the Gaussian Error Linear Unit:

```text
GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
```

It is elementwise and preserves the input shape. When comparing with PyTorch, use `torch.nn.functional.gelu(x, approximate="tanh")`.

`scale()` is also elementwise and multiplies every value by a fixed scalar:

```php
$scaled = $x->scale(0.5);
```

Its backward pass multiplies the upstream gradient by the same scalar. The scalar is a fixed operation attribute, not a trainable Tensor.

The activation must match the loss. A loss that expects logits, such as `CELogitsLabelInt()`, should normally receive the raw output of the final linear layer. Applying softmax before a logits-based loss changes the mathematical expression and can make optimization less stable.

## Losses and reductions

The available loss-related operations include `CE()`, `CELogits()`, and `CELogitsLabelInt()`. The last one is useful for multiclass classification with integer labels:

```php
$loss = $logits
    ->CELogitsLabelInt($labels)
    ->mean();
```

The final `mean()` is important in batch training. It turns per-sample losses into one scalar and normalizes the contribution of the batch. `GraphRuntime::getError()` also reports the mean when a loss tensor contains multiple values, but reducing the loss explicitly in the graph is preferable because it also gives the backward pass the correct reduction operation.

## Batch dimensions

The standard dense-model convention is:

```text
x: [B, D]
y: [B] or [B, C]
```

`B` is the number of samples, `D` is the input feature count, and `C` is the number of classes or target channels. Integer-label classification usually uses `[B]`, while one-hot or distribution targets usually use `[B, C]`.

Every layer must preserve the meaning of the leading batch dimension. A dense layer changes `[B, D]` into `[B, H]`; an elementwise activation keeps `[B, H]`; a reduction may remove one or more dimensions. When designing a model, write the expected shape beside every operation and check both the forward and backward shapes.

## Sequence and embedding operations

Sequence models use a similar principle with an additional time or token dimension:

```php
$mask = $x->paddingMask(0);
$embeddings = $x->embeddings($embeddingTable);
$pooled = $embeddings->meanPooling($mask);
```

A common shape flow is:

```text
x:          [B, T]
embeddings: [B, T, D]
pooled:     [B, D]
```

`paddingMask()` marks valid positions, `embeddings()` maps integer token identifiers to rows in an embedding table, and `meanPooling()` combines valid token representations while ignoring padding. The mask is part of the computation and must have a shape compatible with the sequence representation.

### Sinusoidal positional encoding

`positionalEncoding()` applies the standard sinusoidal encoding to token embeddings with shape `[B, L, D]`:

```php
$x = $x->positionalEncoding();
```

`B` is the batch dimension, `L` is the token position, and `D` is the embedding dimension. Rank `[B, D]` is rejected because batch entries do not represent positions in one sequence. For every position `p` and feature-pair index `i`:

```text
PE(p, 2i)   = sin(p / 10000^(2i / D))
PE(p, 2i+1) = cos(p / 10000^(2i / D))
```

The same `[L, D]` encoding is added independently to every batch item. Positional encoding has derivative 1 with respect to its input, so backward passes the upstream gradient through unchanged.

## Dropout and fixed operation attributes

Dropout is a regularization operation with a fixed hyperparameter:

```php
$hidden = $hidden->dropout(20.0);
```

The argument is a percentage, so `20.0` means that approximately 20 percent of values are dropped during training. Retained values are scaled using inverted dropout. The percentage is serialized as an operation attribute; it is not a Tensor, is not listed in `trainable`, and is never updated by the optimizer.

The runtime uses `training = true` to generate a random mask and `training = false` to return the input unchanged. The exact mask generated in the forward pass is saved and reused by backward. This is essential after `ReLU()`, because an input equal to zero cannot reveal whether it was retained or dropped.

## Parameters, trainability, and gradients

Model parameters are Tensors assigned to model properties:

```php
$this->W = Tensor::init([D, H], 0.05);
$this->b = Tensor::zeros([H]);
```

During graph generation, these values are registered as `param` tensors. The exported graph contains the identifiers of trainable parameters, and the optimizer updates only those identifiers.

A tensor can be excluded from updates:

```php
$tensor->setTrainable(false);
```

Trainability and gradient propagation are separate concepts. `setTrainable(false)` prevents optimizer updates, while `setRequiresGrad(true)` controls whether gradients are propagated through the tensor. A frozen embedding table may therefore remain non-trainable while still allowing gradients to flow through the embedding output to earlier operations.

## Practical shape checklist

When adding an operation or a new model layer, verify:

1. input ranks and dimensions;
2. output shape;
3. flat element count;
4. batch dimension and its position;
5. target shape expected by the loss;
6. backward gradient shape;
7. fixed attributes versus trainable parameters.

A small hand-written example with known values is often more useful than a full training run when debugging a new operation. First verify the forward result, then verify that gradients have the expected shape and sign, and only then test convergence on a real dataset.
