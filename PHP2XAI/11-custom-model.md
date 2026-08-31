# 11. Creating a custom model

## Parameters

Assigning a Tensor to a model property makes it part of the configuration:

```php
$this->W = Tensor::init([inputSize, outputSize]);
$this->b = Tensor::zeros([outputSize]);
```

Use stable names to make diagnostics and saving easier.

## `forward()`

Contains the training path. Dropout and other stochastic operations may appear here.

## `output()`

Contains the inference path. It should return the output needed by the application, normally probabilities or logits, without training-only operations.

## `loss()`

Receives input and target, calls the training path, and produces a scalar loss tensor:

```php
public function loss(Tensor $x, Tensor $y): Tensor
{
    return $this->forward($x)
                ->CELogitsLabelInt($y)
                ->mean();
}
```

## Shapes

For every model, document:

- input shape;
- target shape;
- parameter shapes;
- output shape;
- axes used by `softmax` and `mean`.
