# 10. Dropout and training mode

## Usage

```php
$h = $h->dropout(20.0);
```

The parameter is a real dropout percentage. `dropout(20.0)` drops approximately 20% of the elements and scales retained elements by `1 / 0.8`.

The coefficient is a fixed hyperparameter:

- it is saved as an operation attribute;
- it is not a Tensor;
- it is not included in `trainable`;
- it is not updated by the optimizer.

## Training and evaluation

```text
training = true  → random mask and inverted dropout
training = false → output equals input
```

The runtime saves the mask generated during forward and reuses it during backward. This is especially important after `ReLU()`, because many values can be zero: using `y / x` cannot reconstruct the correct mask when the input is zero.

## Model recommendation

Dropout should be part of the `forward()` path used by the loss. The `output()` path should normally omit it:

```php
public function forward(Tensor $x): Tensor
{
    return $x->matmul($this->W)
            ->add($this->b)
            ->ReLU()
            ->dropout(20.0);
}

public function output(Tensor $x): Tensor
{
    return $x->matmul($this->W)
            ->add($this->b)
            ->ReLU();
}
```

Even though `training = false` neutralizes dropout, an inference graph without that operation is smaller and more explicit.
