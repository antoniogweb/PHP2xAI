# 7. Training

The PHP and C++ training loops follow the same sequence:

```text
resetGrad
setLossGrad
pack batch
setInput / setTarget
forward
getError
backward
optimizer step
```

The core PHP loop is conceptually:

```php
$graph->resetGrad();
$graph->setLossGrad(1.0);
[$x, $y] = $dataset->train->pack();
$graph->setInput($x);
$graph->setTarget($y);
$graph->forward();
$error = $graph->getError();
$graph->backward();
$this->step($graph);
```

The loss should be reduced with `mean()` over the batch. This normalizes the gradient with respect to the number of reduced elements and prevents it from depending directly on batch size.

After each epoch, validation loss is computed. If it improves and a save path was provided, the weights are written to that path.

## Optimizers

Optimizers receive gradients for trainable parameters and update only those tensors. Adam maintains moments; Fixed applies a simple constant-learning-rate update.

## Gradient clipping

For models with embeddings or long sequences, limiting gradients can be useful:

```php
$optimizer->setGradClip(1.0);
```

Clipping protects against numerical spikes; it does not replace a suitable learning rate and initialization.
