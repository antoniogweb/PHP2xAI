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

## Dataset and training input

The training loop receives a `TrainValidateDataset`, which groups the training and validation readers. Both readers implement the shared `BatchDataset` contract: reset an epoch, optionally shuffle it, advance to a batch, and pack that batch into flat row-major `x` and `y` vectors. The loop does not depend on the storage format.

Choose the implementation when creating the readers. `StreamFileDataset` reads the legacy `x|y` text format; `HDF5Dataset` reads named numeric fields from an HDF5 file. For example:

```php
$train = new HDF5Dataset('./Data/train.h5', $batchSize);
$validation = new HDF5Dataset('./Data/validation.h5', $batchSize);
$datasets = new TrainValidateDataset($train, $validation);
$model->train($datasets, $epochs, './weights.json', $logEvery);
```

The same code works with `StreamFileDataset` if the two constructor calls use `.txt` files instead. See [Datasets and batches](06-datasets-and-batches.md) for file layouts, creation examples, shapes, shuffle behavior, and incomplete-batch handling.

When PHP selects the C++ training runtime, `Model::getTrainingConfig()` writes the reader type as `dataset_type` (`TXT` or `HDF5`). The native `Core` uses that setting to construct the matching reader classes. One type is currently shared by training and validation, so both paths in the configuration must point to the same format. Configurations created before this setting existed default to `TXT`.

The graph batch shape must agree with the datasets. `pack()` returns flat arrays, while the input and target placeholders define their tensor dimensions. Both dataset files must contain at least one complete batch. The legacy text reader shuffles batches; HDF5 shuffles individual sample indices before grouping them into batches.

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
