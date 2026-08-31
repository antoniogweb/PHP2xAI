# 6. Datasets and batches

## File format

`StreamFileDataset` reads one line at a time. The default separator is `|`:

```text
0.1 0.2 0.3|1
0.4 0.5 0.6|0
```

The left side is `x`; the right side is `y`. Values are numbers separated by spaces.

## Batch iteration

```php
$dataset->resetEpoch();

while ($dataset->nextBatch()) {
    [$x, $y] = $dataset->pack();
}
```

`pack()` returns flat vectors in row-major order. For three samples with two features:

```text
x = [x00, x01, x10, x11, x20, x21]
```

The runtime uses the placeholder shape to interpret the vector as `[3, 2]`.

## Shuffle

```php
$dataset->resetEpoch();
$dataset->shuffleEpoch();
```

Shuffling changes batch order, not individual sample order. This preserves streaming reads and reduces random file access.

## Incomplete batches

The last batch may contain fewer samples than the configured batch size. The runtime must therefore support the effective batch size, or the dataset must be prepared to produce full batches when the graph requires a fixed shape.
