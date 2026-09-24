# 6. Datasets and batches

PHP2xAI can train and validate from either a legacy text dataset or an HDF5 file. The model receives the same kind of batches from both formats: a pair of flat row-major vectors, one for inputs (`x`) and one for targets (`y`). The graph placeholders supply the shapes that give those vectors their tensor dimensions.

The format-specific readers share the `BatchDataset` contract:

```text
resetEpoch()    rewind the batch cursor
shuffleEpoch()  change the order used in this epoch
nextBatch()     prepare the next batch, or report end of epoch
pack()          return [xRowMajor, yRowMajor]
```

In PHP, `StreamFileDataset` and `HDF5Dataset` extend the abstract `BatchDataset`. In C++, the two reader classes implement the virtual `BatchDataset` base class. `TrainValidateDataset` stores references to this base type, so the training and validation loops can use the same operations without checking the file format in each loop.

## Choosing a format

| Format | PHP class | C++ class | Data layout | Shuffle unit |
|---|---|---|---|---|
| Text | `StreamFileDataset` | `StreamFileDataset` | one `x|y` sample per line | batches |
| HDF5 | `HDF5Dataset` | `HDF5Dataset` | named, typed datasets in an `.h5` file | samples |

Use TXT when the data is already available as text, when you want to inspect or edit rows directly, or when a simple streaming reader is sufficient. Use HDF5 when you want named fields, explicit numeric types, and compact binary storage. Both readers drop incomplete final batches, so choose a batch size that leaves at least one full batch in each split.

The C++ training configuration records the training reader's type as `dataset_type`, with the value `TXT` or `HDF5`. The same configured type is used for both training and validation files, so both files should use the same format. Older configurations without `dataset_type` continue to use `TXT`.

## The legacy TXT format

`StreamFileDataset` reads a text file one line at a time. The default separator between input and target is `|`; numbers on each side are separated by whitespace:

```text
0.1 0.2 0.3|1
0.4 0.5 0.6|0
```

The left side is the input vector and the right side is the target vector. A file with three input features and one scalar target per row could look like this:

```text
0.1 0.2 0.3|1
0.4 0.5 0.6|0
0.7 0.8 0.9|1
```

The reader infers the feature and target widths from the first row when it creates graph placeholders. Rows should therefore have consistent widths. Empty or malformed rows, a missing delimiter, or changing feature counts will lead to invalid data or a parse error.

Create and iterate a text dataset like this:

```php
use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;

$dataset = new StreamFileDataset('./train.txt', 32);
echo $dataset->getType(); // TXT

$dataset->resetEpoch();
while ($dataset->nextBatch()) {
	[$x, $y] = $dataset->pack();
	// Use $x and $y with the graph.
}
```

The delimiter can be supplied as the third constructor argument:

```php
$dataset = new StreamFileDataset('./train.csv-like.txt', 32, ';');
```

This is still a whitespace-separated numeric format around the delimiter; it is not a general CSV parser with quoting or escaped delimiters.

### TXT shuffle and streaming behavior

`StreamFileDataset` records byte offsets for batch starts, seeks to the selected batch, and parses its lines as they are read. It does not need to load the entire text file into memory. Its shuffle permutes the order of batches while preserving the sample order inside each batch. This makes file access relatively sequential, but it is not a full sample-level shuffle.

Call `resetEpoch()` before beginning an epoch. Training code typically calls `shuffleEpoch()` after the reset. `shuffleEpoch()` only changes the batch order; `resetEpoch()` resets the cursor. Validation normally resets the cursor without shuffling.

The text reader excludes a partial final batch. For example, 103 rows with a batch size of 32 produce three complete batches (96 rows); the remaining seven rows are not used in that epoch.

## The HDF5 format

HDF5 stores each field as a named numeric dataset. A typical PHP2xAI file has two fields:

```text
/
├── x   input samples
└── y   target samples
```

Each field has a leading sample axis. `setField('x', ..., [784])` declares a sample with 784 values; the stored HDF5 dimensions become `[N, 784]`, where `N` grows as samples are appended. A target declared with shape `[1]` is stored as `[N, 1]`. More complex per-sample shapes are also supported, such as `[28, 28]` or `[sequenceLength, embeddingSize]`.

Supported element types are:

| PHP constant | Value | C++ type used by HDF5 | Example |
|---|---:|---|---|
| `PHP2XAIHDF5::FLOAT32` | 1 | 32-bit float | image pixels |
| `PHP2XAIHDF5::FLOAT64` | 2 | 64-bit float | high-precision values |
| `PHP2XAIHDF5::INT32` | 3 | 32-bit integer | integer features |
| `PHP2XAIHDF5::INT64` | 4 | 64-bit integer | labels and token IDs |

The `shape` passed to `setField()` describes one sample and must contain positive dimensions. It does not include the sample count. The `HDF5Dataset` reader gets the field shape and type from the file metadata. At the C++ graph boundary, values are packed into `float` vectors; this also converts integer labels to the runtime's numeric scalar type.

### Creating an HDF5 dataset in PHP

`PHP2XAIHDF5` is the low-level PHP FFI wrapper for creating and accessing HDF5 files. It exposes `create`, `open`, `destroy`, `setField`, `fieldMetadata`, `count`, `add`, and `readIndices`:

```php
use PHP2xAI\Runtime\PHP\Datasets\PHP2XAIHDF5;

$file = PHP2XAIHDF5::create('./train.h5');
try {
	$file->setField('x', PHP2XAIHDF5::FLOAT32, [3]);
	$file->setField('y', PHP2XAIHDF5::INT64, [1]);

	$file->add('x', [0.1, 0.2, 0.3]);
	$file->add('y', [1]);
	$file->add('x', [0.4, 0.5, 0.6]);
	$file->add('y', [0]);

	echo $file->count(); // 2 samples
	print_r($file->fieldMetadata('x'));
	// ['dtype' => PHP2XAIHDF5::FLOAT32, 'shape' => [3]]
}
finally {
	$file->destroy();
}
```

`add()` appends one sample to one field. Its data argument must already be a flat row-major array whose length equals the product of that field's sample shape. For shape `[2, 3]`, pass six values in row-major order. Add samples to each field in the same order and keep every field's sample count equal. `count()` verifies that fields in the file have consistent counts.

`create()` truncates an existing file at the given path. `open()` opens an existing file for reading and writing. `destroy()` closes the native HDF5 handle and is safe to call more than once; use `try/finally` when writing files so the handle is closed even if an exception occurs.

For larger datasets, prepare the input samples in a loop and call `add()` once per sample and field. HDF5 stores the values as typed binary data rather than formatting each number as text, and the HDF5 file can be read by other HDF5 tools.

### Reading and batching an HDF5 file

`HDF5Dataset` takes the filename, batch size, and optional names for the input and target fields. The defaults are `x` and `y`:

```php
use PHP2xAI\Runtime\PHP\Datasets\HDF5Dataset;

$dataset = new HDF5Dataset('./train.h5', 32);
echo $dataset->getType(); // HDF5

$dataset->resetEpoch();
while ($dataset->nextBatch()) {
	[$x, $y] = $dataset->pack();
	// $x and $y are flat row-major vectors for the same samples.
}
```

If the file uses other field names, pass them explicitly:

```php
$dataset = new HDF5Dataset('./records.h5', 32, 'features', 'labels');
```

`fieldMetadata($name)` reports the dtype and per-sample shape. `HDF5Dataset` uses this information to create the graph input and target placeholders. For a sample shape `[28, 28]` and training batch size 32, the input placeholder shape is `[32, 28, 28]`; `pack()` still returns a flat vector of `32 * 28 * 28` values.

The reader also implements `IteratorAggregate`, like `StreamFileDataset`. Nested iteration returns each batch and then its individual `[x, y]` samples:

```php
foreach ($dataset as $batch) {
	foreach ($batch as [$x, $y]) {
		// $x is one flat input sample; $y is one flat target sample.
	}
}
```

This form is useful for per-sample inference and metrics. The outer loop yields batches; it does not yield one flat packed batch. Use `nextBatch()` plus `pack()` when passing full batches to the graph.

### HDF5 shuffle and incomplete batches

`HDF5Dataset::shuffleEpoch()` shuffles the sample index list, then `nextBatch()` groups the shuffled indices into batches. Individual samples can therefore move between batches from one epoch to the next. The values for `x` and `y` use the same index list, so their correspondence is retained.

The last incomplete batch is dropped because the graph normally has a fixed batch shape. HDF5 uses `floor(N / batchSize)` batches. Since the full sample index list is shuffled, the samples in the trailing remainder can differ between epochs. If every sample must be used in every epoch, choose a batch size that divides the sample count or add a future variable-size batch path to the graph.

`readIndices()` accepts an ordered list of sample indices and returns a flat row-major buffer in that same order. The C++ implementation combines the requested sample selections and reads a whole field batch with one `H5Dread()` call, then restores the requested order. `HDF5Dataset::pack()` reads `x` and `y` separately, so this is one HDF5 read per field per batch. Random sample shuffling can still touch many HDF5 chunks, but batching the read avoids one HDF5 call per sample.

## Using either format for training

`TrainValidateDataset` accepts two `BatchDataset` objects. Select the format when you construct each dataset; the training and validation loop does not need to know which format is underneath:

```php
use PHP2xAI\Runtime\PHP\Datasets\HDF5Dataset;
use PHP2xAI\Runtime\PHP\Datasets\TrainValidateDataset;

$train = new HDF5Dataset('./Data/train.h5', 32);
$validation = new HDF5Dataset('./Data/validation.h5', 32);
$datasets = new TrainValidateDataset($train, $validation);

$model->train($datasets, 20, './weights.json', 10);
```

The same setup works with `StreamFileDataset`:

```php
$train = new StreamFileDataset('./Data/train.txt', 32);
$validation = new StreamFileDataset('./Data/validation.txt', 32);
$datasets = new TrainValidateDataset($train, $validation);
```

`getType()` returns `TXT` or `HDF5`. When PHP builds a C++ training configuration, `Model::getTrainingConfig()` writes that value as `dataset_type`, along with the training and validation paths and batch size. C++ `Core` constructs `StreamFileDataset` for `TXT` and `HDF5Dataset` for `HDF5`. Configurations created before `dataset_type` existed default to `TXT`.

The training and validation files should have the same type because the configuration currently contains one `dataset_type` value. Both datasets may have distinct file paths and contents, but their fields and per-sample shapes must agree with the model graph. The input and target row-major lengths must match the products of the graph's placeholder dimensions for each batch.

## Runtime and build requirements for HDF5

HDF5 support is provided by the Conan `hdf5` dependency and the C++ HDF5 wrapper. Build the C++ targets from `src/Runtime/CPP` after installing the Conan dependencies:

```bash
cd src/Runtime/CPP
make all
```

The build produces the native training executables and the shared libraries, including `php2xai_hdf5.so` for the PHP FFI wrapper. PHP needs the FFI extension enabled and the HDF5 shared library built for the current platform. `PHP2XAIHDF5` locates `Bin/linux-<architecture>/php2xai_hdf5.so` by default; its optional second argument to `create()` or `open()` can supply a custom library path.

During C++ training, the PHP `HDF5Dataset` closes its HDF5 handles before launching the native process. This releases HDF5's file lock so that C++ can open the same files. The PHP dataset object reopens its handle on the next `pack()` call if it is reused later.

## Choosing batch size and diagnosing data issues

The batch size is part of the graph's input and target shapes. Changing the dataset batch size after graph generation can create a shape mismatch. Make sure the training and validation datasets use the batch size expected by the model and that the dataset has at least one full batch.

For TXT, inspect a few rows and verify the delimiter and vector widths. For HDF5, inspect `fieldMetadata('x')`, `fieldMetadata('y')`, and `count()`. Both fields need a matching sample count; their sample shapes need to match what the model expects. A target field declared with shape `[1]` is typically treated as a scalar label per sample by the graph placeholder, while `pack()` still returns one value per sample in row-major order.

If HDF5 errors mention an inability to lock a file, check that no other process still has the file open for writing. Close handles explicitly with `destroy()` or `HDF5Dataset::close()` before handing the file to another process. The model's C++ training path handles this handoff automatically.
