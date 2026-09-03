# PHP2xAI BPE Tokenizer

## Credits and attribution

PHP2xAI's tokenizer is built on the open-source **Hugging Face Tokenizers** library, which provides efficient implementations of the BPE model, ByteLevel tokenization, normalization, and vocabulary management. The Rust trainer included in this repository directly uses Hugging Face's `tokenizers` crate; the generated `tokenizer.json` follows the tokenizer serialization format produced by the library.

The project acknowledges and thanks Hugging Face and all contributors to the Tokenizers project:

- project: [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers);
- documentation: [Hugging Face Tokenizers documentation](https://huggingface.co/docs/tokenizers);
- license and copyright: the upstream code is distributed under the Apache License 2.0; see `ThirdParty/tokenizers/LICENSE` and the notices in the upstream project.

PHP2xAI is not affiliated with, sponsored by, or endorsed by Hugging Face. Hugging Face and Tokenizers are trademarks and projects of their respective owners. Hugging Face Tokenizers code remains subject to its own license; this attribution does not alter the upstream license terms or those of PHP2xAI.

For the concrete configuration used by the trainer, see `src/Tokenizer/Rust/Trainer/src/main.rs`: the tokenizer combines BPE with ByteLevel pre-tokenization and decoding, NFC normalization, and the special tokens `[PAD]`, `[UNK]`, `[BOS]`, and `[EOS]`.


## Purpose

PHP2xAI uses a ByteLevel BPE tokenizer to convert text into a sequence of integer IDs. The IDs are indices into the embedding table: if the table has shape `[V, D]`, its first dimension `V` must be the actual vocabulary size.

This guide explains how to generate the tokenizer using the Linux binary included in the repository:

```text
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

The generated file is named `tokenizer.json`.

## Complete workflow

1. Prepare a UTF-8 corpus, usually named `corpus.txt`.
2. Run the Rust trainer.
3. Keep `tokenizer.json` and `tokenizer.meta.json` together.
4. Read `vocabulary_size` from the metadata.
5. Use that value as `V` in the `[V, D]` embedding table.
6. Load the same `tokenizer.json` during both training and inference.

The token-to-ID mapping is part of the model. Replacing the tokenizer after creating the embeddings can make all embedding-table weights incompatible.

## Corpus

The corpus is a UTF-8 text file. Each line is one sample:

```text
This is the first document.
The tokenizer learns from the texts included in the corpus.
A line can contain a sentence or a long document.
```

Empty lines and lines containing only spaces are ignored when computing statistics. The corpus should represent the data used by the model: include uppercase letters, punctuation, numbers, accents, and emoji when they occur in real data.

You do not need to add token IDs or special tokens to the corpus. The trainer registers them itself.

Check the encoding before running the trainer. For example:

```bash
file -bi /path/to/data/corpus.txt
```

## Linux prerequisites

The included binary is compiled for Linux `x86_64`:

```bash
uname -m
file src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

The first command should return `x86_64`. If necessary, make the file executable:

```bash
chmod +x src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer
```

## Minimal generation

From the repository root:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /path/to/data/corpus.txt
```

Without `--output`, the trainer writes to the current directory (`pwd`):

```text
tokenizer.json
tokenizer.meta.json
```

The current directory is not the binary's directory. To avoid ambiguity, work in a dedicated directory:

```bash
mkdir -p artifacts/tokenizer
cd artifacts/tokenizer

/path/to/PHP2xAI/src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /path/to/data/corpus.txt
```

## Explicit path

Use `--output` (or `-o`) to choose the tokenizer filename:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input /path/to/data/corpus.txt \
    --output /path/to/model/tokenizer.json
```

Missing parent directories are created automatically. Metadata is written next to the tokenizer with a fixed name:

```text
/path/to/model/tokenizer.json
/path/to/model/tokenizer.meta.json
```

Use the `.json` extension, not `.js`.

## Options

```text
-i, --input <FILE>
    UTF-8 corpus. Required.

-o, --output <FILE>
    Destination tokenizer JSON file.
    Default: tokenizer.json in the current directory.

-v, --vocab-size <NUMBER>
    Maximum vocabulary size.
    Default: 30000.

-f, --min-frequency <NUMBER>
    Minimum frequency of BPE pairs.
    Default: 2.

--pretty
    Writes formatted JSON instead of compact JSON.

-h, --help
    Displays help.
```

To view the help version actually embedded in the binary:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer --help
```

### Maximum vocabulary size

`--vocab-size 30000` is a maximum, not a guarantee of 30,000 tokens. A small corpus or a high minimum frequency can produce a smaller vocabulary. Always use `vocabulary_size` from the metadata file for the embedding table; do not blindly copy the value passed to `--vocab-size`.

Complete example:

```bash
src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input corpus.txt \
    --output model/tokenizer.json \
    --vocab-size 30000 \
    --min-frequency 2 \
    --pretty
```

## Generated files

### `tokenizer.json`

It contains the complete tokenizer: BPE model, token-to-ID vocabulary, BPE merges, NFC Unicode normalizer, ByteLevel pre-tokenizer, ByteLevel decoder, and special tokens. This is the file passed to the Rust/C binding and the PHP tokenizer class.

### `tokenizer.meta.json`

It contains statistics computed after training. An example is:

```json
{
  "vocabulary_size": 18472,
  "samples": 50000,
  "max_length": 3105,
  "average_length": 291.13634,
  "p50": 216,
  "p95": 750,
  "p99": 1139
}
```

The field used by the model is:

```json
"vocabulary_size": 18472
```

In the current trainer, IDs are contiguous starting at zero, therefore:

```text
Valid IDs: 0 ... vocabulary_size - 1
embedding table: [vocabulary_size, D]
```

The other properties mean:

- `samples`: number of non-empty lines;
- `max_length`: maximum length in tokens;
- `average_length`: average length;
- `p50`, `p95`, `p99`: length percentiles;
- `vocabulary_size`: number of rows required by the embedding table.

## Special tokens

The trainer assigns IDs in this order:

```text
0 -> [PAD]
1 -> [UNK]
2 -> [BOS]
3 -> [EOS]
```

`[PAD]` fills sequences, `[UNK]` represents unknown content, `[BOS]` marks the beginning, and `[EOS]` the end. They are already included in `vocabulary_size`: do not add `+4`.

For example, if the metadata reports `30000`, the correct table is `[30000, D]`, with rows from `0` to `29999`; the first four rows are the special tokens.

## Using metadata in PHP

The model must read `vocabulary_size` from the generated file:

```php
<?php

$metadataPath = __DIR__ . '/tokenizer.meta.json';
$json = file_get_contents($metadataPath);
if ($json === false) {
    throw new RuntimeException("Unable to read {$metadataPath}");
}

$metadata = json_decode($json, true, 512, JSON_THROW_ON_ERROR);
$V = (int) ($metadata['vocabulary_size'] ?? 0);
if ($V < 4) {
    throw new RuntimeException('Invalid vocabulary_size');
}

$D = 512;
$embTable = Tensor::init([$V, $D], 0.05);
```
