# 16. Exercises and example projects

PHP2xAI is accompanied by three separate repositories. They are intentionally kept outside the framework repository: the exercises validate individual operations, while the MNIST and sentiment projects are complete applications built with the public API.

## Applications git repositories

The Git remotes are:

- [PHP2xAI exercises](https://github.com/antoniogweb/exercises)
- [PHP2xAI MNIST](https://github.com/antoniogweb/PHP2xAI-mnist)
- [PHP2xAI sentiment](https://github.com/antoniogweb/PHP2xAI-sentiment)

## Low-level exercises


It is the playground for validating tensor primitives, shape rules, forward values, and autograd behavior while the framework is developed. Its source tree is organized by implementation or reference:

```text
src/PHP/    exercises executed by the PHP runtime
src/CPP/    the same graph exercises executed by the C++ runtime
src/Torch/  PyTorch reference implementations
```

Examples cover tensor creation, `matmul`, broadcasting with `add`, `mean`, `softmax`, integer-label cross entropy, `transpose`, `reshape`, `scale`, `gelu`, and `layerNorm`. A new operation should normally receive one small exercise in each directory. The PHP and C++ results are compared with Torch to catch shape mistakes and backward errors early.

Run a PHP exercise from its directory so that its relative Composer autoloader path resolves correctly:

```bash
CD src/PHP
php layer_norm.php
```

The matching C++ runtime exercise is launched through its PHP entry point:

```bash
CD src/CPP
php layer_norm.php
```

The Torch reference can be run with an environment containing PyTorch:

```bash
cd src/Torch
python layer_norm.py
```

## MNIST classification project

It demonstrates an end-to-end image classification workflow: preprocessing images into `DataLabelInt` files, graph generation in PHP, dense-network training, weight serialization, and validation. The model in `src/model.php` uses three dense layers with ReLU activations and ten output logits.

Its prepared training and test files are expected at:

```text
src/DataLabelInt/Training/train.txt
src/DataLabelInt/Training/test.txt
```

Run training and validation from `src/`:

```bash
cd src
php train.php
php validate.php
```

The example is configured to use the C++ runtime and can select the Eigen provider. It writes artifacts such as `model.json` and `weights.json` under `src/`. If the prepared dataset files are missing, `create_data_one_file.php` generates them from the MNIST images expected in `src/images/`.

## Sentiment analysis project

It demonstrates the text pipeline: corpus construction, ByteLevel BPE tokenizer training, conversion to integer-token datasets, embeddings, mean pooling, dense classification layers, dropout, and C++/Eigen training. The model in `src/model.php` predicts two sentiment classes from sequences of up to 1024 tokens.

The preparation sequence, run from the repository root, is:

```bash
php src/corpus.php
vendor/antoniogweb/php2xai/src/Tokenizer/Rust/Trainer/Bin/linux-x86_64/php2xai-tokenizer-trainer \
    --input src/corpus.txt --output src/tokenizer.json \
    --vocab-size 30000 --min-frequency 2
php src/create_dataset.php
```

The tokenizer's effective `vocabulary_size` in `src/tokenizer.meta.json` must match the vocabulary dimension used for the embedding table. The generated numerical datasets are saved to `src/DataLabelInt/train.txt` and `src/DataLabelInt/test.txt`.

Train and validate from `src/`:

```bash
cd src
php train.php
php validate.php
```

Training saves `model.json`, `weights.json`, and related artifacts in `src/`; validation reloads those artifacts and reports test accuracy and elapsed time.

## Development workflow

Use the repositories in this order when extending PHP2xAI:

1. Add and verify the primitive in `PHP2xAI-exercises` against Torch.
2. Add or update the framework implementation and its documentation in `PHP2xAI`.
3. Exercise the new capability in a complete project when it applies to MNIST, sentiment, or another model repository.

This separation keeps low-level numerical validation independent from application datasets and makes it easier to identify whether a regression comes from an operation, a graph, or data preparation.
