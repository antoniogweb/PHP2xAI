# PHP2xAI

AI framework with a PHP frontend and a C++ runtime, designed for training and inference of custom neural networks.

PHP handles dataset preparation, computational graph configuration, and training orchestration, while the C++ runtime performs high-performance numerical computation. PHP2xAI is built for experimentation, full architectural control, and integration into web-based applications.

## Installation

```bash
composer require antoniogweb/php2xai
```

## Technical documentation

The complete technical guide is available below. Each chapter is linked directly from this README.

1. [Introduction and architecture](PHP2XAI/01-introduction.md)
2. [Installation](PHP2XAI/02-installation.md)
3. [First model](PHP2XAI/03-first-model.md)
4. [Tensors and operations](PHP2XAI/04-tensors-and-operations.md)
5. [Computational graph](PHP2XAI/05-computational-graph.md)
6. [Datasets and batches](PHP2XAI/06-datasets-and-batches.md)
7. [Training](PHP2XAI/07-training.md)
8. [PHP and C++ runtimes](PHP2XAI/08-php-and-cpp-runtimes.md)
9. [Inference and model serialization](PHP2XAI/09-inference.md)
10. [Dropout and training mode](PHP2XAI/10-dropout-and-training-mode.md)
11. [Custom models](PHP2XAI/11-custom-model.md)
12. [Extending the runtime](PHP2XAI/12-extending-the-runtime.md)
13. [Debugging](PHP2XAI/13-debugging.md)
14. [Tokenizer](PHP2XAI/14-tokenizer.md)
15. [SIMD and Eigen migration](PHP2XAI/15-simd-and-eigen.md)

## Quick start

```php
use PHP2xAI\Tensor\Tensor;

$x = Tensor::zeros([2, 3]);
```

Browse the complete guide starting with [Introduction and architecture](PHP2XAI/01-introduction.md).
