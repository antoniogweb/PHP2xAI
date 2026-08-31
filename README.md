# PHP2xAI

AI framework with a PHP frontend and a C++ runtime, designed for training and inference of custom neural networks.

PHP handles dataset preparation, computational graph configuration, and training orchestration, while the C++ runtime performs high-performance numerical computation. PHP2xAI is built for experimentation, full architectural control, and integration into web-based applications.

## Installation

```bash
composer require antoniogweb/php2xai
```

## Technical documentation

The complete technical guide is available in the [PHP2XAI documentation](PHP2XAI/01-introduction.md).

It covers:

- architecture and runtime design;
- installation and first model creation;
- tensors, shapes, and operations;
- computational graph construction;
- datasets and batch training;
- PHP and C++ runtimes;
- inference and model serialization;
- dropout and training/evaluation modes;
- custom models and runtime extensions;
- debugging and troubleshooting.

## Quick start

```php
use PHP2xAI\Tensor\Tensor;

$x = Tensor::zeros([2, 3]);
```

Browse the complete guide starting with [Introduction and architecture](PHP2XAI/01-introduction.md).
