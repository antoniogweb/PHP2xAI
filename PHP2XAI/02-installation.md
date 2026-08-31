# 2. Installation

## Composer

PHP2xAI is installed as a Composer package:

```bash
composer require antoniogweb/php2xai
```

In the application, include the autoloader:

```php
require __DIR__ . '/vendor/autoload.php';
```

## Requirements

The PHP runtime requires PHP and Composer. The C++ runtime additionally requires:

- a C++17-compatible compiler;
- a compiled PHP2xAI runtime library;
- numerical dependencies such as Eigen, when enabled by the build;
- access to `proc_open` when C++ training is started from the PHP model.

## Quick check

```php
use PHP2xAI\Tensor\Tensor;

$x = Tensor::zeros([2, 3]);
echo json_encode($x->getShape());
```

A complete check should also create a model, generate its graph, and run one forward pass.
