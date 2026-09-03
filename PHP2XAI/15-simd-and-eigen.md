# 15. SIMD and Eigen migration

The C++ runtime is being migrated incrementally toward SIMD-friendly Eigen implementations for the operations that dominate neural-network execution time. The goal is to improve performance without changing the graph format, Tensor API, numerical semantics, or the PHP runtime reference implementation.

## Why Eigen

Eigen is a header-only C++ linear-algebra library. Its matrix expressions can use SIMD instructions made available by the compiler and target architecture, while keeping the runtime code expressed in terms of row-major tensors and ordinary matrix operations.

PHP2xAI stores tensor data in row-major order. This permits contiguous two-dimensional tensors to be exposed to Eigen through `Eigen::Map`, avoiding an extra copy before a supported operation is evaluated.

## Current state

The C++ runtime has an Eigen implementation for the contiguous `MATMUL_2D_2D` kernel and its backward counterpart. When Eigen is enabled, the dispatcher selects:

```text
MATMUL_2D_2D           -> MATMUL_2D_2D_EIGEN
BACKWARD_MATMUL_2D_2D  -> BACKWARD_MATMUL_2D_2D_EIGEN
```

The generic and specialized kernels remain available as the semantic reference and for layouts that cannot safely be mapped as the required contiguous Eigen matrix. This is intentional: correctness for every supported shape and stride configuration takes precedence over forcing an operation through a matrix-only path.

## Build variants

The runtime is controlled by the `PHP2XAI_USE_EIGEN` preprocessor macro:

```text
PHP2XAI_USE_EIGEN=1  enable Eigen kernels where available
PHP2XAI_USE_EIGEN=0  use the scalar/runtime kernels only
```

The project provides separate native and Eigen build targets, including PHP shared libraries and standalone C++ runtime binaries. The PHP model configuration can select the Eigen provider, which loads the corresponding Eigen runtime artifact.

Optimization flags such as `-O3`, `-DNDEBUG`, and `-march=native` are build choices rather than graph properties. In particular, `-march=native` may enable instructions available on the build machine; binaries built with it are not necessarily portable to older CPUs.

## Migration strategy

Each operation is migrated one kernel at a time:

1. Preserve the existing Tensor and graph operation contract.
2. Identify the contiguous, high-volume shape path.
3. Add an Eigen implementation for forward and backward.
4. Keep the existing scalar or stride-aware kernel as fallback.
5. Compare forward values and gradients against the PHP runtime and small numerical-gradient tests.
6. Benchmark realistic batch and sequence shapes before making the Eigen path the preferred dispatch target.

Priority is given to matrix multiplication, batched matrix multiplication, reductions and normalization, and common elementwise activation paths. Operations with arbitrary strides or complex broadcasting continue to need dedicated generic kernels even after common contiguous paths gain Eigen acceleration.

## SIMD is an optimization, not a semantic change

SIMD and Eigen may alter evaluation order and therefore produce small floating-point differences from scalar code. Tests should use an appropriate tolerance, while requiring identical output shapes, compatible gradients, and the same graph behavior. The PHP runtime remains useful as a readable reference implementation during this migration.
