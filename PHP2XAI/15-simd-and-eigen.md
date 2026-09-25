# 15. SIMD and Eigen migration

This chapter describes the current C++ execution providers. `EIGEN` is a CPU provider: it uses Eigen row-major maps for supported dense operations and lets Eigen and the compiler use the SIMD instruction set available on the build machine. It is not a GPU provider. The C++ build requires C++20 because dtype dispatch uses a templated lambda.

The PHP graph format and Tensor API are shared by the PHP runtime, the native C++ runtime, and the Eigen C++ runtime. Selecting Eigen changes only the C++ kernel implementation when an Eigen override exists.

## Providers and build configuration

The C++ runtime has two providers:

```text
NAIVE  GraphRuntime: typed C++ kernels and general-shape paths
EIGEN  GraphRuntimeEigen: Eigen overrides plus every general C++ kernel
```

The standalone binaries are built with `PHP2XAI_USE_EIGEN=0` or `1`; the compile-time flag chooses the provider for that binary. The shared FFI library selects `NAIVE` or `EIGEN` at runtime. In PHP, `GraphRuntimeCpp` selects `NAIVE` and `GraphRuntimeEigen` selects `EIGEN`; the C++ `Core` receives the provider when constructed. The provider is not written in the graph, so the same graph and weights can be used by either provider.

The standard build uses:

```text
-O3 -DNDEBUG -march=native -flto -fopenmp
```

`-march=native` can enable CPU-specific SIMD instructions such as AVX2 where the build CPU supports them. Binaries built this way are not automatically portable to older or different CPUs. OpenMP controls CPU parallel loops; configure its thread count with `OMP_NUM_THREADS` when needed.

## Eigen provider overrides

`GraphRuntimeEigen` currently overrides the following kernels. All listed matmul paths have Eigen forward and backward implementations; other kernel entry points are inherited from `GraphRuntime` and use NAIVE unless overridden.

| Graph kernel | Tensor layout | Implementation |
|---|---|---|
| `MATMUL_2D_2D` | `[M, K] x [K, N]` | One row-major `Eigen::Map` GEMM. |
| `MATMUL_1B_2D_2D` | `[B, T, K] x [B, K, N]` | One GEMM per batch item; batch items run with OpenMP. |
| `MATMUL_2B_2D_2D` | `[B, H, T, K] x [B, H, K, N]` | One GEMM per `[B, H]` matrix; matrices run with OpenMP. |
| `MATMUL_1B_2D_2D_LINEAR` | `[B, T, K] x [K, N]` | Flattens the first two axes to one `[B*T, K]` GEMM. |
| `GELU` | Any contiguous tensor | Eigen array expression for the tanh GELU approximation. The backward materializes only the tanh term and fuses the remaining expression. |
| `SOFTMAX_4D_LAST` | `[B, H, T, D]`, last axis | Dedicated row-wise stable softmax. Each worker performs max, exponential sum, and normalization for independent rows without auxiliary max or sum tensors. |

The matmul implementations use row-major `Eigen::Map` and `noalias()` assignments or accumulations. This avoids temporary result matrices for the normal dense paths.

The 4D softmax is an optimized Eigen-provider override, but its inner loop is explicit C++ plus OpenMP rather than an Eigen reduction expression. It remains in the Eigen provider because it is the preferred optimized path for attention scores.

## Common C++ SIMD and OpenMP paths

Some optimizations are implemented in `GraphRuntime` itself. They are therefore available to both `NAIVE` and `EIGEN`; they are not Eigen-specific.

| Operation | Available path |
|---|---|
| `dropout` forward | OpenMP element loop with deterministic splitmix random values. Dropout is active only in `ExecutionMode::TRAIN`. |
| `apply_padding_mask` forward and backward | OpenMP over contiguous score rows. Forward writes negative infinity for masked keys; backward propagates only unmasked gradients. |
| `apply_causal_mask` forward and backward | OpenMP over outer `[Lq, Lkv]` matrices. The operation derives `Lq` and `Lkv` from the last two runtime dimensions. For `Lq == 1`, forward is a direct copy because a single decode query has no future key. |
| `SLICE_LAST` and `BACKWARD_SLICE_LAST` | OpenMP copies or accumulates contiguous last-axis slices. |
| `TRANSPOSE_4D_AXIS_1_2` and backward | OpenMP tiles over heads and time, copying each contiguous feature vector as a block for `[B,H,T,D]` to `[B,T,H,D]`. |
| `TRANSPOSE_4D_LAST_TWO` and backward | OpenMP 32×32 tiles transpose each `[T,D]` matrix in `[B,H,T,D]`; backward uses the same mapping to accumulate gradients. |

These paths are memory-bandwidth sensitive. They benefit from contiguous tensors and parallel rows, but they are not substitutes for a dense GEMM backend.

## Typed kernel dispatch and fallbacks

The Tensor API selects a graph kernel from known ranks and axis patterns. The C++ NAIVE kernel entry point dispatches from `Tensor.dtype` to the corresponding C++ template specialization. The current dtype set is `FLOAT32`, `FLOAT64`, `INT32`, and `INT64`; mixed index/data kernels select each input type independently. The Eigen provider overrides selected virtual entry points and uses the same dtype dispatch where those kernels support the type.

NAIVE generic implementations cover add and broadcast, generic matmul, transpose, slice, reductions, softmax, layer and RMS normalization, RoPE, and the three cross-entropy families. They handle generic axes/layouts for the named operation paths. These implementations remain available when the Eigen provider is selected, but do not become Eigen expressions simply because Eigen is enabled. Embeddings, masks, pooling, dropout, and most elementwise operations also use NAIVE kernels unless listed in the Eigen override table.

Correctness takes priority over forcing every operation through Eigen. The optimized paths require the layouts they were written for; the general runtime remains the semantic reference for other valid graph shapes.

## Attention-specific status

BERT and Decoder attention benefit from the Eigen batched matmul kernels, the 4D last-axis softmax path, and the common padding or causal mask paths.

The causal mask is shape-driven:

```text
scores shape: [B, H, Lq, Lkv]
Lq          : scores[-2]
Lkv         : scores[-1]
```

This lets the runtime use the actual dimensions in prefill and decode, rather than relying on static graph attributes.

KV cache is implemented in the C++ runtime as a stateful multi-output operation. In `PREFILL`, it initializes the per-layer key/value cache; in `DECODE`, it appends a step and returns the accumulated cache. `resetKvCache()` resets all layers or a selected layer. `TRAIN` and ordinary stateless `INFER` simply pass the supplied key and value through. KV cache storage follows the tensor dtype. It is currently a NAIVE runtime feature and does not have an Eigen-specific override.

The optimizer is still based on `Scalar` (`float`): Adam's moments and its elementwise parameter update use scalar vectors/accessors. Typed kernel execution therefore does not yet imply typed optimizer arithmetic or end-to-end `FLOAT64` precision during training. Eigen half and bfloat16 support would also require adding those dtypes to graph serialization, storage, kernels, and optimizer dispatch.

## Profiling and expectations

The optional C++ profiler reports time by operation and kernel. Use it to identify the hot shape before optimizing. For Transformer training, `MATMUL_1B_2D_2D_LINEAR.backward`, batched attention matmuls, normalization, and memory traffic around attention commonly dominate.

Eigen improves the supported CPU dense paths, but it does not eliminate the structural cost of Transformer training. The remaining limits are dense matrix multiplication, memory reads and writes, and the CPU core and cache budget. Larger gains beyond this provider generally require more graph fusion, a specialized CPU backend such as oneDNN or MKL, reduced precision where numerically valid, or a GPU backend.

## Verification rule for future migrations

For every new optimized path:

1. Preserve the graph and Tensor contract.
2. Compare forward values with the PHP runtime.
3. Compare backward gradients, including masked and broadcast cases.
4. Retain a general fallback when the optimized layout assumptions do not hold.
5. Measure realistic batch, sequence, head, and hidden dimensions with the C++ profiler.

This keeps provider selection an implementation choice rather than a change in model semantics.
