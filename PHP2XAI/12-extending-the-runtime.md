# 12. Extending the runtime

Adding an operation requires keeping the PHP graph contract, PHP runtime, and C++ runtime aligned. The native implementation is split between the graph dispatcher and per-operation kernel files; operation logic should not accumulate in `runtime.cpp`.

## Procedure

1. Add the public method to `Tensor.php` and register the op and its tensor dtypes in `GraphContext`.
2. Define and serialize fixed attributes such as axes, kernel name, epsilon, or dropout rate.
3. Implement PHP forward and backward behavior when the PHP runtime supports the op.
4. Add the operation dispatch methods and any backend-overridable kernel declarations to `Core/runtime.hpp`.
5. Parse op attributes and implement the forward and reverse-order backward dispatch in `Core/runtime.cpp`.
6. Put the NAIVE kernel declarations and dtype-templated implementations in `Core/Kernels/NAIVE/<op>.hpp`; put validation, dtype dispatch, and kernel entry-point definitions in `Core/Kernels/NAIVE/<op>.cpp`.
7. Add the files to the native build. Keep tensor access typed through `dataAs<T>()` and `gradAs<T>()`; do not assume every graph tensor is `Scalar`.
8. If an Eigen implementation is appropriate, add an override under `Core/Kernels/EIGEN/` and mark the corresponding base kernel virtual. Keep NAIVE as the fallback for shapes without an Eigen override.
9. Verify forward values, backward gradients, supported dtypes, axes and shape errors with the relevant exercises.

## Fixed attributes

A value such as dropout percentage, axis, epsilon, or padding ID is not a model parameter. It should be serialized inside the operation's `attributes`. A tensor's dtype belongs to the tensor definition, not to the operation attributes.

## Kernel and dtype dispatch

The C++ graph dispatcher chooses a named kernel from the graph's kernel field and shape metadata. A kernel receives tensor references. The NAIVE kernel checks compatible shapes and dtypes, dispatches with `dispatchDType`, then invokes a matching function template from the operation header. Mixed index/data operations dispatch each dtype separately. The generic NAIVE path belongs in that operation's existing `.cpp` and `.hpp`; avoid a second generic-kernel directory or placing implementation bodies in the central runtime file.

`Scalar` remains the external data and accumulation type in parts of the current interface, but it is not a substitute for the tensor's storage type. Keep input/output storage accesses templated on the dispatched dtype. C++ graph dtypes currently include `FLOAT32`, `FLOAT64`, `INT32`, and `INT64`; Eigen half and bfloat16 are future work.

## Minimal test

A good test should compare:

- output shape;
- forward values on simple inputs;
- gradients on unambiguous inputs;
- behavior with `training = true` and `false`;
- errors produced by incompatible shapes.
