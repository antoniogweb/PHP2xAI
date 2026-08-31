# 12. Extending the runtime

Adding a new operation requires keeping the frontend and both runtimes aligned.

## Procedure

1. Add the method to `Tensor.php`.
2. Register the operation in `GraphContext`.
3. Define serialized attributes.
4. Implement the PHP forward operation.
5. Implement the PHP backward operation.
6. Add the attribute structure to `runtime.hpp` when needed.
7. Read the attributes in `loadOps()`.
8. Implement forward and backward in `runtime.cpp`.
9. Verify shapes with a minimal PHP and C++ test.

## Fixed attributes

A value such as dropout percentage, axis, or padding is not a model parameter. It should be serialized inside the operation's `attributes`.

## Minimal test

A good test should compare:

- output shape;
- forward values on simple inputs;
- gradients on unambiguous inputs;
- behavior with `training = true` and `false`;
- errors produced by incompatible shapes.
