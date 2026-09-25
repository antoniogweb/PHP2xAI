# PHP2xAI C++ Runtime

This directory contains the C++ runtime for PHP2xAI.

## Directory Structure

```
CPP/
├── Makefile              # Build system
├── README_BUILD.md       # This file
├── main.cpp              # Main entry point
├── types.hpp             # Type definitions
├── Core/                 # Core runtime logic
├── Dataset/              # Dataset handling
├── Optimizers/           # Optimization algorithms
├── Utility/              # Utility functions
├── ThirdParty/           # External dependencies
└── Bin/                  # Build output (auto-created)
    └── linux-x86_64/     # Platform-specific binaries
```

## Building

Use the provided Makefile to compile all binaries:

```bash
cd src/Runtime/CPP

# Build all binaries for the current architecture
make all

# Build only the NAIVE runtime
make Bin/linux-x86_64/php2xai_runtime

# Build only the EIGEN runtime
make Bin/linux-x86_64/php2xai_runtime_eigen

# Build only the shared library
make Bin/linux-x86_64/php2xai_runtime.so

# Show build information
make info

# Clean all build artifacts
make clean
```

## Architecture Detection

The Makefile automatically detects the architecture:

- **x86_64 / amd64** → `Bin/linux-x86_64/`
- **aarch64 / arm64** → `Bin/linux-arm64/`

To override the architecture:

```bash
make ARCH=arm64 all
```

## Generated Binaries

| Binary | Description |
|--------|-------------|
| `php2xai_runtime` | NAIVE runtime executable |
| `php2xai_runtime_eigen` | EIGEN runtime executable |
| `php2xai_runtime.so` | Shared library; provider selected at runtime |

## Usage

### PHP Integration

The PHP code automatically locates the correct binary based on the platform:

```php
use PHP2xAI\Utility\Utility;

$platform = Utility::getPlatform(); // Returns "linux-x86_64" or "linux-arm64"
```

### Standalone Execution

```bash
# NAIVE runtime
./Bin/linux-x86_64/php2xai_runtime config.json

# EIGEN runtime
./Bin/linux-x86_64/php2xai_runtime_eigen config.json
```

## Provider Selection

The native training executables are built separately. `Model::setProvider("EIGEN")`
selects `php2xai_runtime_eigen`; the default `NAIVE` provider selects
`php2xai_runtime`. Prediction through `CoreFFI` and `GraphRuntimeCpp` passes the
provider to the shared library at runtime. The provider is not stored in the graph.

`GraphRuntimeEigen` currently inherits all behavior from `GraphRuntime`. Its
virtual kernel entry points are ready for Eigen overrides to be added incrementally.

## Build Options

The Makefile uses these compiler flags:

- **C++ Standard**: C++20
- **Optimization**: `-O3 -march=native -flto`
- **Debug symbols disabled**: `-DNDEBUG`
- **Shared flags**: `-fPIC -shared` for .so files

## Requirements

- g++ (C++17 compatible)
- make
- Conan packages configured in `build/conan` (nlohmann_json and HDF5)

## Cleaning

```bash
make clean              # Remove entire Bin/ directory
make clean-bin          # Remove only binaries (keep directory)
```

## Troubleshooting

### LTO Warnings

Warnings like `lto-wrapper: warning: using serial compilation` are normal and can be ignored. They indicate LTO is working but cannot parallelize the link phase.

### Missing Headers

Ensure all ThirdParty dependencies are present:
- `ThirdParty/eigen/`
- `ThirdParty/nlohmann/`

### Architecture Mismatch

If you get "Illegal instruction" errors, ensure the compiled binary matches your CPU. Use `make info` to verify the detected architecture.
