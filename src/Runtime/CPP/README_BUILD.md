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

# Build only the native runtime
make Bin/linux-x86_64/php2xai_runtime

# Build only the Eigen runtime
make Bin/linux-x86_64/php2xai_runtime_eigen.so

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

| Binary | Description | USE_EIGEN |
|--------|-------------|-----------|
| `php2xai_runtime` | Native runtime executable | No |
| `php2xai_runtime.so` | Native shared library | No |
| `php2xai_runtime_eigen` | Eigen-powered executable | Yes |
| `php2xai_runtime_eigen.so` | Eigen-powered shared library | Yes |

## Usage

### PHP Integration

The PHP code automatically locates the correct binary based on the platform:

```php
use PHP2xAI\Utility\Utility;

$platform = Utility::getPlatform(); // Returns "linux-x86_64" or "linux-arm64"
```

### Standalone Execution

```bash
# Native runtime
./Bin/linux-x86_64/php2xai_runtime config.json

# Eigen runtime
./Bin/linux-x86_64/php2xai_runtime_eigen config.json
```

## Build Options

The Makefile uses these compiler flags:

- **C++ Standard**: C++17
- **Optimization**: `-O3 -march=native -flto`
- **Debug symbols disabled**: `-DNDEBUG`
- **Shared flags**: `-fPIC -shared` for .so files

## Requirements

- g++ (C++17 compatible)
- make
- Eigen 3 (bundled in ThirdParty/)

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
