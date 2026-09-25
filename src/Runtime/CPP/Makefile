# Makefile per PHP2xAI C++ Runtime
# Compiles binaries into Bin/<platform>

CXX := g++
OMPFLAGS := -fopenmp
CXXFLAGS := -std=c++20 -O3 -DNDEBUG -march=native -flto -pipe $(OMPFLAGS)
SHFLAGS := -std=c++20 -O3 -fPIC -shared $(OMPFLAGS)
CONAN_DIR := $(CURDIR)/build/conan
INCLUDES := -I./ $(shell PKG_CONFIG_PATH=$(CONAN_DIR) pkg-config --cflags nlohmann_json hdf5 eigen3)
LIBS_HDF5 := $(shell PKG_CONFIG_PATH=$(CONAN_DIR) pkg-config --libs hdf5)

ARCH ?= $(shell uname -m)
PLATFORM := linux-$(ARCH)
BINDIR := Bin/$(PLATFORM)

CPP_SOURCES := \
	Core/Core.cpp \
	Core/runtime.cpp \
	Core/Kernels/NAIVE/add.cpp \
	Core/Kernels/NAIVE/matmul.cpp \
	Core/Kernels/NAIVE/relu.cpp \
	Core/Kernels/NAIVE/gelu.cpp \
	Core/Kernels/NAIVE/silu.cpp \
	Core/Kernels/NAIVE/ce_label_input.cpp \
	Core/Kernels/NAIVE/mean.cpp \
	Core/Kernels/NAIVE/softmax.cpp \
	Core/Kernels/NAIVE/embeddings.cpp \
	Core/Kernels/NAIVE/embeddings_mean_pooling.cpp \
	Core/Kernels/NAIVE/mean_pooling.cpp \
	Core/Kernels/NAIVE/padding_mask.cpp \
	Core/Kernels/NAIVE/dropout.cpp \
	Core/Kernels/NAIVE/multiply.cpp \
	Core/Kernels/NAIVE/scale.cpp \
	Core/Kernels/NAIVE/sig.cpp \
	Core/Kernels/NAIVE/reshape.cpp \
	Core/Kernels/NAIVE/transpose.cpp \
	Core/Kernels/NAIVE/slice.cpp \
	Core/Kernels/NAIVE/positional_encoding.cpp \
	Core/Kernels/NAIVE/apply_padding_mask.cpp \
	Core/Kernels/NAIVE/apply_causal_mask.cpp \
	Core/Kernels/NAIVE/layer_norm.cpp \
	Core/Kernels/NAIVE/rms_norm.cpp \
	Core/Kernels/NAIVE/ce.cpp \
	Core/Kernels/NAIVE/ce_logits.cpp \
	Core/Kernels/NAIVE/rope.cpp \
	Core/Kernels/EIGEN/matmul.cpp \
	Core/Kernels/EIGEN/gelu.cpp \
	Core/Kernels/EIGEN/softmax.cpp \
	Dataset/TrainValidateDataset.cpp \
	Dataset/stream_file_dataset.cpp \
	Optimizers/Optimizer.cpp \
	Optimizers/Adam.cpp \
	Optimizers/Fixed.cpp \
	Utility/Utility.cpp \
	Utility/Profiler.cpp \
	Utility/ProfileWriter.cpp

CPP_HDF5_SOURCES := Dataset/PHP2XAIHDF5.cpp Dataset/HDF5Dataset.cpp
NATIVE_SOURCES := $(CPP_SOURCES) $(CPP_HDF5_SOURCES)
FFI_SOURCES := $(CPP_SOURCES) Core/ffi.cpp $(CPP_HDF5_SOURCES) Dataset/ffi.cpp
FFI_SOURCES_HDF5 := Dataset/PHP2XAIHDF5.cpp Dataset/ffi.cpp
CPP_HEADERS := $(shell find Core Dataset Optimizers Utility -type f \( -name '*.hpp' -o -name '*.h' \))
DATASET_HEADERS := $(shell find Dataset -type f \( -name '*.hpp' -o -name '*.h' \))

BIN_NATIVE := php2xai_runtime
BIN_EIGEN := php2xai_runtime_eigen
SO := php2xai_runtime.so
SO_HDF5 := php2xai_hdf5.so

all: $(BINDIR)/$(BIN_NATIVE) $(BINDIR)/$(BIN_EIGEN) $(BINDIR)/$(SO) $(BINDIR)/$(SO_HDF5)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/$(BIN_NATIVE): $(NATIVE_SOURCES) main.cpp $(CPP_HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -DPHP2XAI_USE_EIGEN=0 $(INCLUDES) $(NATIVE_SOURCES) main.cpp -o $@ $(LIBS_HDF5)

$(BINDIR)/$(BIN_EIGEN): $(NATIVE_SOURCES) main.cpp $(CPP_HEADERS) | $(BINDIR)
	$(CXX) $(CXXFLAGS) -DPHP2XAI_USE_EIGEN=1 $(INCLUDES) $(NATIVE_SOURCES) main.cpp -o $@ $(LIBS_HDF5)

# Shared library exposing the single graph runtime through FFI.
$(BINDIR)/$(SO): $(FFI_SOURCES) $(CPP_HEADERS) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) \
		$(FFI_SOURCES) -o $@ $(LIBS_HDF5)

# PHP2XAIHDF5 shared library to be used by means of FFI
$(BINDIR)/$(SO_HDF5): $(FFI_SOURCES_HDF5) $(DATASET_HEADERS) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) \
		$(FFI_SOURCES_HDF5) -o $@ $(LIBS_HDF5)

clean:
	rm -rf Bin/

clean-bin:
	rm -f Bin/*/php2xai_runtime*
	@echo "Binaries cleaned"

info:
	@echo "Platform: $(PLATFORM)"
	@echo "Output dir: $(BINDIR)"
	@echo "Compiler: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SHFLAGS: $(SHFLAGS)"

install: all
	@echo "Installation not implemented. Copy Bin/$(PLATFORM)/* to your system path."

.PHONY: all clean clean-bin info install
