# Makefile per PHP2xAI C++ Runtime
# Compiles binaries into Bin/<platform>

CXX := g++
OMPFLAGS := -fopenmp
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe $(OMPFLAGS)
SHFLAGS := -std=c++17 -O3 -fPIC -shared $(OMPFLAGS)
CONAN_DIR := $(CURDIR)/build/conan
INCLUDES := -I./ $(shell PKG_CONFIG_PATH=$(CONAN_DIR) pkg-config --cflags nlohmann_json hdf5)
LIBS_HDF5 := $(shell PKG_CONFIG_PATH=$(CONAN_DIR) pkg-config --libs hdf5)

ARCH ?= $(shell uname -m)
PLATFORM := linux-$(ARCH)
BINDIR := Bin/$(PLATFORM)

CPP_SOURCES := \
	Core/Core.cpp \
	Core/runtime.cpp \
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

BIN_NATIVE := php2xai_runtime
SO := php2xai_runtime.so
SO_HDF5 := php2xai_hdf5.so

all: $(BINDIR)/$(BIN_NATIVE) $(BINDIR)/$(SO) $(BINDIR)/$(SO_HDF5)

$(BINDIR):
	mkdir -p $(BINDIR)

$(BINDIR)/$(BIN_NATIVE): $(NATIVE_SOURCES) main.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBS_HDF5)

# Shared library exposing the single graph runtime through FFI.
$(BINDIR)/$(SO): $(FFI_SOURCES) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) \
		$^ -o $@ $(LIBS_HDF5)

# PHP2XAIHDF5 shared library to be used by means of FFI
$(BINDIR)/$(SO_HDF5): $(FFI_SOURCES_HDF5) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) \
		$^ -o $@ $(LIBS_HDF5)

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
