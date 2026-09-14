# Makefile per PHP2xAI C++ Runtime
# Compila i binari C++ nella cartella Bin/<platform> corretta

CXX := g++
OMPFLAGS := -fopenmp
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe $(OMPFLAGS)
SHFLAGS := -std=c++17 -O3 -fPIC -shared $(OMPFLAGS)
INCLUDES := -I./ -I./ThirdParty/nlohmann -I./ThirdParty/eigen

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
	Utility/Utility.cpp

FFI_SOURCES := $(CPP_SOURCES) Core/ffi.cpp

BIN_NATIVE := php2xai_runtime
BIN_EIGEN := php2xai_runtime_eigen
SO := php2xai_runtime.so

all: $(BINDIR)/$(BIN_NATIVE) $(BINDIR)/$(BIN_EIGEN) $(BINDIR)/$(SO)

$(BINDIR):
	mkdir -p $(BINDIR)

# Compilazione eseguibile NAIVE
$(BINDIR)/$(BIN_NATIVE): $(CPP_SOURCES) main.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN=0 \
		$^ -o $@

# Compilazione eseguibile EIGEN
$(BINDIR)/$(BIN_EIGEN): $(CPP_SOURCES) main.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN=1 \
		$^ -o $@

# Shared library unica: provider scelto a runtime via FFI
$(BINDIR)/$(SO): $(FFI_SOURCES) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) \
		$^ -o $@

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
