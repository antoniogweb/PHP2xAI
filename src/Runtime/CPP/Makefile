# Makefile per PHP2xAI C++ Runtime
# Compila i binari C++ nella cartella Bin/<platform> corretta

CXX := g++
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe
SHFLAGS := -std=c++17 -O3 -fPIC -shared
INCLUDES := -I./ -I./ThirdParty/nlohmann -I./ThirdParty/eigen

# Rileva l'architettura automaticamente se non specificata
ARCH ?= $(shell uname -m | sed 's/x86_64/x86_64/;s/amd64/x86_64/;s/aarch64/arm64/;s/arm64/arm64/')
PLATFORM := linux-$(ARCH)
BINDIR := Bin/$(PLATFORM)

# File sorgenti comuni
CPP_SOURCES := \
	Core/Core.cpp \
	Core/runtime.cpp \
	Dataset/TrainValidateDataset.cpp \
	Dataset/stream_file_dataset.cpp \
	Optimizers/Optimizer.cpp \
	Optimizers/Adam.cpp \
	Optimizers/Fixed.cpp \
	Utility/Utility.cpp

# File sorgenti per la shared library con FFI
FFI_SOURCES := $(CPP_SOURCES) Core/ffi.cpp

# Target: eseguibili
BIN_NATIVE := php2xai_runtime
BIN_EIGEN := php2xai_runtime_eigen

# Target: shared library
SO_NATIVE := php2xai_runtime.so
SO_EIGEN := php2xai_runtime_eigen.so

# Rule di default
all: $(BINDIR)/$(BIN_NATIVE) $(BINDIR)/$(SO_NATIVE) $(BINDIR)/$(BIN_EIGEN) $(BINDIR)/$(SO_EIGEN)

# Crea la directory di output
$(BINDIR):
	mkdir -p $(BINDIR)

# Compilazione eseguibile nativo (senza Eigen)
$(BINDIR)/$(BIN_NATIVE): $(CPP_SOURCES) main.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN=0 \
		$^ -o $@

# Compilazione shared library nativa
$(BINDIR)/$(SO_NATIVE): $(FFI_SOURCES) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN=0 \
		$^ -o $@

# Compilazione eseguibile con Eigen
$(BINDIR)/$(BIN_EIGEN): $(CPP_SOURCES) main.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN \
		$^ -o $@

# Compilazione shared library con Eigen
$(BINDIR)/$(SO_EIGEN): $(FFI_SOURCES) | $(BINDIR)
	$(CXX) $(SHFLAGS) $(INCLUDES) -DPHP2XAI_USE_EIGEN \
		$^ -o $@

# Pulisci i binari generati
clean:
	rm -rf Bin/

# Pulisci solo i binari (mantieni la directory)
clean-bin:
	rm -f Bin/*/php2xai_runtime*
	@echo "Binaries cleaned"

# Mostra informazioni di build
info:
	@echo "Platform: $(PLATFORM)"
	@echo "Output dir: $(BINDIR)"
	@echo "Compiler: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SHFLAGS: $(SHFLAGS)"

# Installa in una posizione globale (opzionale)
install: all
	@echo "Installation not implemented. Copy Bin/$(PLATFORM)/* to your system path."

.PHONY: all clean clean-bin info install
