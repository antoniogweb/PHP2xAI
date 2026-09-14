#include <exception>
#include <iostream>
#include <string>
#include "Core/runtime.hpp"
#include "Core/Core.hpp"

// COMPILAZIONE NAIVE
// g++ -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe -DPHP2XAI_USE_EIGEN=0 -I./ -I./ThirdParty/nlohmann -I./ThirdParty/eigen Utility/Utility.cpp Core/Core.cpp Core/runtime.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp Optimizers/Fixed.cpp main.cpp -o Bin/linux-x86_64/php2xai_runtime

// COMPILAZIONE EIGEN
// g++ -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe -DPHP2XAI_USE_EIGEN=1 -I./ -I./ThirdParty/nlohmann -I./ThirdParty/eigen Utility/Utility.cpp Core/Core.cpp Core/runtime.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp Optimizers/Fixed.cpp main.cpp -o Bin/linux-x86_64/php2xai_runtime_eigen

// COMPILAZIONE SHARED LIBRARY (provider scelto a runtime)
// g++ -std=c++17 -O3 -fPIC -shared -I./ -I./ThirdParty/nlohmann -I./ThirdParty/eigen Utility/Utility.cpp Core/Core.cpp Core/runtime.cpp Core/ffi.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp Optimizers/Fixed.cpp -o Bin/linux-x86_64/php2xai_runtime.so

#ifndef PHP2XAI_USE_EIGEN
#define PHP2XAI_USE_EIGEN 0
#endif

// ./php2xai_runtime ../../../Exercises/MNIST/config.json

// Runtime/
// └── CPP/
//     ├── Core/
//     │   ├── runtime.hpp
//     │   └── runtime.cpp
//     │
//     ├── Optimizers/
//     │   ├── Optimizer.hpp     (base class)
//     │   ├── SGD.hpp
//     │   ├── Adam.hpp
//     │   └── Optimizers.hpp    ← aggregatore
//     │
//     ├── Dataset/
//     │   ├── StreamFileDataset.hpp
//     │   └── Dataset.hpp       ← aggregatore
//     │
//     └── php2xai_runtime       ← binary output

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << (argc > 0 ? argv[0] : "php2xai_runtime") << " <config.json>\n";
		return 1;
	}

	try
	{
		std::string configPath = argv[1];
		const std::string provider = PHP2XAI_USE_EIGEN ? "EIGEN" : "NAIVE";
		PHP2xAI::Runtime::CPP::Core model(provider, configPath);
		model.train();
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
