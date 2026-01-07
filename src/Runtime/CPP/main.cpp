#include <exception>
#include <iostream>
#include <string>
#include "Core/runtime.hpp"
#include "Core/Core.hpp"

// g++ -std=c++17 -I./ -I./ThirdParty Utility/Utility.cpp Core/Core.cpp Core/runtime.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp Optimizers/Fixed.cpp main.cpp -o php2xai_runtime
// g++ -std=c++17 -O3 -DNDEBUG -march=native -flto -pipe -I./ -I./ThirdPartyUtility/Utility.cpp Core/Core.cpp Core/runtime.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp Optimizers/Fixed.cpp main.cpp -o php2xai_runtime

//.so
// g++ -std=c++17 -O3 -fPIC -shared -I./ -I./ThirdParty Utility/Utility.cpp Core/Core.cpp Core/runtime.cpp Core/ffi.cpp Dataset/TrainValidateDataset.cpp Dataset/stream_file_dataset.cpp Optimizers/Optimizer.cpp Optimizers/Adam.cpp  Optimizers/Fixed.cpp -o php2xai_runtime.so


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
		PHP2xAI::Runtime::CPP::Core model(configPath);
		model.train();
	}
	catch (const std::exception &ex)
	{
		std::cerr << "Error: " << ex.what() << "\n";
		return 1;
	}

	return 0;
}
