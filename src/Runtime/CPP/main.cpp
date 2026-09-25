#include <exception>
#include <iostream>
#include <string>

#include "Core/Core.hpp"

#ifndef PHP2XAI_USE_EIGEN
#define PHP2XAI_USE_EIGEN 0
#endif

// Command-line entry point. The build flag chooses the runtime provider;
// the graph file itself stays independent of the provider.
int main(int argc, char **argv)
{
	if (argc < 2)
	{
		const char *programName = argc > 0 ? argv[0] : "php2xai_runtime";
		std::cerr << "Usage: " << programName << " <config.json>\n";
		return 1;
	}

	try
	{
		const std::string configPath = argv[1];
		const std::string provider = PHP2XAI_USE_EIGEN ? "EIGEN" : "NAIVE";
		PHP2xAI::Runtime::CPP::Core model(provider, configPath);
		model.train();
	}
	catch (const std::exception &error)
	{
		std::cerr << "Error: " << error.what() << "\n";
		return 1;
	}

	return 0;
}
