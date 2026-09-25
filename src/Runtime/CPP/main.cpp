#include <exception>
#include <iostream>
#include <string>

#include "Core/Core.hpp"

// Command-line entry point for the single C++ runtime.
// The graph and its optional training configuration come from configPath.
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
		PHP2xAI::Runtime::CPP::Core model(configPath);
		model.train();
	}
	catch (const std::exception &error)
	{
		std::cerr << "Error: " << error.what() << "\n";
		return 1;
	}

	return 0;
}
