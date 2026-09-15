#pragma once

#include "Profiler.hpp"

#include <cstddef>
#include <string>

namespace PHP2xAI::Runtime::CPP
{
	class ProfileWriter
	{
	public:
		static void appendBatch(const Profiler &profiler, const std::string &filename, std::size_t batchIndex);
	};
}
