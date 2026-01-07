#pragma once

#include <vector>
#include "../types.hpp"

namespace PHP2xAI::Runtime::CPP
{
	class Utility
	{
	public:
		static int argmax(const std::vector<Scalar> &values);
	};
}
