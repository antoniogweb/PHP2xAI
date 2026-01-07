#include "Utility.hpp"

namespace PHP2xAI::Runtime::CPP
{
	int Utility::argmax(const std::vector<Scalar> &values)
	{
		if (values.empty())
			return 0;

		int maxIndex = 0;
		Scalar maxValue = values[0];

		for (std::size_t i = 1; i < values.size(); ++i)
		{
			if (values[i] > maxValue)
			{
				maxValue = values[i];
				maxIndex = static_cast<int>(i);
			}
		}

		return maxIndex;
	}
}
