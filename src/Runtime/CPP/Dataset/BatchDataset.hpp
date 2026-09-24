#pragma once

#include <string>
#include <vector>

namespace PHP2xAI::Runtime::CPP
{
	class BatchDataset
	{
	public:
		virtual ~BatchDataset() = default;

		virtual std::string getType() const = 0;
		virtual void shuffleEpoch() = 0;
		virtual void resetEpoch() = 0;
		virtual bool nextBatch() = 0;
		virtual void pack(std::vector<float>& xPacked, std::vector<float>& yPacked) = 0;
	};
}
