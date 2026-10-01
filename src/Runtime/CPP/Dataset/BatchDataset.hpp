#pragma once

#include <string>
#include <vector>

#include "../types.hpp"

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
		template <typename X, typename Y>
		void pack(std::vector<X>& xPacked, std::vector<Y>& yPacked)
		{
			packRaw(&xPacked, &yPacked);
		}

	protected:
		virtual void packRaw(void* xVector, void* yVector) = 0;
	};
}
