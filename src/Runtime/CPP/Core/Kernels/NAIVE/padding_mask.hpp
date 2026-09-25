#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename TIndex, typename TMask>
	void PADDING_MASK_TEMPLATE(const TIndex *ids, TMask *mask, std::size_t elementCount, int padId)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const long double tokenValue = static_cast<long double>(ids[i]);
			if (!std::isfinite(tokenValue) || std::floor(tokenValue) != tokenValue
				|| tokenValue < static_cast<long double>(std::numeric_limits<int>::min())
				|| tokenValue > static_cast<long double>(std::numeric_limits<int>::max()))
				throw std::runtime_error("padding_mask: token ID must be an integer");

			const int tokenId = static_cast<int>(tokenValue);
			mask[i] = static_cast<TMask>(tokenId == padId ? 0 : 1);
		}
	}
}
