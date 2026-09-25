#pragma once

#include <cstddef>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void APPLY_CAUSAL_MASK_TEMPLATE(const T *input, T *output,
		std::size_t outer, int queryLength, int keyLength)
	{
		const std::size_t total = outer * static_cast<std::size_t>(queryLength)
			* static_cast<std::size_t>(keyLength);
		for (std::size_t i = 0; i < total; ++i)
			output[i] = input[i];

		const int offset = keyLength - queryLength;
		const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
		for (std::size_t batch = 0; batch < outer; ++batch)
		{
			for (int query = 0; query < queryLength; ++query)
			{
				const int firstMasked = offset + query + 1;
				const std::size_t row = (batch * queryLength + query) * keyLength;
				for (int key = firstMasked; key < keyLength; ++key)
					output[row + static_cast<std::size_t>(key)] = static_cast<T>(negativeInfinity);
			}
		}
	}

	template <typename T>
	void BACKWARD_APPLY_CAUSAL_MASK_TEMPLATE(T *inputGrad, const T *outputGrad,
		std::size_t outer, int queryLength, int keyLength)
	{
		const int offset = keyLength - queryLength;
		for (std::size_t batch = 0; batch < outer; ++batch)
		{
			for (int query = 0; query < queryLength; ++query)
			{
				const int firstMasked = offset + query + 1;
				const std::size_t row = (batch * queryLength + query) * keyLength;
				for (int key = 0; key < firstMasked; ++key)
				{
					const std::size_t index = row + static_cast<std::size_t>(key);
					inputGrad[index] = static_cast<T>(inputGrad[index] + outputGrad[index]);
				}
			}
		}
	}
}
