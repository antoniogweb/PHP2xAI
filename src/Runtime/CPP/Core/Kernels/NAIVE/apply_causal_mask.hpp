#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void APPLY_CAUSAL_MASK_TEMPLATE(const T *input, T *output,
		std::size_t outer, int queryLength, int keyLength)
	{
		const int offset = keyLength - queryLength;
		const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();

		if (queryLength == 1)
		{
			const std::size_t total = outer * static_cast<std::size_t>(keyLength);
			#pragma omp parallel for schedule(static)
			for (std::int64_t index = 0;
				index < static_cast<std::int64_t>(total); ++index)
				output[index] = input[index];
			return;
		}

		const std::size_t rowCount = outer * static_cast<std::size_t>(queryLength);
		#pragma omp parallel for schedule(static)
		for (std::int64_t rowIndex = 0;
			rowIndex < static_cast<std::int64_t>(rowCount); ++rowIndex)
		{
			const std::size_t rowNumber = static_cast<std::size_t>(rowIndex);
			const int query = static_cast<int>(rowNumber % queryLength);
			const std::size_t row = rowNumber * static_cast<std::size_t>(keyLength);
			const int firstMasked = offset + query + 1;
			for (int key = 0; key < keyLength; ++key)
			{
				const std::size_t index = row + static_cast<std::size_t>(key);
				output[index] = key < firstMasked
					? input[index]
					: static_cast<T>(negativeInfinity);
			}
		}
	}

	template <typename T>
	void BACKWARD_APPLY_CAUSAL_MASK_TEMPLATE(T *inputGrad, const T *outputGrad,
		std::size_t outer, int queryLength, int keyLength)
	{
		const int offset = keyLength - queryLength;
		const std::size_t rowCount = outer * static_cast<std::size_t>(queryLength);
		#pragma omp parallel for schedule(static)
		for (std::int64_t rowIndex = 0;
			rowIndex < static_cast<std::int64_t>(rowCount); ++rowIndex)
		{
			const std::size_t rowNumber = static_cast<std::size_t>(rowIndex);
			const int query = static_cast<int>(rowNumber % queryLength);
			const std::size_t row = rowNumber * static_cast<std::size_t>(keyLength);
			const int firstMasked = offset + query + 1;
			for (int key = 0; key < firstMasked; ++key)
			{
				const std::size_t index = row + static_cast<std::size_t>(key);
				inputGrad[index] = static_cast<T>(inputGrad[index] + outputGrad[index]);
			}
		}
	}
}
