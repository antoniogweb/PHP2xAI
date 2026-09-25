#pragma once

#include <cstddef>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T, typename M>
	void APPLY_PADDING_MASK_TEMPLATE(const T *scores, const M *mask, T *output,
		std::size_t outer, int batchRows, int rowSize)
	{
		const Scalar negativeInfinity = -std::numeric_limits<Scalar>::infinity();
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t batch = row / static_cast<std::size_t>(batchRows);
			for (int column = 0; column < rowSize; ++column)
			{
				const std::size_t index = row * static_cast<std::size_t>(rowSize) + column;
				output[index] = mask[batch * static_cast<std::size_t>(rowSize) + column]
					== static_cast<M>(0) ? static_cast<T>(negativeInfinity) : scores[index];
			}
		}
	}

	template <typename T, typename M>
	void BACKWARD_APPLY_PADDING_MASK_TEMPLATE(const M *mask, T *inputGrad,
		const T *outputGrad, std::size_t outer, int batchRows, int rowSize)
	{
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t batch = row / static_cast<std::size_t>(batchRows);
			for (int column = 0; column < rowSize; ++column)
			{
				const std::size_t index = row * static_cast<std::size_t>(rowSize) + column;
				if (mask[batch * static_cast<std::size_t>(rowSize) + column] != static_cast<M>(0))
					inputGrad[index] = static_cast<T>(inputGrad[index] + outputGrad[index]);
			}
		}
	}
}
