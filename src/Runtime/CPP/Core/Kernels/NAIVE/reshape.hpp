#pragma once

#include <cstddef>

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void RESHAPE_TEMPLATE(const T *input, T *output, std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
			output[i] = input[i];
	}

	template <typename T>
	void BACKWARD_RESHAPE_TEMPLATE(const T *outputGrad, T *inputGrad,
		std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
			inputGrad[i] = static_cast<T>(inputGrad[i] + outputGrad[i]);
	}
}
