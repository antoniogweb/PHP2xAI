#pragma once

#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void SCALE_TEMPLATE(const T *input, T *output, std::size_t count, Scalar scale)
	{
		for (std::size_t i = 0; i < count; ++i)
			output[i] = static_cast<T>(static_cast<Scalar>(input[i]) * scale);
	}

	template <typename T>
	void BACKWARD_SCALE_TEMPLATE(T *inputGrad, const T *outputGrad,
		std::size_t count, Scalar scale)
	{
		for (std::size_t i = 0; i < count; ++i)
			inputGrad[i] = static_cast<T>(static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * scale);
	}
}
