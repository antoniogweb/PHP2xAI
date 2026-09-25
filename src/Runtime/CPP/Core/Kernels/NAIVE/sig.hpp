#pragma once

#include <cmath>
#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void SIG_TEMPLATE(const T *input, T *output, std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
		{
			const Scalar value = static_cast<Scalar>(input[i]);
			output[i] = static_cast<T>(1.0f / (1.0f + std::exp(-value)));
		}
	}

	template <typename T>
	void BACKWARD_SIG_TEMPLATE(const T *output, T *inputGrad,
		const T *outputGrad, std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
		{
			const Scalar value = static_cast<Scalar>(output[i]);
			inputGrad[i] = static_cast<T>(static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * value * (1.0f - value));
		}
	}
}
