#pragma once

#include <cmath>
#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void SILU_TEMPLATE(const T *input, T *output, std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar x = static_cast<Scalar>(input[i]);
			const Scalar sigmoid = 1.0f / (1.0f + std::exp(-x));
			output[i] = static_cast<T>(x * sigmoid);
		}
	}

	template <typename T>
	void BACKWARD_SILU_TEMPLATE(
		const T *input,
		T *inputGrad,
		const T *outputGrad,
		std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar x = static_cast<Scalar>(input[i]);
			const Scalar sigmoid = 1.0f / (1.0f + std::exp(-x));
			const Scalar localGrad = sigmoid * (1.0f + x * (1.0f - sigmoid));
			const Scalar grad = static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * localGrad;
			inputGrad[i] = static_cast<T>(grad);
		}
	}
}
