#pragma once

#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void RELU_TEMPLATE(const T *input, T *output, std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar value = static_cast<Scalar>(input[i]);
			output[i] = static_cast<T>(value > 0.0f ? value : 0.0f);
		}
	}

	template <typename T>
	void BACKWARD_RELU_TEMPLATE(
		const T *input,
		T *inputGrad,
		const T *outputGrad,
		std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar inputValue = static_cast<Scalar>(input[i]);
			const Scalar localGrad = inputValue > 0.0f ? 1.0f : 0.0f;
			const Scalar gradient = static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * localGrad;
			inputGrad[i] = static_cast<T>(gradient);
		}
	}
}
