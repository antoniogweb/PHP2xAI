#pragma once

#include <cmath>
#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void GELU_TEMPLATE(const T *input, T *output, std::size_t elementCount)
	{
		const Scalar scale = std::sqrt(2.0f / 3.14159265358979323846f);
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar x = static_cast<Scalar>(input[i]);
			const Scalar xCubed = x * x * x;
			const Scalar u = scale * (x + 0.044715f * xCubed);
			const Scalar value = 0.5f * x * (1.0f + std::tanh(u));
			output[i] = static_cast<T>(value);
		}
	}

	template <typename T>
	void BACKWARD_GELU_TEMPLATE(
		const T *input,
		T *inputGrad,
		const T *outputGrad,
		std::size_t elementCount)
	{
		const Scalar scale = std::sqrt(2.0f / 3.14159265358979323846f);
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar x = static_cast<Scalar>(input[i]);
			const Scalar xSquared = x * x;
			const Scalar u = scale * (x + 0.044715f * xSquared * x);
			const Scalar tanhU = std::tanh(u);
			const Scalar du = scale * (1.0f + 3.0f * 0.044715f * xSquared);
			const Scalar localGrad = 0.5f * (1.0f + tanhU)
				+ 0.5f * x * (1.0f - tanhU * tanhU) * du;
			const Scalar grad = static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * localGrad;
			inputGrad[i] = static_cast<T>(grad);
		}
	}
}
