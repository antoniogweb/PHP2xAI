#pragma once

#include <cstddef>
#include <cmath>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void POSITIONAL_ENCODING_TEMPLATE(const T *input, T *output,
		int batch, int length, int dimension)
	{
		for (int b = 0; b < batch; ++b)
		{
			for (int position = 0; position < length; ++position)
			{
				for (int d = 0; d < dimension; ++d)
				{
					const int pairDimension = d - d % 2;
					const Scalar exponent = static_cast<Scalar>(pairDimension)
						/ static_cast<Scalar>(dimension);
					const Scalar angle = static_cast<Scalar>(position)
						/ std::pow(10000.0f, exponent);
					const Scalar encoding = d % 2 == 0 ? std::sin(angle) : std::cos(angle);
					const std::size_t index = static_cast<std::size_t>(
						(b * length + position) * dimension + d);
					output[index] = static_cast<T>(static_cast<Scalar>(input[index]) + encoding);
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_POSITIONAL_ENCODING_TEMPLATE(T *inputGrad, const T *outputGrad,
		std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
			inputGrad[i] = static_cast<T>(inputGrad[i] + outputGrad[i]);
	}
}
