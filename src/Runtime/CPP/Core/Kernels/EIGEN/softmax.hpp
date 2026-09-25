#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void SOFTMAX_4D_LAST_TEMPLATE_EIGEN(const T *input, T *output,
		int batch, int heads, int time, int width)
	{
		const long long rowCount = static_cast<long long>(batch) * heads * time;
		#pragma omp parallel for schedule(static)
		for (long long row = 0; row < rowCount; ++row)
		{
			const std::size_t offset = static_cast<std::size_t>(row * width);
			Scalar maximum = -std::numeric_limits<Scalar>::infinity();
			for (int column = 0; column < width; ++column)
				maximum = std::max(maximum, static_cast<Scalar>(input[offset + column]));
			if (maximum == -std::numeric_limits<Scalar>::infinity())
				maximum = 0.0f;

			Scalar sum = 0.0f;
			for (int column = 0; column < width; ++column)
			{
				const Scalar value = std::exp(
					static_cast<Scalar>(input[offset + column]) - maximum);
				output[offset + column] = static_cast<T>(value);
				sum += value;
			}

			const Scalar inverseSum = sum > 0.0f ? 1.0f / sum : 1.0f;
			for (int column = 0; column < width; ++column)
				output[offset + column] = static_cast<T>(
					static_cast<Scalar>(output[offset + column]) * inverseSum);
		}
	}
}
