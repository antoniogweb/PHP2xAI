#pragma once

#include <cmath>
#include <cstddef>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	namespace
	{
		template <typename T>
		void softmaxRow(const T *input, T *output, int featureCount)
		{
			Scalar maxValue = static_cast<Scalar>(input[0]);
			for (int feature = 1; feature < featureCount; ++feature)
			{
				const Scalar value = static_cast<Scalar>(input[feature]);
				if (value > maxValue)
					maxValue = value;
			}

			if (maxValue == -std::numeric_limits<Scalar>::infinity())
			{
				for (int feature = 0; feature < featureCount; ++feature)
					output[feature] = static_cast<T>(0);
				return;
			}

			Scalar sum = 0.0f;
			for (int feature = 0; feature < featureCount; ++feature)
				sum += std::exp(static_cast<Scalar>(input[feature]) - maxValue);

			const Scalar inverseSum = sum == 0.0f ? 0.0f : 1.0f / sum;
			for (int feature = 0; feature < featureCount; ++feature)
			{
				const Scalar value = std::exp(static_cast<Scalar>(input[feature]) - maxValue);
				output[feature] = static_cast<T>(value * inverseSum);
			}
		}

		template <typename T>
		void backwardSoftmaxRows(
			const T *output,
			T *inputGrad,
			const T *outputGrad,
			int rowCount,
			int featureCount)
		{
			for (int row = 0; row < rowCount; ++row)
			{
				const int rowStart = row * featureCount;
				Scalar dot = 0.0f;
				for (int feature = 0; feature < featureCount; ++feature)
				{
					const std::size_t index = static_cast<std::size_t>(rowStart + feature);
					dot += static_cast<Scalar>(outputGrad[index])
						* static_cast<Scalar>(output[index]);
				}

				for (int feature = 0; feature < featureCount; ++feature)
				{
					const std::size_t index = static_cast<std::size_t>(rowStart + feature);
					const Scalar value = static_cast<Scalar>(output[index]);
					const Scalar contribution = value
						* (static_cast<Scalar>(outputGrad[index]) - dot);
					inputGrad[index] = static_cast<T>(
						static_cast<Scalar>(inputGrad[index]) + contribution);
				}
			}
		}
	}

	template <typename T>
	void SOFTMAX_1D_LAST_TEMPLATE(const T *input, T *output, int featureCount)
	{
		softmaxRow(input, output, featureCount);
	}

	template <typename T>
	void SOFTMAX_2D_LAST_TEMPLATE(
		const T *input,
		T *output,
		int batchSize,
		int featureCount)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const std::size_t offset = static_cast<std::size_t>(batch) * featureCount;
			softmaxRow(input + offset, output + offset, featureCount);
		}
	}

	template <typename T>
	void SOFTMAX_3D_LAST_TEMPLATE(
		const T *input,
		T *output,
		int batchSize,
		int timeSize,
		int featureCount)
	{
		const int rowCount = batchSize * timeSize;
		for (int row = 0; row < rowCount; ++row)
		{
			const std::size_t offset = static_cast<std::size_t>(row) * featureCount;
			softmaxRow(input + offset, output + offset, featureCount);
		}
	}

	template <typename T>
	void SOFTMAX_4D_LAST_TEMPLATE(
		const T *input,
		T *output,
		int batchSize,
		int headCount,
		int timeSize,
		int featureCount)
	{
		const int rowCount = batchSize * headCount * timeSize;
		for (int row = 0; row < rowCount; ++row)
		{
			const std::size_t offset = static_cast<std::size_t>(row) * featureCount;
			softmaxRow(input + offset, output + offset, featureCount);
		}
	}

	template <typename T>
	void BACKWORD_SOFTMAX_1D_LAST_TEMPLATE(
		const T *output,
		T *inputGrad,
		const T *outputGrad,
		int featureCount)
	{
		backwardSoftmaxRows(output, inputGrad, outputGrad, 1, featureCount);
	}

	template <typename T>
	void BACKWORD_SOFTMAX_2D_LAST_TEMPLATE(
		const T *output,
		T *inputGrad,
		const T *outputGrad,
		int batchSize,
		int featureCount)
	{
		backwardSoftmaxRows(output, inputGrad, outputGrad, batchSize, featureCount);
	}

	template <typename T>
	void BACKWORD_SOFTMAX_3D_LAST_TEMPLATE(
		const T *output,
		T *inputGrad,
		const T *outputGrad,
		int batchSize,
		int timeSize,
		int featureCount)
	{
		backwardSoftmaxRows(output, inputGrad, outputGrad, batchSize * timeSize, featureCount);
	}

	template <typename T>
	void BACKWORD_SOFTMAX_4D_LAST_TEMPLATE(
		const T *output,
		T *inputGrad,
		const T *outputGrad,
		int batchSize,
		int headCount,
		int timeSize,
		int featureCount)
	{
		backwardSoftmaxRows(
			output, inputGrad, outputGrad, batchSize * headCount * timeSize, featureCount);
	}
}
