#pragma once

#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void MEAN_GENERIC_AXIS_TEMPLATE(const T *input, T *output,
		std::size_t outer, std::size_t inner, int axisSize)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar sum = 0.0f;
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t inputIndex =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					sum += static_cast<Scalar>(input[inputIndex]);
				}
				output[outerIndex * inner + innerIndex] =
					static_cast<T>(sum / static_cast<Scalar>(axisSize));
			}
		}
	}

	template <typename T>
	void BACKWARD_MEAN_GENERIC_AXIS_TEMPLATE(T *inputGrad,
		const T *outputGrad, std::size_t outer, std::size_t inner, int axisSize)
	{
		const Scalar inverseAxis = 1.0f / static_cast<Scalar>(axisSize);
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
			{
				for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
				{
					const std::size_t inputIndex =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					const std::size_t outputIndex = outerIndex * inner + innerIndex;
					const Scalar contribution =
						static_cast<Scalar>(outputGrad[outputIndex]) * inverseAxis;
					inputGrad[inputIndex] = static_cast<T>(
						static_cast<Scalar>(inputGrad[inputIndex]) + contribution);
				}
			}
		}
	}

	template <typename T>
	void MEAN_1D_FIRST_TEMPLATE(const T *input, T *output, std::size_t elementCount)
	{
		Scalar sum = 0.0f;
		for (std::size_t i = 0; i < elementCount; ++i)
			sum += static_cast<Scalar>(input[i]);

		output[0] = static_cast<T>(sum / static_cast<Scalar>(elementCount));
	}

	template <typename T>
	void MEAN_2D_FIRST_TEMPLATE(
		const T *input,
		T *output,
		int batchSize,
		int featureCount)
	{
		for (int feature = 0; feature < featureCount; ++feature)
		{
			Scalar sum = 0.0f;
			for (int batch = 0; batch < batchSize; ++batch)
			{
				const std::size_t inputIndex = static_cast<std::size_t>(batch * featureCount + feature);
				sum += static_cast<Scalar>(input[inputIndex]);
			}
			output[feature] = static_cast<T>(sum / static_cast<Scalar>(batchSize));
		}
	}

	template <typename T>
	void MEAN_3D_FIRST_TEMPLATE(
		const T *input,
		T *output,
		int batchSize,
		int timeSize,
		int featureCount)
	{
		const int rowSize = timeSize * featureCount;
		for (int index = 0; index < rowSize; ++index)
		{
			Scalar sum = 0.0f;
			for (int batch = 0; batch < batchSize; ++batch)
			{
				const std::size_t inputIndex = static_cast<std::size_t>(batch * rowSize + index);
				sum += static_cast<Scalar>(input[inputIndex]);
			}
			output[index] = static_cast<T>(sum / static_cast<Scalar>(batchSize));
		}
	}

	template <typename T>
	void BACKWARD_MEAN_1D_FIRST_TEMPLATE(
		T *inputGrad,
		const T *outputGrad,
		std::size_t elementCount)
	{
		const Scalar contribution = static_cast<Scalar>(outputGrad[0])
			/ static_cast<Scalar>(elementCount);
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			inputGrad[i] = static_cast<T>(
				static_cast<Scalar>(inputGrad[i]) + contribution);
		}
	}

	template <typename T>
	void BACKWARD_MEAN_2D_FIRST_TEMPLATE(
		T *inputGrad,
		const T *outputGrad,
		int batchSize,
		int featureCount)
	{
		const Scalar inverseBatch = 1.0f / static_cast<Scalar>(batchSize);
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int feature = 0; feature < featureCount; ++feature)
			{
				const std::size_t inputIndex = static_cast<std::size_t>(batch * featureCount + feature);
				const Scalar contribution = static_cast<Scalar>(outputGrad[feature]) * inverseBatch;
				inputGrad[inputIndex] = static_cast<T>(
					static_cast<Scalar>(inputGrad[inputIndex]) + contribution);
			}
		}
	}

	template <typename T>
	void BACKWARD_MEAN_3D_FIRST_TEMPLATE(
		T *inputGrad,
		const T *outputGrad,
		int batchSize,
		int timeSize,
		int featureCount)
	{
		const int rowSize = timeSize * featureCount;
		const Scalar inverseBatch = 1.0f / static_cast<Scalar>(batchSize);
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const int batchOffset = batch * rowSize;
			for (int index = 0; index < rowSize; ++index)
			{
				const std::size_t inputIndex = static_cast<std::size_t>(batchOffset + index);
				const Scalar contribution = static_cast<Scalar>(outputGrad[index]) * inverseBatch;
				inputGrad[inputIndex] = static_cast<T>(
					static_cast<Scalar>(inputGrad[inputIndex]) + contribution);
			}
		}
	}
}
