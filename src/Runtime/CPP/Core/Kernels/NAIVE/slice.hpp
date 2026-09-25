#pragma once

#include <cstddef>
#include <cstdint>

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void SLICE_GENERIC_AXIS_TEMPLATE(const T *input, T *output,
		std::size_t outer, std::size_t inner, int axisSize,
		int start, int sliceSize)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (int axisIndex = 0; axisIndex < sliceSize; ++axisIndex)
			{
				const std::size_t inputBase =
					(outerIndex * static_cast<std::size_t>(axisSize)
						+ static_cast<std::size_t>(start + axisIndex)) * inner;
				const std::size_t outputBase =
					(outerIndex * static_cast<std::size_t>(sliceSize)
						+ static_cast<std::size_t>(axisIndex)) * inner;
				for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
					output[outputBase + innerIndex] = input[inputBase + innerIndex];
			}
		}
	}

	template <typename T>
	void BACKWARD_SLICE_GENERIC_AXIS_TEMPLATE(const T *outputGrad, T *inputGrad,
		std::size_t outer, std::size_t inner, int axisSize,
		int start, int sliceSize)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (int axisIndex = 0; axisIndex < sliceSize; ++axisIndex)
			{
				const std::size_t inputBase =
					(outerIndex * static_cast<std::size_t>(axisSize)
						+ static_cast<std::size_t>(start + axisIndex)) * inner;
				const std::size_t outputBase =
					(outerIndex * static_cast<std::size_t>(sliceSize)
						+ static_cast<std::size_t>(axisIndex)) * inner;
				for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
				{
					const std::size_t inputIndex = inputBase + innerIndex;
					inputGrad[inputIndex] = static_cast<T>(inputGrad[inputIndex]
						+ outputGrad[outputBase + innerIndex]);
				}
			}
		}
	}

	template <typename T>
	void SLICE_LAST_TEMPLATE(const T *input, T *output, std::size_t outer,
		int axisSize, int start, int sliceSize)
	{
		#pragma omp parallel for schedule(static)
		for (std::int64_t rowIndex = 0;
			rowIndex < static_cast<std::int64_t>(outer); ++rowIndex)
		{
			const std::size_t row = static_cast<std::size_t>(rowIndex);
			for (int i = 0; i < sliceSize; ++i)
				output[row * static_cast<std::size_t>(sliceSize) + i] =
					input[row * static_cast<std::size_t>(axisSize) + start + i];
		}
	}

	template <typename T>
	void BACKWARD_SLICE_LAST_TEMPLATE(const T *outputGrad, T *inputGrad,
		std::size_t outer, int axisSize, int start, int sliceSize)
	{
		#pragma omp parallel for schedule(static)
		for (std::int64_t rowIndex = 0;
			rowIndex < static_cast<std::int64_t>(outer); ++rowIndex)
		{
			const std::size_t row = static_cast<std::size_t>(rowIndex);
			for (int i = 0; i < sliceSize; ++i)
			{
				const std::size_t inputIndex = row * static_cast<std::size_t>(axisSize) + start + i;
				inputGrad[inputIndex] = static_cast<T>(inputGrad[inputIndex]
					+ outputGrad[row * static_cast<std::size_t>(sliceSize) + i]);
			}
		}
	}
}
