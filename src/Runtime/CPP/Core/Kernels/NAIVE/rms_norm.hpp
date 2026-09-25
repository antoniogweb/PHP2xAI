#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void RMS_NORM_GENERIC_TEMPLATE(const T *input, const T *gamma, T *output,
		std::size_t outer, std::size_t inner, int width, Scalar epsilon)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar meanSquare = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar value = static_cast<Scalar>(input[index]);
					meanSquare += value * value;
				}
				const Scalar inverseRms = 1.0f / std::sqrt(meanSquare / width + epsilon);
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					output[index] = static_cast<T>(static_cast<Scalar>(input[index])
						* inverseRms * static_cast<Scalar>(gamma[i]));
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_RMS_NORM_GENERIC_TEMPLATE(const T *input, const T *gamma,
		const T *outputGrad, T *inputGrad, T *gammaGrad,
		std::size_t outer, std::size_t inner, int width, Scalar epsilon,
		bool inputNeedsGrad, bool gammaNeedsGrad)
	{
		std::vector<Scalar> gammaAccum(static_cast<std::size_t>(width), 0.0f);
		if (gammaNeedsGrad)
			for (int i = 0; i < width; ++i)
				gammaAccum[static_cast<std::size_t>(i)] = static_cast<Scalar>(gammaGrad[i]);
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar meanSquare = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar value = static_cast<Scalar>(input[index]);
					meanSquare += value * value;
				}
				const Scalar inverseRms = 1.0f / std::sqrt(meanSquare / width + epsilon);
				Scalar sumGradTimesInput = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					sumGradTimesInput += static_cast<Scalar>(outputGrad[index])
						* static_cast<Scalar>(gamma[i]) * static_cast<Scalar>(input[index]);
					if (gammaNeedsGrad)
						gammaAccum[static_cast<std::size_t>(i)] +=
							static_cast<Scalar>(outputGrad[index])
							* static_cast<Scalar>(input[index]) * inverseRms;
				}
				if (inputNeedsGrad)
				{
					for (int i = 0; i < width; ++i)
					{
						const std::size_t index =
							(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
						const Scalar value = static_cast<Scalar>(input[index]);
						const Scalar contribution = inverseRms * static_cast<Scalar>(gamma[i])
							* static_cast<Scalar>(outputGrad[index])
							- value * inverseRms * inverseRms * inverseRms
								* sumGradTimesInput / width;
						inputGrad[index] = static_cast<T>(static_cast<Scalar>(inputGrad[index]) + contribution);
					}
				}
			}
		}
		if (gammaNeedsGrad)
			for (int i = 0; i < width; ++i)
				gammaGrad[i] = static_cast<T>(gammaAccum[static_cast<std::size_t>(i)]);
	}

	template <typename T>
	void RMS_NORM_LAST_AXIS_TEMPLATE(const T *input, const T *gamma, T *output,
		std::size_t outer, int width, Scalar epsilon)
	{
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t base = row * static_cast<std::size_t>(width);
			Scalar meanSquare = 0.0f;
			for (int i = 0; i < width; ++i)
			{
				const Scalar value = static_cast<Scalar>(input[base + i]);
				meanSquare += value * value;
			}
			const Scalar inverseRms = 1.0f / std::sqrt(meanSquare / width + epsilon);
			for (int i = 0; i < width; ++i)
				output[base + i] = static_cast<T>(static_cast<Scalar>(input[base + i])
					* inverseRms * static_cast<Scalar>(gamma[i]));
		}
	}

	template <typename T>
	void BACKWARD_RMS_NORM_LAST_AXIS_TEMPLATE(const T *input, const T *gamma,
		const T *outputGrad, T *inputGrad, T *gammaGrad, std::size_t outer,
		int width, Scalar epsilon, bool inputNeedsGrad, bool gammaNeedsGrad)
	{
		std::vector<Scalar> gammaAccum(static_cast<std::size_t>(width), 0.0f);
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t base = row * static_cast<std::size_t>(width);
			Scalar meanSquare = 0.0f;
			for (int i = 0; i < width; ++i)
			{
				const Scalar value = static_cast<Scalar>(input[base + i]);
				meanSquare += value * value;
			}
			const Scalar inverseRms = 1.0f / std::sqrt(meanSquare / width + epsilon);
			Scalar sum = 0.0f;
			for (int i = 0; i < width; ++i)
				sum += static_cast<Scalar>(outputGrad[base + i])
					* static_cast<Scalar>(gamma[i]) * static_cast<Scalar>(input[base + i]);
				
			if (inputNeedsGrad)
			{
				for (int i = 0; i < width; ++i)
				{
					const Scalar value = static_cast<Scalar>(input[base + i]);
					const Scalar grad = inverseRms * static_cast<Scalar>(gamma[i])
						* static_cast<Scalar>(outputGrad[base + i])
						- value * inverseRms * inverseRms * inverseRms * sum / width;
					inputGrad[base + i] = static_cast<T>(static_cast<Scalar>(inputGrad[base + i]) + grad);
				}
			}
			if (gammaNeedsGrad)
				for (int i = 0; i < width; ++i)
					gammaAccum[static_cast<std::size_t>(i)] += static_cast<Scalar>(outputGrad[base + i])
						* static_cast<Scalar>(input[base + i]) * inverseRms;
		}
		if (gammaNeedsGrad)
			for (int i = 0; i < width; ++i)
				gammaGrad[i] = static_cast<T>(static_cast<Scalar>(gammaGrad[i])
					+ gammaAccum[static_cast<std::size_t>(i)]);
	}
}
