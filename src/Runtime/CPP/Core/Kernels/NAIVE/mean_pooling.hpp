#pragma once

#include <cstddef>
#include <stdexcept>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	namespace MeanPoolingDetail
	{
		template <typename TMask>
		bool isValidToken(TMask value)
		{
			const long double maskValue = static_cast<long double>(value);
			if (maskValue != 0.0L && maskValue != 1.0L)
				throw std::runtime_error("mean_pooling: mask values must be 0 or 1");
			return maskValue == 1.0L;
		}
	}

	// The mask can use a different dtype from the pooled values.
	template <typename T, typename TMask>
	void MEAN_POOLING_TEMPLATE(const T *input, const TMask *mask, T *output,
		int batchSize, int sequenceLength, int featureSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			int validTokens = 0;
			for (int token = 0; token < sequenceLength; ++token)
			{
				const std::size_t maskIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
				if (MeanPoolingDetail::isValidToken(mask[maskIndex]))
					++validTokens;
			}

			const std::size_t outputOffset = static_cast<std::size_t>(batch) * featureSize;
			for (int feature = 0; feature < featureSize; ++feature)
			{
				Scalar sum = 0.0f;
				if (validTokens > 0)
				{
					for (int token = 0; token < sequenceLength; ++token)
					{
						const std::size_t maskIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
						if (!MeanPoolingDetail::isValidToken(mask[maskIndex]))
							continue;
						const std::size_t inputIndex = maskIndex * featureSize + feature;
						sum += static_cast<Scalar>(input[inputIndex]);
					}
					sum /= static_cast<Scalar>(validTokens);
				}
				output[outputOffset + feature] = static_cast<T>(sum);
			}
		}
	}

	template <typename T, typename TMask>
	void BACKWARD_MEAN_POOLING_TEMPLATE(const T *outputGrad, const TMask *mask, T *inputGrad,
		int batchSize, int sequenceLength, int featureSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			int validTokens = 0;
			for (int token = 0; token < sequenceLength; ++token)
			{
				const std::size_t maskIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
				if (MeanPoolingDetail::isValidToken(mask[maskIndex]))
					++validTokens;
			}
			if (validTokens == 0)
				continue;

			const Scalar scale = 1.0f / static_cast<Scalar>(validTokens);
			const std::size_t outputOffset = static_cast<std::size_t>(batch) * featureSize;
			for (int token = 0; token < sequenceLength; ++token)
			{
				const std::size_t maskIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
				if (!MeanPoolingDetail::isValidToken(mask[maskIndex]))
					continue;
				const std::size_t inputOffset = maskIndex * featureSize;

				for (int feature = 0; feature < featureSize; ++feature)
				{
					const Scalar accumulated = static_cast<Scalar>(inputGrad[inputOffset + feature])
						+ static_cast<Scalar>(outputGrad[outputOffset + feature]) * scale;
					inputGrad[inputOffset + feature] = static_cast<T>(accumulated);
				}
			}
		}
	}
}
