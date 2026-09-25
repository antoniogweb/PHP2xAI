#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#include "embeddings.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename TIndex, typename T>
	void EMBEDDINGS_MEAN_POOLING_TEMPLATE(const TIndex *ids, const T *table, T *output,
		int batchSize, int sequenceLength, int vocabularySize, int embeddingSize, int padId)
	{
		std::vector<Scalar> sums(static_cast<std::size_t>(embeddingSize));
		for (int batch = 0; batch < batchSize; ++batch)
		{
			int validTokens = 0;
			const std::size_t idsOffset = static_cast<std::size_t>(batch) * sequenceLength;
			std::fill(sums.begin(), sums.end(), 0.0f);

			for (int token = 0; token < sequenceLength; ++token)
			{
				const int tokenId = EmbeddingsDetail::checkedTokenId(
					ids[idsOffset + token], vocabularySize);
				if (tokenId == padId)
					continue;

				++validTokens;
				const std::size_t tableOffset = static_cast<std::size_t>(tokenId) * embeddingSize;
				for (int dimension = 0; dimension < embeddingSize; ++dimension)
					sums[static_cast<std::size_t>(dimension)] +=
						static_cast<Scalar>(table[tableOffset + dimension]);
			}

			const std::size_t outputOffset = static_cast<std::size_t>(batch) * embeddingSize;
			for (int dimension = 0; dimension < embeddingSize; ++dimension)
			{
				Scalar average = 0.0f;
				if (validTokens > 0)
					average = sums[static_cast<std::size_t>(dimension)] / static_cast<Scalar>(validTokens);
				output[outputOffset + dimension] = static_cast<T>(average);
			}
		}
	}

	template <typename TIndex, typename T>
	void BACKWARD_EMBEDDINGS_MEAN_POOLING_TEMPLATE(const TIndex *ids, T *tableGrad,
		const T *outputGrad, int batchSize, int sequenceLength,
		int vocabularySize, int embeddingSize, int padId)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			int validTokens = 0;
			const std::size_t idsOffset = static_cast<std::size_t>(batch) * sequenceLength;
			for (int token = 0; token < sequenceLength; ++token)
			{
				const int tokenId = EmbeddingsDetail::checkedTokenId(
					ids[idsOffset + token], vocabularySize);
				if (tokenId != padId)
					++validTokens;
			}
			if (validTokens == 0)
				continue;

			const Scalar scale = 1.0f / static_cast<Scalar>(validTokens);
			const std::size_t outputOffset = static_cast<std::size_t>(batch) * embeddingSize;
			for (int token = 0; token < sequenceLength; ++token)
			{
				const int tokenId = EmbeddingsDetail::checkedTokenId(
					ids[idsOffset + token], vocabularySize);
				if (tokenId == padId)
					continue;
				const std::size_t tableOffset = static_cast<std::size_t>(tokenId) * embeddingSize;

				for (int dimension = 0; dimension < embeddingSize; ++dimension)
				{
					const Scalar accumulated = static_cast<Scalar>(tableGrad[tableOffset + dimension])
						+ static_cast<Scalar>(outputGrad[outputOffset + dimension]) * scale;
					tableGrad[tableOffset + dimension] = static_cast<T>(accumulated);
				}
			}
		}
	}
}
