#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	namespace EmbeddingsDetail
	{
		template <typename TIndex>
		int checkedTokenId(TIndex value, int vocabularySize)
		{
			const long double tokenValue = static_cast<long double>(value);
			if (!std::isfinite(tokenValue) || tokenValue < 0.0L
				|| tokenValue >= static_cast<long double>(vocabularySize)
				|| std::floor(tokenValue) != tokenValue)
				throw std::runtime_error("embeddings: token ID must be an in-range integer");

			return static_cast<int>(tokenValue);
		}
	}

	// IDs and embedding values may have different tensor dtypes.
	template <typename TIndex, typename T>
	void EMBEDDINGS_TEMPLATE(const TIndex *ids, const T *table, T *output,
		int batchSize, int sequenceLength, int vocabularySize, int embeddingSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int token = 0; token < sequenceLength; ++token)
			{
				const std::size_t idsIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
				const int tokenId = EmbeddingsDetail::checkedTokenId(ids[idsIndex], vocabularySize);
				const std::size_t tableOffset = static_cast<std::size_t>(tokenId) * embeddingSize;
				const std::size_t outputOffset = idsIndex * embeddingSize;

				for (int dimension = 0; dimension < embeddingSize; ++dimension)
					output[outputOffset + dimension] = table[tableOffset + dimension];
			}
		}
	}

	template <typename TIndex, typename T>
	void BACKWARD_EMBEDDINGS_TEMPLATE(const TIndex *ids, T *tableGrad, const T *outputGrad,
		int batchSize, int sequenceLength, int vocabularySize, int embeddingSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int token = 0; token < sequenceLength; ++token)
			{
				const std::size_t idsIndex = static_cast<std::size_t>(batch) * sequenceLength + token;
				const int tokenId = EmbeddingsDetail::checkedTokenId(ids[idsIndex], vocabularySize);
				const std::size_t tableOffset = static_cast<std::size_t>(tokenId) * embeddingSize;
				const std::size_t outputOffset = idsIndex * embeddingSize;

				for (int dimension = 0; dimension < embeddingSize; ++dimension)
				{
					const Scalar accumulated = static_cast<Scalar>(tableGrad[tableOffset + dimension])
						+ static_cast<Scalar>(outputGrad[outputOffset + dimension]);
					tableGrad[tableOffset + dimension] = static_cast<T>(accumulated);
				}
			}
		}
	}
}
