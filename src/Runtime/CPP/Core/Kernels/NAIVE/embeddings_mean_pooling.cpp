#include "embeddings_mean_pooling.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::EMBEDDINGS_MEAN_POOLING(
		Tensor &idsTensor, Tensor &tableTensor, Tensor &outputTensor, int padId)
	{
		TensorAccess ids = accessTensor(idsTensor);
		TensorAccess table = accessTensor(tableTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (ids.shape.size() != 2 || table.shape.size() != 2 || output.shape.size() != 2)
			throw std::runtime_error("embeddings_mean_pooling: expected IDs [B,L], table [V,D], output [B,D]");

		const int batchSize = ids.shape[0];
		const int sequenceLength = ids.shape[1];
		const int vocabularySize = table.shape[0];
		const int embeddingSize = table.shape[1];
		if (output.shape[0] != batchSize || output.shape[1] != embeddingSize
			|| ids.size != static_cast<std::size_t>(batchSize) * sequenceLength
			|| table.size != static_cast<std::size_t>(vocabularySize) * embeddingSize
			|| output.size != static_cast<std::size_t>(batchSize) * embeddingSize)
			throw std::runtime_error("embeddings_mean_pooling: dimension mismatch");
		if (output.dtype != table.dtype)
			throw std::runtime_error("embeddings_mean_pooling: output dtype must match embedding table dtype");

		dispatchDType(ids.dtype, [&]<typename TIndex>()
		{
			dispatchDType(table.dtype, [&]<typename T>()
			{
				Templates::EMBEDDINGS_MEAN_POOLING_TEMPLATE<TIndex, T>(
					ids.dataAs<TIndex>(), table.dataAs<T>(), output.dataAs<T>(),
					batchSize, sequenceLength, vocabularySize, embeddingSize, padId);
			});
		});
	}

	void GraphRuntime::BACKWARD_EMBEDDINGS_MEAN_POOLING(
		Tensor &idsTensor, Tensor &tableTensor, Tensor &outputTensor, int padId)
	{
		TensorAccess ids = accessTensor(idsTensor);
		TensorAccess table = accessTensor(tableTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (!table.requiresGrad)
			return;
		if (ids.shape.size() != 2 || table.shape.size() != 2 || output.shape.size() != 2
			|| output.shape[0] != ids.shape[0] || output.shape[1] != table.shape[1])
			throw std::runtime_error("embeddings_mean_pooling backward: dimension mismatch");
		if (output.dtype != table.dtype)
			throw std::runtime_error("embeddings_mean_pooling backward: output dtype must match embedding table dtype");

		const int batchSize = ids.shape[0];
		const int sequenceLength = ids.shape[1];
		const int vocabularySize = table.shape[0];
		const int embeddingSize = table.shape[1];
		dispatchDType(ids.dtype, [&]<typename TIndex>()
		{
			dispatchDType(table.dtype, [&]<typename T>()
			{
				Templates::BACKWARD_EMBEDDINGS_MEAN_POOLING_TEMPLATE<TIndex, T>(
					ids.dataAs<TIndex>(), table.gradAs<T>(), output.gradAs<T>(),
					batchSize, sequenceLength, vocabularySize, embeddingSize, padId);
			});
		});
	}
}
