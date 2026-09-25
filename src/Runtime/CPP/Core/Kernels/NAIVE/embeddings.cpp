#include "embeddings.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::EMBEDDINGS(Tensor &idsTensor, Tensor &tableTensor, Tensor &outputTensor)
	{
		TensorAccess ids = accessTensor(idsTensor);
		TensorAccess table = accessTensor(tableTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (ids.shape.size() != 2 || table.shape.size() != 2 || output.shape.size() != 3)
			throw std::runtime_error("embeddings: expected IDs [B,L], table [V,D], output [B,L,D]");

		const int batchSize = ids.shape[0];
		const int sequenceLength = ids.shape[1];
		const int vocabularySize = table.shape[0];
		const int embeddingSize = table.shape[1];
		if (output.shape[0] != batchSize || output.shape[1] != sequenceLength
			|| output.shape[2] != embeddingSize
			|| ids.size != static_cast<std::size_t>(batchSize) * sequenceLength
			|| table.size != static_cast<std::size_t>(vocabularySize) * embeddingSize
			|| output.size != static_cast<std::size_t>(batchSize) * sequenceLength * embeddingSize)
			throw std::runtime_error("embeddings: dimension mismatch");
		if (output.dtype != table.dtype)
			throw std::runtime_error("embeddings: output dtype must match embedding table dtype");

		dispatchDType(ids.dtype, [&]<typename TIndex>()
		{
			dispatchDType(table.dtype, [&]<typename T>()
			{
				Templates::EMBEDDINGS_TEMPLATE<TIndex, T>(ids.dataAs<TIndex>(),
					table.dataAs<T>(), output.dataAs<T>(), batchSize,
					sequenceLength, vocabularySize, embeddingSize);
			});
		});
	}

	void GraphRuntime::BACKWARD_EMBEDDINGS(Tensor &idsTensor, Tensor &tableTensor, Tensor &outputTensor)
	{
		TensorAccess ids = accessTensor(idsTensor);
		TensorAccess table = accessTensor(tableTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (!table.requiresGrad)
			return;

		if (ids.shape.size() != 2 || table.shape.size() != 2 || output.shape.size() != 3
			|| output.shape[0] != ids.shape[0] || output.shape[1] != ids.shape[1]
			|| output.shape[2] != table.shape[1])
			throw std::runtime_error("embeddings backward: dimension mismatch");
		if (output.dtype != table.dtype)
			throw std::runtime_error("embeddings backward: output dtype must match embedding table dtype");

		const int batchSize = ids.shape[0];
		const int sequenceLength = ids.shape[1];
		const int vocabularySize = table.shape[0];
		const int embeddingSize = table.shape[1];
		dispatchDType(ids.dtype, [&]<typename TIndex>()
		{
			dispatchDType(table.dtype, [&]<typename T>()
			{
				Templates::BACKWARD_EMBEDDINGS_TEMPLATE<TIndex, T>(ids.dataAs<TIndex>(),
					table.gradAs<T>(), output.gradAs<T>(), batchSize,
					sequenceLength, vocabularySize, embeddingSize);
			});
		});
	}
}
