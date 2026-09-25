#include "mean_pooling.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::MEAN_POOLING(Tensor &inputTensor, Tensor &maskTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess mask = accessTensor(maskTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 3 || mask.shape.size() != 2 || output.shape.size() != 2)
			throw std::runtime_error("mean_pooling: expected input [B,L,D], mask [B,L], output [B,D]");

		const int batchSize = input.shape[0];
		const int sequenceLength = input.shape[1];
		const int featureSize = input.shape[2];
		if (mask.shape[0] != batchSize || mask.shape[1] != sequenceLength
			|| output.shape[0] != batchSize || output.shape[1] != featureSize
			|| input.size != static_cast<std::size_t>(batchSize) * sequenceLength * featureSize
			|| mask.size != static_cast<std::size_t>(batchSize) * sequenceLength
			|| output.size != static_cast<std::size_t>(batchSize) * featureSize)
			throw std::runtime_error("mean_pooling: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean_pooling: output dtype must match input dtype");

		dispatchDType(input.dtype, [&]<typename T>()
		{
			dispatchDType(mask.dtype, [&]<typename TMask>()
			{
				Templates::MEAN_POOLING_TEMPLATE<T, TMask>(input.dataAs<T>(), mask.dataAs<TMask>(),
					output.dataAs<T>(), batchSize, sequenceLength, featureSize);
			});
		});
	}

	void GraphRuntime::BACKWARD_MEAN_POOLING(Tensor &inputTensor, Tensor &maskTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess mask = accessTensor(maskTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (!input.requiresGrad)
			return;
		if (input.shape.size() != 3 || mask.shape.size() != 2 || output.shape.size() != 2
			|| mask.shape[0] != input.shape[0] || mask.shape[1] != input.shape[1]
			|| output.shape[0] != input.shape[0] || output.shape[1] != input.shape[2])
			throw std::runtime_error("mean_pooling backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean_pooling backward: output dtype must match input dtype");

		const int batchSize = input.shape[0];
		const int sequenceLength = input.shape[1];
		const int featureSize = input.shape[2];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			dispatchDType(mask.dtype, [&]<typename TMask>()
			{
				Templates::BACKWARD_MEAN_POOLING_TEMPLATE<T, TMask>(output.gradAs<T>(),
					mask.dataAs<TMask>(), input.gradAs<T>(), batchSize, sequenceLength, featureSize);
			});
		});
	}
}
