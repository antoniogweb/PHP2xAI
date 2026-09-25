#include "dropout.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::DROPOUT(Tensor &inputTensor, Tensor &outputTensor,
		Scalar dropoutPerc, Scalar *mask, std::uint64_t seed, bool training)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape != output.shape || input.size != output.size)
			throw std::runtime_error("dropout: input and output dimensions must match");
		if (input.dtype != output.dtype)
			throw std::runtime_error("dropout: input and output dtypes must match");
		if (dropoutPerc < 0.0f || dropoutPerc > 100.0f)
			throw std::runtime_error("dropout: percentage must be between 0 and 100");
		if (training && input.size > 0 && mask == 0)
			throw std::runtime_error("dropout: training mask is missing");

		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::DROPOUT_TEMPLATE<T>(input.dataAs<T>(), output.dataAs<T>(), mask,
				input.size, dropoutPerc, seed, training);
		});
	}

	void GraphRuntime::BACKWARD_DROPOUT(Tensor &inputTensor, Tensor &outputTensor, const Scalar *mask)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (!input.requiresGrad)
			return;
		if (input.shape != output.shape || input.size != output.size)
			throw std::runtime_error("dropout backward: input and output dimensions must match");
		if (input.dtype != output.dtype)
			throw std::runtime_error("dropout backward: input and output dtypes must match");
		if (input.size > 0 && mask == 0)
			throw std::runtime_error("dropout backward: forward mask is missing");

		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_DROPOUT_TEMPLATE<T>(input.gradAs<T>(),
				output.gradAs<T>(), mask, input.size);
		});
	}
}
