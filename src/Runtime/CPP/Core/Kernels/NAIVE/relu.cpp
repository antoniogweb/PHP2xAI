#include "relu.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::RELU(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess Y = accessTensor(outputTensor);

		if (X.shape != Y.shape || X.size != Y.size)
			throw std::runtime_error("ReLU: input and output dimensions must match");
		if (X.dtype != Y.dtype)
			throw std::runtime_error("ReLU: input and output dtypes must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::RELU_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size);
		});
	}

	void GraphRuntime::BACKWARD_RELU(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess Y = accessTensor(outputTensor);

		if (X.shape != Y.shape || X.size != Y.size)
			throw std::runtime_error("ReLU backward: input and output dimensions must match");
		if (X.dtype != Y.dtype)
			throw std::runtime_error("ReLU backward: input and output dtypes must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_RELU_TEMPLATE<T>(
				X.dataAs<T>(), X.gradAs<T>(), Y.gradAs<T>(), X.size);
		});
	}
}
