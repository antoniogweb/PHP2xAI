#include "silu.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SILU(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess Y = accessTensor(outputTensor);

		if (X.shape != Y.shape || X.size != Y.size)
			throw std::runtime_error("silu: input and output dimensions must match");
		if (X.dtype != Y.dtype)
			throw std::runtime_error("silu: input and output dtypes must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SILU_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size);
		});
	}

	void GraphRuntime::BACKWARD_SILU(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess Y = accessTensor(outputTensor);

		if (X.shape != Y.shape || X.size != Y.size)
			throw std::runtime_error("silu backward: input and output dimensions must match");
		if (X.dtype != Y.dtype)
			throw std::runtime_error("silu backward: input and output dtypes must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SILU_TEMPLATE<T>(
				X.dataAs<T>(), X.gradAs<T>(), Y.gradAs<T>(), X.size);
		});
	}
}
