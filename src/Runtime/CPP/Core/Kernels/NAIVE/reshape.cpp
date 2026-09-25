#include "reshape.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::RESHAPE(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("reshape: element counts or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::RESHAPE_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size);
		});
	}

	void GraphRuntime::BACKWARD_RESHAPE(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("reshape backward: element counts or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_RESHAPE_TEMPLATE<T>(
				Y.gradAs<T>(), X.gradAs<T>(), X.size);
		});
	}
}
