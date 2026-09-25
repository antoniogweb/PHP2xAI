#include "gelu.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntimeEigen::GELU(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape != Y.shape || X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("gelu Eigen: input and output tensors must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::GELU_TEMPLATE_EIGEN<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size);
		});
	}

	void GraphRuntimeEigen::BACKWARD_GELU(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (!X.requiresGrad)
			return;
		if (X.shape != Y.shape || X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("gelu Eigen backward: input and output tensors must match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_GELU_TEMPLATE_EIGEN<T>(
				X.dataAs<T>(), X.gradAs<T>(), Y.gradAs<T>(), X.size);
		});
	}
}
