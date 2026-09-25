#include "scale.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SCALE(Tensor &input, Tensor &output, Scalar scale)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("scale: tensor dimensions or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SCALE_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size, scale);
		});
	}

	void GraphRuntime::BACKWARD_SCALE(Tensor &input, Tensor &output, Scalar scale)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("scale backward: tensor dimensions or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SCALE_TEMPLATE<T>(
				X.gradAs<T>(), Y.gradAs<T>(), X.size, scale);
		});
	}
}
