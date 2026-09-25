#include "sig.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SIG(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("sig: tensor dimensions or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SIG_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.size);
		});
	}

	void GraphRuntime::BACKWARD_SIG(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("sig backward: tensor dimensions or dtypes differ");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SIG_TEMPLATE<T>(
				Y.dataAs<T>(), X.gradAs<T>(), Y.gradAs<T>(), X.size);
		});
	}
}
