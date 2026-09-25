#include "softmax.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntimeEigen::SOFTMAX_4D_LAST(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.size() != 4 || X.shape != Y.shape || X.dtype != Y.dtype
			|| X.shape[3] <= 0 || X.size != Y.size)
			throw std::runtime_error("softmax Eigen 4D: dimensions or dtypes do not match");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_4D_LAST_TEMPLATE_EIGEN<T>(X.dataAs<T>(), Y.dataAs<T>(),
				X.shape[0], X.shape[1], X.shape[2], X.shape[3]);
		});
	}
}
