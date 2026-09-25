#include "layer_norm.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::LAYER_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &beta, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess B = accessTensor(beta);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0 || X.shape != Y.shape || G.shape.size() != 1
			|| B.shape != G.shape || G.shape[0] != X.shape.back()
			|| X.dtype != G.dtype || X.dtype != B.dtype || X.dtype != Y.dtype)
			throw std::runtime_error("layer_norm: dimensions or dtypes do not match");

		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::LAYER_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(), G.dataAs<T>(),
				B.dataAs<T>(), Y.dataAs<T>(), outer, width);
		});
	}

	void GraphRuntime::BACKWARD_LAYER_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &beta, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess B = accessTensor(beta);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0)
			throw std::runtime_error("layer_norm backward: invalid feature width");
		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_LAYER_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(),
				G.dataAs<T>(), Y.dataAs<T>(), Y.gradAs<T>(), X.gradAs<T>(),
				G.gradAs<T>(), B.gradAs<T>(), outer, width, X.requiresGrad,
				G.requiresGrad, B.requiresGrad);
		});
	}
}
