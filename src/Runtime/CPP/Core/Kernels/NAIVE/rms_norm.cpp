#include "rms_norm.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::RMS_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &output, Scalar epsilon)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0 || X.shape != Y.shape || G.shape.size() != 1
			|| G.shape[0] != X.shape.back() || X.dtype != G.dtype
			|| X.dtype != Y.dtype || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm: dimensions, dtypes, or epsilon are invalid");

		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::RMS_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(), G.dataAs<T>(),
				Y.dataAs<T>(), outer, width, epsilon);
		});
	}

	void GraphRuntime::BACKWARD_RMS_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &output, Scalar epsilon)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0 || X.shape != Y.shape
			|| G.shape.size() != 1 || G.shape[0] != X.shape.back()
			|| X.dtype != G.dtype || X.dtype != Y.dtype || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm backward: invalid tensors or epsilon");
		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_RMS_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(),
				G.dataAs<T>(), Y.gradAs<T>(), X.gradAs<T>(), G.gradAs<T>(),
				outer, width, epsilon, X.requiresGrad, G.requiresGrad);
		});
	}
}
