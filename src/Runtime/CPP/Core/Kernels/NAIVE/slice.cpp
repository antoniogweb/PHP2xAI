#include "slice.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SLICE_LAST(Tensor &input, Tensor &output, int start, int end)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.dtype != Y.dtype || start < 0 || end <= start
			|| end > X.shape.back())
			throw std::runtime_error("slice: invalid last-axis range or dtype");

		const int axisSize = X.shape.back();
		const int sliceSize = end - start;
		const std::size_t outer = X.size / static_cast<std::size_t>(axisSize);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SLICE_LAST_TEMPLATE<T>(
				X.dataAs<T>(), Y.dataAs<T>(), outer, axisSize, start, sliceSize);
		});
	}

	void GraphRuntime::BACKWARD_SLICE_LAST(Tensor &input, Tensor &output,
		int start, int end)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.dtype != Y.dtype || start < 0 || end <= start
			|| end > X.shape.back())
			throw std::runtime_error("slice backward: invalid last-axis range or dtype");

		const int axisSize = X.shape.back();
		const int sliceSize = end - start;
		const std::size_t outer = X.size / static_cast<std::size_t>(axisSize);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SLICE_LAST_TEMPLATE<T>(
				Y.gradAs<T>(), X.gradAs<T>(), outer, axisSize, start, sliceSize);
		});
	}
}
