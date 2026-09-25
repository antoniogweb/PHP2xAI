#include "apply_causal_mask.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::APPLY_CAUSAL_MASK(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.size() < 2 || X.shape != Y.shape || X.dtype != Y.dtype)
			throw std::runtime_error("apply_causal_mask: expected matching tensors of rank >= 2");

		const int queryLength = X.shape[X.shape.size() - 2];
		const int keyLength = X.shape.back();
		if (queryLength <= 0 || keyLength < queryLength)
			throw std::runtime_error("apply_causal_mask: invalid query and key lengths");
		const std::size_t outer = X.size
			/ (static_cast<std::size_t>(queryLength) * keyLength);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::APPLY_CAUSAL_MASK_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(),
				outer, queryLength, keyLength);
		});
	}

	void GraphRuntime::BACKWARD_APPLY_CAUSAL_MASK(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		const int queryLength = X.shape[X.shape.size() - 2];
		const int keyLength = X.shape.back();
		const std::size_t outer = X.size
			/ (static_cast<std::size_t>(queryLength) * keyLength);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_APPLY_CAUSAL_MASK_TEMPLATE<T>(
				X.gradAs<T>(), Y.gradAs<T>(), outer, queryLength, keyLength);
		});
	}
}
