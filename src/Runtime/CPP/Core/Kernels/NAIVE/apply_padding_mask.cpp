#include "apply_padding_mask.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::APPLY_PADDING_MASK(Tensor &scores, Tensor &mask, Tensor &output)
	{
		TensorAccess X = accessTensor(scores);
		TensorAccess M = accessTensor(mask);
		TensorAccess Y = accessTensor(output);
		if (X.shape.size() < 2 || X.shape != Y.shape || X.shape[0] <= 0
			|| X.shape.back() <= 0 || X.dtype != Y.dtype
			|| M.shape.size() != 2 || M.shape[0] != X.shape[0]
			|| M.shape[1] != X.shape.back())
			throw std::runtime_error("apply_padding_mask: tensor dimensions do not match");

		const int batchRows = static_cast<int>(X.size
			/ (static_cast<std::size_t>(X.shape[0]) * X.shape.back()));
		const std::size_t outer = X.size / static_cast<std::size_t>(X.shape.back());
		dispatchDType(X.dtype, [&]<typename T>()
		{
			dispatchDType(M.dtype, [&]<typename MType>()
			{
				Templates::APPLY_PADDING_MASK_TEMPLATE<T, MType>(
					X.dataAs<T>(), M.dataAs<MType>(), Y.dataAs<T>(), outer,
					batchRows, X.shape.back());
			});
		});
	}

	void GraphRuntime::BACKWARD_APPLY_PADDING_MASK(
		Tensor &scores, Tensor &mask, Tensor &output)
	{
		TensorAccess X = accessTensor(scores);
		TensorAccess M = accessTensor(mask);
		TensorAccess Y = accessTensor(output);
		const int batchRows = static_cast<int>(X.size
			/ (static_cast<std::size_t>(X.shape[0]) * X.shape.back()));
		const std::size_t outer = X.size / static_cast<std::size_t>(X.shape.back());
		dispatchDType(X.dtype, [&]<typename T>()
		{
			dispatchDType(M.dtype, [&]<typename MType>()
			{
				Templates::BACKWARD_APPLY_PADDING_MASK_TEMPLATE<T, MType>(
					M.dataAs<MType>(), X.gradAs<T>(), Y.gradAs<T>(), outer,
					batchRows, X.shape.back());
			});
		});
	}
}
