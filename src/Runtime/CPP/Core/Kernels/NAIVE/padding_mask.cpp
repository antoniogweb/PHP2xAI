#include "padding_mask.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::PADDING_MASK(Tensor &idsTensor, Tensor &outputTensor, int padId)
	{
		TensorAccess ids = accessTensor(idsTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (ids.shape.size() != 2 || output.shape.size() != 2 || ids.shape != output.shape
			|| ids.size != output.size)
			throw std::runtime_error("padding_mask: input and output dimensions must match [B,L]");
		if (ids.dtype != output.dtype)
			throw std::runtime_error("padding_mask: output dtype must match token ID dtype");

		dispatchDType(ids.dtype, [&]<typename T>()
		{
			Templates::PADDING_MASK_TEMPLATE<T, T>(
				ids.dataAs<T>(), output.dataAs<T>(), ids.size, padId);
		});
	}
}
