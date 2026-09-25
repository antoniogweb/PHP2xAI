#include "positional_encoding.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::POSITIONAL_ENCODING(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.size() != 3 || X.shape != Y.shape || X.dtype != Y.dtype)
			throw std::runtime_error("positional_encoding: expected matching rank-3 tensors");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::POSITIONAL_ENCODING_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(),
				X.shape[0], X.shape[1], X.shape[2]);
		});
	}

	void GraphRuntime::BACKWARD_POSITIONAL_ENCODING(Tensor &input, Tensor &output)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.size != Y.size || X.dtype != Y.dtype)
			throw std::runtime_error("positional_encoding backward: tensor mismatch");

		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_POSITIONAL_ENCODING_TEMPLATE<T>(
				X.gradAs<T>(), Y.gradAs<T>(), X.size);
		});
	}
}
