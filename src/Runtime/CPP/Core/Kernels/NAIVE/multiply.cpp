#include "multiply.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::MULTIPLY(Tensor &a, Tensor &b, Tensor &output)
	{
		TensorAccess A = accessTensor(a);
		TensorAccess B = accessTensor(b);
		TensorAccess O = accessTensor(output);

		if (A.size != B.size || A.size != O.size
			|| A.shape != B.shape || A.shape != O.shape
			|| A.dtype != B.dtype || A.dtype != O.dtype)
		{
			throw std::runtime_error("multiply: tensor dimensions or dtypes differ");
		}

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::MULTIPLY_TEMPLATE<T>(
				A.dataAs<T>(), B.dataAs<T>(), O.dataAs<T>(), O.size);
		});
	}

	void GraphRuntime::BACKWARD_MULTIPLY(Tensor &a, Tensor &b, Tensor &output)
	{
		TensorAccess A = accessTensor(a);
		TensorAccess B = accessTensor(b);
		TensorAccess O = accessTensor(output);

		if (A.size != B.size || A.size != O.size
			|| A.dtype != B.dtype || A.dtype != O.dtype)
		{
			throw std::runtime_error("multiply backward: tensor dimensions or dtypes differ");
		}

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MULTIPLY_TEMPLATE<T>(
				A.dataAs<T>(), B.dataAs<T>(), A.gradAs<T>(), B.gradAs<T>(),
				O.gradAs<T>(), O.size, A.requiresGrad, B.requiresGrad);
		});
	}
}
