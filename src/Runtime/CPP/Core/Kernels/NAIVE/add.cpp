#include "add.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::ADD_1D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape != B.shape || A.shape != C.shape)
			throw std::runtime_error("add: 1D kernel requires equal shapes");
		if (A.size != B.size || A.size != C.size)
			throw std::runtime_error("add: 1D kernel data size mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::ADD_1D_LAST_TEMPLATE<T>(
				A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(), A.size);
		});
	}

	void GraphRuntime::ADD_2D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 2 || B.shape.size() != 1 || C.shape != A.shape)
			throw std::runtime_error("add: 2D kernel expects [batch, features] plus [features]");

		const int batchSize = A.shape[0];
		const int featureCount = A.shape[1];
		if (B.shape[0] != featureCount
			|| A.size != static_cast<std::size_t>(batchSize * featureCount)
			|| C.size != A.size)
			throw std::runtime_error("add: 2D kernel dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::ADD_2D_LAST_TEMPLATE<T>(
				A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(), batchSize, featureCount);
		});
	}

	void GraphRuntime::ADD_3D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 1 || C.shape != A.shape)
			throw std::runtime_error("add: 3D kernel expects [batch, time, features] plus [features]");

		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int featureCount = A.shape[2];
		const std::size_t expectedSize = static_cast<std::size_t>(batchSize)
			* static_cast<std::size_t>(timeSize)
			* static_cast<std::size_t>(featureCount);
		if (B.shape[0] != featureCount || A.size != expectedSize || C.size != A.size)
			throw std::runtime_error("add: 3D kernel dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::ADD_3D_LAST_TEMPLATE<T>(
				A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(),
				batchSize, timeSize, featureCount);
		});
	}

	void GraphRuntime::BACKWARD_ADD_1D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape != B.shape || A.shape != C.shape
			|| A.size != B.size || A.size != C.size)
			throw std::runtime_error("add backward: 1D kernel dimensions do not match");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add backward: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_ADD_1D_LAST_TEMPLATE<T>(
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(), A.size,
				A.requiresGrad, B.requiresGrad);
		});
	}

	void GraphRuntime::BACKWARD_ADD_2D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 2 || B.shape.size() != 1 || C.shape != A.shape)
			throw std::runtime_error("add backward: 2D kernel shape mismatch");

		const int batchSize = A.shape[0];
		const int featureCount = A.shape[1];
		if (B.shape[0] != featureCount || C.size != A.size)
			throw std::runtime_error("add backward: 2D kernel dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add backward: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_ADD_2D_LAST_TEMPLATE<T>(
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(),
				batchSize, featureCount, A.requiresGrad, B.requiresGrad);
		});
	}

	void GraphRuntime::BACKWARD_ADD_3D_LAST(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 1 || C.shape != A.shape)
			throw std::runtime_error("add backward: 3D kernel shape mismatch");

		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int featureCount = A.shape[2];
		if (B.shape[0] != featureCount || C.size != A.size)
			throw std::runtime_error("add backward: 3D kernel dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("add backward: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_ADD_3D_LAST_TEMPLATE<T>(
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(),
				batchSize, timeSize, featureCount, A.requiresGrad, B.requiresGrad);
		});
	}

}
