#include "matmul.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntimeEigen::MATMUL_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2)
			throw std::runtime_error("matmul: dimension mismatch");
		const int batchSize = A.shape[0];
		const int inputSize = A.shape[1];
		const int outputSize = B.shape[1];
		if (inputSize != B.shape[0] || C.shape[0] != batchSize || C.shape[1] != outputSize
			|| A.size != static_cast<std::size_t>(batchSize) * inputSize
			|| B.size != static_cast<std::size_t>(inputSize) * outputSize
			|| C.size != static_cast<std::size_t>(batchSize) * outputSize)
			throw std::runtime_error("matmul: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::MATMUL_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(),
				batchSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::MATMUL_1B_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 3 || C.shape.size() != 3)
			throw std::runtime_error("matmul: dimension mismatch");
		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int inputSize = A.shape[2];
		const int outputSize = B.shape[2];
		if (batchSize != B.shape[0] || inputSize != B.shape[1]
			|| C.shape[0] != batchSize || C.shape[1] != timeSize || C.shape[2] != outputSize
			|| A.size != static_cast<std::size_t>(batchSize) * timeSize * inputSize
			|| B.size != static_cast<std::size_t>(batchSize) * inputSize * outputSize
			|| C.size != static_cast<std::size_t>(batchSize) * timeSize * outputSize)
			throw std::runtime_error("matmul: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::MATMUL_1B_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(),
				batchSize, timeSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::MATMUL_2B_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 4 || B.shape.size() != 4 || C.shape.size() != 4)
			throw std::runtime_error("matmul: dimension mismatch");
		const int batchSize = A.shape[0];
		const int headCount = A.shape[1];
		const int timeSize = A.shape[2];
		const int inputSize = A.shape[3];
		const int outputSize = B.shape[3];
		if (batchSize != B.shape[0] || headCount != B.shape[1] || inputSize != B.shape[2]
			|| C.shape[0] != batchSize || C.shape[1] != headCount
			|| C.shape[2] != timeSize || C.shape[3] != outputSize
			|| A.size != static_cast<std::size_t>(batchSize) * headCount * timeSize * inputSize
			|| B.size != static_cast<std::size_t>(batchSize) * headCount * inputSize * outputSize
			|| C.size != static_cast<std::size_t>(batchSize) * headCount * timeSize * outputSize)
			throw std::runtime_error("matmul: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::MATMUL_2B_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(),
				batchSize, headCount, timeSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::MATMUL_1B_2D_2D_LINEAR(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 2 || C.shape.size() != 3)
			throw std::runtime_error("matmul: dimension mismatch");
		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int inputSize = A.shape[2];
		const int outputSize = B.shape[1];
		if (inputSize != B.shape[0] || C.shape[0] != batchSize
			|| C.shape[1] != timeSize || C.shape[2] != outputSize
			|| A.size != static_cast<std::size_t>(batchSize) * timeSize * inputSize
			|| B.size != static_cast<std::size_t>(inputSize) * outputSize
			|| C.size != static_cast<std::size_t>(batchSize) * timeSize * outputSize)
			throw std::runtime_error("matmul: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul: input and output dtypes must match");

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::MATMUL_1B_2D_2D_LINEAR_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(), C.dataAs<T>(),
				batchSize, timeSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::BACKWARD_MATMUL_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 2 || B.shape.size() != 2 || C.shape.size() != 2
			|| A.shape[1] != B.shape[0] || C.shape[0] != A.shape[0] || C.shape[1] != B.shape[1])
			throw std::runtime_error("matmul backward: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul backward: input and output dtypes must match");
		const int batchSize = A.shape[0];
		const int inputSize = A.shape[1];
		const int outputSize = B.shape[1];

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MATMUL_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(),
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(), batchSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::BACKWARD_MATMUL_1B_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 3 || C.shape.size() != 3
			|| A.shape[0] != B.shape[0] || A.shape[2] != B.shape[1]
			|| C.shape[0] != A.shape[0] || C.shape[1] != A.shape[1] || C.shape[2] != B.shape[2])
			throw std::runtime_error("matmul backward: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul backward: input and output dtypes must match");
		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int inputSize = A.shape[2];
		const int outputSize = B.shape[2];

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MATMUL_1B_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(),
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(),
				batchSize, timeSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::BACKWARD_MATMUL_2B_2D_2D(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 4 || B.shape.size() != 4 || C.shape.size() != 4
			|| A.shape[0] != B.shape[0] || A.shape[1] != B.shape[1] || A.shape[3] != B.shape[2]
			|| C.shape[0] != A.shape[0] || C.shape[1] != A.shape[1]
			|| C.shape[2] != A.shape[2] || C.shape[3] != B.shape[3])
			throw std::runtime_error("matmul backward: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul backward: input and output dtypes must match");
		const int batchSize = A.shape[0];
		const int headCount = A.shape[1];
		const int timeSize = A.shape[2];
		const int inputSize = A.shape[3];
		const int outputSize = B.shape[3];

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MATMUL_2B_2D_2D_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(),
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(),
				batchSize, headCount, timeSize, inputSize, outputSize);
		});
	}

	void GraphRuntimeEigen::BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &aTensor, Tensor &bTensor, Tensor &cTensor)
	{
		TensorAccess A = accessTensor(aTensor);
		TensorAccess B = accessTensor(bTensor);
		TensorAccess C = accessTensor(cTensor);
		if (A.shape.size() != 3 || B.shape.size() != 2 || C.shape.size() != 3
			|| A.shape[2] != B.shape[0] || C.shape[0] != A.shape[0]
			|| C.shape[1] != A.shape[1] || C.shape[2] != B.shape[1])
			throw std::runtime_error("matmul backward: dimension mismatch");
		if (A.dtype != B.dtype || A.dtype != C.dtype)
			throw std::runtime_error("matmul backward: input and output dtypes must match");
		const int batchSize = A.shape[0];
		const int timeSize = A.shape[1];
		const int inputSize = A.shape[2];
		const int outputSize = B.shape[1];

		dispatchDType(A.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MATMUL_1B_2D_2D_LINEAR_TEMPLATE_EIGEN<T>(A.dataAs<T>(), B.dataAs<T>(),
				A.gradAs<T>(), B.gradAs<T>(), C.gradAs<T>(),
				batchSize, timeSize, inputSize, outputSize);
		});
	}
}
