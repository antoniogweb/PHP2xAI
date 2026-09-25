#include "mean.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::MEAN_1D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 1 || input.size == 0
			|| !output.shape.empty() || output.size != 1)
			throw std::runtime_error("mean 1D: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean: input and output dtypes must match");

		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::MEAN_1D_FIRST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), input.size);
		});
	}

	void GraphRuntime::MEAN_2D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 2 || input.shape[0] <= 0 || input.shape[1] <= 0
			|| output.shape.size() != 1 || output.shape[0] != input.shape[1]
			|| input.size != static_cast<std::size_t>(input.shape[0]) * input.shape[1]
			|| output.size != static_cast<std::size_t>(input.shape[1]))
			throw std::runtime_error("mean 2D: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int featureCount = input.shape[1];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::MEAN_2D_FIRST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), batchSize, featureCount);
		});
	}

	void GraphRuntime::MEAN_3D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 3 || input.shape[0] <= 0
			|| input.shape[1] <= 0 || input.shape[2] <= 0
			|| output.shape.size() != 2 || output.shape[0] != input.shape[1]
			|| output.shape[1] != input.shape[2])
			throw std::runtime_error("mean 3D: dimension mismatch");
		const std::size_t rowSize = static_cast<std::size_t>(input.shape[1]) * input.shape[2];
		if (input.size != static_cast<std::size_t>(input.shape[0]) * rowSize
			|| output.size != rowSize)
			throw std::runtime_error("mean 3D: data size mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int timeSize = input.shape[1];
		const int featureCount = input.shape[2];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::MEAN_3D_FIRST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), batchSize, timeSize, featureCount);
		});
	}

	void GraphRuntime::MEAN_GENERIC_AXIS(Tensor &, Tensor &)
	{
		throw std::runtime_error("mean: generic axis kernel is not implemented for the NAIVE backend");
	}

	void GraphRuntime::BACKWARD_MEAN_1D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 1 || input.size == 0
			|| !output.shape.empty() || output.size != 1)
			throw std::runtime_error("mean 1D backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean backward: input and output dtypes must match");

		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MEAN_1D_FIRST_TEMPLATE<T>(
				input.gradAs<T>(), output.gradAs<T>(), input.size);
		});
	}

	void GraphRuntime::BACKWARD_MEAN_2D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 2 || input.shape[0] <= 0 || input.shape[1] <= 0
			|| output.shape.size() != 1 || output.shape[0] != input.shape[1])
			throw std::runtime_error("mean 2D backward: dimension mismatch");
		const std::size_t inputSize = static_cast<std::size_t>(input.shape[0]) * input.shape[1];
		if (input.size != inputSize || output.size != static_cast<std::size_t>(input.shape[1]))
			throw std::runtime_error("mean 2D backward: data size mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean backward: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int featureCount = input.shape[1];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MEAN_2D_FIRST_TEMPLATE<T>(
				input.gradAs<T>(), output.gradAs<T>(), batchSize, featureCount);
		});
	}

	void GraphRuntime::BACKWARD_MEAN_3D_FIRST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 3 || input.shape[0] <= 0
			|| input.shape[1] <= 0 || input.shape[2] <= 0
			|| output.shape.size() != 2 || output.shape[0] != input.shape[1]
			|| output.shape[1] != input.shape[2])
			throw std::runtime_error("mean 3D backward: dimension mismatch");
		const std::size_t rowSize = static_cast<std::size_t>(input.shape[1]) * input.shape[2];
		if (input.size != static_cast<std::size_t>(input.shape[0]) * rowSize
			|| output.size != rowSize)
			throw std::runtime_error("mean 3D backward: data size mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean backward: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int timeSize = input.shape[1];
		const int featureCount = input.shape[2];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MEAN_3D_FIRST_TEMPLATE<T>(
				input.gradAs<T>(), output.gradAs<T>(), batchSize, timeSize, featureCount);
		});
	}

	void GraphRuntime::BACKWARD_MEAN_GENERIC_AXIS(Tensor &, Tensor &)
	{
		throw std::runtime_error(
			"mean backward: generic axis kernel is not implemented for the NAIVE backend");
	}
}
