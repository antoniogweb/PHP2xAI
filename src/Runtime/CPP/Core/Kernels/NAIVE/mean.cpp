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

	void GraphRuntime::MEAN_GENERIC_AXIS(Tensor &inputTensor, Tensor &outputTensor, int axis)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean: input and output dtypes must match");

		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0)
		{
			if (input.size != 1 || output.size != 1 || !output.shape.empty())
				throw std::runtime_error("mean: scalar input/output mismatch");
			dispatchDType(input.dtype, [&]<typename T>()
			{
				output.dataAs<T>()[0] = input.dataAs<T>()[0];
			});
			return;
		}

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || input.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("mean: axis is invalid or empty");

		std::vector<int> expectedShape = input.shape;
		expectedShape.erase(expectedShape.begin() + axis);
		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		if (output.shape != expectedShape
			|| output.size != input.size / static_cast<std::size_t>(axisSize))
			throw std::runtime_error("mean: output shape mismatch");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::MEAN_GENERIC_AXIS_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), outer, inner, axisSize);
		});
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

	void GraphRuntime::BACKWARD_MEAN_GENERIC_AXIS(Tensor &inputTensor, Tensor &outputTensor, int axis)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.dtype != output.dtype)
			throw std::runtime_error("mean backward: input and output dtypes must match");

		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0)
		{
			if (input.size != 1 || output.size != 1)
				throw std::runtime_error("mean backward: scalar input/output mismatch");
			dispatchDType(input.dtype, [&]<typename T>()
			{
				input.gradAs<T>()[0] += output.gradAs<T>()[0];
			});
			return;
		}

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || input.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("mean backward: axis is invalid or empty");

		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		std::vector<int> expectedShape = input.shape;
		expectedShape.erase(expectedShape.begin() + axis);
		if (output.shape != expectedShape
			|| output.size != input.size / static_cast<std::size_t>(axisSize))
			throw std::runtime_error("mean backward: output shape mismatch");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_MEAN_GENERIC_AXIS_TEMPLATE<T>(
				input.gradAs<T>(), output.gradAs<T>(), outer, inner, axisSize);
		});
	}
}
