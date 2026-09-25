#include "softmax.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SOFTMAX_1D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 1 || input.shape != output.shape || input.size == 0)
			throw std::runtime_error("softmax 1D: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax: input and output dtypes must match");

		const int featureCount = input.shape[0];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_1D_LAST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), featureCount);
		});
	}

	void GraphRuntime::SOFTMAX_2D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 2 || input.shape != output.shape
			|| input.shape[1] <= 0
			|| input.size != static_cast<std::size_t>(input.shape[0]) * input.shape[1]
			|| output.size != input.size)
			throw std::runtime_error("softmax 2D: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int featureCount = input.shape[1];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_2D_LAST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), batchSize, featureCount);
		});
	}

	void GraphRuntime::SOFTMAX_3D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 3 || input.shape != output.shape
			|| input.shape[2] <= 0)
			throw std::runtime_error("softmax 3D: dimension mismatch");
		const std::size_t expectedSize = static_cast<std::size_t>(input.shape[0])
			* static_cast<std::size_t>(input.shape[1]) * input.shape[2];
		if (input.size != expectedSize || output.size != expectedSize)
			throw std::runtime_error("softmax 3D: data size mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int timeSize = input.shape[1];
		const int featureCount = input.shape[2];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_3D_LAST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), batchSize, timeSize, featureCount);
		});
	}

	void GraphRuntime::SOFTMAX_4D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 4 || input.shape != output.shape
			|| input.shape[3] <= 0)
			throw std::runtime_error("softmax 4D: dimension mismatch");
		const std::size_t expectedSize = static_cast<std::size_t>(input.shape[0])
			* input.shape[1] * input.shape[2] * input.shape[3];
		if (input.size != expectedSize || output.size != expectedSize)
			throw std::runtime_error("softmax 4D: data size mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int headCount = input.shape[1];
		const int timeSize = input.shape[2];
		const int featureCount = input.shape[3];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_4D_LAST_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(),
				batchSize, headCount, timeSize, featureCount);
		});
	}

	void GraphRuntime::SOFTMAX_GENERIC_AXIS(Tensor &inputTensor,
		Tensor &outputTensor, int axis)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0 || input.shape != output.shape || input.size != output.size
			|| input.dtype != output.dtype)
			throw std::runtime_error("softmax generic: input and output dimensions or dtypes differ");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || input.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("softmax generic: axis is invalid or empty");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SOFTMAX_GENERIC_AXIS_TEMPLATE<T>(input.dataAs<T>(),
				output.dataAs<T>(), outer, inner, axisSize);
		});
	}

	void GraphRuntime::BACKWORD_SOFTMAX_1D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 1 || input.shape != output.shape || input.size == 0)
			throw std::runtime_error("softmax 1D backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax backward: input and output dtypes must match");

		const int featureCount = input.shape[0];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWORD_SOFTMAX_1D_LAST_TEMPLATE<T>(
				output.dataAs<T>(), input.gradAs<T>(), output.gradAs<T>(), featureCount);
		});
	}

	void GraphRuntime::BACKWORD_SOFTMAX_2D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 2 || input.shape != output.shape || input.shape[1] <= 0
			|| input.size != output.size)
			throw std::runtime_error("softmax 2D backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax backward: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int featureCount = input.shape[1];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWORD_SOFTMAX_2D_LAST_TEMPLATE<T>(
				output.dataAs<T>(), input.gradAs<T>(), output.gradAs<T>(),
				batchSize, featureCount);
		});
	}

	void GraphRuntime::BACKWORD_SOFTMAX_3D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 3 || input.shape != output.shape || input.shape[2] <= 0
			|| input.size != output.size)
			throw std::runtime_error("softmax 3D backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax backward: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int timeSize = input.shape[1];
		const int featureCount = input.shape[2];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWORD_SOFTMAX_3D_LAST_TEMPLATE<T>(
				output.dataAs<T>(), input.gradAs<T>(), output.gradAs<T>(),
				batchSize, timeSize, featureCount);
		});
	}

	void GraphRuntime::BACKWORD_SOFTMAX_4D_LAST(Tensor &inputTensor, Tensor &outputTensor)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		if (input.shape.size() != 4 || input.shape != output.shape || input.shape[3] <= 0
			|| input.size != output.size)
			throw std::runtime_error("softmax 4D backward: dimension mismatch");
		if (input.dtype != output.dtype)
			throw std::runtime_error("softmax backward: input and output dtypes must match");

		const int batchSize = input.shape[0];
		const int headCount = input.shape[1];
		const int timeSize = input.shape[2];
		const int featureCount = input.shape[3];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWORD_SOFTMAX_4D_LAST_TEMPLATE<T>(
				output.dataAs<T>(), input.gradAs<T>(), output.gradAs<T>(),
				batchSize, headCount, timeSize, featureCount);
		});
	}

	void GraphRuntime::BACKWORD_SOFTMAX_GENERIC_AXIS(Tensor &inputTensor,
		Tensor &outputTensor, int axis)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0 || input.shape != output.shape || input.size != output.size
			|| input.dtype != output.dtype)
			throw std::runtime_error("softmax generic backward: input and output dimensions or dtypes differ");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || input.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("softmax generic backward: axis is invalid or empty");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SOFTMAX_GENERIC_AXIS_TEMPLATE<T>(
				output.dataAs<T>(), output.gradAs<T>(), input.gradAs<T>(),
				outer, inner, axisSize);
		});
	}
}
