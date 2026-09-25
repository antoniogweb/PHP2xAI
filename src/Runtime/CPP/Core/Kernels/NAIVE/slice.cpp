#include "slice.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::SLICE_LAST(Tensor &input, Tensor &output, int start, int end)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.dtype != Y.dtype || start < 0 || end <= start
			|| end > X.shape.back())
			throw std::runtime_error("slice: invalid last-axis range or dtype");

		const int axisSize = X.shape.back();
		const int sliceSize = end - start;
		const std::size_t outer = X.size / static_cast<std::size_t>(axisSize);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::SLICE_LAST_TEMPLATE<T>(
				X.dataAs<T>(), Y.dataAs<T>(), outer, axisSize, start, sliceSize);
		});
	}

	void GraphRuntime::BACKWARD_SLICE_LAST(Tensor &input, Tensor &output,
		int start, int end)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.dtype != Y.dtype || start < 0 || end <= start
			|| end > X.shape.back())
			throw std::runtime_error("slice backward: invalid last-axis range or dtype");

		const int axisSize = X.shape.back();
		const int sliceSize = end - start;
		const std::size_t outer = X.size / static_cast<std::size_t>(axisSize);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SLICE_LAST_TEMPLATE<T>(
				Y.gradAs<T>(), X.gradAs<T>(), outer, axisSize, start, sliceSize);
		});
	}

	void GraphRuntime::SLICE_GENERIC_AXIS(Tensor &inputTensor, Tensor &outputTensor,
		int axis, int start, int end)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0 || input.dtype != output.dtype)
			throw std::runtime_error("slice: generic kernel requires a non-scalar tensor of matching dtype");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || start < 0 || end <= start
			|| end > input.shape[static_cast<std::size_t>(axis)])
			throw std::runtime_error("slice: invalid generic axis or range");

		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		const int sliceSize = end - start;
		std::vector<int> expectedShape = input.shape;
		expectedShape[static_cast<std::size_t>(axis)] = sliceSize;
		if (output.shape != expectedShape
			|| output.size != input.size / static_cast<std::size_t>(axisSize)
				* static_cast<std::size_t>(sliceSize))
			throw std::runtime_error("slice: generic output shape mismatch");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::SLICE_GENERIC_AXIS_TEMPLATE<T>(
				input.dataAs<T>(), output.dataAs<T>(), outer, inner,
				axisSize, start, sliceSize);
		});
	}

	void GraphRuntime::BACKWARD_SLICE_GENERIC_AXIS(Tensor &inputTensor,
		Tensor &outputTensor, int axis, int start, int end)
	{
		TensorAccess input = accessTensor(inputTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(input.shape.size());
		if (rank == 0 || input.dtype != output.dtype)
			throw std::runtime_error("slice backward: generic kernel requires matching non-scalar tensors");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || start < 0 || end <= start
			|| end > input.shape[static_cast<std::size_t>(axis)])
			throw std::runtime_error("slice backward: invalid generic axis or range");

		const int axisSize = input.shape[static_cast<std::size_t>(axis)];
		const int sliceSize = end - start;
		std::vector<int> expectedShape = input.shape;
		expectedShape[static_cast<std::size_t>(axis)] = sliceSize;
		if (output.shape != expectedShape
			|| output.size != input.size / static_cast<std::size_t>(axisSize)
				* static_cast<std::size_t>(sliceSize))
			throw std::runtime_error("slice backward: generic output shape mismatch");

		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(input.shape[static_cast<std::size_t>(i)]);
		dispatchDType(input.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_SLICE_GENERIC_AXIS_TEMPLATE<T>(
				output.gradAs<T>(), input.gradAs<T>(), outer, inner,
				axisSize, start, sliceSize);
		});
	}
}
