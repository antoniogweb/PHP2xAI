#pragma once

#include <cstddef>
#include <vector>

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void TRANSPOSE_TEMPLATE(const T *input, T *output,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		std::vector<int> outputShape(shape.size());
		std::vector<std::size_t> inputStrides(shape.size());
		std::vector<std::size_t> outputStrides(shape.size());
		std::size_t count = 1;
		for (std::size_t axis = 0; axis < shape.size(); ++axis)
		{
			outputShape[axis] = shape[static_cast<std::size_t>(permutation[axis])];
			count *= static_cast<std::size_t>(outputShape[axis]);
		}
		std::size_t stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			inputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
		}
		stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			outputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(outputShape[static_cast<std::size_t>(axis)]);
		}

		for (std::size_t outputIndex = 0; outputIndex < count; ++outputIndex)
		{
			std::size_t inputIndex = 0;
			for (std::size_t axis = 0; axis < shape.size(); ++axis)
			{
				const std::size_t coordinate =
					(outputIndex / outputStrides[axis]) % static_cast<std::size_t>(outputShape[axis]);
				inputIndex += coordinate * inputStrides[static_cast<std::size_t>(permutation[axis])];
			}
			output[outputIndex] = input[inputIndex];
		}
	}

	template <typename T>
	void BACKWARD_TRANSPOSE_TEMPLATE(const T *outputGrad, T *inputGrad,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		std::vector<int> outputShape(shape.size());
		std::vector<std::size_t> inputStrides(shape.size());
		std::vector<std::size_t> outputStrides(shape.size());
		std::size_t count = 1;
		for (std::size_t axis = 0; axis < shape.size(); ++axis)
		{
			outputShape[axis] = shape[static_cast<std::size_t>(permutation[axis])];
			count *= static_cast<std::size_t>(outputShape[axis]);
		}
		std::size_t stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			inputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
		}
		stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			outputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(outputShape[static_cast<std::size_t>(axis)]);
		}

		for (std::size_t outputIndex = 0; outputIndex < count; ++outputIndex)
		{
			std::size_t inputIndex = 0;
			for (std::size_t axis = 0; axis < shape.size(); ++axis)
			{
				const std::size_t coordinate =
					(outputIndex / outputStrides[axis]) % static_cast<std::size_t>(outputShape[axis]);
				inputIndex += coordinate * inputStrides[static_cast<std::size_t>(permutation[axis])];
			}
			inputGrad[inputIndex] = static_cast<T>(inputGrad[inputIndex] + outputGrad[outputIndex]);
		}
	}

	template <typename T>
	void TRANSPOSE_GENERIC_TEMPLATE(const T *input, T *output,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		TRANSPOSE_TEMPLATE<T>(input, output, shape, permutation);
	}

	template <typename T>
	void BACKWARD_TRANSPOSE_GENERIC_TEMPLATE(const T *outputGrad, T *inputGrad,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(outputGrad, inputGrad, shape, permutation);
	}

	template <typename T>
	void TRANSPOSE_2D_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {1, 0});
	}
	template <typename T>
	void TRANSPOSE_3D_LAST_TWO_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {0, 2, 1});
	}
	template <typename T>
	void TRANSPOSE_4D_LAST_TWO_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {0, 1, 3, 2});
	}
	template <typename T>
	void TRANSPOSE_4D_AXIS_1_2_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {0, 2, 1, 3});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_2D_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {1, 0});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_3D_LAST_TWO_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {0, 2, 1});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_4D_LAST_TWO_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {0, 1, 3, 2});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_4D_AXIS_1_2_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {0, 2, 1, 3});
	}
}
