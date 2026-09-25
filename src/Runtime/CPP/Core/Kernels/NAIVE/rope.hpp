#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void ROPE_GENERIC_TEMPLATE(const T *input, T *output,
		const std::vector<int> &shape, int positionAxis, int rotationAxis,
		int offset, Scalar base, bool rotateHalf)
	{
		const std::size_t rank = shape.size();
		std::vector<std::size_t> strides(rank, 1);
		for (std::size_t axis = rank; axis > 1; --axis)
			strides[axis - 2] = strides[axis - 1]
				* static_cast<std::size_t>(shape[axis - 1]);
		const int length = shape[static_cast<std::size_t>(positionAxis)];
		const int dimension = shape[static_cast<std::size_t>(rotationAxis)];
		const int half = dimension / 2;
		std::size_t total = 1;
		for (std::size_t i = 0; i < shape.size(); ++i)
			total *= static_cast<std::size_t>(shape[i]);
		const std::size_t sliceCount = total / static_cast<std::size_t>(length * dimension);

		for (std::size_t slice = 0; slice < sliceCount; ++slice)
		{
			std::size_t remaining = slice;
			std::size_t baseIndex = 0;
			for (std::size_t axis = rank; axis > 0; --axis)
			{
				const std::size_t current = axis - 1;
				if (static_cast<int>(current) == positionAxis
					|| static_cast<int>(current) == rotationAxis)
					continue;
				const std::size_t coordinate = remaining
					% static_cast<std::size_t>(shape[current]);
				remaining /= static_cast<std::size_t>(shape[current]);
				baseIndex += coordinate * strides[current];
			}
			for (int position = 0; position < length; ++position)
			{
				for (int i = 0; i < half; ++i)
				{
					const Scalar exponent = 2.0f * i / static_cast<Scalar>(dimension);
					const Scalar angle = (position + offset) / std::pow(base, exponent);
					const Scalar cosine = std::cos(angle);
					const Scalar sine = std::sin(angle);
					const int left = rotateHalf ? i : 2 * i;
					const int right = rotateHalf ? i + half : 2 * i + 1;
					const std::size_t positionOffset = baseIndex
						+ static_cast<std::size_t>(position) * strides[static_cast<std::size_t>(positionAxis)];
					const std::size_t leftIndex = positionOffset
						+ static_cast<std::size_t>(left) * strides[static_cast<std::size_t>(rotationAxis)];
					const std::size_t rightIndex = positionOffset
						+ static_cast<std::size_t>(right) * strides[static_cast<std::size_t>(rotationAxis)];
					const Scalar x0 = static_cast<Scalar>(input[leftIndex]);
					const Scalar x1 = static_cast<Scalar>(input[rightIndex]);
					output[leftIndex] = static_cast<T>(x0 * cosine - x1 * sine);
					output[rightIndex] = static_cast<T>(x0 * sine + x1 * cosine);
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_ROPE_GENERIC_TEMPLATE(const T *outputGrad, T *inputGrad,
		const std::vector<int> &shape, int positionAxis, int rotationAxis,
		int offset, Scalar base, bool rotateHalf)
	{
		const std::size_t rank = shape.size();
		std::vector<std::size_t> strides(rank, 1);
		for (std::size_t axis = rank; axis > 1; --axis)
			strides[axis - 2] = strides[axis - 1]
				* static_cast<std::size_t>(shape[axis - 1]);
		const int length = shape[static_cast<std::size_t>(positionAxis)];
		const int dimension = shape[static_cast<std::size_t>(rotationAxis)];
		const int half = dimension / 2;
		std::size_t total = 1;
		for (std::size_t i = 0; i < shape.size(); ++i)
			total *= static_cast<std::size_t>(shape[i]);
		const std::size_t sliceCount = total / static_cast<std::size_t>(length * dimension);

		for (std::size_t slice = 0; slice < sliceCount; ++slice)
		{
			std::size_t remaining = slice;
			std::size_t baseIndex = 0;
			for (std::size_t axis = rank; axis > 0; --axis)
			{
				const std::size_t current = axis - 1;
				if (static_cast<int>(current) == positionAxis
					|| static_cast<int>(current) == rotationAxis)
					continue;
				const std::size_t coordinate = remaining
					% static_cast<std::size_t>(shape[current]);
				remaining /= static_cast<std::size_t>(shape[current]);
				baseIndex += coordinate * strides[current];
			}
			for (int position = 0; position < length; ++position)
			{
				for (int i = 0; i < half; ++i)
				{
					const Scalar exponent = 2.0f * i / static_cast<Scalar>(dimension);
					const Scalar angle = (position + offset) / std::pow(base, exponent);
					const Scalar cosine = std::cos(angle);
					const Scalar sine = std::sin(angle);
					const int left = rotateHalf ? i : 2 * i;
					const int right = rotateHalf ? i + half : 2 * i + 1;
					const std::size_t positionOffset = baseIndex
						+ static_cast<std::size_t>(position) * strides[static_cast<std::size_t>(positionAxis)];
					const std::size_t leftIndex = positionOffset
						+ static_cast<std::size_t>(left) * strides[static_cast<std::size_t>(rotationAxis)];
					const std::size_t rightIndex = positionOffset
						+ static_cast<std::size_t>(right) * strides[static_cast<std::size_t>(rotationAxis)];
					const Scalar g0 = static_cast<Scalar>(outputGrad[leftIndex]);
					const Scalar g1 = static_cast<Scalar>(outputGrad[rightIndex]);
					inputGrad[leftIndex] = static_cast<T>(static_cast<Scalar>(inputGrad[leftIndex])
						+ g0 * cosine + g1 * sine);
					inputGrad[rightIndex] = static_cast<T>(static_cast<Scalar>(inputGrad[rightIndex])
						- g0 * sine + g1 * cosine);
				}
			}
		}
	}
	template <typename T>
	void ROPE_LAST_TWO_TEMPLATE(const T *input, T *output, int length,
		int dimension, int offset, Scalar base, bool rotateHalf)
	{
		if (dimension % 2 != 0) throw std::runtime_error("rope: rotation dimension must be even");
		const int half = dimension / 2;
		for (int position = 0; position < length; ++position)
		{
			for (int i = 0; i < half; ++i)
			{
				const Scalar exponent = 2.0f * i / static_cast<Scalar>(dimension);
				const Scalar angle = (position + offset) / std::pow(base, exponent);
				const Scalar cosine = std::cos(angle);
				const Scalar sine = std::sin(angle);
				const int left = rotateHalf ? i : 2 * i;
				const int right = rotateHalf ? i + half : 2 * i + 1;
				const std::size_t leftIndex = static_cast<std::size_t>(position * dimension + left);
				const std::size_t rightIndex = static_cast<std::size_t>(position * dimension + right);
				const Scalar x0 = static_cast<Scalar>(input[leftIndex]);
				const Scalar x1 = static_cast<Scalar>(input[rightIndex]);
				output[leftIndex] = static_cast<T>(x0 * cosine - x1 * sine);
				output[rightIndex] = static_cast<T>(x0 * sine + x1 * cosine);
			}
		}
	}

	template <typename T>
	void BACKWARD_ROPE_LAST_TWO_TEMPLATE(const T *outputGrad, T *inputGrad,
		int length, int dimension, int offset, Scalar base, bool rotateHalf)
	{
		const int half = dimension / 2;
		for (int position = 0; position < length; ++position)
			for (int i = 0; i < half; ++i)
			{
				const Scalar exponent = 2.0f * i / static_cast<Scalar>(dimension);
				const Scalar angle = (position + offset) / std::pow(base, exponent);
				const Scalar cosine = std::cos(angle);
				const Scalar sine = std::sin(angle);
				const int left = rotateHalf ? i : 2 * i;
				const int right = rotateHalf ? i + half : 2 * i + 1;
				const std::size_t leftIndex = static_cast<std::size_t>(position * dimension + left);
				const std::size_t rightIndex = static_cast<std::size_t>(position * dimension + right);
				const Scalar g0 = static_cast<Scalar>(outputGrad[leftIndex]);
				const Scalar g1 = static_cast<Scalar>(outputGrad[rightIndex]);
				inputGrad[leftIndex] = static_cast<T>(static_cast<Scalar>(inputGrad[leftIndex]) + g0 * cosine + g1 * sine);
				inputGrad[rightIndex] = static_cast<T>(static_cast<Scalar>(inputGrad[rightIndex]) - g0 * sine + g1 * cosine);
			}
	}

	template <typename T>
	void ROPE_INTERLEAVED_LAST_TWO_TEMPLATE(const T *x, T *y, int length,
		int dimension, int offset, Scalar base)
	{
		ROPE_LAST_TWO_TEMPLATE<T>(x, y, length, dimension, offset, base, false);
	}
	template <typename T>
	void ROPE_ROTATE_HALF_LAST_TWO_TEMPLATE(const T *x, T *y, int length,
		int dimension, int offset, Scalar base)
	{
		ROPE_LAST_TWO_TEMPLATE<T>(x, y, length, dimension, offset, base, true);
	}
	template <typename T>
	void BACKWARD_ROPE_INTERLEAVED_LAST_TWO_TEMPLATE(const T *yg, T *xg,
		int length, int dimension, int offset, Scalar base)
	{
		BACKWARD_ROPE_LAST_TWO_TEMPLATE<T>(yg, xg, length, dimension, offset, base, false);
	}
	template <typename T>
	void BACKWARD_ROPE_ROTATE_HALF_LAST_TWO_TEMPLATE(const T *yg, T *xg,
		int length, int dimension, int offset, Scalar base)
	{
		BACKWARD_ROPE_LAST_TWO_TEMPLATE<T>(yg, xg, length, dimension, offset, base, true);
	}
}
