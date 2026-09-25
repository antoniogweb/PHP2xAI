#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
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
