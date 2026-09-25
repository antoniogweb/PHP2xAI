#pragma once

#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void MULTIPLY_TEMPLATE(const T *a, const T *b, T *output, std::size_t count)
	{
		for (std::size_t i = 0; i < count; ++i)
			output[i] = static_cast<T>(
				static_cast<Scalar>(a[i]) * static_cast<Scalar>(b[i]));
	}

	template <typename T>
	void BACKWARD_MULTIPLY_TEMPLATE(
		const T *a, const T *b, T *aGrad, T *bGrad, const T *outputGrad,
		std::size_t count, bool aRequiresGrad, bool bRequiresGrad)
	{
		for (std::size_t i = 0; i < count; ++i)
		{
			const Scalar grad = static_cast<Scalar>(outputGrad[i]);
			if (aRequiresGrad)
				aGrad[i] = static_cast<T>(static_cast<Scalar>(aGrad[i]) + grad * b[i]);
			if (bRequiresGrad)
				bGrad[i] = static_cast<T>(static_cast<Scalar>(bGrad[i]) + grad * a[i]);
		}
	}
}
