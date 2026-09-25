#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <Eigen/Dense>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void GELU_TEMPLATE_EIGEN(const T *input, T *output, std::size_t count)
	{
		if constexpr (std::is_floating_point<T>::value)
		{
			using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
			const Eigen::Map<const Array> x(input, static_cast<Eigen::Index>(count));
			Eigen::Map<Array> y(output, static_cast<Eigen::Index>(count));
			const T scale = static_cast<T>(std::sqrt(2.0 / 3.14159265358979323846));
			y = static_cast<T>(0.5) * x
				* (static_cast<T>(1) + (scale
					* (x + static_cast<T>(0.044715) * x.cube())).tanh());
		}
		else
		{
			const Scalar scale = std::sqrt(2.0f / 3.14159265358979323846f);
			for (std::size_t i = 0; i < count; ++i)
			{
				const Scalar x = static_cast<Scalar>(input[i]);
				const Scalar u = scale * (x + 0.044715f * x * x * x);
				output[i] = static_cast<T>(0.5f * x * (1.0f + std::tanh(u)));
			}
		}
	}

	template <typename T>
	void BACKWARD_GELU_TEMPLATE_EIGEN(const T *input, T *inputGrad,
		const T *outputGrad, std::size_t count)
	{
		if constexpr (std::is_floating_point<T>::value)
		{
			using Array = Eigen::Array<T, Eigen::Dynamic, 1>;
			const Eigen::Map<const Array> x(input, static_cast<Eigen::Index>(count));
			const Eigen::Map<const Array> grad(outputGrad, static_cast<Eigen::Index>(count));
			Eigen::Map<Array> xGrad(inputGrad, static_cast<Eigen::Index>(count));
			const T scale = static_cast<T>(std::sqrt(2.0 / 3.14159265358979323846));
			const Array tanhU = (scale
				* (x + static_cast<T>(0.044715) * x.cube())).tanh();
			xGrad += grad * (static_cast<T>(0.5) * (static_cast<T>(1) + tanhU)
				+ static_cast<T>(0.5) * x * (static_cast<T>(1) - tanhU.square())
					* scale * (static_cast<T>(1) + static_cast<T>(3 * 0.044715) * x.square()));
		}
		else
		{
			const Scalar scale = std::sqrt(2.0f / 3.14159265358979323846f);
			for (std::size_t i = 0; i < count; ++i)
			{
				const Scalar x = static_cast<Scalar>(input[i]);
				const Scalar xSquared = x * x;
				const Scalar u = scale * (x + 0.044715f * xSquared * x);
				const Scalar tanhU = std::tanh(u);
				const Scalar localGrad = 0.5f * (1.0f + tanhU)
					+ 0.5f * x * (1.0f - tanhU * tanhU)
						* scale * (1.0f + 3.0f * 0.044715f * xSquared);
				inputGrad[i] = static_cast<T>(static_cast<Scalar>(inputGrad[i])
					+ static_cast<Scalar>(outputGrad[i]) * localGrad);
			}
		}
	}
}
