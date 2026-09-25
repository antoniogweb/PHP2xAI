#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void CE_LAST_TEMPLATE(const T *prediction, const T *target, T *output,
		int rows, int classes)
	{
		const Scalar epsilon = 1.0e-12f;
		for (int row = 0; row < rows; ++row)
		{
			Scalar loss = 0.0f;
			for (int column = 0; column < classes; ++column)
			{
				const std::size_t index = static_cast<std::size_t>(row * classes + column);
				loss -= static_cast<Scalar>(target[index])
					* std::log(static_cast<Scalar>(prediction[index]) + epsilon);
			}
			output[row] = static_cast<T>(loss);
		}
	}

	template <typename T>
	void BACKWARD_CE_LAST_TEMPLATE(const T *prediction, const T *target,
		const T *outputGrad, T *predictionGrad, int rows, int classes,
		bool predictionNeedsGrad)
	{
		if (!predictionNeedsGrad) return;
		const Scalar epsilon = 1.0e-12f;
		for (int row = 0; row < rows; ++row)
			for (int column = 0; column < classes; ++column)
			{
				const std::size_t index = static_cast<std::size_t>(row * classes + column);
				const Scalar grad = -static_cast<Scalar>(target[index])
					/ (static_cast<Scalar>(prediction[index]) + epsilon)
					* static_cast<Scalar>(outputGrad[row]);
				predictionGrad[index] = static_cast<T>(
					static_cast<Scalar>(predictionGrad[index]) + grad);
			}
	}

	template <typename T>
	void CE_1D_LAST_TEMPLATE(const T *p, const T *t, T *o, int rows, int classes)
	{
		CE_LAST_TEMPLATE<T>(p, t, o, rows, classes);
	}
	template <typename T>
	void CE_2D_LAST_TEMPLATE(const T *p, const T *t, T *o, int rows, int classes)
	{
		CE_LAST_TEMPLATE<T>(p, t, o, rows, classes);
	}
	template <typename T>
	void CE_3D_LAST_TEMPLATE(const T *p, const T *t, T *o, int rows, int classes)
	{
		CE_LAST_TEMPLATE<T>(p, t, o, rows, classes);
	}
	template <typename T>
	void BACKWARD_CE_1D_LAST_TEMPLATE(const T *p, const T *t, const T *og,
		T *pg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LAST_TEMPLATE<T>(p, t, og, pg, rows, classes, needsGrad);
	}
	template <typename T>
	void BACKWARD_CE_2D_LAST_TEMPLATE(const T *p, const T *t, const T *og,
		T *pg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LAST_TEMPLATE<T>(p, t, og, pg, rows, classes, needsGrad);
	}
	template <typename T>
	void BACKWARD_CE_3D_LAST_TEMPLATE(const T *p, const T *t, const T *og,
		T *pg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LAST_TEMPLATE<T>(p, t, og, pg, rows, classes, needsGrad);
	}
}
