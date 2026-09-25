#pragma once

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <limits>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void CE_LOGITS_GENERIC_AXIS_TEMPLATE(const T *logits, const T *target,
		T *output, std::size_t outer, std::size_t inner, int axisSize)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar maximum = -std::numeric_limits<Scalar>::infinity();
				Scalar targetSum = 0.0f;
				Scalar targetLogit = 0.0f;
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					maximum = std::max(maximum, static_cast<Scalar>(logits[index]));
					targetSum += static_cast<Scalar>(target[index]);
					targetLogit += static_cast<Scalar>(target[index])
						* static_cast<Scalar>(logits[index]);
				}

				Scalar sumExp = 0.0f;
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					sumExp += std::exp(static_cast<Scalar>(logits[index]) - maximum);
				}
				output[outerIndex * inner + innerIndex] = static_cast<T>(
					maximum + std::log(sumExp) * targetSum - targetLogit);
			}
		}
	}

	template <typename T>
	void BACKWARD_CE_LOGITS_GENERIC_AXIS_TEMPLATE(const T *logits, const T *target,
		const T *outputGrad, T *logitsGrad, std::size_t outer,
		std::size_t inner, int axisSize, bool logitsNeedGrad)
	{
		if (!logitsNeedGrad)
			return;
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar maximum = -std::numeric_limits<Scalar>::infinity();
				Scalar targetSum = 0.0f;
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					maximum = std::max(maximum, static_cast<Scalar>(logits[index]));
					targetSum += static_cast<Scalar>(target[index]);
				}
				Scalar sumExp = 0.0f;
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					sumExp += std::exp(static_cast<Scalar>(logits[index]) - maximum);
				}
				const Scalar outputGradient = static_cast<Scalar>(
					outputGrad[outerIndex * inner + innerIndex]);
				for (int axisIndex = 0; axisIndex < axisSize; ++axisIndex)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(axisSize)
							+ static_cast<std::size_t>(axisIndex)) * inner + innerIndex;
					const Scalar probability =
						std::exp(static_cast<Scalar>(logits[index]) - maximum) / sumExp;
					const Scalar contribution = (probability * targetSum
						- static_cast<Scalar>(target[index])) * outputGradient;
					logitsGrad[index] = static_cast<T>(
						static_cast<Scalar>(logitsGrad[index]) + contribution);
				}
			}
		}
	}

	template <typename T>
	void CE_LOGITS_LAST_TEMPLATE(const T *logits, const T *target, T *output,
		int rows, int classes)
	{
		for (int row = 0; row < rows; ++row)
		{
			const std::size_t base = static_cast<std::size_t>(row * classes);
			Scalar maximum = static_cast<Scalar>(logits[base]);
			for (int i = 1; i < classes; ++i)
				maximum = std::max(maximum, static_cast<Scalar>(logits[base + i]));
			Scalar sumExp = 0.0f;
			Scalar targetSum = 0.0f;
			Scalar targetLogit = 0.0f;
			for (int i = 0; i < classes; ++i)
			{
				const Scalar t = static_cast<Scalar>(target[base + i]);
				sumExp += std::exp(static_cast<Scalar>(logits[base + i]) - maximum);
				targetSum += t;
				targetLogit += t * static_cast<Scalar>(logits[base + i]);
			}
			output[row] = static_cast<T>(maximum + std::log(sumExp) * targetSum - targetLogit);
		}
	}

	template <typename T>
	void BACKWARD_CE_LOGITS_LAST_TEMPLATE(const T *logits, const T *target,
		const T *outputGrad, T *logitsGrad, int rows, int classes,
		bool logitsNeedsGrad)
	{
		if (!logitsNeedsGrad) return;
		for (int row = 0; row < rows; ++row)
		{
			const std::size_t base = static_cast<std::size_t>(row * classes);
			Scalar maximum = static_cast<Scalar>(logits[base]);
			for (int i = 1; i < classes; ++i)
				maximum = std::max(maximum, static_cast<Scalar>(logits[base + i]));
			Scalar sumExp = 0.0f;
			for (int i = 0; i < classes; ++i)
				sumExp += std::exp(static_cast<Scalar>(logits[base + i]) - maximum);
			Scalar targetSum = 0.0f;
			for (int i = 0; i < classes; ++i)
				targetSum += static_cast<Scalar>(target[base + i]);
			for (int i = 0; i < classes; ++i)
			{
				const Scalar probability = std::exp(static_cast<Scalar>(logits[base + i]) - maximum) / sumExp;
				const Scalar grad = (probability * targetSum - static_cast<Scalar>(target[base + i]))
					* static_cast<Scalar>(outputGrad[row]);
				logitsGrad[base + i] = static_cast<T>(static_cast<Scalar>(logitsGrad[base + i]) + grad);
			}
		}
	}

	template <typename T>
	void CE_LOGITS_1D_LAST_TEMPLATE(const T *x, const T *t, T *o, int rows, int classes)
	{
		CE_LOGITS_LAST_TEMPLATE<T>(x, t, o, rows, classes);
	}
	template <typename T>
	void CE_LOGITS_2D_LAST_TEMPLATE(const T *x, const T *t, T *o, int rows, int classes)
	{
		CE_LOGITS_LAST_TEMPLATE<T>(x, t, o, rows, classes);
	}
	template <typename T>
	void CE_LOGITS_3D_LAST_TEMPLATE(const T *x, const T *t, T *o, int rows, int classes)
	{
		CE_LOGITS_LAST_TEMPLATE<T>(x, t, o, rows, classes);
	}
	template <typename T>
	void BACKWARD_CE_LOGITS_1D_LAST_TEMPLATE(const T *x, const T *t, const T *og,
		T *xg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LOGITS_LAST_TEMPLATE<T>(x, t, og, xg, rows, classes, needsGrad);
	}
	template <typename T>
	void BACKWARD_CE_LOGITS_2D_LAST_TEMPLATE(const T *x, const T *t, const T *og,
		T *xg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LOGITS_LAST_TEMPLATE<T>(x, t, og, xg, rows, classes, needsGrad);
	}
	template <typename T>
	void BACKWARD_CE_LOGITS_3D_LAST_TEMPLATE(const T *x, const T *t, const T *og,
		T *xg, int rows, int classes, bool needsGrad)
	{
		BACKWARD_CE_LOGITS_LAST_TEMPLATE<T>(x, t, og, xg, rows, classes, needsGrad);
	}
}
