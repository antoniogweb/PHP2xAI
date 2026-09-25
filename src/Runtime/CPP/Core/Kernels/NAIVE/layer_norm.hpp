#pragma once

#include <cstddef>
#include <cmath>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void LAYER_NORM_GENERIC_TEMPLATE(const T *x, const T *gamma, const T *beta,
		T *y, std::size_t outer, std::size_t inner, int width)
	{
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar mean = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					mean += static_cast<Scalar>(x[index]);
				}
				mean /= static_cast<Scalar>(width);
				Scalar variance = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar centered = static_cast<Scalar>(x[index]) - mean;
					variance += centered * centered;
				}
				const Scalar inverseStd = 1.0f / std::sqrt(variance / width + 1.0e-5f);
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar normalized = (static_cast<Scalar>(x[index]) - mean) * inverseStd;
					y[index] = static_cast<T>(normalized * static_cast<Scalar>(gamma[i])
						+ static_cast<Scalar>(beta[i]));
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_LAYER_NORM_GENERIC_TEMPLATE(const T *x, const T *gamma,
		const T *yGrad, T *xGrad, T *gammaGrad, T *betaGrad,
		std::size_t outer, std::size_t inner, int width,
		bool xNeedsGrad, bool gammaNeedsGrad, bool betaNeedsGrad)
	{
		std::vector<Scalar> gammaAccum(static_cast<std::size_t>(width), 0.0f);
		std::vector<Scalar> betaAccum(static_cast<std::size_t>(width), 0.0f);
		for (int i = 0; i < width; ++i)
		{
			if (gammaNeedsGrad)
				gammaAccum[static_cast<std::size_t>(i)] = static_cast<Scalar>(gammaGrad[i]);
			if (betaNeedsGrad)
				betaAccum[static_cast<std::size_t>(i)] = static_cast<Scalar>(betaGrad[i]);
		}
		for (std::size_t outerIndex = 0; outerIndex < outer; ++outerIndex)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				Scalar mean = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					mean += static_cast<Scalar>(x[index]);
				}
				mean /= static_cast<Scalar>(width);
				Scalar variance = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar centered = static_cast<Scalar>(x[index]) - mean;
					variance += centered * centered;
				}
				const Scalar inverseStd = 1.0f / std::sqrt(variance / width + 1.0e-5f);
				Scalar sumGrad = 0.0f;
				Scalar sumGradNormalized = 0.0f;
				for (int i = 0; i < width; ++i)
				{
					const std::size_t index =
						(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
					const Scalar normalized = (static_cast<Scalar>(x[index]) - mean) * inverseStd;
					const Scalar grad = static_cast<Scalar>(yGrad[index]);
					const Scalar scaledGrad = grad * static_cast<Scalar>(gamma[i]);
					sumGrad += scaledGrad;
					sumGradNormalized += scaledGrad * normalized;
					if (gammaNeedsGrad)
						gammaAccum[static_cast<std::size_t>(i)] += grad * normalized;
					if (betaNeedsGrad)
						betaAccum[static_cast<std::size_t>(i)] += grad;
				}
				if (xNeedsGrad)
				{
					for (int i = 0; i < width; ++i)
					{
						const std::size_t index =
							(outerIndex * static_cast<std::size_t>(width) + i) * inner + innerIndex;
						const Scalar normalized = (static_cast<Scalar>(x[index]) - mean) * inverseStd;
						const Scalar scaledGrad = static_cast<Scalar>(yGrad[index])
							* static_cast<Scalar>(gamma[i]);
						const Scalar contribution = inverseStd * (width * scaledGrad - sumGrad
							- normalized * sumGradNormalized) / width;
						xGrad[index] = static_cast<T>(static_cast<Scalar>(xGrad[index]) + contribution);
					}
				}
			}
		}
		if (gammaNeedsGrad)
			for (int i = 0; i < width; ++i)
				gammaGrad[i] = static_cast<T>(gammaAccum[static_cast<std::size_t>(i)]);
		if (betaNeedsGrad)
			for (int i = 0; i < width; ++i)
				betaGrad[i] = static_cast<T>(betaAccum[static_cast<std::size_t>(i)]);
	}

	template <typename T>
	void LAYER_NORM_LAST_AXIS_TEMPLATE(const T *x, const T *gamma, const T *beta,
		T *y, std::size_t outer, int width)
	{
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t base = row * static_cast<std::size_t>(width);
			Scalar mean = 0.0f;
			for (int i = 0; i < width; ++i) mean += static_cast<Scalar>(x[base + i]);
			mean /= static_cast<Scalar>(width);
			Scalar variance = 0.0f;
			for (int i = 0; i < width; ++i)
			{
				const Scalar centered = static_cast<Scalar>(x[base + i]) - mean;
				variance += centered * centered;
			}
			const Scalar inverseStd = 1.0f / std::sqrt(variance / width + 1.0e-5f);
			for (int i = 0; i < width; ++i)
			{
				const Scalar normalized = (static_cast<Scalar>(x[base + i]) - mean) * inverseStd;
				y[base + i] = static_cast<T>(normalized * gamma[i] + beta[i]);
			}
		}
	}

	template <typename T>
	void BACKWARD_LAYER_NORM_LAST_AXIS_TEMPLATE(const T *x, const T *gamma,
		const T *y, const T *yGrad, T *xGrad, T *gammaGrad, T *betaGrad,
		std::size_t outer, int width, bool xNeedsGrad, bool gammaNeedsGrad,
		bool betaNeedsGrad)
	{
		std::vector<Scalar> gammaAccum(static_cast<std::size_t>(width), 0.0f);
		std::vector<Scalar> betaAccum(static_cast<std::size_t>(width), 0.0f);
		for (std::size_t row = 0; row < outer; ++row)
		{
			const std::size_t base = row * static_cast<std::size_t>(width);
			Scalar mean = 0.0f;
			for (int i = 0; i < width; ++i) mean += static_cast<Scalar>(x[base + i]);
			mean /= static_cast<Scalar>(width);
			Scalar variance = 0.0f;
			for (int i = 0; i < width; ++i)
			{
				const Scalar centered = static_cast<Scalar>(x[base + i]) - mean;
				variance += centered * centered;
			}
			const Scalar inverseStd = 1.0f / std::sqrt(variance / width + 1.0e-5f);
			Scalar sumGrad = 0.0f;
			Scalar sumGradNormalized = 0.0f;
			for (int i = 0; i < width; ++i)
			{
				const Scalar normalized = (static_cast<Scalar>(x[base + i]) - mean) * inverseStd;
				const Scalar grad = static_cast<Scalar>(yGrad[base + i]);
				const Scalar scaledGrad = grad * static_cast<Scalar>(gamma[i]);
				sumGrad += scaledGrad;
				sumGradNormalized += scaledGrad * normalized;
				gammaAccum[static_cast<std::size_t>(i)] += grad * normalized;
				betaAccum[static_cast<std::size_t>(i)] += grad;
			}
			if (xNeedsGrad)
				for (int i = 0; i < width; ++i)
				{
					const Scalar normalized = (static_cast<Scalar>(x[base + i]) - mean) * inverseStd;
					const Scalar scaledGrad = static_cast<Scalar>(yGrad[base + i]) * gamma[i];
					const Scalar grad = inverseStd * (width * scaledGrad - sumGrad
						- normalized * sumGradNormalized) / width;
					xGrad[base + i] = static_cast<T>(static_cast<Scalar>(xGrad[base + i]) + grad);
				}
		}
		if (gammaNeedsGrad)
			for (int i = 0; i < width; ++i)
				gammaGrad[i] = static_cast<T>(static_cast<Scalar>(gammaGrad[i]) + gammaAccum[i]);
		if (betaNeedsGrad)
			for (int i = 0; i < width; ++i)
				betaGrad[i] = static_cast<T>(static_cast<Scalar>(betaGrad[i]) + betaAccum[i]);
	}
}
