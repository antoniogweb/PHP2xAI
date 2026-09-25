#pragma once

#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	// Equal-shape element-wise addition.
	template <typename T>
	void ADD_1D_LAST_TEMPLATE(
		const T *a,
		const T *b,
		T *c,
		std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			// Keep arithmetic intermediates in Scalar. The stored result uses T.
			const Scalar sum = static_cast<Scalar>(a[i]) + static_cast<Scalar>(b[i]);
			c[i] = static_cast<T>(sum);
		}
	}

	// Add a rank-1 bias to every row of a rank-2 tensor.
	template <typename T>
	void ADD_2D_LAST_TEMPLATE(
		const T *a,
		const T *bias,
		T *c,
		int batchSize,
		int featureCount)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const int rowStart = batch * featureCount;
			for (int feature = 0; feature < featureCount; ++feature)
			{
				const std::size_t tensorIndex = static_cast<std::size_t>(rowStart + feature);
				const Scalar sum = static_cast<Scalar>(a[tensorIndex])
					+ static_cast<Scalar>(bias[feature]);
				c[tensorIndex] = static_cast<T>(sum);
			}
		}
	}

	// Add a rank-1 bias to every row of a rank-3 tensor.
	template <typename T>
	void ADD_3D_LAST_TEMPLATE(
		const T *a,
		const T *bias,
		T *c,
		int batchSize,
		int timeSize,
		int featureCount)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				const int rowStart = (batch * timeSize + time) * featureCount;
				for (int feature = 0; feature < featureCount; ++feature)
				{
					const std::size_t tensorIndex = static_cast<std::size_t>(rowStart + feature);
					const Scalar sum = static_cast<Scalar>(a[tensorIndex])
						+ static_cast<Scalar>(bias[feature]);
					c[tensorIndex] = static_cast<T>(sum);
				}
			}
		}
	}

	// Equal-shape addition sends the output gradient to both inputs.
	template <typename T>
	void BACKWARD_ADD_1D_LAST_TEMPLATE(
		T *aGrad,
		T *bGrad,
		const T *cGrad,
		std::size_t elementCount,
		bool aRequiresGrad,
		bool bRequiresGrad)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar outputGrad = static_cast<Scalar>(cGrad[i]);
			if (aRequiresGrad)
				aGrad[i] = static_cast<T>(static_cast<Scalar>(aGrad[i]) + outputGrad);
			if (bRequiresGrad)
				bGrad[i] = static_cast<T>(static_cast<Scalar>(bGrad[i]) + outputGrad);
		}
	}

	// Backward for a rank-2 tensor plus rank-1 bias. Bias gradients reduce
	// across the batch using Scalar precision, then are stored in T once.
	template <typename T>
	void BACKWARD_ADD_2D_LAST_TEMPLATE(
		T *aGrad,
		T *bGrad,
		const T *cGrad,
		int batchSize,
		int featureCount,
		bool aRequiresGrad,
		bool bRequiresGrad)
	{
		std::vector<Scalar> biasGrad;
		if (bRequiresGrad)
		{
			biasGrad.resize(static_cast<std::size_t>(featureCount));
			for (int feature = 0; feature < featureCount; ++feature)
				biasGrad[static_cast<std::size_t>(feature)] = static_cast<Scalar>(bGrad[feature]);
		}

		for (int batch = 0; batch < batchSize; ++batch)
		{
			const int rowStart = batch * featureCount;
			for (int feature = 0; feature < featureCount; ++feature)
			{
				const std::size_t tensorIndex = static_cast<std::size_t>(rowStart + feature);
				const Scalar outputGrad = static_cast<Scalar>(cGrad[tensorIndex]);
				if (aRequiresGrad)
				{
					aGrad[tensorIndex] = static_cast<T>(
						static_cast<Scalar>(aGrad[tensorIndex]) + outputGrad);
				}
				if (bRequiresGrad)
					biasGrad[static_cast<std::size_t>(feature)] += outputGrad;
			}
		}

		if (bRequiresGrad)
		{
			for (int feature = 0; feature < featureCount; ++feature)
				bGrad[feature] = static_cast<T>(biasGrad[static_cast<std::size_t>(feature)]);
		}
	}

	// Backward for a rank-3 tensor plus rank-1 bias. Both batch and time axes
	// contribute to each bias gradient, accumulated in Scalar precision.
	template <typename T>
	void BACKWARD_ADD_3D_LAST_TEMPLATE(
		T *aGrad,
		T *bGrad,
		const T *cGrad,
		int batchSize,
		int timeSize,
		int featureCount,
		bool aRequiresGrad,
		bool bRequiresGrad)
	{
		std::vector<Scalar> biasGrad;
		if (bRequiresGrad)
		{
			biasGrad.resize(static_cast<std::size_t>(featureCount));
			for (int feature = 0; feature < featureCount; ++feature)
				biasGrad[static_cast<std::size_t>(feature)] = static_cast<Scalar>(bGrad[feature]);
		}

		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				const int rowStart = (batch * timeSize + time) * featureCount;
				for (int feature = 0; feature < featureCount; ++feature)
				{
					const std::size_t tensorIndex = static_cast<std::size_t>(rowStart + feature);
					const Scalar outputGrad = static_cast<Scalar>(cGrad[tensorIndex]);
					if (aRequiresGrad)
					{
						aGrad[tensorIndex] = static_cast<T>(
							static_cast<Scalar>(aGrad[tensorIndex]) + outputGrad);
					}
					if (bRequiresGrad)
						biasGrad[static_cast<std::size_t>(feature)] += outputGrad;
				}
			}
		}

		if (bRequiresGrad)
		{
			for (int feature = 0; feature < featureCount; ++feature)
				bGrad[feature] = static_cast<T>(biasGrad[static_cast<std::size_t>(feature)]);
		}
	}

}
