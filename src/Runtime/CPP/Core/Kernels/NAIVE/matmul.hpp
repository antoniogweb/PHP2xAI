#pragma once

#include <cstddef>
#include <vector>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	inline std::vector<std::size_t> contiguousStrides(const std::vector<int> &shape)
	{
		std::vector<std::size_t> strides(shape.size());
		std::size_t stride = 1;
		for (std::size_t axis = shape.size(); axis > 0; --axis)
		{
			strides[axis - 1] = stride;
			stride *= static_cast<std::size_t>(shape[axis - 1]);
		}
		return strides;
	}

	template <typename T>
	void MATMUL_GENERIC_B_2D_2D_BROADCAST_TEMPLATE(
		const T *a, const T *b, T *c,
		const std::vector<int> &aShape, const std::vector<int> &bShape,
		const std::vector<int> &cShape)
	{
		const std::size_t rankA = aShape.size();
		const std::size_t rankB = bShape.size();
		const std::size_t batchRank = cShape.size() - 2;
		const int rows = aShape[rankA - 2];
		const int innerSize = aShape[rankA - 1];
		const int columns = bShape[rankB - 1];
		const std::vector<std::size_t> aStrides = contiguousStrides(aShape);
		const std::vector<std::size_t> bStrides = contiguousStrides(bShape);
		const std::vector<std::size_t> cStrides = contiguousStrides(cShape);
		std::vector<int> batchShape(batchRank, 1);
		std::vector<std::size_t> aBatchStrides(batchRank, 0);
		std::vector<std::size_t> bBatchStrides(batchRank, 0);

		for (std::size_t axis = 0; axis < batchRank; ++axis)
		{
			const std::size_t aOffset = batchRank - (rankA - 2);
			const std::size_t bOffset = batchRank - (rankB - 2);
			const bool hasA = axis >= aOffset;
			const bool hasB = axis >= bOffset;
			const int aDim = hasA ? aShape[axis - aOffset] : 1;
			const int bDim = hasB ? bShape[axis - bOffset] : 1;
			batchShape[axis] = aDim > bDim ? aDim : bDim;
			if (hasA && aDim != 1)
				aBatchStrides[axis] = aStrides[axis - aOffset];
			if (hasB && bDim != 1)
				bBatchStrides[axis] = bStrides[axis - bOffset];
		}

		std::size_t batchCount = 1;
		for (std::size_t axis = 0; axis < batchRank; ++axis)
			batchCount *= static_cast<std::size_t>(batchShape[axis]);

		for (std::size_t batch = 0; batch < batchCount; ++batch)
		{
			std::size_t remaining = batch;
			std::size_t baseA = 0;
			std::size_t baseB = 0;
			std::size_t baseC = 0;
			for (std::size_t axis = batchRank; axis > 0; --axis)
			{
				const std::size_t current = axis - 1;
				const std::size_t coordinate = remaining
					% static_cast<std::size_t>(batchShape[current]);
				remaining /= static_cast<std::size_t>(batchShape[current]);
				baseA += coordinate * aBatchStrides[current];
				baseB += coordinate * bBatchStrides[current];
				baseC += coordinate * cStrides[current];
			}

			for (int row = 0; row < rows; ++row)
			{
				for (int column = 0; column < columns; ++column)
				{
					Scalar sum = 0.0f;
					for (int inner = 0; inner < innerSize; ++inner)
					{
						const std::size_t aIndex = baseA
							+ static_cast<std::size_t>(row) * aStrides[rankA - 2]
							+ static_cast<std::size_t>(inner) * aStrides[rankA - 1];
						const std::size_t bIndex = baseB
							+ static_cast<std::size_t>(inner) * bStrides[rankB - 2]
							+ static_cast<std::size_t>(column) * bStrides[rankB - 1];
						sum += static_cast<Scalar>(a[aIndex]) * static_cast<Scalar>(b[bIndex]);
					}
					const std::size_t cIndex = baseC
						+ static_cast<std::size_t>(row) * cStrides[batchRank]
						+ static_cast<std::size_t>(column) * cStrides[batchRank + 1];
					c[cIndex] = static_cast<T>(sum);
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST_TEMPLATE(
		const T *a, const T *b, T *aGrad, T *bGrad, const T *cGrad,
		const std::vector<int> &aShape, const std::vector<int> &bShape,
		const std::vector<int> &cShape, bool aNeedsGrad, bool bNeedsGrad)
	{
		const std::size_t rankA = aShape.size();
		const std::size_t rankB = bShape.size();
		const std::size_t batchRank = cShape.size() - 2;
		const int rows = aShape[rankA - 2];
		const int innerSize = aShape[rankA - 1];
		const int columns = bShape[rankB - 1];
		const std::vector<std::size_t> aStrides = contiguousStrides(aShape);
		const std::vector<std::size_t> bStrides = contiguousStrides(bShape);
		const std::vector<std::size_t> cStrides = contiguousStrides(cShape);
		std::vector<int> batchShape(batchRank, 1);
		std::vector<std::size_t> aBatchStrides(batchRank, 0);
		std::vector<std::size_t> bBatchStrides(batchRank, 0);
		for (std::size_t axis = 0; axis < batchRank; ++axis)
		{
			const std::size_t aOffset = batchRank - (rankA - 2);
			const std::size_t bOffset = batchRank - (rankB - 2);
			const bool hasA = axis >= aOffset;
			const bool hasB = axis >= bOffset;
			const int aDim = hasA ? aShape[axis - aOffset] : 1;
			const int bDim = hasB ? bShape[axis - bOffset] : 1;
			batchShape[axis] = aDim > bDim ? aDim : bDim;
			if (hasA && aDim != 1)
				aBatchStrides[axis] = aStrides[axis - aOffset];
			if (hasB && bDim != 1)
				bBatchStrides[axis] = bStrides[axis - bOffset];
		}

		std::size_t batchCount = 1;
		for (std::size_t axis = 0; axis < batchRank; ++axis)
			batchCount *= static_cast<std::size_t>(batchShape[axis]);
		std::size_t aSize = 1;
		std::size_t bSize = 1;
		for (std::size_t axis = 0; axis < rankA; ++axis)
			aSize *= static_cast<std::size_t>(aShape[axis]);
		for (std::size_t axis = 0; axis < rankB; ++axis)
			bSize *= static_cast<std::size_t>(bShape[axis]);
		std::vector<Scalar> accumulatedA(aSize, 0.0f);
		std::vector<Scalar> accumulatedB(bSize, 0.0f);
		if (aNeedsGrad)
			for (std::size_t i = 0; i < aSize; ++i)
				accumulatedA[i] = static_cast<Scalar>(aGrad[i]);
		if (bNeedsGrad)
			for (std::size_t i = 0; i < bSize; ++i)
				accumulatedB[i] = static_cast<Scalar>(bGrad[i]);

		for (std::size_t batch = 0; batch < batchCount; ++batch)
		{
			std::size_t remaining = batch;
			std::size_t baseA = 0;
			std::size_t baseB = 0;
			std::size_t baseC = 0;
			for (std::size_t axis = batchRank; axis > 0; --axis)
			{
				const std::size_t current = axis - 1;
				const std::size_t coordinate = remaining
					% static_cast<std::size_t>(batchShape[current]);
				remaining /= static_cast<std::size_t>(batchShape[current]);
				baseA += coordinate * aBatchStrides[current];
				baseB += coordinate * bBatchStrides[current];
				baseC += coordinate * cStrides[current];
			}

			for (int row = 0; row < rows; ++row)
			{
				for (int column = 0; column < columns; ++column)
				{
					const std::size_t gradientIndex = baseC
						+ static_cast<std::size_t>(row) * cStrides[batchRank]
						+ static_cast<std::size_t>(column) * cStrides[batchRank + 1];
					const Scalar gradient = static_cast<Scalar>(cGrad[gradientIndex]);
					for (int inner = 0; inner < innerSize; ++inner)
					{
						const std::size_t aIndex = baseA
							+ static_cast<std::size_t>(row) * aStrides[rankA - 2]
							+ static_cast<std::size_t>(inner) * aStrides[rankA - 1];
						const std::size_t bIndex = baseB
							+ static_cast<std::size_t>(inner) * bStrides[rankB - 2]
							+ static_cast<std::size_t>(column) * bStrides[rankB - 1];
						if (aNeedsGrad)
							accumulatedA[aIndex] += gradient * static_cast<Scalar>(b[bIndex]);
						if (bNeedsGrad)
							accumulatedB[bIndex] += static_cast<Scalar>(a[aIndex]) * gradient;
					}
				}
			}
		}

		if (aNeedsGrad)
			for (std::size_t i = 0; i < aSize; ++i)
				aGrad[i] = static_cast<T>(accumulatedA[i]);
		if (bNeedsGrad)
			for (std::size_t i = 0; i < bSize; ++i)
				bGrad[i] = static_cast<T>(accumulatedB[i]);
	}

	// Matrix multiplication over the final dimension of A and the first
	// dimension of B: [batch, K] x [K, N] -> [batch, N].
	template <typename T>
	void MATMUL_2D_2D_TEMPLATE(const T *a, const T *b, T *c,
		int batchSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int output = 0; output < outputSize; ++output)
			{
				Scalar sum = 0;
				for (int input = 0; input < inputSize; ++input)
				{
					const std::size_t aIndex = static_cast<std::size_t>(batch * inputSize + input);
					const std::size_t bIndex = static_cast<std::size_t>(input * outputSize + output);
					sum += static_cast<Scalar>(a[aIndex]) * static_cast<Scalar>(b[bIndex]);
				}
				const std::size_t cIndex = static_cast<std::size_t>(batch * outputSize + output);
				c[cIndex] = static_cast<T>(sum);
			}
		}
	}

	// Batched matrix multiplication: [B, T, K] x [B, K, N].
	template <typename T>
	void MATMUL_1B_2D_2D_TEMPLATE(const T *a, const T *b, T *c,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				for (int output = 0; output < outputSize; ++output)
				{
					Scalar sum = 0;
					for (int input = 0; input < inputSize; ++input)
					{
						const std::size_t aIndex = static_cast<std::size_t>((batch * timeSize + time) * inputSize + input);
						const std::size_t bIndex = static_cast<std::size_t>((batch * inputSize + input) * outputSize + output);
						sum += static_cast<Scalar>(a[aIndex]) * static_cast<Scalar>(b[bIndex]);
					}
					const std::size_t cIndex = static_cast<std::size_t>((batch * timeSize + time) * outputSize + output);
					c[cIndex] = static_cast<T>(sum);
				}
			}
		}
	}

	// Batched matrix multiplication with batch and head dimensions:
	// [B, H, T, K] x [B, H, K, N].
	template <typename T>
	void MATMUL_2B_2D_2D_TEMPLATE(const T *a, const T *b, T *c,
		int batchSize, int headCount, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int head = 0; head < headCount; ++head)
			{
				for (int time = 0; time < timeSize; ++time)
				{
					for (int output = 0; output < outputSize; ++output)
					{
						Scalar sum = 0;
						for (int input = 0; input < inputSize; ++input)
						{
							const std::size_t aIndex = static_cast<std::size_t>(((batch * headCount + head) * timeSize + time) * inputSize + input);
							const std::size_t bIndex = static_cast<std::size_t>(((batch * headCount + head) * inputSize + input) * outputSize + output);
							sum += static_cast<Scalar>(a[aIndex]) * static_cast<Scalar>(b[bIndex]);
						}
						const std::size_t cIndex = static_cast<std::size_t>(((batch * headCount + head) * timeSize + time) * outputSize + output);
						c[cIndex] = static_cast<T>(sum);
					}
				}
			}
		}
	}

	// Apply one [K, N] weight matrix independently to every [T, K] batch.
	template <typename T>
	void MATMUL_1B_2D_2D_LINEAR_TEMPLATE(const T *a, const T *b, T *c,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				for (int output = 0; output < outputSize; ++output)
				{
					Scalar sum = 0;
					for (int input = 0; input < inputSize; ++input)
					{
						const std::size_t aIndex = static_cast<std::size_t>((batch * timeSize + time) * inputSize + input);
						const std::size_t bIndex = static_cast<std::size_t>(input * outputSize + output);
						sum += static_cast<Scalar>(a[aIndex]) * static_cast<Scalar>(b[bIndex]);
					}
					const std::size_t cIndex = static_cast<std::size_t>((batch * timeSize + time) * outputSize + output);
					c[cIndex] = static_cast<T>(sum);
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_2D_2D_TEMPLATE(const T *a, const T *b, T *aGrad, T *bGrad,
		const T *cGrad, int batchSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int input = 0; input < inputSize; ++input)
			{
				const std::size_t aIndex = static_cast<std::size_t>(batch * inputSize + input);
				Scalar gradA = static_cast<Scalar>(aGrad[aIndex]);
				for (int output = 0; output < outputSize; ++output)
				{
					const std::size_t bIndex = static_cast<std::size_t>(input * outputSize + output);
					const std::size_t cIndex = static_cast<std::size_t>(batch * outputSize + output);
					const Scalar grad = static_cast<Scalar>(cGrad[cIndex]);
					gradA += grad * static_cast<Scalar>(b[bIndex]);
					bGrad[bIndex] = static_cast<T>(static_cast<Scalar>(bGrad[bIndex])
						+ static_cast<Scalar>(a[aIndex]) * grad);
				}
				aGrad[aIndex] = static_cast<T>(gradA);
			}
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_1B_2D_2D_TEMPLATE(const T *a, const T *b, T *aGrad, T *bGrad,
		const T *cGrad, int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				for (int input = 0; input < inputSize; ++input)
				{
					const std::size_t aIndex = static_cast<std::size_t>((batch * timeSize + time) * inputSize + input);
					Scalar gradA = static_cast<Scalar>(aGrad[aIndex]);
					for (int output = 0; output < outputSize; ++output)
					{
						const std::size_t bIndex = static_cast<std::size_t>((batch * inputSize + input) * outputSize + output);
						const std::size_t cIndex = static_cast<std::size_t>((batch * timeSize + time) * outputSize + output);
						const Scalar grad = static_cast<Scalar>(cGrad[cIndex]);
						gradA += grad * static_cast<Scalar>(b[bIndex]);
						bGrad[bIndex] = static_cast<T>(static_cast<Scalar>(bGrad[bIndex])
							+ static_cast<Scalar>(a[aIndex]) * grad);
					}
					aGrad[aIndex] = static_cast<T>(gradA);
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_2B_2D_2D_TEMPLATE(const T *a, const T *b, T *aGrad, T *bGrad,
		const T *cGrad, int batchSize, int headCount, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int head = 0; head < headCount; ++head)
			{
				for (int time = 0; time < timeSize; ++time)
				{
					for (int input = 0; input < inputSize; ++input)
					{
						const std::size_t aIndex = static_cast<std::size_t>(((batch * headCount + head) * timeSize + time) * inputSize + input);
						Scalar gradA = static_cast<Scalar>(aGrad[aIndex]);
						for (int output = 0; output < outputSize; ++output)
						{
							const std::size_t bIndex = static_cast<std::size_t>(((batch * headCount + head) * inputSize + input) * outputSize + output);
							const std::size_t cIndex = static_cast<std::size_t>(((batch * headCount + head) * timeSize + time) * outputSize + output);
							const Scalar grad = static_cast<Scalar>(cGrad[cIndex]);
							gradA += grad * static_cast<Scalar>(b[bIndex]);
							bGrad[bIndex] = static_cast<T>(static_cast<Scalar>(bGrad[bIndex])
								+ static_cast<Scalar>(a[aIndex]) * grad);
						}
						aGrad[aIndex] = static_cast<T>(gradA);
					}
				}
			}
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_1B_2D_2D_LINEAR_TEMPLATE(const T *a, const T *b, T *aGrad, T *bGrad,
		const T *cGrad, int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			for (int time = 0; time < timeSize; ++time)
			{
				for (int input = 0; input < inputSize; ++input)
				{
					const std::size_t aIndex = static_cast<std::size_t>((batch * timeSize + time) * inputSize + input);
					Scalar gradA = static_cast<Scalar>(aGrad[aIndex]);
					for (int output = 0; output < outputSize; ++output)
					{
						const std::size_t bIndex = static_cast<std::size_t>(input * outputSize + output);
						const std::size_t cIndex = static_cast<std::size_t>((batch * timeSize + time) * outputSize + output);
						const Scalar grad = static_cast<Scalar>(cGrad[cIndex]);
						gradA += grad * static_cast<Scalar>(b[bIndex]);
						bGrad[bIndex] = static_cast<T>(static_cast<Scalar>(bGrad[bIndex])
							+ static_cast<Scalar>(a[aIndex]) * grad);
					}
					aGrad[aIndex] = static_cast<T>(gradA);
				}
			}
		}
	}
}
