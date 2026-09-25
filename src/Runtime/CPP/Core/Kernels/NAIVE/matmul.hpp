#pragma once

#include <cstddef>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
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
