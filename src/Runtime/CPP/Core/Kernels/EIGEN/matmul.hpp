#pragma once

#include <cstddef>

#include <Eigen/Dense>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	namespace EigenMatmulDetail
	{
		template <typename T>
		using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

		// Eigen performs each dot product in Scalar so reduced-precision tensor
		// types keep the same accumulation precision as the NAIVE kernels.
		template <typename T>
		void multiply(const T *a, const T *b, T *c, int rows, int inputSize, int outputSize)
		{
			using ScalarMatrix = Matrix<Scalar>;
			using InputMap = Eigen::Map<const Matrix<T>>;
			using OutputMap = Eigen::Map<Matrix<T>>;

			const InputMap aMap(a, rows, inputSize);
			const InputMap bMap(b, inputSize, outputSize);
			OutputMap cMap(c, rows, outputSize);

			const ScalarMatrix result = aMap.template cast<Scalar>() * bMap.template cast<Scalar>();
			cMap = result.template cast<T>();
		}

		template <typename T>
		void backward(const T *a, const T *b, T *aGrad, T *bGrad,
			const T *cGrad, int rows, int inputSize, int outputSize)
		{
			using ScalarMatrix = Matrix<Scalar>;
			using InputMap = Eigen::Map<const Matrix<T>>;
			using GradMap = Eigen::Map<Matrix<T>>;

			const InputMap aMap(a, rows, inputSize);
			const InputMap bMap(b, inputSize, outputSize);
			const InputMap cGradMap(cGrad, rows, outputSize);
			GradMap aGradMap(aGrad, rows, inputSize);
			GradMap bGradMap(bGrad, inputSize, outputSize);

			const ScalarMatrix nextAGrad = aGradMap.template cast<Scalar>()
				+ cGradMap.template cast<Scalar>() * bMap.template cast<Scalar>().transpose();
			const ScalarMatrix nextBGrad = bGradMap.template cast<Scalar>()
				+ aMap.template cast<Scalar>().transpose() * cGradMap.template cast<Scalar>();

			aGradMap = nextAGrad.template cast<T>();
			bGradMap = nextBGrad.template cast<T>();
		}
	}

	template <typename T>
	void MATMUL_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b, T *c,
		int batchSize, int inputSize, int outputSize)
	{
		EigenMatmulDetail::multiply(a, b, c, batchSize, inputSize, outputSize);
	}

	template <typename T>
	void MATMUL_1B_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b, T *c,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const std::size_t aOffset = static_cast<std::size_t>(batch) * timeSize * inputSize;
			const std::size_t bOffset = static_cast<std::size_t>(batch) * inputSize * outputSize;
			const std::size_t cOffset = static_cast<std::size_t>(batch) * timeSize * outputSize;
			EigenMatmulDetail::multiply(a + aOffset, b + bOffset, c + cOffset,
				timeSize, inputSize, outputSize);
		}
	}

	template <typename T>
	void MATMUL_2B_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b, T *c,
		int batchSize, int headCount, int timeSize, int inputSize, int outputSize)
	{
		const int matrixCount = batchSize * headCount;
		for (int matrix = 0; matrix < matrixCount; ++matrix)
		{
			const std::size_t aOffset = static_cast<std::size_t>(matrix) * timeSize * inputSize;
			const std::size_t bOffset = static_cast<std::size_t>(matrix) * inputSize * outputSize;
			const std::size_t cOffset = static_cast<std::size_t>(matrix) * timeSize * outputSize;
			EigenMatmulDetail::multiply(a + aOffset, b + bOffset, c + cOffset,
				timeSize, inputSize, outputSize);
		}
	}

	template <typename T>
	void MATMUL_1B_2D_2D_LINEAR_TEMPLATE_EIGEN(const T *a, const T *b, T *c,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		EigenMatmulDetail::multiply(a, b, c, batchSize * timeSize, inputSize, outputSize);
	}

	template <typename T>
	void BACKWARD_MATMUL_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b,
		T *aGrad, T *bGrad, const T *cGrad, int batchSize, int inputSize, int outputSize)
	{
		EigenMatmulDetail::backward(a, b, aGrad, bGrad, cGrad,
			batchSize, inputSize, outputSize);
	}

	template <typename T>
	void BACKWARD_MATMUL_1B_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b,
		T *aGrad, T *bGrad, const T *cGrad,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const std::size_t aOffset = static_cast<std::size_t>(batch) * timeSize * inputSize;
			const std::size_t bOffset = static_cast<std::size_t>(batch) * inputSize * outputSize;
			const std::size_t cOffset = static_cast<std::size_t>(batch) * timeSize * outputSize;
			EigenMatmulDetail::backward(a + aOffset, b + bOffset,
				aGrad + aOffset, bGrad + bOffset, cGrad + cOffset,
				timeSize, inputSize, outputSize);
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_2B_2D_2D_TEMPLATE_EIGEN(const T *a, const T *b,
		T *aGrad, T *bGrad, const T *cGrad,
		int batchSize, int headCount, int timeSize, int inputSize, int outputSize)
	{
		const int matrixCount = batchSize * headCount;
		for (int matrix = 0; matrix < matrixCount; ++matrix)
		{
			const std::size_t aOffset = static_cast<std::size_t>(matrix) * timeSize * inputSize;
			const std::size_t bOffset = static_cast<std::size_t>(matrix) * inputSize * outputSize;
			const std::size_t cOffset = static_cast<std::size_t>(matrix) * timeSize * outputSize;
			EigenMatmulDetail::backward(a + aOffset, b + bOffset,
				aGrad + aOffset, bGrad + bOffset, cGrad + cOffset,
				timeSize, inputSize, outputSize);
		}
	}

	template <typename T>
	void BACKWARD_MATMUL_1B_2D_2D_LINEAR_TEMPLATE_EIGEN(const T *a, const T *b,
		T *aGrad, T *bGrad, const T *cGrad,
		int batchSize, int timeSize, int inputSize, int outputSize)
	{
		EigenMatmulDetail::backward(a, b, aGrad, bGrad, cGrad,
			batchSize * timeSize, inputSize, outputSize);
	}
}
