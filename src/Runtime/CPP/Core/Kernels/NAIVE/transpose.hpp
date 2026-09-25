#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T>
	void TRANSPOSE_TEMPLATE(const T *input, T *output,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		std::vector<int> outputShape(shape.size());
		std::vector<std::size_t> inputStrides(shape.size());
		std::vector<std::size_t> outputStrides(shape.size());
		std::size_t count = 1;
		for (std::size_t axis = 0; axis < shape.size(); ++axis)
		{
			outputShape[axis] = shape[static_cast<std::size_t>(permutation[axis])];
			count *= static_cast<std::size_t>(outputShape[axis]);
		}
		std::size_t stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			inputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
		}
		stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			outputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(outputShape[static_cast<std::size_t>(axis)]);
		}

		for (std::size_t outputIndex = 0; outputIndex < count; ++outputIndex)
		{
			std::size_t inputIndex = 0;
			for (std::size_t axis = 0; axis < shape.size(); ++axis)
			{
				const std::size_t coordinate =
					(outputIndex / outputStrides[axis]) % static_cast<std::size_t>(outputShape[axis]);
				inputIndex += coordinate * inputStrides[static_cast<std::size_t>(permutation[axis])];
			}
			output[outputIndex] = input[inputIndex];
		}
	}

	template <typename T>
	void BACKWARD_TRANSPOSE_TEMPLATE(const T *outputGrad, T *inputGrad,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		std::vector<int> outputShape(shape.size());
		std::vector<std::size_t> inputStrides(shape.size());
		std::vector<std::size_t> outputStrides(shape.size());
		std::size_t count = 1;
		for (std::size_t axis = 0; axis < shape.size(); ++axis)
		{
			outputShape[axis] = shape[static_cast<std::size_t>(permutation[axis])];
			count *= static_cast<std::size_t>(outputShape[axis]);
		}
		std::size_t stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			inputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(shape[static_cast<std::size_t>(axis)]);
		}
		stride = 1;
		for (int axis = static_cast<int>(shape.size()) - 1; axis >= 0; --axis)
		{
			outputStrides[static_cast<std::size_t>(axis)] = stride;
			stride *= static_cast<std::size_t>(outputShape[static_cast<std::size_t>(axis)]);
		}

		for (std::size_t outputIndex = 0; outputIndex < count; ++outputIndex)
		{
			std::size_t inputIndex = 0;
			for (std::size_t axis = 0; axis < shape.size(); ++axis)
			{
				const std::size_t coordinate =
					(outputIndex / outputStrides[axis]) % static_cast<std::size_t>(outputShape[axis]);
				inputIndex += coordinate * inputStrides[static_cast<std::size_t>(permutation[axis])];
			}
			inputGrad[inputIndex] = static_cast<T>(inputGrad[inputIndex] + outputGrad[outputIndex]);
		}
	}

	template <typename T>
	void TRANSPOSE_GENERIC_TEMPLATE(const T *input, T *output,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		TRANSPOSE_TEMPLATE<T>(input, output, shape, permutation);
	}

	template <typename T>
	void BACKWARD_TRANSPOSE_GENERIC_TEMPLATE(const T *outputGrad, T *inputGrad,
		const std::vector<int> &shape, const std::vector<int> &permutation)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(outputGrad, inputGrad, shape, permutation);
	}

	template <typename T>
	void TRANSPOSE_2D_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {1, 0});
	}
	template <typename T>
	void TRANSPOSE_3D_LAST_TWO_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		TRANSPOSE_TEMPLATE<T>(x, y, shape, {0, 2, 1});
	}
	template <typename T>
	void TRANSPOSE_4D_LAST_TWO_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		// Treat every [time, width] matrix independently and transpose it in
		// cache-sized tiles. Each OpenMP task owns a distinct output tile.
		const int time = shape[2];
		const int width = shape[3];
		const std::size_t matrixCount =
			static_cast<std::size_t>(shape[0]) * shape[1];
		const int tileSize = 32;
		const int timeTiles = (time + tileSize - 1) / tileSize;
		const int widthTiles = (width + tileSize - 1) / tileSize;
		const std::size_t tasks = matrixCount * timeTiles * widthTiles;

		#pragma omp parallel for schedule(static)
		for (std::int64_t taskIndex = 0;
			taskIndex < static_cast<std::int64_t>(tasks); ++taskIndex)
		{
			const std::size_t task = static_cast<std::size_t>(taskIndex);
			const int tileColumn = static_cast<int>(task % widthTiles);
			const int tileRow = static_cast<int>((task / widthTiles) % timeTiles);
			const std::size_t matrix = task
				/ (static_cast<std::size_t>(widthTiles) * timeTiles);
			const int rowStart = tileRow * tileSize;
			const int columnStart = tileColumn * tileSize;
			const std::size_t inputBase = matrix
				* static_cast<std::size_t>(time) * width;
			const std::size_t outputBase = matrix
				* static_cast<std::size_t>(time) * width;

			for (int row = rowStart; row < std::min(rowStart + tileSize, time); ++row)
			{
				for (int column = columnStart;
					column < std::min(columnStart + tileSize, width); ++column)
				{
					const std::size_t inputIndex = inputBase
						+ static_cast<std::size_t>(row) * width + column;
					const std::size_t outputIndex = outputBase
						+ static_cast<std::size_t>(column) * time + row;
					y[outputIndex] = x[inputIndex];
				}
			}
		}
	}
	template <typename T>
	void TRANSPOSE_4D_AXIS_1_2_TEMPLATE(const T *x, T *y, const std::vector<int> &shape)
	{
		// The final feature dimension is contiguous in both tensors. Transpose
		// tiles of [heads, time] and copy each contiguous feature block at once.
		const int heads = shape[1];
		const int time = shape[2];
		const int width = shape[3];
		const std::size_t batch = static_cast<std::size_t>(shape[0]);
		const int headTileSize = 16;
		const int timeTileSize = 32;
		const int headTiles = (heads + headTileSize - 1) / headTileSize;
		const int timeTiles = (time + timeTileSize - 1) / timeTileSize;
		const std::size_t tasks = batch * headTiles * timeTiles;
		const std::size_t batchSize =
			static_cast<std::size_t>(heads) * time * width;

		#pragma omp parallel for schedule(static)
		for (std::int64_t taskIndex = 0;
			taskIndex < static_cast<std::int64_t>(tasks); ++taskIndex)
		{
			const std::size_t task = static_cast<std::size_t>(taskIndex);
			const int timeTile = static_cast<int>(task % timeTiles);
			const int headTile = static_cast<int>((task / timeTiles) % headTiles);
			const std::size_t batchIndex = task
				/ (static_cast<std::size_t>(headTiles) * timeTiles);
			const int firstHead = headTile * headTileSize;
			const int firstTime = timeTile * timeTileSize;
			const std::size_t inputBatchOffset = batchIndex * batchSize;
			const std::size_t outputBatchOffset = batchIndex * batchSize;

			for (int timeIndex = firstTime;
				timeIndex < std::min(firstTime + timeTileSize, time); ++timeIndex)
			{
				for (int headIndex = firstHead;
					headIndex < std::min(firstHead + headTileSize, heads); ++headIndex)
				{
					const std::size_t inputOffset = inputBatchOffset
						+ (static_cast<std::size_t>(headIndex) * time + timeIndex) * width;
					const std::size_t outputOffset = outputBatchOffset
						+ (static_cast<std::size_t>(timeIndex) * heads + headIndex) * width;
					std::copy_n(x + inputOffset, width, y + outputOffset);
				}
			}
		}
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_2D_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {1, 0});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_3D_LAST_TWO_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		BACKWARD_TRANSPOSE_TEMPLATE<T>(yGrad, xGrad, shape, {0, 2, 1});
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_4D_LAST_TWO_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		const int time = shape[2];
		const int width = shape[3];
		const std::size_t matrixCount =
			static_cast<std::size_t>(shape[0]) * shape[1];
		const int tileSize = 32;
		const int timeTiles = (time + tileSize - 1) / tileSize;
		const int widthTiles = (width + tileSize - 1) / tileSize;
		const std::size_t tasks = matrixCount * timeTiles * widthTiles;

		#pragma omp parallel for schedule(static)
		for (std::int64_t taskIndex = 0;
			taskIndex < static_cast<std::int64_t>(tasks); ++taskIndex)
		{
			const std::size_t task = static_cast<std::size_t>(taskIndex);
			const int tileColumn = static_cast<int>(task % widthTiles);
			const int tileRow = static_cast<int>((task / widthTiles) % timeTiles);
			const std::size_t matrix = task
				/ (static_cast<std::size_t>(widthTiles) * timeTiles);
			const int rowStart = tileRow * tileSize;
			const int columnStart = tileColumn * tileSize;
			const std::size_t inputBase = matrix
				* static_cast<std::size_t>(time) * width;
			const std::size_t outputBase = matrix
				* static_cast<std::size_t>(time) * width;

			for (int row = rowStart; row < std::min(rowStart + tileSize, time); ++row)
			{
				for (int column = columnStart;
					column < std::min(columnStart + tileSize, width); ++column)
				{
					const std::size_t inputIndex = inputBase
						+ static_cast<std::size_t>(row) * width + column;
					const std::size_t outputIndex = outputBase
						+ static_cast<std::size_t>(column) * time + row;
					xGrad[inputIndex] = static_cast<T>(xGrad[inputIndex] + yGrad[outputIndex]);
				}
			}
		}
	}
	template <typename T>
	void BACKWARD_TRANSPOSE_4D_AXIS_1_2_TEMPLATE(const T *yGrad, T *xGrad, const std::vector<int> &shape)
	{
		const int heads = shape[1];
		const int time = shape[2];
		const int width = shape[3];
		const std::size_t batch = static_cast<std::size_t>(shape[0]);
		const int headTileSize = 16;
		const int timeTileSize = 32;
		const int headTiles = (heads + headTileSize - 1) / headTileSize;
		const int timeTiles = (time + timeTileSize - 1) / timeTileSize;
		const std::size_t tasks = batch * headTiles * timeTiles;
		const std::size_t batchSize =
			static_cast<std::size_t>(heads) * time * width;

		#pragma omp parallel for schedule(static)
		for (std::int64_t taskIndex = 0;
			taskIndex < static_cast<std::int64_t>(tasks); ++taskIndex)
		{
			const std::size_t task = static_cast<std::size_t>(taskIndex);
			const int timeTile = static_cast<int>(task % timeTiles);
			const int headTile = static_cast<int>((task / timeTiles) % headTiles);
			const std::size_t batchIndex = task
				/ (static_cast<std::size_t>(headTiles) * timeTiles);
			const int firstHead = headTile * headTileSize;
			const int firstTime = timeTile * timeTileSize;
			const std::size_t inputBatchOffset = batchIndex * batchSize;
			const std::size_t outputBatchOffset = batchIndex * batchSize;

			for (int timeIndex = firstTime;
				timeIndex < std::min(firstTime + timeTileSize, time); ++timeIndex)
			{
				for (int headIndex = firstHead;
					headIndex < std::min(firstHead + headTileSize, heads); ++headIndex)
				{
					const std::size_t inputOffset = inputBatchOffset
						+ (static_cast<std::size_t>(headIndex) * time + timeIndex) * width;
					const std::size_t outputOffset = outputBatchOffset
						+ (static_cast<std::size_t>(timeIndex) * heads + headIndex) * width;
					for (int feature = 0; feature < width; ++feature)
					{
						const std::size_t inputIndex = inputOffset + feature;
						const std::size_t outputIndex = outputOffset + feature;
						xGrad[inputIndex] = static_cast<T>(xGrad[inputIndex]
							+ yGrad[outputIndex]);
					}
				}
			}
		}
	}
}
