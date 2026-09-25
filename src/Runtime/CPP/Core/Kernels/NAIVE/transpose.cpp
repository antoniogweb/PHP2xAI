#include "transpose.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		void transposeKernel(Tensor &input, Tensor &output,
			const std::vector<int> &permutation, bool backward)
		{
			TensorAccess X = accessTensor(input);
			TensorAccess Y = accessTensor(output);
			if (X.dtype != Y.dtype || permutation.size() != X.shape.size())
				throw std::runtime_error("transpose: dtype or rank mismatch");
			std::vector<int> expectedShape(permutation.size());
			for (std::size_t axis = 0; axis < permutation.size(); ++axis)
			{
				if (permutation[axis] < 0
					|| static_cast<std::size_t>(permutation[axis]) >= X.shape.size())
					throw std::runtime_error("transpose: invalid permutation");
				expectedShape[axis] = X.shape[static_cast<std::size_t>(permutation[axis])];
			}
			if (Y.shape != expectedShape || X.size != Y.size)
				throw std::runtime_error("transpose: output dimensions do not match permutation");

			dispatchDType(X.dtype, [&]<typename T>()
			{
				if (backward)
				{
					if (permutation == std::vector<int>{1, 0})
						Templates::BACKWARD_TRANSPOSE_2D_TEMPLATE<T>(Y.gradAs<T>(), X.gradAs<T>(), X.shape);
					else if (permutation == std::vector<int>{0, 2, 1})
						Templates::BACKWARD_TRANSPOSE_3D_LAST_TWO_TEMPLATE<T>(Y.gradAs<T>(), X.gradAs<T>(), X.shape);
					else if (permutation == std::vector<int>{0, 1, 3, 2})
						Templates::BACKWARD_TRANSPOSE_4D_LAST_TWO_TEMPLATE<T>(Y.gradAs<T>(), X.gradAs<T>(), X.shape);
					else
						Templates::BACKWARD_TRANSPOSE_4D_AXIS_1_2_TEMPLATE<T>(Y.gradAs<T>(), X.gradAs<T>(), X.shape);
				}
				else
				{
					if (permutation == std::vector<int>{1, 0})
						Templates::TRANSPOSE_2D_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.shape);
					else if (permutation == std::vector<int>{0, 2, 1})
						Templates::TRANSPOSE_3D_LAST_TWO_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.shape);
					else if (permutation == std::vector<int>{0, 1, 3, 2})
						Templates::TRANSPOSE_4D_LAST_TWO_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.shape);
					else
						Templates::TRANSPOSE_4D_AXIS_1_2_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(), X.shape);
				}
			});
		}
	}

	void GraphRuntime::TRANSPOSE_2D(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {1, 0}, false);
	}

	void GraphRuntime::TRANSPOSE_3D_LAST_TWO(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 2, 1}, false);
	}

	void GraphRuntime::TRANSPOSE_4D_LAST_TWO(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 1, 3, 2}, false);
	}

	void GraphRuntime::TRANSPOSE_4D_AXIS_1_2(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 2, 1, 3}, false);
	}

	void GraphRuntime::BACKWARD_TRANSPOSE_2D(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {1, 0}, true);
	}

	void GraphRuntime::BACKWARD_TRANSPOSE_3D_LAST_TWO(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 2, 1}, true);
	}

	void GraphRuntime::BACKWARD_TRANSPOSE_4D_LAST_TWO(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 1, 3, 2}, true);
	}

	void GraphRuntime::BACKWARD_TRANSPOSE_4D_AXIS_1_2(Tensor &input, Tensor &output)
	{
		transposeKernel(input, output, {0, 2, 1, 3}, true);
	}
}
