#include "rope.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		void runRope(Tensor &input, Tensor &output, int offset, Scalar base,
			bool rotateHalf, bool backward)
		{
			TensorAccess X = accessTensor(input);
			TensorAccess Y = accessTensor(output);
			if (X.shape.size() < 2 || X.shape != Y.shape || X.dtype != Y.dtype
				|| base <= 0.0f)
				throw std::runtime_error("rope: invalid tensors or base");
			const int length = X.shape[X.shape.size() - 2];
			const int dimension = X.shape.back();
			if (length <= 0 || dimension <= 0)
				throw std::runtime_error("rope: sequence and feature dimensions must be positive");
			const std::size_t outer = X.size / static_cast<std::size_t>(length * dimension);
			dispatchDType(X.dtype, [&]<typename T>()
			{
				for (std::size_t batch = 0; batch < outer; ++batch)
				{
					const std::size_t offsetData = batch * length * dimension;
					if (backward)
					{
						if (rotateHalf)
							Templates::BACKWARD_ROPE_ROTATE_HALF_LAST_TWO_TEMPLATE<T>(
								Y.gradAs<T>() + offsetData, X.gradAs<T>() + offsetData,
								length, dimension, offset, base);
						else
							Templates::BACKWARD_ROPE_INTERLEAVED_LAST_TWO_TEMPLATE<T>(
								Y.gradAs<T>() + offsetData, X.gradAs<T>() + offsetData,
								length, dimension, offset, base);
					}
					else
					{
						if (rotateHalf)
							Templates::ROPE_ROTATE_HALF_LAST_TWO_TEMPLATE<T>(
								X.dataAs<T>() + offsetData, Y.dataAs<T>() + offsetData,
								length, dimension, offset, base);
						else
							Templates::ROPE_INTERLEAVED_LAST_TWO_TEMPLATE<T>(
								X.dataAs<T>() + offsetData, Y.dataAs<T>() + offsetData,
								length, dimension, offset, base);
					}
				}
			});
		}
	}

	void GraphRuntime::ROPE_INTERLEAVED_LAST_TWO(
		Tensor &input, Tensor &output, int offset, Scalar base)
	{
		runRope(input, output, offset, base, false, false);
	}

	void GraphRuntime::ROPE_ROTATE_HALF_LAST_TWO(
		Tensor &input, Tensor &output, int offset, Scalar base)
	{
		runRope(input, output, offset, base, true, false);
	}

	void GraphRuntime::BACKWARD_ROPE_INTERLEAVED_LAST_TWO(
		Tensor &input, Tensor &output, int offset, Scalar base)
	{
		runRope(input, output, offset, base, false, true);
	}

	void GraphRuntime::BACKWARD_ROPE_ROTATE_HALF_LAST_TWO(
		Tensor &input, Tensor &output, int offset, Scalar base)
	{
		runRope(input, output, offset, base, true, true);
	}
}
