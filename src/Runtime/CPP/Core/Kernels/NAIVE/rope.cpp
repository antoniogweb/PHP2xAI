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

		void runRopeGeneric(Tensor &input, Tensor &output, int positionAxis,
			int rotationAxis, int offset, Scalar base, bool rotateHalf, bool backward)
		{
			TensorAccess X = accessTensor(input);
			TensorAccess Y = accessTensor(output);
			const int rank = static_cast<int>(X.shape.size());
			if (rank < 2 || X.shape != Y.shape || X.dtype != Y.dtype || base <= 0.0f)
				throw std::runtime_error("rope generic: invalid tensors or base");
			if (positionAxis < 0)
				positionAxis += rank;
			if (rotationAxis < 0)
				rotationAxis += rank;
			if (positionAxis < 0 || positionAxis >= rank || rotationAxis < 0
				|| rotationAxis >= rank || positionAxis == rotationAxis)
				throw std::runtime_error("rope generic: axes are invalid");
			if (X.shape[static_cast<std::size_t>(positionAxis)] <= 0
				|| X.shape[static_cast<std::size_t>(rotationAxis)] <= 0
				|| X.shape[static_cast<std::size_t>(rotationAxis)] % 2 != 0)
				throw std::runtime_error("rope generic: axis dimensions are invalid");

			dispatchDType(X.dtype, [&]<typename T>()
			{
				if (backward)
					Templates::BACKWARD_ROPE_GENERIC_TEMPLATE<T>(Y.gradAs<T>(),
						X.gradAs<T>(), X.shape, positionAxis, rotationAxis,
						offset, base, rotateHalf);
				else
					Templates::ROPE_GENERIC_TEMPLATE<T>(X.dataAs<T>(), Y.dataAs<T>(),
						X.shape, positionAxis, rotationAxis, offset, base, rotateHalf);
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

	void GraphRuntime::ROPE_INTERLEAVED_GENERIC(Tensor &input, Tensor &output,
		int positionAxis, int rotationAxis, int offset, Scalar base)
	{
		runRopeGeneric(input, output, positionAxis, rotationAxis,
			offset, base, false, false);
	}

	void GraphRuntime::ROPE_ROTATE_HALF_GENERIC(Tensor &input, Tensor &output,
		int positionAxis, int rotationAxis, int offset, Scalar base)
	{
		runRopeGeneric(input, output, positionAxis, rotationAxis,
			offset, base, true, false);
	}

	void GraphRuntime::BACKWARD_ROPE_INTERLEAVED_GENERIC(Tensor &input,
		Tensor &output, int positionAxis, int rotationAxis, int offset, Scalar base)
	{
		runRopeGeneric(input, output, positionAxis, rotationAxis,
			offset, base, false, true);
	}

	void GraphRuntime::BACKWARD_ROPE_ROTATE_HALF_GENERIC(Tensor &input,
		Tensor &output, int positionAxis, int rotationAxis, int offset, Scalar base)
	{
		runRopeGeneric(input, output, positionAxis, rotationAxis,
			offset, base, true, true);
	}
}
