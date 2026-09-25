#include "rms_norm.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::RMS_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &output, Scalar epsilon)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0 || X.shape != Y.shape || G.shape.size() != 1
			|| G.shape[0] != X.shape.back() || X.dtype != G.dtype
			|| X.dtype != Y.dtype || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm: dimensions, dtypes, or epsilon are invalid");

		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::RMS_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(), G.dataAs<T>(),
				Y.dataAs<T>(), outer, width, epsilon);
		});
	}

	void GraphRuntime::BACKWARD_RMS_NORM_LAST_AXIS(Tensor &input, Tensor &gamma,
		Tensor &output, Scalar epsilon)
	{
		TensorAccess X = accessTensor(input);
		TensorAccess G = accessTensor(gamma);
		TensorAccess Y = accessTensor(output);
		if (X.shape.empty() || X.shape.back() <= 0 || X.shape != Y.shape
			|| G.shape.size() != 1 || G.shape[0] != X.shape.back()
			|| X.dtype != G.dtype || X.dtype != Y.dtype || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm backward: invalid tensors or epsilon");
		const int width = X.shape.back();
		const std::size_t outer = X.size / static_cast<std::size_t>(width);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_RMS_NORM_LAST_AXIS_TEMPLATE<T>(X.dataAs<T>(),
				G.dataAs<T>(), Y.gradAs<T>(), X.gradAs<T>(), G.gradAs<T>(),
				outer, width, epsilon, X.requiresGrad, G.requiresGrad);
		});
	}

	void GraphRuntime::RMS_NORM_GENERIC(Tensor &inputTensor, Tensor &gammaTensor,
		Tensor &outputTensor, int axis, Scalar epsilon)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess G = accessTensor(gammaTensor);
		TensorAccess Y = accessTensor(outputTensor);
		const int rank = static_cast<int>(X.shape.size());
		if (rank == 0 || X.shape != Y.shape || X.dtype != G.dtype || X.dtype != Y.dtype)
			throw std::runtime_error("rms_norm generic: tensor shapes or dtypes do not match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::runtime_error("rms_norm generic: invalid axis");
		const int width = X.shape[static_cast<std::size_t>(axis)];
		if (width <= 0 || G.shape != std::vector<int>{width} || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm generic: gamma shape or epsilon is invalid");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(X.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(X.shape[static_cast<std::size_t>(i)]);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::RMS_NORM_GENERIC_TEMPLATE<T>(X.dataAs<T>(), G.dataAs<T>(),
				Y.dataAs<T>(), outer, inner, width, epsilon);
		});
	}

	void GraphRuntime::BACKWARD_RMS_NORM_GENERIC(Tensor &inputTensor,
		Tensor &gammaTensor, Tensor &outputTensor, int axis, Scalar epsilon)
	{
		TensorAccess X = accessTensor(inputTensor);
		TensorAccess G = accessTensor(gammaTensor);
		TensorAccess Y = accessTensor(outputTensor);
		const int rank = static_cast<int>(X.shape.size());
		if (rank == 0 || X.shape != Y.shape || X.dtype != G.dtype || X.dtype != Y.dtype)
			throw std::runtime_error("rms_norm generic backward: tensor shapes or dtypes do not match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::runtime_error("rms_norm generic backward: invalid axis");
		const int width = X.shape[static_cast<std::size_t>(axis)];
		if (width <= 0 || G.shape != std::vector<int>{width} || epsilon <= 0.0f)
			throw std::runtime_error("rms_norm generic backward: gamma shape or epsilon is invalid");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(X.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(X.shape[static_cast<std::size_t>(i)]);
		dispatchDType(X.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_RMS_NORM_GENERIC_TEMPLATE<T>(X.dataAs<T>(),
				G.dataAs<T>(), Y.gradAs<T>(), X.gradAs<T>(), G.gradAs<T>(),
				outer, inner, width, epsilon, X.requiresGrad, G.requiresGrad);
		});
	}
}
