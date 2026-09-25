#include "ce.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		void runCe(Tensor &prediction, Tensor &target, Tensor &output, bool backward)
		{
			TensorAccess P = accessTensor(prediction);
			TensorAccess T = accessTensor(target);
			TensorAccess O = accessTensor(output);
			if (P.shape.empty() || P.shape != T.shape || P.shape.size() > 3
				|| P.dtype != T.dtype || P.dtype != O.dtype || P.shape.back() <= 0)
				throw std::runtime_error("CE: expected matching rank 1-3 tensors");

			const int classes = P.shape.back();
			const int rows = static_cast<int>(P.size / static_cast<std::size_t>(classes));
			if (O.size != static_cast<std::size_t>(rows))
				throw std::runtime_error("CE: output must contain one loss per row");
			dispatchDType(P.dtype, [&]<typename V>()
			{
				if (backward)
				{
					if (P.shape.size() == 1)
						Templates::BACKWARD_CE_1D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), P.gradAs<V>(), rows, classes, P.requiresGrad);
					else if (P.shape.size() == 2)
						Templates::BACKWARD_CE_2D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), P.gradAs<V>(), rows, classes, P.requiresGrad);
					else
						Templates::BACKWARD_CE_3D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), P.gradAs<V>(), rows, classes, P.requiresGrad);
				}
				else if (P.shape.size() == 1)
					Templates::CE_1D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
				else if (P.shape.size() == 2)
					Templates::CE_2D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
				else
					Templates::CE_3D_LAST_TEMPLATE<V>(P.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
			});
		}
	}

	void GraphRuntime::CE_1D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, false);
	}

	void GraphRuntime::CE_2D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, false);
	}

	void GraphRuntime::CE_3D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, false);
	}

	void GraphRuntime::BACKWARD_CE_1D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, true);
	}

	void GraphRuntime::BACKWARD_CE_2D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, true);
	}

	void GraphRuntime::BACKWARD_CE_3D_LAST(Tensor &prediction, Tensor &target, Tensor &output)
	{
		runCe(prediction, target, output, true);
	}

	void GraphRuntime::CE_GENERIC_AXIS(Tensor &predictionTensor, Tensor &targetTensor,
		Tensor &outputTensor, int axis)
	{
		TensorAccess prediction = accessTensor(predictionTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(prediction.shape.size());
		if (rank == 0 || prediction.shape != target.shape
			|| prediction.dtype != target.dtype || prediction.dtype != output.dtype)
			throw std::runtime_error("CE generic: prediction and target must match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || prediction.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("CE generic: invalid or empty class axis");

		std::vector<int> outputShape = prediction.shape;
		outputShape.erase(outputShape.begin() + axis);
		const int classes = prediction.shape[static_cast<std::size_t>(axis)];
		if (output.shape != outputShape
			|| output.size != prediction.size / static_cast<std::size_t>(classes))
			throw std::runtime_error("CE generic: reduced output shape mismatch");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(prediction.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(prediction.shape[static_cast<std::size_t>(i)]);

		dispatchDType(prediction.dtype, [&]<typename T>()
		{
			Templates::CE_GENERIC_AXIS_TEMPLATE<T>(prediction.dataAs<T>(),
				target.dataAs<T>(), output.dataAs<T>(), outer, inner, classes);
		});
	}

	void GraphRuntime::BACKWARD_CE_GENERIC_AXIS(Tensor &predictionTensor,
		Tensor &targetTensor, Tensor &outputTensor, int axis)
	{
		TensorAccess prediction = accessTensor(predictionTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(prediction.shape.size());
		if (rank == 0 || prediction.shape != target.shape
			|| prediction.dtype != target.dtype || prediction.dtype != output.dtype)
			throw std::runtime_error("CE generic backward: prediction and target must match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || prediction.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("CE generic backward: invalid or empty class axis");

		std::vector<int> outputShape = prediction.shape;
		outputShape.erase(outputShape.begin() + axis);
		const int classes = prediction.shape[static_cast<std::size_t>(axis)];
		if (output.shape != outputShape
			|| output.size != prediction.size / static_cast<std::size_t>(classes))
			throw std::runtime_error("CE generic backward: reduced output shape mismatch");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(prediction.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(prediction.shape[static_cast<std::size_t>(i)]);

		dispatchDType(prediction.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_CE_GENERIC_AXIS_TEMPLATE<T>(prediction.dataAs<T>(),
				target.dataAs<T>(), output.gradAs<T>(), prediction.gradAs<T>(),
				outer, inner, classes, prediction.requiresGrad);
		});
	}
}
