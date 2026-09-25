#include "ce_logits.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		void runCeLogits(Tensor &logits, Tensor &target, Tensor &output, bool backward)
		{
			TensorAccess L = accessTensor(logits);
			TensorAccess T = accessTensor(target);
			TensorAccess O = accessTensor(output);
			if (L.shape.empty() || L.shape != T.shape || L.shape.size() > 3
				|| L.dtype != T.dtype || L.dtype != O.dtype || L.shape.back() <= 0)
				throw std::runtime_error("softmax_ce_logits: expected matching rank 1-3 tensors");

			const int classes = L.shape.back();
			const int rows = static_cast<int>(L.size / static_cast<std::size_t>(classes));
			if (O.size != static_cast<std::size_t>(rows))
				throw std::runtime_error("softmax_ce_logits: output must contain one loss per row");
			dispatchDType(L.dtype, [&]<typename V>()
			{
				if (backward)
				{
					if (L.shape.size() == 1)
						Templates::BACKWARD_CE_LOGITS_1D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), L.gradAs<V>(), rows, classes, L.requiresGrad);
					else if (L.shape.size() == 2)
						Templates::BACKWARD_CE_LOGITS_2D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), L.gradAs<V>(), rows, classes, L.requiresGrad);
					else
						Templates::BACKWARD_CE_LOGITS_3D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.gradAs<V>(), L.gradAs<V>(), rows, classes, L.requiresGrad);
				}
				else if (L.shape.size() == 1)
					Templates::CE_LOGITS_1D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
				else if (L.shape.size() == 2)
					Templates::CE_LOGITS_2D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
				else
					Templates::CE_LOGITS_3D_LAST_TEMPLATE<V>(L.dataAs<V>(), T.dataAs<V>(), O.dataAs<V>(), rows, classes);
			});
		}
	}

	void GraphRuntime::CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, false);
	}

	void GraphRuntime::CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, false);
	}

	void GraphRuntime::CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, false);
	}

	void GraphRuntime::BACKWARD_CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, true);
	}

	void GraphRuntime::BACKWARD_CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, true);
	}

	void GraphRuntime::BACKWARD_CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &output)
	{
		runCeLogits(logits, target, output, true);
	}

	void GraphRuntime::CE_LOGITS_GENERIC_AXIS(Tensor &logitsTensor,
		Tensor &targetTensor, Tensor &outputTensor, int axis)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(logits.shape.size());
		if (rank == 0 || logits.shape != target.shape
			|| logits.dtype != target.dtype || logits.dtype != output.dtype)
			throw std::runtime_error("CE logits generic: logits and target must match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || logits.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("CE logits generic: invalid or empty class axis");

		std::vector<int> outputShape = logits.shape;
		outputShape.erase(outputShape.begin() + axis);
		const int classes = logits.shape[static_cast<std::size_t>(axis)];
		if (output.shape != outputShape
			|| output.size != logits.size / static_cast<std::size_t>(classes))
			throw std::runtime_error("CE logits generic: reduced output shape mismatch");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(logits.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(logits.shape[static_cast<std::size_t>(i)]);

		dispatchDType(logits.dtype, [&]<typename T>()
		{
			Templates::CE_LOGITS_GENERIC_AXIS_TEMPLATE<T>(logits.dataAs<T>(),
				target.dataAs<T>(), output.dataAs<T>(), outer, inner, classes);
		});
	}

	void GraphRuntime::BACKWARD_CE_LOGITS_GENERIC_AXIS(Tensor &logitsTensor,
		Tensor &targetTensor, Tensor &outputTensor, int axis)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);
		const int rank = static_cast<int>(logits.shape.size());
		if (rank == 0 || logits.shape != target.shape
			|| logits.dtype != target.dtype || logits.dtype != output.dtype)
			throw std::runtime_error("CE logits generic backward: logits and target must match");
		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank || logits.shape[static_cast<std::size_t>(axis)] <= 0)
			throw std::runtime_error("CE logits generic backward: invalid or empty class axis");

		std::vector<int> outputShape = logits.shape;
		outputShape.erase(outputShape.begin() + axis);
		const int classes = logits.shape[static_cast<std::size_t>(axis)];
		if (output.shape != outputShape
			|| output.size != logits.size / static_cast<std::size_t>(classes))
			throw std::runtime_error("CE logits generic backward: reduced output shape mismatch");
		std::size_t outer = 1;
		std::size_t inner = 1;
		for (int i = 0; i < axis; ++i)
			outer *= static_cast<std::size_t>(logits.shape[static_cast<std::size_t>(i)]);
		for (int i = axis + 1; i < rank; ++i)
			inner *= static_cast<std::size_t>(logits.shape[static_cast<std::size_t>(i)]);

		dispatchDType(logits.dtype, [&]<typename T>()
		{
			Templates::BACKWARD_CE_LOGITS_GENERIC_AXIS_TEMPLATE<T>(
				logits.dataAs<T>(), target.dataAs<T>(), output.gradAs<T>(),
				logits.gradAs<T>(), outer, inner, classes, logits.requiresGrad);
		});
	}
}
