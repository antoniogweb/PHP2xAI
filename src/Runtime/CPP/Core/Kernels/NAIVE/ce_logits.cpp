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
}
