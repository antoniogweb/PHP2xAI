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
}
