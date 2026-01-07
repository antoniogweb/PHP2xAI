#include "Optimizer.hpp"
#include "../Core/runtime.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	Scalar Optimizer::getError() const
	{
		if (errorCounter_ == 0)
			return 0;

		return error_ / static_cast<Scalar>(errorCounter_);
	}

	void Optimizer::setGradClip(std::optional<Scalar> clip)
	{
		gradClip_ = clip;
	}

	void Optimizer::addError(Scalar error)
	{
		error_ += error;
		++errorCounter_;
	}

	void Optimizer::zeroGrads(GraphRuntime& graph)
	{
		error_ = 0;
		errorCounter_ = 0;
		graph.resetGrad();
	}
}
