#pragma once

#include "Optimizer.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	class Fixed final : public Optimizer
	{
	public:
		explicit Fixed(Scalar learningRate = 0.1f);

		void step(GraphRuntime& graph) override;

	private:
		Scalar learningRate_;
	};
}

