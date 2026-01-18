#pragma once

#include <cstddef>
#include <optional>

namespace PHP2xAI::Runtime::CPP
{
	class GraphRuntime;
}

#include "../types.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	class Optimizer
	{
	public:
		virtual ~Optimizer() = default;

		virtual void step(GraphRuntime& graph) = 0;
		// Scalar getError() const;
		void setGradClip(std::optional<Scalar> clip);
		// void addError(Scalar error);
		// void zeroGrads(GraphRuntime& graph);

	protected:
		std::optional<Scalar> gradClip_;
		// Scalar error_ = 0;
		// std::size_t errorCounter_ = 0;
	};
}
