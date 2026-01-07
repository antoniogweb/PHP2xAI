#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "Optimizer.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	class Adam final : public Optimizer
	{
	public:
		explicit Adam(Scalar learningRate = 0.1f,
					Scalar beta1 = 0.9f,
					Scalar beta2 = 0.999f,
					Scalar eps = 0.00000001f);

		void step(GraphRuntime& graph) override;

	private:
		Scalar learningRate_;
		Scalar beta1_;
		Scalar beta2_;
		Scalar eps_;

		std::unordered_map<int, std::vector<Scalar>> mp_;
		std::unordered_map<int, std::vector<Scalar>> vp_;

		std::size_t stepNumber_ = 1;
	};
}

