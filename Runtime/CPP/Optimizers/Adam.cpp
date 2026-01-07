#include <algorithm>
#include <cmath>

#include "Adam.hpp"
#include "../Core/runtime.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	Adam::Adam(Scalar learningRate, Scalar beta1, Scalar beta2, Scalar eps)
		: learningRate_(learningRate),
		beta1_(beta1),
		beta2_(beta2),
		eps_(eps)
	{
	}

	void Adam::step(GraphRuntime& graph)
	{
		const auto n = std::max(1, graph.accSteps);
		const auto beta1PowT = std::pow(beta1_, static_cast<Scalar>(stepNumber_));
		const auto beta2PowT = std::pow(beta2_, static_cast<Scalar>(stepNumber_));

		for (int tid : graph.trainable)
		{
			auto& t = graph.tensors[static_cast<std::size_t>(tid)];

			auto& mVec = mp_[tid];
			auto& vVec = vp_[tid];

			const auto size = t.data.size();
			if (mVec.size() < size)
				mVec.resize(size, static_cast<Scalar>(0));
			if (vVec.size() < size)
				vVec.resize(size, static_cast<Scalar>(0));

			for (std::size_t i = 0; i < size; ++i)
			{
				Scalar g = t.grad[i] / static_cast<Scalar>(n);

				if (gradClip_)
				{
					const auto clip = *gradClip_;
					if (g > clip)
						g = clip;
					else if (g < -clip)
						g = -clip;
				}

				const auto mtp = mVec[i];
				const auto vtp = vVec[i];

				const auto mt = beta1_ * mtp + (1 - beta1_) * g;
				const auto vt = beta2_ * vtp + (1 - beta2_) * (g * g);

				mVec[i] = mt;
				vVec[i] = vt;

				const auto mtHat = mt / (1 - beta1PowT);
				const auto vtHat = vt / (1 - beta2PowT);

				t.data[i] -= learningRate_ * (mtHat / (std::sqrt(vtHat) + eps_));
			}
		}

		++stepNumber_;
	}
}
