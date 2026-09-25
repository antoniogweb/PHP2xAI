#include <algorithm>
#include <cmath>
#include <stdexcept>

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
		const auto beta1PowT = std::pow(beta1_, static_cast<Scalar>(stepNumber_));
		const auto beta2PowT = std::pow(beta2_, static_cast<Scalar>(stepNumber_));

		const std::vector<int> &trainable = graph.getTrainableTensorIds();
		for (std::size_t parameter = 0; parameter < trainable.size(); ++parameter)
		{
			const int tid = trainable[parameter];
			if (!graph.tensorHasFloatingPointDType(tid))
				throw std::runtime_error("Adam can only update floating point tensors");

			auto& mVec = mp_[tid];
			auto& vVec = vp_[tid];

			const std::size_t size = graph.getTensorSize(tid);
			if (mVec.size() < size)
				mVec.resize(size, static_cast<Scalar>(0));
			if (vVec.size() < size)
				vVec.resize(size, static_cast<Scalar>(0));

			for (std::size_t i = 0; i < size; ++i)
			{
				Scalar g = graph.getTensorGradValue(tid, i);

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

				const Scalar value = graph.getTensorDataValue(tid, i);
				graph.setTensorDataValue(tid, i, value - learningRate_ * (mtHat / (std::sqrt(vtHat) + eps_)));
			}
		}

		++stepNumber_;
	}
}
