#pragma once

#include <cstddef>
#include <cstdint>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	namespace DropoutDetail
	{
		inline std::uint64_t splitmix64(std::uint64_t value)
		{
			value += 0x9e3779b97f4a7c15ULL;
			value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
			value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
			return value ^ (value >> 31U);
		}
	}

	template <typename T>
	void DROPOUT_TEMPLATE(const T *input, T *output, Scalar *mask,
		std::size_t elementCount, Scalar dropoutPerc, std::uint64_t seed, bool training)
	{
		if (!training)
		{
			for (std::size_t i = 0; i < elementCount; ++i)
				output[i] = input[i];
			return;
		}

		const Scalar keepProbability = 1.0f - dropoutPerc / 100.0f;
		const Scalar scale = keepProbability > 0.0f ? 1.0f / keepProbability : 0.0f;
		// Each index gets its own deterministic random value, so workers can
		// fill disjoint output and mask elements without sharing RNG state.
		#pragma omp parallel for schedule(static)
		for (std::int64_t index = 0;
			index < static_cast<std::int64_t>(elementCount); ++index)
		{
			const std::size_t i = static_cast<std::size_t>(index);
			const std::uint64_t randomBits = DropoutDetail::splitmix64(seed + i);
			const Scalar randomUnit = static_cast<Scalar>(randomBits >> 40U)
				* (1.0f / 16777216.0f);
			const Scalar multiplier = randomUnit >= dropoutPerc / 100.0f ? scale : 0.0f;
			mask[i] = multiplier;
			output[i] = static_cast<T>(static_cast<Scalar>(input[i]) * multiplier);
		}
	}

	template <typename T>
	void BACKWARD_DROPOUT_TEMPLATE(T *inputGrad, const T *outputGrad,
		const Scalar *mask, std::size_t elementCount)
	{
		for (std::size_t i = 0; i < elementCount; ++i)
		{
			const Scalar accumulated = static_cast<Scalar>(inputGrad[i])
				+ static_cast<Scalar>(outputGrad[i]) * mask[i];
			inputGrad[i] = static_cast<T>(accumulated);
		}
	}
}
