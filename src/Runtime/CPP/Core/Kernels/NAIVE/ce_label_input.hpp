#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "../../../types.hpp"

namespace PHP2xAI::Runtime::CPP::Templates
{
	template <typename T, typename TargetT>
	void CE_LOGITS_LABEL_INT_GENERIC_AXIS_TEMPLATE(const T *logits,
		const TargetT *target, T *output, std::size_t outer, std::size_t inner,
		int classCount)
	{
		for (std::size_t row = 0; row < outer; ++row)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				const int label = static_cast<int>(target[row * inner + innerIndex]);
				if (label < 0 || label >= classCount)
					throw std::out_of_range("CE logits label int: label is outside the class range");
				const std::size_t base = row * static_cast<std::size_t>(classCount) * inner + innerIndex;
				Scalar maximum = static_cast<Scalar>(logits[base]);
				for (int classIndex = 1; classIndex < classCount; ++classIndex)
				{
					const std::size_t index = base + static_cast<std::size_t>(classIndex) * inner;
					maximum = std::max(maximum, static_cast<Scalar>(logits[index]));
				}
				Scalar sumExp = 0.0f;
				for (int classIndex = 0; classIndex < classCount; ++classIndex)
				{
					const std::size_t index = base + static_cast<std::size_t>(classIndex) * inner;
					sumExp += std::exp(static_cast<Scalar>(logits[index]) - maximum);
				}
				const std::size_t labelIndex = base + static_cast<std::size_t>(label) * inner;
				output[row * inner + innerIndex] = static_cast<T>(
					std::log(sumExp) + maximum - static_cast<Scalar>(logits[labelIndex]));
			}
		}
	}

	template <typename T, typename TargetT>
	void BACKWARD_CE_LOGITS_LABEL_INT_GENERIC_AXIS_TEMPLATE(const T *logits,
		const TargetT *target, T *logitsGrad, const T *outputGrad,
		std::size_t outer, std::size_t inner, int classCount)
	{
		for (std::size_t row = 0; row < outer; ++row)
		{
			for (std::size_t innerIndex = 0; innerIndex < inner; ++innerIndex)
			{
				const int label = static_cast<int>(target[row * inner + innerIndex]);
				if (label < 0 || label >= classCount)
					throw std::out_of_range("CE logits label int: label is outside the class range");
				const std::size_t base = row * static_cast<std::size_t>(classCount) * inner + innerIndex;
				Scalar maximum = static_cast<Scalar>(logits[base]);
				for (int classIndex = 1; classIndex < classCount; ++classIndex)
				{
					const std::size_t index = base + static_cast<std::size_t>(classIndex) * inner;
					maximum = std::max(maximum, static_cast<Scalar>(logits[index]));
				}
				Scalar sumExp = 0.0f;
				for (int classIndex = 0; classIndex < classCount; ++classIndex)
				{
					const std::size_t index = base + static_cast<std::size_t>(classIndex) * inner;
					sumExp += std::exp(static_cast<Scalar>(logits[index]) - maximum);
				}
				const Scalar outputGradient = static_cast<Scalar>(
					outputGrad[row * inner + innerIndex]);
				for (int classIndex = 0; classIndex < classCount; ++classIndex)
				{
					const std::size_t index = base + static_cast<std::size_t>(classIndex) * inner;
					Scalar probability = std::exp(static_cast<Scalar>(logits[index]) - maximum) / sumExp;
					if (classIndex == label)
						probability -= 1.0f;
					logitsGrad[index] = static_cast<T>(static_cast<Scalar>(logitsGrad[index])
						+ outputGradient * probability);
				}
			}
		}
	}

	namespace
	{
		template <typename T>
		Scalar crossEntropyLabelRow(const T *logits, int label, int classCount)
		{
			if (label < 0 || label >= classCount)
				throw std::out_of_range("CE logits label int: label is outside the class range");

			Scalar maxLogit = static_cast<Scalar>(logits[0]);
			for (int i = 1; i < classCount; ++i)
			{
				const Scalar value = static_cast<Scalar>(logits[i]);
				if (value > maxLogit)
					maxLogit = value;
			}

			Scalar sumExp = 0.0f;
			for (int i = 0; i < classCount; ++i)
				sumExp += std::exp(static_cast<Scalar>(logits[i]) - maxLogit);

			return std::log(sumExp) + maxLogit - static_cast<Scalar>(logits[label]);
		}

		template <typename T, typename TargetT>
		void backwardCrossEntropyLabelRow(
			const T *logits,
			const TargetT *target,
			T *logitsGrad,
			Scalar outputGrad,
			int classCount,
			std::size_t targetIndex)
		{
			const int label = static_cast<int>(target[targetIndex]);
			if (label < 0 || label >= classCount)
				throw std::out_of_range("CE logits label int: label is outside the class range");

			Scalar maxLogit = static_cast<Scalar>(logits[0]);
			for (int i = 1; i < classCount; ++i)
			{
				const Scalar value = static_cast<Scalar>(logits[i]);
				if (value > maxLogit)
					maxLogit = value;
			}

			Scalar sumExp = 0.0f;
			for (int i = 0; i < classCount; ++i)
				sumExp += std::exp(static_cast<Scalar>(logits[i]) - maxLogit);

			const Scalar inverseSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
			for (int i = 0; i < classCount; ++i)
			{
				Scalar probability = std::exp(static_cast<Scalar>(logits[i]) - maxLogit) * inverseSum;
				if (i == label)
					probability -= 1.0f;

				const Scalar gradient = static_cast<Scalar>(logitsGrad[i])
					+ outputGrad * probability;
				logitsGrad[i] = static_cast<T>(gradient);
			}
		}
	}

	template <typename T, typename TargetT>
	void CE_LOGITS_LABEL_INT_1D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *output,
		int classCount)
	{
		const int label = static_cast<int>(target[0]);
		output[0] = static_cast<T>(crossEntropyLabelRow(logits, label, classCount));
	}

	template <typename T, typename TargetT>
	void CE_LOGITS_LABEL_INT_2D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *output,
		int batchSize,
		int classCount)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const std::size_t logitsOffset = static_cast<std::size_t>(batch) * classCount;
			output[batch] = static_cast<T>(crossEntropyLabelRow(
				logits + logitsOffset, static_cast<int>(target[batch]), classCount));
		}
	}

	template <typename T, typename TargetT>
	void CE_LOGITS_LABEL_INT_3D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *output,
		int batchSize,
		int stepCount,
		int classCount)
	{
		const int rowCount = batchSize * stepCount;
		for (int row = 0; row < rowCount; ++row)
		{
			const std::size_t logitsOffset = static_cast<std::size_t>(row) * classCount;
			output[row] = static_cast<T>(crossEntropyLabelRow(
				logits + logitsOffset, static_cast<int>(target[row]), classCount));
		}
	}

	template <typename T, typename TargetT>
	void BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *logitsGrad,
		const T *outputGrad,
		int classCount)
	{
		backwardCrossEntropyLabelRow(
			logits, target, logitsGrad, static_cast<Scalar>(outputGrad[0]), classCount, 0);
	}

	template <typename T, typename TargetT>
	void BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *logitsGrad,
		const T *outputGrad,
		int batchSize,
		int classCount)
	{
		for (int batch = 0; batch < batchSize; ++batch)
		{
			const std::size_t offset = static_cast<std::size_t>(batch) * classCount;
			backwardCrossEntropyLabelRow(
				logits + offset, target, logitsGrad + offset,
				static_cast<Scalar>(outputGrad[batch]), classCount,
				static_cast<std::size_t>(batch));
		}
	}

	template <typename T, typename TargetT>
	void BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST_TEMPLATE(
		const T *logits,
		const TargetT *target,
		T *logitsGrad,
		const T *outputGrad,
		int batchSize,
		int stepCount,
		int classCount)
	{
		const int rowCount = batchSize * stepCount;
		for (int row = 0; row < rowCount; ++row)
		{
			const std::size_t offset = static_cast<std::size_t>(row) * classCount;
			backwardCrossEntropyLabelRow(
				logits + offset, target, logitsGrad + offset,
				static_cast<Scalar>(outputGrad[row]), classCount,
				static_cast<std::size_t>(row));
		}
	}
}
