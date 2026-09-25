#include "ce_label_input.hpp"

#include "../DTypeDispatch.hpp"
#include "../TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	void GraphRuntime::CE_LOGITS_LABEL_INT_1D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 1 || logits.size == 0
			|| target.size == 0 || !output.shape.empty() || output.size != 1)
			throw std::runtime_error("CE logits label int 1D: dimension mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int: output dtype must match logits");

		const int classCount = logits.shape[0];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::CE_LOGITS_LABEL_INT_1D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), output.dataAs<T>(), classCount);
			});
		});
	}

	void GraphRuntime::CE_LOGITS_LABEL_INT_2D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 2 || target.shape.size() != 1
			|| output.shape.size() != 1 || logits.shape[0] != target.shape[0]
			|| output.shape[0] != logits.shape[0] || logits.shape[1] <= 0
			|| logits.size != static_cast<std::size_t>(logits.shape[0]) * logits.shape[1]
			|| target.size != static_cast<std::size_t>(logits.shape[0])
			|| output.size != static_cast<std::size_t>(logits.shape[0]))
			throw std::runtime_error("CE logits label int 2D: dimension mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int: output dtype must match logits");

		const int batchSize = logits.shape[0];
		const int classCount = logits.shape[1];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::CE_LOGITS_LABEL_INT_2D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), output.dataAs<T>(),
					batchSize, classCount);
			});
		});
	}

	void GraphRuntime::CE_LOGITS_LABEL_INT_3D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 3 || target.shape.size() != 2
			|| output.shape.size() != 2 || logits.shape[0] != target.shape[0]
			|| logits.shape[1] != target.shape[1] || output.shape != target.shape
			|| logits.shape[2] <= 0)
			throw std::runtime_error("CE logits label int 3D: dimension mismatch");
		const std::size_t rowCount = static_cast<std::size_t>(logits.shape[0])
			* static_cast<std::size_t>(logits.shape[1]);
		if (target.size != rowCount || output.size != rowCount
			|| logits.size != rowCount * static_cast<std::size_t>(logits.shape[2]))
			throw std::runtime_error("CE logits label int 3D: data size mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int: output dtype must match logits");

		const int batchSize = logits.shape[0];
		const int stepCount = logits.shape[1];
		const int classCount = logits.shape[2];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::CE_LOGITS_LABEL_INT_3D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), output.dataAs<T>(),
					batchSize, stepCount, classCount);
			});
		});
	}

	void GraphRuntime::CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &, Tensor &, Tensor &)
	{
		throw std::runtime_error(
			"CE logits label int: generic axis kernel is not implemented for the NAIVE backend");
	}

	void GraphRuntime::BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 1 || logits.size == 0 || target.size == 0
			|| output.size != 1)
			throw std::runtime_error("CE logits label int 1D backward: dimension mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int backward: output dtype must match logits");

		const int classCount = logits.shape[0];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), logits.gradAs<T>(),
					output.gradAs<T>(), classCount);
			});
		});
	}

	void GraphRuntime::BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 2 || target.shape.size() != 1
			|| output.shape.size() != 1 || logits.shape[0] != target.shape[0]
			|| output.shape[0] != logits.shape[0] || logits.shape[1] <= 0
			|| logits.size != static_cast<std::size_t>(logits.shape[0]) * logits.shape[1]
			|| target.size != static_cast<std::size_t>(logits.shape[0])
			|| output.size != static_cast<std::size_t>(logits.shape[0]))
			throw std::runtime_error("CE logits label int 2D backward: dimension mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int backward: output dtype must match logits");

		const int batchSize = logits.shape[0];
		const int classCount = logits.shape[1];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), logits.gradAs<T>(),
					output.gradAs<T>(), batchSize, classCount);
			});
		});
	}

	void GraphRuntime::BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST(
		Tensor &logitsTensor,
		Tensor &targetTensor,
		Tensor &outputTensor)
	{
		TensorAccess logits = accessTensor(logitsTensor);
		TensorAccess target = accessTensor(targetTensor);
		TensorAccess output = accessTensor(outputTensor);

		if (logits.shape.size() != 3 || target.shape.size() != 2
			|| output.shape.size() != 2 || logits.shape[0] != target.shape[0]
			|| logits.shape[1] != target.shape[1] || output.shape != target.shape
			|| logits.shape[2] <= 0)
			throw std::runtime_error("CE logits label int 3D backward: dimension mismatch");
		const std::size_t rowCount = static_cast<std::size_t>(logits.shape[0])
			* static_cast<std::size_t>(logits.shape[1]);
		if (target.size != rowCount || output.size != rowCount
			|| logits.size != rowCount * static_cast<std::size_t>(logits.shape[2]))
			throw std::runtime_error("CE logits label int 3D backward: data size mismatch");
		if (output.dtype != logits.dtype)
			throw std::runtime_error("CE logits label int backward: output dtype must match logits");

		const int batchSize = logits.shape[0];
		const int stepCount = logits.shape[1];
		const int classCount = logits.shape[2];
		dispatchDType(logits.dtype, [&]<typename T>()
		{
			dispatchDType(target.dtype, [&]<typename TargetT>()
			{
				Templates::BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST_TEMPLATE<T, TargetT>(
					logits.dataAs<T>(), target.dataAs<TargetT>(), logits.gradAs<T>(),
					output.gradAs<T>(), batchSize, stepCount, classCount);
			});
		});
	}

	void GraphRuntime::BACKWORD_CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &, Tensor &, Tensor &)
	{
		throw std::runtime_error(
			"CE logits label int backward: generic axis kernel is not implemented for the NAIVE backend");
	}
}
