#include "Tensor.hpp"

#include <stdexcept>

#include "../Kernels/DTypeDispatch.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		template <typename T>
		void fillZeros(T *values, std::size_t count)
		{
			for (std::size_t i = 0; i < count; ++i)
				values[i] = static_cast<T>(0);
		}
	}

	Tensor::Tensor()
		: id(-1), dtype(DType::FLOAT32), requiresGrad(false),
		  data(0), grad(0), size(0)
	{
	}

	void Tensor::allocate(DType tensorDType, std::size_t elementSize)
	{
		dtype = tensorDType;
		size = elementSize;

		// Allocate real objects of the requested type, not just untyped bytes.
		// dataAs<T>() can therefore safely expose a pointer to typed elements.
		switch (dtype)
		{
			case DType::FLOAT32:
				dataOwner.allocate<float>(size);
				gradOwner.allocate<float>(size);
				break;
			case DType::FLOAT64:
				dataOwner.allocate<double>(size);
				gradOwner.allocate<double>(size);
				break;
			case DType::INT32:
				dataOwner.allocate<std::int32_t>(size);
				gradOwner.allocate<std::int32_t>(size);
				break;
			case DType::INT64:
				dataOwner.allocate<std::int64_t>(size);
				gradOwner.allocate<std::int64_t>(size);
				break;
		}

		// These runtime pointers are non-owning views. TensorBuffer still owns the
		// allocations and releases them when this Tensor is destroyed or resized.
		data = dataOwner.data();
		grad = gradOwner.data();
		fillStorageWithZeros(data);
		fillStorageWithZeros(grad);
	}

	Scalar Tensor::readData(std::size_t index) const
	{
		checkIndex(index);
		Scalar value = 0.0f;
		dispatchDType(dtype, [&]<typename T>()
		{
			value = static_cast<Scalar>(dataAs<T>()[index]);
		});
		return value;
	}

	Scalar Tensor::readGrad(std::size_t index) const
	{
		checkIndex(index);
		Scalar value = 0.0f;
		dispatchDType(dtype, [&]<typename T>()
		{
			value = static_cast<Scalar>(gradAs<T>()[index]);
		});
		return value;
	}

	void Tensor::writeData(std::size_t index, Scalar value)
	{
		checkIndex(index);
		dispatchDType(dtype, [&]<typename T>()
		{
			dataAs<T>()[index] = static_cast<T>(value);
		});
	}

	void Tensor::writeGrad(std::size_t index, Scalar value)
	{
		checkIndex(index);
		dispatchDType(dtype, [&]<typename T>()
		{
			gradAs<T>()[index] = static_cast<T>(value);
		});
	}

	void Tensor::fillStorageWithZeros(void *buffer)
	{
		dispatchDType(dtype, [&]<typename T>()
		{
			fillZeros(static_cast<T *>(buffer), size);
		});
	}

	void Tensor::fillGradWithZeros()
	{
		// Select the typed pointer once per tensor, then clear its elements.
		switch (dtype)
		{
			case DType::FLOAT32:
				fillZeros(gradAs<float>(), size);
				break;
			case DType::FLOAT64:
				fillZeros(gradAs<double>(), size);
				break;
			case DType::INT32:
				fillZeros(gradAs<std::int32_t>(), size);
				break;
			case DType::INT64:
				fillZeros(gradAs<std::int64_t>(), size);
				break;
		}
	}

	void Tensor::checkIndex(std::size_t index) const
	{
		if (index >= size)
			throw std::out_of_range("Tensor element index out of range");
	}
}
