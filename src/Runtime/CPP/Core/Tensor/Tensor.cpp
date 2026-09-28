#include "Tensor.hpp"

#include <stdexcept>

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
		switch (dtype)
		{
			case DType::FLOAT32: return static_cast<Scalar>(dataAs<float>()[index]);
			case DType::FLOAT64: return static_cast<Scalar>(dataAs<double>()[index]);
			case DType::INT32: return static_cast<Scalar>(dataAs<std::int32_t>()[index]);
			case DType::INT64: return static_cast<Scalar>(dataAs<std::int64_t>()[index]);
		}
		throw std::runtime_error("Invalid tensor dtype");
	}

	Scalar Tensor::readGrad(std::size_t index) const
	{
		checkIndex(index);
		switch (dtype)
		{
			case DType::FLOAT32: return static_cast<Scalar>(gradAs<float>()[index]);
			case DType::FLOAT64: return static_cast<Scalar>(gradAs<double>()[index]);
			case DType::INT32: return static_cast<Scalar>(gradAs<std::int32_t>()[index]);
			case DType::INT64: return static_cast<Scalar>(gradAs<std::int64_t>()[index]);
		}
		throw std::runtime_error("Invalid tensor dtype");
	}

	void Tensor::writeData(std::size_t index, Scalar value)
	{
		checkIndex(index);
		switch (dtype)
		{
			case DType::FLOAT32: dataAs<float>()[index] = static_cast<float>(value); break;
			case DType::FLOAT64: dataAs<double>()[index] = static_cast<double>(value); break;
			case DType::INT32: dataAs<std::int32_t>()[index] = static_cast<std::int32_t>(value); break;
			case DType::INT64: dataAs<std::int64_t>()[index] = static_cast<std::int64_t>(value); break;
		}
	}

	void Tensor::writeGrad(std::size_t index, Scalar value)
	{
		checkIndex(index);
		switch (dtype)
		{
			case DType::FLOAT32: gradAs<float>()[index] = static_cast<float>(value); break;
			case DType::FLOAT64: gradAs<double>()[index] = static_cast<double>(value); break;
			case DType::INT32: gradAs<std::int32_t>()[index] = static_cast<std::int32_t>(value); break;
			case DType::INT64: gradAs<std::int64_t>()[index] = static_cast<std::int64_t>(value); break;
		}
	}

	void Tensor::fillStorageWithZeros(void *buffer)
	{
		switch (dtype)
		{
			case DType::FLOAT32: fillZeros(static_cast<float *>(buffer), size); break;
			case DType::FLOAT64: fillZeros(static_cast<double *>(buffer), size); break;
			case DType::INT32: fillZeros(static_cast<std::int32_t *>(buffer), size); break;
			case DType::INT64: fillZeros(static_cast<std::int64_t *>(buffer), size); break;
		}
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
