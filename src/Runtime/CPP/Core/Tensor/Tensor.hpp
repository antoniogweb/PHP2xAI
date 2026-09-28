#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "../runtime.hpp"
#include "TensorBuffer.hpp"

namespace PHP2xAI::Runtime::CPP
{
	// Tensor owns typed data and gradient buffers while exposing typed views to
	// kernels. Its dtype and shape come from the serialized graph definition.
	struct Tensor
	{
		int id;
		DType dtype;
		std::string kind;
		std::string name;
		std::vector<int> shape;
		std::vector<int> strides;
		bool requiresGrad;
		void *data;
		void *grad;
		TensorBuffer dataOwner;
		TensorBuffer gradOwner;
		std::size_t size;

		Tensor();

		void allocate(DType tensorDType, std::size_t elementSize);

		template <typename T>
		T *dataAs()
		{
			return static_cast<T *>(data);
		}

		template <typename T>
		const T *dataAs() const
		{
			return static_cast<const T *>(data);
		}

		template <typename T>
		T *gradAs()
		{
			return static_cast<T *>(grad);
		}

		template <typename T>
		const T *gradAs() const
		{
			return static_cast<const T *>(grad);
		}

		Scalar readData(std::size_t index) const;
		Scalar readGrad(std::size_t index) const;
		void writeData(std::size_t index, Scalar value);
		void writeGrad(std::size_t index, Scalar value);
		void fillStorageWithZeros(void *buffer);
		void fillGradWithZeros();
		void checkIndex(std::size_t index) const;
	};
}
