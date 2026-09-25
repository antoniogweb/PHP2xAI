#pragma once

#include <cstddef>
#include <vector>

#include "../runtime.hpp"

namespace PHP2xAI::Runtime::CPP
{
	// A non-owning view lets backend kernel files use tensor storage without
	// moving the private Tensor definition out of runtime.cpp.
	struct TensorAccess
	{
		DType dtype;
		const std::vector<int> &shape;
		bool requiresGrad;
		void *data;
		void *grad;
		std::size_t size;

		template <typename T>
		T *dataAs()
		{
			return static_cast<T *>(data);
		}

		template <typename T>
		T *gradAs()
		{
			return static_cast<T *>(grad);
		}
	};

	TensorAccess accessTensor(Tensor &tensor);
}
