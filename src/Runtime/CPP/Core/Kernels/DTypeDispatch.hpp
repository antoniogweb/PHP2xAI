#pragma once

#include <cstdint>
#include <stdexcept>

#include "../runtime.hpp"

namespace PHP2xAI::Runtime::CPP
{
	template <typename Func>
	void dispatchDType(DType dtype, Func &&func)
	{
		switch (dtype)
		{
			case DType::FLOAT32:
				func.template operator()<float>();
				break;
			case DType::FLOAT64:
				func.template operator()<double>();
				break;
			case DType::INT32:
				func.template operator()<std::int32_t>();
				break;
			case DType::INT64:
				func.template operator()<std::int64_t>();
				break;
			default:
				throw std::invalid_argument("Unsupported dtype");
		}
	}
}
