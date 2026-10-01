#pragma once

#include <cstdint>
#include <type_traits>

namespace PHP2xAI::Runtime::CPP
{
	using Scalar = float;

	// Numeric values match Tensor.php and the HDF5 dataset format.
	enum class DType : int
	{
		FLOAT32 = 1,
		FLOAT64 = 2,
		INT32 = 3,
		INT64 = 4
	};

	template <typename T>
	constexpr DType dtypeOf()
	{
		using Value = typename std::remove_cv<T>::type;
		if constexpr (std::is_same<Value, float>::value)
			return DType::FLOAT32;
		else if constexpr (std::is_same<Value, double>::value)
			return DType::FLOAT64;
		else if constexpr (std::is_same<Value, std::int32_t>::value)
			return DType::INT32;
		else if constexpr (std::is_same<Value, std::int64_t>::value)
			return DType::INT64;
		else
			static_assert(!std::is_same<Value, Value>::value, "Unsupported tensor value type");
	}
}
