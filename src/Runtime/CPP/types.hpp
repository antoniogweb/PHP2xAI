#pragma once

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
}
