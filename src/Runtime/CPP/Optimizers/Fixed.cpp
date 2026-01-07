#include "Fixed.hpp"
#include "../Core/runtime.hpp"

namespace PHP2xAI::Runtime::CPP::Optimizers
{
	Fixed::Fixed(Scalar learningRate) : learningRate_(learningRate)
	{
	}

	void Fixed::step(GraphRuntime& /*graph*/)
	{
		// Intentionally left blank: matches the PHP Fixed optimizer stub.
	}
}
