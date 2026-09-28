#include "TensorBuffer.hpp"

namespace PHP2xAI::Runtime::CPP
{
	TensorBuffer::TensorBuffer() : pointer(0), deleter(0)
	{
	}

	TensorBuffer::~TensorBuffer()
	{
		clear();
	}

	TensorBuffer::TensorBuffer(TensorBuffer &&other)
		: pointer(other.pointer), deleter(other.deleter)
	{
		other.pointer = 0;
		other.deleter = 0;
	}

	TensorBuffer &TensorBuffer::operator=(TensorBuffer &&other)
	{
		if (this != &other)
		{
			clear();
			pointer = other.pointer;
			deleter = other.deleter;
			other.pointer = 0;
			other.deleter = 0;
		}
		return *this;
	}

	void *TensorBuffer::data()
	{
		return pointer;
	}

	const void *TensorBuffer::data() const
	{
		return pointer;
	}

	void TensorBuffer::clear()
	{
		if (pointer != 0 && deleter != 0)
			deleter(pointer);
		pointer = 0;
		deleter = 0;
	}
}
