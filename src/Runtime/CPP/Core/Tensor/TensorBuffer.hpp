#pragma once

#include <cstddef>

namespace PHP2xAI::Runtime::CPP
{
	// Owns one typed allocation. Keeping its deleter beside the pointer means the
	// matching delete[] is used when the Tensor is destroyed or resized.
	struct TensorBuffer
	{
		TensorBuffer();
		~TensorBuffer();

		TensorBuffer(const TensorBuffer &) = delete;
		TensorBuffer &operator=(const TensorBuffer &) = delete;

		TensorBuffer(TensorBuffer &&other);
		TensorBuffer &operator=(TensorBuffer &&other);

		template <typename T>
		void allocate(std::size_t count)
		{
			clear();
			if (count == 0)
				return;

			pointer = new T[count];
			deleter = &deleteArray<T>;
		}

		void *data();
		const void *data() const;

	private:
		void *pointer;
		void (*deleter)(void *);

		template <typename T>
		static void deleteArray(void *memory)
		{
			delete[] static_cast<T *>(memory);
		}

		void clear();
	};
}
