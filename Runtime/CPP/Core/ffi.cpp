#include "ffi.hpp"
#include <new>
#include <vector>
#include "Core.hpp"

using PHP2xAI::Runtime::CPP::Core;
using PHP2xAI::Runtime::CPP::Scalar;

struct PHP2xAI_Core
{
	Core *core = nullptr;
};

extern "C" {
	PHP2xAI_Core* php2xai_core_create(const char* model_path, const char* weights_path)
	{
		try
		{
			auto *handle = new PHP2xAI_Core();
			handle->core = new Core(model_path, weights_path ? weights_path : "");
			return handle;
		}
		catch (...)
		{
			return nullptr;
		}
	}

	void php2xai_core_destroy(PHP2xAI_Core* core)
	{
		if (!core)
			return;
		delete core->core;
		core->core = nullptr;
		delete core;
	}

	std::size_t php2xai_core_input_size(PHP2xAI_Core* core)
	{
		if (!core || !core->core)
			return 0;
		try
		{
			return core->core->inputSize();
		}
		catch (...)
		{
			return 0;
		}
	}

	std::size_t php2xai_core_output_size(PHP2xAI_Core* core)
	{
		if (!core || !core->core)
			return 0;
		try
		{
			return core->core->outputSize();
		}
		catch (...)
		{
			return 0;
		}
	}

	int php2xai_core_predict(
		PHP2xAI_Core* core,
		const float* x,
		std::size_t x_len,
		float* out,
		std::size_t out_len)
	{
		if (!core || !core->core || !x || !out)
			return 1;
		try
		{
			if (x_len != core->core->inputSize())
				return 2;
			if (out_len != core->core->outputSize())
				return 3;

			std::vector<Scalar> input(x, x + x_len);
			const auto output = core->core->predict(input);

			if (output.size() != out_len)
				return 4;

			for (std::size_t i = 0; i < out_len; ++i)
				out[i] = output[i];
		}
		catch (...)
		{
			return 5;
		}
		return 0;
	}

	int php2xai_core_predict_label_int(
		PHP2xAI_Core* core,
		const float* x,
		std::size_t x_len,
		int* out_label)
	{
		if (!core || !core->core || !x || !out_label)
			return 1;
		try
		{
			if (x_len != core->core->inputSize())
				return 2;

			std::vector<Scalar> input(x, x + x_len);
			*out_label = core->core->predictLabelInt(input);
		}
		catch (...)
		{
			return 3;
		}
		return 0;
	}
}
