#include "ffi.hpp"
#include <new>
#include <string>
#include <vector>
#include "Core.hpp"
#include "runtime.hpp"

using PHP2xAI::Runtime::CPP::Core;
using PHP2xAI::Runtime::CPP::GraphRuntime;
using PHP2xAI::Runtime::CPP::Scalar;
using PHP2xAI::Runtime::CPP::json;

struct PHP2xAI_Core
{
	Core *core = nullptr;
};

struct PHP2xAI_Runtime
{
	GraphRuntime *runtime = nullptr;
	std::vector<int> shapeBuffer;
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

	PHP2xAI_Runtime* php2xai_runtime_create(const char* graph_json)
	{
		if (!graph_json)
			return nullptr;
		try
		{
			auto *handle = new PHP2xAI_Runtime();
			json graphDef = json::parse(std::string(graph_json));
			handle->runtime = new GraphRuntime(graphDef, "");
			return handle;
		}
		catch (...)
		{
			return nullptr;
		}
	}

	void php2xai_runtime_destroy(PHP2xAI_Runtime* runtime)
	{
		if (!runtime)
			return;
		delete runtime->runtime;
		runtime->runtime = nullptr;
		delete runtime;
	}

	int php2xai_runtime_forward(PHP2xAI_Runtime* runtime)
	{
		if (!runtime || !runtime->runtime)
			return 1;
		try
		{
			runtime->runtime->forward();
		}
		catch (...)
		{
			return 2;
		}
		return 0;
	}

	int php2xai_runtime_backward(PHP2xAI_Runtime* runtime)
	{
		if (!runtime || !runtime->runtime)
			return 1;
		try
		{
			runtime->runtime->backward();
		}
		catch (...)
		{
			return 2;
		}
		return 0;
	}

	int* php2xai_runtime_get_tensor_shape(PHP2xAI_Runtime* runtime, int id)
	{
		if (!runtime || !runtime->runtime)
			return nullptr;
		try
		{
			const auto &tensor = runtime->runtime->getTensor(id);
			runtime->shapeBuffer = tensor.shape;
			if (runtime->shapeBuffer.empty())
				return nullptr;
			return runtime->shapeBuffer.data();
		}
		catch (...)
		{
			return nullptr;
		}
	}

	int php2xai_runtime_get_tensor_data(PHP2xAI_Runtime* runtime, int id, float* out, int n)
	{
		if (!runtime || !runtime->runtime || !out)
			return 1;
		if (n < 0)
			return 2;
		try
		{
			const auto &tensor = runtime->runtime->getTensor(id);
			const auto size = tensor.data.size();
			if (static_cast<std::size_t>(n) != size)
				return 3;
			for (int i = 0; i < n; ++i)
				out[i] = tensor.data[static_cast<std::size_t>(i)];
		}
		catch (...)
		{
			return 4;
		}
		return 0;
	}

	int php2xai_runtime_get_tensor_grad(PHP2xAI_Runtime* runtime, int id, float* out, int n)
	{
		if (!runtime || !runtime->runtime || !out)
			return 1;
		if (n < 0)
			return 2;
		try
		{
			const auto &tensor = runtime->runtime->getTensor(id);
			const auto size = tensor.grad.size();
			if (static_cast<std::size_t>(n) != size)
				return 3;
			for (int i = 0; i < n; ++i)
				out[i] = tensor.grad[static_cast<std::size_t>(i)];
		}
		catch (...)
		{
			return 4;
		}
		return 0;
	}
}
