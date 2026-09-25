#pragma once

#include <cstddef>

extern "C" {
	struct PHP2xAI_Core;
	struct PHP2xAI_Runtime;

	PHP2xAI_Core* php2xai_core_create(const char* model_path, const char* weights_path);
	void php2xai_core_destroy(PHP2xAI_Core* core);

	std::size_t php2xai_core_input_size(PHP2xAI_Core* core);
	std::size_t php2xai_core_output_size(PHP2xAI_Core* core);

	int php2xai_core_predict(
		PHP2xAI_Core* core,
		const float* x,
		std::size_t x_len,
		float* out,
		std::size_t out_len);

	int php2xai_core_predict_label_int(
		PHP2xAI_Core* core,
		const float* x,
		std::size_t x_len,
		int* out_label);

	PHP2xAI_Runtime* php2xai_runtime_create(const char* graph_json);
	void php2xai_runtime_destroy(PHP2xAI_Runtime* runtime);
	int php2xai_runtime_forward(PHP2xAI_Runtime* runtime);
	int php2xai_runtime_backward(PHP2xAI_Runtime* runtime);
	int* php2xai_runtime_get_tensor_shape(PHP2xAI_Runtime* runtime, int id);
	int php2xai_runtime_get_tensor_data(PHP2xAI_Runtime* runtime, int id, float* out, int n);
	int php2xai_runtime_get_tensor_grad(PHP2xAI_Runtime* runtime, int id, float* out, int n);
}
