#pragma once

#include <cstddef>

extern "C" {
	struct PHP2xAI_Core;

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
}
