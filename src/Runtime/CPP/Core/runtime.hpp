#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../types.hpp"
#include "../Utility/Profiler.hpp"

namespace PHP2xAI::Runtime::CPP
{
	using nlohmann::json;

	// Tensor is implemented in runtime.cpp; kernels only need references to it.
	struct Tensor;

	// The numeric values match Tensor.php and the HDF5 dataset format.
	enum class DType : int
	{
		FLOAT32 = 1,
		FLOAT64 = 2,
		INT32 = 3,
		INT64 = 4
	};

	enum class ExecutionMode
	{
		TRAIN,
		INFER,
		PREFILL,
		DECODE
	};

	class GraphRuntime
	{
	public:
		explicit GraphRuntime(const json &graphDef, const std::string &weightsPath = "");
		virtual ~GraphRuntime();

		GraphRuntime(const GraphRuntime &) = delete;
		GraphRuntime &operator=(const GraphRuntime &) = delete;

		// Execute the operations registered in the graph.
		void forward();
		void backward();

		// Graph input/output helpers used by Core and the FFI layer.
		std::size_t inputSize() const;
		std::size_t outputSize() const;
		void setInput(const std::vector<Scalar> &values);
		void setTarget(const std::vector<Scalar> &values);
		std::vector<Scalar> getOutput() const;
		std::vector<Scalar> getLoss() const;
		Scalar getError() const;

		// Tensor inspection and optimizer access. Tensor's storage type stays private.
		std::vector<int> getTensorShape(int id) const;
		std::size_t getTensorSize(int id) const;
		int getTensorDType(int id) const;
		Scalar getTensorDataValue(int id, std::size_t index) const;
		Scalar getTensorGradValue(int id, std::size_t index) const;
		void setTensorDataValue(int id, std::size_t index, Scalar value);
		bool tensorHasFloatingPointDType(int id) const;
		const std::vector<int> &getTrainableTensorIds() const;

		void resetGrad();
		void setLossGrad(Scalar lossGrad = 1.0f);
		void saveWeightsToJson(const std::string &path) const;
		void saveToJson(const std::string &path) const;

		void setMode(ExecutionMode mode);
		void resetKvCache(int layer = -1);
		void enableProfiler();
		bool isProfilingEnabled() const;
		void setProfilingEnabled(bool enabled);
		Profiler &getProfiler();

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;

		void opAdd(int aId, int bId, int outId, const std::string &kernel);
		void backwardAdd(int aId, int bId, int outId, const std::string &kernel);
		void opMatmul(int aId, int bId, int outId, const std::string &kernel);
		void backwardMatmul(int aId, int bId, int outId, const std::string &kernel);
		void opRelu(int inputId, int outputId);
		void backwardRelu(int inputId, int outputId);
		void opGelu(int inputId, int outputId);
		void backwardGelu(int inputId, int outputId);
		void opSilu(int inputId, int outputId);
		void backwardSilu(int inputId, int outputId);
		void opEmbeddings(int idsId, int tableId, int outputId);
		void backwardEmbeddings(int idsId, int tableId, int outputId);
		void opEmbeddingsMeanPooling(int idsId, int tableId, int outputId, int padId);
		void backwardEmbeddingsMeanPooling(int idsId, int tableId, int outputId, int padId);
		void opMeanPooling(int inputId, int maskId, int outputId);
		void backwardMeanPooling(int inputId, int maskId, int outputId);
		void opPaddingMask(int inputId, int outputId, int padId);
		void opDropout(int inputId, int outputId, Scalar dropoutPerc);
		void backwardDropout(int inputId, int outputId);
		void opCeLogitsLabelInt(int logitsId, int targetId, int outputId, const std::string &kernel);
		void backwardCeLogitsLabelInt(int logitsId, int targetId, int outputId, const std::string &kernel);
		void opMean(int inputId, int outputId, const std::string &kernel);
		void backwardMean(int inputId, int outputId, const std::string &kernel);
		void opSoftmax(int inputId, int outputId, const std::string &kernel);
		void backwardSoftmax(int inputId, int outputId, const std::string &kernel);
		void opMultiply(int aId, int bId, int outputId);
		void backwardMultiply(int aId, int bId, int outputId);
		void opScale(int inputId, int outputId, Scalar scale);
		void backwardScale(int inputId, int outputId, Scalar scale);
		void opSig(int inputId, int outputId);
		void backwardSig(int inputId, int outputId);
		void opReshape(int inputId, int outputId);
		void backwardReshape(int inputId, int outputId);
		void opTranspose(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes);
		void backwardTranspose(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes);
		void opSlice(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes, int start, int end);
		void backwardSlice(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes, int start, int end);
		void opPositionalEncoding(int inputId, int outputId);
		void backwardPositionalEncoding(int inputId, int outputId);
		void opApplyPaddingMask(int inputId, int maskId, int outputId);
		void backwardApplyPaddingMask(int inputId, int maskId, int outputId);
		void opApplyCausalMask(int inputId, int outputId);
		void backwardApplyCausalMask(int inputId, int outputId);
		void opLayerNorm(int inputId, int gammaId, int betaId, int outputId, const std::string &kernel);
		void backwardLayerNorm(int inputId, int gammaId, int betaId, int outputId, const std::string &kernel);
		void opRMSNorm(int inputId, int gammaId, int outputId, const std::string &kernel, Scalar eps);
		void backwardRMSNorm(int inputId, int gammaId, int outputId, const std::string &kernel, Scalar eps);
		void opRope(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes, int offset, Scalar base);
		void backwardRope(int inputId, int outputId, const std::string &kernel, const std::vector<int> &axes, int offset, Scalar base);
		void opCe(int predId, int targetId, int outputId, const std::string &kernel);
		void backwardCe(int predId, int targetId, int outputId, const std::string &kernel);
		void opCeLogits(int logitsId, int targetId, int outputId, const std::string &kernel);
		void backwardCeLogits(int logitsId, int targetId, int outputId, const std::string &kernel);
		void opKvCache(int keyId, int valueId, int keyOutputId, int valueOutputId, int layer);
		void backwardKvCache(int keyId, int valueId, int keyOutputId, int valueOutputId);

	protected:
		// Backend-specific kernel entry points. GraphRuntime provides NAIVE;
		// Eigen and CUDA runtimes can override these methods later.
		virtual void ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);

		virtual void MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C);
		void MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C);

		virtual void RELU(Tensor &X, Tensor &Y);
		virtual void BACKWARD_RELU(Tensor &X, Tensor &Y);
		virtual void GELU(Tensor &X, Tensor &Y);
		virtual void BACKWARD_GELU(Tensor &X, Tensor &Y);
		virtual void SILU(Tensor &X, Tensor &Y);
		virtual void BACKWARD_SILU(Tensor &X, Tensor &Y);
		virtual void EMBEDDINGS(Tensor &ids, Tensor &table, Tensor &output);
		virtual void BACKWARD_EMBEDDINGS(Tensor &ids, Tensor &table, Tensor &output);
		virtual void EMBEDDINGS_MEAN_POOLING(Tensor &ids, Tensor &table, Tensor &output, int padId);
		virtual void BACKWARD_EMBEDDINGS_MEAN_POOLING(Tensor &ids, Tensor &table,
			Tensor &output, int padId);
		virtual void MEAN_POOLING(Tensor &input, Tensor &mask, Tensor &output);
		virtual void BACKWARD_MEAN_POOLING(Tensor &input, Tensor &mask, Tensor &output);
		virtual void PADDING_MASK(Tensor &ids, Tensor &output, int padId);
		virtual void DROPOUT(Tensor &input, Tensor &output, Scalar dropoutPerc,
			Scalar *mask, std::uint64_t seed, bool training);
		virtual void BACKWARD_DROPOUT(Tensor &input, Tensor &output, const Scalar *mask);
		virtual void CE_LOGITS_LABEL_INT_1D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_LABEL_INT_2D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_LABEL_INT_3D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWORD_CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &output);
		virtual void MEAN_1D_FIRST(Tensor &input, Tensor &output);
		virtual void MEAN_2D_FIRST(Tensor &input, Tensor &output);
		virtual void MEAN_3D_FIRST(Tensor &input, Tensor &output);
		virtual void MEAN_GENERIC_AXIS(Tensor &input, Tensor &output);
		virtual void BACKWARD_MEAN_1D_FIRST(Tensor &input, Tensor &output);
		virtual void BACKWARD_MEAN_2D_FIRST(Tensor &input, Tensor &output);
		virtual void BACKWARD_MEAN_3D_FIRST(Tensor &input, Tensor &output);
		virtual void BACKWARD_MEAN_GENERIC_AXIS(Tensor &input, Tensor &output);
		virtual void SOFTMAX_1D_LAST(Tensor &input, Tensor &output);
		virtual void SOFTMAX_2D_LAST(Tensor &input, Tensor &output);
		virtual void SOFTMAX_3D_LAST(Tensor &input, Tensor &output);
		virtual void SOFTMAX_4D_LAST(Tensor &input, Tensor &output);
		virtual void SOFTMAX_GENERIC_AXIS(Tensor &input, Tensor &output);
		virtual void BACKWORD_SOFTMAX_1D_LAST(Tensor &input, Tensor &output);
		virtual void BACKWORD_SOFTMAX_2D_LAST(Tensor &input, Tensor &output);
		virtual void BACKWORD_SOFTMAX_3D_LAST(Tensor &input, Tensor &output);
		virtual void BACKWORD_SOFTMAX_4D_LAST(Tensor &input, Tensor &output);
		virtual void BACKWORD_SOFTMAX_GENERIC_AXIS(Tensor &input, Tensor &output);
		virtual void MULTIPLY(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MULTIPLY(Tensor &A, Tensor &B, Tensor &C);
		virtual void SCALE(Tensor &X, Tensor &Y, Scalar scale);
		virtual void BACKWARD_SCALE(Tensor &X, Tensor &Y, Scalar scale);
		virtual void SIG(Tensor &X, Tensor &Y);
		virtual void BACKWARD_SIG(Tensor &X, Tensor &Y);
		virtual void RESHAPE(Tensor &X, Tensor &Y);
		virtual void BACKWARD_RESHAPE(Tensor &X, Tensor &Y);
		virtual void TRANSPOSE_2D(Tensor &X, Tensor &Y);
		virtual void TRANSPOSE_3D_LAST_TWO(Tensor &X, Tensor &Y);
		virtual void TRANSPOSE_4D_LAST_TWO(Tensor &X, Tensor &Y);
		virtual void TRANSPOSE_4D_AXIS_1_2(Tensor &X, Tensor &Y);
		virtual void BACKWARD_TRANSPOSE_2D(Tensor &X, Tensor &Y);
		virtual void BACKWARD_TRANSPOSE_3D_LAST_TWO(Tensor &X, Tensor &Y);
		virtual void BACKWARD_TRANSPOSE_4D_LAST_TWO(Tensor &X, Tensor &Y);
		virtual void BACKWARD_TRANSPOSE_4D_AXIS_1_2(Tensor &X, Tensor &Y);
		virtual void SLICE_LAST(Tensor &X, Tensor &Y, int start, int end);
		virtual void BACKWARD_SLICE_LAST(Tensor &X, Tensor &Y, int start, int end);
		virtual void POSITIONAL_ENCODING(Tensor &X, Tensor &Y);
		virtual void BACKWARD_POSITIONAL_ENCODING(Tensor &X, Tensor &Y);
		virtual void APPLY_PADDING_MASK(Tensor &X, Tensor &mask, Tensor &Y);
		virtual void BACKWARD_APPLY_PADDING_MASK(Tensor &X, Tensor &mask, Tensor &Y);
		virtual void APPLY_CAUSAL_MASK(Tensor &X, Tensor &Y);
		virtual void BACKWARD_APPLY_CAUSAL_MASK(Tensor &X, Tensor &Y);
		virtual void LAYER_NORM_LAST_AXIS(Tensor &X, Tensor &gamma, Tensor &beta, Tensor &Y);
		virtual void BACKWARD_LAYER_NORM_LAST_AXIS(Tensor &X, Tensor &gamma, Tensor &beta, Tensor &Y);
		virtual void RMS_NORM_LAST_AXIS(Tensor &X, Tensor &gamma, Tensor &Y, Scalar eps);
		virtual void BACKWARD_RMS_NORM_LAST_AXIS(Tensor &X, Tensor &gamma, Tensor &Y, Scalar eps);
		virtual void ROPE_INTERLEAVED_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		virtual void ROPE_ROTATE_HALF_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		virtual void BACKWARD_ROPE_INTERLEAVED_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		virtual void BACKWARD_ROPE_ROTATE_HALF_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		virtual void CE_1D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void CE_2D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void CE_3D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_1D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_2D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_3D_LAST(Tensor &pred, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &output);
		virtual void BACKWARD_CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &output);
	};

	// Eigen runtime starts with the same behavior as GraphRuntime. Kernel
	// methods are virtual on the base class so Eigen implementations can be
	// added one at a time without changing Core or the graph format.
	class GraphRuntimeEigen : public GraphRuntime
	{
	public:
		using GraphRuntime::GraphRuntime;

	protected:
		void MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C) override;
		void GELU(Tensor &X, Tensor &Y) override;
		void BACKWARD_GELU(Tensor &X, Tensor &Y) override;
		void SOFTMAX_4D_LAST(Tensor &X, Tensor &Y) override;
	};
}
