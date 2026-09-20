#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>
#include "../ThirdParty/nlohmann/json.hpp"
#include "../types.hpp"
#include "../Utility/Profiler.hpp"

#include <Eigen/Dense>

namespace PHP2xAI::Runtime::CPP
{
	using nlohmann::json;

	struct Tensor
	{
		int id{};
		std::vector<Scalar> data;
		std::vector<Scalar> grad;
		std::vector<int> shape;
		std::string name;
		std::string kind;
		bool requiresGrad{};
		int baseOffset{};
		std::vector<int> strides;

		static std::vector<int> computeStrides(const std::vector<int> &shape)
		{
			const int rank = static_cast<int>(shape.size());
			std::vector<int> stridesLocal(rank, 0);
			int acc = 1;

			for (int a = rank - 1; a >= 0; --a)
			{
				stridesLocal[a] = acc;
				acc *= shape[a];
			}

			return stridesLocal;
		}

		int offset(const std::vector<int> &indices) const
		{
			const int rank = static_cast<int>(shape.size());

			if (static_cast<int>(indices.size()) != rank)
				throw std::runtime_error("Wrong rank: expected " + std::to_string(rank) + " indices");

			int off = 0;

			for (int a = 0; a < rank; ++a)
			{
				const int i = indices[a];
				const int d = shape[a];

				if (i < 0 || i >= d)
					throw std::runtime_error("Index out of bounds at axis " + std::to_string(a)
						+ ": " + std::to_string(i) + " (dim=" + std::to_string(d) + ")");

				off += i * strides[a];
			}

			return off;
		}

		Scalar get(const std::vector<int> &indices) const
		{
			const int off = offset(indices);
			return data[static_cast<std::size_t>(off)];
		}

		void set(const std::vector<int> &indices, Scalar value)
		{
			const int off = offset(indices);
			data[static_cast<std::size_t>(off)] = value;
		}

		int getRank() const
		{
			return static_cast<int>(shape.size());
		}

		bool isContiguous() const
		{
			return strides == Tensor::computeStrides(shape) && baseOffset == 0;
		}
	};

	struct Op
	{
		int id{};
		std::string op;
		std::vector<int> inputs;
		int output{};
		std::vector<int> outputs;
		int layer{-1};
		std::string kernel{};
		int padId{};
		std::vector<int> axes;
		Scalar dropoutPerc{50.0f};
		Scalar scale{1.0f};
		int start{};
		int end{};
		int offset{};
		Scalar base{10000.0f};
	};

	// Controls execution-specific behavior such as dropout and, later, GPT KV-cache handling.
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
		virtual ~GraphRuntime() = default;
		std::vector<Tensor> tensors;
		std::vector<Op> ops;
		int lossId{};
		std::vector<int> trainable;
		std::unordered_map<int, std::vector<Scalar>> dropoutMasks;
		ExecutionMode mode_ = ExecutionMode::INFER;
		int inputId{};
		int targetId{};
		int outputId{};

		std::vector<Scalar> getLoss() const;
		Scalar getError() const;
		std::vector<Scalar> getOutput() const;
		void setInput(const std::vector<Scalar> &x);
		void setTarget(const std::vector<Scalar> &y);
		void resetGrad();
		void saveWeightsToJson(const std::string &path) const;
		void saveToJson(const std::string &path) const;

		void forward();
		void backward();

		Tensor &getTensor(int id);
		const Tensor &getTensor(int id) const;
		
		void setLossGrad(Scalar lossGrad = 1.0f);
		void setMode(ExecutionMode mode);
		void enableProfiler();
		bool isProfilingEnabled() const;
		void setProfilingEnabled(bool enabled);
		Profiler &getProfiler();
		
		explicit GraphRuntime(const json &graphDef, const std::string &weightsPath = "");

	private:
		std::string graphPath_;
		json graphDef_;
		std::unique_ptr<Profiler> profiler_;
		bool profilingEnabled_ = false;
		std::uint64_t dropoutSeed_ = 0x9e3779b97f4a7c15ULL;
		struct KVCacheSlot { std::vector<Scalar> key, value; std::vector<int> shape; };
		std::unordered_map<int, KVCacheSlot> kvCaches_;
		int ropeOffset_ = -1;
		int ropeOffsetIncrement_ = 1;
		bool hasKvCacheInForward_ = false;

		void opMatmul(int, int, int, const std::string &kernel);
		void opLayerNorm(int inputId, int gammaId, int betaId, int outId, const std::string &kernel, const std::vector<int> &axes);
		void opApplyCausalMask(int inputId, int outId);

		void opEmbeddings(int xIdsId, int embeddingsId, int outId);
		void opEmbeddingsMeanPooling(int xIdsId, int embeddingsId, int outId, int padId);
		void opMeanPooling(int inputId, int maskId, int outId);
		void opPaddingMask(int inputId, int outId, int padId);
		void opApplyPaddingMask(int inputId, int maskId, int outId);
		void opScale(int inputId, int outId, Scalar scale);
		virtual void opGelu(int inputId, int outId);
		void opPositionalEncoding(int inputId, int outId);
		void opReshape(int inputId, int outId);
		void opSlice(int inputId, int outId, const std::string &kernel, const std::vector<int> &axes, int start, int end);
		void opTranspose(int inputId, int outId, const std::string &kernel, const std::vector<int> &axes);
		void opAdd(int aId, int bId, int outId, const std::string &kernel);
		// void opSub(int, int, int);
		// void opDot(int, int, int);
		void opDropout(int, int, Scalar);
		void opSig(int, int);
		void opRelu(int, int);
		// void opLRelu(int, int);
		// void opMse(int, int);
		// void opMae(int, int);
		void opMean(int, int, const std::string &kernel, const std::vector<int> &axes);
		void opKvCache(int, int, int, int, int);
		void opRope(int, int, const std::string &kernel, const std::vector<int> &axes, int offset, Scalar base);
		void opSoftmax(int, int, const std::string &kernel, const std::vector<int> &axes);
		void opCe(int, int, int, const std::string &kernel, const std::vector<int> &axes);
		void opCeLogits(int, int, int, const std::string &kernel, const std::vector<int> &axes);
		void opCeLogitsLabelInt(int, int, int, const std::string &kernel, const std::vector<int> &axes);

		void backwardMatmul(int, int, int, const std::string &kernel);
		void backwardLayerNorm(int inputId, int gammaId, int betaId, int outId, const std::string &kernel, const std::vector<int> &axes);
		void backwardScale(int inputId, int outId, Scalar scale);
		virtual void backwardGelu(int inputId, int outId);
		void backwardPositionalEncoding(int inputId, int outId);
		void backwardReshape(int inputId, int outId);
		void backwardSlice(int inputId, int outId, const std::string &kernel, const std::vector<int> &axes, int start, int end);
		void backwardTranspose(int inputId, int outId, const std::string &kernel, const std::vector<int> &axes);
		void backwardAdd(int, int, int, const std::string &kernel);
		void backwardApplyCausalMask(int inputId, int outId);

		void backwardEmbeddings(int xIdsId, int embeddingsId, int outId);
		void backwardEmbeddingsMeanPooling(int xIdsId, int embeddingsId, int outId, int padId);
		void backwardMeanPooling(int inputId, int maskId, int outId);
		void backwardPaddingMask(int inputId, int outId);
		void backwardApplyPaddingMask(int inputId, int maskId, int outId);
		// void backwardSub(int, int, int);
		// void backwardDot(int, int, int);
		void backwardDropout(int, int);
		void backwardSig(int, int);
		void backwardRelu(int, int);
		// void backwardLRelu(int, int);
		// void backwardMse(int, int);
		// void backwardMae(int, int);
		void backwardMean(int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardKvCache(int, int, int, int);
		void backwardRope(int, int, const std::string &kernel, const std::vector<int> &axes, int offset, Scalar base);
		void backwardSoftmax(int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardCe(int, int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardCeLogits(int, int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardCeLogitsLabelInt(int, int, int, const std::string &kernel, const std::vector<int> &axes);

		void ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C);
		
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

		void LAYER_NORM_LAST_AXIS(Tensor &X, Tensor &Gamma, Tensor &Beta, Tensor &Y);
		void LAYER_NORM_GENERIC(Tensor &X, Tensor &Gamma, Tensor &Beta, Tensor &Y, const std::vector<int> &axes);
		void BACKWARD_LAYER_NORM_LAST_AXIS(Tensor &X, Tensor &Gamma, Tensor &Beta, Tensor &Y);
		void BACKWARD_LAYER_NORM_GENERIC(Tensor &X, Tensor &Gamma, Tensor &Beta, Tensor &Y, const std::vector<int> &axes);

		void TRANSPOSE_2D(Tensor &A, Tensor &C);
		void TRANSPOSE_3D_LAST_TWO(Tensor &A, Tensor &C);
		void TRANSPOSE_4D_LAST_TWO(Tensor &A, Tensor &C);
		void TRANSPOSE_4D_AXIS_1_2(Tensor &A, Tensor &C);
		void TRANSPOSE_GENERIC(Tensor &A, Tensor &C, const std::vector<int> &axes);
		void BACKWARD_TRANSPOSE_2D(Tensor &A, Tensor &C);
		void BACKWARD_TRANSPOSE_3D_LAST_TWO(Tensor &A, Tensor &C);
		void BACKWARD_TRANSPOSE_4D_LAST_TWO(Tensor &A, Tensor &C);
		void BACKWARD_TRANSPOSE_4D_AXIS_1_2(Tensor &A, Tensor &C);
		void BACKWARD_TRANSPOSE_GENERIC(Tensor &A, Tensor &C, const std::vector<int> &axes);
		void SLICE_LAST(Tensor &A, Tensor &C, int start, int end);
		void SLICE_GENERIC_AXIS(Tensor &A, Tensor &C, int axis, int start, int end);
		void BACKWARD_SLICE_LAST(Tensor &A, Tensor &C, int start, int end);
		void BACKWARD_SLICE_GENERIC_AXIS(Tensor &A, Tensor &C, int axis, int start, int end);

		void ROPE_INTERLEAVED_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		void ROPE_ROTATE_HALF_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		void ROPE_INTERLEAVED_GENERIC(Tensor &X, Tensor &Y, int positionAxis, int rotationAxis, int offset, Scalar base);
		void ROPE_ROTATE_HALF_GENERIC(Tensor &X, Tensor &Y, int positionAxis, int rotationAxis, int offset, Scalar base);
		void BACKWARD_ROPE_INTERLEAVED_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		void BACKWARD_ROPE_ROTATE_HALF_LAST_TWO(Tensor &X, Tensor &Y, int offset, Scalar base);
		void BACKWARD_ROPE_INTERLEAVED_GENERIC(Tensor &X, Tensor &Y, int positionAxis, int rotationAxis, int offset, Scalar base);
		void BACKWARD_ROPE_ROTATE_HALF_GENERIC(Tensor &X, Tensor &Y, int positionAxis, int rotationAxis, int offset, Scalar base);

		void ropeLastTwo(Tensor &X, Tensor &Y, int offset, Scalar base, bool rotateHalf, bool backward);
		void ropeGeneric(Tensor &X, Tensor &Y, int positionAxis, int rotationAxis, int offset, Scalar base, bool rotateHalf, bool backward);

		void SOFTMAX_1D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_2D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_3D_LAST(Tensor &X, Tensor &Y);
		virtual void SOFTMAX_4D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis);
		void BACKWORD_SOFTMAX_1D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_2D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_3D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_4D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis);

		void CE_1D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void CE_2D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void CE_3D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void CE_GENERIC_AXIS(Tensor &pred, Tensor &target, Tensor &out, int axis);
		void CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &out, int axis);
		void BACKWORD_CE_1D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void BACKWORD_CE_2D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void BACKWORD_CE_3D_LAST(Tensor &pred, Tensor &target, Tensor &out);
		void BACKWORD_CE_GENERIC_AXIS(Tensor &pred, Tensor &target, Tensor &out, int axis);
		void BACKWORD_CE_LOGITS_1D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_2D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_3D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &out, int axis);

		void CE_LOGITS_LABEL_INT_1D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_LABEL_INT_2D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_LABEL_INT_3D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &out, int axis);
		void BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST(Tensor &logits, Tensor &target, Tensor &out);
		void BACKWORD_CE_LOGITS_LABEL_INT_GENERIC_AXIS(Tensor &logits, Tensor &target, Tensor &out, int axis);
		
		void MEAN_1D_FIRST(Tensor &A, Tensor &out);
		void MEAN_2D_FIRST(Tensor &A, Tensor &out);
		void MEAN_3D_FIRST(Tensor &A, Tensor &out);
		void MEAN_GENERIC_AXIS(Tensor &A, Tensor &out, int axis);
		void BACKWARD_MEAN_1D_FIRST(Tensor &A, Tensor &out);
		void BACKWARD_MEAN_2D_FIRST(Tensor &A, Tensor &out);
		void BACKWARD_MEAN_3D_FIRST(Tensor &A, Tensor &out);
		void BACKWARD_MEAN_GENERIC_AXIS(Tensor &A, Tensor &out, int axis);

		void softmaxAlongAxisInPlace(
			std::vector<Scalar> &data,
			const std::vector<int> &shape,
			const std::vector<int> &strides,
			int axis = -1) const;

		static json loadJson(const std::string &path);
		void loadTensors(const json &graphDef, const json *weightsDef);
		void loadOps(const json &graphDef);

		template <class Callback>
		void forEachSliceAlongAxisIncremental(
			const std::vector<int> &shape,
			const std::vector<int> &strides,
			int axis,
			Callback onSlice) const;

		void addAlongAxisInPlace(
			std::vector<Scalar> &zData,
			const std::vector<int> &zShape,
			const std::vector<int> &zStrides,
			const std::vector<Scalar> &xData,
			const std::vector<int> &xStrides,
			const std::vector<Scalar> &yData,
			const std::vector<int> &yStrides,
			int axis = -1) const;
	};

	class GraphRuntimeEigen final : public GraphRuntime
	{
	public:
		using GraphRuntime::GraphRuntime;

	private:
		void opGelu(int inputId, int outId) override;
		void backwardGelu(int inputId, int outId) override;
		void SOFTMAX_4D_LAST(Tensor &X, Tensor &Y) override;
		void MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C) override;
		void BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C) override;
	};
}
