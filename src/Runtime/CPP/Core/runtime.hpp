#pragma once

#include <functional>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include "../ThirdParty/nlohmann/json.hpp"
#include "../types.hpp"

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
		int baseOffset{};
		std::vector<int> strides;

		std::vector<int> computeStrides(const std::vector<int> &shape) const
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
			return strides == computeStrides(shape) && baseOffset == 0;
		}
	};

	struct Op
	{
		int id{};
		std::string op;
		std::vector<int> inputs;
		int output{};
		std::string kernel{};
		std::vector<int> axes;
	};

	class GraphRuntime
	{
	public:
		std::vector<Tensor> tensors;
		std::vector<Op> ops;
		int lossId{};
		std::vector<int> trainable;
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
		
		explicit GraphRuntime(const json &graphDef, const std::string &weightsPath = "");

	private:
		std::string graphPath_;
		json graphDef_;

		void opMatmul(int, int, int);
		void opAdd(int aId, int bId, int outId, const std::string &kernel);
		// void opSub(int, int, int);
		// void opDot(int, int, int);
		void opDropout(int, int);
		void opSig(int, int);
		void opRelu(int, int);
		// void opLRelu(int, int);
		// void opMse(int, int);
		// void opMae(int, int);
		void opMean(int, int, const std::string &kernel, const std::vector<int> &axes);
		void opSoftmax(int, int, const std::string &kernel, const std::vector<int> &axes);
		void opCe(int, int, int);
		void opCeLogits(int, int, int);
		void opCeLogitsLabelInt(int, int, int);

		void backwardMatmul(int, int, int);
		void backwardAdd(int, int, int, const std::string &kernel);
		// void backwardSub(int, int, int);
		// void backwardDot(int, int, int);
		void backwardDropout(int, int);
		void backwardSig(int, int);
		void backwardRelu(int, int);
		// void backwardLRelu(int, int);
		// void backwardMse(int, int);
		// void backwardMae(int, int);
		void backwardMean(int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardSoftmax(int, int, const std::string &kernel, const std::vector<int> &axes);
		void backwardCe(int, int, int);
		void backwardCeLogits(int, int, int);
		void backwardCeLogitsLabelInt(int, int, int);

		void ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C);
		void BACKWARD_ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C);

		void SOFTMAX_1D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_2D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_3D_LAST(Tensor &X, Tensor &Y);
		void SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis);
		void BACKWORD_SOFTMAX_1D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_2D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_3D_LAST(Tensor &X, Tensor &Y);
		void BACKWORD_SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis);
		
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

		std::vector<int> alignStridesToRank(
			const std::vector<int> &shape,
			const std::vector<int> &strides,
			int targetRank) const;

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
}
