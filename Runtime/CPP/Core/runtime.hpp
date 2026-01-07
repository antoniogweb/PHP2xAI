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
	};

	struct Op
	{
		int id{};
		std::string op;
		std::vector<int> inputs;
		int output{};
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
		int accSteps = 0;

		Scalar getLoss() const;
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

		explicit GraphRuntime(const json &graphDef, const std::string &weightsPath = "");

	private:
		std::string graphPath_;
		json graphDef_;

		void opMatmul(int, int, int);
		void opAdd(int aId, int bId, int outId);
		void opSub(int, int, int);
		void opDot(int, int, int);
		void opDropout(int, int);
		void opSig(int, int);
		void opRelu(int, int);
		void opLRelu(int, int);
		void opMse(int, int);
		void opMae(int, int);
		void opSoftmax(int, int);
		void opCe(int, int, int);
		void opCeLogits(int, int, int);
		void opCeLogitsLabelInt(int, int, int);

		void backwardMatmul(int, int, int);
		void backwardAdd(int, int, int);
		void backwardSub(int, int, int);
		void backwardDot(int, int, int);
		void backwardDropout(int, int);
		void backwardSig(int, int);
		void backwardRelu(int, int);
		void backwardLRelu(int, int);
		void backwardMse(int, int);
		void backwardMae(int, int);
		void backwardSoftmax(int, int);
		void backwardCe(int, int, int);
		void backwardCeLogits(int, int, int);
		void backwardCeLogitsLabelInt(int, int, int);

		static json loadJson(const std::string &path);
		void loadTensors(const json &graphDef, const json *weightsDef);
		void loadOps(const json &graphDef);
	};
}
