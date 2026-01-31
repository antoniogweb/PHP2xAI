#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <utility>
#include "runtime.hpp"

// enum OpType { SOFTMAX, REDUCE_MEAN, TRANSPOSE, MATMUL, ... };
// 
// struct SoftmaxAttr { int axis; };
// struct ReduceAttr { int axis; bool keepdims; };
// struct TransposeAttr { std::vector<int> axes; };
// 
// using OpAttr = std::variant<std::monostate, SoftmaxAttr, ReduceAttr, TransposeAttr, ...>;
// 
// struct OpNode {
//   OpType type;
//   std::vector<int> inputs;
//   std::vector<int> outputs;
//   OpAttr attr;
// };

namespace PHP2xAI::Runtime::CPP
{
	GraphRuntime::GraphRuntime(const json &graphDef, const std::string &weightsPath)
		: graphDef_(graphDef)
	{
		json weightsDef;
		const json *weightsPtr = nullptr;

		if (!weightsPath.empty())
		{
			weightsDef = loadJson(weightsPath);
			weightsPtr = &weightsDef;
		}

		loadTensors(graphDef_, weightsPtr);
		loadOps(graphDef_);

		if (graphDef_.contains("loss"))
			lossId = graphDef_.at("loss").get<int>();

		if (graphDef_.contains("output"))
			outputId = graphDef_.at("output").get<int>();

		if (graphDef_.contains("trainable"))
			trainable = graphDef_.at("trainable").get<std::vector<int>>();
	}

	void GraphRuntime::forward()
	{
		for (const auto &op : ops)
		{
			const auto &name = op.op;
			const auto &inputs = op.inputs;
			const auto outId = op.output;

			if (name == "matmul")
				opMatmul(inputs[0], inputs[1], outId);
			else if (name == "add")
				opAdd(inputs[0], inputs[1], outId);
			else if (name == "sub")
				opSub(inputs[0], inputs[1], outId);
			else if (name == "dot")
				opDot(inputs[0], inputs[1], outId);
			else if (name == "dropout")
				opDropout(inputs[0], outId);
			else if (name == "sig")
				opSig(inputs[0], outId);
			else if (name == "ReLU" || name == "relu")
				opRelu(inputs[0], outId);
			else if (name == "LReLU")
				opLRelu(inputs[0], outId);
			else if (name == "MSE")
				opMse(inputs[0], outId);
			else if (name == "MAE")
				opMae(inputs[0], outId);
			else if (name == "mean")
				opMean(inputs[0], outId);
			else if (name == "softmax")
				opSoftmax(inputs[0], outId);
			else if (name == "CE")
				opCe(inputs[0], inputs[1], outId);
			else if (name == "softmax_ce_logits")
				opCeLogits(inputs[0], inputs[1], outId);
			else if (name == "softmax_ce_logits_label_int")
				opCeLogitsLabelInt(inputs[0], inputs[1], outId);
			else
				throw std::runtime_error("Op not supported: " + name);
		}
	}

	void GraphRuntime::backward()
	{
		for (auto &tensor : tensors)
		{
			if (tensor.kind != "param")
				tensor.grad.assign(tensor.grad.size(), 0.0f);
		}

		setLossGrad(1.0f);

		for (int i = static_cast<int>(ops.size()) - 1; i >= 0; --i)
		{
			const auto &op = ops[static_cast<std::size_t>(i)];
			const auto &name = op.op;
			const auto &inputs = op.inputs;
			auto outId = op.output;

			if (name == "matmul")
				backwardMatmul(inputs[0], inputs[1], outId);
			else if (name == "add")
				backwardAdd(inputs[0], inputs[1], outId);
			else if (name == "sub")
				backwardSub(inputs[0], inputs[1], outId);
			else if (name == "dot")
				backwardDot(inputs[0], inputs[1], outId);
			else if (name == "dropout")
				backwardDropout(inputs[0], outId);
			else if (name == "sig")
				backwardSig(inputs[0], outId);
			else if (name == "relu" || name == "ReLU")
				backwardRelu(inputs[0], outId);
			else if (name == "LReLU")
				backwardLRelu(inputs[0], outId);
			else if (name == "MSE")
				backwardMse(inputs[0], outId);
			else if (name == "MAE")
				backwardMae(inputs[0], outId);
			else if (name == "mean")
				backwardMean(inputs[0], outId);
			else if (name == "softmax")
				backwardSoftmax(inputs[0], outId);
			else if (name == "CE")
				backwardCe(inputs[0], inputs[1], outId);
			else if (name == "softmax_ce_logits")
				backwardCeLogits(inputs[0], inputs[1], outId);
			else if (name == "softmax_ce_logits_label_int")
				backwardCeLogitsLabelInt(inputs[0], inputs[1], outId);
			else
				throw std::runtime_error("Op not supported: " + name);
		}
	}

	Tensor &GraphRuntime::getTensor(int id)
	{
		if (id < 0 || static_cast<std::size_t>(id) >= tensors.size())
			throw std::out_of_range("Tensor id out of range");

		return tensors[id];
	}

	const Tensor &GraphRuntime::getTensor(int id) const
	{
		if (id < 0 || static_cast<std::size_t>(id) >= tensors.size())
			throw std::out_of_range("Tensor id out of range");

		return tensors[id];
	}

	std::vector<Scalar> GraphRuntime::getLoss() const
	{
		const auto &tensor = tensors[lossId];
		return tensor.data;
	}
	
	Scalar GraphRuntime::getError() const
	{
		const auto &loss = tensors[lossId];
		
		if (loss.data.size() > 1)
		{
			Scalar mean = std::accumulate(loss.data.begin(), loss.data.end(), 0.0f)
			/ static_cast<Scalar>(loss.data.size());
			
			return mean;
		}
		else
			return loss.data[0];
	}
	
	std::vector<Scalar> GraphRuntime::getOutput() const
	{
		const auto &tensor = tensors[outputId];
		
		if (tensor.data.empty())
		{
			std::vector<Scalar> data;
			
			auto size = std::accumulate(
				tensor.shape.begin(),
				tensor.shape.end(),
				1,
				std::multiplies<int>());
			data.assign(size, 0.0f);
			
			return data;
		}
		else
			return tensor.data;
	}
	
	void GraphRuntime::setLossGrad(Scalar lossGrad)
	{
		if (lossId != 0)
		{
			auto &tensor = tensors[lossId];
		
			tensor.grad.assign(tensor.data.size(), lossGrad);
		}
	}
	
	void GraphRuntime::setInput(const std::vector<Scalar> &x)
	{
		auto &tensor = tensors[inputId];

		if (tensor.data.size() != x.size())
			throw std::runtime_error("Inserting incompatible dimensions");

		tensor.data = x;
	}

	void GraphRuntime::setTarget(const std::vector<Scalar> &y)
	{
		auto &tensor = tensors[targetId];

		if (tensor.data.size() != y.size())
			throw std::runtime_error("Inserting incompatible dimensions");

		tensor.data = y;
	}

	void GraphRuntime::resetGrad()
	{
		for (auto &tensor : tensors)
		{
			tensor.grad.assign(tensor.grad.size(), 0.0f);
		}
	}

	void GraphRuntime::saveToJson(const std::string &path) const
	{
		nlohmann::json tensorsJson = nlohmann::json::object();

		for (const auto &t : tensors)
		{
			tensorsJson[std::to_string(t.id)] = t.data;
		}

		nlohmann::json jsonArray = {
			{"graph", graphDef_},
			{"tensors", std::move(tensorsJson)}
		};

		std::ofstream file(path, std::ios::trunc);
		if (!file.is_open())
			throw std::runtime_error("Unable to open file for writing: " + path);

		file << jsonArray.dump();
	}

	void GraphRuntime::saveWeightsToJson(const std::string &path) const
	{
		nlohmann::json tensorsJson = nlohmann::json::object();

		for (const auto &t : tensors)
		{
			if (std::find(trainable.begin(), trainable.end(), t.id) != trainable.end())
			{
				tensorsJson[std::to_string(t.id)] = {
					{"data", t.data},
					{"shape", t.shape}
				};
			}
		}

		nlohmann::json jsonArray = {
			{"tensors", std::move(tensorsJson)}
		};

		std::ofstream file(path, std::ios::trunc);
		if (!file.is_open())
			throw std::runtime_error("Unable to open file for writing: " + path);

		file << jsonArray.dump();
	}

	// TODO: implement operations; currently stubbed to signal unimplemented functionality.
	void GraphRuntime::opMatmul(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (A.shape.size() != 2)
			throw std::runtime_error("matmul: left operand must be a matrix");

		if (B.shape.size() == 1)
		{
			const auto m = A.shape[0];
			const auto n = A.shape[1];

			if (B.shape[0] != n)
				throw std::runtime_error("matmul: dimension mismatch");

			C.shape = {m};
			C.data.assign(static_cast<std::size_t>(m), 0.0f);

			for (int i = 0; i < m; ++i)
			{
				Scalar sum = 0.0f;

				for (int k = 0; k < n; ++k)
					sum += A.data[static_cast<std::size_t>(i * n + k)] * B.data[static_cast<std::size_t>(k)];

				C.data[static_cast<std::size_t>(i)] = sum;
			}
		}
		else if (B.shape.size() == 2)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];
			const auto dimB = B.shape[0];
			const auto outDim = B.shape[1];

			if (dim != dimB)
				throw std::runtime_error("matmul: dimension mismatch");

			C.shape = {batch, outDim};
			C.data.assign(static_cast<std::size_t>(batch * outDim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto aRow = b * dim;
				const auto cRow = b * outDim;

				for (int d = 0; d < dim; ++d)
				{
					Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const auto bRow = d * outDim;

					for (int n = 0; n < outDim; ++n)
					{
						C.data[static_cast<std::size_t>(cRow + n)] += aVal * B.data[static_cast<std::size_t>(bRow + n)];
					}
				}
			}
		}
		else
		{
			throw std::runtime_error("matmul: caso non implementato");
		}
	}

	void GraphRuntime::opAdd(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			C.shape = {batch, dim};
			C.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto aRow = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const auto idx = static_cast<std::size_t>(aRow + n);
					C.data[idx] = A.data[idx] + B.data[static_cast<std::size_t>(n)];
				}
			}

			return;
		}

		auto size = A.data.size();

		if (size != B.data.size())
			throw std::runtime_error("add: dimension mismatch");

		C.shape = A.shape;
		C.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			C.data[i] = A.data[i] + B.data[i];
	}

	void GraphRuntime::opSub(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("sub: dimension mismatch");

			C.shape = {batch, dim};
			C.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto aRow = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const auto idx = static_cast<std::size_t>(aRow + n);
					C.data[idx] = A.data[idx] - B.data[static_cast<std::size_t>(n)];
				}
			}

			return;
		}

		auto size = A.data.size();

		if (size != B.data.size())
			throw std::runtime_error("sub: dimension mismatch");

		C.shape = A.shape;
		C.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			C.data[i] = A.data[i] - B.data[i];
	}

	void GraphRuntime::opDot(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		auto size = A.data.size();

		if (size != B.data.size())
			throw std::runtime_error("dot: dimension mismatch");

		Scalar sum = 0.0f;

		for (std::size_t i = 0; i < size; ++i)
			sum += A.data[i] * B.data[i];

		C.shape.clear();
		C.data = {sum};
	}

	void GraphRuntime::opDropout(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		Y.shape = X.shape;
		auto size = X.data.size();
		Y.data.assign(size, 0.0f);

		int dropPerc = 50;
		dropPerc = std::max(0, std::min(100, dropPerc));
		Scalar keepProb = 1.0f - (static_cast<Scalar>(dropPerc) / 100.0f);
		Scalar scale = keepProb > 0.0f ? 1.0f / keepProb : 0.0f;

		for (std::size_t i = 0; i < size; ++i)
		{
			bool keep = (std::rand() % 100) + 1 > dropPerc;
			Scalar mask = keep ? scale : 0.0f;
			Y.data[i] = X.data[i] * mask;
		}
	}

	void GraphRuntime::opSig(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		Y.shape = X.shape;
		auto size = X.data.size();
		Y.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			Y.data[i] = 1.0f / (1.0f + std::exp(-1.0f * X.data[i]));
	}

	void GraphRuntime::opRelu(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		Y.shape = X.shape;
		auto size = X.data.size();
		Y.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			Y.data[i] = X.data[i] > 0.0f ? X.data[i] : 0.0f;
	}

	void GraphRuntime::opLRelu(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		Scalar alpha = 0.01f;

		Y.shape = X.shape;
		auto size = X.data.size();
		Y.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar val = X.data[i];
			Y.data[i] = val > 0.0f ? val : alpha * val;
		}
	}

	void GraphRuntime::opMse(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		auto size = X.data.size();

		if (size == 0)
		{
			Y.shape.clear();
			Y.data = {0.0f};
			return;
		}

		if (X.shape.empty())
		{
			Scalar val = X.data[0];
			Y.shape.clear();
			Y.data = {0.5f * val * val};
			return;
		}

		if (X.shape.size() == 2)
		{
			const auto batch = X.shape[0];
			const auto dim = X.shape[1];

			Y.shape = {batch, 1};
			Y.data.assign(static_cast<std::size_t>(batch), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar sum = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar val = X.data[static_cast<std::size_t>(rowStart + i)];
					sum += val * val;
				}

				Y.data[static_cast<std::size_t>(b)] = dim > 0 ? sum / static_cast<Scalar>(dim) : 0.0f;
			}

			return;
		}

		Y.shape.clear();
		Scalar sum = 0.0f;
		for (std::size_t i = 0; i < size; ++i)
			sum += X.data[i] * X.data[i];

		Y.data = {sum / static_cast<Scalar>(size)};
	}

	void GraphRuntime::opMae(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		auto size = X.data.size();

		if (size == 0)
		{
			Y.shape.clear();
			Y.data = {0.0f};
			return;
		}

		if (X.shape.empty())
		{
			Scalar val = X.data[0];
			Y.shape.clear();
			Y.data = {0.5f * std::fabs(val)};
			return;
		}

		if (X.shape.size() == 2)
		{
			const auto batch = X.shape[0];
			const auto dim = X.shape[1];

			Y.shape = {batch, 1};
			Y.data.assign(static_cast<std::size_t>(batch), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar sum = 0.0f;

				for (int i = 0; i < dim; ++i)
					sum += std::fabs(X.data[static_cast<std::size_t>(rowStart + i)]);

				Y.data[static_cast<std::size_t>(b)] = dim > 0 ? sum / static_cast<Scalar>(dim) : 0.0f;
			}

			return;
		}

		Y.shape.clear();
		Scalar sum = 0.0f;
		for (std::size_t i = 0; i < size; ++i)
			sum += std::fabs(X.data[i]);

		Y.data = {sum / static_cast<Scalar>(size)};
	}

	void GraphRuntime::opSoftmax(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];

		Y.shape = X.shape;
		auto size = X.data.size();

		if (size == 0)
		{
			Y.data.clear();
			return;
		}

		if (X.shape.size() == 2)
		{
			const auto batch = X.shape[0];
			const auto dim = X.shape[1];
			Y.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar maxVal = X.data[static_cast<std::size_t>(rowStart)];

				for (int i = 1; i < dim; ++i)
				{
					Scalar val = X.data[static_cast<std::size_t>(rowStart + i)];
					if (val > maxVal)
						maxVal = val;
				}

				std::vector<Scalar> expValues(static_cast<std::size_t>(dim), 0.0f);
				Scalar sum = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					expValues[static_cast<std::size_t>(i)] = std::exp(X.data[static_cast<std::size_t>(rowStart + i)] - maxVal);
					sum += expValues[static_cast<std::size_t>(i)];
				}

				Scalar invSum = sum == 0.0f ? 0.0f : 1.0f / sum;

				for (int i = 0; i < dim; ++i)
				{
					Y.data[static_cast<std::size_t>(rowStart + i)] = expValues[static_cast<std::size_t>(i)] * invSum;
				}
			}

			return;
		}

		Scalar maxVal = X.data[0];
		for (std::size_t i = 1; i < size; ++i)
			if (X.data[i] > maxVal)
				maxVal = X.data[i];

		std::vector<Scalar> expValues(size, 0.0f);
		Scalar sum = 0.0f;
		for (std::size_t i = 0; i < size; ++i)
		{
			expValues[i] = std::exp(X.data[i] - maxVal);
			sum += expValues[i];
		}

		Scalar invSum = sum == 0.0f ? 0.0f : 1.0f / sum;
		Y.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			Y.data[i] = expValues[i] * invSum;
	}

	void GraphRuntime::opCe(int predId, int targetId, int outId)
	{
		auto &pred = tensors[predId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = pred.data.size();

		if (classes == 0 || classes != target.data.size())
		{
			out.shape.clear();
			out.data = {0.0f};
			return;
		}

		if (pred.shape.size() == 2 && target.shape.size() == 2)
		{
			const auto batch = pred.shape[0];
			const auto dim = pred.shape[1];

			if (target.shape[0] != batch || target.shape[1] != dim)
				throw std::runtime_error("CE: dimension mismatch");

			out.shape = {batch};
			out.data.assign(static_cast<std::size_t>(batch), 0.0f);

			const Scalar eps = 1.0e-12f;

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				int activeIndex = -1;
				bool isOneHot = true;

				for (int i = 0; i < dim; ++i)
				{
					Scalar val = target.data[static_cast<std::size_t>(rowStart + i)];

					if (val > 0.5f)
					{
						if (activeIndex != -1)
						{
							isOneHot = false;
							break;
						}

						activeIndex = i;
					}
					else if (std::fabs(val) > 1.0e-9f)
					{
						isOneHot = false;
						break;
					}
				}

				if (isOneHot && activeIndex != -1)
				{
					Scalar prob = pred.data[static_cast<std::size_t>(rowStart + activeIndex)];
					out.data[static_cast<std::size_t>(b)] = -std::log(prob + eps);
					continue;
				}

				Scalar loss = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					const auto idx = static_cast<std::size_t>(rowStart + i);
					loss += target.data[idx] * std::log(pred.data[idx] + eps);
				}

				out.data[static_cast<std::size_t>(b)] = -loss;
			}

			return;
		}

		out.shape.clear();
		int activeIndex = -1;
		bool isOneHot = true;

		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar val = target.data[i];

			if (val > 0.5f)
			{
				if (activeIndex != -1)
				{
					isOneHot = false;
					break;
				}

				activeIndex = static_cast<int>(i);
			}
			else if (std::fabs(val) > 1.0e-9f)
			{
				isOneHot = false;
				break;
			}
		}

		const Scalar eps = 1.0e-12f;

		if (isOneHot && activeIndex != -1)
		{
			Scalar prob = activeIndex < static_cast<int>(pred.data.size()) ? pred.data[static_cast<std::size_t>(activeIndex)] : 0.0f;
			out.data = {-std::log(prob + eps)};
			return;
		}

		Scalar loss = 0.0f;

		for (std::size_t i = 0; i < classes; ++i)
			loss += target.data[i] * std::log((pred.data[i]) + eps);

		out.data = {-loss};
	}

	void GraphRuntime::opCeLogits(int logitsId, int targetId, int outId)
	{
		auto &logits = tensors[logitsId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = logits.data.size();

		if (classes == 0 || classes != target.data.size())
		{
			out.shape.clear();
			out.data = {0.0f};
			return;
		}

		if (logits.shape.size() == 2 && target.shape.size() == 2)
		{
			const auto batch = logits.shape[0];
			const auto dim = logits.shape[1];

			if (target.shape[0] != batch || target.shape[1] != dim)
				throw std::runtime_error("CE logits: dimension mismatch");

			out.shape = {batch};
			out.data.assign(static_cast<std::size_t>(batch), 0.0f);

			const Scalar eps = 1.0e-12f;

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar maxVal = logits.data[static_cast<std::size_t>(rowStart)];

				for (int i = 1; i < dim; ++i)
				{
					Scalar val = logits.data[static_cast<std::size_t>(rowStart + i)];
					if (val > maxVal)
						maxVal = val;
				}

				std::vector<Scalar> probs(static_cast<std::size_t>(dim), 0.0f);
				Scalar sumExp = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar expVal = std::exp(logits.data[static_cast<std::size_t>(rowStart + i)] - maxVal);
					probs[static_cast<std::size_t>(i)] = expVal;
					sumExp += expVal;
				}

				Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
				for (int i = 0; i < dim; ++i)
					probs[static_cast<std::size_t>(i)] *= invSum;

				Scalar loss = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar t = target.data[static_cast<std::size_t>(rowStart + i)];
					if (t > 0.0f)
						loss += -t * std::log(probs[static_cast<std::size_t>(i)] + eps);
				}

				out.data[static_cast<std::size_t>(b)] = loss;
			}

			return;
		}

		out.shape.clear();
		Scalar maxVal = logits.data[0];
		for (std::size_t i = 1; i < classes; ++i)
			if (logits.data[i] > maxVal)
				maxVal = logits.data[i];

		std::vector<Scalar> probs(classes, 0.0f);
		Scalar sumExp = 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar expVal = std::exp(logits.data[i] - maxVal);
			probs[i] = expVal;
			sumExp += expVal;
		}

		Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
			probs[i] *= invSum;

		Scalar loss = 0.0f;
		const Scalar eps = 1.0e-12f;

		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar t = target.data[i];
			if (t > 0.0f)
				loss += -t * std::log(probs[i] + eps);
		}

		out.data = {loss};
	}

	void GraphRuntime::opCeLogitsLabelInt(int logitsId, int targetId, int outId)
	{
		auto &logits = tensors[logitsId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = logits.data.size();

		if (classes == 0)
		{
			out.shape.clear();
			out.data = {0.0f};
			return;
		}

		if (logits.shape.size() == 2)
		{
			const auto batch = logits.shape[0];
			const auto dim = logits.shape[1];

			if (target.shape.size() != 1 || target.shape[0] != batch)
				throw std::runtime_error("CE logits label int: dimension mismatch");

			out.shape = {batch};
			out.data.assign(static_cast<std::size_t>(batch), 0.0f);

			const Scalar eps = 1.0e-12f;

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar labelInt = target.data[static_cast<std::size_t>(b)];
				Scalar maxVal = logits.data[static_cast<std::size_t>(rowStart)];

				for (int i = 1; i < dim; ++i)
				{
					Scalar val = logits.data[static_cast<std::size_t>(rowStart + i)];
					if (val > maxVal)
						maxVal = val;
				}

				std::vector<Scalar> probs(static_cast<std::size_t>(dim), 0.0f);
				Scalar sumExp = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar expVal = std::exp(logits.data[static_cast<std::size_t>(rowStart + i)] - maxVal);
					probs[static_cast<std::size_t>(i)] = expVal;
					sumExp += expVal;
				}

				Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
				for (int i = 0; i < dim; ++i)
					probs[static_cast<std::size_t>(i)] *= invSum;

				Scalar loss = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					if (i == static_cast<int>(labelInt))
						loss += -1.0f * std::log(probs[static_cast<std::size_t>(i)] + eps);
				}

				out.data[static_cast<std::size_t>(b)] = loss;
			}

			return;
		}

		out.shape.clear();
		Scalar maxVal = logits.data[0];
		Scalar labelInt = target.data.empty() ? 0.0f : target.data[0];

		for (std::size_t i = 1; i < classes; ++i)
			if (logits.data[i] > maxVal)
				maxVal = logits.data[i];

		std::vector<Scalar> probs(classes, 0.0f);
		Scalar sumExp = 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar expVal = std::exp(logits.data[i] - maxVal);
			probs[i] = expVal;
			sumExp += expVal;
		}

		Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
			probs[i] *= invSum;

		Scalar loss = 0.0f;
		const Scalar eps = 1.0e-12f;

		for (std::size_t i = 0; i < classes; ++i)
		{
			if (static_cast<int>(i) == static_cast<int>(labelInt))
				loss += -1.0f * std::log(probs[i] + eps);
		}

		out.data = {loss};
	}

	void GraphRuntime::opMean(int aId, int outId)
	{
		auto &A = tensors[aId];
		auto &out = tensors[outId];

		out.shape.clear();

		if (A.shape.size() != 1 || A.data.empty())
			throw std::runtime_error("Mean: dimension mismatch");

		Scalar mean = std::accumulate(A.data.begin(), A.data.end(), 0.0f)
			/ static_cast<Scalar>(A.data.size());

		out.data = {mean};
	}

	void GraphRuntime::backwardMatmul(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (B.shape.size() == 1)
		{
			const auto m = A.shape[0];
			const auto n = A.shape[1];

			for (int i = 0; i < m; ++i)
			{
				Scalar gradC = C.grad[static_cast<std::size_t>(i)];

				for (int k = 0; k < n; ++k)
				{
					auto aIdx = static_cast<std::size_t>(i * n + k);
					A.grad[aIdx] += gradC * B.data[static_cast<std::size_t>(k)];
					B.grad[static_cast<std::size_t>(k)] += gradC * A.data[aIdx];
				}
			}

			return;
		}

		if (B.shape.size() == 2)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];
			const auto dimB = B.shape[0];
			const auto outDim = B.shape[1];

			if (dim != dimB)
				throw std::runtime_error("matmul: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const auto aRow = b * dim;
				const auto cRow = b * outDim;

				for (int d = 0; d < dim; ++d)
				{
					Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const auto bRow = d * outDim;

					for (int n = 0; n < outDim; ++n)
					{
						Scalar gradC = C.grad[static_cast<std::size_t>(cRow + n)];
						A.grad[static_cast<std::size_t>(aRow + d)] += gradC * B.data[static_cast<std::size_t>(bRow + n)];
						B.grad[static_cast<std::size_t>(bRow + n)] += aVal * gradC;
					}
				}
			}

			return;
		}

		throw std::runtime_error("matmul backward: caso non implementato");
	}

	void GraphRuntime::backwardAdd(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const auto idx = static_cast<std::size_t>(rowStart + n);
					Scalar grad = C.grad[idx];
					A.grad[idx] += grad;
					B.grad[static_cast<std::size_t>(n)] += grad;
				}
			}

			return;
		}

		auto size = C.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			A.grad[i] += C.grad[i];
			B.grad[i] += C.grad[i];
		}
	}
	void GraphRuntime::backwardSub(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const auto batch = A.shape[0];
			const auto dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("sub: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const auto idx = static_cast<std::size_t>(rowStart + n);
					Scalar grad = C.grad[idx];
					A.grad[idx] += grad;
					B.grad[static_cast<std::size_t>(n)] -= grad;
				}
			}

			return;
		}

		auto size = C.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			A.grad[i] += C.grad[i];
			B.grad[i] -= C.grad[i];
		}
	}

	void GraphRuntime::backwardDot(int aId, int bId, int outId)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		Scalar gradOut = C.grad.empty() ? 0.0f : C.grad[0];
		auto size = A.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			A.grad[i] += gradOut * B.data[i];
			B.grad[i] += gradOut * A.data[i];
		}
	}

	void GraphRuntime::backwardDropout(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = X.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar x = X.data[i];
			Scalar y = Y.data[i];
			Scalar mask = (x != 0.0f) ? (y / x) : (y == 0.0f ? 0.0f : 1.0f);
			X.grad[i] += Y.grad[i] * mask;
		}
	}

	void GraphRuntime::backwardSig(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = X.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar y = Y.data[i];
			Scalar local = y * (1.0f - y);
			X.grad[i] += Y.grad[i] * local;
		}
	}

	void GraphRuntime::backwardRelu(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = X.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar local = X.data[i] > 0.0f ? 1.0f : 0.0f;
			X.grad[i] += Y.grad[i] * local;
		}
	}

	void GraphRuntime::backwardLRelu(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		Scalar alpha = 0.01f;
		auto size = X.data.size();

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar local = X.data[i] > 0.0f ? 1.0f : alpha;
			X.grad[i] += Y.grad[i] * local;
		}
	}

	void GraphRuntime::backwardMse(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = X.data.size();

		if (size == 0)
			return;

		if (X.shape.empty())
		{
			Scalar val = X.data[0];
			Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
			X.grad[0] += gradOut * val;
			return;
		}

		if (X.shape.size() == 2)
		{
			const auto batch = X.shape[0];
			const auto dim = X.shape[1];

			for (int b = 0; b < batch; ++b)
			{
				Scalar gradOut = (static_cast<std::size_t>(b) < Y.grad.size()) ? Y.grad[static_cast<std::size_t>(b)] : 0.0f;
				Scalar scale = dim > 0 ? (2.0f / static_cast<Scalar>(dim)) * gradOut : 0.0f;
				const auto rowStart = b * dim;

				for (int i = 0; i < dim; ++i)
				{
					const auto idx = static_cast<std::size_t>(rowStart + i);
					X.grad[idx] += scale * X.data[idx];
				}
			}

			return;
		}

		Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
		Scalar scale = (2.0f / static_cast<Scalar>(size)) * gradOut;

		for (std::size_t i = 0; i < size; ++i)
			X.grad[i] += scale * X.data[i];
	}

	void GraphRuntime::backwardMae(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = X.data.size();

		if (size == 0)
			return;

		if (X.shape.empty())
		{
			Scalar val = X.data[0];
			Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
			Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
			X.grad[0] += gradOut * 0.5f * sign;
			return;
		}

		if (X.shape.size() == 2)
		{
			const auto batch = X.shape[0];
			const auto dim = X.shape[1];

			for (int b = 0; b < batch; ++b)
			{
				Scalar gradOut = (static_cast<std::size_t>(b) < Y.grad.size()) ? Y.grad[static_cast<std::size_t>(b)] : 0.0f;
				Scalar scale = dim > 0 ? (1.0f / static_cast<Scalar>(dim)) * gradOut : 0.0f;
				const auto rowStart = b * dim;

				for (int i = 0; i < dim; ++i)
				{
					const auto idx = static_cast<std::size_t>(rowStart + i);
					Scalar val = X.data[idx];
					Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
					X.grad[idx] += scale * sign;
				}
			}

			return;
		}

		Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
		Scalar scale = size > 0 ? (1.0f / static_cast<Scalar>(size)) * gradOut : 0.0f;

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar val = X.data[i];
			Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
			X.grad[i] += scale * sign;
		}
	}

	void GraphRuntime::backwardSoftmax(int inpId, int outId)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		auto size = Y.data.size();

		if (Y.shape.size() == 2)
		{
			const auto batch = Y.shape[0];
			const auto dim = Y.shape[1];

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;

				for (int i = 0; i < dim; ++i)
				{
					Scalar grad = 0.0f;
					const auto yi = Y.data[static_cast<std::size_t>(rowStart + i)];

					for (int j = 0; j < dim; ++j)
					{
						Scalar delta = (i == j) ? 1.0f : 0.0f;
						const auto yj = Y.data[static_cast<std::size_t>(rowStart + j)];
						Scalar jac = yj * (delta - yi);
						grad += Y.grad[static_cast<std::size_t>(rowStart + j)] * jac;
					}

					X.grad[static_cast<std::size_t>(rowStart + i)] += grad;
				}
			}

			return;
		}

		for (std::size_t i = 0; i < size; ++i)
		{
			Scalar grad = 0.0f;

			for (std::size_t j = 0; j < size; ++j)
			{
				Scalar delta = (i == j) ? 1.0f : 0.0f;
				Scalar jac = Y.data[j] * (delta - Y.data[i]);
				grad += Y.grad[j] * jac;
			}

			X.grad[i] += grad;
		}
	}

	void GraphRuntime::backwardCe(int predId, int targetId, int outId)
	{
		auto &pred = tensors[predId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = pred.data.size();
		if (classes == 0 || classes != target.data.size())
			return;

		if (pred.shape.size() == 2 && target.shape.size() == 2)
		{
			const auto batch = pred.shape[0];
			const auto dim = pred.shape[1];

			if (target.shape[0] != batch || target.shape[1] != dim)
				throw std::runtime_error("CE backward: dimension mismatch");

			const Scalar eps = 1.0e-12f;

			for (int b = 0; b < batch; ++b)
			{
				Scalar gradOut = (static_cast<std::size_t>(b) < out.grad.size()) ? out.grad[static_cast<std::size_t>(b)] : 0.0f;
				Scalar scale = gradOut;
				const auto rowStart = b * dim;

				for (int i = 0; i < dim; ++i)
				{
					const auto idx = static_cast<std::size_t>(rowStart + i);
					Scalar p = pred.data[idx];
					Scalar t = target.data[idx];
					pred.grad[idx] += -scale * (t / (p + eps));
				}
			}

			return;
		}

		Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[0];
		const Scalar eps = 1.0e-12f;
		Scalar scale = gradOut;

		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar p = pred.data[i];
			Scalar t = target.data[i];
			pred.grad[i] += -scale * (t / (p + eps));
		}
	}

	void GraphRuntime::backwardCeLogits(int logitsId, int targetId, int outId)
	{
		auto &logits = tensors[logitsId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = logits.data.size();
		if (classes == 0 || classes != target.data.size())
			return;

		if (logits.shape.size() == 2 && target.shape.size() == 2)
		{
			const auto batch = logits.shape[0];
			const auto dim = logits.shape[1];

			if (target.shape[0] != batch || target.shape[1] != dim)
				throw std::runtime_error("CE logits backward: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar maxVal = logits.data[static_cast<std::size_t>(rowStart)];

				for (int i = 1; i < dim; ++i)
				{
					Scalar val = logits.data[static_cast<std::size_t>(rowStart + i)];
					if (val > maxVal)
						maxVal = val;
				}

				std::vector<Scalar> probs(static_cast<std::size_t>(dim), 0.0f);
				Scalar sumExp = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar expVal = std::exp(logits.data[static_cast<std::size_t>(rowStart + i)] - maxVal);
					probs[static_cast<std::size_t>(i)] = expVal;
					sumExp += expVal;
				}

				Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
				for (int i = 0; i < dim; ++i)
					probs[static_cast<std::size_t>(i)] *= invSum;

				Scalar gradOut = (static_cast<std::size_t>(b) < out.grad.size()) ? out.grad[static_cast<std::size_t>(b)] : 0.0f;
				Scalar scale = gradOut;

				for (int i = 0; i < dim; ++i)
				{
					Scalar t = target.data[static_cast<std::size_t>(rowStart + i)];
					logits.grad[static_cast<std::size_t>(rowStart + i)] += scale * (probs[static_cast<std::size_t>(i)] - t);
				}
			}

			return;
		}

		Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[0];
		Scalar maxVal = logits.data[0];
		for (std::size_t i = 1; i < classes; ++i)
			if (logits.data[i] > maxVal)
				maxVal = logits.data[i];

		std::vector<Scalar> probs(classes, 0.0f);
		Scalar sumExp = 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar expVal = std::exp(logits.data[i] - maxVal);
			probs[i] = expVal;
			sumExp += expVal;
		}

		Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
			probs[i] *= invSum;

		Scalar scale = gradOut;

		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar t = target.data[i];
			logits.grad[i] += scale * (probs[i] - t);
		}
	}

	void GraphRuntime::backwardCeLogitsLabelInt(int logitsId, int targetId, int outId)
	{
		auto &logits = tensors[logitsId];
		auto &target = tensors[targetId];
		auto &out = tensors[outId];

		auto classes = logits.data.size();
		if (classes == 0)
			return;

		if (logits.shape.size() == 2)
		{
			const auto batch = logits.shape[0];
			const auto dim = logits.shape[1];

			if (target.shape.size() != 1 || target.shape[0] != batch)
				throw std::runtime_error("CE logits label int backward: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const auto rowStart = b * dim;
				Scalar labelInt = target.data[static_cast<std::size_t>(b)];
				Scalar maxVal = logits.data[static_cast<std::size_t>(rowStart)];

				for (int i = 1; i < dim; ++i)
				{
					Scalar val = logits.data[static_cast<std::size_t>(rowStart + i)];
					if (val > maxVal)
						maxVal = val;
				}

				std::vector<Scalar> probs(static_cast<std::size_t>(dim), 0.0f);
				Scalar sumExp = 0.0f;

				for (int i = 0; i < dim; ++i)
				{
					Scalar expVal = std::exp(logits.data[static_cast<std::size_t>(rowStart + i)] - maxVal);
					probs[static_cast<std::size_t>(i)] = expVal;
					sumExp += expVal;
				}

				Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
				for (int i = 0; i < dim; ++i)
					probs[static_cast<std::size_t>(i)] *= invSum;

				Scalar gradOut = (static_cast<std::size_t>(b) < out.grad.size()) ? out.grad[static_cast<std::size_t>(b)] : 0.0f;
				Scalar scale = gradOut;

				for (int i = 0; i < dim; ++i)
				{
					if (i == static_cast<int>(labelInt))
						logits.grad[static_cast<std::size_t>(rowStart + i)] += scale * (probs[static_cast<std::size_t>(i)] - 1.0f);
					else
						logits.grad[static_cast<std::size_t>(rowStart + i)] += scale * (probs[static_cast<std::size_t>(i)]);
				}
			}

			return;
		}

		Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[0];
		Scalar maxVal = logits.data[0];
		Scalar labelInt = target.data.empty() ? 0.0f : target.data[0];

		for (std::size_t i = 1; i < classes; ++i)
			if (logits.data[i] > maxVal)
				maxVal = logits.data[i];

		std::vector<Scalar> probs(classes, 0.0f);
		Scalar sumExp = 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
		{
			Scalar expVal = std::exp(logits.data[i] - maxVal);
			probs[i] = expVal;
			sumExp += expVal;
		}

		Scalar invSum = sumExp > 0.0f ? 1.0f / sumExp : 0.0f;
		for (std::size_t i = 0; i < classes; ++i)
			probs[i] *= invSum;

		Scalar scale = gradOut;

		for (std::size_t i = 0; i < classes; ++i)
		{
			if (static_cast<int>(i) == static_cast<int>(labelInt))
			{
				logits.grad[i] += scale * (probs[i] - 1.0f);
			}
			else
				logits.grad[i] += scale * (probs[i]);
		}
	}

	void GraphRuntime::backwardMean(int aId, int outId)
	{
		auto &A = tensors[aId];
		auto &out = tensors[outId];
		auto size = A.data.size();

		if (size == 0)
			return;

		if (A.shape.size() != 1)
			throw std::runtime_error("Mean backward: dimension mismatch");

		Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[0];
		Scalar scale = gradOut / static_cast<Scalar>(size);

		for (std::size_t i = 0; i < size; ++i)
			A.grad[i] += scale;
	}

	json GraphRuntime::loadJson(const std::string &path)
	{
		std::ifstream file(path);

		if (!file.is_open())
			throw std::runtime_error("Unable to open graph file: " + path);

		json parsed;
		file >> parsed;
		return parsed;
	}

	void GraphRuntime::loadTensors(const json &graphDef, const json *weightsDef)
	{
		const auto &jsonTensors = graphDef.at("tensors");
		tensors.reserve(jsonTensors.size());

		for (const auto &t : jsonTensors)
		{
			Tensor tensor;
			tensor.id = t.at("id").get<int>();
			tensor.kind = t.at("kind").get<std::string>();
			tensor.name = t.value("name", "");
			tensor.shape = t.at("shape").get<std::vector<int>>();

			// Initialize data/grad; if data provided, use it, otherwise fill zeros with shape product (or 1).
			if (t.contains("data"))
			{
				tensor.data = t.at("data").get<std::vector<Scalar>>();
			}
			else
			{
				auto size = std::accumulate(
					tensor.shape.begin(),
					tensor.shape.end(),
					1,
					std::multiplies<int>());
				tensor.data.assign(size, 0.0f);
			}

			if (weightsDef != nullptr && tensor.kind == "param" && weightsDef->contains("tensors"))
			{
				const auto &weightsTensors = weightsDef->at("tensors");
				const auto tensorKey = std::to_string(tensor.id);

				if (weightsTensors.contains(tensorKey))
				{
					const auto &weightsTensor = weightsTensors.at(tensorKey);
					const auto weightsShape = weightsTensor.at("shape").get<std::vector<int>>();

					if (weightsShape == tensor.shape)
						tensor.data = weightsTensor.at("data").get<std::vector<Scalar>>();
				}
			}

			tensor.grad.assign(tensor.data.size(), 0.0f);

			if (tensor.kind == "input")
				inputId = tensor.id;
			else if (tensor.kind == "target")
				targetId = tensor.id;

			tensors.push_back(std::move(tensor));
		}
	}

	void GraphRuntime::loadOps(const json &graphDef)
	{
		const auto &jsonOps = graphDef.at("ops");
		ops.reserve(jsonOps.size());

		for (const auto &o : jsonOps)
		{
			Op op;
			op.id = o.at("id").get<int>();
			op.op = o.at("op").get<std::string>();
			op.inputs = o.at("inputs").get<std::vector<int>>();
			op.output = o.at("output").get<int>();
			ops.push_back(std::move(op));
		}
	}
}
