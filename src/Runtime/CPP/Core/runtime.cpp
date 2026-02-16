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
	namespace
	{
		static std::size_t shapeElementCount(const std::vector<int> &shape)
		{
			std::size_t count = 1;
			for (int d : shape)
				count *= static_cast<std::size_t>(d);
			return count;
		}

		static std::pair<std::vector<int>, std::vector<int>> alignBatchShapeStrides(
			const std::vector<int> &shape,
			const std::vector<int> &strides,
			int targetBatchRank)
		{
			const int rank = static_cast<int>(shape.size());
			if (rank > targetBatchRank)
				throw std::invalid_argument("rank > targetBatchRank");

			std::vector<int> alignedShape(static_cast<std::size_t>(targetBatchRank), 1);
			std::vector<int> alignedStride(static_cast<std::size_t>(targetBatchRank), 0);

			const int off = targetBatchRank - rank;
			for (int i = 0; i < rank; ++i)
			{
				const int dim = shape[static_cast<std::size_t>(i)];
				alignedShape[static_cast<std::size_t>(off + i)] = dim;
				alignedStride[static_cast<std::size_t>(off + i)] =
					(dim == 1) ? 0 : strides[static_cast<std::size_t>(i)];
			}

			return {alignedShape, alignedStride};
		}

		static void bmmGenericBroadcast(
			const std::vector<Scalar> &aData,
			const std::vector<int> &aShape,
			const std::vector<int> &aStrides,
			const std::vector<Scalar> &bData,
			const std::vector<int> &bShape,
			const std::vector<int> &bStrides,
			std::vector<Scalar> &cData,
			const std::vector<int> &cShape,
			const std::vector<int> &cStrides)
		{
			const int rankA = static_cast<int>(aShape.size());
			const int rankB = static_cast<int>(bShape.size());
			const int rankC = static_cast<int>(cShape.size());
			if (rankA < 2 || rankB < 2)
				throw std::invalid_argument("A,B need rank>=2");

			const int M = aShape[static_cast<std::size_t>(rankA - 2)];
			const int K = aShape[static_cast<std::size_t>(rankA - 1)];
			const int Kb = bShape[static_cast<std::size_t>(rankB - 2)];
			const int N = bShape[static_cast<std::size_t>(rankB - 1)];
			if (K != Kb)
				throw std::invalid_argument("K mismatch");

			const int aStrideM = aStrides[static_cast<std::size_t>(rankA - 2)];
			const int aStrideK = aStrides[static_cast<std::size_t>(rankA - 1)];
			const int bStrideK = bStrides[static_cast<std::size_t>(rankB - 2)];
			const int bStrideN = bStrides[static_cast<std::size_t>(rankB - 1)];

			const int batchRank = std::max(rankA - 2, rankB - 2);

			if (rankC != batchRank + 2)
				throw std::invalid_argument("C rank mismatch");

			const std::vector<int> aBatchShapeRaw(aShape.begin(), aShape.end() - 2);
			const std::vector<int> aBatchStridesRaw(aStrides.begin(), aStrides.end() - 2);
			const std::vector<int> bBatchShapeRaw(bShape.begin(), bShape.end() - 2);
			const std::vector<int> bBatchStridesRaw(bStrides.begin(), bStrides.end() - 2);

			const auto [aBatchShape, aBatchStrideEff] = alignBatchShapeStrides(aBatchShapeRaw, aBatchStridesRaw, batchRank);
			const auto [bBatchShape, bBatchStrideEff] = alignBatchShapeStrides(bBatchShapeRaw, bBatchStridesRaw, batchRank);

			std::vector<int> outBatchShape(static_cast<std::size_t>(batchRank), 1);
			for (int d = 0; d < batchRank; ++d)
			{
				const int ad = aBatchShape[static_cast<std::size_t>(d)];
				const int bd = bBatchShape[static_cast<std::size_t>(d)];
				if (ad != bd && ad != 1 && bd != 1)
					throw std::invalid_argument("batch dim not broadcastable");

				outBatchShape[static_cast<std::size_t>(d)] = std::max(ad, bd);

				if (cShape[static_cast<std::size_t>(d)] != outBatchShape[static_cast<std::size_t>(d)])
					throw std::invalid_argument("C batch shape mismatch");
			}

			if (cShape[static_cast<std::size_t>(batchRank)] != M
				|| cShape[static_cast<std::size_t>(batchRank + 1)] != N)
				throw std::invalid_argument("C last dims mismatch");

			if (cData.size() != shapeElementCount(cShape))
				throw std::invalid_argument("cData size mismatch");

			const int cStrideM = cStrides[static_cast<std::size_t>(batchRank)];
			const int cStrideN = cStrides[static_cast<std::size_t>(batchRank + 1)];

			long long outerCount = 1;
			for (int d : outBatchShape)
				outerCount *= d;

			std::vector<int> idx(static_cast<std::size_t>(batchRank), 0);

			for (long long t = 0; t < outerCount; ++t)
			{
				int baseA = 0;
				int baseB = 0;
				int baseC = 0;
				for (int d = 0; d < batchRank; ++d)
				{
					const int i = idx[static_cast<std::size_t>(d)];
					baseA += i * aBatchStrideEff[static_cast<std::size_t>(d)];
					baseB += i * bBatchStrideEff[static_cast<std::size_t>(d)];
					baseC += i * cStrides[static_cast<std::size_t>(d)];
				}

				for (int m = 0; m < M; ++m)
				{
					const int aRowBase = baseA + m * aStrideM;
					const int cRowBase = baseC + m * cStrideM;

					for (int n = 0; n < N; ++n)
					{
						Scalar sum = 0.0f;
						int aOff = aRowBase;
						int bOff = baseB + n * bStrideN;

						for (int k = 0; k < K; ++k)
						{
							sum += aData[static_cast<std::size_t>(aOff)] * bData[static_cast<std::size_t>(bOff)];
							aOff += aStrideK;
							bOff += bStrideK;
						}

						cData[static_cast<std::size_t>(cRowBase + n * cStrideN)] = sum;
					}
				}

				for (int d = batchRank - 1; d >= 0; --d)
				{
					idx[static_cast<std::size_t>(d)]++;
					if (idx[static_cast<std::size_t>(d)] < outBatchShape[static_cast<std::size_t>(d)])
						break;
					idx[static_cast<std::size_t>(d)] = 0;
				}
			}
		}

		static void bmmGenericBroadcastBackward(
			const std::vector<Scalar> &aData,
			const std::vector<int> &aShape,
			const std::vector<int> &aStrides,
			std::vector<Scalar> &aGrad,
			const std::vector<Scalar> &bData,
			const std::vector<int> &bShape,
			const std::vector<int> &bStrides,
			std::vector<Scalar> &bGrad,
			const std::vector<Scalar> &cGrad,
			const std::vector<int> &cShape,
			const std::vector<int> &cStrides)
		{
			const int rankA = static_cast<int>(aShape.size());
			const int rankB = static_cast<int>(bShape.size());
			const int rankC = static_cast<int>(cShape.size());
			if (rankA < 2 || rankB < 2)
				throw std::invalid_argument("A,B need rank>=2");

			const int M = aShape[static_cast<std::size_t>(rankA - 2)];
			const int K = aShape[static_cast<std::size_t>(rankA - 1)];
			const int Kb = bShape[static_cast<std::size_t>(rankB - 2)];
			const int N = bShape[static_cast<std::size_t>(rankB - 1)];
			if (K != Kb)
				throw std::invalid_argument("K mismatch");

			const int aStrideM = aStrides[static_cast<std::size_t>(rankA - 2)];
			const int aStrideK = aStrides[static_cast<std::size_t>(rankA - 1)];
			const int bStrideK = bStrides[static_cast<std::size_t>(rankB - 2)];
			const int bStrideN = bStrides[static_cast<std::size_t>(rankB - 1)];

			const int batchRank = std::max(rankA - 2, rankB - 2);

			if (rankC != batchRank + 2)
				throw std::invalid_argument("C rank mismatch");

			const std::vector<int> aBatchShapeRaw(aShape.begin(), aShape.end() - 2);
			const std::vector<int> aBatchStridesRaw(aStrides.begin(), aStrides.end() - 2);
			const std::vector<int> bBatchShapeRaw(bShape.begin(), bShape.end() - 2);
			const std::vector<int> bBatchStridesRaw(bStrides.begin(), bStrides.end() - 2);

			const auto [aBatchShape, aBatchStrideEff] = alignBatchShapeStrides(aBatchShapeRaw, aBatchStridesRaw, batchRank);
			const auto [bBatchShape, bBatchStrideEff] = alignBatchShapeStrides(bBatchShapeRaw, bBatchStridesRaw, batchRank);

			for (int d = 0; d < batchRank; ++d)
			{
				const int ad = aBatchShape[static_cast<std::size_t>(d)];
				const int bd = bBatchShape[static_cast<std::size_t>(d)];
				if (ad != bd && ad != 1 && bd != 1)
					throw std::invalid_argument("batch dim not broadcastable");
			}

			if (cShape[static_cast<std::size_t>(batchRank)] != M
				|| cShape[static_cast<std::size_t>(batchRank + 1)] != N)
				throw std::invalid_argument("C last dims mismatch");

			if (cGrad.size() != shapeElementCount(cShape))
				throw std::invalid_argument("cGrad size mismatch");

			const int cStrideM = cStrides[static_cast<std::size_t>(batchRank)];
			const int cStrideN = cStrides[static_cast<std::size_t>(batchRank + 1)];

			std::vector<int> outBatchShape(static_cast<std::size_t>(batchRank), 1);
			for (int d = 0; d < batchRank; ++d)
				outBatchShape[static_cast<std::size_t>(d)] = cShape[static_cast<std::size_t>(d)];

			long long outerCount = 1;
			for (int d : outBatchShape)
				outerCount *= d;

			std::vector<int> idx(static_cast<std::size_t>(batchRank), 0);

			for (long long t = 0; t < outerCount; ++t)
			{
				int baseA = 0;
				int baseB = 0;
				int baseC = 0;
				for (int d = 0; d < batchRank; ++d)
				{
					const int i = idx[static_cast<std::size_t>(d)];
					baseA += i * aBatchStrideEff[static_cast<std::size_t>(d)];
					baseB += i * bBatchStrideEff[static_cast<std::size_t>(d)];
					baseC += i * cStrides[static_cast<std::size_t>(d)];
				}

				for (int m = 0; m < M; ++m)
				{
					const int aRowBase = baseA + m * aStrideM;
					const int cRowBase = baseC + m * cStrideM;

					for (int n = 0; n < N; ++n)
					{
						const Scalar gradC = cGrad[static_cast<std::size_t>(cRowBase + n * cStrideN)];
						const int bColBase = baseB + n * bStrideN;

						int aOff = aRowBase;
						int bOff = bColBase;
						for (int k = 0; k < K; ++k)
						{
							aGrad[static_cast<std::size_t>(aOff)] += gradC * bData[static_cast<std::size_t>(bOff)];
							bGrad[static_cast<std::size_t>(bOff)] += aData[static_cast<std::size_t>(aOff)] * gradC;
							aOff += aStrideK;
							bOff += bStrideK;
						}
					}
				}

				for (int d = batchRank - 1; d >= 0; --d)
				{
					idx[static_cast<std::size_t>(d)]++;
					if (idx[static_cast<std::size_t>(d)] < outBatchShape[static_cast<std::size_t>(d)])
						break;
					idx[static_cast<std::size_t>(d)] = 0;
				}
			}
		}
	}

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
				opMatmul(inputs[0], inputs[1], outId, op.kernel);
			else if (name == "add")
				opAdd(inputs[0], inputs[1], outId, op.kernel);
			// else if (name == "sub")
			// 	opSub(inputs[0], inputs[1], outId);
			// else if (name == "dot")
			// 	opDot(inputs[0], inputs[1], outId);
			else if (name == "dropout")
				opDropout(inputs[0], outId);
			else if (name == "sig")
				opSig(inputs[0], outId);
			else if (name == "ReLU" || name == "relu")
				opRelu(inputs[0], outId);
			// else if (name == "LReLU")
			// 	opLRelu(inputs[0], outId);
			// else if (name == "MSE")
			// 	opMse(inputs[0], outId);
			// else if (name == "MAE")
			// 	opMae(inputs[0], outId);
			else if (name == "mean")
				opMean(inputs[0], outId, op.kernel, op.axes);
			else if (name == "softmax")
				opSoftmax(inputs[0], outId, op.kernel, op.axes);
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
				backwardMatmul(inputs[0], inputs[1], outId, op.kernel);
			else if (name == "add")
				backwardAdd(inputs[0], inputs[1], outId, op.kernel);
			// else if (name == "sub")
			// 	backwardSub(inputs[0], inputs[1], outId);
			// else if (name == "dot")
			// 	backwardDot(inputs[0], inputs[1], outId);
			else if (name == "dropout")
				backwardDropout(inputs[0], outId);
			else if (name == "sig")
				backwardSig(inputs[0], outId);
			else if (name == "relu" || name == "ReLU")
				backwardRelu(inputs[0], outId);
			// else if (name == "LReLU")
			// 	backwardLRelu(inputs[0], outId);
			// else if (name == "MSE")
			// 	backwardMse(inputs[0], outId);
			// else if (name == "MAE")
			// 	backwardMae(inputs[0], outId);
			else if (name == "mean")
				backwardMean(inputs[0], outId, op.kernel, op.axes);
			else if (name == "softmax")
				backwardSoftmax(inputs[0], outId, op.kernel, op.axes);
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

	void GraphRuntime::opMatmul(int aId, int bId, int outId, const std::string &kernel)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		const std::string kernelName = kernel.empty() ? "MATMUL_GENERIC_B_2D_2D_BROADCAST" : kernel;

		if (kernelName == "MATMUL_2D_2D")
			return MATMUL_2D_2D(A, B, C);
		if (kernelName == "MATMUL_1B_2D_2D")
			return MATMUL_1B_2D_2D(A, B, C);
		if (kernelName == "MATMUL_2B_2D_2D")
			return MATMUL_2B_2D_2D(A, B, C);
		if (kernelName == "MATMUL_1B_2D_2D_LINEAR")
			return MATMUL_1B_2D_2D_LINEAR(A, B, C);
		if (kernelName == "MATMUL_GENERIC_B_2D_2D_BROADCAST")
			return MATMUL_GENERIC_B_2D_2D_BROADCAST(A, B, C);

		throw std::runtime_error("matmul: kernel not supported");
	}

	void GraphRuntime::MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 2 || B.shape.size() != 2)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int dim = A.shape[1];
		const int dimB = B.shape[0];
		const int outDim = B.shape[1];

		if (dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		C.shape = {batch, outDim};
		C.data.assign(static_cast<std::size_t>(batch * outDim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int aRow = b * dim;
			const int cRow = b * outDim;

			for (int d = 0; d < dim; ++d)
			{
				const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
				const int bRow = d * outDim;
				for (int n = 0; n < outDim; ++n)
					C.data[static_cast<std::size_t>(cRow + n)] += aVal * B.data[static_cast<std::size_t>(bRow + n)];
			}
		}
	}

	void GraphRuntime::MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 3 || B.shape.size() != 3)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];
		const int batchB = B.shape[0];
		const int dimB = B.shape[1];
		const int outDim = B.shape[2];

		if (batch != batchB || dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		C.shape = {batch, time, outDim};
		C.data.assign(static_cast<std::size_t>(batch * time * outDim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * time * dim;
			const int bBatch = b * dim * outDim;
			const int cBatch = b * time * outDim;

			for (int t = 0; t < time; ++t)
			{
				const int aRow = aBatch + t * dim;
				const int cRow = cBatch + t * outDim;

				for (int d = 0; d < dim; ++d)
				{
					const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const int bRow = bBatch + d * outDim;
					for (int n = 0; n < outDim; ++n)
						C.data[static_cast<std::size_t>(cRow + n)] += aVal * B.data[static_cast<std::size_t>(bRow + n)];
				}
			}
		}
	}

	void GraphRuntime::MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 4 || B.shape.size() != 4)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int heads = A.shape[1];
		const int time = A.shape[2];
		const int dim = A.shape[3];
		const int batchB = B.shape[0];
		const int headsB = B.shape[1];
		const int dimB = B.shape[2];
		const int outTime = B.shape[3];

		if (batch != batchB || heads != headsB || dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		C.shape = {batch, heads, time, outTime};
		C.data.assign(static_cast<std::size_t>(batch * heads * time * outTime), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * heads * time * dim;
			const int bBatch = b * heads * dim * outTime;
			const int cBatch = b * heads * time * outTime;

			for (int h = 0; h < heads; ++h)
			{
				const int aHead = aBatch + h * time * dim;
				const int bHead = bBatch + h * dim * outTime;
				const int cHead = cBatch + h * time * outTime;

				for (int t = 0; t < time; ++t)
				{
					const int aRow = aHead + t * dim;
					const int cRow = cHead + t * outTime;

					for (int d = 0; d < dim; ++d)
					{
						const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
						const int bRow = bHead + d * outTime;
						for (int n = 0; n < outTime; ++n)
							C.data[static_cast<std::size_t>(cRow + n)] += aVal * B.data[static_cast<std::size_t>(bRow + n)];
					}
				}
			}
		}
	}

	void GraphRuntime::MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 3 || B.shape.size() != 2)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];
		const int dimB = B.shape[0];
		const int hidden = B.shape[1];

		if (dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		C.shape = {batch, time, hidden};
		C.data.assign(static_cast<std::size_t>(batch * time * hidden), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * time * dim;
			const int cBatch = b * time * hidden;

			for (int t = 0; t < time; ++t)
			{
				const int aRow = aBatch + t * dim;
				const int cRow = cBatch + t * hidden;

				for (int d = 0; d < dim; ++d)
				{
					const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const int bRow = d * hidden;
					for (int h = 0; h < hidden; ++h)
						C.data[static_cast<std::size_t>(cRow + h)] += aVal * B.data[static_cast<std::size_t>(bRow + h)];
				}
			}
		}
	}

	void GraphRuntime::MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C)
	{
		bmmGenericBroadcast(A.data, A.shape, A.strides, B.data, B.shape, B.strides, C.data, C.shape, C.strides);
	}

	void GraphRuntime::opAdd(int aId, int bId, int outId, const std::string &kernel)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (kernel == "ADD_1D_LAST")
			return ADD_1D_LAST(A, B, C);
		if (kernel == "ADD_2D_LAST")
			return ADD_2D_LAST(A, B, C);
		if (kernel == "ADD_3D_LAST")
			return ADD_3D_LAST(A, B, C);
		if (kernel == "ADD_GENERIC_LAST")
			return ADD_GENERIC_LAST(A, B, C);

		throw std::runtime_error("add: kernel not supported");
	}

	// void GraphRuntime::opSub(int aId, int bId, int outId)
	// {
	// 	auto &A = tensors[aId];
	// 	auto &B = tensors[bId];
	// 	auto &C = tensors[outId];
 // 
	// 	if (A.shape.size() == 2 && B.shape.size() == 1)
	// 	{
	// 		const auto batch = A.shape[0];
	// 		const auto dim = A.shape[1];
 // 
	// 		if (B.shape[0] != dim)
	// 			throw std::runtime_error("sub: dimension mismatch");
 // 
	// 		C.shape = {batch, dim};
	// 		C.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			const auto aRow = b * dim;
 // 
	// 			for (int n = 0; n < dim; ++n)
	// 			{
	// 				const auto idx = static_cast<std::size_t>(aRow + n);
	// 				C.data[idx] = A.data[idx] - B.data[static_cast<std::size_t>(n)];
	// 			}
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	auto size = A.data.size();
 // 
	// 	if (size != B.data.size())
	// 		throw std::runtime_error("sub: dimension mismatch");
 // 
	// 	C.shape = A.shape;
	// 	C.data.assign(size, 0.0f);
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 		C.data[i] = A.data[i] - B.data[i];
	// }

	// void GraphRuntime::opDot(int aId, int bId, int outId)
	// {
	// 	auto &A = tensors[aId];
	// 	auto &B = tensors[bId];
	// 	auto &C = tensors[outId];
 // 
	// 	auto size = A.data.size();
 // 
	// 	if (size != B.data.size())
	// 		throw std::runtime_error("dot: dimension mismatch");
 // 
	// 	Scalar sum = 0.0f;
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 		sum += A.data[i] * B.data[i];
 // 
	// 	C.shape.clear();
	// 	C.data = {sum};
	// }

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

	// void GraphRuntime::opLRelu(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
	// 	Scalar alpha = 0.01f;
 // 
	// 	Y.shape = X.shape;
	// 	auto size = X.data.size();
	// 	Y.data.assign(size, 0.0f);
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 	{
	// 		Scalar val = X.data[i];
	// 		Y.data[i] = val > 0.0f ? val : alpha * val;
	// 	}
	// }

	// void GraphRuntime::opMse(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
 // 
	// 	auto size = X.data.size();
 // 
	// 	if (size == 0)
	// 	{
	// 		Y.shape.clear();
	// 		Y.data = {0.0f};
	// 		return;
	// 	}
 // 
	// 	if (X.shape.empty())
	// 	{
	// 		Scalar val = X.data[0];
	// 		Y.shape.clear();
	// 		Y.data = {0.5f * val * val};
	// 		return;
	// 	}
 // 
	// 	if (X.shape.size() == 2)
	// 	{
	// 		const auto batch = X.shape[0];
	// 		const auto dim = X.shape[1];
 // 
	// 		Y.shape = {batch, 1};
	// 		Y.data.assign(static_cast<std::size_t>(batch), 0.0f);
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			const auto rowStart = b * dim;
	// 			Scalar sum = 0.0f;
 // 
	// 			for (int i = 0; i < dim; ++i)
	// 			{
	// 				Scalar val = X.data[static_cast<std::size_t>(rowStart + i)];
	// 				sum += val * val;
	// 			}
 // 
	// 			Y.data[static_cast<std::size_t>(b)] = dim > 0 ? sum / static_cast<Scalar>(dim) : 0.0f;
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	Y.shape.clear();
	// 	Scalar sum = 0.0f;
	// 	for (std::size_t i = 0; i < size; ++i)
	// 		sum += X.data[i] * X.data[i];
 // 
	// 	Y.data = {sum / static_cast<Scalar>(size)};
	// }

	// void GraphRuntime::opMae(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
 // 
	// 	auto size = X.data.size();
 // 
	// 	if (size == 0)
	// 	{
	// 		Y.shape.clear();
	// 		Y.data = {0.0f};
	// 		return;
	// 	}
 // 
	// 	if (X.shape.empty())
	// 	{
	// 		Scalar val = X.data[0];
	// 		Y.shape.clear();
	// 		Y.data = {0.5f * std::fabs(val)};
	// 		return;
	// 	}
 // 
	// 	if (X.shape.size() == 2)
	// 	{
	// 		const auto batch = X.shape[0];
	// 		const auto dim = X.shape[1];
 // 
	// 		Y.shape = {batch, 1};
	// 		Y.data.assign(static_cast<std::size_t>(batch), 0.0f);
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			const auto rowStart = b * dim;
	// 			Scalar sum = 0.0f;
 // 
	// 			for (int i = 0; i < dim; ++i)
	// 				sum += std::fabs(X.data[static_cast<std::size_t>(rowStart + i)]);
 // 
	// 			Y.data[static_cast<std::size_t>(b)] = dim > 0 ? sum / static_cast<Scalar>(dim) : 0.0f;
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	Y.shape.clear();
	// 	Scalar sum = 0.0f;
	// 	for (std::size_t i = 0; i < size; ++i)
	// 		sum += std::fabs(X.data[i]);
 // 
	// 	Y.data = {sum / static_cast<Scalar>(size)};
	// }

	void GraphRuntime::opSoftmax(int inpId, int outId, const std::string &kernel, const std::vector<int> &axes)
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

		const std::string kernelName = kernel.empty() ? "SOFTMAX_GENERIC_AXIS" : kernel;
		const int axis = axes.empty() ? -1 : axes[0];

		if (kernelName == "SOFTMAX_1D_LAST")
			return SOFTMAX_1D_LAST(X, Y);
		if (kernelName == "SOFTMAX_2D_LAST")
			return SOFTMAX_2D_LAST(X, Y);
		if (kernelName == "SOFTMAX_3D_LAST")
			return SOFTMAX_3D_LAST(X, Y);
		if (kernelName == "SOFTMAX_GENERIC_AXIS")
			return SOFTMAX_GENERIC_AXIS(X, Y, axis);

		throw std::runtime_error("softmax: kernel not supported");
	}

	void GraphRuntime::SOFTMAX_1D_LAST(Tensor &X, Tensor &Y)
	{
		auto size = X.data.size();
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

	void GraphRuntime::SOFTMAX_2D_LAST(Tensor &X, Tensor &Y)
	{
		const int batch = X.shape[0];
		const int dim = X.shape[1];
		Y.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int rowStart = b * dim;
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
	}

	void GraphRuntime::SOFTMAX_3D_LAST(Tensor &X, Tensor &Y)
	{
		const int batch = X.shape[0];
		const int time = X.shape[1];
		const int dim = X.shape[2];
		Y.data.assign(static_cast<std::size_t>(batch * time * dim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int batchOffset = b * time * dim;

			for (int t = 0; t < time; ++t)
			{
				const int rowStart = batchOffset + t * dim;
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
		}
	}

	void GraphRuntime::SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis)
	{
		Y.shape = X.shape;
		Y.data = X.data;
		softmaxAlongAxisInPlace(Y.data, Y.shape, X.strides, axis);
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

	void GraphRuntime::opMean(int aId, int outId, const std::string &kernel, const std::vector<int> &axes)
	{
		auto &A = tensors[aId];
		auto &out = tensors[outId];

		const std::string selectedKernel = kernel.empty() ? "MEAN_GENERIC_AXIS" : kernel;
		const int axis = !axes.empty() ? axes[0] : 0;

		if (selectedKernel == "MEAN_1D_FIRST")
			MEAN_1D_FIRST(A, out);
		else if (selectedKernel == "MEAN_2D_FIRST")
			MEAN_2D_FIRST(A, out);
		else if (selectedKernel == "MEAN_3D_FIRST")
			MEAN_3D_FIRST(A, out);
		else if (selectedKernel == "MEAN_GENERIC_AXIS")
			MEAN_GENERIC_AXIS(A, out, axis);
		else
			throw std::runtime_error("Mean: kernel not supported");
	}

	void GraphRuntime::MEAN_1D_FIRST(Tensor &A, Tensor &out)
	{
		out.shape.clear();

		if (A.shape.size() != 1 || A.data.empty())
			throw std::runtime_error("Mean: dimension mismatch");

		Scalar mean = std::accumulate(A.data.begin(), A.data.end(), 0.0f)
			/ static_cast<Scalar>(A.data.size());

		out.data = {mean};
	}

	void GraphRuntime::MEAN_2D_FIRST(Tensor &A, Tensor &out)
	{
		if (A.shape.size() != 2 || A.data.empty())
			throw std::runtime_error("Mean: dimension mismatch");

		const int batch = A.shape[0];
		const int dim = A.shape[1];

		out.shape = {dim};
		out.data.assign(static_cast<std::size_t>(dim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int rowStart = b * dim;
			for (int i = 0; i < dim; ++i)
			{
				out.data[static_cast<std::size_t>(i)] += A.data[static_cast<std::size_t>(rowStart + i)];
			}
		}

		const Scalar invBatch = batch > 0 ? (1.0f / static_cast<Scalar>(batch)) : 0.0f;
		for (int i = 0; i < dim; ++i)
			out.data[static_cast<std::size_t>(i)] *= invBatch;
	}

	void GraphRuntime::MEAN_3D_FIRST(Tensor &A, Tensor &out)
	{
		if (A.shape.size() != 3 || A.data.empty())
			throw std::runtime_error("Mean: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];

		out.shape = {time, dim};
		out.data.assign(static_cast<std::size_t>(time * dim), 0.0f);

		for (int b = 0; b < batch; ++b)
		{
			const int batchOffset = b * time * dim;
			for (int t = 0; t < time; ++t)
			{
				const int rowOffset = batchOffset + t * dim;
				const int outRow = t * dim;
				for (int i = 0; i < dim; ++i)
				{
					out.data[static_cast<std::size_t>(outRow + i)]
						+= A.data[static_cast<std::size_t>(rowOffset + i)];
				}
			}
		}

		const Scalar invBatch = batch > 0 ? (1.0f / static_cast<Scalar>(batch)) : 0.0f;
		for (int i = 0; i < time * dim; ++i)
			out.data[static_cast<std::size_t>(i)] *= invBatch;
	}

	void GraphRuntime::MEAN_GENERIC_AXIS(Tensor &A, Tensor &out, int axis)
	{
		const int rank = static_cast<int>(A.shape.size());

		if (rank == 0)
		{
			out.shape.clear();
			out.data = A.data;
			return;
		}

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::invalid_argument("axis out of range");

		const int axisLen = A.shape[static_cast<std::size_t>(axis)];
		if (axisLen <= 0)
		{
			out.shape = A.shape;
			out.shape.erase(out.shape.begin() + axis);
			out.data.clear();
			return;
		}

		out.shape = A.shape;
		out.shape.erase(out.shape.begin() + axis);

		long long outCount = 1;
		for (int n : out.shape)
			outCount *= n;

		out.data.assign(static_cast<std::size_t>(outCount), 0.0f);
		out.strides = Tensor::computeStrides(out.shape);

		const Scalar invAxisLen = 1.0f / static_cast<Scalar>(axisLen);
		std::size_t outPos = 0;

		forEachSliceAlongAxisIncremental(
			A.shape,
			A.strides,
			axis,
			[&](int base, int strideAxis, int axisLenInner, const std::vector<int> &idxNoAxis)
			{
				(void)idxNoAxis;
				Scalar sum = 0.0f;
				int off = base;
				for (int i = 0; i < axisLenInner; ++i)
				{
					sum += A.data[static_cast<std::size_t>(off)];
					off += strideAxis;
				}

				out.data[outPos++] = sum * invAxisLen;
			});
	}

	void GraphRuntime::BACKWARD_MEAN_1D_FIRST(Tensor &A, Tensor &out)
	{
		if (A.shape.size() != 1)
			throw std::runtime_error("Mean backward: dimension mismatch");

		const std::size_t size = A.data.size();
		if (size == 0)
			return;

		const Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[0];
		const Scalar scale = gradOut / static_cast<Scalar>(size);

		for (std::size_t i = 0; i < size; ++i)
			A.grad[i] += scale;
	}

	void GraphRuntime::BACKWARD_MEAN_2D_FIRST(Tensor &A, Tensor &out)
	{
		if (A.shape.size() != 2)
			throw std::runtime_error("Mean backward: dimension mismatch");

		const int batch = A.shape[0];
		const int dim = A.shape[1];
		const Scalar invBatch = batch > 0 ? (1.0f / static_cast<Scalar>(batch)) : 0.0f;

		for (int b = 0; b < batch; ++b)
		{
			const int rowStart = b * dim;
			for (int i = 0; i < dim; ++i)
			{
				const Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[static_cast<std::size_t>(i)];
				A.grad[static_cast<std::size_t>(rowStart + i)] += gradOut * invBatch;
			}
		}
	}

	void GraphRuntime::BACKWARD_MEAN_3D_FIRST(Tensor &A, Tensor &out)
	{
		if (A.shape.size() != 3)
			throw std::runtime_error("Mean backward: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];
		const Scalar invBatch = batch > 0 ? (1.0f / static_cast<Scalar>(batch)) : 0.0f;

		for (int b = 0; b < batch; ++b)
		{
			const int batchOffset = b * time * dim;
			for (int t = 0; t < time; ++t)
			{
				const int rowOffset = batchOffset + t * dim;
				const int outRow = t * dim;
				for (int i = 0; i < dim; ++i)
				{
					const Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[static_cast<std::size_t>(outRow + i)];
					A.grad[static_cast<std::size_t>(rowOffset + i)] += gradOut * invBatch;
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_MEAN_GENERIC_AXIS(Tensor &A, Tensor &out, int axis)
	{
		const int rank = static_cast<int>(A.shape.size());

		if (rank == 0)
		{
			A.grad[0] += out.grad.empty() ? 0.0f : out.grad[0];
			return;
		}

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::invalid_argument("axis out of range");

		const int axisLen = A.shape[static_cast<std::size_t>(axis)];
		if (axisLen <= 0)
			return;

		const Scalar invAxisLen = 1.0f / static_cast<Scalar>(axisLen);
		std::size_t outPos = 0;

		forEachSliceAlongAxisIncremental(
			A.shape,
			A.strides,
			axis,
			[&](int base, int strideAxis, int axisLenInner, const std::vector<int> &idxNoAxis)
			{
				(void)idxNoAxis;
				const Scalar gradOut = out.grad.empty() ? 0.0f : out.grad[outPos++];
				const Scalar scale = gradOut * invAxisLen;

				int off = base;
				for (int i = 0; i < axisLenInner; ++i)
				{
					A.grad[static_cast<std::size_t>(off)] += scale;
					off += strideAxis;
				}
			});
	}

	void GraphRuntime::backwardMatmul(int aId, int bId, int outId, const std::string &kernel)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		const std::string kernelName = kernel.empty() ? "MATMUL_GENERIC_B_2D_2D_BROADCAST" : kernel;

		if (kernelName == "MATMUL_2D_2D")
			return BACKWARD_MATMUL_2D_2D(A, B, C);
		if (kernelName == "MATMUL_1B_2D_2D")
			return BACKWARD_MATMUL_1B_2D_2D(A, B, C);
		if (kernelName == "MATMUL_2B_2D_2D")
			return BACKWARD_MATMUL_2B_2D_2D(A, B, C);
		if (kernelName == "MATMUL_1B_2D_2D_LINEAR")
			return BACKWARD_MATMUL_1B_2D_2D_LINEAR(A, B, C);
		if (kernelName == "MATMUL_GENERIC_B_2D_2D_BROADCAST")
			return BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST(A, B, C);

		throw std::runtime_error("matmul backward: kernel not supported");
	}

	void GraphRuntime::BACKWARD_MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 2 || B.shape.size() != 2)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int dim = A.shape[1];
		const int dimB = B.shape[0];
		const int outDim = B.shape[1];

		if (dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		for (int b = 0; b < batch; ++b)
		{
			const int aRow = b * dim;
			const int cRow = b * outDim;

			for (int d = 0; d < dim; ++d)
			{
				const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
				const int bRow = d * outDim;

				for (int n = 0; n < outDim; ++n)
				{
					const Scalar gradC = C.grad[static_cast<std::size_t>(cRow + n)];
					A.grad[static_cast<std::size_t>(aRow + d)] += gradC * B.data[static_cast<std::size_t>(bRow + n)];
					B.grad[static_cast<std::size_t>(bRow + n)] += aVal * gradC;
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 3 || B.shape.size() != 3)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];
		const int batchB = B.shape[0];
		const int dimB = B.shape[1];
		const int outDim = B.shape[2];

		if (batch != batchB || dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * time * dim;
			const int bBatch = b * dim * outDim;
			const int cBatch = b * time * outDim;

			for (int t = 0; t < time; ++t)
			{
				const int aRow = aBatch + t * dim;
				const int cRow = cBatch + t * outDim;

				for (int d = 0; d < dim; ++d)
				{
					const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const int bRow = bBatch + d * outDim;

					for (int n = 0; n < outDim; ++n)
					{
						const Scalar gradC = C.grad[static_cast<std::size_t>(cRow + n)];
						A.grad[static_cast<std::size_t>(aRow + d)] += gradC * B.data[static_cast<std::size_t>(bRow + n)];
						B.grad[static_cast<std::size_t>(bRow + n)] += aVal * gradC;
					}
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 4 || B.shape.size() != 4)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int heads = A.shape[1];
		const int time = A.shape[2];
		const int dim = A.shape[3];
		const int batchB = B.shape[0];
		const int headsB = B.shape[1];
		const int dimB = B.shape[2];
		const int outTime = B.shape[3];

		if (batch != batchB || heads != headsB || dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * heads * time * dim;
			const int bBatch = b * heads * dim * outTime;
			const int cBatch = b * heads * time * outTime;

			for (int h = 0; h < heads; ++h)
			{
				const int aHead = aBatch + h * time * dim;
				const int bHead = bBatch + h * dim * outTime;
				const int cHead = cBatch + h * time * outTime;

				for (int t = 0; t < time; ++t)
				{
					const int aRow = aHead + t * dim;
					const int cRow = cHead + t * outTime;

					for (int d = 0; d < dim; ++d)
					{
						const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
						const int bRow = bHead + d * outTime;

						for (int n = 0; n < outTime; ++n)
						{
							const Scalar gradC = C.grad[static_cast<std::size_t>(cRow + n)];
							A.grad[static_cast<std::size_t>(aRow + d)] += gradC * B.data[static_cast<std::size_t>(bRow + n)];
							B.grad[static_cast<std::size_t>(bRow + n)] += aVal * gradC;
						}
					}
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() != 3 || B.shape.size() != 2)
			throw std::runtime_error("matmul: dimension mismatch");

		const int batch = A.shape[0];
		const int time = A.shape[1];
		const int dim = A.shape[2];
		const int dimB = B.shape[0];
		const int hidden = B.shape[1];

		if (dim != dimB)
			throw std::runtime_error("matmul: dimension mismatch");

		for (int b = 0; b < batch; ++b)
		{
			const int aBatch = b * time * dim;
			const int cBatch = b * time * hidden;

			for (int t = 0; t < time; ++t)
			{
				const int aRow = aBatch + t * dim;
				const int cRow = cBatch + t * hidden;

				for (int d = 0; d < dim; ++d)
				{
					const Scalar aVal = A.data[static_cast<std::size_t>(aRow + d)];
					const int bRow = d * hidden;

					for (int h = 0; h < hidden; ++h)
					{
						const Scalar gradC = C.grad[static_cast<std::size_t>(cRow + h)];
						A.grad[static_cast<std::size_t>(aRow + d)] += gradC * B.data[static_cast<std::size_t>(bRow + h)];
						B.grad[static_cast<std::size_t>(bRow + h)] += aVal * gradC;
					}
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C)
	{
		bmmGenericBroadcastBackward(
			A.data, A.shape, A.strides, A.grad,
			B.data, B.shape, B.strides, B.grad,
			C.grad, C.shape, C.strides);
	}

	void GraphRuntime::backwardAdd(int aId, int bId, int outId, const std::string &kernel)
	{
		auto &A = tensors[aId];
		auto &B = tensors[bId];
		auto &C = tensors[outId];

		if (kernel == "ADD_1D_LAST")
			return BACKWARD_ADD_1D_LAST(A, B, C);
		if (kernel == "ADD_2D_LAST")
			return BACKWARD_ADD_2D_LAST(A, B, C);
		if (kernel == "ADD_3D_LAST")
			return BACKWARD_ADD_3D_LAST(A, B, C);
		if (kernel == "ADD_GENERIC_LAST")
			return BACKWARD_ADD_GENERIC_LAST(A, B, C);

		throw std::runtime_error("add backward: kernel not supported");
	}

	void GraphRuntime::ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		auto size = A.data.size();
		if (size != B.data.size())
			throw std::runtime_error("add: dimension mismatch");

		C.shape = A.shape;
		C.data.assign(size, 0.0f);

		for (std::size_t i = 0; i < size; ++i)
			C.data[i] = A.data[i] + B.data[i];
	}

	void GraphRuntime::ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const int batch = A.shape[0];
			const int dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			C.shape = {batch, dim};
			C.data.assign(static_cast<std::size_t>(batch * dim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const int aRow = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const std::size_t idx = static_cast<std::size_t>(aRow + n);
					C.data[idx] = A.data[idx] + B.data[static_cast<std::size_t>(n)];
				}
			}
		}
	}

	void GraphRuntime::ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() == 3 && B.shape.size() == 1)
		{
			const int batch = A.shape[0];
			const int time = A.shape[1];
			const int dim = A.shape[2];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			C.shape = {batch, time, dim};
			C.data.assign(static_cast<std::size_t>(batch * time * dim), 0.0f);

			for (int b = 0; b < batch; ++b)
			{
				const int batchOffset = b * time * dim;

				for (int t = 0; t < time; ++t)
				{
					const int rowOffset = batchOffset + t * dim;

					for (int n = 0; n < dim; ++n)
					{
						const std::size_t idx = static_cast<std::size_t>(rowOffset + n);
						C.data[idx] = A.data[idx] + B.data[static_cast<std::size_t>(n)];
					}
				}
			}
		}
	}

	void GraphRuntime::ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		const int rank = A.getRank();
		if (rank < 1 || B.shape.size() != 1)
			throw std::runtime_error("add: dimension mismatch");

		const int lastDim = A.shape[static_cast<std::size_t>(rank - 1)];
		if (B.shape[0] != lastDim)
			throw std::runtime_error("add: dimension mismatch");

		C.shape = A.shape;
		const auto size = std::accumulate(
			A.shape.begin(),
			A.shape.end(),
			1,
			std::multiplies<int>());
		C.data.assign(static_cast<std::size_t>(size == 0 ? 1 : size), 0.0f);

		auto bStrides = alignBatchShapeStrides(B.shape, B.strides, C.getRank()).second;
		addAlongAxisInPlace(C.data, C.shape, C.strides, A.data, A.strides, B.data, bStrides);
	}

	void GraphRuntime::BACKWARD_ADD_1D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		auto size = C.data.size();
		for (std::size_t i = 0; i < size; ++i)
		{
			A.grad[i] += C.grad[i];
			B.grad[i] += C.grad[i];
		}
	}

	void GraphRuntime::BACKWARD_ADD_2D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() == 2 && B.shape.size() == 1)
		{
			const int batch = A.shape[0];
			const int dim = A.shape[1];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const int rowStart = b * dim;

				for (int n = 0; n < dim; ++n)
				{
					const std::size_t idx = static_cast<std::size_t>(rowStart + n);
					Scalar grad = C.grad[idx];
					A.grad[idx] += grad;
					B.grad[static_cast<std::size_t>(n)] += grad;
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_ADD_3D_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		if (A.shape.size() == 3 && B.shape.size() == 1)
		{
			const int batch = A.shape[0];
			const int time = A.shape[1];
			const int dim = A.shape[2];

			if (B.shape[0] != dim)
				throw std::runtime_error("add: dimension mismatch");

			for (int b = 0; b < batch; ++b)
			{
				const int batchOffset = b * time * dim;

				for (int t = 0; t < time; ++t)
				{
					const int rowOffset = batchOffset + t * dim;

					for (int n = 0; n < dim; ++n)
					{
						const std::size_t idx = static_cast<std::size_t>(rowOffset + n);
						Scalar grad = C.grad[idx];
						A.grad[idx] += grad;
						B.grad[static_cast<std::size_t>(n)] += grad;
					}
				}
			}
		}
	}

	void GraphRuntime::BACKWARD_ADD_GENERIC_LAST(Tensor &A, Tensor &B, Tensor &C)
	{
		const int rank = C.getRank();
		if (rank < 1 || B.shape.size() != 1)
			throw std::runtime_error("add: dimension mismatch");

		const int lastDim = C.shape[static_cast<std::size_t>(rank - 1)];
		if (B.shape[0] != lastDim)
			throw std::runtime_error("add: dimension mismatch");

		auto bStrides = alignBatchShapeStrides(B.shape, B.strides, rank).second;
		const int axis = rank - 1;

		forEachSliceAlongAxisIncremental(
			C.shape,
			C.strides,
			axis,
			[&](int baseC, int strideCAxis, int axisLen, const std::vector<int> &idxNoAxis)
			{
				int baseA = 0;
				int baseB = 0;

				for (int d = 0; d < rank; ++d)
				{
					const int i = idxNoAxis[static_cast<std::size_t>(d)];
					if (i < 0)
						continue;

					baseA += i * A.strides[static_cast<std::size_t>(d)];
					baseB += i * bStrides[static_cast<std::size_t>(d)];
				}

				const int strideA = A.strides[static_cast<std::size_t>(axis)];
				const int strideB = bStrides[static_cast<std::size_t>(axis)];

				int offC = baseC;
				int offA = baseA;
				int offB = baseB;

				for (int i = 0; i < axisLen; ++i)
				{
					Scalar grad = C.grad[static_cast<std::size_t>(offC)];
					A.grad[static_cast<std::size_t>(offA)] += grad;
					B.grad[static_cast<std::size_t>(offB)] += grad;

					offC += strideCAxis;
					offA += strideA;
					offB += strideB;
				}
			});
	}
	
	// void GraphRuntime::backwardSub(int aId, int bId, int outId)
	// {
	// 	auto &A = tensors[aId];
	// 	auto &B = tensors[bId];
	// 	auto &C = tensors[outId];
 // 
	// 	if (A.shape.size() == 2 && B.shape.size() == 1)
	// 	{
	// 		const auto batch = A.shape[0];
	// 		const auto dim = A.shape[1];
 // 
	// 		if (B.shape[0] != dim)
	// 			throw std::runtime_error("sub: dimension mismatch");
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			const auto rowStart = b * dim;
 // 
	// 			for (int n = 0; n < dim; ++n)
	// 			{
	// 				const auto idx = static_cast<std::size_t>(rowStart + n);
	// 				Scalar grad = C.grad[idx];
	// 				A.grad[idx] += grad;
	// 				B.grad[static_cast<std::size_t>(n)] -= grad;
	// 			}
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	auto size = C.data.size();
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 	{
	// 		A.grad[i] += C.grad[i];
	// 		B.grad[i] -= C.grad[i];
	// 	}
	// }

	// void GraphRuntime::backwardDot(int aId, int bId, int outId)
	// {
	// 	auto &A = tensors[aId];
	// 	auto &B = tensors[bId];
	// 	auto &C = tensors[outId];
 // 
	// 	Scalar gradOut = C.grad.empty() ? 0.0f : C.grad[0];
	// 	auto size = A.data.size();
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 	{
	// 		A.grad[i] += gradOut * B.data[i];
	// 		B.grad[i] += gradOut * A.data[i];
	// 	}
	// }

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

	// void GraphRuntime::backwardLRelu(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
	// 	Scalar alpha = 0.01f;
	// 	auto size = X.data.size();
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 	{
	// 		Scalar local = X.data[i] > 0.0f ? 1.0f : alpha;
	// 		X.grad[i] += Y.grad[i] * local;
	// 	}
	// }

	// void GraphRuntime::backwardMse(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
	// 	auto size = X.data.size();
 // 
	// 	if (size == 0)
	// 		return;
 // 
	// 	if (X.shape.empty())
	// 	{
	// 		Scalar val = X.data[0];
	// 		Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
	// 		X.grad[0] += gradOut * val;
	// 		return;
	// 	}
 // 
	// 	if (X.shape.size() == 2)
	// 	{
	// 		const auto batch = X.shape[0];
	// 		const auto dim = X.shape[1];
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			Scalar gradOut = (static_cast<std::size_t>(b) < Y.grad.size()) ? Y.grad[static_cast<std::size_t>(b)] : 0.0f;
	// 			Scalar scale = dim > 0 ? (2.0f / static_cast<Scalar>(dim)) * gradOut : 0.0f;
	// 			const auto rowStart = b * dim;
 // 
	// 			for (int i = 0; i < dim; ++i)
	// 			{
	// 				const auto idx = static_cast<std::size_t>(rowStart + i);
	// 				X.grad[idx] += scale * X.data[idx];
	// 			}
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
	// 	Scalar scale = (2.0f / static_cast<Scalar>(size)) * gradOut;
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 		X.grad[i] += scale * X.data[i];
	// }

	// void GraphRuntime::backwardMae(int inpId, int outId)
	// {
	// 	auto &X = tensors[inpId];
	// 	auto &Y = tensors[outId];
	// 	auto size = X.data.size();
 // 
	// 	if (size == 0)
	// 		return;
 // 
	// 	if (X.shape.empty())
	// 	{
	// 		Scalar val = X.data[0];
	// 		Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
	// 		Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
	// 		X.grad[0] += gradOut * 0.5f * sign;
	// 		return;
	// 	}
 // 
	// 	if (X.shape.size() == 2)
	// 	{
	// 		const auto batch = X.shape[0];
	// 		const auto dim = X.shape[1];
 // 
	// 		for (int b = 0; b < batch; ++b)
	// 		{
	// 			Scalar gradOut = (static_cast<std::size_t>(b) < Y.grad.size()) ? Y.grad[static_cast<std::size_t>(b)] : 0.0f;
	// 			Scalar scale = dim > 0 ? (1.0f / static_cast<Scalar>(dim)) * gradOut : 0.0f;
	// 			const auto rowStart = b * dim;
 // 
	// 			for (int i = 0; i < dim; ++i)
	// 			{
	// 				const auto idx = static_cast<std::size_t>(rowStart + i);
	// 				Scalar val = X.data[idx];
	// 				Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
	// 				X.grad[idx] += scale * sign;
	// 			}
	// 		}
 // 
	// 		return;
	// 	}
 // 
	// 	Scalar gradOut = Y.grad.empty() ? 0.0f : Y.grad[0];
	// 	Scalar scale = size > 0 ? (1.0f / static_cast<Scalar>(size)) * gradOut : 0.0f;
 // 
	// 	for (std::size_t i = 0; i < size; ++i)
	// 	{
	// 		Scalar val = X.data[i];
	// 		Scalar sign = val > 0.0f ? 1.0f : (val < 0.0f ? -1.0f : 0.0f);
	// 		X.grad[i] += scale * sign;
	// 	}
	// }

	void GraphRuntime::backwardSoftmax(int inpId, int outId, const std::string &kernel, const std::vector<int> &axes)
	{
		auto &X = tensors[inpId];
		auto &Y = tensors[outId];
		const std::string kernelName = kernel.empty() ? "SOFTMAX_GENERIC_AXIS" : kernel;
		const int axis = axes.empty() ? -1 : axes[0];

		if (kernelName == "SOFTMAX_1D_LAST")
			return BACKWORD_SOFTMAX_1D_LAST(X, Y);
		if (kernelName == "SOFTMAX_2D_LAST")
			return BACKWORD_SOFTMAX_2D_LAST(X, Y);
		if (kernelName == "SOFTMAX_3D_LAST")
			return BACKWORD_SOFTMAX_3D_LAST(X, Y);
		if (kernelName == "SOFTMAX_GENERIC_AXIS")
			return BACKWORD_SOFTMAX_GENERIC_AXIS(X, Y, axis);

		throw std::runtime_error("softmax backward: kernel not supported");
	}

	void GraphRuntime::BACKWORD_SOFTMAX_1D_LAST(Tensor &X, Tensor &Y)
	{
		const auto size = Y.data.size();
		Scalar dot = 0.0f;
		for (std::size_t i = 0; i < size; ++i)
			dot += Y.grad[i] * Y.data[i];

		for (std::size_t i = 0; i < size; ++i)
			X.grad[i] += Y.data[i] * (Y.grad[i] - dot);
	}

	void GraphRuntime::BACKWORD_SOFTMAX_2D_LAST(Tensor &X, Tensor &Y)
	{
		if (Y.shape.size() == 2)
		{
			const int batch = Y.shape[0];
			const int dim = Y.shape[1];

			for (int b = 0; b < batch; ++b)
			{
				const int rowStart = b * dim;
				Scalar dot = 0.0f;

				for (int i = 0; i < dim; ++i)
					dot += Y.grad[static_cast<std::size_t>(rowStart + i)]
						* Y.data[static_cast<std::size_t>(rowStart + i)];

				for (int i = 0; i < dim; ++i)
				{
					const std::size_t idx = static_cast<std::size_t>(rowStart + i);
					X.grad[idx] += Y.data[idx] * (Y.grad[idx] - dot);
				}
			}
		}
	}

	void GraphRuntime::BACKWORD_SOFTMAX_3D_LAST(Tensor &X, Tensor &Y)
	{
		if (Y.shape.size() == 3)
		{
			const int batch = Y.shape[0];
			const int time = Y.shape[1];
			const int dim = Y.shape[2];

			for (int b = 0; b < batch; ++b)
			{
				const int batchOffset = b * time * dim;

				for (int t = 0; t < time; ++t)
				{
					const int rowStart = batchOffset + t * dim;
					Scalar dot = 0.0f;

					for (int i = 0; i < dim; ++i)
					{
						const std::size_t idx = static_cast<std::size_t>(rowStart + i);
						dot += Y.grad[idx] * Y.data[idx];
					}

					for (int i = 0; i < dim; ++i)
					{
						const std::size_t idx = static_cast<std::size_t>(rowStart + i);
						X.grad[idx] += Y.data[idx] * (Y.grad[idx] - dot);
					}
				}
			}
		}
	}

	void GraphRuntime::BACKWORD_SOFTMAX_GENERIC_AXIS(Tensor &X, Tensor &Y, int axis)
	{
		const int rank = static_cast<int>(Y.shape.size());
		if (rank == 0)
			return;

		int axisNorm = axis < 0 ? axis + rank : axis;
		if (axisNorm < 0 || axisNorm >= rank)
			throw std::invalid_argument("axis out of range");

		const int axisLen = Y.shape[static_cast<std::size_t>(axisNorm)];
		if (axisLen <= 0)
			return;

		std::vector<Scalar> tmpY(static_cast<std::size_t>(axisLen), 0.0f);
		std::vector<Scalar> tmpGrad(static_cast<std::size_t>(axisLen), 0.0f);

		forEachSliceAlongAxisIncremental(
			Y.shape,
			Y.strides,
			axis,
			[&](int base, int strideAxis, int axisLenInner, const std::vector<int> &idxNoAxis)
			{
				(void)idxNoAxis;
				int off = base;
				tmpY[0] = Y.data[static_cast<std::size_t>(off)];
				tmpGrad[0] = Y.grad[static_cast<std::size_t>(off)];

				for (int i = 1; i < axisLenInner; ++i)
				{
					off += strideAxis;
					tmpY[static_cast<std::size_t>(i)] = Y.data[static_cast<std::size_t>(off)];
					tmpGrad[static_cast<std::size_t>(i)] = Y.grad[static_cast<std::size_t>(off)];
				}

				Scalar dot = 0.0f;
				for (int i = 0; i < axisLenInner; ++i)
					dot += tmpGrad[static_cast<std::size_t>(i)] * tmpY[static_cast<std::size_t>(i)];

				off = base;
				for (int i = 0; i < axisLenInner; ++i)
				{
					X.grad[static_cast<std::size_t>(off)] +=
						tmpY[static_cast<std::size_t>(i)] * (tmpGrad[static_cast<std::size_t>(i)] - dot);
					off += strideAxis;
				}
			});
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

	void GraphRuntime::backwardMean(int aId, int outId, const std::string &kernel, const std::vector<int> &axes)
	{
		auto &A = tensors[aId];
		auto &out = tensors[outId];

		const std::string selectedKernel = kernel.empty() ? "MEAN_GENERIC_AXIS" : kernel;
		const int axis = !axes.empty() ? axes[0] : 0;

		if (selectedKernel == "MEAN_1D_FIRST")
			BACKWARD_MEAN_1D_FIRST(A, out);
		else if (selectedKernel == "MEAN_2D_FIRST")
			BACKWARD_MEAN_2D_FIRST(A, out);
		else if (selectedKernel == "MEAN_3D_FIRST")
			BACKWARD_MEAN_3D_FIRST(A, out);
		else if (selectedKernel == "MEAN_GENERIC_AXIS")
			BACKWARD_MEAN_GENERIC_AXIS(A, out, axis);
		else
			throw std::runtime_error("Mean backward: kernel not supported");
	}

	template <class Callback>
	void GraphRuntime::forEachSliceAlongAxisIncremental(
		const std::vector<int> &shape,
		const std::vector<int> &strides,
		int axis,
		Callback onSlice) const
	{
		const int rank = static_cast<int>(shape.size());
		if (rank == 0)
			return;

		if (static_cast<int>(strides.size()) != rank)
			throw std::invalid_argument("shape/strides rank mismatch");

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::invalid_argument("axis out of range");

		const int axisLen = shape[static_cast<std::size_t>(axis)];
		if (axisLen <= 0)
			return;
		const int strideAxis = strides[static_cast<std::size_t>(axis)];

		std::vector<int> outerDims;
		outerDims.reserve(static_cast<std::size_t>(rank - 1));
		for (int d = 0; d < rank; ++d)
		{
			if (d != axis)
				outerDims.push_back(d);
		}

		if (outerDims.empty())
		{
			std::vector<int> idxNoAxis(static_cast<std::size_t>(rank), 0);
			idxNoAxis[static_cast<std::size_t>(axis)] = -1;
			onSlice(0, strideAxis, axisLen, idxNoAxis);
			return;
		}

		std::vector<int> idx(static_cast<std::size_t>(rank), 0);

		long long outerCount = 1;
		for (int d : outerDims)
		{
			const int n = shape[static_cast<std::size_t>(d)];
			if (n <= 0)
				return;
			outerCount *= n;
		}

		int baseOffset = 0;

		for (long long t = 0; t < outerCount; ++t)
		{
			std::vector<int> idxNoAxis = idx;
			idxNoAxis[static_cast<std::size_t>(axis)] = -1;

			onSlice(baseOffset, strideAxis, axisLen, idxNoAxis);

			for (int k = static_cast<int>(outerDims.size()) - 1; k >= 0; --k)
			{
				const int d = outerDims[static_cast<std::size_t>(k)];
				idx[static_cast<std::size_t>(d)]++;

				if (idx[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
				{
					baseOffset += strides[static_cast<std::size_t>(d)];
					break;
				}

				idx[static_cast<std::size_t>(d)] = 0;
				baseOffset -= (shape[static_cast<std::size_t>(d)] - 1) * strides[static_cast<std::size_t>(d)];
			}
		}
	}

	void GraphRuntime::addAlongAxisInPlace(
		std::vector<Scalar> &zData,
		const std::vector<int> &zShape,
		const std::vector<int> &zStrides,
		const std::vector<Scalar> &xData,
		const std::vector<int> &xStrides,
		const std::vector<Scalar> &yData,
		const std::vector<int> &yStrides,
		int axis) const
	{
		const int rank = static_cast<int>(zShape.size());
		if (rank == 0)
			return;

		if (axis < 0)
			axis += rank;
		if (axis < 0 || axis >= rank)
			throw std::invalid_argument("axis out of range");

		this->forEachSliceAlongAxisIncremental(
			zShape,
			zStrides,
			axis,
			[&](
				int baseZ,
				int strideZAxis,
				int axisLen,
				const std::vector<int> &idxNoAxis)
			{
				int baseX = 0;
				int baseY = 0;

				for (int d = 0; d < rank; ++d)
				{
					const int i = idxNoAxis[static_cast<std::size_t>(d)];
					if (i < 0)
						continue;

					baseX += i * xStrides[static_cast<std::size_t>(d)];
					baseY += i * yStrides[static_cast<std::size_t>(d)];
				}

				const int strideX = xStrides[static_cast<std::size_t>(axis)];
				const int strideY = yStrides[static_cast<std::size_t>(axis)];

				int offZ = baseZ;
				int offX = baseX;
				int offY = baseY;

				for (int i = 0; i < axisLen; ++i)
				{
					zData[static_cast<std::size_t>(offZ)] =
						xData[static_cast<std::size_t>(offX)] + yData[static_cast<std::size_t>(offY)];

					offZ += strideZAxis;
					offX += strideX;
					offY += strideY;
				}
			});
	}

	void GraphRuntime::softmaxAlongAxisInPlace(
		std::vector<Scalar> &data,
		const std::vector<int> &shape,
		const std::vector<int> &strides,
		int axis) const
	{
		const int rank = static_cast<int>(shape.size());
		if (rank == 0)
			return;

		int axisNorm = axis < 0 ? axis + rank : axis;
		if (axisNorm < 0 || axisNorm >= rank)
			throw std::invalid_argument("axis out of range");

		const int axisLen = shape[static_cast<std::size_t>(axisNorm)];
		if (axisLen <= 0)
			return;

		std::vector<Scalar> tmp(static_cast<std::size_t>(axisLen), 0.0f);

		forEachSliceAlongAxisIncremental(
			shape,
			strides,
			axis,
			[&](int base, int strideAxis, int axisLenInner, const std::vector<int> &idxNoAxis)
			{
				(void)idxNoAxis;
				int off = base;
				Scalar maxVal = data[static_cast<std::size_t>(off)];
				tmp[0] = maxVal;

				for (int i = 1; i < axisLenInner; ++i)
				{
					off += strideAxis;
					const Scalar v = data[static_cast<std::size_t>(off)];
					tmp[static_cast<std::size_t>(i)] = v;
					if (v > maxVal)
						maxVal = v;
				}

				Scalar sum = 0.0f;
				for (int i = 0; i < axisLenInner; ++i)
				{
					const Scalar e = std::exp(tmp[static_cast<std::size_t>(i)] - maxVal);
					tmp[static_cast<std::size_t>(i)] = e;
					sum += e;
				}

				const Scalar invSum = sum != 0.0f ? (1.0f / sum) : 0.0f;

				off = base;
				data[static_cast<std::size_t>(off)] = tmp[0] * invSum;
				for (int i = 1; i < axisLenInner; ++i)
				{
					off += strideAxis;
					data[static_cast<std::size_t>(off)] = tmp[static_cast<std::size_t>(i)] * invSum;
				}
			});
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
			tensor.baseOffset = 0;
			tensor.strides = Tensor::computeStrides(tensor.shape);

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
			if (o.contains("attributes"))
			{
				const auto &attrs = o.at("attributes");
				if (attrs.contains("kernel"))
					op.kernel = attrs.at("kernel").get<std::string>();
				if (attrs.contains("axes"))
					op.axes = attrs.at("axes").get<std::vector<int>>();
			}

			ops.push_back(std::move(op));
		}
	}
}
