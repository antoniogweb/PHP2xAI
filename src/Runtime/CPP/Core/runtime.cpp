#include "runtime.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

#include "Kernels/TensorAccess.hpp"

namespace PHP2xAI::Runtime::CPP
{
	namespace
	{
		std::size_t elementCount(const std::vector<int> &shape)
		{
			std::size_t count = 1;
			for (std::size_t i = 0; i < shape.size(); ++i)
			{
				if (shape[i] < 0)
					throw std::invalid_argument("Tensor dimensions cannot be negative");
				count *= static_cast<std::size_t>(shape[i]);
			}
			return count;
		}

		DType readDType(const json &definition)
		{
			const int value = definition.value("dtype", static_cast<int>(DType::FLOAT32));
			switch (value)
			{
				case static_cast<int>(DType::FLOAT32): return DType::FLOAT32;
				case static_cast<int>(DType::FLOAT64): return DType::FLOAT64;
				case static_cast<int>(DType::INT32): return DType::INT32;
				case static_cast<int>(DType::INT64): return DType::INT64;
				default: throw std::invalid_argument("Unsupported tensor dtype: " + std::to_string(value));
			}
		}

		template <typename T>
		void fillZeros(T *values, std::size_t count)
		{
			for (std::size_t i = 0; i < count; ++i)
				values[i] = static_cast<T>(0);
		}
	}

	// Owns one typed allocation. Keeping its deleter beside the pointer means the
	// matching delete[] is used when the Tensor is destroyed or resized.
	struct TensorBuffer
	{
		void *pointer;
		void (*deleter)(void *);

		TensorBuffer() : pointer(0), deleter(0) {}

		~TensorBuffer()
		{
			clear();
		}

		TensorBuffer(const TensorBuffer &) = delete;
		TensorBuffer &operator=(const TensorBuffer &) = delete;

		TensorBuffer(TensorBuffer &&other) : pointer(other.pointer), deleter(other.deleter)
		{
			other.pointer = 0;
			other.deleter = 0;
		}

		TensorBuffer &operator=(TensorBuffer &&other)
		{
			if (this != &other)
			{
				clear();
				pointer = other.pointer;
				deleter = other.deleter;
				other.pointer = 0;
				other.deleter = 0;
			}
			return *this;
		}

		template <typename T>
		void allocate(std::size_t count)
		{
			clear();
			if (count == 0)
				return;

			pointer = new T[count];
			deleter = &deleteArray<T>;
		}

		void *data()
		{
			return pointer;
		}

		const void *data() const
		{
			return pointer;
		}

	private:
		template <typename T>
		static void deleteArray(void *memory)
		{
			delete[] static_cast<T *>(memory);
		}

		void clear()
		{
			if (pointer != 0 && deleter != 0)
				deleter(pointer);
			pointer = 0;
			deleter = 0;
		}
	};

	// Tensor is intentionally defined only in this file. Its two memory buffers
	// are RAII-owned and can hold any dtype supported by the graph format.
	struct Tensor
	{
		int id;
		DType dtype;
		std::string kind;
		std::string name;
		std::vector<int> shape;
		std::vector<int> strides;
		bool requiresGrad;
		void *data;
		void *grad;
		TensorBuffer dataOwner;
		TensorBuffer gradOwner;
		std::size_t size;

		Tensor()
			: id(-1), dtype(DType::FLOAT32), requiresGrad(false), data(0), grad(0), size(0)
		{
		}

		void allocate(DType tensorDType, std::size_t elementSize)
		{
			dtype = tensorDType;
			size = elementSize;

			// Allocate real objects of the requested type, not just untyped bytes.
			// dataAs<T>() can therefore safely expose a pointer to typed elements.
				switch (dtype)
			{
				case DType::FLOAT32:
					dataOwner.allocate<float>(size);
					gradOwner.allocate<float>(size);
					break;
				case DType::FLOAT64:
					dataOwner.allocate<double>(size);
					gradOwner.allocate<double>(size);
					break;
				case DType::INT32:
					dataOwner.allocate<std::int32_t>(size);
					gradOwner.allocate<std::int32_t>(size);
					break;
				case DType::INT64:
					dataOwner.allocate<std::int64_t>(size);
					gradOwner.allocate<std::int64_t>(size);
					break;
			}

			// These public-to-the-runtime pointers are non-owning views. The two
			// TensorBuffer members above remain responsible for releasing memory.
			data = dataOwner.data();
			grad = gradOwner.data();
			fillStorageWithZeros(data);
			fillStorageWithZeros(grad);
		}

		template <typename T>
		T *dataAs()
		{
			return static_cast<T *>(data);
		}

		template <typename T>
		const T *dataAs() const
		{
			return static_cast<const T *>(data);
		}

		template <typename T>
		T *gradAs()
		{
			return static_cast<T *>(grad);
		}

		template <typename T>
		const T *gradAs() const
		{
			return static_cast<const T *>(grad);
		}

		Scalar readData(std::size_t index) const
		{
			checkIndex(index);
			switch (dtype)
			{
				case DType::FLOAT32: return static_cast<Scalar>(dataAs<float>()[index]);
				case DType::FLOAT64: return static_cast<Scalar>(dataAs<double>()[index]);
				case DType::INT32: return static_cast<Scalar>(dataAs<std::int32_t>()[index]);
				case DType::INT64: return static_cast<Scalar>(dataAs<std::int64_t>()[index]);
			}
			throw std::runtime_error("Invalid tensor dtype");
		}

		Scalar readGrad(std::size_t index) const
		{
			checkIndex(index);
			switch (dtype)
			{
				case DType::FLOAT32: return static_cast<Scalar>(gradAs<float>()[index]);
				case DType::FLOAT64: return static_cast<Scalar>(gradAs<double>()[index]);
				case DType::INT32: return static_cast<Scalar>(gradAs<std::int32_t>()[index]);
				case DType::INT64: return static_cast<Scalar>(gradAs<std::int64_t>()[index]);
			}
			throw std::runtime_error("Invalid tensor dtype");
		}

		void writeData(std::size_t index, Scalar value)
		{
			checkIndex(index);
			switch (dtype)
			{
				case DType::FLOAT32: dataAs<float>()[index] = static_cast<float>(value); break;
				case DType::FLOAT64: dataAs<double>()[index] = static_cast<double>(value); break;
				case DType::INT32: dataAs<std::int32_t>()[index] = static_cast<std::int32_t>(value); break;
				case DType::INT64: dataAs<std::int64_t>()[index] = static_cast<std::int64_t>(value); break;
			}
		}

		void writeGrad(std::size_t index, Scalar value)
		{
			checkIndex(index);
			switch (dtype)
			{
				case DType::FLOAT32: gradAs<float>()[index] = static_cast<float>(value); break;
				case DType::FLOAT64: gradAs<double>()[index] = static_cast<double>(value); break;
				case DType::INT32: gradAs<std::int32_t>()[index] = static_cast<std::int32_t>(value); break;
				case DType::INT64: gradAs<std::int64_t>()[index] = static_cast<std::int64_t>(value); break;
			}
		}

		void fillStorageWithZeros(void *buffer)
		{
			switch (dtype)
			{
				case DType::FLOAT32: fillZeros(static_cast<float *>(buffer), size); break;
				case DType::FLOAT64: fillZeros(static_cast<double *>(buffer), size); break;
				case DType::INT32: fillZeros(static_cast<std::int32_t *>(buffer), size); break;
				case DType::INT64: fillZeros(static_cast<std::int64_t *>(buffer), size); break;
			}
		}

		void fillGradWithZeros()
		{
			// Select the typed pointer once per tensor, then clear its elements.
			switch (dtype)
			{
				case DType::FLOAT32:
					fillZeros(gradAs<float>(), size);
					break;
				case DType::FLOAT64:
					fillZeros(gradAs<double>(), size);
					break;
				case DType::INT32:
					fillZeros(gradAs<std::int32_t>(), size);
					break;
				case DType::INT64:
					fillZeros(gradAs<std::int64_t>(), size);
					break;
			}
		}

		void checkIndex(std::size_t index) const
		{
			if (index >= size)
				throw std::out_of_range("Tensor element index out of range");
		}
	};

	// Give kernel implementation files a small non-owning view of Tensor while
	// keeping the owning Tensor definition private to this translation unit.
	TensorAccess accessTensor(Tensor &tensor)
	{
		return TensorAccess{
			tensor.dtype,
			tensor.shape,
			tensor.requiresGrad,
			tensor.data,
			tensor.grad,
			tensor.size};
	}

	struct RuntimeOp
	{
		std::string name;
		std::vector<int> inputs;
		int output;
		std::string kernel;
		Scalar dropoutPerc;
		int padId;

		RuntimeOp() : output(-1), dropoutPerc(50.0f), padId(0) {}
	};

	struct GraphRuntime::Impl
	{
		json graphDef;
		std::vector<Tensor> tensors;
		std::vector<RuntimeOp> ops;
		std::unordered_map<int, std::size_t> tensorIndices;
		std::vector<int> trainable;
		std::unordered_map<int, std::vector<Scalar> > dropoutMasks;
		std::uint64_t dropoutSeed;
		int inputId;
		int targetId;
		int outputId;
		int lossId;
		ExecutionMode mode;
		std::unique_ptr<Profiler> profiler;
		bool profilingEnabled;

		Impl(const json &definition)
			: graphDef(definition), dropoutSeed(0x9e3779b97f4a7c15ULL),
			  inputId(-1), targetId(-1), outputId(-1), lossId(-1),
			  mode(ExecutionMode::INFER), profilingEnabled(false)
		{
		}

		Tensor &tensor(int id)
		{
			std::unordered_map<int, std::size_t>::iterator found = tensorIndices.find(id);
			if (found == tensorIndices.end())
				throw std::out_of_range("Unknown tensor id: " + std::to_string(id));
			return tensors[found->second];
		}

		const Tensor &tensor(int id) const
		{
			std::unordered_map<int, std::size_t>::const_iterator found = tensorIndices.find(id);
			if (found == tensorIndices.end())
				throw std::out_of_range("Unknown tensor id: " + std::to_string(id));
			return tensors[found->second];
		}

		void loadTensors(const json &weights)
		{
			const json &definitions = graphDef.at("tensors");
			tensors.reserve(definitions.size());

			for (std::size_t i = 0; i < definitions.size(); ++i)
			{
				const json &definition = definitions[i];
				Tensor tensorValue;
				tensorValue.id = definition.at("id").get<int>();
				tensorValue.kind = definition.value("kind", std::string("intermediate"));
				tensorValue.name = definition.value("name", std::string());
				tensorValue.shape = definition.at("shape").get<std::vector<int> >();
				tensorValue.strides.resize(tensorValue.shape.size());
				int stride = 1;
				for (int axis = static_cast<int>(tensorValue.shape.size()) - 1; axis >= 0; --axis)
				{
					tensorValue.strides[static_cast<std::size_t>(axis)] = stride;
					stride *= tensorValue.shape[static_cast<std::size_t>(axis)];
				}
				tensorValue.requiresGrad = definition.value("requiresGrad", false);
				tensorValue.allocate(readDType(definition), elementCount(tensorValue.shape));

				bool loaded = false;
				const std::string key = std::to_string(tensorValue.id);
				if (tensorValue.kind == "param" && weights.is_object()
					&& weights.contains("tensors") && weights.at("tensors").contains(key))
				{
					const json &weight = weights.at("tensors").at(key);
					if (weight.at("shape").get<std::vector<int> >() == tensorValue.shape)
					{
						loadValues(tensorValue, weight.at("data"));
						loaded = true;
					}
				}

				if (!loaded && definition.contains("data") && !definition.at("data").empty())
				{
					loadValues(tensorValue, definition.at("data"));
					loaded = true;
				}

				const std::string initType = definition.value("init_type", std::string());
				if (!loaded && !initType.empty() && initType != "rand" && initType != "zeros")
					throw std::invalid_argument("Unsupported tensor init type: " + initType);

				if (!loaded && initType == "rand")
				{
					const Scalar scale = definition.value("init_scale", 0.05f);
					std::uint32_t state = definition.value("init_seed", std::uint32_t(0));
					for (std::size_t element = 0; element < tensorValue.size; ++element)
					{
						state = std::uint32_t(1664525) * state + std::uint32_t(1013904223);
						const double uniform = static_cast<double>(state) / 4294967296.0;
						tensorValue.writeData(element, static_cast<Scalar>((uniform * 2.0 - 1.0) * scale));
					}
				}

				if (tensorValue.kind == "input")
					inputId = tensorValue.id;
				if (tensorValue.kind == "target")
					targetId = tensorValue.id;
				if (tensorValue.kind == "loss")
					lossId = tensorValue.id;

				tensorIndices[tensorValue.id] = tensors.size();
				tensors.push_back(std::move(tensorValue));
			}

			if (graphDef.contains("loss"))
				lossId = graphDef.at("loss").get<int>();
			if (graphDef.contains("output"))
				outputId = graphDef.at("output").get<int>();

			if (outputId < 0 && !tensors.empty())
				outputId = tensors.back().id;

			if (graphDef.contains("trainable"))
				trainable = graphDef.at("trainable").get<std::vector<int> >();
			else
			{
				for (std::size_t i = 0; i < tensors.size(); ++i)
					if (tensors[i].kind == "param" && tensors[i].requiresGrad)
						trainable.push_back(tensors[i].id);
			}
		}

		void loadOps()
		{
			const json &definitions = graphDef.at("ops");
			ops.reserve(definitions.size());

			for (std::size_t i = 0; i < definitions.size(); ++i)
			{
				const json &definition = definitions[i];
				RuntimeOp op;
				op.name = definition.at("op").get<std::string>();
				op.inputs = definition.at("inputs").get<std::vector<int> >();
				if (definition.contains("output"))
					op.output = definition.at("output").get<int>();
				if (definition.contains("attributes")
					&& definition.at("attributes").contains("kernel"))
				{
					op.kernel = definition.at("attributes").at("kernel").get<std::string>();
				}
				if (definition.contains("attributes"))
				{
					const json &attributes = definition.at("attributes");
					op.dropoutPerc = attributes.value("dropoutPerc", 50.0f);
					op.padId = attributes.value("padId", 0);
				}
				ops.push_back(op);
			}
		}

		void loadValues(Tensor &tensorValue, const json &values)
		{
			if (!values.is_array() || values.size() != tensorValue.size)
				throw std::invalid_argument("Tensor data size does not match its shape");

			// Read numbers directly into their declared type. In particular, this
			// avoids rounding FLOAT64 values through the current Scalar alias (float).
			for (std::size_t i = 0; i < tensorValue.size; ++i)
			{
				switch (tensorValue.dtype)
				{
					case DType::FLOAT32:
						tensorValue.dataAs<float>()[i] = values[i].get<float>();
						break;
					case DType::FLOAT64:
						tensorValue.dataAs<double>()[i] = values[i].get<double>();
						break;
					case DType::INT32:
						tensorValue.dataAs<std::int32_t>()[i] = values[i].get<std::int32_t>();
						break;
					case DType::INT64:
						tensorValue.dataAs<std::int64_t>()[i] = values[i].get<std::int64_t>();
						break;
				}
			}
		}

		json tensorValues(const Tensor &tensorValue, bool gradients) const
		{
			json values = json::array();
			for (std::size_t i = 0; i < tensorValue.size; ++i)
			{
				switch (tensorValue.dtype)
				{
					case DType::FLOAT32:
						values.push_back(gradients ? tensorValue.gradAs<float>()[i] : tensorValue.dataAs<float>()[i]);
						break;
					case DType::FLOAT64:
						values.push_back(gradients ? tensorValue.gradAs<double>()[i] : tensorValue.dataAs<double>()[i]);
						break;
					case DType::INT32:
						values.push_back(gradients ? tensorValue.gradAs<std::int32_t>()[i] : tensorValue.dataAs<std::int32_t>()[i]);
						break;
					case DType::INT64:
						values.push_back(gradients ? tensorValue.gradAs<std::int64_t>()[i] : tensorValue.dataAs<std::int64_t>()[i]);
						break;
				}
			}
			return values;
		}
	};

	GraphRuntime::GraphRuntime(const json &graphDef, const std::string &weightsPath)
		: impl_(new Impl(graphDef))
	{
		json weights;
		if (!weightsPath.empty())
		{
			std::ifstream file(weightsPath.c_str());
			if (!file.is_open())
				throw std::runtime_error("Unable to open weights file: " + weightsPath);
			file >> weights;
		}
		impl_->loadTensors(weights);
		impl_->loadOps();
	}

	GraphRuntime::~GraphRuntime() {}

	void GraphRuntime::forward()
	{
		for (std::size_t i = 0; i < impl_->ops.size(); ++i)
		{
			const RuntimeOp &op = impl_->ops[i];
			if (op.name == "add")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("add: expected two inputs and one output");
				opAdd(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "matmul")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("matmul: expected two inputs and one output");
				opMatmul(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "embeddings")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("embeddings: expected two inputs and one output");
				opEmbeddings(op.inputs[0], op.inputs[1], op.output);
			}
			else if (op.name == "embeddings_mean_pooling")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("embeddings_mean_pooling: expected two inputs and one output");
				opEmbeddingsMeanPooling(op.inputs[0], op.inputs[1], op.output, op.padId);
			}
			else if (op.name == "padding_mask")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("padding_mask: expected one input and one output");
				opPaddingMask(op.inputs[0], op.output, op.padId);
			}
			else if (op.name == "mean_pooling")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("mean_pooling: expected input, mask, and output");
				opMeanPooling(op.inputs[0], op.inputs[1], op.output);
			}
			else if (op.name == "dropout")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("dropout: expected one input and one output");
				opDropout(op.inputs[0], op.output, op.dropoutPerc);
			}
			else if (op.name == "ReLU" || op.name == "relu")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("ReLU: expected one input and one output");
				opRelu(op.inputs[0], op.output);
			}
			else if (op.name == "gelu")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("gelu: expected one input and one output");
				opGelu(op.inputs[0], op.output);
			}
			else if (op.name == "silu")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("silu: expected one input and one output");
				opSilu(op.inputs[0], op.output);
			}
			else if (op.name == "softmax_ce_logits_label_int")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("CE logits label int: expected two inputs and one output");
				opCeLogitsLabelInt(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "mean")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("mean: expected one input and one output");
				opMean(op.inputs[0], op.output, op.kernel);
			}
			else if (op.name == "softmax")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("softmax: expected one input and one output");
				opSoftmax(op.inputs[0], op.output, op.kernel);
			}
			else
			{
				throw std::runtime_error("Op not supported: " + op.name);
			}
		}
	}

	void GraphRuntime::backward()
	{
		// Clear intermediate gradients but retain parameter gradients, matching
		// the behavior expected by the optimizer and by the previous runtime.
		for (std::size_t i = 0; i < impl_->tensors.size(); ++i)
		{
			Tensor &tensor = impl_->tensors[i];
			if (tensor.kind != "param")
				tensor.fillGradWithZeros();
		}

		// A graph created from a loss/output tensor starts with gradient one.
		setLossGrad(1.0f);

		for (std::size_t i = impl_->ops.size(); i > 0; --i)
		{
			const RuntimeOp &op = impl_->ops[i - 1];
			if (op.name == "add")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("add backward: expected two inputs and one output");
				backwardAdd(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "matmul")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("matmul backward: expected two inputs and one output");
				backwardMatmul(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "embeddings")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("embeddings backward: expected two inputs and one output");
				backwardEmbeddings(op.inputs[0], op.inputs[1], op.output);
			}
			else if (op.name == "embeddings_mean_pooling")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("embeddings_mean_pooling backward: expected two inputs and one output");
				backwardEmbeddingsMeanPooling(op.inputs[0], op.inputs[1], op.output, op.padId);
			}
			else if (op.name == "padding_mask")
			{
				// A discrete mask has no derivative with respect to token IDs.
			}
			else if (op.name == "mean_pooling")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("mean_pooling backward: expected input, mask, and output");
				backwardMeanPooling(op.inputs[0], op.inputs[1], op.output);
			}
			else if (op.name == "dropout")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("dropout backward: expected one input and one output");
				backwardDropout(op.inputs[0], op.output);
			}
			else if (op.name == "relu" || op.name == "ReLU")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("ReLU backward: expected one input and one output");
				backwardRelu(op.inputs[0], op.output);
			}
			else if (op.name == "gelu")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("gelu backward: expected one input and one output");
				backwardGelu(op.inputs[0], op.output);
			}
			else if (op.name == "silu")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("silu backward: expected one input and one output");
				backwardSilu(op.inputs[0], op.output);
			}
			else if (op.name == "softmax_ce_logits_label_int")
			{
				if (op.inputs.size() != 2 || op.output < 0)
					throw std::runtime_error("CE logits label int backward: expected two inputs and one output");
				backwardCeLogitsLabelInt(op.inputs[0], op.inputs[1], op.output, op.kernel);
			}
			else if (op.name == "mean")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("mean backward: expected one input and one output");
				backwardMean(op.inputs[0], op.output, op.kernel);
			}
			else if (op.name == "softmax")
			{
				if (op.inputs.size() != 1 || op.output < 0)
					throw std::runtime_error("softmax backward: expected one input and one output");
				backwardSoftmax(op.inputs[0], op.output, op.kernel);
			}
			else
			{
				throw std::runtime_error("Op backward not supported: " + op.name);
			}
		}
	}

	void GraphRuntime::opAdd(int aId, int bId, int outId, const std::string &kernel)
	{
		Tensor &A = impl_->tensor(aId);
		Tensor &B = impl_->tensor(bId);
		Tensor &C = impl_->tensor(outId);

		if (kernel == "ADD_1D_LAST")
			ADD_1D_LAST(A, B, C);
		else if (kernel == "ADD_2D_LAST")
			ADD_2D_LAST(A, B, C);
		else if (kernel == "ADD_3D_LAST")
			ADD_3D_LAST(A, B, C);
		else
			throw std::runtime_error("add: kernel not supported: " + kernel);
	}

	void GraphRuntime::backwardAdd(int aId, int bId, int outId, const std::string &kernel)
	{
		Tensor &A = impl_->tensor(aId);
		Tensor &B = impl_->tensor(bId);
		Tensor &C = impl_->tensor(outId);

		if (!A.requiresGrad && !B.requiresGrad)
			return;

		if (kernel == "ADD_1D_LAST")
			BACKWARD_ADD_1D_LAST(A, B, C);
		else if (kernel == "ADD_2D_LAST")
			BACKWARD_ADD_2D_LAST(A, B, C);
		else if (kernel == "ADD_3D_LAST")
			BACKWARD_ADD_3D_LAST(A, B, C);
		else
			throw std::runtime_error("add backward: kernel not supported: " + kernel);
	}

	void GraphRuntime::opMatmul(int aId, int bId, int outId, const std::string &kernel)
	{
		Tensor &A = impl_->tensor(aId);
		Tensor &B = impl_->tensor(bId);
		Tensor &C = impl_->tensor(outId);
		const std::string kernelName = kernel.empty()
			? "MATMUL_GENERIC_B_2D_2D_BROADCAST" : kernel;

		if (kernelName == "MATMUL_2D_2D")
			MATMUL_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_1B_2D_2D")
			MATMUL_1B_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_2B_2D_2D")
			MATMUL_2B_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_1B_2D_2D_LINEAR")
			MATMUL_1B_2D_2D_LINEAR(A, B, C);
		else if (kernelName == "MATMUL_GENERIC_B_2D_2D_BROADCAST")
			MATMUL_GENERIC_B_2D_2D_BROADCAST(A, B, C);
		else
			throw std::runtime_error("matmul: kernel not supported: " + kernelName);
	}

	void GraphRuntime::backwardMatmul(int aId, int bId, int outId, const std::string &kernel)
	{
		Tensor &A = impl_->tensor(aId);
		Tensor &B = impl_->tensor(bId);
		Tensor &C = impl_->tensor(outId);
		const std::string kernelName = kernel.empty()
			? "MATMUL_GENERIC_B_2D_2D_BROADCAST" : kernel;

		if (!A.requiresGrad && !B.requiresGrad)
			return;

		if (kernelName == "MATMUL_2D_2D")
			BACKWARD_MATMUL_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_1B_2D_2D")
			BACKWARD_MATMUL_1B_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_2B_2D_2D")
			BACKWARD_MATMUL_2B_2D_2D(A, B, C);
		else if (kernelName == "MATMUL_1B_2D_2D_LINEAR")
			BACKWARD_MATMUL_1B_2D_2D_LINEAR(A, B, C);
		else if (kernelName == "MATMUL_GENERIC_B_2D_2D_BROADCAST")
			BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST(A, B, C);
		else
			throw std::runtime_error("matmul backward: kernel not supported: " + kernelName);
	}

	void GraphRuntime::opRelu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		RELU(X, Y);
	}

	void GraphRuntime::backwardRelu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		if (!X.requiresGrad)
			return;
		BACKWARD_RELU(X, Y);
	}

	void GraphRuntime::opGelu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		GELU(X, Y);
	}

	void GraphRuntime::backwardGelu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		if (!X.requiresGrad)
			return;
		BACKWARD_GELU(X, Y);
	}

	void GraphRuntime::opSilu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		SILU(X, Y);
	}

	void GraphRuntime::backwardSilu(int inputId, int outputId)
	{
		Tensor &X = impl_->tensor(inputId);
		Tensor &Y = impl_->tensor(outputId);
		if (!X.requiresGrad)
			return;
		BACKWARD_SILU(X, Y);
	}

	void GraphRuntime::opEmbeddings(int idsId, int tableId, int outputId)
	{
		Tensor &ids = impl_->tensor(idsId);
		Tensor &table = impl_->tensor(tableId);
		Tensor &output = impl_->tensor(outputId);
		EMBEDDINGS(ids, table, output);
	}

	void GraphRuntime::backwardEmbeddings(int idsId, int tableId, int outputId)
	{
		Tensor &ids = impl_->tensor(idsId);
		Tensor &table = impl_->tensor(tableId);
		Tensor &output = impl_->tensor(outputId);
		if (!table.requiresGrad)
			return;
		BACKWARD_EMBEDDINGS(ids, table, output);
	}

	void GraphRuntime::opEmbeddingsMeanPooling(int idsId, int tableId, int outputId, int padId)
	{
		Tensor &ids = impl_->tensor(idsId);
		Tensor &table = impl_->tensor(tableId);
		Tensor &output = impl_->tensor(outputId);
		EMBEDDINGS_MEAN_POOLING(ids, table, output, padId);
	}

	void GraphRuntime::backwardEmbeddingsMeanPooling(int idsId, int tableId, int outputId, int padId)
	{
		Tensor &ids = impl_->tensor(idsId);
		Tensor &table = impl_->tensor(tableId);
		Tensor &output = impl_->tensor(outputId);
		if (!table.requiresGrad)
			return;
		BACKWARD_EMBEDDINGS_MEAN_POOLING(ids, table, output, padId);
	}

	void GraphRuntime::opPaddingMask(int inputId, int outputId, int padId)
	{
		Tensor &ids = impl_->tensor(inputId);
		Tensor &output = impl_->tensor(outputId);
		PADDING_MASK(ids, output, padId);
	}

	void GraphRuntime::opMeanPooling(int inputId, int maskId, int outputId)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &mask = impl_->tensor(maskId);
		Tensor &output = impl_->tensor(outputId);
		MEAN_POOLING(input, mask, output);
	}

	void GraphRuntime::backwardMeanPooling(int inputId, int maskId, int outputId)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &mask = impl_->tensor(maskId);
		Tensor &output = impl_->tensor(outputId);
		if (!input.requiresGrad)
			return;
		BACKWARD_MEAN_POOLING(input, mask, output);
	}

	void GraphRuntime::opDropout(int inputId, int outputId, Scalar dropoutPerc)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &output = impl_->tensor(outputId);
		const bool training = impl_->mode == ExecutionMode::TRAIN;
		if (training)
		{
			std::vector<Scalar> &mask = impl_->dropoutMasks[outputId];
			mask.resize(input.size);
			impl_->dropoutSeed += 0x9e3779b97f4a7c15ULL;
			DROPOUT(input, output, dropoutPerc, mask.empty() ? 0 : mask.data(),
				impl_->dropoutSeed, true);
		}
		else
		{
			impl_->dropoutMasks.erase(outputId);
			DROPOUT(input, output, dropoutPerc, 0, 0, false);
		}
	}

	void GraphRuntime::backwardDropout(int inputId, int outputId)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &output = impl_->tensor(outputId);
		if (!input.requiresGrad)
			return;
		std::unordered_map<int, std::vector<Scalar> >::const_iterator found =
			impl_->dropoutMasks.find(outputId);
		if (found == impl_->dropoutMasks.end())
			throw std::runtime_error("dropout backward: forward mask is missing");
		const std::vector<Scalar> &mask = found->second;
		if (mask.size() != input.size)
			throw std::runtime_error("dropout backward: forward mask size mismatch");
		BACKWARD_DROPOUT(input, output, mask.empty() ? 0 : mask.data());
	}

	void GraphRuntime::opCeLogitsLabelInt(
		int logitsId,
		int targetId,
		int outputId,
		const std::string &kernel)
	{
		Tensor &logits = impl_->tensor(logitsId);
		Tensor &target = impl_->tensor(targetId);
		Tensor &output = impl_->tensor(outputId);
		const std::string kernelName = kernel.empty()
			? "CE_LOGITS_LABEL_INT_GENERIC_AXIS" : kernel;

		if (kernelName == "CE_LOGITS_LABEL_INT_1D_LAST")
			CE_LOGITS_LABEL_INT_1D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_2D_LAST")
			CE_LOGITS_LABEL_INT_2D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_3D_LAST")
			CE_LOGITS_LABEL_INT_3D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_GENERIC_AXIS")
			CE_LOGITS_LABEL_INT_GENERIC_AXIS(logits, target, output);
		else
			throw std::runtime_error("CE logits label int: kernel not supported: " + kernelName);
	}

	void GraphRuntime::backwardCeLogitsLabelInt(
		int logitsId,
		int targetId,
		int outputId,
		const std::string &kernel)
	{
		Tensor &logits = impl_->tensor(logitsId);
		if (!logits.requiresGrad || logits.size == 0)
			return;

		Tensor &target = impl_->tensor(targetId);
		Tensor &output = impl_->tensor(outputId);
		const std::string kernelName = kernel.empty()
			? "CE_LOGITS_LABEL_INT_GENERIC_AXIS" : kernel;

		if (kernelName == "CE_LOGITS_LABEL_INT_1D_LAST")
			BACKWORD_CE_LOGITS_LABEL_INT_1D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_2D_LAST")
			BACKWORD_CE_LOGITS_LABEL_INT_2D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_3D_LAST")
			BACKWORD_CE_LOGITS_LABEL_INT_3D_LAST(logits, target, output);
		else if (kernelName == "CE_LOGITS_LABEL_INT_GENERIC_AXIS")
			BACKWORD_CE_LOGITS_LABEL_INT_GENERIC_AXIS(logits, target, output);
		else
			throw std::runtime_error("CE logits label int backward: kernel not supported: " + kernelName);
	}

	void GraphRuntime::opMean(int inputId, int outputId, const std::string &kernel)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &output = impl_->tensor(outputId);
		const std::string kernelName = kernel.empty() ? "MEAN_GENERIC_AXIS" : kernel;

		if (kernelName == "MEAN_1D_FIRST")
			MEAN_1D_FIRST(input, output);
		else if (kernelName == "MEAN_2D_FIRST")
			MEAN_2D_FIRST(input, output);
		else if (kernelName == "MEAN_3D_FIRST")
			MEAN_3D_FIRST(input, output);
		else if (kernelName == "MEAN_GENERIC_AXIS")
			MEAN_GENERIC_AXIS(input, output);
		else
			throw std::runtime_error("mean: kernel not supported: " + kernelName);
	}

	void GraphRuntime::backwardMean(int inputId, int outputId, const std::string &kernel)
	{
		Tensor &input = impl_->tensor(inputId);
		if (!input.requiresGrad)
			return;
		Tensor &output = impl_->tensor(outputId);
		const std::string kernelName = kernel.empty() ? "MEAN_GENERIC_AXIS" : kernel;

		if (kernelName == "MEAN_1D_FIRST")
			BACKWARD_MEAN_1D_FIRST(input, output);
		else if (kernelName == "MEAN_2D_FIRST")
			BACKWARD_MEAN_2D_FIRST(input, output);
		else if (kernelName == "MEAN_3D_FIRST")
			BACKWARD_MEAN_3D_FIRST(input, output);
		else if (kernelName == "MEAN_GENERIC_AXIS")
			BACKWARD_MEAN_GENERIC_AXIS(input, output);
		else
			throw std::runtime_error("mean backward: kernel not supported: " + kernelName);
	}

	void GraphRuntime::opSoftmax(int inputId, int outputId, const std::string &kernel)
	{
		Tensor &input = impl_->tensor(inputId);
		Tensor &output = impl_->tensor(outputId);
		if (input.size == 0)
			return;
		const std::string kernelName = kernel.empty() ? "SOFTMAX_GENERIC_AXIS" : kernel;

		if (kernelName == "SOFTMAX_1D_LAST")
			SOFTMAX_1D_LAST(input, output);
		else if (kernelName == "SOFTMAX_2D_LAST")
			SOFTMAX_2D_LAST(input, output);
		else if (kernelName == "SOFTMAX_3D_LAST")
			SOFTMAX_3D_LAST(input, output);
		else if (kernelName == "SOFTMAX_4D_LAST")
			SOFTMAX_4D_LAST(input, output);
		else if (kernelName == "SOFTMAX_GENERIC_AXIS")
			SOFTMAX_GENERIC_AXIS(input, output);
		else
			throw std::runtime_error("softmax: kernel not supported: " + kernelName);
	}

	void GraphRuntime::backwardSoftmax(int inputId, int outputId, const std::string &kernel)
	{
		Tensor &input = impl_->tensor(inputId);
		if (!input.requiresGrad || input.size == 0)
			return;
		Tensor &output = impl_->tensor(outputId);
		const std::string kernelName = kernel.empty() ? "SOFTMAX_GENERIC_AXIS" : kernel;

		if (kernelName == "SOFTMAX_1D_LAST")
			BACKWORD_SOFTMAX_1D_LAST(input, output);
		else if (kernelName == "SOFTMAX_2D_LAST")
			BACKWORD_SOFTMAX_2D_LAST(input, output);
		else if (kernelName == "SOFTMAX_3D_LAST")
			BACKWORD_SOFTMAX_3D_LAST(input, output);
		else if (kernelName == "SOFTMAX_4D_LAST")
			BACKWORD_SOFTMAX_4D_LAST(input, output);
		else if (kernelName == "SOFTMAX_GENERIC_AXIS")
			BACKWORD_SOFTMAX_GENERIC_AXIS(input, output);
		else
			throw std::runtime_error("softmax backward: kernel not supported: " + kernelName);
	}

	std::size_t GraphRuntime::inputSize() const
	{
		return getTensorSize(impl_->inputId);
	}

	std::size_t GraphRuntime::outputSize() const
	{
		return getTensorSize(impl_->outputId);
	}

	void GraphRuntime::setInput(const std::vector<Scalar> &values)
	{
		Tensor &input = impl_->tensor(impl_->inputId);
		if (values.size() != input.size)
			throw std::invalid_argument("Input data size does not match input tensor shape");
		for (std::size_t i = 0; i < values.size(); ++i)
			input.writeData(i, values[i]);
	}

	void GraphRuntime::setTarget(const std::vector<Scalar> &values)
	{
		Tensor &target = impl_->tensor(impl_->targetId);
		if (values.size() != target.size)
			throw std::invalid_argument("Target data size does not match target tensor shape");
		for (std::size_t i = 0; i < values.size(); ++i)
			target.writeData(i, values[i]);
	}

	std::vector<Scalar> GraphRuntime::getOutput() const
	{
		const Tensor &output = impl_->tensor(impl_->outputId);
		std::vector<Scalar> values(output.size);
		for (std::size_t i = 0; i < output.size; ++i)
			values[i] = output.readData(i);
		return values;
	}

	std::vector<Scalar> GraphRuntime::getLoss() const
	{
		const Tensor &loss = impl_->tensor(impl_->lossId);
		std::vector<Scalar> values(loss.size);
		for (std::size_t i = 0; i < loss.size; ++i)
			values[i] = loss.readData(i);
		return values;
	}

	Scalar GraphRuntime::getError() const
	{
		const std::vector<Scalar> values = getLoss();
		if (values.empty())
			return 0.0f;
		Scalar sum = 0.0f;
		for (std::size_t i = 0; i < values.size(); ++i)
			sum += values[i];
		return sum / static_cast<Scalar>(values.size());
	}

	std::vector<int> GraphRuntime::getTensorShape(int id) const
	{
		return impl_->tensor(id).shape;
	}

	std::size_t GraphRuntime::getTensorSize(int id) const
	{
		return impl_->tensor(id).size;
	}

	int GraphRuntime::getTensorDType(int id) const
	{
		return static_cast<int>(impl_->tensor(id).dtype);
	}

	Scalar GraphRuntime::getTensorDataValue(int id, std::size_t index) const
	{
		return impl_->tensor(id).readData(index);
	}

	Scalar GraphRuntime::getTensorGradValue(int id, std::size_t index) const
	{
		return impl_->tensor(id).readGrad(index);
	}

	void GraphRuntime::setTensorDataValue(int id, std::size_t index, Scalar value)
	{
		impl_->tensor(id).writeData(index, value);
	}

	bool GraphRuntime::tensorHasFloatingPointDType(int id) const
	{
		const DType dtype = impl_->tensor(id).dtype;
		return dtype == DType::FLOAT32 || dtype == DType::FLOAT64;
	}

	const std::vector<int> &GraphRuntime::getTrainableTensorIds() const
	{
		return impl_->trainable;
	}

	void GraphRuntime::resetGrad()
	{
		for (std::size_t tensorIndex = 0; tensorIndex < impl_->tensors.size(); ++tensorIndex)
		{
			Tensor &tensorValue = impl_->tensors[tensorIndex];
			tensorValue.fillGradWithZeros();
		}
	}

	void GraphRuntime::setLossGrad(Scalar lossGrad)
	{
		if (impl_->lossId < 0)
			return;
		Tensor &loss = impl_->tensor(impl_->lossId);
		for (std::size_t element = 0; element < loss.size; ++element)
			loss.writeGrad(element, lossGrad);
	}

	void GraphRuntime::saveWeightsToJson(const std::string &path) const
	{
		json tensorDefinitions = json::object();
		for (std::size_t i = 0; i < impl_->trainable.size(); ++i)
		{
			const Tensor &tensorValue = impl_->tensor(impl_->trainable[i]);
			tensorDefinitions[std::to_string(tensorValue.id)] = {
				{"data", impl_->tensorValues(tensorValue, false)},
				{"shape", tensorValue.shape},
				{"dtype", static_cast<int>(tensorValue.dtype)}
			};
		}

		std::ofstream file(path.c_str(), std::ios::trunc);
		if (!file.is_open())
			throw std::runtime_error("Unable to open weights file for writing: " + path);
		file << json({{"tensors", tensorDefinitions}}).dump();
	}

	void GraphRuntime::saveToJson(const std::string &path) const
	{
		json tensorDefinitions = json::object();
		for (std::size_t i = 0; i < impl_->tensors.size(); ++i)
		{
			const Tensor &tensorValue = impl_->tensors[i];
			tensorDefinitions[std::to_string(tensorValue.id)] = impl_->tensorValues(tensorValue, false);
		}

		std::ofstream file(path.c_str(), std::ios::trunc);
		if (!file.is_open())
			throw std::runtime_error("Unable to open graph output file: " + path);
		file << json({{"graph", impl_->graphDef}, {"tensors", tensorDefinitions}}).dump();
	}

	void GraphRuntime::setMode(ExecutionMode mode)
	{
		impl_->mode = mode;
	}

	void GraphRuntime::enableProfiler()
	{
		impl_->profiler.reset(new Profiler());
		impl_->profilingEnabled = true;
	}

	bool GraphRuntime::isProfilingEnabled() const
	{
		return impl_->profilingEnabled;
	}

	void GraphRuntime::setProfilingEnabled(bool enabled)
	{
		impl_->profilingEnabled = enabled && impl_->profiler.get() != 0;
	}

	Profiler &GraphRuntime::getProfiler()
	{
		if (!impl_->profiler)
			throw std::runtime_error("Profiler is not enabled");
		return *impl_->profiler;
	}
}
