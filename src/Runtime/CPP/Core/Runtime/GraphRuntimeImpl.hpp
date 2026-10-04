#include "../../types.hpp"
#include "../runtime.hpp"
#include "../Tensor/Tensor.hpp"
#include "../Kernels/DTypeDispatch.hpp"

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
		void copyValues(const T *source, T *destination, std::size_t count)
		{
			for (std::size_t i = 0; i < count; ++i)
				destination[i] = source[i];
		}

		void computeStrides(const std::vector<int> &shape, std::vector<int> &strides)
		{
			strides.resize(shape.size());
			int stride = 1;
			for (std::size_t i = shape.size(); i > 0; --i)
			{
				strides[i - 1] = stride;
				stride *= shape[i - 1];
			}
		}
	}
	
	struct RuntimeOp
	{
		std::string name;
		std::vector<int> inputs;
		std::vector<int> outputs;
		int output;
		int layer;
		std::string kernel;
		Scalar dropoutPerc;
		int padId;
		std::vector<int> axes;
		int start;
		int end;
		int offset;
		Scalar scale;
		Scalar base;
		Scalar eps;

		RuntimeOp() : output(-1), layer(-1), dropoutPerc(50.0f), padId(0), start(0), end(0),
			offset(0), scale(1.0f), base(10000.0f), eps(1.0e-5f) {}
	};

	struct KvCacheEntry
	{
		DType dtype;
		std::vector<int> shape;
		TensorBuffer key;
		TensorBuffer value;

		KvCacheEntry() : dtype(DType::FLOAT32) {}
	};
	
	struct GraphRuntime::Impl
	{
		json graphDef;
		std::vector<Tensor> tensors;
		std::vector<RuntimeOp> ops;
		std::unordered_map<int, std::size_t> tensorIndices;
		std::vector<int> trainable;
		std::unordered_map<int, std::vector<Scalar> > dropoutMasks;
		std::unordered_map<int, KvCacheEntry> kvCaches;
		std::uint64_t dropoutSeed;
		int inputId;
		int targetId;
		int outputId;
		int lossId;
		ExecutionMode mode;
		int ropeOffset;
		int ropeOffsetIncrement;
		bool hasKvCacheInForward;
		std::unique_ptr<Profiler> profiler;
		bool profilingEnabled;

		Impl(const json &definition)
			: graphDef(definition), dropoutSeed(0x9e3779b97f4a7c15ULL),
				inputId(-1), targetId(-1), outputId(-1), lossId(-1),
				mode(ExecutionMode::INFER), ropeOffset(-1), ropeOffsetIncrement(1),
				hasKvCacheInForward(false), profilingEnabled(false)
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
				if (definition.contains("outputs"))
					op.outputs = definition.at("outputs").get<std::vector<int> >();
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
					op.axes = attributes.value("axes", std::vector<int>());
					op.start = attributes.value("start", 0);
					op.end = attributes.value("end", 0);
					op.offset = attributes.value("offset", 0);
					op.scale = attributes.value("scale", 1.0f);
					op.base = attributes.value("base", 10000.0f);
					op.eps = attributes.value("eps", 1.0e-5f);
					op.layer = attributes.value("layer", -1);
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
				dispatchDType(tensorValue.dtype, [&]<typename T>()
				{
					tensorValue.dataAs<T>()[i] = values[i].get<T>();
				});
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
}
