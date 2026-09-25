#pragma once

#include <cstddef>
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
		virtual void MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_1B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_2B_2D_2D(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_1B_2D_2D_LINEAR(Tensor &A, Tensor &B, Tensor &C);
		virtual void BACKWARD_MATMUL_GENERIC_B_2D_2D_BROADCAST(Tensor &A, Tensor &B, Tensor &C);
	};
}
