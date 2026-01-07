#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include "../types.hpp"
#include "../Dataset/TrainValidateDataset.hpp"
#include "../Optimizers/Optimizer.hpp"
#include "../ThirdParty/nlohmann/json.hpp"
#include "runtime.hpp"

namespace PHP2xAI::Runtime::CPP
{
	using nlohmann::json;
	
	class Core
	{
	public:
		explicit Core(const std::string &configPath, const std::string &weightsPath = "");

		void train();
		Scalar validationLoss();
		int predictLabelInt(const std::vector<Scalar> &x);
		std::vector<Scalar> predict(const std::vector<Scalar> &x);
		std::size_t inputSize() const;
		std::size_t outputSize() const;

	private:
		std::string graphPath_;
		std::string weightsPath_;
		std::unique_ptr<Optimizers::Optimizer> optimizer_;
		std::optional<StreamFileDataset> trainDataset_;
		std::optional<StreamFileDataset> valDataset_;
		std::optional<TrainValidateDataset> trainValDataset_;
		std::optional<GraphRuntime> graphRuntime_;
		std::string outputPath_;
		int epochsNumber_{};
		int logOnEachXBatch_ = 1;
		
		static json loadJson(const std::string &path);
		void loadGraphRuntime(const json &configDef);
		void loadOptimizer(const json &configDef);
		void loadTrainValidateDataset(const json &configDef);
		void loadOutputPath(const json &configDef);
		void loadEpochsNumber(const json &configDef);
	};
}
