#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "Core.hpp"
#include "../Optimizers/Optimizers.hpp"
#include "../Dataset/stream_file_dataset.hpp"
#include "../Dataset/HDF5Dataset.hpp"
#include "../Utility/Utility.hpp"
#include "../Utility/ProfileWriter.hpp"

namespace PHP2xAI::Runtime::CPP
{
	Core::Core(const std::string &configPath, const std::string &weightsPath)
		: graphPath_(configPath), weightsPath_(weightsPath)
	{
		auto configDef = loadJson(graphPath_);
		loadGraphRuntime(configDef);

		if (configDef.contains("profiler_output_path"))
			loadProfilerOutputPath(configDef);

		if (configDef.contains("optimizer"))
			loadOptimizer(configDef);

		if (configDef.contains("train_data_file") && configDef.contains("val_data_file") && configDef.contains("batch_size"))
			loadTrainValidateDataset(configDef);

		if (configDef.contains("save_Path"))
			loadOutputPath(configDef);

		if (configDef.contains("epochs_number"))
			loadEpochsNumber(configDef);

		if (configDef.contains("log_on_each_x_batch"))
			logOnEachXBatch_ = configDef.at("log_on_each_x_batch").get<int>();
	}
	
	json Core::loadJson(const std::string &path)
	{
		std::ifstream file(path);

		if (!file.is_open())
			throw std::runtime_error("Unable to open graph file: " + path);

		json parsed;
		file >> parsed;
		return parsed;
	}
	
	void Core::loadGraphRuntime(const json &configDef)
	{
		const auto &graphDef = configDef.at("graph");
		graphRuntime_.reset(new GraphRuntime(graphDef, weightsPath_));
	}
	
	void Core::loadOptimizer(const json &configDef)
	{
		const auto &optimizerDef = configDef.at("optimizer");
		const auto name = optimizerDef.at("name").get<std::string>();
		const auto &params = optimizerDef.at("params");

		if (name == "Adam")
		{
			const auto learningRate = params.value("learningRate", 0.1f);
			const auto beta1 = params.value("beta1", 0.9f);
			const auto beta2 = params.value("beta2", 0.999f);
			const auto eps = params.value("eps", 0.00000001f);

			optimizer_ = std::make_unique<Optimizers::Adam>(learningRate, beta1, beta2, eps);
		}
		else if (name == "Fixed")
		{
			const auto learningRate = params.value("learningRate", 0.1f);
			optimizer_ = std::make_unique<Optimizers::Fixed>(learningRate);
		}
		else
		{
			throw std::runtime_error("Unsupported optimizer: " + name);
		}
	}
	
	void Core::loadTrainValidateDataset(const json &configDef)
	{
		const auto trainPath = configDef.at("train_data_file").get<std::string>();
		const auto valPath = configDef.at("val_data_file").get<std::string>();
		const auto batchSize = static_cast<std::size_t>(configDef.at("batch_size").get<int>());
		const auto datasetType = configDef.value("dataset_type", std::string("TXT"));

		if (datasetType == "TXT")
		{
			trainDataset_ = std::make_unique<StreamFileDataset>(trainPath, batchSize);
			valDataset_ = std::make_unique<StreamFileDataset>(valPath, batchSize);
		}
		else if (datasetType == "HDF5")
		{
			trainDataset_ = std::make_unique<HDF5Dataset>(trainPath, batchSize);
			valDataset_ = std::make_unique<HDF5Dataset>(valPath, batchSize);
		}
		else
		{
			throw std::runtime_error("Unsupported dataset type: " + datasetType);
		}

		trainValDataset_.emplace(*trainDataset_, *valDataset_);
	}
	
	void Core::loadOutputPath(const json &configDef)
	{
		outputPath_ = configDef.at("save_Path").get<std::string>();
	}
	
	void Core::loadEpochsNumber(const json &configDef)
	{
		epochsNumber_ = configDef.at("epochs_number").get<int>();
	}

	void Core::loadProfilerOutputPath(const json &configDef)
	{
		profilerOutputPath_ = configDef.at("profiler_output_path").get<std::string>();
		if (!profilerOutputPath_.empty())
			graphRuntime_->enableProfiler();
	}

	int Core::predictLabelInt(const std::vector<Scalar> &x)
	{
		const auto output = predict(x);
		return Utility::argmax(output);
	}

	std::size_t Core::inputSize() const
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		return graphRuntime_->inputSize();
	}
	
	std::size_t Core::outputSize() const
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		return graphRuntime_->outputSize();
	}

	std::vector<Scalar> Core::predict(const std::vector<Scalar> &x)
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		auto *graph = graphRuntime_.get();
		graph->setMode(ExecutionMode::INFER);
		graph->setInput(x);
		graph->forward();
		
		return graph->getOutput();
	}
	
	void Core::train()
	{
		if (!trainValDataset_ || !graphRuntime_ || !optimizer_)
			throw std::runtime_error("Core not initialized");

		auto &dataset = *trainValDataset_;
		auto *graph = graphRuntime_.get();
		graph->setMode(ExecutionMode::TRAIN);

		std::vector<Scalar> x;
		std::vector<Scalar> y;
		auto betterValidationLoss = std::numeric_limits<Scalar>::max();
		std::size_t profileBatchIndex = 0;
		
		for (int i = 0; i < epochsNumber_; ++i)
		{
			std::cout << "Epoch " << (i + 1) << "\n";
			std::cout << "------------------------\n";
			std::cout.flush();
			
			dataset.train.shuffleEpoch();
			std::size_t indice = 0;
			
			while (dataset.train.nextBatch())
			{
				graph->resetGrad();
				graph->setLossGrad(1.0f);
				
				dataset.train.pack(x, y);
				
				graph->setInput(x);
				graph->setTarget(y);
				graph->forward();
				
				const auto error = graph->getError();
				
				graph->backward();
				optimizer_->step(*graph);

				if (graph->isProfilingEnabled())
				{
					ProfileWriter::appendBatch(graph->getProfiler(), profilerOutputPath_, ++profileBatchIndex);
					graph->getProfiler().clear();
				}
				
				++indice;
				
				if (logOnEachXBatch_ > 0 && (indice % logOnEachXBatch_) == 0)
				{
					std::cout << "Train error batch " << indice << ": " << error << "\n";
					std::cout.flush();
				}
			}
			
			const bool profileTraining = graph->isProfilingEnabled();
			graph->setProfilingEnabled(false);
			const auto valLoss = validationLoss();
			graph->setProfilingEnabled(profileTraining);
			
			std::cout << "------------------------\n";
			std::cout << "Validation error: " << valLoss << "\n";
			std::cout.flush();

			if (!outputPath_.empty() && valLoss < betterValidationLoss)
			{
				betterValidationLoss = valLoss;
				graph->saveWeightsToJson(outputPath_);
			}
			else
			{
				std::cout << "------------------------\n";
				std::cout << "Validation error increased\n";
				std::cout.flush();
			}
			std::cout << "------------------------\n";
			std::cout.flush();
		}
	}

	Scalar Core::validationLoss()
	{
		if (!trainValDataset_ || !graphRuntime_)
			throw std::runtime_error("Core not initialized");

		auto &dataset = trainValDataset_->val;
		auto *graph = graphRuntime_.get();
		graph->setMode(ExecutionMode::INFER);

		std::vector<Scalar> x;
		std::vector<Scalar> y;
		Scalar loss = 0.0f;
		std::size_t count = 0;

		dataset.resetEpoch();

		try
		{
			while (dataset.nextBatch())
			{
				dataset.pack(x, y);
				
				graph->setInput(x);
				graph->setTarget(y);
				graph->forward();
				
				loss += graph->getError();
				++count;
			}
		}
		catch (...)
		{
			// The same runtime continues the training loop after validation.
			graph->setMode(ExecutionMode::TRAIN);
			throw;
		}

		const Scalar validationLoss = count > 0 ? loss / static_cast<Scalar>(count) : 0.0f;
		// Restore dropout and TRAIN-only backward behavior for the next epoch.
		graph->setMode(ExecutionMode::TRAIN);
		return validationLoss;
	}
}
