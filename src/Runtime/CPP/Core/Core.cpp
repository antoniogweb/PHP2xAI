#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include "Core.hpp"
#include "../Optimizers/Optimizers.hpp"
#include "../Utility/Utility.hpp"

namespace PHP2xAI::Runtime::CPP
{
	Core::Core(const std::string &configPath, const std::string &weightsPath)
		: graphPath_(configPath), weightsPath_(weightsPath)
	{
		auto configDef = loadJson(graphPath_);
		loadGraphRuntime(configDef);

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
		graphRuntime_.emplace(graphDef, weightsPath_);
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

		trainDataset_.emplace(trainPath, batchSize);
		valDataset_.emplace(valPath, batchSize);
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

	int Core::predictLabelInt(const std::vector<Scalar> &x)
	{
		const auto output = predict(x);
		return Utility::argmax(output);
	}

	std::size_t Core::inputSize() const
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		const auto &tensor = graphRuntime_->tensors[graphRuntime_->inputId];
		
		return std::accumulate(
			tensor.shape.begin(),
			tensor.shape.end(),
			static_cast<std::size_t>(1),
			std::multiplies<std::size_t>());
	}
	
	std::size_t Core::outputSize() const
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		const auto &tensor = graphRuntime_->tensors[graphRuntime_->outputId];
		
		return std::accumulate(
			tensor.shape.begin(),
			tensor.shape.end(),
			static_cast<std::size_t>(1),
			std::multiplies<std::size_t>());
	}

	std::vector<Scalar> Core::predict(const std::vector<Scalar> &x)
	{
		if (!graphRuntime_)
			throw std::runtime_error("Core not initialized");
		
		auto &graph = *graphRuntime_;
		
		graph.setInput(x);
		graph.forward();
		
		return graph.getOutput();
	}
	
	void Core::train()
	{
		if (!trainValDataset_ || !graphRuntime_ || !optimizer_)
			throw std::runtime_error("Core not initialized");

		auto &dataset = *trainValDataset_;
		auto &graph = *graphRuntime_;

		std::vector<Scalar> x;
		std::vector<Scalar> y;
		auto betterValidationLoss = std::numeric_limits<Scalar>::max();
		
		for (int i = 0; i < epochsNumber_; ++i)
		{
			std::cout << "Epoch " << (i + 1) << "\n";
			std::cout << "------------------------\n";
			std::cout.flush();
			
			dataset.train.shuffleEpoch();
			std::size_t indice = 0;
			
			while (dataset.train.nextBatch())
			{
				while (dataset.train.nextSampleInBatch(x, y))
				{
					graph.setInput(x);
					graph.setTarget(y);
					graph.forward();
					
					optimizer_->addError(graph.getLoss());
					
					graph.backward();
				}
				
				const auto error = optimizer_->getError();
				optimizer_->step(graph);
				optimizer_->zeroGrads(graph);
				
				++indice;
				
				if (logOnEachXBatch_ > 0 && (indice % logOnEachXBatch_) == 0)
				{
					std::cout << "Train error batch " << indice << ": " << error << "\n";
					std::cout.flush();
				}
			}
			
			const auto valLoss = validationLoss();
			
			std::cout << "------------------------\n";
			std::cout << "Validation error: " << valLoss << "\n";
			std::cout.flush();

			if (!outputPath_.empty() && valLoss < betterValidationLoss)
			{
				betterValidationLoss = valLoss;
				graph.saveWeightsToJson(outputPath_);
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
		auto &graph = *graphRuntime_;

		std::vector<Scalar> x;
		std::vector<Scalar> y;
		Scalar loss = 0.0f;
		std::size_t count = 0;

		dataset.resetEpoch();

		while (dataset.nextBatch())
		{
			while (dataset.nextSampleInBatch(x, y))
			{
				graph.setInput(x);
				graph.setTarget(y);
				graph.forward();
				
				loss += graph.getLoss();
				++count;
			}
		}

		return count > 0 ? loss / static_cast<Scalar>(count) : 0.0f;
	}
}
