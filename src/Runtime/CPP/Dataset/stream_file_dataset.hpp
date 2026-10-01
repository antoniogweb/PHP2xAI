#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "BatchDataset.hpp"

namespace PHP2xAI::Runtime::CPP
{
	class StreamFileDataset : public BatchDataset
	{
	public:
		explicit StreamFileDataset(std::string path,
								std::size_t batchSize,
								char delimiter = '|',
								uint32_t seed = 42);

		std::size_t numBatches() const;
		std::string getType() const override;

		void shuffleEpoch() override;
		void resetEpoch() override;

		// Equivalente del foreach($dataset as $batch)
		bool nextBatch() override;

		// Equivalente del foreach($batch as [$x,$y])
		template <typename X, typename Y>
		bool nextSampleInBatch(std::vector<X>& x, std::vector<Y>& y)
		{
			return nextSampleRaw(&x, dtypeOf<X>(), &y, dtypeOf<Y>());
		}

		// Pack del batch corrente in row-major: ritorna xPacked e yPacked
		template <typename X, typename Y>
		void pack(std::vector<X>& xPacked, std::vector<Y>& yPacked)
		{
			BatchDataset::pack(xPacked, yPacked);
		}
		
		// Print the vector
		template <typename T>
		static void printVec(const char* label, const std::vector<T>& v)
		{
			std::cout << label << "=[";
			for (std::size_t i = 0; i < v.size(); ++i)
			{
				std::cout << v[i];
				if (i + 1 < v.size())
					std::cout << ' ';
			}
			std::cout << "]";
		}

	private:
		std::string path_;
		std::size_t batchSize_;
		char delimiter_;

		std::mt19937 rng_;
		std::ifstream file_;

		std::vector<std::streampos> batchOffsets_; // offset byte di inizio batch (uno ogni batchSize righe)
		std::vector<std::size_t> batchOrder_;      // permutazione dei batch
		std::size_t curBatchPos_ = 0;              // posizione nell'ordine dei batch
		std::size_t curInBatch_ = 0;               // sample letti nel batch corrente
		std::size_t numLines_ = 0;

		void packRaw(void* xVector, void* yVector) override;
		bool nextSampleRaw(void* xVector, DType xDType, void* yVector, DType yDType);

		void resetOrder_();
		void resetEpoch_();
		void buildBatchOffsets_();
		void seekToBatchStart_(std::size_t batchId);

		static bool isBlank_(const std::string& s);
		template <typename X, typename Y>
		void parseLineXY_(const std::string& line, std::vector<X>& x, std::vector<Y>& y) const;
		template <typename T>
		static void parseFloatVector_(std::string_view sv, std::vector<T>& out);
	};
}
