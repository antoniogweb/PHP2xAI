#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <vector>

namespace PHP2xAI::Runtime::CPP
{
	class StreamFileDataset
	{
	public:
		explicit StreamFileDataset(std::string path,
								std::size_t batchSize,
								char delimiter = '|',
								uint32_t seed = 42);

		std::size_t numBatches() const;

		void shuffleEpoch();
		void resetEpoch();

		// Equivalente del foreach($dataset as $batch)
		bool nextBatch();

		// Equivalente del foreach($batch as [$x,$y])
		bool nextSampleInBatch(std::vector<float>& x, std::vector<float>& y);

		// Pack del batch corrente in row-major: ritorna xPacked e yPacked
		void pack(std::vector<float>& xPacked, std::vector<float>& yPacked);
		
		// Print the vector
		static void printVec(const char* label, const std::vector<float>& v);

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

		void resetOrder_();
		void resetEpoch_();
		void buildBatchOffsets_();
		void seekToBatchStart_(std::size_t batchId);

		static bool isBlank_(const std::string& s);
		void parseLineXY_(const std::string& line, std::vector<float>& x, std::vector<float>& y) const;
		static void parseFloatVector_(std::string_view sv, std::vector<float>& out);
	};
}
