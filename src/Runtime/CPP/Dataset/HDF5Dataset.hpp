#pragma once

#include "BatchDataset.hpp"
#include "PHP2XAIHDF5.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace PHP2xAI::Runtime::CPP
{
	class HDF5Dataset : public BatchDataset
	{
	public:
		explicit HDF5Dataset(
			std::string filename,
			std::size_t batchSize,
			std::string xField = "x",
			std::string yField = "y",
			std::uint32_t seed = 42);

		std::size_t numBatches() const;
		std::int64_t sampleCount() const;
		std::size_t batchSize() const;
		std::string getType() const override;

		void shuffleEpoch() override;
		void resetEpoch() override;
		bool nextBatch() override;
		void pack(std::vector<float>& xPacked, std::vector<float>& yPacked) override;

	private:
		std::unique_ptr<PHP2XAIHDF5> dataset_;
		std::string filename_;
		std::size_t batchSize_;
		std::string xField_;
		std::string yField_;
		std::int64_t sampleCount_;
		std::size_t xElementsPerSample_;
		std::size_t yElementsPerSample_;
		PHP2XAIHDF5::FieldMetadata xMetadata_;
		PHP2XAIHDF5::FieldMetadata yMetadata_;
		std::mt19937 rng_;
		std::vector<std::int64_t> sampleOrder_;
		std::vector<std::int64_t> currentIndices_;
		std::size_t batchPosition_ = 0;

		static std::size_t elementCount(const std::vector<std::int64_t>& shape);
		void readFieldAsFloat(
			const std::string& field,
			PHP2XAIHDF5::DType dtype,
			std::size_t elementsPerSample,
			const std::vector<std::int64_t>& indices,
			std::vector<float>& output) const;
	};
}
