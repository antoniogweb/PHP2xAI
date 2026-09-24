#include "HDF5Dataset.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace PHP2xAI::Runtime::CPP
{
	HDF5Dataset::HDF5Dataset(
		std::string filename,
		std::size_t batchSize,
		std::string xField,
		std::string yField,
		std::uint32_t seed)
		: dataset_(PHP2XAIHDF5::open(filename)),
		  filename_(std::move(filename)),
		  batchSize_(batchSize),
		  xField_(std::move(xField)),
		  yField_(std::move(yField)),
		  sampleCount_(0),
		  xElementsPerSample_(0),
		  yElementsPerSample_(0),
		  rng_(seed)
	{
		if (batchSize_ == 0)
			throw std::runtime_error("batchSize must be > 0");

		xMetadata_ = dataset_->fieldMetadata(xField_);
		yMetadata_ = dataset_->fieldMetadata(yField_);
		xElementsPerSample_ = elementCount(xMetadata_.shape);
		yElementsPerSample_ = elementCount(yMetadata_.shape);
		sampleCount_ = dataset_->count();

		if (sampleCount_ < static_cast<std::int64_t>(batchSize_))
			throw std::runtime_error("HDF5 dataset has no complete batches");

		sampleOrder_.resize(static_cast<std::size_t>(sampleCount_));
		std::iota(sampleOrder_.begin(), sampleOrder_.end(), std::int64_t{0});
	}

	std::size_t HDF5Dataset::numBatches() const
	{
		return static_cast<std::size_t>(sampleCount_) / batchSize_;
	}

	std::int64_t HDF5Dataset::sampleCount() const
	{
		return sampleCount_;
	}

	std::size_t HDF5Dataset::batchSize() const
	{
		return batchSize_;
	}

	std::string HDF5Dataset::getType() const
	{
		return "HDF5";
	}

	void HDF5Dataset::shuffleEpoch()
	{
		std::shuffle(sampleOrder_.begin(), sampleOrder_.end(), rng_);
		resetEpoch();
	}

	void HDF5Dataset::resetEpoch()
	{
		batchPosition_ = 0;
		currentIndices_.clear();
	}

	bool HDF5Dataset::nextBatch()
	{
		if (!currentIndices_.empty())
			return true;
		if (batchPosition_ >= numBatches())
			return false;

		const std::size_t first = batchPosition_ * batchSize_;
		currentIndices_.assign(
			sampleOrder_.begin() + static_cast<std::ptrdiff_t>(first),
			sampleOrder_.begin() + static_cast<std::ptrdiff_t>(first + batchSize_));
		return true;
	}

	void HDF5Dataset::pack(std::vector<float>& xPacked, std::vector<float>& yPacked)
	{
		if (currentIndices_.empty())
			throw std::runtime_error("Call nextBatch() before pack()");

		readFieldAsFloat(xField_, xMetadata_.dtype, xElementsPerSample_, currentIndices_, xPacked);
		readFieldAsFloat(yField_, yMetadata_.dtype, yElementsPerSample_, currentIndices_, yPacked);

		++batchPosition_;
		currentIndices_.clear();
	}

	std::size_t HDF5Dataset::elementCount(const std::vector<std::int64_t>& shape)
	{
		if (shape.empty())
			throw std::runtime_error("HDF5 field shape cannot be empty");

		std::size_t count = 1;
		for (const std::int64_t dimension : shape)
		{
			if (dimension <= 0)
				throw std::runtime_error("HDF5 field dimensions must be positive");
			const auto size = static_cast<std::size_t>(dimension);
			if (count > std::numeric_limits<std::size_t>::max() / size)
				throw std::runtime_error("HDF5 field shape is too large");
			count *= size;
		}
		return count;
	}

	void HDF5Dataset::readFieldAsFloat(
		const std::string& field,
		PHP2XAIHDF5::DType dtype,
		std::size_t elementsPerSample,
		const std::vector<std::int64_t>& indices,
		std::vector<float>& output) const
	{
		if (indices.size() > std::numeric_limits<std::size_t>::max() / elementsPerSample)
			throw std::runtime_error("HDF5 output buffer is too large");
		const std::size_t valueCount = indices.size() * elementsPerSample;

		auto readAndConvert = [&](auto value) {
			using Value = decltype(value);
			std::vector<Value> values(valueCount);
			dataset_->readIndices(field, indices, values.data());
			output.resize(valueCount);
			std::transform(values.begin(), values.end(), output.begin(),
				[](Value item) { return static_cast<float>(item); });
		};

		switch (dtype)
		{
			case PHP2XAIHDF5::FLOAT32:
				readAndConvert(float{});
				break;
			case PHP2XAIHDF5::FLOAT64:
				readAndConvert(double{});
				break;
			case PHP2XAIHDF5::INT32:
				readAndConvert(std::int32_t{});
				break;
			case PHP2XAIHDF5::INT64:
				readAndConvert(std::int64_t{});
				break;
			default:
				throw std::runtime_error("Unsupported HDF5 dtype");
		}
	}
}
