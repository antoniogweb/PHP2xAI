#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <iostream>

#include "stream_file_dataset.hpp"
#include "../Core/Kernels/DTypeDispatch.hpp"

namespace PHP2xAI::Runtime::CPP
{
	StreamFileDataset::StreamFileDataset(std::string path,
										std::size_t batchSize,
										char delimiter,
										uint32_t seed)
		: path_(std::move(path)),
		batchSize_(batchSize),
		delimiter_(delimiter),
		rng_(seed)
	{
		if (batchSize_ == 0)
			throw std::runtime_error("batchSize must be > 0");

		file_.open(path_);
		if (!file_.is_open())
			throw std::runtime_error("Unable to open dataset file: " + path_);

		buildBatchOffsets_();
		resetOrder_();
			resetEpoch_();
		}

	std::size_t StreamFileDataset::numBatches() const { return batchOffsets_.size(); }
	std::string StreamFileDataset::getType() const { return "TXT"; }

	void StreamFileDataset::shuffleEpoch()
	{
		std::shuffle(batchOrder_.begin(), batchOrder_.end(), rng_);
		resetEpoch_();
	}

	void StreamFileDataset::resetEpoch()
	{
		resetEpoch_();
	}

	bool StreamFileDataset::nextBatch()
	{
		if (curBatchPos_ >= batchOrder_.size()) return false;

		curInBatch_ = 0;
		seekToBatchStart_(batchOrder_[curBatchPos_]);

		// Il batch corrente è pronto: ora chiamerai nextSampleInBatch()
		return true;
	}

	bool StreamFileDataset::nextSampleRaw(void* xVector, DType xDType, void* yVector, DType yDType)
	{
		bool found = false;
		dispatchDType(xDType, [&]<typename X>()
		{
			dispatchDType(yDType, [&]<typename Y>()
			{
				auto& x = *static_cast<std::vector<X>*>(xVector);
				auto& y = *static_cast<std::vector<Y>*>(yVector);
				x.clear();
				y.clear();

				while (true)
				{
					if (curInBatch_ >= batchSize_)
					{
						++curBatchPos_;
						return;
					}

					std::string line;
					if (!std::getline(file_, line))
					{
						++curBatchPos_;
						return;
					}

					if (isBlank_(line))
						continue;

					parseLineXY_(line, x, y);
					++curInBatch_;
					found = true;
					return;
				}
			});
		});
		return found;
	}

	void StreamFileDataset::packRaw(
		void* xVector, void* yVector)
	{
		auto& xPacked = *static_cast<std::vector<Scalar>*>(xVector);
		auto& yPacked = *static_cast<std::vector<Scalar>*>(yVector);
		xPacked.clear();
		yPacked.clear();
		std::vector<Scalar> x;
		std::vector<Scalar> y;
		while (true)
		{
			if (curInBatch_ >= batchSize_)
			{
				++curBatchPos_;
				break;
			}

			std::string line;
			if (!std::getline(file_, line))
			{
				++curBatchPos_;
				break;
			}

			if (isBlank_(line))
				continue;

			parseLineXY_(line, x, y);
			xPacked.insert(xPacked.end(), x.begin(), x.end());
			yPacked.insert(yPacked.end(), y.begin(), y.end());
			++curInBatch_;
		}
	}

	void StreamFileDataset::resetOrder_()
	{
		batchOrder_.resize(batchOffsets_.size());
		for (std::size_t i = 0; i < batchOrder_.size(); ++i) batchOrder_[i] = i;
	}

	void StreamFileDataset::resetEpoch_()
	{
		curBatchPos_ = 0;
		curInBatch_ = 0;
		file_.clear();
		file_.seekg(0);
	}

	void StreamFileDataset::buildBatchOffsets_()
	{
		batchOffsets_.clear();
		numLines_ = 0;

		file_.clear();
		file_.seekg(0);

		std::string line;
		while (true)
		{
			std::streampos pos = file_.tellg();
			if (!std::getline(file_, line)) break;

			if ((numLines_ % batchSize_) == 0)
			{
				batchOffsets_.push_back(pos);
			}

			++numLines_;
		}

		// Il runtime usa placeholder a batch fisso: escludi l'ultimo batch parziale.
		if ((numLines_ % batchSize_) != 0)
		{
			batchOffsets_.pop_back();
		}

		if (batchOffsets_.empty())
		{
			throw std::runtime_error("Dataset is empty: " + path_);
		}

		file_.clear();
		file_.seekg(0);
	}

	void StreamFileDataset::seekToBatchStart_(std::size_t batchId)
	{
		const std::streampos off = batchOffsets_.at(batchId);
		file_.clear();
		file_.seekg(off);
		if (!file_) throw std::runtime_error("seekg failed on dataset file");
	}

	bool StreamFileDataset::isBlank_(const std::string& s)
	{
		for (unsigned char c : s) if (!std::isspace(c)) return false;
		return true;
	}

	template <typename X, typename Y>
	void StreamFileDataset::parseLineXY_(const std::string& line, std::vector<X>& x, std::vector<Y>& y) const
	{
		const auto p = line.find(delimiter_);
		if (p == std::string::npos) {
			throw std::runtime_error("Invalid line (missing delimiter): " + line);
		}

		std::string_view left(line.data(), p);
		std::string_view right(line.data() + p + 1, line.size() - (p + 1));
		parseFloatVector_(left, x);
		parseFloatVector_(right, y);

		if (x.empty() || y.empty()) {
			throw std::runtime_error("Invalid line (empty x or y): " + line);
		}
	}

	template <typename T>
	void StreamFileDataset::parseFloatVector_(std::string_view sv, std::vector<T>& out)
	{
		out.clear();
		std::string s(sv);
		std::istringstream iss(s);
		std::string value;
		while (iss >> value)
			out.push_back(static_cast<T>(std::stold(value)));
	}
}
