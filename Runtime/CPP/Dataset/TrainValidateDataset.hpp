#pragma once

#include "stream_file_dataset.hpp"

namespace PHP2xAI::Runtime::CPP
{
	class TrainValidateDataset
	{
	public:
		TrainValidateDataset(StreamFileDataset& trainDataset, StreamFileDataset& valDataset);

		StreamFileDataset& train;
		StreamFileDataset& val;
	};
}
