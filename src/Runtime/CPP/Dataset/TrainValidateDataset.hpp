#pragma once

#include "BatchDataset.hpp"

namespace PHP2xAI::Runtime::CPP
{
	class TrainValidateDataset
	{
	public:
		TrainValidateDataset(BatchDataset& trainDataset, BatchDataset& valDataset);

		BatchDataset& train;
		BatchDataset& val;
	};
}
