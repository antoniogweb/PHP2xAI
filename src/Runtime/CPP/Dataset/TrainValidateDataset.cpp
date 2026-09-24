#include "TrainValidateDataset.hpp"

namespace PHP2xAI::Runtime::CPP
{
	TrainValidateDataset::TrainValidateDataset(BatchDataset& trainDataset, BatchDataset& valDataset)
		: train(trainDataset), val(valDataset)
	{
	}
}
