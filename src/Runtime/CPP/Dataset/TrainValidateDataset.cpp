#include "TrainValidateDataset.hpp"

namespace PHP2xAI::Runtime::CPP
{
	TrainValidateDataset::TrainValidateDataset(StreamFileDataset& trainDataset, StreamFileDataset& valDataset)
		: train(trainDataset), val(valDataset)
	{
	}
}
