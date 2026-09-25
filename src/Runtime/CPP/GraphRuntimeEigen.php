<?php

namespace PHP2xAI\Runtime\CPP;

class GraphRuntimeEigen extends GraphRuntimeCpp
{
	protected function getProvider() : string
	{
		return "EIGEN";
	}
}
