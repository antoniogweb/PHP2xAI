<?php

namespace PHP2xAI\Runtime\CPP;

class GraphRuntimeCppEigen extends GraphRuntimeCpp
{
	protected function getProvider() : string
	{
		return "EIGEN";
	}
}
