<?php

namespace PHP2xAI\Runtime\PHP\Optimizers; 

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

class Fixed extends Optimizer
{
	protected float $learningRate = 0.1;
	
	public function __construct(float $learningRate = 0.1)
	{
		$this->learningRate = $learningRate;
	}
	
	public function getConfig() : array
	{
		return array(
			"name"		=>	"Fixed",
			"params"	=>	array(
				"learningRate"	=>	$this->learningRate,
			),
		);
	}
	
	public function step(GraphRuntime $graph)
	{
		
	}
	
	public function stepScalar()
	{
		if (parent::stepScalar())
		{
			$parametersNumber = count($this->parameters);
			
			for ($i=0; $i < $parametersNumber; $i++)
			{
				$this->parameters[$i]->value -= $this->avgGrads[$i] * $this->learningRate;
			}
			
			return true;
		}
		
		return false;
	}
}
