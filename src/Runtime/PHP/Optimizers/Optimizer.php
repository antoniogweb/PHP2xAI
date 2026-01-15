<?php

namespace PHP2xAI\Runtime\PHP\Optimizers; 

use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Tensor\Matrix;
use PHP2xAI\Tensor\Vector;
use PHP2xAI\Tensor\Scalar;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

abstract class Optimizer
{
	// protected array $previousParameterValues = []; // contains the old values of the parameters if it has to step back
	// protected array $parameters = []; // all the params of the model $k => Node
	protected array $grads = []; // array of array of the gradients
	protected array $avgGrads = []; // contains the AVG of all the gradients
	protected float $error = 0; // the total AVG errors of the current STEP
	protected array $errors = []; // contains all the decreasing errors
	// protected array $allErrors = []; // contains all the errors
	protected ?float $gradClip = null; // max absolute value for gradient clipping
	protected int $errorCounter = 0;
	
	abstract public function step(GraphRuntime $graph);
	
	abstract public function getConfig() : array;
	
	// public function getErrorScalar()
	// {
	// 	return $this->error;
	// }
	
	// public function getError()
	// {
	// 	return $this->error / $this->errorCounter;
	// }
	
	public function setGradClip(?float $clip) : void
	{
		$this->gradClip = $clip;
	}
	
	// public function getAllErrors()
	// {
	// 	return $this->allErrors;
	// }
	
	// public function addTensors(array $tensors)
	// {
	// 	foreach ($tensors as $tensor)
	// 	{
	// 		$this->addTensor($tensor);
	// 	}
	// }
	
	// public function addTensor(Tensor $tensor)
	// {
	// 	if ($tensor instanceof Vector)
	// 		$this->parameters = array_merge($this->parameters, $tensor->a);
	// 	else if ($tensor instanceof Matrix)
	// 	{
	// 		for ($i=0; $i < $tensor->m; $i++)
	// 		{
	// 			$this->parameters = array_merge($this->parameters, $tensor->a[$i]);
	// 		}
	// 	}
	// 	else if ($tensor instanceof Scalar)
	// 		$this->parameters[] = $tensor->a;
	// }
	
// 	public function addError(float $error)
// 	{
// 		$this->error += $error;
// 		
// 		$this->errorCounter++;
// 	}
	
	// public function zeroGrads(GraphRuntime $graph)
	// {
	// 	// $this->error = 0;
	// 	// $this->errorCounter = 0;
	// 	$graph->resetGrad();
	// }
	
	// public function loadError()
	// {
	// 	$this->errors[] = $this->error;
	// }
	
	// public function getLastLoadedError()
	// {
	// 	return $this->errors[count($this->errors)-1] ?? 0;
	// }
	
// 	// Calculate the AVG gradients
// 	public function avgGrads() : bool
// 	{
// 		$n = count($this->grads);
// 		
// 		if ($n === 0)
// 			return false;
// 		
// 		$this->error = $this->error / $n;
// 		
// 		$parametersNumber = count($this->parameters);
// 		
// 		$this->avgGrads = array_fill(0, $parametersNumber, 0);
// 		
// 		foreach ($this->grads as $grads)
// 		{
// 			for ($i=0; $i < $parametersNumber; $i++)
// 			{
// 				$this->avgGrads[$i] += ($grads[$i] / $n);
// 			}
// 		}
// 		
// 		$this->allErrors[] = $this->error;
// 		
// 		return true;
// 	}
	
	protected function clipGradients() : void
	{
		if ($this->gradClip === null)
			return;
		
		$clip = $this->gradClip;
		
		foreach ($this->avgGrads as $i => $g)
		{
			if ($g > $clip)
				$this->avgGrads[$i] = $clip;
			else if ($g < -$clip)
				$this->avgGrads[$i] = -$clip;
		}
	}
	
// 	// Save the current parameters
// 	public function saveParameters()
// 	{
// 		$parametersNumber = count($this->parameters);
// 		
// 		for ($i=0; $i < $parametersNumber; $i++)
// 		{
// 			$this->previousParameterValues[$i] = $this->parameters[$i]->value;
// 		}
// 	}
// 	
// 	// Save errors to file
// 	public function saveToFile($logPath) : void
//     {
// 		$data = "#x\ty\n";
// 		
// 		$errors = $this->getAllErrors();
// 		
// 		foreach ($errors as $k => $error)
// 		{
// 			$data .= "$k\t$error\n";
// 		}
// 		
// 		file_put_contents($logPath,$data, LOCK_EX);
//     }
// 	
// 	// Save to the previous values
// 	public function stepBack()
// 	{
// 		$parametersNumber = count($this->parameters);
// 		
// 		for ($i=0; $i < $parametersNumber; $i++)
// 		{
// 			$this->parameters[$i]->value = $this->previousParameterValues[$i];
// 		}
// 	}
// 	
// 	// change the parameters
// 	public function stepScalar()
// 	{
// 		// Get the last error loaded in $this->errors
// 		$lastError = $this->getLastLoadedError();
// 		
// 		// Calculate the AVG gradients in $this->avgGrads and current error in $this->error
// 		if (!$this->avgGrads())
// 			return false;
// 		
// 		if ($lastError <= 0 || $this->error < $lastError)
// 		{
// 			// Save $this->error in $errors array
// 			$this->loadError();
// 			
// 			// Save the current parameters
// 			$this->saveParameters();
// 			
// 			return true;
// 		}
// // 		else
// // 			$this->stepBack();
// 		
// 		return true;
// 	}
}
