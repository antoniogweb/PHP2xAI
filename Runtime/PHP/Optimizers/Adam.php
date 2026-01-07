<?php

namespace PHP2xAI\Runtime\PHP\Optimizers; 

use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

class Adam extends Optimizer
{
	protected float $learningRate = 0.1;
	protected float $beta1 = 0.9;
	protected float $beta2 = 0.999;
	protected float $eps = 0.00000001;
	
	protected array $m = [];
	protected array $v = [];
	
	protected array $mp = []; // array of previous $m
	protected array $vp = []; // array of previuss $v
	
	protected $stepNumber = 1;
	
	public function __construct(float $learningRate = 0.1, float $beta1 = 0.9, float $beta2 = 0.999, float $eps = 0.00000001)
	{
		$this->learningRate = $learningRate;
		$this->beta1 = $beta1;
		$this->beta2 = $beta2;
		$this->eps = $eps;
	}
	
	public function getConfig() : array
	{
		return array(
			"name"		=>	"Adam",
			"params"	=>	array(
				"learningRate"	=>	$this->learningRate,
				"beta1"			=>	$this->beta1,
				"beta2"			=>	$this->beta2,
				"eps"			=>	$this->eps,
			),
		);
	}
	
	public function step(GraphRuntime $graph)
	{
		$n = max(1, $graph->accSteps);
		// echo $graph->accSteps."\n";
		$beta1PowT = pow($this->beta1, $this->stepNumber);
		$beta2PowT = pow($this->beta2, $this->stepNumber);
		
		foreach ($graph->trainable as $tid)
		{
			$t = $graph->tensors[$tid];
			
			$size = count($t->data);
			
			for ($i = 0; $i < $size; $i++)
			{
				$g = $t->grad[$i] / $n;
				
				if ($this->gradClip !== null)
				{
					$clip = $this->gradClip;
					if ($g > $clip) $g = $clip;
					else if ($g < -$clip) $g = -$clip;
				}
				
				$mtp = $this->mp[$tid][$i] ?? 0;
				$vtp = $this->vp[$tid][$i] ?? 0;
				
				$mt = $this->beta1 * $mtp + (1 - $this->beta1) * $g;
				$vt = $this->beta2 * $vtp + (1 - $this->beta2) * ($g * $g);
				
				$this->mp[$tid][$i] = $mt;
				$this->vp[$tid][$i] = $vt;
				
				$mt_ = $mt / (1 - $beta1PowT);
				$vt_ = $vt / (1 - $beta2PowT);
				
				$t->data[$i] -= $this->learningRate * ($mt_ / (sqrt($vt_) + $this->eps));
			}
		}
		
		$this->stepNumber++;
	}
}
