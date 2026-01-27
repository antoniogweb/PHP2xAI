<?php

namespace PHP2xAI\Runtime\PHP\Core;

use RuntimeException;
use PHP2xAI\Tensor\Tensor;

class GraphRuntime
{
	/** @var TensorRuntime[] */
	public array $tensors = [];
	/** @var array[] */
	public array $ops;
	public int $lossId = 0;
	/** @var int[] */
	public array $trainable;
	
	public int $inputId = 0;
	public int $targetId = 0;
	public int $outputId = 0;
	
	private array $graphDef = [];
	
	public function __construct(array $graphDef, ?array $weigths = null)
	{
		$this->graphDef = $graphDef;
		
		// crea i tensori
		foreach ($graphDef['tensors'] as $t)
		{
			$id = $t['id'];
			
			$this->tensors[$id] = new TensorRuntime(
				$t['shape'],
				$t['kind'],
				$t['name'] ?? null
			);
			
			// Initialize with provided data (e.g., parameters) when available
			if (isset($t['data']))
			{
				$data = array_values($t['data']);
				$this->tensors[$id]->data = $data;
				$this->tensors[$id]->grad = array_fill(0, count($data), 1.0);
			}
			
			if ($t["kind"] == "input")
				$this->inputId = $id;
			
			if ($t["kind"] == "target")
				$this->targetId = $id;
			
			if (isset($weigths) && $t["kind"] == "param" && isset($weigths["tensors"][$id]) && $weigths["tensors"][$id]["shape"] == $t['shape'])
			{
				$this->tensors[$id]->data = $weigths["tensors"][$id]["data"];
			}
		}

		$this->ops = $graphDef['ops'];
		
		if (isset($graphDef['loss']))
			$this->lossId = $graphDef['loss'];
		
		if (isset($graphDef['output']))
			$this->outputId = $graphDef['output'];
		
		if (isset($graphDef['trainable']))
			$this->trainable = $graphDef['trainable'];
	}
	
	public static function create(Tensor $tensor) : GraphRuntime
	{
		$context = $tensor->context;
		
		$graphRuntime = new GraphRuntime($context->export());
		
		return $graphRuntime;
	}
	
	public function saveWeightsToJson(string $path)
	{
		$tensors = [];
		
		foreach ($this->tensors as $id => $t)
		{
			if (in_array($id, $this->trainable))
				$tensors[$id] = array(
					"data"	=>	$t->data,
					"shape"	=>	$t->shape,
				);
		}
		
		$jsonArray = array(
			"tensors"	=>	$tensors,
		);
		
		file_put_contents($path, json_encode($jsonArray), LOCK_EX);
	}
	
	public function saveToJson(string $path)
	{
		$tensors = [];
		
		foreach ($this->tensors as $id => $t)
		{
			$tensors[$id] = $t->data;
		}
		
		$jsonArray = array(
			"graph"		=>	$this->graphDef,
			"tensors"	=>	$tensors,
		);
		
		file_put_contents($path, json_encode($jsonArray), LOCK_EX);
	}
	
	public function setLossGrad(float $lossGrad = 1.0)
	{
		if ($this->lossId && isset($this->tensors[$this->lossId]))
		{
			$tensor = $this->tensors[$this->lossId];
		
			$tensor->grad = array_fill(0, count($tensor->data), $lossGrad);
		}
	}
	
	public function getLoss() : array
	{
		$tensor = $this->tensors[$this->lossId];
		
		return $tensor->data;
	}
	
	public function getError() : float
	{
		$loss = $this->getLoss();
		
		if (count($loss) > 1)
			return array_sum($loss)/count($loss);
		else
			return $loss[0];
	}
	
	public function getOutput()
	{
		$tensor = $this->tensors[$this->outputId];
		
		return $tensor->data;
	}
	
	public function setInput(array $x): void
	{
		$tensor = $this->tensors[$this->inputId];
		
		if (count($tensor->data) !== count($x)) {
			throw new RuntimeException("Inserting incompatible dimensions");
		}
		
		// copia i valori nel tensore di runtime
		$tensor->data = array_map('floatval', array_values($x));
	}

	public function setTarget(array $y): void
	{
		$tensor = $this->tensors[$this->targetId];
		
		if (count($tensor->data) !== count($y)) {
			throw new RuntimeException("Inserting incompatible dimensions");
		}
		
		$tensor->data = array_map('floatval', array_values($y));
	}
	
	public function resetGrad(): void
	{
		foreach ($this->tensors as $tensor)
		{
			$tensor->grad = array_fill(0, count($tensor->grad), 0.0);
		}
	}
	
	public function forward(): void
	{
		foreach ($this->ops as $op)
		{
			$name = $op['op'];
			$inputs = $op['inputs'];
			$outId = $op['output'];

			switch ($name)
			{
				case 'matmul':
					$this->opMatmul($inputs[0], $inputs[1], $outId);
					break;
				case 'add':
					$this->opAdd($inputs[0], $inputs[1], $outId);
					break;
				case 'sub':
					$this->opSub($inputs[0], $inputs[1], $outId);
					break;
				case 'dot':
					$this->opDot($inputs[0], $inputs[1], $outId);
					break;
				case 'dropout':
					$this->opDropout($inputs[0], $outId);
					break;
				case 'sig':
					$this->opSig($inputs[0], $outId);
					break;
				case 'ReLU':
				case 'relu':
					$this->opRelu($inputs[0], $outId);
					break;
				case 'LReLU':
					$this->opLRelu($inputs[0], $outId);
					break;
				case 'MSE':
					$this->opMse($inputs[0], $outId);
					break;
				case 'MAE':
					$this->opMae($inputs[0], $outId);
					break;
				case 'softmax':
					$this->opSoftmax($inputs[0], $outId);
					break;
				case 'CE':
					$this->opCe($inputs[0], $inputs[1], $outId);
					break;
				case 'softmax_ce_logits':
					$this->opCeLogits($inputs[0], $inputs[1], $outId);
					break;
				case 'softmax_ce_logits_label_int':
					$this->opCeLogitsLabelInt($inputs[0], $inputs[1], $outId);
					break;
				case 'mean':
					$this->opMean($inputs[0], $outId);
					break;
				default:
					throw new RuntimeException("Op not supported: {$name}");
			}
		}
	}
	
	private function opMatmul(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		if (count($A->shape) !== 2)
			throw new RuntimeException('matmul: left operand must be a matrix');
		
		// N: hidden layer dimension
		// D: number of elements of input tensor
		// B: number of samples in batch
		
		// shapes: A[N, D] * B[D] = C[N]
		if (count($B->shape) == 1)
		{
			[$m, $n] = $A->shape;
			
			if ($B->shape[0] !== $n)
				throw new RuntimeException('matmul: dimension mismatch');

			// A[m, n] * B[n] => C[m]
			$C->shape = [$m];
			$C->data  = array_fill(0, $m, 0.0);
			for ($i = 0; $i < $m; $i++)
			{
				$sum = 0.0;
				
				for ($k = 0; $k < $n; $k++)
				{
					$sum += $A->data[$i * $n + $k] * $B->data[$k];
				}
				
				$C->data[$i] = $sum;
			}
		}
		else if (count($B->shape) == 2) // A[B, D] * B[D, N] = C[B, N]
		{
			[$batch, $dim] = $A->shape;
			[$dimB, $outDim] = $B->shape;
			
			if ($dim !== $dimB)
				throw new RuntimeException('matmul: dimension mismatch');
			
			// A[B, D] * B[D, N] = C[B, N]
			$C->shape = [$batch, $outDim];
			$C->data = array_fill(0, $batch * $outDim, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				$cRow = $b * $outDim;

				for ($d = 0; $d < $dim; $d++)
				{
					$aVal = $A->data[$aRow + $d];
					$bRow = $d * $outDim;

					for ($n = 0; $n < $outDim; $n++)
					{
						$C->data[$cRow + $n] += $aVal * $B->data[$bRow + $n];
					}
				}
			}
		}
		else
		{
			// per ora puoi gestire solo matrice * vettore
			throw new RuntimeException("matmul: caso non implementato");
		}
	}
	
	private function opAdd(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		// broadcast support: A[B, N] + B[N] = C[B, N]
		if (count($A->shape) === 2 && count($B->shape) === 1)
		{
			[$batch, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('add: dimension mismatch');
			
			$C->shape = [$batch, $dim];
			$C->data  = array_fill(0, $batch * $dim, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				
				for ($n = 0; $n < $dim; $n++)
				{
					$C->data[$aRow + $n] = $A->data[$aRow + $n] + $B->data[$n];
				}
			}
			
			return;
		}

		$size = count($A->data);
		
		if ($size !== count($B->data))
			throw new RuntimeException('add: dimension mismatch');

		$C->shape = $A->shape;
		$C->data  = array_fill(0, $size, 0.0);

		for ($i = 0; $i < $size; $i++)
		{
			$C->data[$i] = $A->data[$i] + $B->data[$i];
		}
	}

	private function opSub(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		// broadcast support: A[B, N] - B[N] = C[B, N]
		if (count($A->shape) === 2 && count($B->shape) === 1)
		{
			[$batch, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('sub: dimension mismatch');
			
			$C->shape = [$batch, $dim];
			$C->data  = array_fill(0, $batch * $dim, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				
				for ($n = 0; $n < $dim; $n++)
				{
					$C->data[$aRow + $n] = $A->data[$aRow + $n] - $B->data[$n];
				}
			}
			
			return;
		}

		$size = count($A->data);
		
		if ($size !== count($B->data))
			throw new RuntimeException('sub: dimension mismatch');

		$C->shape = $A->shape;
		$C->data  = array_fill(0, $size, 0.0);

		for ($i = 0; $i < $size; $i++)
		{
			$C->data[$i] = $A->data[$i] - $B->data[$i];
		}
	}

	private function opDot(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		$size = count($A->data);
		
		if ($size !== count($B->data))
			throw new RuntimeException('dot: dimension mismatch');

		$sum = 0.0;

		for ($i = 0; $i < $size; $i++)
		{
			$sum += $A->data[$i] * $B->data[$i];
		}

		$C->shape = [];
		$C->data = [$sum];
	}
	
	private function opRelu(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$Y->shape = $X->shape;
		$size = count($X->data);
		$Y->data = array_fill(0, $size, 0.0);
		
		for ($i = 0; $i < $size; $i++)
		{
			$Y->data[$i] = $X->data[$i] > 0.0 ? $X->data[$i] : 0.0;
		}
	}

	private function opLRelu(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$alpha = 0.01;
		$Y->shape = $X->shape;
		$size = count($X->data);
		$Y->data = array_fill(0, $size, 0.0);
		
		for ($i = 0; $i < $size; $i++)
		{
			$val = $X->data[$i];
			$Y->data[$i] = $val > 0.0 ? $val : $alpha * $val;
		}
	}

	private function opSig(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$Y->shape = $X->shape;
		$size = count($X->data);
		$Y->data = array_fill(0, $size, 0.0);
		
		for ($i = 0; $i < $size; $i++)
		{
			$Y->data[$i] = 1 / (1 + \exp(-1 * $X->data[$i]));
		}
	}

	private function opDropout(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$Y->shape = $X->shape;
		$size = count($X->data);
		$Y->data = array_fill(0, $size, 0.0);

		$dropPerc = 50;
		$dropPerc = max(0, min(100, $dropPerc));
		$keepProb = 1 - ($dropPerc / 100);
		$scale = $keepProb > 0 ? 1 / $keepProb : 0.0;
		
		for ($i = 0; $i < $size; $i++)
		{
			$keep = mt_rand(1, 100) > $dropPerc;
			$mask = $keep ? $scale : 0.0;
			$Y->data[$i] = $X->data[$i] * $mask;
		}
	}

	private function opMse(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		if ($size === 0)
		{
			$Y->shape = [];
			$Y->data = [0.0];
			return;
		}

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$Y->shape = [];
			$Y->data = [0.5 * $val * $val];
			return;
		}

		if (count($X->shape) === 2)
		{
			[$batch, $dim] = $X->shape;
			$Y->shape = [$batch];
			$Y->data = array_fill(0, $batch, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$sum = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$val = $X->data[$rowStart + $i];
					$sum += $val * $val;
				}
				
				$Y->data[$b] = $dim > 0 ? $sum / $dim : 0.0;
			}
			
			return;
		}

		$Y->shape = [];
		$sum = 0.0;
		
		for ($i = 0; $i < $size; $i++)
		{
			$sum += $X->data[$i] * $X->data[$i];
		}
		
		$Y->data = [$sum / $size];
	}

	private function opMae(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		if ($size === 0)
		{
			$Y->shape = [];
			$Y->data = [0.0];
			return;
		}

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$Y->shape = [];
			$Y->data = [0.5 * \abs($val)];
			return;
		}

		if (count($X->shape) === 2)
		{
			[$batch, $dim] = $X->shape;
			$Y->shape = [$batch];
			$Y->data = array_fill(0, $batch, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$sum = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$sum += \abs($X->data[$rowStart + $i]);
				}
				
				$Y->data[$b] = $dim > 0 ? $sum / $dim : 0.0;
			}
			
			return;
		}

		$Y->shape = [];
		$sum = 0.0;
		
		for ($i = 0; $i < $size; $i++)
		{
			$sum += \abs($X->data[$i]);
		}
		
		$Y->data = [$sum / $size];
	}

	private function opSoftmax(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$Y->shape = $X->shape;
		$size = count($X->data);

		if ($size === 0)
		{
			$Y->data = [];
			return;
		}

		if (count($X->shape) === 2)
		{
			[$batch, $dim] = $X->shape;
			$Y->data = array_fill(0, $batch * $dim, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$max = $X->data[$rowStart];
				
				for ($i = 1; $i < $dim; $i++)
				{
					$val = $X->data[$rowStart + $i];
					if ($val > $max)
						$max = $val;
				}
				
				$sum = 0.0;
				$expValues = [];
				
				for ($i = 0; $i < $dim; $i++)
				{
					$expValues[$i] = \exp($X->data[$rowStart + $i] - $max);
					$sum += $expValues[$i];
				}
				
				$invSum = $sum === 0.0 ? 0.0 : 1 / $sum;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$Y->data[$rowStart + $i] = $expValues[$i] * $invSum;
				}
			}
			
			return;
		}

		$max = $X->data[0];
		
		for ($i = 1; $i < $size; $i++)
		{
			if ($X->data[$i] > $max)
				$max = $X->data[$i];
		}
		
		$expValues = [];
		$sum = 0.0;
		
		for ($i = 0; $i < $size; $i++)
		{
			$expValues[$i] = \exp($X->data[$i] - $max);
			$sum += $expValues[$i];
		}
		
		$invSum = $sum === 0.0 ? 0.0 : 1 / $sum;
		$Y->data = array_fill(0, $size, 0.0);
		
		for ($i = 0; $i < $size; $i++)
		{
			$Y->data[$i] = $expValues[$i] * $invSum;
		}
	}

	private function opCe(int $predId, int $targetId, int $outId): void
	{
		$pred = $this->tensors[$predId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$classes = count($pred->data);

		if ($classes === 0 || $classes !== count($target->data))
		{
			$out->shape = [];
			$out->data = [0.0];
			return;
		}

		if (count($pred->shape) === 2 && count($target->shape) === 2)
		{
			[$batch, $dim] = $pred->shape;
			
			if ($target->shape[0] !== $batch || $target->shape[1] !== $dim)
				throw new RuntimeException('CE: dimension mismatch');
			
			$out->shape = [$batch];
			$out->data = array_fill(0, $batch, 0.0);
			$eps = 1.0e-12;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$activeIndex = null;
				$isOneHot = true;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$val = $target->data[$rowStart + $i];
					
					if ($val > 0.5)
					{
						if ($activeIndex !== null)
						{
							$isOneHot = false;
							break;
						}
						
						$activeIndex = $i;
					}
					else if (\abs($val) > 1.0e-9)
					{
						$isOneHot = false;
						break;
					}
				}
				
				if ($isOneHot && $activeIndex !== null)
				{
					$prob = $pred->data[$rowStart + $activeIndex] ?? 0.0;
					$out->data[$b] = -\log($prob + $eps);
					continue;
				}
				
				$loss = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$loss += $target->data[$rowStart + $i] * \log(($pred->data[$rowStart + $i] ?? 0.0) + $eps);
				}
				
				$out->data[$b] = -$loss;
			}
			
			return;
		}

		$out->shape = [];
		$activeIndex = null;
		$isOneHot = true;

		for ($i = 0; $i < $classes; $i++)
		{
			$val = $target->data[$i];
			
			if ($val > 0.5)
			{
				if ($activeIndex !== null)
				{
					$isOneHot = false;
					break;
				}
				
				$activeIndex = $i;
			}
			else if (\abs($val) > 1.0e-9)
			{
				$isOneHot = false;
				break;
			}
		}

		$eps = 1.0e-12;

		if ($isOneHot && $activeIndex !== null)
		{
			$prob = $pred->data[$activeIndex] ?? 0.0;
			$out->data = [-\log($prob + $eps)];
			return;
		}

		$loss = 0.0;

		for ($i = 0; $i < $classes; $i++)
		{
			$loss += $target->data[$i] * \log(($pred->data[$i] ?? 0.0) + $eps);
		}

		$out->data = [-$loss];
	}

	private function opCeLogits(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$classes = count($logits->data);

		if ($classes === 0 || $classes !== count($target->data))
		{
			$out->shape = [];
			$out->data = [0.0];
			return;
		}

		if (count($logits->shape) === 2 && count($target->shape) === 2)
		{
			[$batch, $dim] = $logits->shape;
			
			if ($target->shape[0] !== $batch || $target->shape[1] !== $dim)
				throw new RuntimeException('CE logits: dimension mismatch');
			
			$out->shape = [$batch];
			$out->data = array_fill(0, $batch, 0.0);
			$eps = 1.0e-12;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$max = $logits->data[$rowStart];
				
				for ($i = 1; $i < $dim; $i++)
				{
					$val = $logits->data[$rowStart + $i];
					if ($val > $max)
						$max = $val;
				}
				
				$probs = [];
				$sumExp = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$expVal = \exp($logits->data[$rowStart + $i] - $max);
					$probs[$i] = $expVal;
					$sumExp += $expVal;
				}
				
				$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$probs[$i] *= $invSum;
				}
				
				$loss = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$t = $target->data[$rowStart + $i];
					
					if ($t > 0.0)
						$loss += -$t * \log($probs[$i] + $eps);
				}
				
				$out->data[$b] = $loss;
			}
			
			return;
		}

		$out->shape = [];
		$max = $logits->data[0];
		
		for ($i = 1; $i < $classes; $i++)
		{
			if ($logits->data[$i] > $max)
				$max = $logits->data[$i];
		}
		
		$probs = [];
		$sumExp = 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$expVal = \exp($logits->data[$i] - $max);
			$probs[$i] = $expVal;
			$sumExp += $expVal;
		}
		
		$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$probs[$i] *= $invSum;
		}
		
		$loss = 0.0;
		$eps = 1.0e-12;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$t = $target->data[$i];
			
			if ($t > 0.0)
				$loss += -$t * \log($probs[$i] + $eps);
		}
		
		$out->data = [$loss];
	}
	
	private function opCeLogitsLabelInt(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$classes = count($logits->data);

		if ($classes === 0)
		{
			$out->shape = [];
			$out->data = [0.0];
			return;
		}

		if (count($logits->shape) === 2)
		{
			[$batch, $dim] = $logits->shape;
			
			if (count($target->shape) !== 1 || $target->shape[0] !== $batch)
				throw new RuntimeException('CE logits label int: dimension mismatch');
			
			$out->shape = [$batch];
			$out->data = array_fill(0, $batch, 0.0);
			$eps = 1.0e-12;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$labelInt = $target->data[$b];
				$max = $logits->data[$rowStart];
				
				for ($i = 1; $i < $dim; $i++)
				{
					$val = $logits->data[$rowStart + $i];
					if ($val > $max)
						$max = $val;
				}
				
				$probs = [];
				$sumExp = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$expVal = \exp($logits->data[$rowStart + $i] - $max);
					$probs[$i] = $expVal;
					$sumExp += $expVal;
				}
				
				$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$probs[$i] *= $invSum;
				}
				
				$loss = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					if ((int)$i === (int)$labelInt)
						$loss += -1 * \log($probs[$i] + $eps);
				}
				
				$out->data[$b] = $loss;
			}
			
			return;
		}

		$out->shape = [];
		$max = $logits->data[0];
		$labelInt = $target->data[0];
		
		for ($i = 1; $i < $classes; $i++)
		{
			if ($logits->data[$i] > $max)
				$max = $logits->data[$i];
		}
		
		$probs = [];
		$sumExp = 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$expVal = \exp($logits->data[$i] - $max);
			$probs[$i] = $expVal;
			$sumExp += $expVal;
		}
		
		$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$probs[$i] *= $invSum;
		}
		
		$loss = 0.0;
		$eps = 1.0e-12;
		
		for ($i = 0; $i < $classes; $i++)
		{
			if ((int)$i === (int)$labelInt)
				$loss += -1 * \log($probs[$i] + $eps);
		}
		
		$out->data = [$loss];
	}
	
	private function opMean(int $aId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$out = $this->tensors[$outId];
		
		$out->shape = [];
		
		if (count($A->shape) !== 1 || count($A->data) === 0)
			throw new RuntimeException('Mean: dimension mismatch');
		
		$mean = array_sum($A->data)/count($A->data);
		
		$out->data = [$mean];
	}
	
	public function backward(): void
	{
		// clear grads of non-parameter tensors so intermediate gradients don't snowball across samples
		foreach ($this->tensors as $tensor)
		{
			if ($tensor->kind !== 'param')
				$tensor->grad = array_fill(0, count($tensor->grad), 0.0);
		}

		// grad loss = 1
		$this->setLossGrad(1.0);

		// reverse sulle op
		for ($i = count($this->ops) - 1; $i >= 0; $i--)
		{
			$op = $this->ops[$i];
			$name   = $op['op'];
			$inputs = $op['inputs'];
			$outId  = $op['output'];

			switch ($name)
			{
				case 'matmul':
					$this->backwardMatmul($inputs[0], $inputs[1], $outId);
					break;
				case 'add':
					$this->backwardAdd($inputs[0], $inputs[1], $outId);
					break;
				case 'sub':
					$this->backwardSub($inputs[0], $inputs[1], $outId);
					break;
				case 'dot':
					$this->backwardDot($inputs[0], $inputs[1], $outId);
					break;
				case 'dropout':
					$this->backwardDropout($inputs[0], $outId);
					break;
				case 'sig':
					$this->backwardSig($inputs[0], $outId);
					break;
				case 'relu':
				case 'ReLU':
					$this->backwardRelu($inputs[0], $outId);
					break;
				case 'LReLU':
					$this->backwardLRelu($inputs[0], $outId);
					break;
				case 'MSE':
					$this->backwardMse($inputs[0], $outId);
					break;
				case 'MAE':
					$this->backwardMae($inputs[0], $outId);
					break;
				case 'softmax':
					$this->backwardSoftmax($inputs[0], $outId);
					break;
				case 'CE':
					$this->backwardCe($inputs[0], $inputs[1], $outId);
					break;
				case 'softmax_ce_logits':
					$this->backwardCeLogits($inputs[0], $inputs[1], $outId);
					break;
				case 'softmax_ce_logits_label_int':
					$this->backwardCeLogitsLabelInt($inputs[0], $inputs[1], $outId);
					break;
				case 'mean':
					$this->backwardMean($inputs[0], $outId);
					break;
				default:
					throw new RuntimeException("Op not supported: {$name}");
			}
		}
	}
	
	private function backwardMean(int $aId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$out = $this->tensors[$outId];
		$size = count($A->data);

		if ($size === 0)
			return;

		if (count($A->shape) !== 1)
			throw new RuntimeException('Mean backward: dimension mismatch');

		$gradOut = $out->grad[0] ?? 0.0;
		$scale = $gradOut / $size;

		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $scale;
		}
	}
	
	private function backwardMatmul(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		// caso A[m, n] * B[n] = C[m]
		if (count($B->shape) == 1)
		{
			[$m, $n] = $A->shape;
			
			for ($i = 0; $i < $m; $i++)
			{
				$gradC = $C->grad[$i];
				
				for ($k = 0; $k < $n; $k++)
				{
					$aIdx = $i * $n + $k;
					// dC[i]/dA[i,k] = B[k]
					$A->grad[$aIdx] += $gradC * $B->data[$k];
					// dC[i]/dB[k]   = A[i,k]
					$B->grad[$k]    += $gradC * $A->data[$aIdx];
				}
			}
			
			return;
		}

		// caso A[B, D] * B[D, N] = C[B, N]
		if (count($B->shape) == 2)
		{
			[$batch, $dim] = $A->shape;
			[$dimB, $outDim] = $B->shape;
			
			if ($dim !== $dimB)
				throw new RuntimeException('matmul: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				$cRow = $b * $outDim;
				
				for ($d = 0; $d < $dim; $d++)
				{
					$aVal = $A->data[$aRow + $d];
					$bRow = $d * $outDim;
					
					for ($n = 0; $n < $outDim; $n++)
					{
						$gradC = $C->grad[$cRow + $n];
						$A->grad[$aRow + $d] += $gradC * $B->data[$bRow + $n];
						$B->grad[$bRow + $n] += $aVal * $gradC;
					}
				}
			}
			
			return;
		}

		throw new RuntimeException("matmul backward: caso non implementato");
	}
	
	private function backwardAdd(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		if (count($A->shape) === 2 && count($B->shape) === 1)
		{
			[$batch, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('add: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				
				for ($n = 0; $n < $dim; $n++)
				{
					$grad = $C->grad[$aRow + $n];
					$A->grad[$aRow + $n] += $grad;
					$B->grad[$n] += $grad;
				}
			}
			
			return;
		}

		$size = count($C->data);
		
		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $C->grad[$i];
			$B->grad[$i] += $C->grad[$i];
		}
	}

	private function backwardSub(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		if (count($A->shape) === 2 && count($B->shape) === 1)
		{
			[$batch, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('sub: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$aRow = $b * $dim;
				
				for ($n = 0; $n < $dim; $n++)
				{
					$grad = $C->grad[$aRow + $n];
					$A->grad[$aRow + $n] += $grad;
					$B->grad[$n] -= $grad;
				}
			}
			
			return;
		}

		$size = count($C->data);
		
		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $C->grad[$i];
			$B->grad[$i] -= $C->grad[$i];
		}
	}

	private function backwardDot(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		$gradOut = $C->grad[0] ?? 0.0;
		$size = count($A->data);

		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $gradOut * $B->data[$i];
			$B->grad[$i] += $gradOut * $A->data[$i];
		}
	}
	
	private function backwardRelu(int $inpId, int $outId): void
    {
        $X = $this->tensors[$inpId];
        $Y = $this->tensors[$outId];
        $size = count($X->data);
		
        for ($i = 0; $i < $size; $i++)
		{
            $local = $X->data[$i] > 0.0 ? 1.0 : 0.0;
            $X->grad[$i] += $Y->grad[$i] * $local;
        }
    }

	private function backwardLRelu(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$alpha = 0.01;
		$size = count($X->data);

		for ($i = 0; $i < $size; $i++)
		{
			$local = $X->data[$i] > 0.0 ? 1.0 : $alpha;
			$X->grad[$i] += $Y->grad[$i] * $local;
		}
	}

	private function backwardSig(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		for ($i = 0; $i < $size; $i++)
		{
			$y = $Y->data[$i];
			$local = $y * (1 - $y);
			$X->grad[$i] += $Y->grad[$i] * $local;
		}
	}

	private function backwardDropout(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		for ($i = 0; $i < $size; $i++)
		{
			$x = $X->data[$i];
			$y = $Y->data[$i];
			$mask = ($x != 0.0) ? ($y / $x) : ($y == 0.0 ? 0.0 : 1.0);
			$X->grad[$i] += $Y->grad[$i] * $mask;
		}
	}

	private function backwardMse(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		if ($size === 0)
			return;

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$gradOut = $Y->grad[0] ?? 0.0;
			$X->grad[0] += $gradOut * $val;
			return;
		}

		if (count($X->shape) === 2)
		{
			[$batch, $dim] = $X->shape;

			if (count($Y->shape) !== 1 || $Y->shape[0] !== $batch)
				throw new RuntimeException('MSE backward: output shape mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$gradOut = $Y->grad[$b] ?? 0.0;
				$scale = $dim > 0 ? (2 / $dim) * $gradOut : 0.0;
				$rowStart = $b * $dim;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$X->grad[$rowStart + $i] += $scale * $X->data[$rowStart + $i];
				}
			}
			
			return;
		}

		$gradOut = $Y->grad[0] ?? 0.0;
		$scale = (2 / $size) * $gradOut;

		for ($i = 0; $i < $size; $i++)
		{
			$X->grad[$i] += $scale * $X->data[$i];
		}
	}

	private function backwardMae(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($X->data);

		if ($size === 0)
			return;

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$gradOut = $Y->grad[0] ?? 0.0;
			$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
			$X->grad[0] += $gradOut * 0.5 * $sign;
			return;
		}

		if (count($X->shape) === 2)
		{
			[$batch, $dim] = $X->shape;

			if (count($Y->shape) !== 1 || $Y->shape[0] !== $batch)
				throw new RuntimeException('MAE backward: output shape mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$gradOut = $Y->grad[$b] ?? 0.0;
				$scale = $dim > 0 ? (1 / $dim) * $gradOut : 0.0;
				$rowStart = $b * $dim;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$val = $X->data[$rowStart + $i];
					$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
					$X->grad[$rowStart + $i] += $scale * $sign;
				}
			}
			
			return;
		}

		$gradOut = $Y->grad[0] ?? 0.0;
		$scale = ($size > 0) ? (1 / $size) * $gradOut : 0.0;

		for ($i = 0; $i < $size; $i++)
		{
			$val = $X->data[$i];
			$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
			$X->grad[$i] += $scale * $sign;
		}
	}

	private function backwardSoftmax(int $inpId, int $outId): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($Y->data);

		if (count($Y->shape) === 2)
		{
			[$batch, $dim] = $Y->shape;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$grad = 0.0;
					$yi = $Y->data[$rowStart + $i];
					
					for ($j = 0; $j < $dim; $j++)
					{
						$delta = ($i === $j) ? 1.0 : 0.0;
						$yj = $Y->data[$rowStart + $j];
						$jac = $yj * ($delta - $yi);
						$grad += $Y->grad[$rowStart + $j] * $jac;
					}
					
					$X->grad[$rowStart + $i] += $grad;
				}
			}
			
			return;
		}

		// For each input dimension: dL/dx_i = sum_j dL/dy_j * dy_j/dx_i
		for ($i = 0; $i < $size; $i++)
		{
			$grad = 0.0;

			for ($j = 0; $j < $size; $j++)
			{
				$delta = ($i === $j) ? 1.0 : 0.0;
				$jac = $Y->data[$j] * ($delta - $Y->data[$i]);
				$grad += $Y->grad[$j] * $jac;
			}

			$X->grad[$i] += $grad;
		}
	}

	private function backwardCe(int $predId, int $targetId, int $outId): void
	{
		$pred = $this->tensors[$predId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];

		$classes = count($pred->data);
		if ($classes === 0 || $classes !== count($target->data))
			return;

		if (count($pred->shape) === 2 && count($target->shape) === 2)
		{
			[$batch, $dim] = $pred->shape;
			
			if ($target->shape[0] !== $batch || $target->shape[1] !== $dim)
				throw new RuntimeException('CE backward: dimension mismatch');
			
			$eps = 1.0e-12;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$gradOut = $out->grad[$b] ?? 0.0;
				$scale = $gradOut;
				$rowStart = $b * $dim;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$p = $pred->data[$rowStart + $i];
					$t = $target->data[$rowStart + $i];
					$pred->grad[$rowStart + $i] += -$scale * ($t / ($p + $eps));
				}
			}
			
			return;
		}

		$gradOut = $out->grad[0] ?? 0.0;
		$eps = 1.0e-12;
		$scale = $gradOut;

		for ($i = 0; $i < $classes; $i++)
		{
			$p = $pred->data[$i];
			$t = $target->data[$i];
			$pred->grad[$i] += -$scale * ($t / ($p + $eps));
		}
	}

	private function backwardCeLogits(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];

		$classes = count($logits->data);
		if ($classes === 0 || $classes !== count($target->data))
			return;

		if (count($logits->shape) === 2 && count($target->shape) === 2)
		{
			[$batch, $dim] = $logits->shape;
			
			if ($target->shape[0] !== $batch || $target->shape[1] !== $dim)
				throw new RuntimeException('CE logits backward: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$max = $logits->data[$rowStart];
				
				for ($i = 1; $i < $dim; $i++)
				{
					$val = $logits->data[$rowStart + $i];
					if ($val > $max)
						$max = $val;
				}
				
				$probs = [];
				$sumExp = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$expVal = \exp($logits->data[$rowStart + $i] - $max);
					$probs[$i] = $expVal;
					$sumExp += $expVal;
				}
				
				$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$probs[$i] *= $invSum;
				}
				
				$gradOut = $out->grad[$b] ?? 0.0;
				$scale = $gradOut;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$t = $target->data[$rowStart + $i];
					$logits->grad[$rowStart + $i] += $scale * ($probs[$i] - $t);
				}
			}
			
			return;
		}

		$gradOut = $out->grad[0] ?? 0.0;
		$max = $logits->data[0];
		
		for ($i = 1; $i < $classes; $i++)
		{
			if ($logits->data[$i] > $max)
				$max = $logits->data[$i];
		}
		
		$probs = [];
		$sumExp = 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$expVal = \exp($logits->data[$i] - $max);
			$probs[$i] = $expVal;
			$sumExp += $expVal;
		}
		
		$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$probs[$i] *= $invSum;
		}

		$scale = $gradOut;

		for ($i = 0; $i < $classes; $i++)
		{
			$t = $target->data[$i];
			$logits->grad[$i] += $scale * ($probs[$i] - $t);
		}
	}
	
	private function backwardCeLogitsLabelInt(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];

		$classes = count($logits->data);
		if ($classes === 0)
			return;

		if (count($logits->shape) === 2)
		{
			[$batch, $dim] = $logits->shape;
			
			if (count($target->shape) !== 1 || $target->shape[0] !== $batch)
				throw new RuntimeException('CE logits label int backward: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				$labelInt = $target->data[$b];
				$max = $logits->data[$rowStart];
				
				for ($i = 1; $i < $dim; $i++)
				{
					$val = $logits->data[$rowStart + $i];
					if ($val > $max)
						$max = $val;
				}
				
				$probs = [];
				$sumExp = 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$expVal = \exp($logits->data[$rowStart + $i] - $max);
					$probs[$i] = $expVal;
					$sumExp += $expVal;
				}
				
				$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
				
				for ($i = 0; $i < $dim; $i++)
				{
					$probs[$i] *= $invSum;
				}

				$gradOut = $out->grad[$b] ?? 0.0;
				$scale = $gradOut;

				for ($i = 0; $i < $dim; $i++)
				{
					if ((int)$i === (int)$labelInt)
					{
						$logits->grad[$rowStart + $i] += $scale * ($probs[$i] - 1);
					}
					else
					{
						$logits->grad[$rowStart + $i] += $scale * ($probs[$i]);
					}
				}
			}
			
			return;
		}

		$gradOut = $out->grad[0] ?? 0.0;
		$max = $logits->data[0];
		$labelInt = $target->data[0];
		
		for ($i = 1; $i < $classes; $i++)
		{
			if ($logits->data[$i] > $max)
				$max = $logits->data[$i];
		}
		
		$probs = [];
		$sumExp = 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$expVal = \exp($logits->data[$i] - $max);
			$probs[$i] = $expVal;
			$sumExp += $expVal;
		}
		
		$invSum = $sumExp > 0.0 ? 1 / $sumExp : 0.0;
		
		for ($i = 0; $i < $classes; $i++)
		{
			$probs[$i] *= $invSum;
		}

		$scale = $gradOut;

		for ($i = 0; $i < $classes; $i++)
		{
			if ((int)$i === (int)$labelInt)
			{
				$logits->grad[$i] += $scale * ($probs[$i] - 1);
			}
			else
				$logits->grad[$i] += $scale * ($probs[$i]);
		}
	}
}
