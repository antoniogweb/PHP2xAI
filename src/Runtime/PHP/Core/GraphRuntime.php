<?php

namespace PHP2xAI\Runtime\PHP\Core;

use RuntimeException;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Graph\GraphContext;

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
	private ?GraphContext $context = null;
	
	public function __construct(array $graphDef, ?array $weigths = null)
	{
		$this->graphDef = $graphDef;
		
		// crea i tensori
		foreach ($graphDef['tensors'] as $t)
		{
			$size = array_product($t['shape']) ?: 1;
			
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
			}
			
			$this->tensors[$id]->grad = array_fill(0, $size, 0.0);
			
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
	
	public function setContext(GraphContext $context) : void
	{
		$this->context = $context;
	}
	
	public static function createFromOutputTensor(Tensor $tensor) : GraphRuntime
	{
		$context = $tensor->context;
		
		$graph = $context->export();
		
		$lossId = $context->getTensorId($tensor);
		
		$graph['loss'] = $lossId;
		
		$className = get_called_class();
		
		$graphRuntime = new $className($graph);
		$graphRuntime->setContext($context);
		
		return $graphRuntime;
	}
	
	public function refreshTensorsData()
	{
		$contextTensors = $this->context->getTensors();
		
		foreach ($contextTensors as $contextTensor)
		{
			$tensorId = $contextTensor["id"];
			
			$tensor = $this->context->getTensorFromId($tensorId);
			
			if ($tensor && isset($this->tensors[$tensorId]))
			{
				$tensor->data = $this->getTensorData($tensorId); // $this->tensors[$tensorId]->data;
				$tensor->grad = $this->getTensorGrad($tensorId); //$this->tensors[$tensorId]->grad;
			}
		}
	}
	
	public function getTensorSize(int $tensorId) : int
	{
		if (isset($this->tensors[$tensorId]))
			return array_product($this->tensors[$tensorId]->shape) ?: 1;
		
		return null;
	}
	
	public function getTensorData(int $tensorId) : array
	{
		if (isset($this->tensors[$tensorId]))
			return $this->tensors[$tensorId]->data;
	}
	
	public function getTensorGrad(int $tensorId) : array
	{
		if (isset($this->tensors[$tensorId]))
			return $this->tensors[$tensorId]->grad;
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
			$attributes = $op['attributes'] ?? [];
			
			switch ($name)
			{
				case 'matmul':
					$this->opMatmul($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'add':
					$this->opAdd($inputs[0], $inputs[1], $outId, $attributes);
					break;
				// case 'sub':
				// 	$this->opSub($inputs[0], $inputs[1], $outId, $attributes);
				// 	break;
				// case 'dot':
				// 	$this->opDot($inputs[0], $inputs[1], $outId, $attributes);
				// 	break;
				case 'dropout':
					$this->opDropout($inputs[0], $outId, $attributes);
					break;
				case 'sig':
					$this->opSig($inputs[0], $outId, $attributes);
					break;
				case 'ReLU':
				case 'relu':
					$this->opRelu($inputs[0], $outId, $attributes);
					break;
				case 'LReLU':
					$this->opLRelu($inputs[0], $outId, $attributes);
					break;
				// case 'MSE':
				// 	$this->opMse($inputs[0], $outId, $attributes);
				// 	break;
				// case 'MAE':
				// 	$this->opMae($inputs[0], $outId, $attributes);
				// 	break;
				case 'softmax':
					$this->opSoftmax($inputs[0], $outId, $attributes);
					break;
				case 'CE':
					$this->opCe($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'softmax_ce_logits':
					$this->opCeLogits($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'softmax_ce_logits_label_int':
					$this->opCeLogitsLabelInt($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'mean':
					$this->opMean($inputs[0], $outId, $attributes);
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
	
	private function opAdd(int $aId, int $bId, int $outId, array $attributes): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];
		
		$kernel = $attributes["kernel"] ?? "ADD_GENERIC_LAST";
		
		switch ($kernel)
		{
			case "ADD_1D_LAST":
				$this->ADD_1D_LAST($A, $B, $C);
				break;
			case "ADD_2D_LAST":
				$this->ADD_2D_LAST($A, $B, $C);
				break;
			case "ADD_3D_LAST":
				$this->ADD_3D_LAST($A, $B, $C);
				break;
			case "ADD_GENERIC_LAST":
				$this->ADD_GENERIC_LAST($A, $B, $C);
				break;
		}
	}

// 	private function opSub(int $aId, int $bId, int $outId): void
// 	{
// 		$A = $this->tensors[$aId];
// 		$B = $this->tensors[$bId];
// 		$C = $this->tensors[$outId];
// 
// 		// broadcast support: A[B, N] - B[N] = C[B, N]
// 		if (count($A->shape) === 2 && count($B->shape) === 1)
// 		{
// 			[$batch, $dim] = $A->shape;
// 			
// 			if ($B->shape[0] !== $dim)
// 				throw new RuntimeException('sub: dimension mismatch');
// 			
// 			$C->shape = [$batch, $dim];
// 			$C->data  = array_fill(0, $batch * $dim, 0.0);
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$aRow = $b * $dim;
// 				
// 				for ($n = 0; $n < $dim; $n++)
// 				{
// 					$C->data[$aRow + $n] = $A->data[$aRow + $n] - $B->data[$n];
// 				}
// 			}
// 			
// 			return;
// 		}
// 
// 		$size = count($A->data);
// 		
// 		if ($size !== count($B->data))
// 			throw new RuntimeException('sub: dimension mismatch');
// 
// 		$C->shape = $A->shape;
// 		$C->data  = array_fill(0, $size, 0.0);
// 
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$C->data[$i] = $A->data[$i] - $B->data[$i];
// 		}
// 	}

// 	private function opDot(int $aId, int $bId, int $outId): void
// 	{
// 		$A = $this->tensors[$aId];
// 		$B = $this->tensors[$bId];
// 		$C = $this->tensors[$outId];
// 
// 		$size = count($A->data);
// 		
// 		if ($size !== count($B->data))
// 			throw new RuntimeException('dot: dimension mismatch');
// 
// 		$sum = 0.0;
// 
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$sum += $A->data[$i] * $B->data[$i];
// 		}
// 
// 		$C->shape = [];
// 		$C->data = [$sum];
// 	}
	
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

// 	private function opMse(int $inpId, int $outId): void
// 	{
// 		$X = $this->tensors[$inpId];
// 		$Y = $this->tensors[$outId];
// 		$size = count($X->data);
// 
// 		if ($size === 0)
// 		{
// 			$Y->shape = [];
// 			$Y->data = [0.0];
// 			return;
// 		}
// 
// 		if (count($X->shape) === 0)
// 		{
// 			$val = $X->data[0];
// 			$Y->shape = [];
// 			$Y->data = [0.5 * $val * $val];
// 			return;
// 		}
// 
// 		if (count($X->shape) === 2)
// 		{
// 			[$batch, $dim] = $X->shape;
// 			$Y->shape = [$batch];
// 			$Y->data = array_fill(0, $batch, 0.0);
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$rowStart = $b * $dim;
// 				$sum = 0.0;
// 				
// 				for ($i = 0; $i < $dim; $i++)
// 				{
// 					$val = $X->data[$rowStart + $i];
// 					$sum += $val * $val;
// 				}
// 				
// 				$Y->data[$b] = $dim > 0 ? $sum / $dim : 0.0;
// 			}
// 			
// 			return;
// 		}
// 
// 		$Y->shape = [];
// 		$sum = 0.0;
// 		
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$sum += $X->data[$i] * $X->data[$i];
// 		}
// 		
// 		$Y->data = [$sum / $size];
// 	}

// 	private function opMae(int $inpId, int $outId): void
// 	{
// 		$X = $this->tensors[$inpId];
// 		$Y = $this->tensors[$outId];
// 		$size = count($X->data);
// 
// 		if ($size === 0)
// 		{
// 			$Y->shape = [];
// 			$Y->data = [0.0];
// 			return;
// 		}
// 
// 		if (count($X->shape) === 0)
// 		{
// 			$val = $X->data[0];
// 			$Y->shape = [];
// 			$Y->data = [0.5 * \abs($val)];
// 			return;
// 		}
// 
// 		if (count($X->shape) === 2)
// 		{
// 			[$batch, $dim] = $X->shape;
// 			$Y->shape = [$batch];
// 			$Y->data = array_fill(0, $batch, 0.0);
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$rowStart = $b * $dim;
// 				$sum = 0.0;
// 				
// 				for ($i = 0; $i < $dim; $i++)
// 				{
// 					$sum += \abs($X->data[$rowStart + $i]);
// 				}
// 				
// 				$Y->data[$b] = $dim > 0 ? $sum / $dim : 0.0;
// 			}
// 			
// 			return;
// 		}
// 
// 		$Y->shape = [];
// 		$sum = 0.0;
// 		
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$sum += \abs($X->data[$i]);
// 		}
// 		
// 		$Y->data = [$sum / $size];
// 	}
	
	private function opSoftmax(int $inpId, int $outId, array $attributes): void
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
		
		$kernel = $attributes["kernel"] ?? "SOFTMAX_GENERIC_AXIS";
		$axis = $attributes["axes"][0] ?? -1;
		
		switch ($kernel)
		{
			case "SOFTMAX_1D_LAST":
				$this->SOFTMAX_1D_LAST($X, $Y);
				break;
			case "SOFTMAX_2D_LAST":
				$this->SOFTMAX_2D_LAST($X, $Y);
				break;
			case "SOFTMAX_3D_LAST":
				$this->SOFTMAX_3D_LAST($X, $Y);
				break;
			case "SOFTMAX_GENERIC_AXIS":
				$this->SOFTMAX_GENERIC_AXIS($X, $Y, $axis);
				break;
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
	
	private function opMean(int $aId, int $outId, array $attributes): void
	{
		$A = $this->tensors[$aId];
		$out = $this->tensors[$outId];
		
		$kernel = $attributes["kernel"] ?? "MEAN_GENERIC_AXIS";
		$axis = $attributes["axes"][0] ?? 0;
		
		switch ($kernel)
		{
			case "MEAN_1D_FIRST":
				$this->MEAN_1D_FIRST($A, $out);
				break;
			case "MEAN_2D_FIRST":
				$this->MEAN_2D_FIRST($A, $out);
				break;
			case "MEAN_3D_FIRST":
				$this->MEAN_3D_FIRST($A, $out);
				break;
			case "MEAN_GENERIC_AXIS":
				$this->MEAN_GENERIC_AXIS($A, $out, $axis);
				break;
		}
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
			$attributes = $op['attributes'] ?? [];
			
			switch ($name)
			{
				case 'matmul':
					$this->backwardMatmul($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'add':
					$this->backwardAdd($inputs[0], $inputs[1], $outId, $attributes);
					break;
				// case 'sub':
				// 	$this->backwardSub($inputs[0], $inputs[1], $outId, $attributes);
				// 	break;
				// case 'dot':
				// 	$this->backwardDot($inputs[0], $inputs[1], $outId, $attributes);
				// 	break;
				case 'dropout':
					$this->backwardDropout($inputs[0], $outId, $attributes);
					break;
				case 'sig':
					$this->backwardSig($inputs[0], $outId, $attributes);
					break;
				case 'relu':
				case 'ReLU':
					$this->backwardRelu($inputs[0], $outId, $attributes);
					break;
				case 'LReLU':
					$this->backwardLRelu($inputs[0], $outId, $attributes);
					break;
				// case 'MSE':
				// 	$this->backwardMse($inputs[0], $outId, $attributes);
				// 	break;
				// case 'MAE':
				// 	$this->backwardMae($inputs[0], $outId, $attributes);
				// 	break;
				case 'softmax':
					$this->backwardSoftmax($inputs[0], $outId, $attributes);
					break;
				case 'CE':
					$this->backwardCe($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'softmax_ce_logits':
					$this->backwardCeLogits($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'softmax_ce_logits_label_int':
					$this->backwardCeLogitsLabelInt($inputs[0], $inputs[1], $outId, $attributes);
					break;
				case 'mean':
					$this->backwardMean($inputs[0], $outId, $attributes);
					break;
				default:
					throw new RuntimeException("Op not supported: {$name}");
			}
		}
	}
	
	private function backwardMean(int $aId, int $outId, array $attributes): void
	{
		$A = $this->tensors[$aId];
		$out = $this->tensors[$outId];

		$kernel = $attributes["kernel"] ?? "MEAN_GENERIC_AXIS";
		$axis = $attributes["axes"][0] ?? 0;
		
		switch ($kernel)
		{
			case "MEAN_1D_FIRST":
				$this->BACKWARD_MEAN_1D_FIRST($A, $out);
				break;
			case "MEAN_2D_FIRST":
				$this->BACKWARD_MEAN_2D_FIRST($A, $out);
				break;
			case "MEAN_3D_FIRST":
				$this->BACKWARD_MEAN_3D_FIRST($A, $out);
				break;
			case "MEAN_GENERIC_AXIS":
				$this->BACKWARD_MEAN_GENERIC_AXIS($A, $out, $axis);
				break;
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
	
	private function backwardAdd(int $aId, int $bId, int $outId, array $attributes): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];
		
		switch ($attributes["kernel"])
		{
			case "ADD_1D_LAST":
				$this->BACKWARD_ADD_1D_LAST($A, $B, $C);
				break;
			case "ADD_2D_LAST":
				$this->BACKWARD_ADD_2D_LAST($A, $B, $C);
				break;
			case "ADD_3D_LAST":
				$this->BACKWARD_ADD_3D_LAST($A, $B, $C);
				break;
			case "ADD_GENERIC_LAST":
				$this->BACKWARD_ADD_GENERIC_LAST($A, $B, $C);
				break;
		}
	}

// 	private function backwardSub(int $aId, int $bId, int $outId): void
// 	{
// 		$A = $this->tensors[$aId];
// 		$B = $this->tensors[$bId];
// 		$C = $this->tensors[$outId];
// 
// 		if (count($A->shape) === 2 && count($B->shape) === 1)
// 		{
// 			[$batch, $dim] = $A->shape;
// 			
// 			if ($B->shape[0] !== $dim)
// 				throw new RuntimeException('sub: dimension mismatch');
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$aRow = $b * $dim;
// 				
// 				for ($n = 0; $n < $dim; $n++)
// 				{
// 					$grad = $C->grad[$aRow + $n];
// 					$A->grad[$aRow + $n] += $grad;
// 					$B->grad[$n] -= $grad;
// 				}
// 			}
// 			
// 			return;
// 		}
// 
// 		$size = count($C->data);
// 		
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$A->grad[$i] += $C->grad[$i];
// 			$B->grad[$i] -= $C->grad[$i];
// 		}
// 	}

	// private function backwardDot(int $aId, int $bId, int $outId): void
	// {
	// 	$A = $this->tensors[$aId];
	// 	$B = $this->tensors[$bId];
	// 	$C = $this->tensors[$outId];
 // 
	// 	$gradOut = $C->grad[0] ?? 0.0;
	// 	$size = count($A->data);
 // 
	// 	for ($i = 0; $i < $size; $i++)
	// 	{
	// 		$A->grad[$i] += $gradOut * $B->data[$i];
	// 		$B->grad[$i] += $gradOut * $A->data[$i];
	// 	}
	// }
	
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

// 	private function backwardMse(int $inpId, int $outId): void
// 	{
// 		$X = $this->tensors[$inpId];
// 		$Y = $this->tensors[$outId];
// 		$size = count($X->data);
// 
// 		if ($size === 0)
// 			return;
// 
// 		if (count($X->shape) === 0)
// 		{
// 			$val = $X->data[0];
// 			$gradOut = $Y->grad[0] ?? 0.0;
// 			$X->grad[0] += $gradOut * $val;
// 			return;
// 		}
// 
// 		if (count($X->shape) === 2)
// 		{
// 			[$batch, $dim] = $X->shape;
// 
// 			if (count($Y->shape) !== 1 || $Y->shape[0] !== $batch)
// 				throw new RuntimeException('MSE backward: output shape mismatch');
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$gradOut = $Y->grad[$b] ?? 0.0;
// 				$scale = $dim > 0 ? (2 / $dim) * $gradOut : 0.0;
// 				$rowStart = $b * $dim;
// 				
// 				for ($i = 0; $i < $dim; $i++)
// 				{
// 					$X->grad[$rowStart + $i] += $scale * $X->data[$rowStart + $i];
// 				}
// 			}
// 			
// 			return;
// 		}
// 
// 		$gradOut = $Y->grad[0] ?? 0.0;
// 		$scale = (2 / $size) * $gradOut;
// 
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$X->grad[$i] += $scale * $X->data[$i];
// 		}
// 	}

// 	private function backwardMae(int $inpId, int $outId): void
// 	{
// 		$X = $this->tensors[$inpId];
// 		$Y = $this->tensors[$outId];
// 		$size = count($X->data);
// 
// 		if ($size === 0)
// 			return;
// 
// 		if (count($X->shape) === 0)
// 		{
// 			$val = $X->data[0];
// 			$gradOut = $Y->grad[0] ?? 0.0;
// 			$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
// 			$X->grad[0] += $gradOut * 0.5 * $sign;
// 			return;
// 		}
// 
// 		if (count($X->shape) === 2)
// 		{
// 			[$batch, $dim] = $X->shape;
// 
// 			if (count($Y->shape) !== 1 || $Y->shape[0] !== $batch)
// 				throw new RuntimeException('MAE backward: output shape mismatch');
// 			
// 			for ($b = 0; $b < $batch; $b++)
// 			{
// 				$gradOut = $Y->grad[$b] ?? 0.0;
// 				$scale = $dim > 0 ? (1 / $dim) * $gradOut : 0.0;
// 				$rowStart = $b * $dim;
// 				
// 				for ($i = 0; $i < $dim; $i++)
// 				{
// 					$val = $X->data[$rowStart + $i];
// 					$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
// 					$X->grad[$rowStart + $i] += $scale * $sign;
// 				}
// 			}
// 			
// 			return;
// 		}
// 
// 		$gradOut = $Y->grad[0] ?? 0.0;
// 		$scale = ($size > 0) ? (1 / $size) * $gradOut : 0.0;
// 
// 		for ($i = 0; $i < $size; $i++)
// 		{
// 			$val = $X->data[$i];
// 			$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
// 			$X->grad[$i] += $scale * $sign;
// 		}
// 	}
	
	private function backwardSoftmax(int $inpId, int $outId, array $attributes): void
	{
		$X = $this->tensors[$inpId];
		$Y = $this->tensors[$outId];
		$size = count($Y->data);
		
		$kernel = $attributes["kernel"] ?? "SOFTMAX_GENERIC_AXIS";
		$axis = $attributes["axes"][0] ?? -1;
		
		switch ($kernel)
		{
			case "SOFTMAX_1D_LAST":
				$this->BACKWORD_SOFTMAX_1D_LAST($X, $Y);
				break;
			case "SOFTMAX_2D_LAST":
				$this->BACKWORD_SOFTMAX_2D_LAST($X, $Y);
				break;
			case "SOFTMAX_3D_LAST":
				$this->BACKWORD_SOFTMAX_3D_LAST($X, $Y);
				break;
			case "SOFTMAX_GENERIC_AXIS":
				$this->BACKWORD_SOFTMAX_GENERIC_AXIS($X, $Y, $axis);
				break;
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
	
	// KERNELS REDUCE MEAN
	private function MEAN_1D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{
		$out->shape = [];
		
		if (count($A->shape) !== 1 || count($A->data) === 0)
			throw new RuntimeException('Mean: dimension mismatch');
		
		$mean = array_sum($A->data)/count($A->data);
		
		$out->data = [$mean];
	}
	
	private function MEAN_2D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{

	}
	
	private function MEAN_3D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{
	
	}
	
	private function MEAN_GENERIC_AXIS(TensorRuntime $A, TensorRuntime $out, int $axis)
	{
		$this->reduceMeanAlongAxis($A->data, $A->shape, $A->strides, $axis, $out->shape, $out->data);
	}
	
	private function BACKWARD_MEAN_1D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{
		if (count($A->shape) !== 1)
			throw new RuntimeException('Mean backward: dimension mismatch');
		
		$size = count($A->data);
		
		$gradOut = $out->grad[0] ?? 0.0;
		$scale = $gradOut / $size;

		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $scale;
		}
	}
	
	private function BACKWARD_MEAN_2D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{
		
	}
	
	private function BACKWARD_MEAN_3D_FIRST(TensorRuntime $A, TensorRuntime $out)
	{
		
	}
	
	private function BACKWARD_MEAN_GENERIC_AXIS(TensorRuntime $A, TensorRuntime $out, int $axis)
	{
		
	}
	
	// KERNELS ADD
	private function ADD_1D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
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
	
	private function ADD_2D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
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
		}
	}
	
	private function ADD_3D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
		// broadcast support: A[B, T, N] + B[N] = C[B, T, N]
		if (count($A->shape) === 3 && count($B->shape) === 1)
		{
			[$batch, $time, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('add: dimension mismatch');
			
			$C->shape = [$batch, $time, $dim];
			$C->data = array_fill(0, $batch * $time * $dim, 0.0);
			
			for ($b = 0; $b < $batch; $b++)
			{
				$batchOffset = $b * $time * $dim;
				
				for ($t = 0; $t < $time; $t++)
				{
					$rowOffset = $batchOffset + $t * $dim;
					
					for ($n = 0; $n < $dim; $n++)
					{
						$C->data[$rowOffset + $n] = $A->data[$rowOffset + $n] + $B->data[$n];
					}
				}
			}
		}
	}
	
	private function ADD_GENERIC_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
		$rank = $A->getRank();
		if ($rank < 1 || count($B->shape) !== 1)
			throw new RuntimeException('add: dimension mismatch');
		
		$lastDim = $A->shape[$rank - 1];
		if ($B->shape[0] !== $lastDim)
			throw new RuntimeException('add: dimension mismatch');
		
		$C->shape = $A->shape;
		$C->data = array_fill(0, array_product($A->shape) ?: 1, 0.0);
		
		$BStrides = $this->alignStridesToRank($B->shape, $B->strides, $C->getRank());
		
		$this->addAlongAxisInPlace($C->data,$C->shape,$C->strides,$A->data,$A->strides,$B->data,$BStrides);
	}
	
	private function BACKWARD_ADD_1D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
		$size = count($C->data);
		
		for ($i = 0; $i < $size; $i++)
		{
			$A->grad[$i] += $C->grad[$i];
			$B->grad[$i] += $C->grad[$i];
		}
	}
	
	private function BACKWARD_ADD_2D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
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
		}
	}
	
	private function BACKWARD_ADD_3D_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
		if (count($A->shape) === 3 && count($B->shape) === 1)
		{
			[$batch, $time, $dim] = $A->shape;
			
			if ($B->shape[0] !== $dim)
				throw new RuntimeException('add: dimension mismatch');
			
			for ($b = 0; $b < $batch; $b++)
			{
				$batchOffset = $b * $time * $dim;
				
				for ($t = 0; $t < $time; $t++)
				{
					$rowOffset = $batchOffset + $t * $dim;
					
					for ($n = 0; $n < $dim; $n++)
					{
						$grad = $C->grad[$rowOffset + $n];
						$A->grad[$rowOffset + $n] += $grad;
						$B->grad[$n] += $grad;
					}
				}
			}
		}
	}
	
	private function BACKWARD_ADD_GENERIC_LAST(TensorRuntime $A, TensorRuntime $B, TensorRuntime $C)
	{
		$rank = $C->getRank();
		if ($rank < 1 || count($B->shape) !== 1)
			throw new RuntimeException('add: dimension mismatch');
		
		$lastDim = $C->shape[$rank - 1];
		if ($B->shape[0] !== $lastDim)
			throw new RuntimeException('add: dimension mismatch');
		
		$BStrides = $this->alignStridesToRank($B->shape, $B->strides, $rank);
		$axis = $rank - 1;
		
		$this->forEachSliceAlongAxisIncremental(
			$C->shape,
			$C->strides,
			$axis,
			function (
				int $baseC,
				int $strideCAxis,
				int $axisLen,
				array $idxNoAxis
			) use (
				$A,
				$B,
				$C,
				$BStrides,
				$axis
			) {
				$baseA = 0;
				$baseB = 0;
				
				foreach ($idxNoAxis as $d => $i)
				{
					if ($i === null)
						continue;
					
					$baseA += $i * $A->strides[$d];
					$baseB += $i * $BStrides[$d];
				}
				
				$strideA = $A->strides[$axis];
				$strideB = $BStrides[$axis];
				
				$offC = $baseC;
				$offA = $baseA;
				$offB = $baseB;
				
				for ($i = 0; $i < $axisLen; $i++)
				{
					$grad = $C->grad[$offC];
					$A->grad[$offA] += $grad;
					$B->grad[$offB] += $grad;
					
					$offC += $strideCAxis;
					$offA += $strideA;
					$offB += $strideB;
				}
			}
		);
	}
	
	// KERNELS SOFTMAX
	private function SOFTMAX_1D_LAST(TensorRuntime $X, TensorRuntime $Y)
	{
		$size = count($X->data);
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
	
	private function SOFTMAX_2D_LAST(TensorRuntime $X, TensorRuntime $Y)
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
	}
	
	private function SOFTMAX_3D_LAST(TensorRuntime $X, TensorRuntime $Y)
	{
		[$batch, $time, $dim] = $X->shape;
		$Y->data = array_fill(0, $batch * $time * $dim, 0.0);
		
		for ($b = 0; $b < $batch; $b++)
		{
			$batchOffset = $b * $time * $dim;
			
			for ($t = 0; $t < $time; $t++)
			{
				$rowStart = $batchOffset + $t * $dim;
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
		}
	}
	
	private function SOFTMAX_GENERIC_AXIS(TensorRuntime $X, TensorRuntime $Y, int $axis)
	{
		$Y->shape = $X->shape;
		$Y->data = $X->data;
		
		$this->softmaxAlongAxisInPlace($Y->data, $Y->shape, $X->strides, $axis);
	}
	
	private function BACKWORD_SOFTMAX_1D_LAST($X, $Y)
	{
		$size = count($Y->data);
		
		/**
		 * Backward softmax ottimizzato O(n):
		 *
		 *   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
		 *
		 * Nota: stessa derivazione spiegata nel caso GENERIC_AXIS.
		 *
		 * Dove:
		 * - y_i è l'output softmax
		 * - dL/dy_i è il gradiente in ingresso
		 *
		 * Questo evita il doppio loop O(n^2) del Jacobian esplicito.
		 */
		$dot = 0.0;
		for ($i = 0; $i < $size; $i++)
		{
			$dot += $Y->grad[$i] * $Y->data[$i];
		}
		
		for ($i = 0; $i < $size; $i++)
		{
			$X->grad[$i] += $Y->data[$i] * ($Y->grad[$i] - $dot);
		}
	}
	
	private function BACKWORD_SOFTMAX_2D_LAST(TensorRuntime $X, TensorRuntime $Y)
	{
		if (count($Y->shape) === 2)
		{
			[$batch, $dim] = $Y->shape;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$rowStart = $b * $dim;
				
				/**
				 * Backward softmax per riga (batch, dim) in O(dim):
				 *
				 * 1) dot = sum_j (dL/dy_j * y_j)
				 * 2) dL/dx_i = y_i * (dL/dy_i - dot)
				 *
				 * Nota: stessa derivazione spiegata nel caso GENERIC_AXIS.
				 *
				 * Tutto riferito alla riga corrente.
				 */
				$dot = 0.0;
				for ($i = 0; $i < $dim; $i++)
				{
					$dot += $Y->grad[$rowStart + $i] * $Y->data[$rowStart + $i];
				}
				
				for ($i = 0; $i < $dim; $i++)
				{
					$X->grad[$rowStart + $i] += $Y->data[$rowStart + $i]
						* ($Y->grad[$rowStart + $i] - $dot);
				}
			}
		}
	}
	
	private function BACKWORD_SOFTMAX_3D_LAST(TensorRuntime $X, TensorRuntime $Y)
	{
		if (count($Y->shape) === 3)
		{
			[$batch, $time, $dim] = $Y->shape;
			
			for ($b = 0; $b < $batch; $b++)
			{
				$batchOffset = $b * $time * $dim;
				
				for ($t = 0; $t < $time; $t++)
				{
					$rowStart = $batchOffset + $t * $dim;
					
					/**
					 * Backward softmax per slice [b, t, :]:
					 *
					 * 1) dot = sum_j (dL/dy_j * y_j)
					 * 2) dL/dx_i = y_i * (dL/dy_i - dot)
					 *
					 * Nota: stessa derivazione spiegata nel caso GENERIC_AXIS.
					 *
					 * Tutto sulla dimensione finale (dim).
					 */
					$dot = 0.0;
					for ($i = 0; $i < $dim; $i++)
					{
						$dot += $Y->grad[$rowStart + $i] * $Y->data[$rowStart + $i];
					}
					
					for ($i = 0; $i < $dim; $i++)
					{
						$X->grad[$rowStart + $i] += $Y->data[$rowStart + $i]
							* ($Y->grad[$rowStart + $i] - $dot);
					}
				}
			}
		}
	}
	
	private function BACKWORD_SOFTMAX_GENERIC_AXIS(TensorRuntime $X, TensorRuntime $Y, int $axis)
	{
		$rank = count($Y->shape);
		if ($rank === 0)
			return;
		
		// Normalizza axis solo per ricavare axisLen in modo sicuro.
		$axisNorm = $axis < 0 ? $axis + $rank : $axis;
		if ($axisNorm < 0 || $axisNorm >= $rank)
			throw new InvalidArgumentException("axis out of range");
		
		$axisLen = $Y->shape[$axisNorm];
		if ($axisLen <= 0)
			return;
		
		// Buffer temporanei per una slice:
		// - tmpY: softmax output lungo axis
		// - tmpGrad: grad dL/dY lungo axis
		// Usati per evitare accessi ripetuti al buffer con stride.
		$tmpY = array_fill(0, $axisLen, 0.0);
		$tmpGrad = array_fill(0, $axisLen, 0.0);
		
		$this->forEachSliceAlongAxisIncremental(
			$Y->shape,
			$Y->strides,
			$axis,
			function(int $base, int $strideAxis, int $axisLen, array $idxNoAxis) use ($X, $Y, &$tmpY, &$tmpGrad)
			{
				/**
				 * Backward softmax ottimizzato O(n) per slice:
				 *
				 *   dL/dx_i = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
				 *
				 * Derivazione:
				 *   dy_j/dx_i = y_j * (delta_ij - y_i)
				 *   dL/dx_i = sum_j dL/dy_j * dy_j/dx_i
				 *           = y_i * (dL/dy_i - sum_j dL/dy_j * y_j)
				 *
				 * Calcoliamo prima:
				 *   dot = sum_j (dL/dy_j * y_j)
				 * e poi per ogni i:
				 *   dL/dx_i = y_i * (dL/dy_i - dot)
				 */
				
				// 1) Carica slice Y e gradY in buffer temporanei
				$off = $base;
				$tmpY[0] = $Y->data[$off];
				$tmpGrad[0] = $Y->grad[$off];
				
				for ($i = 1; $i < $axisLen; $i++)
				{
					$off += $strideAxis;
					$tmpY[$i] = $Y->data[$off];
					$tmpGrad[$i] = $Y->grad[$off];
				}
				
				// 2) dot = sum_j (dL/dy_j * y_j)
				$dot = 0.0;
				for ($i = 0; $i < $axisLen; $i++)
				{
					$dot += $tmpGrad[$i] * $tmpY[$i];
				}
				
				// 3) Scrivi dL/dx_i nel tensore X
				$off = $base;
				for ($i = 0; $i < $axisLen; $i++)
				{
					$X->grad[$off] += $tmpY[$i] * ($tmpGrad[$i] - $dot);
					$off += $strideAxis;
				}
			}
		);
	}
	
	/**
	* Itera tutte le "slice" lungo un asse `axis` (es. ultimo asse per softmax)
	* SENZA ricorsione e SENZA ricalcolare ogni volta baseOffset con una somma.
	*
	* Idea:
	* - Scorri tutte le combinazioni degli indici esterni (tutte le dimensioni tranne axis).
	* - Mantieni un "multi-indice" esterno con carry (come un contatore in base mista).
	* - Mantieni anche `baseOffset` aggiornato INCREMENTALMENTE:
	*      quando incrementi idx[d] di 1  => baseOffset += stride[d]
	*      quando fai carry (vai da shape[d]-1 a 0) => baseOffset -= (shape[d]-1) * stride[d]
	*
	* Così eviti di fare, per ogni slice:
	*   base = Σ idx[d] * stride[d]
	* che in PHP costa parecchio.
	*
	* Parametri:
	* - $shape   : [d0, d1, ... d{R-1}]    lunghezze per dimensione
	* - $strides : [s0, s1, ... s{R-1}]    stride per dimensione (in ELEMENTI, non bytes)
	* - $axis    : asse su cui "camminare" dentro ogni slice (0..R-1 oppure negativo tipo -1)
	*
	* Callback:
	*   function(int $baseOffset, int $strideAxis, int $axisLen, array $idxNoAxis): void
	*
	* Dove:
	* - $baseOffset = offset lineare del PRIMO elemento della slice (axis=0)
	* - $strideAxis = quanto devi sommare all'offset per avanzare di 1 lungo axis
	* - $axisLen = numero di elementi lungo axis
	* - $idxNoAxis = gli indici esterni (stesso rank, ma idxNoAxis[axis] = null)
	*
	* Nota: questa callback è PER-SLICE, quindi è perfetta per softmax/layernorm ecc.
	*/
	private function forEachSliceAlongAxisIncremental(
		array $shape,
		array $strides,
		int $axis,
		callable $onSlice
	): void {
		$rank = count($shape);
		if ($rank === 0)
			return;

		if (count($strides) !== $rank)
			throw new InvalidArgumentException("shape/strides rank mismatch");

		// 1) Normalizza axis: -1 => ultimo asse, -2 => penultimo, ecc.
		if ($axis < 0) $axis += $rank;
		if ($axis < 0 || $axis >= $rank)
			throw new InvalidArgumentException("axis out of range");

		// 2) Lunghezza e stride dell'asse interno (quello che vuoi percorrere nella slice)
		$axisLen = $shape[$axis];
		if ($axisLen <= 0)
			return;
		$strideAxis = $strides[$axis];

		// 3) Costruisci la lista delle dimensioni ESTERNE (tutte tranne axis).
		//    L'ordine qui decide l'ordine con cui scorrerai le slice.
		//    Convenzione: facciamo variare più velocemente l'ultima dimensione esterna,
		//    così è simile a un loop annidato "classico".
		$outerDims = [];
		for ($d = 0; $d < $rank; $d++)
		{
			if ($d !== $axis) $outerDims[] = $d;
		}

		// Caso particolare: rank = 1 => c'è UNA sola slice (l'intero tensore).
		if (count($outerDims) === 0)
		{
			$idxNoAxis = array_fill(0, $rank, null);
			$onSlice(0, $strideAxis, $axisLen, $idxNoAxis);
			return;
		}

		// 4) Multi-indice completo (rank), ma useremo solo le posizioni outerDims.
		//    idx[axis] resta sempre 0 (non serve tenerlo aggiornato).
		$idx = array_fill(0, $rank, 0);

		// 5) Numero totale di slice = prodotto delle shape sulle dimensioni esterne.
		$outerCount = 1;
		foreach ($outerDims as $d)
		{
			$n = $shape[$d];
			if ($n <= 0) return;
			$outerCount *= $n;
		}

		// 6) baseOffset iniziale: con tutti gli idx esterni a 0 è 0.
		//    Poi lo aggiorniamo incrementalmente man mano che incrementiamo idx.
		$baseOffset = 0;

		// 7) Loop su tutte le slice (tutte le combinazioni degli indici esterni)
		for ($t = 0; $t < $outerCount; $t++)
		{
			// Costruiamo una vista degli indici esterni per debugging/log (opzionale).
			// È utile per capire "dove sei" nel tensore quando vuoi fare test.
			$idxNoAxis = $idx;
			$idxNoAxis[$axis] = null;

			// Chiama callback PER-SLICE:
			// - baseOffset ti dice dove inizia la slice (axis=0)
			// - strideAxis ti dice come avanzare lungo axis
			// - axisLen quanti elementi ci sono nella slice
			$onSlice($baseOffset, $strideAxis, $axisLen, $idxNoAxis);

			// 8) Incremento con carry del contatore esterno.
			//
			// Immagina dei loop annidati:
			// for idx[d0]=0..shape[d0)-1
			//   for idx[d1]=0..shape[d1)-1
			//     ...
			//
			// Qui lo facciamo a mano con carry, partendo dall'ultima outerDims
			// (che varia più velocemente).
			for ($k = count($outerDims) - 1; $k >= 0; $k--)
			{
				$d = $outerDims[$k];

				// Provo ad incrementare idx[d] di 1
				$idx[$d]++;

				// Se NON ho sforato, è un incremento "semplice":
				// - baseOffset avanza di stride[d]
				if ($idx[$d] < $shape[$d])
				{
					$baseOffset += $strides[$d];
					break; // finito: nessun carry ulteriore
				}

				// Se invece ho sforato, devo fare carry:
				// - idx[d] torna a 0
				// - baseOffset deve tornare indietro di (shape[d]-1)*stride[d]
				//   perché ero passato da shape[d]-1 a shape[d], ma in realtà devo tornare a 0.
				$idx[$d] = 0;
				$baseOffset -= ($shape[$d] - 1) * $strides[$d];

				// e continuo il for per propagare il carry alla dimensione esterna precedente
			}
		}
	}
	
	/**
	* Z += X + Y   (in-place su Z)
	*
	* Usa forEachSliceAlongAxisIncremental per:
	* - fissare tutti gli indici esterni
	* - ciclare lungo axis come inner loop
	*
	* Broadcast:
	* - se una dimensione ha shape 1 sul tensore X o Y,
	*   lo stride effettivo su quella dimensione deve essere 0.
	*
	* Parametri:
	* - &$zData      buffer lineare di Z
	* - $zShape
	* - $zStrides
	* - $xData, $xStrides
	* - $yData, $yStrides
	* - $axis        asse scelto come inner loop (tipicamente -1)
	*/
	private function addAlongAxisInPlace(
		array &$zData,
		array $zShape,
		array $zStrides,
		array $xData,
		array $xStrides,
		array $yData,
		array $yStrides,
		int $axis = -1
	): void {

		$rank = count($zShape);
		if ($rank === 0)
			return;

		// Normalizza axis
		if ($axis < 0) $axis += $rank;
		if ($axis < 0 || $axis >= $rank)
			throw new InvalidArgumentException("axis out of range");

		/**
		* Usiamo l'iteratore per-slice:
		* - lui fissa tutti gli indici esterni
		* - e ci passa baseOffset + strideAxis
		*/
		$this->forEachSliceAlongAxisIncremental(
			$zShape,
			$zStrides,
			$axis,
			function (
				int $baseZ,
				int $strideZAxis,
				int $axisLen,
				array $idxNoAxis
			) use (
				&$zData,
				$xData,
				$yData,
				$xStrides,
				$yStrides,
				$axis
			) {
				/**
				* Per X e Y dobbiamo calcolare:
				* - baseX
				* - baseY
				*
				* usando SOLO gli indici esterni (idxNoAxis).
				* L’asse interno (axis) lo gestiamo dopo col loop.
				*/
				$baseX = 0;
				$baseY = 0;

				foreach ($idxNoAxis as $d => $i)
				{
					if ($i === null)
						continue; // axis

					// se stride è 0 → broadcast → base non cambia
					$baseX += $i * $xStrides[$d];
					$baseY += $i * $yStrides[$d];
				}

				// stride effettivi lungo axis
				$strideX = $xStrides[$axis];
				$strideY = $yStrides[$axis];

				/**
				* Inner loop: qui succede il vero add element-wise
				*
				* Z[baseZ + i*sZ] =
				*     X[baseX + i*sX] + Y[baseY + i*sY]
				*/
				$offZ = $baseZ;
				$offX = $baseX;
				$offY = $baseY;

				for ($i = 0; $i < $axisLen; $i++)
				{
					$zData[$offZ] = $xData[$offX] + $yData[$offY];
					
					$offZ += $strideZAxis;
					$offX += $strideX;
					$offY += $strideY;
				}
			}
		);
	}
	
	/**
	* Softmax IN-PLACE lungo un asse generico (default: ultimo asse).
	*
	* - $data è il buffer lineare (row-major o qualunque: la navigazione la fanno gli strides).
	* - Funziona anche con tensori non contigui (strideAxis può essere >1).
	*
	* Nota prestazioni in PHP:
	* - softmax fa exp() tante volte: quello domina il costo.
	* - però questa struttura minimizza overhead di iterazione sugli indici.
	*/
	private function softmaxAlongAxisInPlace(array &$data, array $shape, array $strides, int $axis = -1): void
	{
		$rank = count($shape);
		if ($rank === 0) return;

		// Normalizza axis qui solo per sapere axisLen e strideAxis per il tmp
		$axisNorm = $axis < 0 ? $axis + $rank : $axis;
		if ($axisNorm < 0 || $axisNorm >= $rank)
			throw new InvalidArgumentException("axis out of range");

		$axisLen = $shape[$axisNorm];
		if ($axisLen <= 0) return;

		// Buffer temporaneo per una slice (evita riletture dal buffer e semplifica la logica)
		$tmp = array_fill(0, $axisLen, 0.0);

		$this->forEachSliceAlongAxisIncremental(
			$shape,
			$strides,
			$axis,
			function(int $base, int $strideAxis, int $axisLen, array $idxNoAxis) use (&$data, &$tmp)
			{
				// 1) Leggi la slice in tmp + trova max per stabilità numerica
				$off = $base;
				$m = (float)$data[$off];
				$tmp[0] = $m;

				for ($i = 1; $i < $axisLen; $i++)
				{
					$off += $strideAxis;
					$v = (float)$data[$off];
					$tmp[$i] = $v;
					if ($v > $m) $m = $v;
				}

				// 2) Calcola exp(v - max) e somma
				$sum = 0.0;
				for ($i = 0; $i < $axisLen; $i++)
				{
					$e = exp($tmp[$i] - $m);
					$tmp[$i] = $e;
					$sum += $e;
				}

				// 3) Normalizza e scrivi in-place nella stessa slice
				$inv = ($sum != 0.0) ? (1.0 / $sum) : 0.0;

				$off = $base;
				$data[$off] = $tmp[0] * $inv;
				
				for ($i = 1; $i < $axisLen; $i++)
				{
					$off += $strideAxis;
					$data[$off] = $tmp[$i] * $inv;
				}
			}
		);
	}
	
	/**
	* reduceMean su asse generico (NO keepdims), scrive direttamente in $outData (passato by ref)
	* e riempie $outShape (passato by ref). Non ritorna nulla.
	*
	* Requisiti:
	* - $outData viene (ri)allocato qui come array contiguo di float
	* - $outShape viene calcolata qui (rank = inRank - 1)
	*
	* Nota: $outData è in row-major contiguo (stride standard).
	*/
	private function reduceMeanAlongAxis(
		array $inData,
		array $inShape,
		array $inStrides,
		int $axis,
		array &$outShape,
		array &$outData
	): void {

		$rank = count($inShape);

		// Caso scalare: mean(scalare) = scalare, output rank 0
		if ($rank === 0) {
			$outShape = [];
			$outData  = $inData; // assume inData = [value]
			return;
		}

		// Normalizza axis (-1 = ultimo asse)
		if ($axis < 0) $axis += $rank;
		if ($axis < 0 || $axis >= $rank) {
			throw new InvalidArgumentException("axis out of range");
		}

		$axisLen = $inShape[$axis];
		if ($axisLen <= 0) {
			// shape degenerata: output coerente ma vuoto
			$outShape = $inShape;
			array_splice($outShape, $axis, 1);
			$outData = [];
			return;
		}

		// 1) outShape = inShape senza la dimensione axis
		$outShape = $inShape;
		array_splice($outShape, $axis, 1);

		// 2) Numero elementi output = prodotto outShape
		//    Se outShape è [] => prodotto vuoto = 1 (una sola media totale)
		$outCount = 1;
		foreach ($outShape as $n) $outCount *= $n;

		// 3) Alloca output contiguo
		$outData = array_fill(0, $outCount, 0.0);

		// 4) Scriviamo sequenzialmente: un valore per slice
		$outPos = 0;
		$invAxisLen = 1.0 / $axisLen;

		$this->forEachSliceAlongAxisIncremental(
			$inShape,
			$inStrides,
			$axis,
			function(int $base, int $strideAxis, int $axisLen, array $idxNoAxis) use (
				$inData,
				&$outData,
				&$outPos,
				$invAxisLen
			) {
				// Somma lungo axis
				$sum = 0.0;
				$off = $base;

				for ($i = 0; $i < $axisLen; $i++) {
					$sum += (float)$inData[$off];
					$off += $strideAxis;
				}

				// Media -> scrivi in output contiguo
				$outData[$outPos++] = $sum * $invAxisLen;
			}
		);
	}
	
	// 	Esempio:
	// 	
	// 	Z.shape = [B, T, D]
	// 
	// 	Y.shape = [D]
	// 
	// 	Allineamento a destra (stile NumPy/PyTorch):
	// 
	// 	Y aligned shape   = [1, 1, D]
	// 	Y aligned strides = [0, 0, 1]
	private function alignStridesToRank
	(
		array $shape,
		array $strides,
		int $targetRank
	): array {
		$rank = count($shape);
		if ($rank > $targetRank)
			throw new InvalidArgumentException("rank > targetRank");

		$aligned = array_fill(0, $targetRank, 0);

		// allinea a destra
		$offset = $targetRank - $rank;

		for ($i = 0; $i < $rank; $i++)
		{
			// se shape==1 → broadcast → stride 0
			$aligned[$offset + $i] = ($shape[$i] == 1) ? 0 : $strides[$i];
		}

		return $aligned;
	}
}
