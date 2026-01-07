<?php

namespace PHP2xAI\Runtime\PHP\Core;

use RuntimeException;

class GraphRuntime
{
	/** @var TensorRuntime[] */
	public array $tensors = [];
	/** @var array[] */
	public array $ops;
	public int $lossId;
	/** @var int[] */
	public array $trainable;
	
	public int $inputId = 0;
	public int $targetId = 0;
	public int $outputId = 0;
	
	public int $accSteps = 0;
	
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
				$this->tensors[$id]->grad = array_fill(0, count($data), 0.0);
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
		
		$this->trainable = $graphDef['trainable'];
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
	
	public function getLoss()
	{
		$tensor = $this->tensors[$this->lossId];
		
		return $tensor->data[0];
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
		$this->accSteps = 0;
		
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

		// shapes: A[m, n], B[n] oppure [n, p]
		[$m, $n] = $A->shape;
		if (count($B->shape) == 1)
		{
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
// 		else if (count($B->shape) == 2)
// 		{
// 			[$nB, $p] = $B->shape;
// 			
// 			if ($nB !== $n)
// 				throw new RuntimeException('matmul: dimension mismatch');
// 			
// 			// A[m, n] * B[n, p] => C[m, p]
// 			$C->shape = [$m, $p];
// 			$C->data = array_fill(0, $m * $p, 0.0);
// 			
// 			for ($i = 0; $i < $m; $i++)
// 			{
// 				for ($j = 0; $j < $p; $j++)
// 				{
// 					$sum = 0.0;
// 					
// 					for ($k = 0; $k < $n; $k++)
// 					{
// 						$sum += $A->data[$i * $n + $k] * $B->data[$k * $p + $j];
// 					}
// 					
// 					$C->data[$i * $p + $j] = $sum;
// 				}
// 			}
// 		}
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

		// shapes identiche nel tuo modello attuale
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
		$Y->shape = [];

		$size = count($X->data);

		if ($size === 0)
		{
			$Y->data = [0.0];
			return;
		}

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$Y->data = [0.5 * $val * $val];
			return;
		}

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
		$Y->shape = [];

		$size = count($X->data);

		if ($size === 0)
		{
			$Y->data = [0.0];
			return;
		}

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$Y->data = [0.5 * \abs($val)];
			return;
		}

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
		$out->shape = [];

		$classes = count($pred->data);

		if ($classes === 0 || $classes !== count($target->data))
		{
			$out->data = [0.0];
			return;
		}

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
			$out->data = [-\log($prob + $eps) / $classes];
			return;
		}

		$loss = 0.0;

		for ($i = 0; $i < $classes; $i++)
		{
			$loss += $target->data[$i] * \log(($pred->data[$i] ?? 0.0) + $eps);
		}

		$out->data = [-$loss / $classes];
	}

	private function opCeLogits(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$out->shape = [];

		$classes = count($logits->data);

		if ($classes === 0 || $classes !== count($target->data))
		{
			$out->data = [0.0];
			return;
		}

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
		
		$out->data = [$loss / $classes];
	}
	
	private function opCeLogitsLabelInt(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$out->shape = [];

		$classes = count($logits->data);

		if ($classes === 0)
		{
			$out->data = [0.0];
			return;
		}

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
		
		$out->data = [$loss / $classes];
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
		$this->tensors[$this->lossId]->grad[0] = 1.0;

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
				default:
					throw new RuntimeException("Op not supported: {$name}");
			}
		}
		
		$this->accSteps++;
	}
	
	private function backwardMatmul(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

		// caso A[m, n] * B[n] = C[m]
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
	}
	
	private function backwardAdd(int $aId, int $bId, int $outId): void
	{
		$A = $this->tensors[$aId];
		$B = $this->tensors[$bId];
		$C = $this->tensors[$outId];

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
		$gradOut = $Y->grad[0] ?? 0.0;
		$size = count($X->data);

		if ($size === 0)
			return;

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$X->grad[0] += $gradOut * $val;
			return;
		}

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
		$gradOut = $Y->grad[0] ?? 0.0;
		$size = count($X->data);

		if ($size === 0)
			return;

		if (count($X->shape) === 0)
		{
			$val = $X->data[0];
			$sign = $val > 0 ? 1.0 : ($val < 0 ? -1.0 : 0.0);
			$X->grad[0] += $gradOut * 0.5 * $sign;
			return;
		}

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
		$gradOut = $out->grad[0] ?? 0.0;

		$classes = count($pred->data);
		if ($classes === 0 || $classes !== count($target->data))
			return;

		$eps = 1.0e-12;
		$scale = $classes > 0 ? $gradOut / $classes : 0.0;

		for ($i = 0; $i < $classes; $i++)
		{
			$p = $pred->data[$i];
			$t = $target->data[$i];
			$pred->grad[$i] += -$scale * ($t / ($p + $eps));
			$target->grad[$i] += -$scale * \log($p + $eps);
		}
	}

	private function backwardCeLogits(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$gradOut = $out->grad[0] ?? 0.0;

		$classes = count($logits->data);
		if ($classes === 0 || $classes !== count($target->data))
			return;

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

		$scale = $classes > 0 ? $gradOut / $classes : 0.0;

		for ($i = 0; $i < $classes; $i++)
		{
			$t = $target->data[$i];
			$logits->grad[$i] += $scale * ($probs[$i] - $t);
			$target->grad[$i] += -$scale * \log($probs[$i] + 1.0e-12);
		}
	}
	
	private function backwardCeLogitsLabelInt(int $logitsId, int $targetId, int $outId): void
	{
		$logits = $this->tensors[$logitsId];
		$target = $this->tensors[$targetId];
		$out = $this->tensors[$outId];
		$gradOut = $out->grad[0] ?? 0.0;

		$classes = count($logits->data);
		if ($classes === 0)
			return;

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

		$scale = $classes > 0 ? $gradOut / $classes : 0.0;

		for ($i = 0; $i < $classes; $i++)
		{
			if ((int)$i === (int)$labelInt)
			{
				$logits->grad[$i] += $scale * ($probs[$i] - 1);
				$target->grad[0] += -$scale * \log($probs[$i] + 1.0e-12);
			}
			else
				$logits->grad[$i] += $scale * ($probs[$i]);
		}
	}
}
