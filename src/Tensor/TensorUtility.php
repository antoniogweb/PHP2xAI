<?php

namespace PHP2xAI\Tensor;

/**
 *
 * useful methods to manage Tensors data
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
trait TensorUtility
{
	public function print(string $type = 'data', array $numPerDimension = [3, 3, 3]) : string
	{
		$values = ($type === 'grad') ? $this->grad : $this->data;
		$rank = count($this->shape);
		
		if ($rank >= 4)
		{
			$out = print_r($values, true);
			echo $out;
			
			return $out;
		}
		
		$limitForDim = function (int $dim) use ($numPerDimension) : int
		{
			$limit = $numPerDimension[$dim] ?? 3;
			
			return ($limit <= 0) ? PHP_INT_MAX : $limit;
		};
		
		$valueAt = function (array $indices) use ($values)
		{
			$off = $this->offset($indices);
			
			return $values[$off] ?? null;
		};
		
		$buildLine = function (callable $getter, int $dimSize, int $limit) : string
		{
			if ($dimSize === 0)
				return '';
			
			if ($dimSize <= $limit)
			{
				$parts = array();
				
				for ($i = 0; $i < $dimSize; $i++)
					$parts[] = (string)$getter($i);
				
				return implode(' ', $parts);
			}
			
			$parts = array();
			
			for ($i = 0; $i < $limit; $i++)
				$parts[] = (string)$getter($i);
			
			$parts[] = '...';
			$parts[] = (string)$getter($dimSize - 1);
			
			return implode(' ', $parts);
		};
		
		if ($rank === 1)
		{
			$line = $buildLine(function (int $i) use ($valueAt) {
				return $valueAt([$i]);
			}, $this->shape[0], $limitForDim(0));
			
			$out = '['.$line.']';
			echo $this->name." ".ucfirst($type).":\n".$out.PHP_EOL;
			
			return $out;
		}
		
		if ($rank === 2)
		{
			$lines = array();
			$lines[] = '[';
			
			$rows = $this->shape[0];
			$cols = $this->shape[1];
			$rowLimit = $limitForDim(0);

			$colLimit = $limitForDim(1);
			$rowLine = function (int $r) use ($buildLine, $valueAt, $cols, $colLimit) : string
			{
				$line = $buildLine(function (int $c) use ($valueAt, $r) {
					return $valueAt([$r, $c]);
				}, $cols, $colLimit);
				
				return '  ['.$line.']';
			};
			
			if ($rows <= $rowLimit)
			{
				for ($r = 0; $r < $rows; $r++)
					$lines[] = $rowLine($r);
			}
			else
			{
				for ($r = 0; $r < $rowLimit; $r++)
					$lines[] = $rowLine($r);
				
				$lines[] = '  ...';
				$lines[] = $rowLine($rows - 1);
			}
			
			$lines[] = ']';
			
			$out = implode(PHP_EOL, $lines);
			echo $this->name." ".ucfirst($type).":\n".$out.PHP_EOL;
			
			return $out;
		}
		
		if ($rank === 3)
		{
			$lines = array();
			$slices = $this->shape[0];
			$rows = $this->shape[1];
			$cols = $this->shape[2];
			
			$sliceLimit = $limitForDim(0);
			$rowLimit = $limitForDim(1);
			$colLimit = $limitForDim(2);
			
			$printSlice = function (int $s) use (&$lines, $rows, $cols, $rowLimit, $colLimit, $buildLine, $valueAt) : void
			{
				$lines[] = '[dim 1 = '.$s.']';
				$lines[] = '[';
				
				$rowLine = function (int $r) use ($buildLine, $valueAt, $cols, $colLimit, $s) : string
				{
					$line = $buildLine(function (int $c) use ($valueAt, $s, $r) {
						return $valueAt([$s, $r, $c]);
					}, $cols, $colLimit);
					
					return '  ['.$line.']';
				};
				
				if ($rows <= $rowLimit)
				{
					for ($r = 0; $r < $rows; $r++)
						$lines[] = $rowLine($r);
				}
				else
				{
					for ($r = 0; $r < $rowLimit; $r++)
						$lines[] = $rowLine($r);
					
					$lines[] = '  ...';
					$lines[] = $rowLine($rows - 1);
				}
				
				$lines[] = ']';
			};
			
			if ($slices <= $sliceLimit)
			{
				for ($s = 0; $s < $slices; $s++)
				{
					$printSlice($s);
					
					if ($s < $slices - 1)
						$lines[] = '';
				}
			}
			else
			{
				for ($s = 0; $s < $sliceLimit; $s++)
				{
					$printSlice($s);
					$lines[] = '';
				}
				
				$lines[] = '...';
				$lines[] = '';
				
				$printSlice($slices - 1);
			}
			
			$out = implode(PHP_EOL, $lines);
			echo $this->name." ".ucfirst($type).":\n".$out.PHP_EOL;
			
			return $out;
		}
		
		$out = $this->name." ".ucfirst($type).":\n".print_r($values, true);
		echo $out;
		
		return $out;
	}
	
	public function computeStrides(): void
	{
		$rank = count($this->shape);
		$strides = array_fill(0, $rank, 0);

		$acc = 1;
		
		for ($a = $rank - 1; $a >= 0; --$a)
		{
			$strides[$a] = $acc;
			$acc *= $this->shape[$a];
		}
		
		$this->strides = $strides;
	}
	
	public function offset(array $indices): int
	{
		$rank = count($this->shape);
		
		if (count($indices) !== $rank)
		{
			throw new Exception("Wrong rank: expected $rank indices");
		}

		$off = 0;
		for ($a = 0; $a < $rank; ++$a)
		{
			$i = $indices[$a];
			$d = $this->shape[$a];

			if ($i < 0 || $i >= $d)
				throw new Exception("Index out of bounds at axis $a: $i (dim=$d)");

			$off += $i * $this->strides[$a];
		}
		
		return $off;
	}
	
	public function get(array $indices): int|float
	{
		$off = $this->offset($indices);
		
		return $this->data[$off];
	}
	
	public function set(array $indices, int|float $value): void
	{
		$off = $this->offset($indices);
		
		$this->data[$off] = $value;
	}
}
