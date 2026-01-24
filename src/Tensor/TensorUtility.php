<?php

namespace PHP2xAI\Tensor;

/**
 * Matrix
 *
 * A zero, one or two dimensional tensor with integer and/or floating point elements.
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
trait TensorUtility
{
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
