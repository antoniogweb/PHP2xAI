<?php

namespace PHP2xAI\Runtime\PHP\Core;

use PHP2xAI\Tensor\TensorUtility;

class TensorRuntime
{
	use TensorUtility;
	
	/** @var float[] */
	public array $data;
	/** @var float[] */
	public array $grad;
	/** @var int[] */
	public array $shape;
	public string $kind;
	public ?string $name;
	
	public int $baseOffset = 0;
	
	public array $strides;
	
	public function __construct(array $shape, string $kind, ?string $name = null)
	{
		$this->shape = $shape;
		$this->kind  = $kind;
		$this->name  = $name;
		$size = array_product($shape) ?: 1;
		$this->data = array_fill(0, $size, 0.0);
		$this->grad = array_fill(0, $size, 0.0);
		$this->strides = $this->computeStrides($shape);
	}
}