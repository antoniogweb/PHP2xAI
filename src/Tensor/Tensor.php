<?php

namespace PHP2xAI\Tensor;

use PHP2xAI\Graph\GraphContext;

/**
 * Matrix
 *
 * A zero, one or two dimensional tensor with integer and/or floating point elements.
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
class Tensor
{
	use TensorUtility;
	
	/**
	* Operation name that produced this tensor (if any)
	*
	* @var string|null
	*/
	public ?string $name = null;
	
	public array $shape = [];
	
	public array $data = []; // tensor data in row-major
	
	public array $grad = []; // tensor grad in row-major
	
	public array $strides = [];
	
	/**
	* Graph context used for IR construction.
	*
	* @var GraphContext|null
	*/
	public ?GraphContext $context = null;
	
	public function __construct(array $shape, array $data, ?string $name = null)
	{
		$this->shape = $shape;
		$this->data = $data;
		$this->name = $name;
		$this->grad = array_fill(0, count($data), 1);
		
		$this->computeStrides();
	}
	
	public static function createFromData(array $multidimensionalArrayOfData, ?string $name = null) : Tensor
	{
		$inferShape = function ($data) use (&$inferShape) : array
		{
			if (!is_array($data))
				return [];
			
			$len = count($data);
			
			if ($len === 0)
				return [0];
			
			$first = reset($data);
			$subshape = $inferShape($first);
			
			foreach ($data as $item)
			{
				if (is_array($item) !== is_array($first))
					throw new \RuntimeException('Inconsistent tensor data: mixed scalar and array at same depth');
				
				if (is_array($item))
				{
					$itemShape = $inferShape($item);
					
					if ($itemShape !== $subshape)
						throw new \RuntimeException('Inconsistent tensor data: non-rectangular shape');
				}
			}
			
			return array_merge([$len], $subshape);
		};
		
		$data = array();
		
		$flatten = function ($input) use (&$flatten, &$data) : void
		{
			if (is_array($input))
			{
				foreach ($input as $item)
				{
					$flatten($item);
				}
				
				return;
			}
			
			$data[] = $input;
		};
		
		$shape = $inferShape($multidimensionalArrayOfData);
		$flatten($multidimensionalArrayOfData);
		
		return new Tensor($shape, $data, $name);
	}
	
	public static function random(array $shape, ?string $name = null) : Tensor
	{
		$count = array_product($shape);
		$data = array();
		
		$max = mt_getrandmax();
		
		for ($i = 0; $i < $count; $i++)
		{
			$data[$i] = mt_rand() / $max;
		}
		
		return new Tensor($shape, $data, $name);
	}
	
	public static function zeros(array $shape, ?string $name = null) : Tensor
	{
		$data = array_fill(0, array_product($shape), 0);
		
		return new Tensor($shape, $data, $name);
	}
	
	public static function init(array $shape, float $scale = 0.05, ?string $name = null) : Tensor
	{
		$tensor = self::random($shape, $name);
		
		for ($i = 0; $i < count($tensor->data); $i++)
		{
			$val = ($tensor->data[$i] - 0.5) * 2 * $scale;
			$tensor->data[$i] = $val;
		}
		
		return $tensor;
	}
	
	// [B, D] x [D, N] = [B, N]
	public function matMul(Tensor $b) : Tensor
	{
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = self::zeros(array($this->shape[0], $b->shape[1]), 'matmul');
		$context->registerOp('matmul', [$leftId, $rightId], $result);
		
		return $result;
	}
	
	public function add(Tensor $b) : Tensor
    {
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = self::zeros($this->shape, 'add');
		$context->registerOp('add', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function sub(Tensor $b) : Tensor
    {
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = self::zeros($this->shape, 'sub');
		$context->registerOp('sub', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function dropout(int $perc = 50) : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shape, 'dropout');
		$context->registerOp('dropout', [$inputId], $result);
		
		return $result;
    }
    
    public function sig() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shape, 'sig');
		$context->registerOp('sig', [$inputId], $result);
		
		return $result;
    }
    
    public function ReLU() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shape, 'ReLU');
		$context->registerOp('ReLU', [$inputId], $result);
		
		return $result;
    }
    
    public function LReLU(float $alfa = 0.01) : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shape, 'LReLU');
		$context->registerOp('LReLU', [$inputId], $result);
		
		return $result;
    }
    
    // Softmax activation
    public function softmax() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shape, 'softmax');
		$context->registerOp('softmax', [$inputId], $result);
		
		return $result;
    }
    
    public function shapeReduced(int $index = 0) : array
	{
		$nSize = $this->shape;
		array_pop($nSize);
		
		return $nSize;
	}
    
    // Mean Squared Error (MSE)
    public function MSE() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shapeReduced(), 'MSE');
		$context->registerOp('MSE', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Absolute Error (MAE)
    public function MAE() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shapeReduced(), 'MAE');
		$context->registerOp('MAE', [$inputId], $result);
		
		return $result;
    }
    
    // Cross Entropy
    public function CE(Tensor $target) : Tensor
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = self::zeros($this->shapeReduced(), 'CE');
		$context->registerOp('CE', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    public function CELogitsLabelInt(Tensor $target) : Tensor
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = self::zeros($this->shapeReduced(), 'CELogitsLabelInt');
		$context->registerOp('softmax_ce_logits_label_int', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    /**
     * Cross Entropy computed directly from logits (numerically stable and no softmax graph).
     * Derivative: dL/dz_i = (softmax_i - target_i) / n
     */
    public function CELogits(Tensor $target) : Tensor
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = self::zeros($this->shapeReduced(), 'CELogits');
		$context->registerOp('softmax_ce_logits', [$logitsId, $targetId], $result);
		
		return $result;
    }
	
	// Mean among batch samples
    public function mean() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->shapeReduced(), 'mean');
		$context->registerOp('mean', [$inputId], $result);
		
		return $result;
    }
	
	public function setName(?string $name) : void
	{
		$this->name = $name;
	}
	
	public function getName() : ?string
	{
		return $this->name;
	}
	
	public function setContext(?GraphContext $context) : void
	{
		$this->context = $context;
	}
	
	public function getContext() : ?GraphContext
	{
		return $this->context;
	}
	
	public function getShape() : array
	{
		return $this->shape;
	}
	
	protected function initContextFrom(Tensor ...$inputs) : GraphContext
	{
		$context = $this->context;
		
		foreach ($inputs as $tensor)
		{
			if ($tensor !== null && $tensor->getContext() !== null)
			{
				$context = $tensor->getContext();
				break;
			}
		}
		
		if ($context === null)
			$context = new GraphContext();
		
		if ($this->context === null)
			$this->context = $context;
		
		foreach ($inputs as $tensor)
		{
			if ($tensor !== null && $tensor->getContext() === null)
				$tensor->setContext($context);
		}
		
		return $context;
	}
	
	protected function registerInContext(GraphContext $context, Tensor $tensor, string $defaultKind = 'intermediate') : int
	{
		if ($context->hasTensor($tensor))
			return (int)$context->getTensorId($tensor);
		
		return $context->registerTensor($tensor, $defaultKind, $tensor->getName(), $tensor->getShape());
	}
}
