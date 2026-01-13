<?php

namespace PHP2xAI\Tensor;

/**
 * Matrix
 *
 * A one dimensional (rank 1) tensor with integer and/or floating point elements.
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
class Vector extends Tensor
{
	 /**
     * The 1-d sequential array that holds the values of the vector.
     *
     * @var []
     */
    public $a;

    /**
     * The number of elements in the vector.
     *
     * @var int
     */
    public $n;
    
    
    /**
     * @param (int|float)[] $a
     * @param bool $build
     * @param string|null $name
     */
    public function __construct(array $a, bool $build = false, ?string $name = null)
    {
		$this->n = count($a);
		
		$this->a = $a;
		
		$this->name = $name;
    }
    
    /**
     * Factory method to build a new random vector
     *
     * @param int $n
     * @param string|null $name
     * @return Vector
     */
    public static function random(int $n, ?string $name = null) : Vector
    {
        $max = mt_getrandmax();

        $a = [];

        while (count($a) < $n) {
            $a[] = mt_rand() / $max;
        }
        
        $vector = new Vector($a, false, $name);
        
        return $vector;
    }
    
    /**
     * Factory method to build a zero vector.
     *
     * @param int $n
     * @param string|null $name
     */
    public static function zeros(int $n, ?string $name = null) : Vector
    {
		$a = [];
		
		$a = array_fill(0, $n, 0.0);
		
		return new Vector($a, false, $name);
    }
    
	public function getShape() : array
	{
		return [$this->n];
	}
    
    public function add(Vector $b) : Vector
    {
		$r = array_fill(0, $b->n, 0.0);
		
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Vector($r, false, 'add');
		$context->registerOp('add', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function sub(Vector $b) : Vector
    {
		$r = array_fill(0, $b->n, 0.0);
		
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Vector($r, false, 'sub');
		$context->registerOp('sub', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function dot(Vector $b) : Scalar
    {
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Scalar(0.0, 'dot');
		$context->registerOp('dot', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function dropout(int $perc = 50) : Vector
    {
		$r = array_fill(0, $this->n, 0.0);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Vector($r, false, 'dropout');
		$context->registerOp('dropout', [$inputId], $result);
		
		return $result;
    }
    
    public function sig() : Vector
    {
		$r = array_fill(0, $this->n, 0.0);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Vector($r, false, 'sig');
		$context->registerOp('sig', [$inputId], $result);
		
		return $result;
    }
    
    public function ReLU() : Vector
    {
		$r = array_fill(0, $this->n, 0.0);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Vector($r, false, 'ReLU');
		$context->registerOp('ReLU', [$inputId], $result);
		
		return $result;
    }
    
    public function LReLU($alfa = 0.01) : Vector
    {
		$r = array_fill(0, $this->n, 0.0);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Vector($r, false, 'LReLU');
		$context->registerOp('LReLU', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Squared Error (MSE)
    public function MSE() : Scalar
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Scalar(0.0, 'MSE');
		
		$context->registerOp('MSE', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Absolute Error (MAE)
    public function MAE() : Scalar
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Scalar(0.0, 'MAE');
		$context->registerOp('MAE', [$inputId], $result);
		
		return $result;
    }
    
    // Softmax activation
    public function softmax() : Vector
    {
		if ($this->n === 0)
			return new Vector([], false, 'softmax');
		
		$r = array_fill(0, $this->n, 0.0);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Vector($r, false, 'softmax');
		$context->registerOp('softmax', [$inputId], $result);
		
		return $result;
    }
    
    // Cross Entropy
    public function CE(Vector $target) : Scalar
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Scalar(0.0, 'CE');
		$context->registerOp('CE', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    public function CELogitsLabelInt(Scalar $target) : Scalar
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Scalar(0.0, 'CELogitsLabelInt');
		$context->registerOp('softmax_ce_logits_label_int', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    /**
     * Cross Entropy computed directly from logits (numerically stable and no softmax graph).
     * Derivative: dL/dz_i = (softmax_i - target_i) / n
     */
    public function CELogits(Vector $target) : Scalar
    {
		$classes = $this->n;
		
		if ($classes === 0)
			return new Scalar(0.0, 'CELogits');
		
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Scalar(0.0, 'CELogits');
		$context->registerOp('softmax_ce_logits', [$logitsId, $targetId], $result);
		
		return $result;
    }
}
