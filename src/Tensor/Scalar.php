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
class Scalar extends Tensor
{
	/**
	* The scalar value
	*
	* @var (Node)
	*/
	public $a;
	
	/**
	* @param float $a
	* @param string|null $name
	*/
	public function __construct(float $a, ?string $name = null)
	{
		// if ($a instanceof Node)
			$this->a = $a;
		// else
		// 	$this->a = new Node($a);
		
		$this->name = $name;
	}
	
	/**
     * Factory method to build a new random scalar
     *
     * @param string|null $name
     * @return Scalar
     */
    public static function random(?string $name = null) : Scalar
    {
		$a = mt_rand() / mt_getrandmax();
		
        return new Scalar($a, $name);
    }
	
	public function getShape() : array
	{
		return [];
	}
	
	public function add(Scalar $b) : Scalar
    {
		// $z = $this->a->add($b->a);
		
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Scalar(0.0, 'add');
		$context->registerOp('add', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function sub(Scalar $b) : Scalar
    {
		// $z = $this->a->sub($b->a);
		
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Scalar(0.0, 'sub');
		$context->registerOp('sub', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function LReLU($alfa = 0.01) : Scalar
    {
		// $z = $this->a->LReLU($alfa);
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Scalar(0.0, 'LReLU');
		$context->registerOp('LReLU', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Squared Error (MSE)
    public function MSE() : Scalar
    {
// 		$z = $this->a->p(2);
// 		
// 		$sc = 1 / 2;
// 		
// 		$z = $z->mul(new Node($sc));
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Scalar(0.0, 'MSE');
		$context->registerOp('MSE', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Absolute Error (MAE)
    public function MAE() : Scalar
    {
// 		$z = $this->a->abs();
// 		
// 		$sc = 1 / 2;
// 		
// 		$z = $z->mul(new Node($sc));
		
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Scalar(0.0, 'MAE');
		$context->registerOp('MAE', [$inputId], $result);
		
		return $result;
    }
}
