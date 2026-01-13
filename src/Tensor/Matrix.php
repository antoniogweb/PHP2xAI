<?php

namespace PHP2xAI\Tensor;

/**
* Matrix
*
* A two dimensional (rank 2) tensor with integer and/or floating point elements.
*
* @category    Scientific Computing
* @package     antoniogweb/PHP2xAI
* @author      Antonio Gallo <info@laboratoriolibero.com>
*/
class Matrix extends Tensor
{
	/**
	* A 2-dimensional sequential array that holds the values of the matrix.
	*
	* @var [][]
	*/
	public $a;

	/**
	* The number of rows in the matrix.
	*
	* @var int
	*/
	public $m;

	/**
	* The number of columns in the matrix.
	*
	* @var int
	*/
	public $n;
	
	
	/**
	* @param array[] $a
	* @param string|null $name
	* @return Matrix
	*/
	public function __construct(array $a, ?string $name = null)
	{
		$m = count($a);
		$n = count(current($a));
		
		$this->a = $a;
		$this->m = $m;
		$this->n = $n;
		$this->name = $name;
	}
	
	/**
	* Factory method to build a new matrix having random values.
	*
	* @param int $m
	* @param int $n
	* @param string|null $name
	* @return self
	*/
	public static function random(int $m, int $n, ?string $name = null) : Matrix
	{
		$max = mt_getrandmax();

		$a = [];

		while (count($a) < $m) {
			$row = [];

			while (count($row) < $n) {
				$row[] = mt_rand() / $max;
			}

			$a[] = $row;
		}
		
		$matrix = new Matrix($a, $name);
		
		return $matrix;
	}
	
	public static function zeros(int $m, int $n, ?string $name = null) : Matrix
	{
		$a = [];

		while (count($a) < $m) {
			$row = [];

			while (count($row) < $n) {
				$row[] = 0.0;
			}

			$a[] = $row;
		}
		
		$matrix = new Matrix($a, $name);
		
		return $matrix;
	}
	
	/**
	 * Factory method to build a scaled random matrix centered on 0.
	 */
	public static function init(int $m, int $n, float $scale = 0.05, ?string $name = null) : Matrix
	{
		$matrix = self::random($m, $n, $name);
		
		for ($i = 0; $i < $m; $i++)
		{
			for ($j = 0; $j < $n; $j++)
			{
				$val = ($matrix->a[$i][$j] - 0.5) * 2 * $scale;
				$matrix->a[$i][$j] = $val;
			}
		}
		
		return $matrix;
	}
	
	public function getShape() : array
	{
		return [$this->m, $this->n];
	}
	
	// [B, D] x [D, N] = [B, N]
	public function matMul(Matrix $b) : Matrix
	{
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = self::zeros($this->m, $b->n, 'matmul');
		// $result = new Vector($r, false, 'matmul');
		$context->registerOp('matmul', [$leftId, $rightId], $result);
		
		return $result;
	}
	
	public function add(Vector $b) : Matrix
    {
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = self::zeros($this->m, $this->n, 'add');
		$context->registerOp('add', [$leftId, $rightId], $result);
		
		return $result;
    }
    
    public function dropout(int $perc = 50) : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, $this->n, 'dropout');
		// $result = new Vector($r, false, 'dropout');
		$context->registerOp('dropout', [$inputId], $result);
		
		return $result;
    }
    
    public function sig() : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, $this->n, 'sig');
		$context->registerOp('sig', [$inputId], $result);
		
		return $result;
    }
    
    public function ReLU() : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, $this->n, 'ReLU');
		$context->registerOp('ReLU', [$inputId], $result);
		
		return $result;
    }
    
    public function LReLU($alfa = 0.01) : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, $this->n, 'LReLU');
		$context->registerOp('LReLU', [$inputId], $result);
		
		return $result;
    }
    
    // Softmax activation
    public function softmax() : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, $this->n, 'softmax');
		$context->registerOp('softmax', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Squared Error (MSE)
    public function MSE() : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, 1, 'MSE');
		$context->registerOp('MSE', [$inputId], $result);
		
		return $result;
    }
    
    // Mean Absolute Error (MAE)
    public function MAE() : Matrix
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = self::zeros($this->m, 1, 'MAE');
		$context->registerOp('MAE', [$inputId], $result);
		
		return $result;
    }
    
    // Cross Entropy
    public function CE(Matrix $target) : Vector
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Vector(array_fill(0, $this->m, 0), true, 'CE');
		// $result = self::zeros($this->m, 1, 'CE');
		$context->registerOp('CE', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    public function CELogitsLabelInt(Vector $target) : Vector
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Vector(array_fill(0, $this->m, 0), true, 'CELogitsLabelInt');
		// $result = self::zeros($this->m, 1, 'CELogitsLabelInt');
		$context->registerOp('softmax_ce_logits_label_int', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    /**
     * Cross Entropy computed directly from logits (numerically stable and no softmax graph).
     * Derivative: dL/dz_i = (softmax_i - target_i) / n
     */
    public function CELogits(Matrix $target) : Vector
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Vector(array_fill(0, $this->m, 0), true, 'CELogits');
		// $result = self::zeros($this->m, 1, 'CELogits');
		$context->registerOp('softmax_ce_logits', [$logitsId, $targetId], $result);
		
		return $result;
    }
}
