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
	* @var (Node)[][]
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
	
	public function matMul(Vector $b) : Vector
	{
		$r = array_fill(0, $this->m, 0.0);
		
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$result = new Vector($r, false, 'matmul');
		$context->registerOp('matmul', [$leftId, $rightId], $result);
		
		return $result;
	}
}
