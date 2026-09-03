<?php

namespace PHP2xAI\Tensor;

use PHP2xAI\Graph\GraphContext;
use Exception;

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

	public ?string $initType = null;

	public float $initScale = 0.0;

	public ?int $initSeed = null;
	
	public array $grad = []; // tensor grad in row-major
	
	public array $strides = [];
	
	public int $baseOffset = 0;
	
	protected bool $trainable = true;

	protected bool $requiresGrad = false;
	
	/**
	* Graph context used for IR construction.
	*
	* @var GraphContext|null
	*/
	public ?GraphContext $context = null;
	
	public function __construct(
		array $shape,
		array $data,
		?string $name = null,
		?string $initType = null,
		float $initScale = 0.05,
		?int $initSeed = null
	)
	{
		if ($initType !== null && !in_array($initType, ['zeros', 'rand'], true))
			throw new Exception("Unsupported tensor init type: {$initType}");

		if ($initScale < 0.0)
			throw new Exception('Tensor init scale must be >= 0');

		if ($initSeed !== null && ($initSeed < 0 || $initSeed > 0xFFFFFFFF))
			throw new Exception('Tensor init seed must be between 0 and 4294967295');

		if ($initType === 'rand' && $initSeed === null)
			$initSeed = random_int(0, 0xFFFFFFFF);

		$this->shape = $shape;
		$this->data = $data;
		$this->name = $name;
		$this->initType = $initType;
		$this->initScale = $initScale;
		$this->initSeed = $initSeed;
		$this->grad = array_fill(0, count($data), 0.0);
		
		$this->strides = self::computeStrides($shape);
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
		return new Tensor($shape, [], $name, 'zeros', 0.0);
	}
	
	public static function init(
		array $shape,
		float $scale = 0.05,
		?string $name = null,
		?int $seed = null
	) : Tensor
	{
		return new Tensor($shape, [], $name, 'rand', $scale, $seed);
	}
	
	public function setTrainable(bool $trainable) : void
	{
		$this->trainable = $trainable;
	}
	
	public function isTrainable() : bool
	{
		return $this->trainable;
	}
	
	public function setRequiresGrad(bool $requiresGrad) : void
	{
		$this->requiresGrad = $requiresGrad;
	}

	public function requiresGrad() : bool
	{
		return $this->requiresGrad;
	}

	public function matMul(Tensor $b) : Tensor
	{
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		$thisRank = $this->getRank();
		$bRank = $b->getRank();
		
		if ($thisRank < 2 || $bRank < 2)
			throw new Exception("Matmul only for rank >= 2");
		
		if ($thisRank === 2 && $bRank === 2)
		{
			// [B, D] x [D, N] = [B, N]
			$kernel = "MATMUL_2D_2D";
			$outputShape = array($this->shape[0], $b->shape[1]);
			
			if ($this->shape[1] != $b->shape[0])
				throw new Exception("Matmul dimensions mismatch");
		}
		else if ($thisRank === 3 && $bRank === 3)
		{
			// [T, B, D] x [T, D, N] = [T, B, N]
			$kernel = "MATMUL_1B_2D_2D";
			$outputShape = array($this->shape[0], $this->shape[1], $b->shape[2]);
			
			if ($this->shape[2] != $b->shape[1])
				throw new Exception("Matmul dimensions mismatch");
		}
		else if ($thisRank === 4 && $bRank === 4)
		{
			// [B, H, T, D_h] * [B, H, D_h, T] = [B, H, T, T]
			$kernel = "MATMUL_2B_2D_2D";
			$outputShape = array($this->shape[0], $this->shape[1], $this->shape[2], $b->shape[3]);
			
			if ($this->shape[3] != $b->shape[2])
				throw new Exception("Matmul dimensions mismatch");
		}
		else if ($thisRank === 3 && $bRank === 2)
		{
			// [B, T, D] * [D, H] = [B, T, H]
			$kernel = "MATMUL_1B_2D_2D_LINEAR";
			$outputShape = array($this->shape[0], $this->shape[1], $b->shape[1]);
			
			if ($this->shape[2] != $b->shape[0])
				throw new Exception("Matmul dimensions mismatch");
		}
		else
		{
			$kernel = "MATMUL_GENERIC_B_2D_2D_BROADCAST";
			$outputShape = $this->shapeReduced(-1);
			$outputShape[] = $b->shape[count($b->shape)-1];
			
			if ($this->shape[count($this->shape)-1] != $b->shape[count($this->shape)-2])
				throw new Exception("Matmul dimensions mismatch");
		}
		
		$result = new Tensor($outputShape, [], 'matmul');
		$context->registerOp('matmul', [$leftId, $rightId], $result, array("kernel" => $kernel));
		
		return $result;
	}

	/**
	 * Swaps two axes of this tensor.
	 *
	 * Negative axes are accepted and are resolved relative to the tensor rank.
	 * For example, transpose([-2, -1]) swaps the last two dimensions.
	 */
	public function transpose(array $axes = [-2, -1]) : Tensor
	{
		if (count($axes) !== 2)
			throw new Exception("Transpose requires exactly two axes");

		$rank = $this->getRank();
		if ($rank < 2)
			throw new Exception("Transpose only for rank >= 2");

		$normalizedAxes = [];
		foreach ($axes as $axis)
		{
			if (!is_int($axis))
				throw new Exception("Transpose axes must be integers");

			if ($axis < 0)
				$axis += $rank;

			if ($axis < 0 || $axis >= $rank)
				throw new Exception("Transpose axis out of range");

			$normalizedAxes[] = $axis;
		}

		[$axisA, $axisB] = $normalizedAxes;
		if ($axisA === $axisB)
			throw new Exception("Transpose axes must be different");

		$outputShape = $this->shape;
		[$outputShape[$axisA], $outputShape[$axisB]] = [$outputShape[$axisB], $outputShape[$axisA]];

		$axisPair = [$axisA, $axisB];
		sort($axisPair);

		if ($rank === 2 && $axisPair === [0, 1])
			$kernel = "TRANSPOSE_2D";
		else if ($rank === 3 && $axisPair === [1, 2])
			$kernel = "TRANSPOSE_3D_LAST_TWO";
		else if ($rank === 4 && $axisPair === [2, 3])
			$kernel = "TRANSPOSE_4D_LAST_TWO";
		else if ($rank === 4 && $axisPair === [1, 2])
			$kernel = "TRANSPOSE_4D_AXIS_1_2";
		else
			$kernel = "TRANSPOSE_GENERIC";

		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		$result = new Tensor($outputShape, [], 'transpose');
		$context->registerOp('transpose', [$inputId], $result, [
			"kernel" => $kernel,
			"axes" => $normalizedAxes,
		]);

		return $result;
	}

	/**
	 * Changes the tensor shape without changing the row-major element order.
	 *
	 * One -1 dimension is allowed and is inferred from the input size.
	 */
	public function reshape(array $shape) : Tensor
	{
		$shape = array_values($shape);
		$inputSize = array_product($this->shape) ?: 1;
		$inferredAxis = null;
		$knownSize = 1;

		foreach ($shape as $axis => $dimension)
		{
			if (!is_int($dimension))
				throw new Exception("Reshape dimensions must be integers");

			if ($dimension === -1)
			{
				if ($inferredAxis !== null)
					throw new Exception("Reshape allows only one inferred dimension");

				$inferredAxis = $axis;
				continue;
			}

			if ($dimension <= 0)
				throw new Exception("Reshape dimensions must be positive or -1");

			$knownSize *= $dimension;
		}

		$outputShape = $shape;
		if ($inferredAxis !== null)
		{
			if ($knownSize === 0 || $inputSize % $knownSize !== 0)
				throw new Exception("Reshape dimensions mismatch");

			$outputShape[$inferredAxis] = intdiv($inputSize, $knownSize);
		}
		else if ($knownSize !== $inputSize)
		{
			throw new Exception("Reshape dimensions mismatch");
		}

		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		$result = new Tensor($outputShape, [], 'reshape');
		$context->registerOp('reshape', [$inputId], $result);

		return $result;
	}

	/**
	 * Adds sinusoidal positional encoding to token embeddings [B, L, D].
	 */
	public function positionalEncoding() : Tensor
	{
		if ($this->getRank() !== 3)
			throw new Exception("Positional encoding requires rank 3 [B, L, D]");

		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		$result = new Tensor($this->shape, [], 'positionalEncoding');
		$context->registerOp('positional_encoding', [$inputId], $result);

		return $result;
	}
	
	public function add(Tensor $b) : Tensor
    {
		$context = $this->initContextFrom($b);
		$leftId = $this->registerInContext($context, $this);
		$rightId = $this->registerInContext($context, $b);
		
		if ($b->getRank() !== 1)
			throw new Exception("BIAS must have rank 1");
		
		if (count($this->shape) === 1)
			$kernel = "ADD_1D_LAST";
		else if (count($this->shape) === 2)
			$kernel = "ADD_2D_LAST";
		else if (count($this->shape) === 3)
			$kernel = "ADD_3D_LAST";
		else 
			$kernel = "ADD_GENERIC_LAST";
		
		$result = new Tensor($this->shape, [], 'add');
		$context->registerOp('add', [$leftId, $rightId], $result, array("kernel" => $kernel));
		
		return $result;
    }

	/**
	 * Looks up token embeddings for a training batch.
	 *
	 * $this is x_ids [B, L] and $embeddings is the embedding table E [V, D].
	 * The resulting tensor has shape [B, L, D].
	 */
	public function embeddings(Tensor $embeddings) : Tensor
	{
		if ($this->getRank() !== 2)
			throw new Exception("x_ids must have rank 2 [B, L]");

		if ($embeddings->getRank() !== 2)
			throw new Exception("Embeddings must have rank 2 [V, D]");

		$context = $this->initContextFrom($embeddings);
		$inputId = $this->registerInContext($context, $this);
		$embeddingsId = $this->registerInContext($context, $embeddings);

		$result = new Tensor([$this->shape[0], $this->shape[1], $embeddings->shape[1]], [], 'embeddings');
		$context->registerOp('embeddings', [$inputId, $embeddingsId], $result);

		return $result;
	}
    
	/**
	 * Builds a binary padding mask for a training batch.
	 *
	 * $this is x_ids [B, L]. The result has shape [B, L], with 1 for each
	 * token different from $padId and 0 for padding tokens.
	 */
	public function paddingMask(int $padId) : Tensor
	{
		if ($this->getRank() !== 2)
			throw new Exception("x_ids must have rank 2 [B, L]");

		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);

		$result = new Tensor($this->shape, [], 'paddingMask');
		$context->registerOp('padding_mask', [$inputId], $result, array("padId" => $padId));

		return $result;
	}

	/**
	 * Applies a [B, L] padding mask to the last axis of this tensor.
	 *
	 * The mask is normally produced by paddingMask() from token IDs, where a
	 * value of 1 marks a valid key position and 0 marks padding. For example,
	 * it applies [B, L] to attention scores [B, H, L, L]. The runtime will set
	 * masked values to negative infinity before softmax.
	 */
	public function applyPaddingMask(Tensor $mask) : Tensor
	{
		$rank = $this->getRank();
		if ($rank < 2)
			throw new Exception("Input must have rank >= 2 [B, ..., L]");

		if ($mask->getRank() !== 2)
			throw new Exception("Mask must have rank 2 [B, L]");

		$lastAxis = $rank - 1;
		if ($this->shape[0] !== $mask->shape[0] || $this->shape[$lastAxis] !== $mask->shape[1])
			throw new Exception("Input and mask dimensions mismatch");

		$context = $this->initContextFrom($mask);
		$inputId = $this->registerInContext($context, $this);
		$maskId = $this->registerInContext($context, $mask);

		$result = new Tensor($this->shape, [], 'applyPaddingMask');
		$context->registerOp('apply_padding_mask', [$inputId, $maskId], $result);

		return $result;
	}

	/**
	 * Mean-pools token representations in a training batch, excluding padding.
	 *
	 * $this is X [B, L, D] and $mask is [B, L], with 1 for a valid token and
	 * 0 for padding tokens. The resulting tensor has shape [B, D].
	 */
	public function meanPooling(Tensor $mask) : Tensor
	{
		if ($this->getRank() !== 3)
			throw new Exception("Input must have rank 3 [B, L, D]");

		if ($mask->getRank() !== 2)
			throw new Exception("Mask must have rank 2 [B, L]");

		if ($this->shape[0] !== $mask->shape[0] || $this->shape[1] !== $mask->shape[1])
			throw new Exception("Input and mask dimensions mismatch");

		$context = $this->initContextFrom($mask);
		$inputId = $this->registerInContext($context, $this);
		$maskId = $this->registerInContext($context, $mask);

		$result = new Tensor([$this->shape[0], $this->shape[2]], [], 'meanPooling');
		$context->registerOp('mean_pooling', [$inputId, $maskId], $result);

		return $result;
	}

//     public function sub(Tensor $b) : Tensor
//     {
// 		$context = $this->initContextFrom($b);
// 		$leftId = $this->registerInContext($context, $this);
// 		$rightId = $this->registerInContext($context, $b);
// 		
// 		$result = self::zeros($this->shape, 'sub');
// 		$context->registerOp('sub', [$leftId, $rightId], $result);
// 		
// 		return $result;
//     }
    
    public function dropout(float $perc = 50.0) : Tensor
    {
		if ($perc < 0.0 || $perc > 100.0)
			throw new Exception("Dropout percentage must be between 0 and 100");

		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Tensor($this->shape, [], 'dropout');
		$context->registerOp('dropout', [$inputId], $result, ['dropoutPerc' => $perc]);
		
		return $result;
    }
    
    public function sig() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Tensor($this->shape, [], 'sig');
		$context->registerOp('sig', [$inputId], $result);
		
		return $result;
    }

	public function gelu() : Tensor
	{
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);

		$result = new Tensor($this->shape, [], 'gelu');
		$context->registerOp('gelu', [$inputId], $result);

		return $result;
	}

	public function scale(float $scale) : Tensor
	{
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);

		$result = new Tensor($this->shape, [], 'scale');
		$context->registerOp('scale', [$inputId], $result, ['scale' => $scale]);

		return $result;
	}

	/**
	 * Applies Layer Normalization over one axis.
	 *
	 * gamma and beta must be rank-1 tensors with one value for every
	 * element along the normalized axis. By default LayerNorm operates on
	 * the last axis, which covers [D], [B, D], [B, L, D], and [B, H, L, D].
	 */
	public function layerNorm(Tensor $gamma, Tensor $beta, int $axis = -1) : Tensor
	{
		$rank = $this->getRank();
		if ($rank === 0)
			throw new Exception("LayerNorm requires rank >= 1");

		if ($axis < -$rank || $axis >= $rank)
			throw new Exception("LayerNorm axis out of range");

		$normalizedAxis = $axis < 0 ? $axis + $rank : $axis;
		$axisSize = $this->shape[$normalizedAxis];

		if ($gamma->getRank() !== 1 || $gamma->shape[0] !== $axisSize)
			throw new Exception("LayerNorm gamma must have shape [" . $axisSize . "]");

		if ($beta->getRank() !== 1 || $beta->shape[0] !== $axisSize)
			throw new Exception("LayerNorm beta must have shape [" . $axisSize . "]");

		$kernel = $axis === -1 ? "LAYER_NORM_LAST_AXIS" : "LAYER_NORM_GENERIC";

		$context = $this->initContextFrom($gamma, $beta);
		$inputId = $this->registerInContext($context, $this);
		$gammaId = $this->registerInContext($context, $gamma);
		$betaId = $this->registerInContext($context, $beta);

		$result = new Tensor($this->shape, [], 'layerNorm');
		$context->registerOp('layer_norm', [$inputId, $gammaId, $betaId], $result, [
			"kernel" => $kernel,
			"axes" => [$normalizedAxis],
		]);

		return $result;
	}
    
    public function ReLU() : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		$result = new Tensor($this->shape, [], 'ReLU');
		$context->registerOp('ReLU', [$inputId], $result);
		
		return $result;
    }
    
//     public function LReLU(float $alfa = 0.01) : Tensor
//     {
// 		$context = $this->initContextFrom();
// 		$inputId = $this->registerInContext($context, $this);
// 		
// 		$result = self::zeros($this->shape, 'LReLU');
// 		$context->registerOp('LReLU', [$inputId], $result);
// 		
// 		return $result;
//     }
    
    // Softmax activation
    public function softmax(int $axis = -1) : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		if (count($this->shape) === 1)
			$kernel = "SOFTMAX_1D_LAST";
		else if (count($this->shape) === 2 && $axis === -1)
			$kernel = "SOFTMAX_2D_LAST";
		else if (count($this->shape) === 3 && $axis === -1)
			$kernel = "SOFTMAX_3D_LAST";
		else
			$kernel = "SOFTMAX_GENERIC_AXIS";
		
		$attributes = array(
			"kernel"	=>	$kernel,
			"axes"		=>	array($axis),
		);
		
		$result = new Tensor($this->shape, [], 'softmax');
		$context->registerOp('softmax', [$inputId], $result, $attributes);
		
		return $result;
    }
    
    public function shapeReduced(int $index = 0) : array
	{
		$nSize = $this->shape;
		array_splice($nSize, $index, 1);
		
		return $nSize;
	}
    
//     // Mean Squared Error (MSE)
//     public function MSE() : Tensor
//     {
// 		$context = $this->initContextFrom();
// 		$inputId = $this->registerInContext($context, $this);
// 		
// 		$result = self::zeros($this->shapeReduced(), 'MSE');
// 		$context->registerOp('MSE', [$inputId], $result);
// 		
// 		return $result;
//     }
//     
//     // Mean Absolute Error (MAE)
//     public function MAE() : Tensor
//     {
// 		$context = $this->initContextFrom();
// 		$inputId = $this->registerInContext($context, $this);
// 		
// 		$result = self::zeros($this->shapeReduced(), 'MAE');
// 		$context->registerOp('MAE', [$inputId], $result);
// 		
// 		return $result;
//     }
    
    // Cross Entropy
    public function CE(Tensor $target) : Tensor
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		$result = new Tensor($this->shapeReduced(-1), [], 'CE');
		$context->registerOp('CE', [$logitsId, $targetId], $result);
		
		return $result;
    }
    
    public function CELogitsLabelInt(Tensor $target, int $axis = -1) : Tensor
    {
		$context = $this->initContextFrom($target);
		$logitsId = $this->registerInContext($context, $this);
		$targetId = $this->registerInContext($context, $target);
		
		if (count($this->shape) === 1)
			$kernel = "CE_LOGITS_LABEL_INT_1D_LAST";
		else if (count($this->shape) === 2 && $axis === -1)
			$kernel = "CE_LOGITS_LABEL_INT_2D_LAST";
		else if (count($this->shape) === 3 && $axis === -1)
			$kernel = "CE_LOGITS_LABEL_INT_3D_LAST";
		else
			$kernel = "CE_LOGITS_LABEL_INT_GENERIC_AXIS";
		
		$attributes = array(
			"kernel"	=>	$kernel,
			"axes"		=>	array($axis),
		);
		
		$result = new Tensor($this->shapeReduced($axis), [], 'CELogitsLabelInt');
		$context->registerOp('softmax_ce_logits_label_int', [$logitsId, $targetId], $result, $attributes);
		
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
		
		$result = new Tensor($this->shapeReduced(-1), [], 'CELogits');
		$context->registerOp('softmax_ce_logits', [$logitsId, $targetId], $result);
		
		return $result;
    }
	
	// Mean among batch samples
    public function mean(int $axis = 0) : Tensor
    {
		$context = $this->initContextFrom();
		$inputId = $this->registerInContext($context, $this);
		
		if (count($this->shape) === 1 && $axis === 0)
			$kernel = "MEAN_1D_FIRST";
		else if (count($this->shape) === 2 && $axis === 0)
			$kernel = "MEAN_2D_FIRST";
		else if (count($this->shape) === 3 && $axis === 0)
			$kernel = "MEAN_3D_FIRST";
		else
			$kernel = "MEAN_GENERIC_AXIS";
		
		$attributes = array(
			"kernel"	=>	$kernel,
			"axes"		=>	array($axis),
		);
		
		$result = new Tensor($this->shapeReduced($axis), [], 'mean');
		$context->registerOp('mean', [$inputId], $result, $attributes);
		
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
