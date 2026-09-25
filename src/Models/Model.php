<?php

namespace PHP2xAI\Models;

use PHP2xAI\Runtime\PHP\Optimizers\Optimizer;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Datasets\TrainValidateDataset;
use PHP2xAI\Runtime\PHP\Datasets\BatchDataset;
use PHP2xAI\Runtime\PHP\Datasets\HDF5Dataset;
use PHP2xAI\Graph\GraphContext;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
use PHP2xAI\Runtime\PHP\Core\ExecutionMode;
use PHP2xAI\Runtime\CPP\CoreFFI;
use PHP2xAI\Utility\Utility;

use RuntimeException;

/**
 * PHP
 *
 * Class to manage batch
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
abstract class Model
{
    private int $inputId;    // id of input tensor in GraphDef
    private int $targetId;   // id of target tensor in GraphDef
	private string $runtime = "PHP";
	private string $provider = "NAIVE";
	private string $modelSavePath = "./model.json";
	private string $configSavePath = "./config.json";
	private ?GraphRuntime $predictRuntime;
	private ?CoreFFI $cppRuntime = null;
	
	protected $p = [];
	/** @var array<string,array<string,Tensor>> */
	protected array $bertEncoderParameters = [];
	/** @var array<int,array<string,Tensor>> Parameters indexed by LLaMA decoder layer. */
	protected array $llamaDecoderParameters = [];
	public Optimizer $optimizer;
	
	// abstract public function forward(Tensor $x) : Tensor;
	
	abstract public function output(Tensor $x) : Tensor;
	abstract public function loss(Tensor $x, Tensor $y) : Tensor;
	
	public function __construct(?Optimizer $optimizer = null)
	{
		if (isset($optimizer))
			$this->optimizer = $optimizer;
		
		// $this->optimizer->addTensors(array_values($this->p));
	}
	
	public function __set(string $name, Tensor $value)
    {
		if ($value instanceof Tensor && $value->getName() === null)
			$value->setName($name);
		
        $this->p[$name] = $value;
    }
    
    public function __get(string $name) : Tensor
    {
        return $this->p[$name] ?? null;
    }
    
	public function createParam(Tensor $tensor) : Tensor
	{
		do
		{
			$name = "param_".bin2hex(random_bytes(8));
		}
		while (array_key_exists($name, $this->p));

		$tensor->setName($name);
		$this->__set($name, $tensor);


		return $tensor;
	}

	public function setRuntime($runtime = "CPP")
	{
		$this->runtime = $runtime;
	}

	public function setProvider(string $provider = "NAIVE")
	{
		$provider = strtoupper($provider);
		if ($provider !== "NAIVE" && $provider !== "EIGEN")
			throw new RuntimeException("Unsupported provider: ".$provider);

		$this->provider = $provider;
	}
	
	public function setModelSavePath($modelSavePath = "./model.json")
	{
		$this->modelSavePath = $modelSavePath;
	}
    
    public function getParameters()
    {
		return $this->p;
    }
    
	// change the parameters
	public function step(GraphRuntime $graph)
	{
		return $this->optimizer->step($graph);
	}
	
	public function exportGrapf(TrainValidateDataset $dataset = null)
	{
		return $this->generateGraph($dataset->train);
	}

	/**
	 * Creates a reusable BERT encoder parameter set. Invoke this from a
	 * concrete model constructor, before the framework exports its graph.
	 */
	protected function initializeBertEncoder(string $key, int $d, int $dff) : void
	{
		if (isset($this->bertEncoderParameters[$key]))
			return;

		if ($dff <= 0)
			throw new RuntimeException("dff must be positive");

		$this->bertEncoderParameters[$key] = [
			'wqkv' => $this->createParam(Tensor::init([$d, 3 * $d], 0.05)),
			'bqkv' => $this->createParam(Tensor::zeros([3 * $d])),
			'wo' => $this->createParam(Tensor::init([$d, $d], 0.05)),
			'bo' => $this->createParam(Tensor::zeros([$d])),
			'w1' => $this->createParam(Tensor::init([$d, $dff], 0.05)),
			'b1' => $this->createParam(Tensor::zeros([$dff])),
			'w2' => $this->createParam(Tensor::init([$dff, $d], 0.05)),
			'b2' => $this->createParam(Tensor::zeros([$d])),
			'gamma1' => $this->createParam(Tensor::createFromData(array_fill(0, $d, 1.0))),
			'beta1' => $this->createParam(Tensor::zeros([$d])),
			'gamma2' => $this->createParam(Tensor::createFromData(array_fill(0, $d, 1.0))),
			'beta2' => $this->createParam(Tensor::zeros([$d])),
		];
	}

	/**
	 * Creates the parameters for one decoder layer before graph construction.
	 *
	 * Invoke this once per layer from the concrete LLaMA model constructor:
	 *
	 *   for ($i = 0; $i < $numLayers; $i++)
	 *       $this->initializeLlamaDecoderParameters($i, $hiddenDim, $ffnDim);
	 *
	 * Parameters must exist before generateGraph()/generateModel() registers
	 * them in GraphContext; they cannot be created lazily by llamaDecoder().
	 */
	protected function initializeLlamaDecoderParameters(int $layer, int $hiddenDim, int $ffnDim): void
	{
		if ($layer < 0)
			throw new RuntimeException('LLaMA decoder layer must be >= 0');
		if ($hiddenDim <= 0 || $ffnDim <= 0)
			throw new RuntimeException('LLaMA hiddenDim and ffnDim must be positive');
		if (isset($this->llamaDecoderParameters[$layer]))
			return;

		$this->llamaDecoderParameters[$layer] = [
			// Standard dense attention projections. GQA/MQA can later replace
			// wk/wv with matrices having fewer output head dimensions.
			'wq' => $this->createParam(Tensor::init([$hiddenDim, $hiddenDim], 0.05)),
			'wk' => $this->createParam(Tensor::init([$hiddenDim, $hiddenDim], 0.05)),
			'wv' => $this->createParam(Tensor::init([$hiddenDim, $hiddenDim], 0.05)),
			'wo' => $this->createParam(Tensor::init([$hiddenDim, $hiddenDim], 0.05)),

			// SwiGLU has two parallel expansions and one down projection.
			'w_gate' => $this->createParam(Tensor::init([$hiddenDim, $ffnDim], 0.05)),
			'w_up' => $this->createParam(Tensor::init([$hiddenDim, $ffnDim], 0.05)),
			'w_down' => $this->createParam(Tensor::init([$ffnDim, $hiddenDim], 0.05)),

			// RMSNorm has only a learned scale parameter, no additive beta.
			'attention_gamma' => $this->createParam(Tensor::createFromData(array_fill(0, $hiddenDim, 1.0))),
			'ffn_gamma' => $this->createParam(Tensor::createFromData(array_fill(0, $hiddenDim, 1.0))),
		];
	}

	/**
	 * Builds one pre-norm LLaMA-style decoder block.
	 *
	 * A concrete model can construct a stack as follows:
	 *
	 *   for ($i = 0; $i < $numLayers; $i++)
	 *       $x = $this->llamaDecoder($x, $i, $numHeads, $mask, ...);
	 *
	 * The layer index selects the weights and is also passed to kvCache, so
	 * every decoder layer owns an independent key/value cache slot.
	 *
	 * @param Tensor $x Input [B, L, hiddenDim]
	 * @param int $layer Decoder-layer index initialized in the constructor
	 * @param int $numHeads Number of query/key/value heads
	 * @param ?Tensor $mask Optional padding mask [B, L]. When omitted only
	 *                      causal masking is used, which is useful for decode.
	 * @param int $ropeOffset Initial absolute position for a prefill request
	 * @param float $ropeBase RoPE frequency base
	 * @param string $ropePairing Tensor::INTERLEAVED or Tensor::ROTATE_HALF
	 */
	public function llamaDecoder(
		Tensor $x,
		int $layer,
		int $numHeads,
		?Tensor $mask = null,
		int $ropeOffset = 0,
		float $ropeBase = 10000.0,
		string $ropePairing = Tensor::INTERLEAVED
	): Tensor
	{
		if ($x->getRank() !== 3)
			throw new RuntimeException('llamaDecoder expects x with shape [B, L, hiddenDim]');
		if ($layer < 0 || !isset($this->llamaDecoderParameters[$layer]))
			throw new RuntimeException('LLaMA decoder parameters must be initialized in the model constructor');

		[$batchSize, $sequenceLength, $hiddenDim] = $x->getShape();
		if ($numHeads <= 0 || $hiddenDim % $numHeads !== 0)
			throw new RuntimeException('LLaMA hiddenDim must be divisible by numHeads');

		if ($mask !== null && ($mask->getRank() !== 2
			|| $mask->shape[0] !== $batchSize
			|| $mask->shape[1] !== $sequenceLength))
			throw new RuntimeException('LLaMA padding mask must have shape [B, L]');

		$params = $this->llamaDecoderParameters[$layer];

		// LLaMA pre-attention RMS normalization.
		$attentionInput = $x->rmsNorm($params['attention_gamma']);

		// Separate dense Q/K/V projections. Keeping them distinct makes the
		// dataflow clear; a fused QKV projection can be added as an optimization.
		$q = $attentionInput->matMul($params['wq']);
		$k = $attentionInput->matMul($params['wk']);
		$v = $attentionInput->matMul($params['wv']);

		// Causal attention is mandatory for a decoder. Padding is composed with
		// it only when a mask is supplied. RoPE rotates Q/K and kvCache stores
		// the already-rotated K plus V for this specific layer.
		$maskType = $mask === null ? 'CAUSAL' : 'PADDING+CAUSAL';
		$attention = self::attention(
			$q,
			$k,
			$v,
			$mask,
			$numHeads,
			$maskType,
			['offset' => $ropeOffset, 'base' => $ropeBase, 'pairing' => $ropePairing],
			$layer
		);

		// Attention output projection followed by the first residual connection.
		$x = $x->add($attention->matMul($params['wo']));

		// Second pre-norm branch: SwiGLU expansion, gate, and down projection.
		$ffnInput = $x->rmsNorm($params['ffn_gamma']);
		$ffnOutput = self::swiGLU(
			$ffnInput,
			$params['w_gate'],
			$params['w_up'],
			$params['w_down']
		);

		// Feed-forward residual. A final model-level norm belongs outside this
		// per-layer block and can be added by the concrete LLaMA model.
		return $x->add($ffnOutput);
	}

	/**
	 * Builds one post-LayerNorm BERT encoder block for an input [B, L, D].
	 */
	public function bertEncoder(
		Tensor $x,
		int $numHeads,
		int $dff,
		?Tensor $mask = null,
		float $dropout = 0.1,
		?string $parameterKey = null
	) : Tensor
	{
		if ($x->getRank() !== 3)
			throw new RuntimeException("bertEncoder expects x with shape [B, L, D]");

		[$batchSize, $sequenceLength, $d] = $x->getShape();

		if ($numHeads <= 0 || $d % $numHeads !== 0)
			throw new RuntimeException("Embedding dimension D must be divisible by numHeads");

		if ($dff <= 0)
			throw new RuntimeException("dff must be positive");

		if ($dropout < 0.0 || $dropout > 1.0)
			throw new RuntimeException("dropout must be a probability between 0 and 1");

		$headDim = intdiv($d, $numHeads);

		if ($mask === null)
			throw new RuntimeException("bertEncoder requires a padding mask with shape [B, L]");

		if ($mask->getRank() !== 2
			|| $mask->shape[0] !== $batchSize
			|| $mask->shape[1] !== $sequenceLength)
			throw new RuntimeException("mask must have shape [B, L]");

		if ($parameterKey === null || !isset($this->bertEncoderParameters[$parameterKey]))
			throw new RuntimeException("BERT encoder parameters must be initialized in the model constructor");

		$params = $this->bertEncoderParameters[$parameterKey];
		$wqkv = $params['wqkv'];
		$bqkv = $params['bqkv'];
		$wo = $params['wo'];
		$bo = $params['bo'];
		$w1 = $params['w1'];
		$b1 = $params['b1'];
		$w2 = $params['w2'];
		$b2 = $params['b2'];
		$gamma1 = $params['gamma1'];
		$beta1 = $params['beta1'];
		$gamma2 = $params['gamma2'];
		$beta2 = $params['beta2'];

		// One [B, L, 3D] projection, then three contiguous last-axis slices.
		$qkv = $x->matMul($wqkv)->add($bqkv);
		$q = $qkv->slice(0, $d);
		$k = $qkv->slice($d, 2 * $d);
		$v = $qkv->slice(2 * $d, 3 * $d);

		$attention = self::attention($q, $k, $v, $mask, $numHeads, "PADDING");
		$attentionOutput = $attention->matMul($wo)->add($bo);
		if ($dropout > 0.0)
			$attentionOutput = $attentionOutput->dropout($dropout * 100.0);

		$y = $x->add($attentionOutput)->layerNorm($gamma1, $beta1);

		$feedForward = $y->matMul($w1)->add($b1)->gelu()->matMul($w2)->add($b2);
		if ($dropout > 0.0)
			$feedForward = $feedForward->dropout($dropout * 100.0);

		return $y->add($feedForward)->layerNorm($gamma2, $beta2);
	}

	/**
	 * SwiGLU feed-forward block.
	 *
	 * For an input x[..., D], the gate and up projections both produce
	 * [..., Dff]. SiLU activates only the gate projection; the resulting
	 * values modulate the up projection element by element. The down
	 * projection maps the hidden dimension back to the requested output one.
	 *
	 *   gate   = x @ wGate                 // [..., Dff]
	 *   up     = x @ wUp                   // [..., Dff]
	 *   hidden = silu(gate) * up           // [..., Dff]
	 *   out    = hidden @ wDown            // [..., Dout]
	 *
	 * Bias terms are deliberately not included here: callers can add them to
	 * each projection when their model architecture requires them.
	 *
	 * @param Tensor $x Input tensor with last dimension D
	 * @param Tensor $wGate Gate projection weights [D, Dff]
	 * @param Tensor $wUp Up projection weights [D, Dff]
	 * @param Tensor $wDown Down projection weights [Dff, Dout]
	 * 
	 *                      ┌─ Wgate ─→ gate ─→ SiLU ─┐
	 * X [B,L,D] ───────────┤                         × ─→ Wdown ─→ [B,L,D]
     *                      └─ Wup   ─→ up ───────────┘
	 */
	public static function swiGLU(Tensor $x, Tensor $wGate, Tensor $wUp, Tensor $wDown): Tensor
	{
		if ($x->getRank() < 1)
			throw new RuntimeException('swiGLU expects x to have at least one dimension');

		if ($wGate->getRank() !== 2 || $wUp->getRank() !== 2 || $wDown->getRank() !== 2)
			throw new RuntimeException('swiGLU weights must be rank-2 matrices');

		$xShape = $x->getShape();
		$inputDim = $xShape[count($xShape) - 1];
		[$gateInputDim, $hiddenDim] = $wGate->getShape();
		[$upInputDim, $upHiddenDim] = $wUp->getShape();
		[$downInputDim] = $wDown->getShape();

		// Both parallel projections must consume x's last dimension.
		if ($gateInputDim !== $inputDim || $upInputDim !== $inputDim)
			throw new RuntimeException('swiGLU gate and up weights must match x last dimension');

		// multiply() intentionally has no broadcast support, so gate and up
		// must produce precisely the same hidden shape.
		if ($hiddenDim !== $upHiddenDim)
			throw new RuntimeException('swiGLU gate and up weights must have the same hidden dimension');

		// The final projection starts from the gated hidden representation.
		if ($downInputDim !== $hiddenDim)
			throw new RuntimeException('swiGLU down weight input dimension must match hidden dimension');

		// Gate branch: SiLU controls how much of each up-projection feature passes.
		$gate = $x->matMul($wGate);
		$up = $x->matMul($wUp);
		$hidden = $gate->silu()->multiply($up);

		// Down-project the gated hidden representation to the model/output dimension.
		return $hidden->matMul($wDown);
	}
	
	/**
	 * Multi-head attention mechanism.
	 *
	 * @param Tensor $Q Query tensor [B, L, D]
	 * @param Tensor $K Key tensor [B, L, D]
	 * @param Tensor $V Value tensor [B, L, D]
	 * @param ?Tensor $mask Optional padding mask [B, Lkv] with 1 for valid tokens, 0 for padding
	 * @param int $numHeads Number of attention heads
	 * @param string $maskType One or more mask types separated by "+", for
	 *                         example "PADDING+CAUSAL". Accepted types are
	 *                         "PADDING" and "CAUSAL".
	 * @return Tensor Attention output with shape [B, L, D]
	 *
	 * Flow:
	 *   Q, K, V [B, L, D]
	 *   → reshape [B, L, H, dk] where dk = D / numHeads
	 *   → transpose axes [1,2] [B, H, L, dk]
	 *   → Qh, Kh, Vh [B, H, L, dk]
	 *
	 *   (Qh @ Khᵀ / sqrt(dk)) + Mask → [B, H, L, L]
	 *   softmax → [B, H, L, L]
	 *   × Vh → [B, H, L, dk]
	 *
	 *   → transpose axes [1,2] [B, L, H, dk]
	 *   → reshape [B, L, D]
	 */
	public static function attention(
		Tensor $Q, 
		Tensor $K, 
		Tensor $V, 
		?Tensor $mask,
		int $numHeads, 
		string $maskType = "PADDING",
		?array $rope = null,
		?int $kvCacheLayer = null
	) : Tensor
	{
		// Validate inputs
		$qRank = $Q->getRank();
		$kRank = $K->getRank();
		$vRank = $V->getRank();

		if ($qRank !== $kRank || $kRank !== $vRank)
			throw new RuntimeException("Q, K, V must have the same rank");

		if ($qRank !== 3)
			throw new RuntimeException("Q, K, V must have rank 3 [B, L, D]");

		if ($numHeads <= 0)
			throw new RuntimeException("numHeads must be positive");

		// Q may be shorter than K/V during decode, but batch, K/V length and embedding dimension must agree.
		[$batch, $Lq, $D] = $Q->shape;
		[$kBatch, $Lkv, $kDim] = $K->shape;
		[$vBatch, $vLength, $vDim] = $V->shape;
		if ($batch !== $kBatch || $batch !== $vBatch || $Lkv !== $vLength || $D !== $kDim || $D !== $vDim)
			throw new RuntimeException("Q, K, V dimensions mismatch");

		if ($D % $numHeads !== 0)
			throw new RuntimeException("Dimension D ({$D}) must be divisible by numHeads ({$numHeads})");

		if ($mask !== null && ($mask->getRank() !== 2 || $mask->shape[0] !== $batch || $mask->shape[1] !== $Lkv))
			throw new RuntimeException("Mask must have shape [B, Lkv]");

		$dk = intdiv($D, $numHeads);

		// A compound mask such as PADDING+CAUSAL is represented as a sequence
		// of masking ops in the graph. Both masks write -INF to masked scores,
		// so their effects compose before softmax.
		$maskTypes = array_map('trim', explode('+', strtoupper($maskType)));
		if ($maskTypes === [] || in_array('', $maskTypes, true))
			throw new RuntimeException('Mask type must contain PADDING, CAUSAL, or both separated by +');

		foreach ($maskTypes as &$currentMaskType)
		{
			if ($currentMaskType !== 'PADDING' && $currentMaskType !== 'CAUSAL')
				throw new RuntimeException("Mask type {$currentMaskType} is not supported. Use PADDING, CAUSAL, or PADDING+CAUSAL.");
		}
		unset($currentMaskType);

		if ($kvCacheLayer !== null && $kvCacheLayer < 0)
			throw new RuntimeException("KV cache layer must be >= 0");
		if ($kvCacheLayer !== null && !in_array('CAUSAL', $maskTypes, true))
			throw new RuntimeException("KV cache requires causal attention");
		if ($rope !== null)
		{
			if (!isset($rope['offset'], $rope['base'], $rope['pairing']))
				throw new RuntimeException("RoPE requires offset, base and pairing");
			if (!is_int($rope['offset']) || !is_numeric($rope['base']) || !is_string($rope['pairing']))
				throw new RuntimeException("Invalid RoPE configuration");
		}

		// Split heads: reshape [B, L, D] -> [B, L, H, dk]
		$Q_reshaped = $Q->reshape([$Q->shape[0], $Q->shape[1], $numHeads, $dk]);
		$K_reshaped = $K->reshape([$K->shape[0], $K->shape[1], $numHeads, $dk]);
		$V_reshaped = $V->reshape([$V->shape[0], $V->shape[1], $numHeads, $dk]);

		// Transpose: [B, L, H, dk] -> [B, H, L, dk]
		$Qh = $Q_reshaped->transpose([1, 2]);
		$Kh = $K_reshaped->transpose([1, 2]);
		$Vh = $V_reshaped->transpose([1, 2]);

		// RoPE rotates queries and keys, never values, in the [B, H, L, Dk] layout.
		if ($rope !== null)
		{
			$Qh = $Qh->rope(-2, -1, $rope['offset'], (float)$rope['base'], $rope['pairing']);
			$Kh = $Kh->rope(-2, -1, $rope['offset'], (float)$rope['base'], $rope['pairing']);
		}

		// Decoder-only: cache already rotated keys and their paired values.
		if ($kvCacheLayer !== null)
			[$Kh, $Vh] = Tensor::kvCache($Kh, $Vh, $kvCacheLayer);

		// Attention: Qh @ Khᵀ / sqrt(dk)
		$Kh_transposed = $Kh->transpose([-2, -1]);
		$scores = $Qh->matMul($Kh_transposed);

		// Scale by sqrt(dk)
		$scaledScores = $scores->scale(1.0 / sqrt($dk));

		$maskedScores = $scaledScores;
		
		// Apply every requested mask in the order given by maskType. For
		// PADDING+CAUSAL, scores invalid because of either condition are -INF.
		foreach ($maskTypes as $currentMaskType)
		{
			if ($currentMaskType === 'PADDING' && $mask !== null)
				$maskedScores = $maskedScores->applyPaddingMask($mask);
			else if ($currentMaskType === 'CAUSAL')
				$maskedScores = $maskedScores->applyCausalMask();
		}

		// Softmax
		$attentionWeights = $maskedScores->softmax();

		// Weighted sum of V: attentionWeights @ Vh
		$attentionOutput = $attentionWeights->matMul($Vh);

		// Merge heads: transpose [B, H, L, dk] -> [B, L, H, dk]
		$merged = $attentionOutput->transpose([1, 2]);

		// Reshape [B, L, H, dk] -> [B, L, D]
		$output = $merged->reshape([$Q->shape[0], $Q->shape[1], $D]);

		return $output;
	}
	
	public function getTrainingConfig(TrainValidateDataset $dataset = null, int $epochsNumber = 10, string $savePath = null, int $logOnEachXBatch = 10, ?string $profilerOutputPath = null) : string
	{
		$graph = $this->generateGraph($dataset->train);
		
		$jsonConfig = array(
			"graph"	=>	$graph,
			"optimizer"	=>	$this->optimizer->getConfig(),
			"train_data_file"	=>	$dataset->train->getPath(),
			"val_data_file"	=>	$dataset->val->getPath(),
			"dataset_type"	=>	$dataset->train->getType(),
			"epochs_number"	=>	$epochsNumber,
			"batch_size"	=>	$dataset->train->getBatchSize(),
			"save_Path"		=>	$savePath ? $savePath : "",
			"log_on_each_x_batch"	=>	$logOnEachXBatch,
		);

		if ($profilerOutputPath !== null && $profilerOutputPath !== '')
			$jsonConfig['profiler_output_path'] = $profilerOutputPath;
		
		return json_encode($jsonConfig);
	}
	
	public function exportModel(BatchDataset $dataset) : string
	{
		$graph = $this->generateModel($dataset);
		
		$modelConfig = array(
			"graph"	=>	$graph,
		);
		
		return json_encode($modelConfig);
	}
	
	// Load the model
	public function loadModel(string $modelPath, string $weightsPath)
	{
		if (!is_file($modelPath))
			throw new \RuntimeException("model path does not exist");

		if (!is_file($weightsPath))
			throw new \RuntimeException("weights path does not exist");

		if ($this->runtime == "CPP")
		{
			$platform = Utility::getPlatform();
			$soPath = realpath(__DIR__ . "/../Runtime/CPP/Bin/".$platform."/php2xai_runtime.so");

			if ($soPath === false)
				throw new \RuntimeException("CPP runtime library not found");

			$this->cppRuntime = new CoreFFI($this->provider, $modelPath, $weightsPath, $soPath);
			$this->predictRuntime = null;
			return;
		}
		
		$modelJson = file_get_contents($modelPath);
		$weightsJson = file_get_contents($weightsPath);
		
		$modelDef = json_decode($modelJson, true);
		$weights = json_decode($weightsJson, true);
		
		$this->predictRuntime = new GraphRuntime($modelDef["graph"], $weights);
	}
	
	public function predict(array $x) : int|array
	{
		if ($this->runtime == "CPP")
		{
			if (!isset($this->cppRuntime))
				throw new \RuntimeException("model not loaded");
			
			return $this->cppRuntime->predict($x);
		}
		
		if (!isset($this->predictRuntime))
			throw new \RuntimeException("model not loaded");
		
		$this->predictRuntime->setInput($x);
		
		$this->predictRuntime->forward();
		
		return $this->predictRuntime->getOutput();
	}
	
	public function predictLabelInt(array $x) : int|array
	{
		if ($this->runtime == "CPP")
		{
			if (!isset($this->cppRuntime))
				throw new \RuntimeException("model not loaded");
			
			return $this->cppRuntime->predictLabelInt($x);
		}
		
		$output = $this->predict($x);
		
		if (is_array($output))
			return Utility::argmax($output);
		
		return $output;
	}
	
	public function trainCpp()
	{
		if (!function_exists('proc_open'))
		{
			fwrite(STDERR, "proc_open is not enabled..\n");
			exit(2);
		}
		
		// run_train.php
		$platform = Utility::getPlatform();
		$binaryName = $this->provider === "EIGEN" ? "php2xai_runtime_eigen" : "php2xai_runtime";
		$bin = realpath(__DIR__ . "/../Runtime/CPP/Bin/".$platform."/".$binaryName);
		
		$jsonPath = realpath($this->configSavePath);
		
		// echo realpath($bin);die();
		
		$cmd = [$bin, $jsonPath];

		$spec = [
			0 => ['pipe', 'r'], // STDIN
			1 => ['pipe', 'w'], // STDOUT
			2 => ['pipe', 'w'], // STDERR
		];

		// bypass_shell evita escaping/injection e problemi quoting
		$proc = proc_open($cmd, $spec, $pipes, null, null, [
			'bypass_shell' => true,
		]);

		if (!is_resource($proc))
		{
			fwrite(STDERR, "Impossibile avviare il processo\n");
			exit(3);
		}

		fclose($pipes[0]);

		// NON bloccare: leggiamo a pezzi
		stream_set_blocking($pipes[1], false);
		stream_set_blocking($pipes[2], false);

		$stdoutBuf = '';
		$stderrBuf = '';

		while (true)
		{
			$status = proc_get_status($proc);
			$running = $status['running'];

			$out = stream_get_contents($pipes[1]);
			if ($out !== false && $out !== '')
			{
				$stdoutBuf .= $out;
				echo $out;               // live stdout
				fflush(STDOUT);
			}

			$err = stream_get_contents($pipes[2]);
			if ($err !== false && $err !== '')
			{
				$stderrBuf .= $err;
				fwrite(STDERR, $err);    // live stderr
				fflush(STDERR);
			}

			if (!$running) break;

			usleep(50_000); // 50ms
		}

		// chiudi pipe
		fclose($pipes[1]);
		fclose($pipes[2]);

		$exitCode = proc_close($proc);

		if ($exitCode !== 0)
		{
			fwrite(STDERR, "\nTrain ended with exit code $exitCode\n");
			// to do: log $stderrBuf
		}

		exit($exitCode);
	}
	
	public function validationLoss(BatchDataset $dataset, GraphRuntime $graph)
	{
		$loss = 0;
		$count = 0;
		
		$dataset->resetEpoch();
		$graph->setMode(ExecutionMode::INFER);

		try
		{
			while ($dataset->nextBatch())
			{
				[$x, $y] = $dataset->pack();
				
				$graph->setInput($x);
				$graph->setTarget($y);
				
				$graph->forward();
				
				$loss += $graph->getError();
				
				$count++;
			}
		}
		finally
		{
			// The training loop reuses this runtime for the next epoch.
			$graph->setMode(ExecutionMode::TRAIN);
		}
		
		if ($count > 0)
			return $loss / $count;
		else
			return 0;
	}
	
	public function train(TrainValidateDataset $dataset = null, int $epochsNumber = 10, string $savePath = null, int $logOnEachXBatch = 10, ?string $profilerOutputPath = null)
	{
		// Save the model JSON graph
		file_put_contents($this->modelSavePath, $this->exportModel($dataset->train), LOCK_EX);
		
		if ($this->runtime == "CPP")
		{
			$config = $this->getTrainingConfig($dataset, $epochsNumber, $savePath, $logOnEachXBatch, $profilerOutputPath);
			file_put_contents($this->configSavePath, $config, LOCK_EX);
			if ($dataset->train instanceof HDF5Dataset)
				$dataset->train->close();
			if ($dataset->val instanceof HDF5Dataset)
				$dataset->val->close();
			$this->trainCpp();
			return;
		}
		
		$graphDef = $this->generateGraph($dataset->train);
		
		$graph = new GraphRuntime($graphDef);
		$graph->setMode(ExecutionMode::TRAIN);
		
		$betterValidationLoss = 99999999;
		
		for ($i=0; $i<$epochsNumber; $i++)
		{
			echo "Epoch ".($i+1)."\n";
			echo "------------------------\n";
			
			$indice = 0;
			
			$dataset->train->resetEpoch(); // reset batch cursor
			$dataset->train->shuffleEpoch(); // shuffle dei batch
			
			while ($dataset->train->nextBatch())
			{
				$graph->resetGrad();
				$graph->setLossGrad(1.0);
				
				[$x, $y] = $dataset->train->pack();
				
				$graph->setInput($x);
				$graph->setTarget($y);
				
				$graph->forward();
				
				$error = $graph->getError();
				
				$graph->backward();
				
				$this->step($graph);
				
				$indice++;
				
				if (($indice % $logOnEachXBatch) === 0)
					echo "Train error batch $indice: ".$error."\n";
			}
			
			$validationLoss = $this->validationLoss($dataset->val, $graph);
			
			echo "------------------------\n";
			echo "Validation error: ".$validationLoss."\n";
			
			if ($validationLoss < $betterValidationLoss && $savePath)
			{
				$betterValidationLoss = $validationLoss;
				
				$graph->saveWeightsToJson($savePath);
				// $graph->saveToJson($savePath);
			}
			else
			{
				echo "------------------------\n";
				echo "Validation error increased\n";
			}
			
			echo "------------------------\n";
		}
	}
	
	public function generateModel(BatchDataset $dataset) : array
	{
		$dataset->initPlaceholders(false);
		$placeholders = $dataset->getPlaceholders();
		
		$x = $placeholders['x'] ?? null;
		
		if ($x === null)
			throw new \RuntimeException("Input placeholders (x, y) are required to generate IR.");
		
		$context = new GraphContext();
		
		// --- create tensors
		$xId = $context->registerTensor($x, 'input', $x->getName(), $x->getShape());
		
		foreach ($this->p as $name => $tensor)
		{
			if (!($tensor instanceof Tensor))
				continue;
			
			if ($tensor->getName() === null)
				$tensor->setName($name);
			
			$context->registerTensor($tensor, 'param', $tensor->getName(), $tensor->getShape());
		}
		
		// --- create ops
		$x->setContext($context);
		$output = $this->output($x);
		
		$graph = $context->export();
		
		$outputId = $context->getTensorId($output);
		
		$graph['output'] = $outputId;
		
		return $graph;
	}
	
	public function generateGraph(BatchDataset $dataset) : array
	{
		$dataset->initPlaceholders();
		$placeholders = $dataset->getPlaceholders();
		
		$x = $placeholders['x'] ?? null;
		$y = $placeholders['y'] ?? null;
		
		if ($x === null || $y === null)
			throw new \RuntimeException("Input placeholders (x, y) are required to generate IR.");
		
		$context = new GraphContext();
		
		// --- create tensors
		$xId = $context->registerTensor($x, 'input', $x->getName(), $x->getShape());
		
		foreach ($this->p as $name => $tensor)
		{
			if (!($tensor instanceof Tensor))
				continue;
			
			if ($tensor->getName() === null)
				$tensor->setName($name);
			
			$context->registerTensor($tensor, 'param', $tensor->getName(), $tensor->getShape());
		}
		
		$yId = $context->registerTensor($y, 'target', $y->getName(), $y->getShape());
		
		// --- create ops
		$x->setContext($context);
		$y->setContext($context);
		$loss = $this->loss($x, $y);
		
		$graph = $context->export();
		
		$lossId = $context->getTensorId($loss);
		$lossIdx = $context->getTensorIndex($loss);
		
		$graph['tensors'][$lossIdx]['kind'] = 'loss';
		$graph['tensors'][$lossIdx]['name'] = $graph['tensors'][$lossIdx]['name'] ?? ($loss->getName() ?? 'loss');
		
		// --- add lossId
		$graph['loss'] = $lossId;
		
		return $graph;
	}
}
