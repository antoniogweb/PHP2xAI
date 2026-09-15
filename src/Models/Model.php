<?php

namespace PHP2xAI\Models;

use PHP2xAI\Runtime\PHP\Optimizers\Optimizer;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Runtime\PHP\Datasets\TrainValidateDataset;
use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;
use PHP2xAI\Graph\GraphContext;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;
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
	private string $provider = "";
	private string $modelSavePath = "./model.json";
	private string $configSavePath = "./config.json";
	private ?GraphRuntime $predictRuntime;
	private ?CoreFFI $cppRuntime = null;
	
	protected $p = [];
	/** @var array<string,array<string,Tensor>> */
	protected array $bertEncoderParameters = [];
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
	
	public function setProvider($provider = "")
	{
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
			'wq' => $this->createParam(Tensor::init([$d, $d], 0.05)),
			'wk' => $this->createParam(Tensor::init([$d, $d], 0.05)),
			'wv' => $this->createParam(Tensor::init([$d, $d], 0.05)),
			'bq' => $this->createParam(Tensor::zeros([$d])),
			'bk' => $this->createParam(Tensor::zeros([$d])),
			'bv' => $this->createParam(Tensor::zeros([$d])),
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

		if ($mask !== null && ($mask->getRank() !== 2
			|| $mask->shape[0] !== $batchSize
			|| $mask->shape[1] !== $sequenceLength))
			throw new RuntimeException("mask must have shape [B, L]");

		if ($mask === null)
		{
			$mask = Tensor::createFromData(
				array_fill(0, $batchSize, array_fill(0, $sequenceLength, 1.0))
			);
			$mask->setTrainable(false);
		}

		if ($parameterKey === null || !isset($this->bertEncoderParameters[$parameterKey]))
			throw new RuntimeException("BERT encoder parameters must be initialized in the model constructor");

		$params = $this->bertEncoderParameters[$parameterKey];
		$wq = $params['wq'];
		$wk = $params['wk'];
		$wv = $params['wv'];
		$bq = $params['bq'];
		$bk = $params['bk'];
		$bv = $params['bv'];
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

		$q = $x->matMul($wq)->add($bq);
		$k = $x->matMul($wk)->add($bk);
		$v = $x->matMul($wv)->add($bv);

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
	 * Multi-head attention mechanism.
	 *
	 * @param Tensor $Q Query tensor [B, L, D]
	 * @param Tensor $K Key tensor [B, L, D]
	 * @param Tensor $V Value tensor [B, L, D]
	 * @param Tensor $mask Padding mask [B, L] with 1 for valid tokens, 0 for padding
	 * @param int $numHeads Number of attention heads
	 * @param string $maskType Mask type: "PADDING" (CASUAL not supported yet)
	 * @return Tensor Attention output with shape [B, L, D]
	 *
	 * Flow:
	 *   Q, K, V [B, L, D]
	 *   → reshape [B, L, H, dk] where dk = D / numHeads
	 *   → transpose axes [1,2] [B, H, L, dk]
	 *   → Qh, Kh, Vh [B, H, L, dk]
	 *
	 *   Qh @ Khᵀ / sqrt(dk) → [B, H, L, L]
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
		Tensor $mask, 
		int $numHeads, 
		string $maskType = "PADDING"
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

		if ($mask->getRank() !== 2)
			throw new RuntimeException("Mask must have rank 2 [B, L]");

		if ($numHeads <= 0)
			throw new RuntimeException("numHeads must be positive");

		// Get D (last dimension) and validate divisibility
		$D = $Q->shape[2];
		if ($D % $numHeads !== 0)
			throw new RuntimeException("Dimension D ({$D}) must be divisible by numHeads ({$numHeads})");

		$dk = intdiv($D, $numHeads);

		// For now, only PADDING mask is supported
		if ($maskType !== "PADDING")
			throw new RuntimeException("Mask type '{$maskType}' not supported yet. Only 'PADDING' is available.");

		// Split heads: reshape [B, L, D] -> [B, L, H, dk]
		$Q_reshaped = $Q->reshape([$Q->shape[0], $Q->shape[1], $numHeads, $dk]);
		$K_reshaped = $K->reshape([$K->shape[0], $K->shape[1], $numHeads, $dk]);
		$V_reshaped = $V->reshape([$V->shape[0], $V->shape[1], $numHeads, $dk]);

		// Transpose: [B, L, H, dk] -> [B, H, L, dk]
		$Qh = $Q_reshaped->transpose([1, 2]);
		$Kh = $K_reshaped->transpose([1, 2]);
		$Vh = $V_reshaped->transpose([1, 2]);

		// Attention: Qh @ Khᵀ / sqrt(dk)
		$Kh_transposed = $Kh->transpose([-2, -1]);
		$scores = $Qh->matMul($Kh_transposed);

		// Scale by sqrt(dk)
		$scaledScores = $scores->scale(1.0 / sqrt($dk));

		// Apply mask
		$maskedScores = $scaledScores->applyPaddingMask($mask);

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
			"epochs_number"	=>	$epochsNumber,
			"batch_size"	=>	$dataset->train->getBatchSize(),
			"save_Path"		=>	$savePath ? $savePath : "",
			"log_on_each_x_batch"	=>	$logOnEachXBatch,
		);

		if ($profilerOutputPath !== null && $profilerOutputPath !== '')
			$jsonConfig['profiler_output_path'] = $profilerOutputPath;
		
		return json_encode($jsonConfig);
	}
	
	public function exportModel(StreamFileDataset $dataset) : string
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

			$this->cppRuntime = new CoreFFI($this->provider ?: "NAIVE", $modelPath, $weightsPath, $soPath);
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
		if ($this->provider == "EIGEN")
			$bin = realpath(__DIR__ . "/../Runtime/CPP/Bin/".$platform."/php2xai_runtime_eigen");
		else
			$bin = realpath(__DIR__ . "/../Runtime/CPP/Bin/".$platform."/php2xai_runtime");
		
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
	
	public function validationLoss(StreamFileDataset $dataset, GraphRuntime $graph)
	{
		$loss = 0;
		$count = 0;
		
		$dataset->resetEpoch();
		$graph->setTraining(false);
		
		while ($dataset->nextBatch())
		{
			[$x, $y] = $dataset->pack();
			
			$graph->setInput($x);
			$graph->setTarget($y);
			
			$graph->forward();
			
			$loss += $graph->getError();
			
			$count++;
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
			$this->trainCpp();
			return;
		}
		
		$graphDef = $this->generateGraph($dataset->train);
		
		$graph = new GraphRuntime($graphDef);
		$graph->setTraining(true);
		
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
	
	public function generateModel(StreamFileDataset $dataset) : array
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
	
	public function generateGraph(StreamFileDataset $dataset) : array
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
