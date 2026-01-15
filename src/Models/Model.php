<?php

namespace PHP2xAI\Models;

use PHP2xAI\Runtime\PHP\Optimizers\Optimizer;
use PHP2xAI\Tensor\Tensor;
use PHP2xAI\Tensor\Matrix;
use PHP2xAI\Tensor\Vector;
use PHP2xAI\Tensor\Scalar;
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
	private string $modelSavePath = "./model.json";
	private string $configSavePath = "./config.json";
	private ?GraphRuntime $predictRuntime;
	private ?CoreFFI $cppRuntime = null;
	
	protected $p = [];
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
	
	public function __set(string $name, Tensor|fTensor $value)
    {
		if ($value instanceof Tensor && $value->getName() === null)
			$value->setName($name);
		
        $this->p[$name] = $value;
    }
    
    public function __get(string $name) : Tensor|fTensor
    {
        return $this->p[$name] ?? null;
    }
    
    public function setRuntime($runtime = "CPP")
	{
		$this->runtime = $runtime;
	}
	
	public function setModelSavePath($modelSavePath = "./model.json")
	{
		$this->modelSavePath = $modelSavePath;
	}
    
    public function getParameters()
    {
		return $this->p;
    }
    
 //    public function addError(float $error)
	// {
	// 	$this->optimizer->addError($error);
	// }
    
	// change the parameters
	public function step(GraphRuntime $graph)
	{
		return $this->optimizer->step($graph);
	}
	
	public function exportGrapf(TrainValidateDataset $dataset = null)
	{
		return $this->generateGraph($dataset->train);
	}
	
	public function getTrainingConfig(TrainValidateDataset $dataset = null, int $epochsNumber = 10, string $savePath = null, int $logOnEachXBatch = 10) : string
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
			$soPath = realpath(__DIR__ . '/../Runtime/CPP/php2xai_runtime.so');
			
			if ($soPath === false)
				throw new \RuntimeException("CPP runtime library not found");
			
			$this->cppRuntime = new CoreFFI($modelPath, $weightsPath, $soPath);
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
		$bin = realpath(__DIR__ . '/../Runtime/CPP/php2xai_runtime');          // il tuo eseguibile C++
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
	
	public function train(TrainValidateDataset $dataset = null, int $epochsNumber = 10, string $savePath = null, int $logOnEachXBatch = 10)
	{
		// Save the model JSON graph
		file_put_contents($this->modelSavePath, $this->exportModel($dataset->train), LOCK_EX);
		
		if ($this->runtime == "CPP")
		{
			$config = $this->getTrainingConfig($dataset, $epochsNumber, $savePath, $logOnEachXBatch);
			file_put_contents($this->configSavePath, $config, LOCK_EX);
			$this->trainCpp();
			return;
		}
		
		$graphDef = $this->generateGraph($dataset->train);
		
		$graph = new GraphRuntime($graphDef);
		
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
		
		$trainableIds = [];
		
		// --- create tensors
		$xId = $context->registerTensor($x, 'input', $x->getName(), $x->getShape());
		
		foreach ($this->p as $name => $tensor)
		{
			if (!($tensor instanceof Tensor))
				continue;
			
			if ($tensor->getName() === null)
				$tensor->setName($name);
			
			$paramId = $context->registerTensor($tensor, 'param', $tensor->getName(), $tensor->getShape());
			
			$trainableIds[] = $paramId;
		}
		
		// --- create ops
		$x->setContext($context);
		
		$output = $this->output($x);
		
		$graph = $context->export();
		
		$outputId = $context->getTensorId($output);
		
		$graph['trainable'] = $trainableIds;
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
		
		$trainableIds = [];
		
		// --- create tensors
		$xId = $context->registerTensor($x, 'input', $x->getName(), $x->getShape());
		
		foreach ($this->p as $name => $tensor)
		{
			if (!($tensor instanceof Tensor))
				continue;
			
			if ($tensor->getName() === null)
				$tensor->setName($name);
			
			$paramId = $context->registerTensor($tensor, 'param', $tensor->getName(), $tensor->getShape());
			
			$trainableIds[] = $paramId;
		}
		
		$yId = $context->registerTensor($y, 'target', $y->getName(), $y->getShape());
		
		// --- create ops
		$x->setContext($context);
		$y->setContext($context);
		
		$loss = $this->loss($x, $y);
		
		$graph = $context->export();
		
		$lossId = $context->getTensorId($loss);
		$lossShape = $loss->getShape();
		
		$idToIndex = [];
		
		foreach ($graph['tensors'] as $idx => $tInfo)
		{
			$idToIndex[$tInfo['id']] = $idx;
		}
		
		$lossIdx = $idToIndex[$lossId];
		
		$graph['tensors'][$lossIdx]['kind'] = 'loss';
		$graph['tensors'][$lossIdx]['name'] = $graph['tensors'][$lossIdx]['name'] ?? ($loss->getName() ?? 'loss');
		$graph['tensors'][$lossIdx]['shape'] = $lossShape;
		
		// --- inject initial parameter values into graph tensors and track mapping
		foreach ($this->p as $tensor)
		{
			if (!($tensor instanceof Tensor))
				continue;
			
			$tid = $context->getTensorId($tensor);
			
			if ($tid === null || !isset($idToIndex[$tid]))
				continue;
			
			$data = [];
			
			if ($tensor instanceof Vector)
			{
				foreach ($tensor->a as $node)
				{
					$data[] = $node;
				}
			}
			else if ($tensor instanceof Matrix)
			{
				for ($i = 0; $i < $tensor->m; $i++)
				{
					for ($j = 0; $j < $tensor->n; $j++)
					{
						$node = $tensor->a[$i][$j];
						$data[] = $node;
					}
				}
			}
			else if ($tensor instanceof Scalar)
			{
				$node = $tensor->a;
				$data[] = $node;
			}
			
			$graph['tensors'][$idToIndex[$tid]]['data'] = $data;
		}
		
		// --- add lossId and list of trainable
		$graph['loss'] = $lossId;
		$graph['trainable'] = $trainableIds;
		
		return $graph;
	}
}
