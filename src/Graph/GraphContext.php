<?php

namespace PHP2xAI\Graph;

use PHP2xAI\Tensor\Tensor;

/**
* Lightweight context to track tensors and operations during graph building.

$graph = [
    'tensors' => [
        ['id' => 0, 'kind' => 'input',        'name' => 'x',      'shape' => [784],     'trainable' => true, 'requiresGrad' => false],
        ['id' => 1, 'kind' => 'target',       'name' => 'target', 'shape' => [10],      'trainable' => true, 'requiresGrad' => false],

        ['id' => 2, 'kind' => 'param',        'name' => 'W',      'shape' => [10, 784], 'trainable' => true, 'requiresGrad' => true],
        ['id' => 3, 'kind' => 'param',        'name' => 'b',      'shape' => [10],      'trainable' => true, 'requiresGrad' => true],

        ['id' => 4, 'kind' => 'intermediate', 'name' => 'logits', 'shape' => [10],      'trainable' => true, 'requiresGrad' => true],
        ['id' => 5, 'kind' => 'intermediate', 'name' => 'loss',   'shape' => [],        'trainable' => true, 'requiresGrad' => true],
    ],
    'ops' => [
        ['id' => 0, 'op' => 'matmul',            'inputs' => [2, 0], 'output' => 4], // logits = W*x
        ['id' => 1, 'op' => 'add',               'inputs' => [4, 3], 'output' => 4], // logits += b
        ['id' => 2, 'op' => 'softmax_ce_logits', 'inputs' => [4, 1], 'output' => 5], // loss
    ],
    'loss'      => 5,
    'trainable' => [2, 3],
];

*/
class GraphContext
{
	/** @var array<int,array> */
	protected array $tensors = [];
	
	/** @var array<int,array> */
	protected array $ops = [];
	
	/** @var array<int,int> maps spl_object_id => tensor id */
	protected array $tensorIds = [];
	
	/** @var array<int,int> maps tensor id => spl_object_id */
	protected ?array $tensorIdo = null;
	
	/** @var array<int,\PHP2xAI\Tensor\Backward\Tensor> strong refs to tensors to prevent GC reuse */
	protected array $tensorRefs = [];
	
	protected int $nextTensorId = 0;
	protected int $nextOpId = 0;
	
	public function registerTensor(Tensor $tensor, string $kind, ?string $name = null, array $shape = []) : int
	{
		$oid = spl_object_id($tensor);
		
		if (isset($this->tensorIds[$oid]))
			return $this->tensorIds[$oid];
		
		$id = $this->nextTensorId++;
		
		$this->tensorIds[$oid] = $id;
		$this->tensorRefs[$oid] = $tensor; // keep reference so spl_object_id is not recycled
		
		$requiresGrad = $tensor->requiresGrad() || ($kind === 'param' && $tensor->isTrainable());
		$tensor->setRequiresGrad($requiresGrad);

		$definition = [
			'id' => $id,
			'kind' => $kind,
			'name' => $name ?? $tensor->getName(),
			'shape' => $shape,
			'trainable' => $tensor->isTrainable(),
			'requiresGrad' => $requiresGrad,
		];

		if ($tensor->data !== [])
			$definition['data'] = $tensor->data;

		$this->tensors[] = $definition;
		
		$tensor->setContext($this);
		
		return $id;
	}
	
	public function registerOp(string $op, array $inputs, Tensor $output, array $attributes = []) : int
	{
		$outputId = $this->registerTensor($output, 'intermediate', $output->getName() ?? $op, $output->getShape());
		$requiresGrad = $output->requiresGrad();

		foreach ($inputs as $inputId)
		{
			if ($this->tensors[$inputId]['requiresGrad'])
			{
				$requiresGrad = true;
				break;
			}
		}

		$output->setRequiresGrad($requiresGrad);
		$this->tensors[$outputId]['requiresGrad'] = $requiresGrad;
		
		$opId = $this->nextOpId++;
		
		$this->ops[] = [
			'id' => $opId,
			'op' => $op,
			'inputs' => $inputs,
			'output' => $outputId,
			'attributes'	=>	$attributes,
		];
		
		return $opId;
	}
	
	public function getTensors() : array
	{
		return $this->tensors;
	}
	
	public function hasTensor(Tensor $tensor) : bool
	{
		return isset($this->tensorIds[spl_object_id($tensor)]);
	}
	
	public function getTensorId(Tensor $tensor) : ?int
	{
		return $this->tensorIds[spl_object_id($tensor)] ?? null;
	}
	
	public function getTensorFromId(int $id) : ?Tensor
	{
		if (!isset($this->tensorIdo))
			$this->tensorIdo = array_flip($this->tensorIds);
		
		$ido = $this->tensorIdo[$id] ?? null;
		
		if (!isset($ido))
			return null;
		
		return $this->tensorRefs[$ido] ?? null;
	}
	
	public function getTensorIndex(Tensor $tensor) : ?int
	{
		$tensorId = $this->getTensorId($tensor);
		
		foreach ($this->tensors as $idx => $tInfo)
		{
			if ((int)$tInfo['id'] === (int)$tensorId)
				return $idx;
		}
		
		return 0;
	}
	
	public function export() : array
	{
		$trainableIds = [];

		foreach ($this->tensors as $tensor)
		{
			if ($tensor['kind'] === 'param' && ($tensor['trainable'] ?? true))
				$trainableIds[] = $tensor['id'];
		}

		return [
			'tensors' => $this->tensors,
			'ops' => $this->ops,
			'trainable' => $trainableIds,
		];
	}
}
