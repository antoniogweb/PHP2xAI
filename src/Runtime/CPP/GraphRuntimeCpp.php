<?php

namespace PHP2xAI\Runtime\CPP;

use FFI;
use RuntimeException;
use PHP2xAI\Runtime\PHP\Core\GraphRuntime;

class GraphRuntimeCpp extends GraphRuntime
{
	private FFI $ffi;
	private $handle;
	/** @var array<int,array> */
	private array $tensorShapes = [];
	/** @var array<int,int> */
	private array $tensorSizes = [];

	public function __construct(array $graphDef)
	{
		$graphJson = json_encode($graphDef);
		
		if (!extension_loaded('ffi'))
			throw new RuntimeException("FFI extension is not enabled");

		$soPath = realpath(__DIR__ . '/php2xai_runtime.so');
		
		if (!is_file($soPath))
			throw new RuntimeException("FFI library not found: ".$soPath);

		$this->ffi = FFI::cdef($this->getCdef(), $soPath);
		$this->handle = $this->ffi->php2xai_runtime_create($graphJson);

		if ($this->handle === null)
			throw new RuntimeException("Unable to initialize CPP runtime");

		parent::__construct($graphDef);
	}

	public function __destruct()
	{
		if (isset($this->handle) && $this->handle !== null)
		{
			$this->ffi->php2xai_runtime_destroy($this->handle);
			$this->handle = null;
		}
	}

	public function forward() : void
	{
		$rc = $this->ffi->php2xai_runtime_forward($this->handle);
		if ($rc !== 0)
			throw new RuntimeException("CPP forward failed: ".$rc);
	}

	public function backward() : void
	{
		$rc = $this->ffi->php2xai_runtime_backward($this->handle);
		if ($rc !== 0)
			throw new RuntimeException("CPP backward failed: ".$rc);
	}

	public function getTensorShape(int $id) : array
	{
		$shape = $this->tensors[$id]->shape ?? null;
		if ($shape === null)
			throw new RuntimeException("Unknown tensor id: ".$id);

		$len = count($shape);
		if ($len === 0)
			return [];

		$ptr = $this->ffi->php2xai_runtime_get_tensor_shape($this->handle, $id);
		if ($ptr === null)
			throw new RuntimeException("CPP getTensorShape failed: ".$id);

		$out = [];
		for ($i = 0; $i < $len; $i++)
			$out[] = (int)$ptr[$i];

		return $out;
	}

	public function getTensorData(int $id) : array
	{
		$n = $this->getTensorSize($id);
		if ($n === null)
			throw new RuntimeException("Unknown tensor id: ".$id);

		$buf = FFI::new("float[".$n."]");
		$rc = $this->ffi->php2xai_runtime_get_tensor_data($this->handle, $id, $buf, $n);
		if ($rc !== 0)
			throw new RuntimeException("CPP getTensorData failed: ".$rc);

		$out = [];
		for ($i = 0; $i < $n; $i++)
			$out[] = (float)$buf[$i];

		return $out;
	}

	public function getTensorGrad(int $id) : array
	{
		$n = $this->getTensorSize($id);
		if ($n === null)
			throw new RuntimeException("Unknown tensor id: ".$id);

		$buf = FFI::new("float[".$n."]");
		$rc = $this->ffi->php2xai_runtime_get_tensor_grad($this->handle, $id, $buf, $n);
		if ($rc !== 0)
			throw new RuntimeException("CPP getTensorGrad failed: ".$rc);

		$out = [];
		for ($i = 0; $i < $n; $i++)
			$out[] = (float)$buf[$i];

		return $out;
	}

	private function getCdef() : string
	{
		return <<<CDEF
			typedef struct PHP2xAI_Runtime PHP2xAI_Runtime;
			PHP2xAI_Runtime* php2xai_runtime_create(const char* graph_json);
			void php2xai_runtime_destroy(PHP2xAI_Runtime* runtime);
			int php2xai_runtime_forward(PHP2xAI_Runtime* runtime);
			int php2xai_runtime_backward(PHP2xAI_Runtime* runtime);
			int* php2xai_runtime_get_tensor_shape(PHP2xAI_Runtime* runtime, int id);
			int php2xai_runtime_get_tensor_data(PHP2xAI_Runtime* runtime, int id, float* out, int n);
			int php2xai_runtime_get_tensor_grad(PHP2xAI_Runtime* runtime, int id, float* out, int n);
		CDEF;
	}
}
