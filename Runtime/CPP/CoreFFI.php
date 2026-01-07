<?php

namespace PHP2xAI\Runtime\CPP;

use FFI;
use RuntimeException;

class CoreFFI
{
	private FFI $ffi;
	private $handle;
	private int $inputSize;
	private int $outputSize;

	public function __construct(string $modelPath, string $weightsPath, string $soPath)
	{
		if (!extension_loaded('ffi'))
			throw new RuntimeException("FFI extension is not enabled");
		
		if (!is_file($soPath))
			throw new RuntimeException("FFI library not found: ".$soPath);
		
		$this->ffi = FFI::cdef($this->getCdef(), $soPath);
		$this->handle = $this->ffi->php2xai_core_create($modelPath, $weightsPath);
		
		if ($this->handle === null)
			throw new RuntimeException("Unable to initialize CPP runtime");
		
		$this->inputSize = (int)$this->ffi->php2xai_core_input_size($this->handle);
		$this->outputSize = (int)$this->ffi->php2xai_core_output_size($this->handle);
		
		if ($this->inputSize <= 0 || $this->outputSize <= 0)
			throw new RuntimeException("Invalid CPP runtime sizes");
	}
	
	public function __destruct()
	{
		if (isset($this->handle) && $this->handle !== null)
		{
			$this->ffi->php2xai_core_destroy($this->handle);
			$this->handle = null;
		}
	}
	
	public function predict(array $x) : array
	{
		$values = array_values($x);
		
		if (count($values) !== $this->inputSize)
			throw new RuntimeException("Inserting incompatible dimensions");
		
		$input = FFI::new("float[".$this->inputSize."]");
		
		for ($i = 0; $i < $this->inputSize; $i++)
			$input[$i] = (float)$values[$i];
		
		$output = FFI::new("float[".$this->outputSize."]");
		
		$rc = $this->ffi->php2xai_core_predict(
			$this->handle,
			$input,
			$this->inputSize,
			$output,
			$this->outputSize
		);
		
		if ($rc !== 0)
			throw new RuntimeException("CPP predict failed: ".$rc);
		
		$result = [];
		
		for ($i = 0; $i < $this->outputSize; $i++)
			$result[] = $output[$i];
		
		return $result;
	}
	
	public function predictLabelInt(array $x) : int
	{
		$values = array_values($x);
		
		if (count($values) !== $this->inputSize)
			throw new RuntimeException("Inserting incompatible dimensions");
		
		$input = FFI::new("float[".$this->inputSize."]");
		
		for ($i = 0; $i < $this->inputSize; $i++)
			$input[$i] = (float)$values[$i];
		
		$outLabel = FFI::new("int[1]");
		
		$rc = $this->ffi->php2xai_core_predict_label_int(
			$this->handle,
			$input,
			$this->inputSize,
			$outLabel
		);
		
		if ($rc !== 0)
			throw new RuntimeException("CPP predictLabelInt failed: ".$rc);
		
		return (int)$outLabel[0];
	}
	
	private function getCdef() : string
	{
		return <<<CDEF
			typedef unsigned long size_t;
			typedef struct PHP2xAI_Core PHP2xAI_Core;
			PHP2xAI_Core* php2xai_core_create(const char* model_path, const char* weights_path);
			void php2xai_core_destroy(PHP2xAI_Core* core);
			size_t php2xai_core_input_size(PHP2xAI_Core* core);
			size_t php2xai_core_output_size(PHP2xAI_Core* core);
			int php2xai_core_predict(PHP2xAI_Core* core, const float* x, size_t x_len, float* out, size_t out_len);
			int php2xai_core_predict_label_int(PHP2xAI_Core* core, const float* x, size_t x_len, int* out_label);
		CDEF;
	}
}
