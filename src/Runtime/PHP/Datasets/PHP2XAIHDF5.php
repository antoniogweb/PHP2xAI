<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

use FFI;
use RuntimeException;

/** PHP wrapper for the PHP2xAI HDF5 dataset API. */
class PHP2XAIHDF5
{
	public const FLOAT32 = 1;
	public const FLOAT64 = 2;
	public const INT32 = 3;
	public const INT64 = 4;

	private FFI $ffi;
	private $handle;

	private function __construct(FFI $ffi, $handle)
	{
		$this->ffi = $ffi;
		$this->handle = $handle;
	}

	/** Create a new HDF5 file, truncating it if it already exists. */
	public static function create(string $filename, ?string $soPath = null): self
	{
		$ffi = self::loadFFI($soPath);
		$handle = $ffi->php2xai_hdf5_create($filename);
		if ($handle === null)
			throw new RuntimeException(self::errorMessage($ffi, null, 'Unable to create HDF5 file'));

		return new self($ffi, $handle);
	}

	/** Open an existing HDF5 file for reading and writing. */
	public static function open(string $filename, ?string $soPath = null): self
	{
		$ffi = self::loadFFI($soPath);
		$handle = $ffi->php2xai_hdf5_open($filename);
		if ($handle === null)
			throw new RuntimeException(self::errorMessage($ffi, null, 'Unable to open HDF5 file'));

		return new self($ffi, $handle);
	}

	public function __destruct()
	{
		$this->destroy();
	}

	/** Release the native HDF5 handle. Safe to call more than once. */
	public function destroy(): void
	{
		if ($this->handle !== null) {
			$this->ffi->php2xai_hdf5_destroy($this->handle);
			$this->handle = null;
		}
	}

	/** Define a field whose shape describes one sample (without the sample axis). */
	public function setField(string $name, int $dtype, array $shape): void
	{
		$this->assertOpen();
		if (!in_array($dtype, [self::FLOAT32, self::FLOAT64, self::INT32, self::INT64], true))
			throw new \InvalidArgumentException('Unsupported HDF5 dtype: ' . $dtype);
		if (!$shape)
			throw new \InvalidArgumentException('Field shape cannot be empty');

		$shapeBuffer = $this->ffi->new('int64_t[' . count($shape) . ']');
		foreach (array_values($shape) as $i => $dimension) {
			if (!is_int($dimension) || $dimension <= 0)
				throw new \InvalidArgumentException('Field dimensions must be positive integers');
			$shapeBuffer[$i] = $dimension;
		}

		$rc = $this->ffi->php2xai_hdf5_set_field(
			$this->handle, $name, $dtype, $shapeBuffer, count($shape)
		);
		$this->checkResult($rc, 'setField');
	}

	/** Return ['dtype' => int, 'shape' => int[]] for a field. */
	public function fieldMetadata(string $name): array
	{
		$this->assertOpen();
		$dtype = $this->ffi->new('int[1]');
		$rank = $this->ffi->new('size_t[1]');

		// The first call reports the required shape rank (and returns code 2).
		$rc = $this->ffi->php2xai_hdf5_field_metadata(
			$this->handle, $name, $dtype, null, 0, $rank
		);
		if ($rc !== 2)
			$this->checkResult($rc, 'fieldMetadata');

		$shape = $this->ffi->new('int64_t[' . max(1, (int)$rank[0]) . ']');
		$rc = $this->ffi->php2xai_hdf5_field_metadata(
			$this->handle, $name, $dtype, $shape, (int)$rank[0], $rank
		);
		$this->checkResult($rc, 'fieldMetadata');

		$dimensions = [];
		for ($i = 0; $i < (int)$rank[0]; $i++)
			$dimensions[] = (int)$shape[$i];

		return ['dtype' => (int)$dtype[0], 'shape' => $dimensions];
	}

	public function count(): int
	{
		$this->assertOpen();
		$count = $this->ffi->php2xai_hdf5_count($this->handle);
		if ($count < 0)
			throw new RuntimeException($this->lastError('count failed'));
		return (int)$count;
	}

	/** Append one sample. $data is a flat, row-major PHP array. */
	public function add(string $field, array $data): void
	{
		$this->assertOpen();
		$metadata = $this->fieldMetadata($field);
		$values = array_values($data);
		$expected = self::elementCount($metadata['shape']);
		if (count($values) !== $expected)
			throw new \InvalidArgumentException("Field '{$field}' expects {$expected} values, got " . count($values));

		[$ctype, $convert] = self::bufferType($metadata['dtype']);
		$buffer = $this->ffi->new($ctype . '[' . max(1, $expected) . ']');
		foreach ($values as $i => $value)
			$buffer[$i] = $convert($value);

		$rc = $this->ffi->php2xai_hdf5_add($this->handle, $field, $buffer);
		$this->checkResult($rc, 'add');
	}

	/** Read samples by index as one flat row-major PHP array. */
	public function readIndices(string $field, array $indices): array
	{
		$this->assertOpen();
		if (!$indices)
			return [];
		$metadata = $this->fieldMetadata($field);
		$indexBuffer = $this->ffi->new('int64_t[' . count($indices) . ']');
		foreach (array_values($indices) as $i => $index) {
			if (!is_int($index) || $index < 0)
				throw new \InvalidArgumentException('Indices must be non-negative integers');
			$indexBuffer[$i] = $index;
		}

		$sampleSize = self::elementCount($metadata['shape']);
		$total = $sampleSize * count($indices);
		[$ctype] = self::bufferType($metadata['dtype']);
		$output = $this->ffi->new($ctype . '[' . max(1, $total) . ']');
		$rc = $this->ffi->php2xai_hdf5_read_indices(
			$this->handle, $field, $indexBuffer, count($indices), $output
		);
		$this->checkResult($rc, 'readIndices');

		$values = [];
		for ($i = 0; $i < $total; $i++)
			$values[] = $output[$i];
		return $values;
	}

	private static function loadFFI(?string $soPath): FFI
	{
		if (!extension_loaded('ffi'))
			throw new RuntimeException('FFI extension is not enabled');

		if ($soPath === null) {
			$platform = 'linux-' . php_uname('m');
			$soPath = dirname(__DIR__, 2) . '/CPP/Bin/' . $platform . '/php2xai_hdf5.so';
		}
		if (!is_file($soPath))
			throw new RuntimeException('HDF5 FFI library not found: ' . $soPath);

		return FFI::cdef(self::getCdef(), $soPath);
	}

	private static function getCdef(): string
	{
		return <<<CDEF
			typedef unsigned long size_t;
			typedef signed int int32_t;
			typedef signed long int64_t;
			typedef struct PHP2XAIHDF5_Handle PHP2XAIHDF5_Handle;
			PHP2XAIHDF5_Handle* php2xai_hdf5_create(const char* filename);
			PHP2XAIHDF5_Handle* php2xai_hdf5_open(const char* filename);
			void php2xai_hdf5_destroy(PHP2XAIHDF5_Handle* handle);
			int php2xai_hdf5_set_field(PHP2XAIHDF5_Handle* handle, const char* name, int dtype, const int64_t* shape, size_t rank);
			int php2xai_hdf5_field_metadata(PHP2XAIHDF5_Handle* handle, const char* name, int* dtype, int64_t* shape, size_t shape_capacity, size_t* shape_rank);
			int64_t php2xai_hdf5_count(PHP2XAIHDF5_Handle* handle);
			int php2xai_hdf5_add(PHP2XAIHDF5_Handle* handle, const char* field, const void* data);
			int php2xai_hdf5_read_indices(PHP2XAIHDF5_Handle* handle, const char* field, const int64_t* indices, size_t indices_count, void* output);
			const char* php2xai_hdf5_last_error(PHP2XAIHDF5_Handle* handle);
		CDEF;
	}

	private static function errorMessage(FFI $ffi, $handle, string $fallback): string
	{
		$error = $ffi->php2xai_hdf5_last_error($handle);
		return $error !== null ? FFI::string($error) : $fallback;
	}

	private function assertOpen(): void
	{
		if ($this->handle === null)
			throw new RuntimeException('HDF5 dataset is already destroyed');
	}

	private function lastError(string $fallback): string
	{
		return self::errorMessage($this->ffi, $this->handle, $fallback);
	}

	private function checkResult(int $code, string $operation): void
	{
		if ($code !== 0)
			throw new RuntimeException($operation . ' failed: ' . $this->lastError('error code ' . $code));
	}

	private static function bufferType(int $dtype): array
	{
		switch ($dtype) {
			case self::FLOAT32: return ['float', static function ($v) { return (float)$v; }];
			case self::FLOAT64: return ['double', static function ($v) { return (float)$v; }];
			case self::INT32: return ['int32_t', static function ($v) { return (int)$v; }];
			case self::INT64: return ['int64_t', static function ($v) { return (int)$v; }];
			default: throw new RuntimeException('Unsupported HDF5 dtype: ' . $dtype);
		}
	}

	private static function elementCount(array $shape): int
	{
		$count = 1;
		foreach ($shape as $dimension)
			$count *= $dimension;
		return $count;
	}

}
