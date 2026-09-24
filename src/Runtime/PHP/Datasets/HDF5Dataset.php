<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

use PHP2xAI\Tensor\Tensor;
use RuntimeException;

/** Batch iterator over feature and target fields stored in an HDF5 file. */
class HDF5Dataset extends BatchDataset implements \IteratorAggregate
{
	private ?PHP2XAIHDF5 $dataset;
	private string $path;
	private int $batchSize;
	private string $xField;
	private string $yField;
	private int $sampleCount;
	private array $sampleOrder = [];
	private int $batchPosition = 0;
	private ?array $currentIndices = null;
	private array $xMetadata;
	private array $yMetadata;
	private ?Tensor $xPlaceholder = null;
	private ?Tensor $yPlaceholder = null;

	public function __construct(
		string $filename,
		int $batchSize,
		string $xField = 'x',
		string $yField = 'y'
	) {
		if ($batchSize <= 0)
			throw new \InvalidArgumentException('batchSize must be > 0');

		$this->path = $filename;
		$this->dataset = PHP2XAIHDF5::open($filename);
		$this->batchSize = $batchSize;
		$this->xField = $xField;
		$this->yField = $yField;
		$this->xMetadata = $this->dataset->fieldMetadata($xField);
		$this->yMetadata = $this->dataset->fieldMetadata($yField);
		$this->sampleCount = $this->dataset->count();

		if ($this->sampleCount < $this->batchSize)
			throw new RuntimeException('HDF5 dataset has no complete batches');

		$this->sampleOrder = range(0, $this->sampleCount - 1);
	}

	public function getBatchSize(): int
	{
		return $this->batchSize;
	}

	public function getPath(): string
	{
		return $this->path;
	}

	public function getType(): string
	{
		return 'HDF5';
	}

	public function resetEpoch(): void
	{
		$this->batchPosition = 0;
		$this->currentIndices = null;
	}

	public function shuffleEpoch(?int $seed = null): void
	{
		if ($seed !== null)
			mt_srand($seed);
		shuffle($this->sampleOrder);
	}

	public function nextBatch(): bool
	{
		if ($this->currentIndices !== null)
			return true;
		if ($this->batchPosition >= intdiv($this->sampleCount, $this->batchSize))
			return false;

		$firstPosition = $this->batchPosition * $this->batchSize;
		$this->currentIndices = array_slice($this->sampleOrder, $firstPosition, $this->batchSize);
		return true;
	}

	/** Return [xRowMajor, yRowMajor] for the current batch. */
	public function pack(): array
	{
		if ($this->currentIndices === null)
			throw new RuntimeException('Call nextBatch() before pack()');
		$this->ensureOpen();

		$indices = $this->currentIndices;
		$x = $this->dataset->readIndices($this->xField, $indices);
		$y = $this->dataset->readIndices($this->yField, $indices);

		$this->batchPosition++;
		$this->currentIndices = null;

		return [$x, $y];
	}

	/** Iterate batches, each yielding individual [x, y] row-major samples. */
	public function getIterator(): \Traversable
	{
		$numBatches = intdiv($this->sampleCount, $this->batchSize);
		for ($batch = 0; $batch < $numBatches; $batch++) {
			$first = $batch * $this->batchSize;
			$indices = array_slice($this->sampleOrder, $first, $this->batchSize);
			yield $this->iterateBatch($indices);
		}
	}

	/** Yield each sample separately, like StreamFileDataset's iterator. */
	private function iterateBatch(array $indices): \Generator
	{
		$this->ensureOpen();
		$x = $this->dataset->readIndices($this->xField, $indices);
		$y = $this->dataset->readIndices($this->yField, $indices);
		$xSize = self::elementCount($this->xMetadata['shape']);
		$ySize = self::elementCount($this->yMetadata['shape']);

		for ($i = 0; $i < count($indices); $i++) {
			yield [
				array_slice($x, $i * $xSize, $xSize),
				array_slice($y, $i * $ySize, $ySize),
			];
		}
	}

	public function initPlaceholders(bool $train = true): void
	{
		$batch = $train ? $this->batchSize : 1;
		$xShape = array_merge([$batch], $this->xMetadata['shape']);
		$ySampleElements = self::elementCount($this->yMetadata['shape']);
		$yShape = $ySampleElements === 1
			? [$batch]
			: array_merge([$batch], $this->yMetadata['shape']);

		$this->xPlaceholder = Tensor::zeros($xShape, 'x');
		$this->yPlaceholder = Tensor::zeros($yShape, 'y');
	}

	public function getPlaceholders(): array
	{
		return ['x' => $this->xPlaceholder, 'y' => $this->yPlaceholder];
	}

	public function getXPlaceholder(): ?Tensor
	{
		return $this->xPlaceholder;
	}

	public function getYPlaceholder(): ?Tensor
	{
		return $this->yPlaceholder;
	}

	/** Close the HDF5 handle so another process can open the file. */
	public function close(): void
	{
		if ($this->dataset !== null) {
			$this->dataset->destroy();
			$this->dataset = null;
		}
	}

	private function ensureOpen(): void
	{
		if ($this->dataset === null)
			$this->dataset = PHP2XAIHDF5::open($this->path);
	}

	private static function elementCount(array $shape): int
	{
		$count = 1;
		foreach ($shape as $dimension)
			$count *= $dimension;
		return $count;
	}
}
