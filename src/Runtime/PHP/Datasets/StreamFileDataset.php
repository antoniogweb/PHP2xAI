<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

use PHP2xAI\Tensor\Vector;
use PHP2xAI\Tensor\Scalar;

class StreamFileDataset implements \IteratorAggregate
{
	private string $path;
	private int $batchSize;
	private string $delimiter;

	/** @var resource */
	private $fh;

	/** @var int[] offset byte di inizio batch (uno ogni batchSize righe) */
	private array $batchOffsets = [];

	/** @var int[] ordine dei batch (indici su batchOffsets) */
	private array $batchOrder = [];

	private int $numLines = 0;

	public function __construct(string $path, int $batchSize, string $delimiter = '|')
	{
		if ($batchSize <= 0)
			throw new \InvalidArgumentException("batchSize must be > 0");
		
		if (!is_file($path))
			throw new \RuntimeException("Dataset file not found: {$path}");

		$this->path = realpath($path);
		$this->batchSize = $batchSize;
		$this->delimiter = $delimiter;

		$this->fh = fopen($this->path, 'rb');
		
		if (!$this->fh)
			throw new \RuntimeException("Unable to open dataset: {$path}");

		$this->buildBatchOffsets();
		$this->resetOrder();
		$this->initPlaceholders();
	}

	public function __destruct()
	{
		if (is_resource($this->fh))
		{
			fclose($this->fh);
		}
	}
	
	public function getBatchSize() : int
	{
		return $this->batchSize;
	}
	
	public function getPath() : string
	{
		return $this->path;
	}
	
	public function getXPlaceholder() : ?Vector
	{
		return $this->xPlaceholder;
	}
	
	public function getYPlaceholder() : ?Vector
	{
		return $this->yPlaceholder;
	}
	
	public function getPlaceholders() : array
	{
		return [
			'x' => $this->xPlaceholder,
			'y' => $this->yPlaceholder,
		];
	}
	
	protected function initPlaceholders() : void
	{
		fseek($this->fh, 0, SEEK_SET);
		$line = fgets($this->fh);
		[$x, $y] = $this->parseLineXY($line);
		
		$xShape = count($x);
		
		$this->xPlaceholder = new Vector(array_fill(0, $xShape, 0), true, 'x');
		
		if (count($y) > 1)
		{
			$yShape = count($y);
			$this->yPlaceholder = new Vector(array_fill(0, $yShape, 0), true, 'y');
		}
		else
			$this->yPlaceholder = new Scalar(0, 'y');
	}
	
	/** Ricostruisce l'ordine dei batch (senza shuffle). */
	public function resetOrder(): void
	{
		$this->batchOrder = range(0, count($this->batchOffsets) - 1);
	}

	/** Shuffle dell'ordine dei batch (shuffle degli indici su batchOffsets). */
	public function shuffleEpoch(?int $seed = null): void
	{
		if ($seed !== null)
			mt_srand($seed);
		
		shuffle($this->batchOrder);
	}

	public function numBatches(): int
	{
		return count($this->batchOffsets);
	}

	public function numLines(): int
	{
		return $this->numLines;
	}

	/**
	* Itera i batch nell'ordine corrente (batchOrder).
	* Ogni batch restituisce un generatore di coppie [$x, $y] riga-per-riga.
	*/
	public function getIterator()
	{
		foreach ($this->batchOrder as $batchIndex)
		{
			yield $this->iterateBatch($batchIndex);
		}
	}

	/**
	* Generatore che produce, per un batch, le coppie [$xArray, $yArray].
	* Non carica in RAM tutto il batch: legge riga per riga.
	*/
	private function iterateBatch(int $batchIndex)
	{
		$offset = $this->batchOffsets[$batchIndex];
		fseek($this->fh, $offset, SEEK_SET);

		for ($k = 0; $k < $this->batchSize; $k++)
		{
			$line = fgets($this->fh);
			
			if ($line === false)
				break; // EOF → ultimo batch più corto

			$line = trim($line);
			if ($line === '')
				continue; // ignora righe vuote

			[$x, $y] = $this->parseLineXY($line);
			
			yield [$x, $y];
		}
	}

	/**
	* Costruisce batchOffsets: salva offset byte della prima riga di ogni batch.
	* batch 0: riga 0, batch 1: riga batchSize, ecc.
	*/
	private function buildBatchOffsets(): void
	{
		$this->batchOffsets = [];
		$this->numLines = 0;

		fseek($this->fh, 0, SEEK_SET);

		while (!feof($this->fh))
		{
			$pos = ftell($this->fh);
			$line = fgets($this->fh);
			if ($line === false)
				break;

			// Offset del primo elemento del batch
			if (($this->numLines % $this->batchSize) === 0)
				$this->batchOffsets[] = $pos;

			$this->numLines++;
		}

		if (count($this->batchOffsets) === 0) {
			throw new \RuntimeException("Dataset is empty: {$this->path}");
		}
	}

	/**
	* Parse riga "x1 x2 x3 | y1 y2".
	* Ritorna due array di float.
	*/
	private function parseLineXY(string $line): array
	{
		$parts = explode($this->delimiter, $line, 2);
		if (count($parts) !== 2) {
			throw new \RuntimeException("Invalid line (missing delimiter '{$this->delimiter}'): {$line}");
		}

		$xStr = trim($parts[0]);
		$yStr = trim($parts[1]);

		$x = $this->parseFloatVector($xStr);
		$y = $this->parseFloatVector($yStr);

		return [$x, $y];
	}

	private function parseFloatVector(string $s): array
	{
		if ($s === '')
			return [];

		// split su spazi multipli
		$tokens = preg_split('/\s+/', $s);
		$out = [];
		foreach ($tokens as $tok)
		{
			if ($tok === '')
				continue;
			
			$out[] = (float)$tok;
		}
		
		return $out;
	}
}
