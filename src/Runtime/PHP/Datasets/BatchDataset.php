<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

use PHP2xAI\Tensor\Tensor;

/** Shared batch iteration contract for datasets backed by different formats. */
abstract class BatchDataset
{
	abstract public function getType(): string;

	abstract public function getPath(): string;

	abstract public function resetEpoch(): void;

	abstract public function shuffleEpoch(?int $seed = null): void;

	abstract public function nextBatch(): bool;

	/** Return [xRowMajor, yRowMajor] for the current batch. */
	abstract public function pack(): array;

	abstract public function initPlaceholders(bool $train = true): void;

	abstract public function getPlaceholders(): array;

	abstract public function getXPlaceholder(): ?Tensor;

	abstract public function getYPlaceholder(): ?Tensor;

	abstract public function getBatchSize(): int;
}
