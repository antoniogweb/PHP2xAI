<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

/**
 * PHP
 *
 * Class to manage batch
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
class TrainValidateDataset
{
	public BatchDataset $train;
	public BatchDataset $val;
	
	public function __construct(BatchDataset $train, BatchDataset $val)
	{
		$this->train = $train;
		$this->val = $val;
	}
}
