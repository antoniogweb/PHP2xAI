<?php

namespace PHP2xAI\Runtime\PHP\Datasets;

use PHP2xAI\Runtime\PHP\Datasets\StreamFileDataset;


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
	public StreamFileDataset $train;
    public StreamFileDataset $val;
	
	public function __construct(StreamFileDataset $train, StreamFileDataset $val)
	{
		$this->train = $train;
		$this->val = $val;
	}
}
