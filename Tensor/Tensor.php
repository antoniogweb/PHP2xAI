<?php

namespace PHP2xAI\Tensor;

use PHP2xAI\Graph\GraphContext;

/**
 * Matrix
 *
 * A zero, one or two dimensional tensor with integer and/or floating point elements.
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 * @author      Antonio Gallo <info@laboratoriolibero.com>
 */
class Tensor
{
	/**
	* Operation name that produced this tensor (if any)
	*
	* @var string|null
	*/
	public ?string $name = null;
	
	/**
	* Graph context used for IR construction.
	*
	* @var GraphContext|null
	*/
	protected ?GraphContext $context = null;
	
	public function toForward() {}
	
	public function setName(?string $name) : void
	{
		$this->name = $name;
	}
	
	public function getName() : ?string
	{
		return $this->name;
	}
	
	public function setContext(?GraphContext $context) : void
	{
		$this->context = $context;
	}
	
	public function getContext() : ?GraphContext
	{
		return $this->context;
	}
	
	public function getShape() : array
	{
		return [];
	}
	
	protected function initContextFrom(Tensor ...$inputs) : GraphContext
	{
		$context = $this->context;
		
		foreach ($inputs as $tensor)
		{
			if ($tensor !== null && $tensor->getContext() !== null)
			{
				$context = $tensor->getContext();
				break;
			}
		}
		
		if ($context === null)
			$context = new GraphContext();
		
		if ($this->context === null)
			$this->context = $context;
		
		foreach ($inputs as $tensor)
		{
			if ($tensor !== null && $tensor->getContext() === null)
				$tensor->setContext($context);
		}
		
		return $context;
	}
	
	protected function registerInContext(GraphContext $context, Tensor $tensor, string $defaultKind = 'intermediate') : int
	{
		if ($context->hasTensor($tensor))
			return (int)$context->getTensorId($tensor);
		
		return $context->registerTensor($tensor, $defaultKind, $tensor->getName(), $tensor->getShape());
	}
}
