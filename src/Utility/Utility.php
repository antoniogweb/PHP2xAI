<?php

namespace PHP2xAI\Utility;

/**
 * Lightweight profiler to count object creations and accumulate timings.
 */
class Utility
{
	public static function argmax(array $values) : int
	{
		$maxIndex = 0;
		$maxValue = $values[0] ?? null;
		
		foreach ($values as $i => $v)
		{
			if ($maxValue === null || $v > $maxValue)
			{
				$maxValue = $v;
				$maxIndex = $i;
			}
		}
		
		return $maxIndex;
	}
}
