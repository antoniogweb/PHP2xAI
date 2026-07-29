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

	public static function getPlatform(): string
	{
		$os = match (PHP_OS_FAMILY) {
			'Linux'   => 'linux',
			'Darwin'  => 'macos',
			'Windows' => 'windows',
			default   => throw new \RuntimeException('Unsupported OS'),
		};

		$arch = match (strtolower(php_uname('m'))) {
			'x86_64', 'amd64' => 'x86_64',
			'aarch64', 'arm64' => 'arm64',
			default => throw new \RuntimeException('Unsupported architecture'),
		};

		return "$os-$arch";
	}
}
