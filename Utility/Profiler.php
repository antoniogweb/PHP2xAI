<?php

namespace PHP2xAI\Utility;

/**
 * Lightweight profiler to count object creations and accumulate timings.
 */
class Profiler
{
	private static bool $enabled = false;
	private static array $counts = [];
	private static array $timers = [];
	
	public static function enable() : void
	{
		self::$enabled = true;
	}
	
	public static function disable() : void
	{
		self::$enabled = false;
	}
	
	public static function isEnabled() : bool
	{
		return self::$enabled;
	}
	
	public static function inc(string $name, int $by = 1) : void
	{
		if (!self::$enabled)
			return;
		
		self::$counts[$name] = (self::$counts[$name] ?? 0) + $by;
	}
	
	public static function start(string $name) : void
	{
		if (!self::$enabled)
			return;
		
		self::$timers[$name]['start'] = microtime(true);
	}
	
	public static function stop(string $name) : void
	{
		if (!self::$enabled || !isset(self::$timers[$name]['start']))
			return;
		
		$elapsed = microtime(true) - self::$timers[$name]['start'];
		
		self::$timers[$name]['time'] = (self::$timers[$name]['time'] ?? 0) + $elapsed;
		self::$timers[$name]['calls'] = (self::$timers[$name]['calls'] ?? 0) + 1;
		
		unset(self::$timers[$name]['start']);
	}
	
	public static function snapshot() : array
	{
		return [
			'counts' => self::$counts,
			'timers' => self::$timers,
		];
	}
	
	public static function reset() : void
	{
		self::$counts = [];
		self::$timers = [];
	}
}
