<?php

namespace PHP2xAI\Utility\Images;

/**
 * Vectorizer
 *
 * Converts a PNG image to a 1-D vector of grayscale values.
 *
 * @category    Scientific Computing
 * @package     antoniogweb/PHP2xAI
 */
class Vectorizer
{
	/**
	 * @var float[]
	 */
	protected $data = [];
	
	/**
	 * @var int
	 */
	protected $width;
	
	/**
	 * @var int
	 */
	protected $height;
	
	public function __construct(string $path)
	{
		if (!is_file($path))
		{
			throw new \InvalidArgumentException("File not found: ".$path);
		}
		
		$image = @imagecreatefrompng($path);
		
		if ($image === false)
		{
			throw new \RuntimeException("Unable to load PNG: ".$path);
		}
		
		$this->width = imagesx($image);
		$this->height = imagesy($image);
		
		for ($y = 0; $y < $this->height; $y++)
		{
			for ($x = 0; $x < $this->width; $x++)
			{
				$rgb = imagecolorat($image, $x, $y);
				
				$r = ($rgb >> 16) & 0xFF;
				$g = ($rgb >> 8) & 0xFF;
				$b = $rgb & 0xFF;
				
				$gray = ($r + $g + $b) / (3 * 255);
				
				// 0 = black, 1 = white
				$this->data[] = $gray;
			}
		}
		
		imagedestroy($image);
	}
	
	public function toArray(bool $binarize = false, float $threshold = 0.1) : array
	{
		if (!$binarize)
			return $this->data;
		
		$r = [];
		
		foreach ($this->data as $value)
		{
			$r[] = ($value > $threshold) ? 1.0 : 0.0;
		}
		
		return $r;
	}
	
	public function getWidth() : int
	{
		return $this->width;
	}
	
	public function getHeight() : int
	{
		return $this->height;
	}
}
