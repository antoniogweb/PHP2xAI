<?php

namespace PHP2xAI\Tokenizer\PHP;

use FFI;
use RuntimeException;
use PHP2xAI\Utility;

class Tokenizer
{
	private FFI $ffi;
	private $handle = null;
	private string $tokenizerPath;
	private string $libraryPath;

	public function __construct(string $tokenizerPath, ?string $libraryPath = null)
	{
		if (!extension_loaded('ffi'))
			throw new RuntimeException('FFI extension is not enabled');

		if (!is_file($tokenizerPath))
			throw new RuntimeException('Tokenizer file not found: '.$tokenizerPath);

		$this->tokenizerPath = $tokenizerPath;
		$this->libraryPath = $libraryPath ?? self::resolveLibraryPath();

		if (!is_file($this->libraryPath))
			throw new RuntimeException('Tokenizer FFI library not found: '.$this->libraryPath);

		$this->ffi = FFI::cdef($this->getCdef(), $this->libraryPath);
		$this->handle = $this->ffi->php2xai_tokenizer_load($this->tokenizerPath);

		if ($this->handle === null)
			throw new RuntimeException('Unable to load tokenizer: '.$this->lastError());
	}

	public function __destruct()
	{
		$this->free();
	}

	public static function resolveLibraryPath(): string
	{
		$platform = Utility::getPlatform();
		$path = __DIR__.'/../Rust/Bindings/Bin/'.$platform.'/libtokenizers.so';
		$realPath = realpath($path);

		return $realPath !== false ? $realPath : $path;
	}

	/**
	 * @return int[]
	 */
	public function encode(string $text, bool $addSpecialTokens = true): array
	{
		$this->assertLoaded();

		$encoding = $this->ffi->php2xai_tokenizer_encode(
			$this->handle,
			$text,
			$addSpecialTokens
		);

		if ($encoding === null)
			throw new RuntimeException('Tokenizer encode failed: '.$this->lastError());

		try
		{
			$length = (int)$this->ffi->php2xai_tokenizer_encoding_length($encoding);
			$idsPtr = $this->ffi->php2xai_tokenizer_encoding_ids($encoding);

			if ($idsPtr === null && $length > 0)
				throw new RuntimeException('Tokenizer encode failed: returned NULL ids');

			$ids = [];
			for ($i = 0; $i < $length; $i++)
				$ids[] = (int)$idsPtr[$i];

			return $ids;
		}
		finally
		{
			$this->ffi->php2xai_tokenizer_encoding_free($encoding);
		}
	}

	/**
	 * @return int[]
	 */
	public function encodeFixed(
		string $text,
		int $length,
		bool $addSpecialTokens = true,
		int $padTokenId = 0
	): array
	{
		if ($length < 0)
			throw new RuntimeException('Tokenizer fixed length must be >= 0');

		$ids = array_slice($this->encode($text, $addSpecialTokens), 0, $length);

		if (count($ids) < $length)
			$ids = array_pad($ids, $length, $padTokenId);

		return $ids;
	}

	/**
	 * @param int[] $ids
	 */
	public function decode(array $ids, bool $skipSpecialTokens = true): string
	{
		$this->assertLoaded();

		$length = count($ids);
		$idsBuffer = null;

		if ($length > 0)
		{
			$idsBuffer = FFI::new('uint32_t['.$length.']');
			foreach (array_values($ids) as $i => $id)
				$idsBuffer[$i] = (int)$id;
		}

		$textPtr = $this->ffi->php2xai_tokenizer_decode(
			$this->handle,
			$idsBuffer,
			$length,
			$skipSpecialTokens
		);

		if ($textPtr === null)
			throw new RuntimeException('Tokenizer decode failed: '.$this->lastError());

		try
		{
			return $this->stringFromPointer($textPtr);
		}
		finally
		{
			$this->ffi->php2xai_tokenizer_string_free($textPtr);
		}
	}

	public function free(): void
	{
		if ($this->handle !== null)
		{
			$this->ffi->php2xai_tokenizer_free($this->handle);
			$this->handle = null;
		}
	}

	public function getTokenizerPath(): string
	{
		return $this->tokenizerPath;
	}

	public function getLibraryPath(): string
	{
		return $this->libraryPath;
	}

	private function assertLoaded(): void
	{
		if ($this->handle === null)
			throw new RuntimeException('Tokenizer is closed');
	}

	private function lastError(): string
	{
		$error = $this->ffi->php2xai_tokenizer_last_error();

		if ($error === null)
			return 'unknown error';

		$message = $this->stringFromPointer($error);

		return $message !== '' ? $message : 'unknown error';
	}

	private function stringFromPointer($value): string
	{
		if (is_string($value))
			return $value;

		return FFI::string($value);
	}

	private function getCdef(): string
	{
		return <<<CDEF
			typedef unsigned char bool;
			typedef unsigned long size_t;
			typedef unsigned int uint32_t;
			typedef struct PHP2xAITokenizer PHP2xAITokenizer;
			typedef struct PHP2xAIEncoding PHP2xAIEncoding;

			const char *php2xai_tokenizer_last_error(void);
			PHP2xAITokenizer *php2xai_tokenizer_load(const char *tokenizer_json_path);
			void php2xai_tokenizer_free(PHP2xAITokenizer *handle);
			PHP2xAIEncoding *php2xai_tokenizer_encode(const PHP2xAITokenizer *handle, const char *text, bool add_special_tokens);
			const uint32_t *php2xai_tokenizer_encoding_ids(const PHP2xAIEncoding *encoding);
			size_t php2xai_tokenizer_encoding_length(const PHP2xAIEncoding *encoding);
			void php2xai_tokenizer_encoding_free(PHP2xAIEncoding *encoding);
			char *php2xai_tokenizer_decode(const PHP2xAITokenizer *handle, const uint32_t *ids, size_t length, bool skip_special_tokens);
			void php2xai_tokenizer_string_free(char *value);
		CDEF;
	}
}
