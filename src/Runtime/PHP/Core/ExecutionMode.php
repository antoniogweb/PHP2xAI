<?php

namespace PHP2xAI\Runtime\PHP\Core;

/**
 * Controls execution-specific behavior in the graph runtime.
 */
enum ExecutionMode
{
	case TRAIN;
	case INFER;
	case PREFILL;
	case DECODE;
}
