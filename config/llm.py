"""LLM configuration via LiteLLM."""

# Default model — override with env var or CLI --model flag
# Examples:
#   openai/gpt-4o
#   openai/gpt-4o-mini
#   anthropic/claude-sonnet-4-20250514
#   ollama/qwen2.5:14b

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4096

# API base override for local models (Ollama, vLLM)
# e.g. "http://localhost:11434/v1"
LOCAL_API_BASE = None
