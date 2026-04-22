
# python
"""
config.py

Configuration management for Numeric Multiplication Assistant.

Responsibilities implemented here:
- Environment variable loading (via pydantic BaseSettings)
- API key management (explicit getter with error handling)
- LLM configuration and defaults (system prompt, user prompt template, few-shot examples)
- Domain-specific settings (max_digit_length, precision_mode, rounding, caches, timeouts)
- Validation and error handling for misconfiguration
- Default values and safe fallbacks

Usage:
    from config import settings, get_openai_api_key, llm_config
    api_key = get_openai_api_key(required=True)
    cfg = settings
    llm = llm_config()
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseSettings, Field, validator, root_validator
import os


# ---- Exceptions ----
class ConfigError(Exception):
    """General configuration error."""


class MissingAPIKeyError(ConfigError):
    """Raised when an API key is required but missing."""


class InvalidConfigError(ConfigError):
    """Raised when a configuration value is invalid."""


# ---- Default LLM Prompts & Examples (from agent design) ----
_DEFAULT_SYSTEM_PROMPT = (
    "You are Numeric Multiplication Assistant, a professional and precise agent. "
    "Validate that the user has provided exactly two numeric values (integers, floats, or scientific notation). "
    "If validation passes, compute the product deterministically using the appropriate precision engine. "
    "Return the numeric product first. Only include an explanation if the user explicitly requests it or if the computation uses fallback/arbitrary precision. "
    "Do not execute user-supplied code. Redact any PII and refuse malicious inputs."
)

_DEFAULT_USER_PROMPT_TEMPLATE = (
    "Multiply {value1} and {value2}. Respond with the numeric product. "
    "If I ask, also provide a brief explanation of the steps and precision used."
)

_DEFAULT_FEW_SHOT_EXAMPLES = [
    "Multiply 3 and 4 -> 12",
    "Multiply 2.5 and 0.12 -> 0.3",
    "Multiply 123456789123456789 and 987654321987654321 -> 121932631356500531347203169112635269",
]


# ---- Settings (env-driven) ----
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables (or .env file if present).

    Key naming convention: environment variables are upper-case with underscores.
    """

    # API / LLM credentials
    OPENAI_API_KEY: Optional[str] = Field(None, env="OPENAI_API_KEY")
    # Allow legacy OPENAI_KEY as fallback
    OPENAI_KEY_LEGACY: Optional[str] = Field(None, env="OPENAI_KEY")

    # LLM runtime config
    OPENAI_MODEL: str = Field("gpt-5-mini", env="OPENAI_MODEL")
    OPENAI_TEMPERATURE: float = Field(0.7, env="OPENAI_TEMPERATURE")
    OPENAI_MAX_TOKENS: int = Field(2000, env="OPENAI_MAX_TOKENS")
    LLM_SYSTEM_PROMPT: str = Field(_DEFAULT_SYSTEM_PROMPT, env="LLM_SYSTEM_PROMPT")
    LLM_USER_PROMPT_TEMPLATE: str = Field(_DEFAULT_USER_PROMPT_TEMPLATE, env="LLM_USER_PROMPT_TEMPLATE")
    LLM_FEW_SHOT_EXAMPLES: Optional[str] = Field(None, env="LLM_FEW_SHOT_EXAMPLES")  # JSON or newline-delimited; parsed below

    # Domain-specific settings
    MAX_DIGIT_LENGTH: int = Field(1000, env="MAX_DIGIT_LENGTH")  # protects arbitrary-precision resource use
    PRECISION_MODE: str = Field("auto", env="PRECISION_MODE")  # 'auto' | 'standard' | 'arbitrary'
    RESULT_ROUNDING_MODE: str = Field("ROUND_HALF_EVEN", env="RESULT_ROUNDING_MODE")

    # Caching & persistence
    REDIS_URL: Optional[str] = Field(None, env="REDIS_URL")
    CACHE_TTL_SECONDS: int = Field(24 * 3600, env="CACHE_TTL_SECONDS")  # 24h default
    AUDIT_LOG_PATH: str = Field("audit_log.jsonl", env="AUDIT_LOG_PATH")

    # Timeouts & retry policy
    REQUEST_TIMEOUT_SECONDS: int = Field(30, env="REQUEST_TIMEOUT_SECONDS")
    RETRY_ATTEMPTS: int = Field(3, env="RETRY_ATTEMPTS")
    BACKOFF_BASE_MS: int = Field(200, env="BACKOFF_BASE_MS")
    BACKOFF_MULTIPLIER: int = Field(2, env="BACKOFF_MULTIPLIER")

    # Security / operational toggles
    ENABLE_LLM_EXPLANATIONS: bool = Field(True, env="ENABLE_LLM_EXPLANATIONS")
    ENFORCE_PII_REJECTION: bool = Field(True, env="ENFORCE_PII_REJECTION")

    class Config:
        env_file = ".env"
        case_sensitive = True

    # Validators
    @validator("OPENAI_TEMPERATURE")
    def check_temperature(cls, v):
        if not 0.0 <= float(v) <= 2.0:
            raise InvalidConfigError("OPENAI_TEMPERATURE must be between 0.0 and 2.0")
        return float(v)

    @validator("MAX_DIGIT_LENGTH")
    def check_max_digit_length(cls, v):
        if int(v) <= 0:
            raise InvalidConfigError("MAX_DIGIT_LENGTH must be a positive integer")
        if int(v) > 1000000:
            # overly large values likely indicate misconfiguration/typo
            raise InvalidConfigError("MAX_DIGIT_LENGTH is unreasonably large; reduce to a sane upper bound")
        return int(v)

    @validator("PRECISION_MODE")
    def check_precision_mode(cls, v):
        if v not in {"auto", "standard", "arbitrary"}:
            raise InvalidConfigError("PRECISION_MODE must be one of 'auto', 'standard', or 'arbitrary'")
        return v

    @validator("REQUEST_TIMEOUT_SECONDS", "CACHE_TTL_SECONDS", "RETRY_ATTEMPTS", "BACKOFF_BASE_MS", "BACKOFF_MULTIPLIER")
    def check_positive(cls, v, field):
        if isinstance(v, int) and v < 0:
            raise InvalidConfigError(f"{field.name} must be non-negative")
        return v

    @root_validator(pre=True)
    def handle_legacy_keys_and_few_shot(cls, values):
        # If legacy OPENAI_KEY present and OPENAI_API_KEY absent, use legacy value
        openai_api = values.get("OPENAI_API_KEY") or values.get("OPENAI_KEY_LEGACY")
        if not values.get("OPENAI_API_KEY") and values.get("OPENAI_KEY_LEGACY"):
            values["OPENAI_API_KEY"] = values.get("OPENAI_KEY_LEGACY")

        # Parse LLM_FEW_SHOT_EXAMPLES if provided as newline or JSON list in env
        raw_examples = values.get("LLM_FEW_SHOT_EXAMPLES")
        if raw_examples:
            parsed = None
            # attempt JSON
            try:
                import json
                parsed = json.loads(raw_examples)
                if isinstance(parsed, list):
                    values["LLM_FEW_SHOT_EXAMPLES"] = parsed
                else:
                    values["LLM_FEW_SHOT_EXAMPLES"] = [str(raw_examples)]
            except Exception:
                # fallback: split on newline
                values["LLM_FEW_SHOT_EXAMPLES"] = [line.strip() for line in str(raw_examples).splitlines() if line.strip()]
        else:
            # set default few-shot examples if not provided
            values["LLM_FEW_SHOT_EXAMPLES"] = _DEFAULT_FEW_SHOT_EXAMPLES
        return values


# instantiate global settings (safe to import)
try:
    settings = Settings()
except Exception as e:
    # On import-time validation failure, raise a clear error so the service fails fast during startup.
    raise InvalidConfigError(f"Configuration validation failed: {e}")


# ---- API Key Management Helpers ----
def get_openai_api_key(required: bool = True) -> Optional[str]:
    """
    Returns the configured OpenAI API key.

    If required is True and no key is configured, raises MissingAPIKeyError with an actionable message.
    If required is False, returns None when not configured.
    """
    api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    # also check legacy env directly
    api_key_legacy = os.getenv("OPENAI_KEY")
    if api_key_legacy:
        return api_key_legacy
    if required:
        raise MissingAPIKeyError(
            "OpenAI API key is not configured. Set the OPENAI_API_KEY environment variable or provide credentials via the platform secret manager."
        )
    return None


def is_openai_api_key_configured() -> bool:
    try:
        return bool(get_openai_api_key(required=False))
    except MissingAPIKeyError:
        return False


# ---- LLM Configuration Getter ----
def llm_config() -> Dict[str, Any]:
    """
    Return a dictionary containing the LLM configuration to be used by the agent.
    This centralizes model name, temperature, token limits and the prompts/few-shot examples.
    """
    return {
        "provider": "openai",
        "model": settings.OPENAI_MODEL,
        "temperature": settings.OPENAI_TEMPERATURE,
        "max_tokens": settings.OPENAI_MAX_TOKENS,
        "system_prompt": settings.LLM_SYSTEM_PROMPT,
        "user_prompt_template": settings.LLM_USER_PROMPT_TEMPLATE,
        "few_shot_examples": settings.LLM_FEW_SHOT_EXAMPLES,
    }


# ---- Domain-specific getters & validators ----
def get_max_digit_length() -> int:
    return settings.MAX_DIGIT_LENGTH


def validate_operand_digit_length(digit_length: int) -> None:
    """
    Raise InvalidConfigError if digit_length exceeds configured max.
    """
    if digit_length > settings.MAX_DIGIT_LENGTH:
        raise InvalidConfigError(
            f"Operand digit length {digit_length} exceeds configured MAX_DIGIT_LENGTH ({settings.MAX_DIGIT_LENGTH})"
        )


# ---- Utility helpers for other modules ----
def get_cache_ttl() -> int:
    return settings.CACHE_TTL_SECONDS


def get_precision_mode() -> str:
    return settings.PRECISION_MODE


def get_result_rounding_mode() -> str:
    return settings.RESULT_ROUNDING_MODE


def get_request_timeout_seconds() -> int:
    return settings.REQUEST_TIMEOUT_SECONDS


def get_retry_policy() -> Dict[str, int]:
    return {
        "attempts": settings.RETRY_ATTEMPTS,
        "base_backoff_ms": settings.BACKOFF_BASE_MS,
        "multiplier": settings.BACKOFF_MULTIPLIER,
    }


# ---- Exported symbols ----
__all__ = [
    "settings",
    "Settings",
    "ConfigError",
    "MissingAPIKeyError",
    "InvalidConfigError",
    "get_openai_api_key",
    "is_openai_api_key_configured",
    "llm_config",
    "get_max_digit_length",
    "validate_operand_digit_length",
    "get_cache_ttl",
    "get_precision_mode",
    "get_result_rounding_mode",
    "get_request_timeout_seconds",
    "get_retry_policy",
]
