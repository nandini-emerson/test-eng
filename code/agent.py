try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': True,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 2,
 'runtime_enabled': True,
 'sanitize_pii': False}

<![CDATA[
#!/usr/bin/env python3
"""
Numeric Multiplication Assistant - agent.py

A FastAPI-based microservice implementing a layered multiplication service with:
- Strict input validation and numeric parsing
- Precision selection (standard double vs arbitrary precision)
- Deterministic computation using Decimal (and optional gmpy2)
- Result formatting and audit logging
- Optional Redis-backed caching
- Safe, lazy OpenAI integration for explanation generation

This module is designed to be import-safe even if API keys or external services are not configured.
Observability helpers (trace_step, trace_step_sync) are used when available via the runtime; safe no-op fallbacks are provided.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext, localcontext
from typing import Any, Dict, Optional, Tuple, Union, List

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError, field_validator
from starlette.middleware.cors import CORSMiddleware

# Optional native libs (import lazily where used)
_gmpy2 = None
try:
    import gmpy2 as _gmpy2  # type: ignore
except Exception:
    _gmpy2 = None

# Optional Redis client
_redis = None
try:
    import redis
except Exception:
    redis = None  # type: ignore

# Optional OpenAI Async client (lazy created)
try:
    import openai
except Exception:
    openai = None  # type: ignore

# Observability wrappers (injected by runtime). Provide safe fallbacks if absent.
def _get_async_trace_step(name: str, **kwargs):
    """
    Return the runtime-provided async trace_step context manager if available;
    otherwise return a dummy no-op async context manager implementing the same API.
    """
    try:
        # trace_step is injected by runtime
        return trace_step(name, **kwargs)  # type: ignore  # noqa: F821
    except Exception:
        @asynccontextmanager
        async def _dummy():
            class DummyStep:
                def capture(self, _r: Any):
                    pass
            step = DummyStep()
            try:
                yield step
            finally:
                pass
        return _dummy()


def _get_sync_trace_step(name: str, **kwargs):
    """
    Return the runtime-provided trace_step_sync context manager if available;
    otherwise return a dummy no-op sync context manager.
    """
    try:
        return trace_step_sync(name, **kwargs)  # type: ignore  # noqa: F821
    except Exception:
        @contextmanager
        def _dummy():
            class DummyStep:
                def capture(self, _r: Any):
                    pass
            step = DummyStep()
            try:
                yield step
            finally:
                pass
        return _dummy()


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("numeric-multiplication-agent")


# ---------------------------
# Configuration & Utilities
# ---------------------------
class Config:
    """
    Lazy configuration loader using environment variables.
    Access properties to read values. No validation at import time.
    """

    @staticmethod
    def get_openai_model() -> str:
        return os.getenv("OPENAI_MODEL", "gpt-5-mini")

    @staticmethod
    def get_openai_temperature() -> float:
        try:
            return float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
        except Exception:
            return 0.7

    @staticmethod
    def get_openai_max_tokens() -> int:
        try:
            return int(os.getenv("OPENAI_MAX_TOKENS", "2000"))
        except Exception:
            return 2000

    @staticmethod
    def get_max_digit_length() -> int:
        try:
            return int(os.getenv("MAX_DIGIT_LENGTH", "1000"))
        except Exception:
            return 1000

    @staticmethod
    def get_precision_mode() -> str:
        return os.getenv("PRECISION_MODE", "auto")  # 'auto' | 'standard' | 'arbitrary'

    @staticmethod
    def get_result_rounding_mode() -> str:
        return os.getenv("RESULT_ROUNDING_MODE", "ROUND_HALF_EVEN")

    @staticmethod
    def get_redis_url() -> Optional[str]:
        return os.getenv("REDIS_URL")


# ---------------------------
# Exceptions & Error Codes
# ---------------------------
class AgentError(Exception):
    error_code: str = "AGENT_ERROR"
    description: str = "An error occurred in the agent"

    def to_response(self):
        return {"success": False, "error": {"code": self.error_code, "message": self.description}}


class ValidationErrorInput(AgentError):
    error_code = "INVALID_INPUT"

    def __init__(self, description: str):
        super().__init__(description)
        self.description = description


class SizeLimitError(AgentError):
    error_code = "VALUE_TOO_LARGE"

    def __init__(self, description: str):
        super().__init__(description)
        self.description = description


class ComputationError(AgentError):
    error_code = "COMPUTATION_ERROR"

    def __init__(self, description: str):
        super().__init__(description)
        self.description = description


class TimeoutErrorAgent(AgentError):
    error_code = "TIMEOUT"

    def __init__(self, description: str):
        super().__init__(description)
        self.description = description


# ---------------------------
# Models
# ---------------------------
class MultiplyRequestModel(BaseModel):
    """
    Request body for /multiply
    Accepts either raw_input_text OR explicit value_a and value_b.
    """
    request_id: Optional[str] = None
    raw_input_text: Optional[str] = None
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    format_options: Optional[Dict[str, Any]] = None
    explanation: Optional[bool] = False

    @field_validator("raw_input_text", mode="before")
    @classmethod
    def _clean_raw_input(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("raw_input_text must be a string")
        if len(v) > 50000:
            raise ValueError("raw_input_text exceeds maximum allowed length (50,000)")
        # normalize whitespace
        return v.strip()

    @field_validator("value_a", "value_b", mode="before")
    @classmethod
    def _clean_values(cls, v):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError("value must be provided as string to preserve formatting")
        if len(v) > 50000:
            raise ValueError("value parameter exceeds maximum allowed length (50,000)")
        return v.strip()

    @field_validator("format_options", mode="before")
    @classmethod
    def _clean_format_options(cls, v):
        if v is None:
            return v
        if not isinstance(v, dict):
            raise ValueError("format_options must be a JSON object")
        return v


class BatchItemModel(BaseModel):
    request_id: Optional[str] = None
    value_a: Optional[str] = None
    value_b: Optional[str] = None
    raw_input_text: Optional[str] = None
    format_options: Optional[Dict[str, Any]] = None
    explanation: Optional[bool] = False


class BatchRequestModel(BaseModel):
    request_id: Optional[str] = None
    items: List[BatchItemModel] = Field(..., min_length=1, max_length=100)


class MultiplyResponseModel(BaseModel):
    success: bool
    request_id: str
    formatted_result: Optional[str] = None
    raw_result: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    error: Optional[Dict[str, Any]] = None


# ---------------------------
# Numeric Representation
# ---------------------------
@dataclass
class NumericRepresentation:
    """
    Internal normalized representation of numeric token.
    value: Decimal representing the numeric value
    type: 'integer' | 'float'
    sign: 1 | -1 | 0
    digits: int  # total significant digits (approx)
    frac_digits: int
    exponent: int
    original_token: str
    """
    value: Decimal
    type: str
    sign: int
    digits: int
    frac_digits: int
    exponent: int
    original_token: str


# ---------------------------
# Security & PII Redaction (simplified)
# ---------------------------
class SecurityManager:
    """
    Responsible for authentication, authorization (not implemented here), PII detection and redaction.
    """

    PII_PATTERNS = [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
        re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),  # Visa-like CC (very rough)
        # Add more patterns as needed
    ]

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def redact_pii(text: str) -> Tuple[str, bool]:
        """
        Redact obvious PII matches. Returns (redacted_text, pii_detected_flag).
        This is intentionally conservative—real deployments should use robust PII detectors.
        """
        if not text:
            return text, False
        redacted = text
        detected = False
        for p in SecurityManager.PII_PATTERNS:
            if p.search(redacted):
                detected = True
                redacted = p.sub("[REDACTED]", redacted)
        return redacted, detected

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def authenticate(request: Request) -> Optional[str]:
        """
        Simple API key authentication via header 'x-api-key'. Returns identity string or None.
        """
        api_key = request.headers.get("x-api-key")
        if api_key:
            # In production validate against AuthStore.
            return f"api_key:{api_key[:6]}..."
        return None


# ---------------------------
# InputValidator
# ---------------------------
class InputValidator:
    """
    Validate that the input contains exactly two numeric tokens, detect malicious content,
    and return the tokens for parsing.
    """

    NUMBER_RE = re.compile(
        r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    )

    @staticmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_and_extract(raw_input_text: str, config: Config) -> Tuple[str, str]:
        """
        Extract two numeric tokens from the raw text.
        Raises ValidationErrorInput if validation fails.
        """
        if not raw_input_text or not raw_input_text.strip():
            raise ValidationErrorInput("Input is empty. Provide two numeric values.")

        # Redact PII and check
        redacted, pii_detected = SecurityManager.redact_pii(raw_input_text)
        if pii_detected:
            # For safety, reject or redact but inform user.
            raise ValidationErrorInput(
                "Input contains personally identifiable information (PII) and cannot be processed."
            )

        # Reject inputs that look like code fragments or have suspicious characters
        suspicious = re.search(r"[;{}<>\\$`|&]", raw_input_text)
        if suspicious:
            raise ValidationErrorInput("Input appears to contain disallowed characters or code fragments.")

        # Find numeric tokens
        tokens = InputValidator.NUMBER_RE.findall(raw_input_text)
        if len(tokens) != 2:
            # Try splitting on whitespace or comma for structured inputs
            simple_tokens = re.split(r"[,\s]+", raw_input_text.strip())
            simple_tokens = [t for t in simple_tokens if t != ""]
            # Filter tokens that look numeric
            numeric_tokens = [t for t in simple_tokens if InputValidator.NUMBER_RE.fullmatch(t)]
            if len(numeric_tokens) == 2:
                return numeric_tokens[0], numeric_tokens[1]
            raise ValidationErrorInput(
                f"Expected exactly two numeric values but found {len(tokens)}. Provide two numbers (integers, decimals, or scientific notation)."
            )
        return tokens[0], tokens[1]


# ---------------------------
# NumericParser
# ---------------------------
class NumericParser:
    """
    Parse numeric tokens using Decimal and optional gmpy2 for integers.
    """

    @staticmethod
    def parse_numeric_token(token: str, parse_strict: bool = True) -> NumericRepresentation:
        """
        Parses token into NumericRepresentation.
        Raises ValidationErrorInput on parse failure.
        """
        if not isinstance(token, str):
            raise ValidationErrorInput("Token must be a string")

        token = token.strip()
        if token == "":
            raise ValidationErrorInput("Empty numeric token")

        # Reject tokens that contain letters (disallow hex/binary) or suspicious characters
        if re.search(r"[A-Za-z&&[^eE]]", token):  # allow 'e' or 'E' for exponent
            # fallback rejection
            raise ValidationErrorInput(f"Invalid numeric token: {token}")

        try:
            d = Decimal(token)
        except (InvalidOperation, ValueError):
            raise ValidationErrorInput(f"Unable to parse numeric token: {token}")

        sign = 0
        if d == 0:
            sign = 0
        elif d > 0:
            sign = 1
        else:
            sign = -1

        # Determine type: integer if no fractional part and exponent non-negative
        tup = d.as_tuple()
        digits = len(tup.digits)
        exponent = tup.exponent  # negative for fractional
        frac_digits = -exponent if exponent < 0 else 0
        is_integer = frac_digits == 0

        num_type = "integer" if is_integer else "float"

        return NumericRepresentation(
            value=d,
            type=num_type,
            sign=sign,
            digits=digits,
            frac_digits=frac_digits,
            exponent=exponent,
            original_token=token,
        )


# ---------------------------
# PrecisionSelector
# ---------------------------
class PrecisionSelector:
    """
    Decide whether to use standard double precision or arbitrary-precision evaluator.
    """

    @staticmethod
    def select_precision(a: NumericRepresentation, b: NumericRepresentation, config: Config) -> Dict[str, str]:
        max_digits = max(a.digits, b.digits)
        precision_mode = config.get_precision_mode()
        if precision_mode == "standard":
            return {"mode": "standard", "reason": "Forced to standard by config"}
        if precision_mode == "arbitrary":
            return {"mode": "arbitrary", "reason": "Forced to arbitrary by config"}
        # auto mode: if either operand has more than 15 digits -> arbitrary
        if max_digits > 15 or a.type == "float" and a.frac_digits > 15 or b.type == "float" and b.frac_digits > 15:
            return {"mode": "arbitrary", "reason": "High digit count or fractional precision requires arbitrary precision"}
        return {"mode": "standard", "reason": "Within double-precision safe limits"}


# ---------------------------
# MathEvaluator (double precision)
# ---------------------------
class MathEvaluator:
    """
    Fast deterministic double-precision multiplication with overflow/NaN detection.
    """

    @staticmethod
    def multiply_double(a: Decimal, b: Decimal) -> Dict[str, Any]:
        """
        Convert to float and multiply. Detect NaN/Inf/Overflow.
        Returns dict {result: float, metadata: {...}}
        """
        try:
            fa = float(a)
            fb = float(b)
            result = fa * fb
            if math.isinf(result) or math.isnan(result):
                return {"result": None, "metadata": {"status": "invalid_result", "reason": "overflow_or_nan"}}
            return {"result": result, "metadata": {"status": "ok", "engine": "double", "operands_float": (fa, fb)}}
        except Exception as e:
            return {"result": None, "metadata": {"status": "error", "reason": str(e)}}


# ---------------------------
# ArbitraryPrecisionEvaluator
# ---------------------------
class ArbitraryPrecisionEvaluator:
    """
    Use Decimal and optionally gmpy2 for integer multiplication. Observes max_digit_length limit.
    """

    @staticmethod
    def multiply_bigint_or_decimal(
        a_repr: NumericRepresentation,
        b_repr: NumericRepresentation,
        max_digit_length: int,
        rounding_mode: str = "ROUND_HALF_EVEN",
    ) -> Dict[str, Any]:
        """
        Perform arbitrary precision multiplication.
        Returns {"result": str, "metadata": {...}}
        """
        start = time.time()
        # Enforce digit limits
        if max(a_repr.digits, b_repr.digits) > max_digit_length:
            raise SizeLimitError("One or more operands exceed configured max_digit_length")

        # If both are integers and gmpy2 is available, use it for speed
        try:
            if a_repr.type == "integer" and b_repr.type == "integer" and _gmpy2:
                ai = _gmpy2.mpz(int(a_repr.value))
                bi = _gmpy2.mpz(int(b_repr.value))
                prod = ai * bi
                result_str = _gmpy2.to_binary(prod)  # not desired string; better to use str(prod)
                result_str = str(prod)
                metadata = {
                    "engine": "gmpy2",
                    "operand_digits": (a_repr.digits, b_repr.digits),
                    "compute_time_ms": int((time.time() - start) * 1000),
                }
                return {"result": result_str, "metadata": metadata}
        except Exception:
            # Fall back to Decimal-based multiplication
            logger.warning("gmpy2 multiplication failed, falling back to Decimal", exc_info=True)

        # Decimal-based approach
        try:
            # Set precision to accommodate digits of both operands plus a margin
            precision = max(a_repr.digits + b_repr.digits + 5, 28)
            if precision > max_digit_length * 2:
                precision = max_digit_length * 2 + 10
            with localcontext() as ctx:
                ctx.prec = precision
                ctx.rounding = getattr(Decimal, rounding_mode, None) or getattr(
                    Decimal, Config.get_result_rounding_mode(), None
                )  # best-effort
                product = a_repr.value * b_repr.value
                # Normalize to plain string without scientific when reasonable
                prod_str = format(product, "f")
            metadata = {
                "engine": "decimal",
                "operand_digits": (a_repr.digits, b_repr.digits),
                "compute_time_ms": int((time.time() - start) * 1000),
            }
            return {"result": prod_str, "metadata": metadata}
        except Exception as e:
            raise ComputationError(f"Arbitrary-precision multiplication failed: {e}")


# ---------------------------
# Formatter
# ---------------------------
class Formatter:
    """
    Format numeric product according to user options.
    """

    @staticmethod
    def format_result(raw_product: Union[str, float, Decimal], format_options: Optional[Dict[str, Any]] = None) -> str:
        """
        format_options may include:
            - notation: 'plain' | 'scientific' | 'commas'
            - trim_trailing_zeros: bool
            - max_decimal_places: int
        """
        fmt_opts = format_options or {}
        notation = fmt_opts.get("notation", "plain")
        trim_trailing = fmt_opts.get("trim_trailing_zeros", True)
        max_dp = fmt_opts.get("max_decimal_places", None)

        # Normalize product to Decimal for safe formatting
        try:
            if isinstance(raw_product, float):
                raw_product = Decimal(str(raw_product))
            elif isinstance(raw_product, str):
                # Could be very large integer or decimal as string
                raw_product = Decimal(raw_product)
            elif isinstance(raw_product, Decimal):
                pass
            else:
                raw_product = Decimal(str(raw_product))
        except Exception:
            # If parsing fails, return string fallback
            return str(raw_product)

        # Apply max decimal places if requested
        if max_dp is not None:
            try:
                max_dp_int = int(max_dp)
                quant = Decimal(1).scaleb(-max_dp_int)  # 1E-max_dp_int
                raw_product = raw_product.quantize(quant)
            except Exception:
                # ignore invalid max_dp and proceed
                pass

        if notation == "scientific":
            s = "{:E}".format(raw_product)
        else:
            # plain or commas
            s = format(raw_product, "f")
            if fmt_opts.get("commas", False) or notation == "commas":
                # insert commas for integer part
                if "." in s:
                    int_part, frac_part = s.split(".", 1)
                    int_part_commas = "{:,}".format(int(int_part))
                    s = f"{int_part_commas}.{frac_part}"
                else:
                    s = "{:,}".format(int(s))

        if trim_trailing and "." in s:
            s = s.rstrip("0").rstrip(".")
        return s


# ---------------------------
# CacheClient (Redis optional)
# ---------------------------
class CacheClient:
    """
    Result-level caching keyed by normalized operand pair.
    """

    def __init__(self):
        self._client = None
        self._memory_cache: Dict[str, Tuple[str, float]] = {}  # key -> (value_json, expiry_timestamp)
        self._redis_url = Config.get_redis_url()
        self._ttl_default = 24 * 3600  # 24h

    def _ensure_redis(self):
        if self._client is not None:
            return
        if self._redis_url and redis:
            try:
                self._client = redis.from_url(self._redis_url)
            except Exception:
                logger.warning("Failed to connect to Redis; falling back to in-memory cache", exc_info=True)
                self._client = None
        else:
            self._client = None

    def _mem_get(self, key: str) -> Optional[str]:
        v = self._memory_cache.get(key)
        if not v:
            return None
        value_json, expiry = v
        if expiry < time.time():
            del self._memory_cache[key]
            return None
        return value_json

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        self._ensure_redis()
        if self._client:
            try:
                _obs_t0 = _time.time()
                raw = self._client.get(key)
                try:
                    trace_tool_call(
                        tool_name='_client.get',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=str(raw)[:200] if raw is not None else None,
                        status="success",
                    )
                except Exception:
                    pass
                if raw:
                    return json.loads(raw)
                return None
            except Exception:
                logger.warning("Redis get failed; using memory cache", exc_info=True)
                return json.loads(self._mem_get(key)) if self._mem_get(key) else None
        else:
            raw = self._mem_get(key)
            return json.loads(raw) if raw else None

    def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None):
        ttl = ttl or self._ttl_default
        self._ensure_redis()
        v = json.dumps(value)
        if self._client:
            try:
                self._client.set(key, v, ex=ttl)
                return
            except Exception:
                logger.warning("Redis set failed; falling back to memory cache", exc_info=True)
        self._memory_cache[key] = (v, time.time() + ttl)

    def invalidate(self, key: str):
        self._ensure_redis()
        if self._client:
            try:
                _obs_t0 = _time.time()
                self._client.delete(key)
                try:
                    trace_tool_call(
                        tool_name='_client.delete',
                        latency_ms=int((_time.time() - _obs_t0) * 1000),
                        output=None,
                        status="success",
                    )
                except Exception:
                    pass
            except Exception:
                logger.warning("Redis delete failed", exc_info=True)
        if key in self._memory_cache:
            del self._memory_cache[key]


# ---------------------------
# AuditLoggerClient (append-only JSONL)
# ---------------------------
class AuditLoggerClient:
    """
    Append-only audit logging to a file with redaction.
    """

    def __init__(self, path: str = "audit_log.jsonl"):
        self.path = path

    def log(self, request_id: str, user_id: Optional[str], input_snapshot: str, result: Dict[str, Any], status: str, metadata: Dict[str, Any]) -> str:
        entry = {
            "ts": time.time(),
            "request_id": request_id,
            "user_id": user_id,
            "input_snapshot": input_snapshot,
            "result": result,
            "status": status,
            "metadata": metadata,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return f"log:{request_id}"
        except Exception:
            logger.exception("Failed to write audit log")
            # Non-blocking: return an indicator
            return f"log_failed:{request_id}"


# ---------------------------
# ErrorHandler
# ---------------------------
class ErrorHandler:
    """
    Map exceptions to user-facing responses and error codes.
    """

    @staticmethod
    def handle_error(context: str, exc: Exception) -> Dict[str, Any]:
        logger.error("Error in %s: %s", context, str(exc), exc_info=True)
        if isinstance(exc, ValidationErrorInput):
            return {"success": False, "error": {"code": exc.error_code, "message": exc.description, "tips": "Provide exactly two numeric values (e.g., '3 4' or '2.5, 0.12')."}}
        if isinstance(exc, SizeLimitError):
            return {"success": False, "error": {"code": exc.error_code, "message": exc.description, "tips": f"Ensure each value has <= {Config.get_max_digit_length()} digits or request asynchronous HITL processing."}}
        if isinstance(exc, ComputationError):
            return {"success": False, "error": {"code": exc.error_code, "message": exc.description, "tips": "Try smaller inputs or contact support."}}
        if isinstance(exc, TimeoutErrorAgent):
            return {"success": False, "error": {"code": exc.error_code, "message": exc.description, "tips": "Request timed out; try again or use smaller inputs."}}
        # generic
        return {"success": False, "error": {"code": "INTERNAL_ERROR", "message": str(exc), "tips": "Contact support if the issue persists."}}


# ---------------------------
# HITLManager (stub)
# ---------------------------
class HITLManager:
    """
    Create a placeholder HTIL review task.
    """

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def create_review_task(self, payload: Dict[str, Any]) -> str:
        task_id = "hitl:" + str(uuid.uuid4())
        # In real system, push to queue. Here we just log.
        logger.info("Created HITL task %s for payload summary", task_id)
        return task_id

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def query_task(self, task_id: str) -> Dict[str, Any]:
        return {"task_id": task_id, "status": "pending"}


# ---------------------------
# LLM Integration (OpenAI async) - Lazy client creation
# ---------------------------
@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    """
    Lazy initialization of OpenAI Async client.
    Raises ValueError if OPENAI_API_KEY is not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    if openai is None:
        raise RuntimeError("openai package is not installed in this environment")
    return openai.AsyncOpenAI(api_key=api_key)


@with_content_safety(config=GUARDRAILS_CONFIG)
async def ask_llm_for_explanation(value1: str, value2: str, raw_result: str, system_prompt: str, user_template: str) -> str:
    """
    Ask LLM to produce a brief explanation of the computation.
    This function is tolerant to missing API key and will raise a ValueError in that case.
    """
    client = get_llm_client()
    model = Config.get_openai_model()
    temp = Config.get_openai_temperature()
    max_toks = Config.get_openai_max_tokens()

    system_msg = system_prompt
    user_msg = user_template.format(value1=value1, value2=value2) + f" The computed product is {raw_result}. Provide a brief explanation of steps and precision used."

    # wrap in trace step
    async with _get_async_trace_step(
        "llm_explanation", step_type="llm_call",
        decision_summary="Generate brief explanation for multiplication",
        output_fn=lambda r: f"len={len(r) if r else 0}",
    ) as step:
        try:
            _obs_t0 = _time.time()
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temp,
                max_tokens=max_toks,
            )
            try:
                trace_model_call(
                    provider='openai',
                    model_name=(getattr(self, "model", None) or getattr(getattr(self, "config", None), "model", None) or "unknown"),
                    prompt_tokens=(getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0),
                    completion_tokens=(getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0),
                    latency_ms=int((_time.time() - _obs_t0) * 1000),
                )
            except Exception:
                pass
            content = None
            try:
                content = response.choices[0].message.content
            except Exception:
                # Best-effort extraction for different response shapes
                try:
                    content = response.choices[0].text
                except Exception:
                    content = None
            step.capture({"llm_response_snippet": (content or "")[:200]})
            return content or ""
        except Exception as e:
            logger.warning("LLM explanation request failed: %s", str(e))
            step.capture({"llm_error": str(e)})
            raise


# ---------------------------
# Orchestrator
# ---------------------------
class Orchestrator:
    """
    Coordinates validation, parsing, precision selection, computation, formatting, caching, and audit logging.
    """

    def __init__(self):
        self.validator = InputValidator()
        self.parser = NumericParser()
        self.selector = PrecisionSelector()
        self.double_eval = MathEvaluator()
        self.big_eval = ArbitraryPrecisionEvaluator()
        self.formatter = Formatter()
        self.cache = CacheClient()
        self.audit = AuditLoggerClient()
        self.error_handler = ErrorHandler()
        self.hitl = HITLManager()
        self.config = Config()

    async def execute_multiplication(
        self,
        request_id: str,
        raw_input_text: Optional[str] = None,
        explicit_a: Optional[str] = None,
        explicit_b: Optional[str] = None,
        format_options: Optional[Dict[str, Any]] = None,
        explanation: bool = False,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Orchestrate the multiplication for a single request. Returns structured response dict.
        """
        # Stage: parse/validate input
        async with _get_async_trace_step(
            "parse_input", step_type="parse",
            decision_summary="Extract and validate two numeric tokens",
            output_fn=lambda r: f"tokens={r if isinstance(r, list) else '?'}",
        ) as step:
            try:
                if explicit_a is not None and explicit_b is not None:
                    token_a, token_b = explicit_a, explicit_b
                elif raw_input_text is not None:
                    token_a, token_b = self.validator.validate_and_extract(raw_input_text, self.config)
                else:
                    raise ValidationErrorInput("No input provided")
                step.capture([token_a, token_b])
            except Exception as e:
                step.capture({"error": str(e)})
                raise

        # Stage: parse numeric tokens
        async with _get_async_trace_step(
            "numeric_parse", step_type="parse",
            decision_summary="Parse numeric tokens into normalized representations",
            output_fn=lambda r: f"a_type={r[0].type if r else '?'}",
        ) as step:
            try:
                a_repr = self.parser.parse_numeric_token(token_a, parse_strict=True)
                b_repr = self.parser.parse_numeric_token(token_b, parse_strict=True)
                step.capture({"a_digits": a_repr.digits, "b_digits": b_repr.digits})
            except Exception as e:
                step.capture({"error": str(e)})
                raise

        # Stage: precision selection
        async with _get_async_trace_step(
            "select_precision", step_type="process",
            decision_summary="Choose computation engine",
            output_fn=lambda r: f"mode={r.get('mode','?')}",
        ) as step:
            try:
                mode_choice = self.selector.select_precision(a_repr, b_repr, self.config)
                mode = mode_choice["mode"]
                step.capture(mode_choice)
            except Exception as e:
                step.capture({"error": str(e)})
                raise

        # Normalize cache key
        cache_key = f"prod:{a_repr.original_token}:{b_repr.original_token}:mode={mode}"
        cached = self.cache.get(cache_key)
        if cached:
            # Return cached entry but still write an audit record (non-blocking)
            self.audit.log(request_id, user_id, f"{a_repr.original_token},{b_repr.original_token}", cached.get("result", {}), "ok_cached", {"cache_hit": True})
            return {"success": True, "request_id": request_id, "formatted_result": cached.get("formatted_result"), "raw_result": cached.get("raw_result"), "metadata": {"cached": True}}

        # Stage: compute
        async with _get_async_trace_step(
            "generate_response", step_type="llm_call" if explanation else "tool_call",
            decision_summary="Call computation engine to produce product",
            output_fn=lambda r: f"len_raw_result={len(str(r)) if r else 0}",
        ) as step:
            raw_result = None
            compute_metadata = {}
            try:
                if mode == "standard":
                    # Convert to floats and multiply deterministically
                    double_out = self.double_eval.multiply_double(a_repr.value, b_repr.value)
                    if double_out["result"] is None:
                        # Escalate to arbitrary precision
                        logger.warning("Double-precision produced invalid result; escalating to arbitrary precision")
                        mode = "arbitrary"
                        compute_metadata["fallback_reason"] = double_out["metadata"].get("reason")
                        arb = self.big_eval.multiply_bigint_or_decimal(a_repr, b_repr, self.config.get_max_digit_length())
                        raw_result = arb["result"]
                        compute_metadata.update(arb.get("metadata", {}))
                    else:
                        raw_result = str(Decimal(str(double_out["result"])))
                        compute_metadata.update(double_out.get("metadata", {}))
                else:
                    arb = self.big_eval.multiply_bigint_or_decimal(a_repr, b_repr, self.config.get_max_digit_length())
                    raw_result = arb["result"]
                    compute_metadata.update(arb.get("metadata", {}))
                step.capture({"raw_result_snippet": (raw_result or "")[:200], "mode": mode})
            except SizeLimitError as sle:
                step.capture({"error": str(sle)})
                # Create HITL task and inform user
                task_id = self.hitl.create_review_task({"request_id": request_id, "a": a_repr.original_token, "b": b_repr.original_token})
                self.audit.log(request_id, user_id, f"{a_repr.original_token},{b_repr.original_token}", {}, "escalated", {"hitl_task": task_id})
                raise SizeLimitError(str(sle))
            except Exception as e:
                step.capture({"error": str(e)})
                raise ComputationError(f"Computation failed: {e}")

        # Stage: formatting
        async with _get_async_trace_step(
            "format_result", step_type="format",
            decision_summary="Format the numeric product per options",
            output_fn=lambda r: f"len_formatted={len(r) if r else 0}",
        ) as step:
            try:
                formatted = self.formatter.format_result(raw_result, format_options)
                step.capture(formatted)
            except Exception as e:
                step.capture({"error": str(e)})
                formatted = str(raw_result)

        # Stage: audit logging
        async with _get_async_trace_step(
            "audit_log", step_type="tool_call",
            decision_summary="Write redacted audit log",
            output_fn=lambda r: f"log_id={r}",
        ) as step:
            try:
                redacted_input, _pii = SecurityManager.redact_pii(f"{a_repr.original_token},{b_repr.original_token}")
                log_id = self.audit.log(request_id, user_id, redacted_input, {"raw_result": raw_result, "formatted": formatted}, "ok", {"compute_metadata": compute_metadata})
                step.capture(log_id)
            except Exception as e:
                logger.warning("Audit logging failed: %s", str(e))
                step.capture({"audit_error": str(e)})

        # Store in cache
        try:
            cache_entry = {"formatted_result": formatted, "raw_result": raw_result, "metadata": compute_metadata}
            self.cache.set(cache_key, cache_entry)
        except Exception:
            logger.warning("Cache set failed", exc_info=True)

        # Optionally ask LLM for explanation
        explanation_text = None
        if explanation:
            try:
                system_prompt = (
                    os.getenv("LLM_SYSTEM_PROMPT")
                    or "You are Numeric Multiplication Assistant, a professional and precise agent. Validate that the user has provided exactly two numeric values. If validation passes, compute the product deterministically using the appropriate precision engine. Return the numeric product first. Only include an explanation if explicitly requested."
                )
                user_template = os.getenv("LLM_USER_PROMPT_TEMPLATE") or "Multiply {value1} and {value2}."
                explanation_text = await ask_llm_for_explanation(a_repr.original_token, b_repr.original_token, raw_result, system_prompt, user_template)
            except Exception as e:
                # LLM errors should not block primary result
                logger.warning("LLM explanation failed: %s", str(e))

        return {
            "success": True,
            "request_id": request_id,
            "formatted_result": formatted,
            "raw_result": raw_result,
            "metadata": {"mode": mode, **compute_metadata},
            "explanation": explanation_text,
        }

    async def execute_batch_multiplication(self, request_id: str, items: List[BatchItemModel]) -> List[Dict[str, Any]]:
        """
        Process a batch of multiplication items. Each item is processed independently.
        """
        results = []
        for it in items:
            rid = it.request_id or f"{request_id}:{uuid.uuid4()}"
            try:
                r = await self.execute_multiplication(
                    request_id=rid,
                    raw_input_text=it.raw_input_text,
                    explicit_a=it.value_a,
                    explicit_b=it.value_b,
                    format_options=it.format_options,
                    explanation=it.explanation,
                )
                results.append(r)
            except Exception as e:
                results.append(ErrorHandler.handle_error("batch_item", e))
        return results


# ---------------------------
# API Layer (FastAPI)
# ---------------------------
app = FastAPI(title="Numeric Multiplication Assistant", version="1.0.0")

# Allow CORS for interactive use if configured
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = Orchestrator()


# JSON parsing error handler
@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"success": False, "error": {"code": "INVALID_INPUT", "message": "Malformed request input", "details": str(exc), "tips": "Check JSON formatting, required fields, and value types."}},
    )


@app.exception_handler(json.JSONDecodeError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def json_decode_exception_handler(request: Request, exc: json.JSONDecodeError):
    logger.warning("JSON decode error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": {"code": "INVALID_JSON", "message": "Malformed JSON in request body", "details": str(exc), "tips": "Ensure request body is valid JSON (check quotes, commas, braces)."}},
    )


@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    resp = ErrorHandler.handle_error("api", exc)
    status_code = 500
    if isinstance(exc, AgentError):
        if exc.error_code == "INVALID_INPUT":
            status_code = 400
        elif exc.error_code == "VALUE_TOO_LARGE":
            status_code = 400
        elif exc.error_code == "TIMEOUT":
            status_code = 504
    return JSONResponse(status_code=status_code, content=resp)


@app.get("/health")
async def health():
    return PlainTextResponse("ok")


@app.get("/metrics")
async def metrics():
    # Minimal metrics endpoint - replace with Prometheus exporter in production.
    return JSONResponse({"uptime": time.time(), "service": "numeric_multiplication_assistant"})


@app.post("/multiply", response_model=MultiplyResponseModel)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def multiply(req: MultiplyRequestModel, request: Request):
    """
    Single multiply endpoint. Accepts either raw_input_text or explicit value_a and value_b.
    Returns formatted_result (string) and metadata.
    """
    request_id = req.request_id or str(uuid.uuid4())
    user_id = SecurityManager.authenticate(request)
    try:
        # entry trace
        with _get_sync_trace_step(
            "api_entry",
            step_type="process",
            decision_summary="API entry and auth",
            output_fn=lambda r: f"request_id={r}",
        ) as step:
            step.capture(request_id)

        resp = await orchestrator.execute_multiplication(
            request_id=request_id,
            raw_input_text=req.raw_input_text,
            explicit_a=req.value_a,
            explicit_b=req.value_b,
            format_options=req.format_options,
            explanation=bool(req.explanation),
            user_id=user_id,
        )
        if not resp.get("success"):
            raise Exception(resp.get("error") or "Computation failed")
        return MultiplyResponseModel(
            success=True,
            request_id=request_id,
            formatted_result=resp.get("formatted_result"),
            raw_result=resp.get("raw_result"),
            metadata=resp.get("metadata"),
            explanation=resp.get("explanation"),
            error=None,
        )
    except Exception as e:
        handled = ErrorHandler.handle_error("multiply_endpoint", e)
        return JSONResponse(status_code=400 if isinstance(e, ValidationErrorInput) else 500, content=handled)


@app.post("/batch_multiply")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def batch_multiply(req: BatchRequestModel, request: Request):
    request_id = req.request_id or str(uuid.uuid4())
    user_id = SecurityManager.authenticate(request)
    try:
        results = await orchestrator.execute_batch_multiplication(request_id, req.items)
        return JSONResponse({"success": True, "request_id": request_id, "results": results})
    except Exception as e:
        handled = ErrorHandler.handle_error("batch_multiply", e)
        return JSONResponse(status_code=500, content=handled)


# ---------------------------
# CLI / execution entrypoint
# ---------------------------


async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        logger.info("Starting Numeric Multiplication Assistant on %s:%s", host, port)
        uvicorn.run("agent:app", host=host, port=port, log_level="info")
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())