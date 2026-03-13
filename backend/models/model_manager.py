"""
Model Manager for handling multiple LLM providers with fallbacks and load balancing.

This module provides a centralized interface for managing different LLM providers
and automatically handling fallbacks when primary models are unavailable.
"""

import os
import time
import logging
from collections import deque
from threading import Lock
from typing import Dict, Any, Optional, List, Deque, Tuple
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import (
    ConfigurationManager,
    TaskType,
    LLMProvider,
    ModelConfig,
    get_config
)

logger = logging.getLogger(__name__)


class ModelInstantiationError(Exception):
    """Exception raised when model instantiation fails."""

    pass


class SimpleRateLimiter:
    """Simple per-minute rate limiter for blocking LLM calls."""

    def __init__(
        self,
        requests_per_minute: Optional[int] = None,
        tokens_per_minute: Optional[int] = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._lock = Lock()
        self._request_times: Deque[float] = deque()
        self._token_usage: Deque[Tuple[float, int]] = deque()
        self._token_total = 0
        self.window_seconds = 60

    @property
    def enabled(self) -> bool:
        return bool(self.requests_per_minute or self.tokens_per_minute)

    def _cleanup(self, now: float):
        """Drop entries outside the rolling window."""
        while self._request_times and now - self._request_times[0] > self.window_seconds:
            self._request_times.popleft()

        while self._token_usage and now - self._token_usage[0][0] > self.window_seconds:
            _, tokens = self._token_usage.popleft()
            self._token_total -= tokens

    def _required_wait(self, now: float, tokens: int) -> float:
        wait_time = 0.0

        if self.requests_per_minute and len(self._request_times) >= self.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = max(wait_time, self.window_seconds - (now - oldest))

        if self.tokens_per_minute and (self._token_total + tokens) > self.tokens_per_minute:
            if self._token_usage:
                oldest = self._token_usage[0][0]
                wait_time = max(wait_time, self.window_seconds - (now - oldest))
            else:
                wait_time = max(wait_time, self.window_seconds)

        return wait_time

    def acquire(self, tokens: int = 0):
        """Block until a request slot is available."""
        if not self.enabled:
            return

        while True:
            with self._lock:
                now = time.time()
                self._cleanup(now)
                wait_time = self._required_wait(now, tokens)

                if wait_time <= 0:
                    self._request_times.append(now)
                    if self.tokens_per_minute and tokens:
                        self._token_usage.append((now, tokens))
                        self._token_total += tokens
                    return

            time.sleep(min(wait_time, 1.0))

    def time_until_ready(self) -> float:
        """Return seconds until the next request would be allowed."""
        if not self.enabled:
            return 0.0

        with self._lock:
            now = time.time()
            self._cleanup(now)
            return self._required_wait(now, 0)

    def is_available(self) -> bool:
        return self.time_until_ready() <= 0


class RateLimitedModelWrapper:
    """Proxy object that enforces rate limits before delegating to the model."""

    def __init__(self, model: BaseChatModel, limiter: SimpleRateLimiter, name: str):
        self._model = model
        self._limiter = limiter
        self._name = name

    def invoke(self, *args, **kwargs):
        estimated_tokens = kwargs.pop("_estimated_tokens", 0)
        self._limiter.acquire(estimated_tokens)
        return self._model.invoke(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        estimated_tokens = kwargs.pop("_estimated_tokens", 0)
        self._limiter.acquire(estimated_tokens)
        return await self._model.ainvoke(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._model, item)


class ModelManager:
    """
    Central manager for LLM instances with fallback support and load balancing.

    This class handles the creation and management of LLM instances from different
    providers, implements fallback logic when models are unavailable, and provides
    a unified interface for all LLM interactions.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initialize the model manager.

        Args:
            config_manager: Optional configuration manager instance
        """
        self.config_manager = config_manager or get_config()
        self._model_cache: Dict[str, BaseChatModel] = {}
        self._model_health: Dict[str, bool] = {}
        self._last_health_check: Dict[str, float] = {}
        self.health_check_interval = 300  # 5 minutes
        self._rate_limiters: Dict[LLMProvider, SimpleRateLimiter] = {}
        self._init_rate_limiters()

    def _init_rate_limiters(self):
        """Initialize provider-level rate limiter objects."""
        for provider in LLMProvider:
            rate_limit = self.config_manager.config.get_rate_limit(provider)
            if not rate_limit:
                continue

            limiter = SimpleRateLimiter(
                requests_per_minute=rate_limit.requests_per_minute,
                tokens_per_minute=rate_limit.tokens_per_minute,
            )
            if limiter.enabled:
                self._rate_limiters[provider] = limiter

    def _wrap_with_rate_limiter(
        self, model: BaseChatModel, provider: LLMProvider, model_name: str
    ) -> BaseChatModel:
        limiter = self._rate_limiters.get(provider)
        if limiter:
            return RateLimitedModelWrapper(model, limiter, model_name)
        return model

    def _create_openai_model(self, config: ModelConfig) -> ChatOpenAI:
        """Create an OpenAI model instance."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.max_tokens:
            params["max_tokens"] = config.max_tokens

        if config.base_url:
            params["base_url"] = config.base_url

        params.update(config.custom_params)

        return ChatOpenAI(**params)

    def _create_anthropic_model(self, config: ModelConfig) -> ChatAnthropic:
        """Create an Anthropic model instance."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.max_tokens:
            params["max_tokens"] = config.max_tokens

        params.update(config.custom_params)

        return ChatAnthropic(**params)

    def _create_groq_model(self, config: ModelConfig) -> ChatGroq:
        """Create a Groq model instance."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "groq_api_key": api_key,
            "max_retries": config.max_retries,
        }

        if config.max_tokens:
            params["max_tokens"] = config.max_tokens

        params.update(config.custom_params)

        return ChatGroq(**params)

    def _create_google_model(self, config: ModelConfig) -> ChatGoogleGenerativeAI:
        """Create a Gemini model instance (uses Google Generative AI API)."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
        }

        if config.max_tokens:
            params["max_tokens"] = config.max_tokens

        params.update(config.custom_params)

        return ChatGoogleGenerativeAI(**params)

    def _create_deepseek_model(self, config: ModelConfig) -> ChatOpenAI:
        """Create a DeepSeek R1 model instance via the OpenAI-compatible API."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": api_key,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "base_url": config.base_url or "https://api.deepseek.com",
        }

        if config.max_tokens:
            params["max_tokens"] = config.max_tokens

        params.update(config.custom_params)

        return ChatOpenAI(**params)

    def _create_model_instance(self, model_name: str) -> BaseChatModel:
        """Create a model instance based on provider."""
        config = self.config_manager.config.get_model_config(model_name)

        try:
            if config.provider == LLMProvider.OPENAI:
                model_instance = self._create_openai_model(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                model_instance = self._create_anthropic_model(config)
            elif config.provider == LLMProvider.GROQ:
                model_instance = self._create_groq_model(config)
            elif config.provider == LLMProvider.GOOGLE:
                model_instance = self._create_google_model(config)
            elif config.provider == LLMProvider.DEEPSEEK:
                model_instance = self._create_deepseek_model(config)
            else:
                raise ModelInstantiationError(
                    f"Unsupported provider: {config.provider}"
                )

            return self._wrap_with_rate_limiter(model_instance, config.provider, model_name)

        except Exception as e:
            logger.error(f"Failed to create model {model_name}: {e}")
            raise ModelInstantiationError(f"Failed to create model {model_name}: {e}")

    def _is_model_healthy(self, model_name: str) -> bool:
        """Check if a model is healthy and available."""
        current_time = time.time()
        last_check = self._last_health_check.get(model_name, 0)

        # Use cached health status if recent
        if current_time - last_check < self.health_check_interval:
            return self._model_health.get(model_name, False)

        try:
            # Simple health check - try to get the model instance
            model = self.get_model(model_name, use_fallback=False)
            self._model_health[model_name] = True
            self._last_health_check[model_name] = current_time
            return True

        except Exception as e:
            logger.warning(f"Health check failed for {model_name}: {e}")
            self._model_health[model_name] = False
            self._last_health_check[model_name] = current_time
            return False

    def get_model(self, model_name: str, use_fallback: bool = True) -> BaseChatModel:
        """
        Get a model instance with optional fallback support.

        Args:
            model_name: Name of the model to retrieve
            use_fallback: Whether to use fallback models if primary fails

        Returns:
            BaseChatModel instance

        Raises:
            ModelInstantiationError: If model creation fails and no fallbacks available
        """
        # Try to get from cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name]

        # Try to create the requested model
        try:
            model = self._create_model_instance(model_name)
            self._model_cache[model_name] = model
            self._model_health[model_name] = True
            return model

        except ModelInstantiationError as e:
            logger.warning(f"Failed to create model {model_name}: {e}")

            if not use_fallback:
                raise

            # Mark model as unhealthy
            self._model_health[model_name] = False

            # Try fallback models if enabled
            if self.config_manager.config.enable_fallbacks:
                logger.info(f"Attempting fallback for {model_name}")
                return self._try_fallback_models(model_name)

            raise ModelInstantiationError(
                f"Model {model_name} unavailable and fallbacks disabled"
            )

    def _try_fallback_models(self, original_model: str) -> BaseChatModel:
        """Try fallback models based on task type."""
        # Determine task type for the original model
        task_type = None
        for t_type, assigned_model in self.config_manager.config.task_models.items():
            if assigned_model == original_model:
                task_type = t_type
                break

        if not task_type:
            # Default fallback chain
            fallback_models = ["llama-3.3-70b-versatile", "gpt-4o-mini"]
        else:
            fallback_models = self.config_manager.config.get_fallback_chain(task_type)

        for fallback_model in fallback_models:
            if fallback_model == original_model:
                continue  # Skip the original model

            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                model = self._create_model_instance(fallback_model)
                self._model_cache[fallback_model] = model
                self._model_health[fallback_model] = True
                return model

            except ModelInstantiationError as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue

        raise ModelInstantiationError(
            f"All fallback models failed for {original_model}"
        )

    def _get_model_cost(self, model_name: str) -> float:
        """Return combined input/output costs for ordering."""
        config = self.config_manager.config.get_model_config(model_name)
        if config.cost_per_1k_input is None and config.cost_per_1k_output is None:
            return float("inf")
        return (config.cost_per_1k_input or 0.0) + (config.cost_per_1k_output or 0.0)

    def _is_model_selectable(self, model_name: str) -> bool:
        """Determine if a model can be used right now."""
        try:
            config = self.config_manager.config.get_model_config(model_name)
        except ValueError:
            return False

        api_key = os.getenv(config.api_key_env)
        if not api_key:
            return False

        if self._model_health.get(model_name) is False:
            return False

        limiter = self._rate_limiters.get(config.provider)
        if limiter and not limiter.is_available():
            return False

        return True

    def _select_cost_optimized_model(self, task_type: TaskType, primary_model: str) -> str:
        """Choose the cheapest healthy model for the given task."""
        candidate_chain = [primary_model]
        candidate_chain.extend(self.config_manager.config.get_fallback_chain(task_type))

        candidates: List[Tuple[float, str]] = []
        seen = set()

        for candidate in candidate_chain:
            if candidate in seen:
                continue
            seen.add(candidate)

            if not self._is_model_selectable(candidate):
                continue

            candidates.append((self._get_model_cost(candidate), candidate))

        if not candidates:
            return primary_model

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates[0][1]

    def get_model_for_task(self, task_type: TaskType) -> BaseChatModel:
        """
        Get the appropriate model for a specific task type.

        Args:
            task_type: Type of task requiring a model

        Returns:
            BaseChatModel instance
        """
        model_name = self.config_manager.config.get_model_for_task(task_type)
        if self.config_manager.config.cost_optimization:
            model_name = self._select_cost_optimized_model(task_type, model_name)
        return self.get_model(model_name)

    def list_available_models(self) -> List[str]:
        """Get list of currently available models."""
        available_models = []

        for model_name in self.config_manager.config.models.keys():
            if self._is_model_selectable(model_name):
                available_models.append(model_name)

        return available_models

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
        self._model_health.clear()
        self._last_health_check.clear()

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a specific model."""
        config = self.config_manager.config.get_model_config(model_name)
        is_healthy = self._is_model_healthy(model_name)
        is_cached = model_name in self._model_cache

        return {
            "name": model_name,
            "provider": config.provider.value,
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "is_healthy": is_healthy,
            "is_cached": is_cached,
            "api_key_configured": bool(os.getenv(config.api_key_env)),
        }

    @property
    def config(self):
        """Backwards compatibility accessor for the configuration object."""
        return self.config_manager.config
