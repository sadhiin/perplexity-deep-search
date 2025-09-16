"""
Model Manager for handling multiple LLM providers with fallbacks and load balancing.

This module provides a centralized interface for managing different LLM providers
and automatically handling fallbacks when primary models are unavailable.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List, Union
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from groq import Groq

from config import (
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

    def _create_groq_model(self, config: ModelConfig) -> 'GroqChatModel':
        """Create a Groq model instance."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        # Return a wrapper that implements the ChatModel interface
        return GroqChatModel(config)

    def _create_deepseek_model(self, config: ModelConfig) -> ChatOpenAI:
        """Create a DeepSeek model instance (uses OpenAI-compatible API)."""
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        params = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": api_key,
            "base_url": config.base_url or "https://api.deepseek.com",
            "timeout": config.timeout,
            "max_retries": config.max_retries,
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
                return self._create_openai_model(config)
            elif config.provider == LLMProvider.ANTHROPIC:
                return self._create_anthropic_model(config)
            elif config.provider == LLMProvider.GROQ:
                return self._create_groq_model(config)
            elif config.provider == LLMProvider.DEEPSEEK:
                return self._create_deepseek_model(config)
            else:
                raise ModelInstantiationError(f"Unsupported provider: {config.provider}")

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

            raise ModelInstantiationError(f"Model {model_name} unavailable and fallbacks disabled")

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

        raise ModelInstantiationError(f"All fallback models failed for {original_model}")

    def get_model_for_task(self, task_type: TaskType) -> BaseChatModel:
        """
        Get the appropriate model for a specific task type.

        Args:
            task_type: Type of task requiring a model

        Returns:
            BaseChatModel instance
        """
        model_name = self.config_manager.config.get_model_for_task(task_type)
        return self.get_model(model_name)

    def list_available_models(self) -> List[str]:
        """Get list of currently available models."""
        available_models = []

        for model_name in self.config_manager.config.models.keys():
            if self._is_model_healthy(model_name):
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
            "api_key_configured": bool(os.getenv(config.api_key_env))
        }


class GroqChatModel(BaseChatModel):
    """
    Wrapper for Groq models to implement ChatModel interface.

    This class wraps the Groq client to be compatible with LangChain's
    BaseChatModel interface.
    """

    def __init__(self, config: ModelConfig):
        """Initialize Groq chat model."""
        super().__init__()
        api_key = os.getenv(config.api_key_env)
        if not api_key:
            raise ModelInstantiationError(f"Missing API key: {config.api_key_env}")

        self.client = Groq(api_key=api_key)
        self.model_name = config.model_name
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.timeout = config.timeout
        self.max_retries = config.max_retries

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """Generate response using Groq API."""
        # Convert LangChain messages to Groq format
        groq_messages = []
        for msg in messages:
            if hasattr(msg, 'content'):
                if msg.__class__.__name__ == 'HumanMessage':
                    groq_messages.append({"role": "user", "content": msg.content})
                elif msg.__class__.__name__ == 'AIMessage':
                    groq_messages.append({"role": "assistant", "content": msg.content})
                elif msg.__class__.__name__ == 'SystemMessage':
                    groq_messages.append({"role": "system", "content": msg.content})

        # Make API call to Groq
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=groq_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **kwargs
            )

            # Convert response to LangChain format
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            ai_message = AIMessage(content=response.choices[0].message.content)
            generation = ChatGeneration(message=ai_message)
            return ChatResult(generations=[generation])

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM type."""
        return "groq_chat"