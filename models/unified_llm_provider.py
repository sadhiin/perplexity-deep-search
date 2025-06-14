"""
Unified LLM Provider Interface

This module provides a unified interface for working with different LLM providers
(OpenAI, Groq, Anthropic, Google, DeepSeek, etc.) in a modular and maintainable way.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from config import ModelConfig, LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers."""
    content: str
    provider: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, config: ModelConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self.provider_name = config.provider.value
        self.model_name = config.model_name
        self.api_key = os.getenv(config.api_key_env)

        if not self.api_key:
            raise ValueError(f"API key not found: {config.api_key_env}")

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the provider-specific client."""
        pass

    @abstractmethod
    def invoke(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Invoke the LLM with messages and return standardized response."""
        pass

    @abstractmethod
    def stream(self, messages: List[BaseMessage], **kwargs):
        """Stream response from the LLM."""
        pass

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to provider-specific format."""
        formatted_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted_messages.append({"role": "system", "content": msg.content})
            else:
                # Fallback for unknown message types
                formatted_messages.append({"role": "user", "content": str(msg.content)})

        return formatted_messages

    def _get_request_params(self, **kwargs) -> Dict[str, Any]:
        """Get standardized request parameters."""
        params = {
            "model": self.model_name,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }

        # Add custom parameters from config
        params.update(self.config.custom_params)

        # Override with any provided kwargs
        params.update(kwargs)

        # Remove None values
        return {k: v for k, v in params.items() if v is not None}


class GroqProvider(BaseLLMProvider):
    """Groq LLM provider implementation."""

    def _create_client(self):
        """Create Groq client."""
        from groq import Groq
        return Groq(api_key=self.api_key)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = self._create_client()

    def invoke(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Invoke Groq model."""
        try:
            formatted_messages = self._format_messages(messages)
            params = self._get_request_params(**kwargs)

            response = self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.provider_name,
                model=self.model_name,
                usage=response.usage._asdict() if hasattr(response, 'usage') else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            raise

    def stream(self, messages: List[BaseMessage], **kwargs):
        """Stream response from Groq."""
        try:
            formatted_messages = self._format_messages(messages)
            params = self._get_request_params(stream=True, **kwargs)

            stream = self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Groq streaming failed: {e}")
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def _create_client(self):
        """Create OpenAI client."""
        from openai import OpenAI
        client_kwargs = {"api_key": self.api_key}

        if self.config.base_url:
            client_kwargs["base_url"] = self.config.base_url

        return OpenAI(**client_kwargs)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = self._create_client()

    def invoke(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Invoke OpenAI model."""
        try:
            formatted_messages = self._format_messages(messages)
            params = self._get_request_params(**kwargs)

            response = self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.provider_name,
                model=self.model_name,
                usage=response.usage.model_dump() if hasattr(response, 'usage') else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )

        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise

    def stream(self, messages: List[BaseMessage], **kwargs):
        """Stream response from OpenAI."""
        try:
            formatted_messages = self._format_messages(messages)
            params = self._get_request_params(stream=True, **kwargs)

            stream = self.client.chat.completions.create(
                messages=formatted_messages,
                **params
            )

            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic LLM provider implementation."""

    def _create_client(self):
        """Create Anthropic client."""
        from anthropic import Anthropic
        return Anthropic(api_key=self.api_key)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = self._create_client()

    def invoke(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Invoke Anthropic model."""
        try:
            # Anthropic requires system message to be separate
            system_message = ""
            user_messages = []

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system_message = msg.content
                else:
                    user_messages.append(msg)

            formatted_messages = self._format_messages(user_messages)
            params = self._get_request_params(**kwargs)

            # Remove unsupported parameters for Anthropic
            anthropic_params = {k: v for k, v in params.items()
                              if k in ["model", "max_tokens", "temperature"]}

            if system_message:
                anthropic_params["system"] = system_message

            response = self.client.messages.create(
                messages=formatted_messages,
                **anthropic_params
            )

            return LLMResponse(
                content=response.content[0].text,
                provider=self.provider_name,
                model=self.model_name,
                usage=response.usage.__dict__ if hasattr(response, 'usage') else None,
                metadata={"stop_reason": response.stop_reason}
            )

        except Exception as e:
            logger.error(f"Anthropic API call failed: {e}")
            raise

    def stream(self, messages: List[BaseMessage], **kwargs):
        """Stream response from Anthropic."""
        try:
            # Similar preprocessing as invoke
            system_message = ""
            user_messages = []

            for msg in messages:
                if isinstance(msg, SystemMessage):
                    system_message = msg.content
                else:
                    user_messages.append(msg)

            formatted_messages = self._format_messages(user_messages)
            params = self._get_request_params(stream=True, **kwargs)

            anthropic_params = {k: v for k, v in params.items()
                              if k in ["model", "max_tokens", "temperature", "stream"]}

            if system_message:
                anthropic_params["system"] = system_message

            stream = self.client.messages.create(
                messages=formatted_messages,
                **anthropic_params
            )

            for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text

        except Exception as e:
            logger.error(f"Anthropic streaming failed: {e}")
            raise


class GoogleProvider(BaseLLMProvider):
    """Google LLM provider implementation."""

    def _create_client(self):
        """Create Google client."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.client = self._create_client()

    def invoke(self, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """Invoke Google model."""
        try:
            # Convert to Google format
            prompt = self._convert_messages_to_prompt(messages)

            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }

            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )

            return LLMResponse(
                content=response.text,
                provider=self.provider_name,
                model=self.model_name,
                usage={"input_tokens": response.usage_metadata.prompt_token_count,
                       "output_tokens": response.usage_metadata.candidates_token_count} if hasattr(response, 'usage_metadata') else None,
                metadata={"finish_reason": response.candidates[0].finish_reason if response.candidates else None}
            )

        except Exception as e:
            logger.error(f"Google API call failed: {e}")
            raise

    def stream(self, messages: List[BaseMessage], **kwargs):
        """Stream response from Google."""
        try:
            prompt = self._convert_messages_to_prompt(messages)

            generation_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            }

            stream = self.client.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True
            )

            for chunk in stream:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Google streaming failed: {e}")
            raise

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Convert messages to a single prompt for Google."""
        prompt_parts = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                prompt_parts.append(f"System: {msg.content}")
            elif isinstance(msg, HumanMessage):
                prompt_parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_parts.append(f"Assistant: {msg.content}")

        return "\n\n".join(prompt_parts)


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider (uses OpenAI-compatible API)."""

    def __init__(self, config: ModelConfig):
        # DeepSeek uses OpenAI-compatible API
        super().__init__(config)


class UnifiedLLMProvider:
    """
    Unified interface for all LLM providers.

    This class provides a single entry point to work with any LLM provider
    in a consistent manner.
    """

    PROVIDER_CLASSES = {
        LLMProvider.GROQ: GroqProvider,
        LLMProvider.OPENAI: OpenAIProvider,
        LLMProvider.ANTHROPIC: AnthropicProvider,
        LLMProvider.GOOGLE: GoogleProvider,
        LLMProvider.DEEPSEEK: DeepSeekProvider,
    }

    @classmethod
    def create_provider(cls, config: ModelConfig) -> BaseLLMProvider:
        """
        Create a provider instance based on the configuration.

        Args:
            config: Model configuration

        Returns:
            Provider instance
        """
        provider_class = cls.PROVIDER_CLASSES.get(config.provider)

        if not provider_class:
            raise ValueError(f"Unsupported provider: {config.provider}")

        try:
            return provider_class(config)
        except Exception as e:
            logger.error(f"Failed to create provider {config.provider}: {e}")
            raise

    @classmethod
    def invoke(cls, config: ModelConfig, messages: List[BaseMessage], **kwargs) -> LLMResponse:
        """
        Invoke an LLM with the given configuration and messages.

        Args:
            config: Model configuration
            messages: Messages to send to the LLM
            **kwargs: Additional parameters

        Returns:
            Standardized LLM response
        """
        provider = cls.create_provider(config)
        return provider.invoke(messages, **kwargs)

    @classmethod
    def stream(cls, config: ModelConfig, messages: List[BaseMessage], **kwargs):
        """
        Stream response from an LLM.

        Args:
            config: Model configuration
            messages: Messages to send to the LLM
            **kwargs: Additional parameters

        Yields:
            Stream of text chunks
        """
        provider = cls.create_provider(config)
        yield from provider.stream(messages, **kwargs)

    @classmethod
    def is_provider_available(cls, provider: LLMProvider, api_key_env: str) -> bool:
        """
        Check if a provider is available (has valid API key and dependencies).

        Args:
            provider: Provider to check
            api_key_env: Environment variable for API key

        Returns:
            True if provider is available
        """
        # Check API key
        api_key = os.getenv(api_key_env)
        if not api_key or not api_key.strip():
            return False

        # Check if provider class exists and dependencies are available
        try:
            provider_class = cls.PROVIDER_CLASSES.get(provider)
            if not provider_class:
                return False

            # Try to import required dependencies
            if provider == LLMProvider.GROQ:
                import groq
            elif provider == LLMProvider.OPENAI or provider == LLMProvider.DEEPSEEK:
                import openai
            elif provider == LLMProvider.ANTHROPIC:
                import anthropic
            elif provider == LLMProvider.GOOGLE:
                import google.generativeai

            return True

        except ImportError:
            logger.warning(f"Dependencies not available for provider {provider}")
            return False
        except Exception as e:
            logger.warning(f"Provider {provider} availability check failed: {e}")
            return False

