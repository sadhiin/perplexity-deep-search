"""
Configuration system for multi-LLM provider management.

This module provides a centralized configuration system for managing multiple LLM providers
including OpenAI, Groq, DeepSeek, and Anthropic with support for fallbacks and load balancing.
"""

import os
from enum import Enum
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class TaskType(Enum):
    """Types of tasks that require different model configurations."""
    SEARCH_QUERY_GENERATION = "search_query_generation"
    THINKING_REASONING = "thinking_reasoning"
    REPORT_GENERATION = "report_generation"
    CHAT_RESPONSE = "chat_response"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: LLMProvider
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3
    api_key_env: str = ""
    base_url: Optional[str] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMConfiguration:
    """Main configuration class for LLM management."""

    # Task-specific model assignments
    task_models: Dict[TaskType, str] = field(default_factory=dict)

    # Model configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)

    # Fallback chains for each task type
    fallback_chains: Dict[TaskType, List[str]] = field(default_factory=dict)

    # Global settings
    enable_fallbacks: bool = True
    enable_load_balancing: bool = False
    cost_optimization: bool = True

    def __post_init__(self):
        """Initialize default configurations."""
        if not self.models:
            self._setup_default_models()
        if not self.task_models:
            self._setup_default_task_assignments()
        if not self.fallback_chains:
            self._setup_default_fallbacks()

        def _setup_default_models(self):
        """Setup default model configurations."""
        self.models = {
            # Primary models from Groq (as requested)
            "llama-3.3-70b-versatile": ModelConfig(
                provider=LLMProvider.GROQ,
                model_name="llama-3.3-70b-versatile",
                temperature=0.3,  # Lower for search queries
                max_tokens=2000,
                api_key_env="GROQ_API_KEY"
            ),
            "qwen-2.5-72b-instruct": ModelConfig(
                provider=LLMProvider.GROQ,
                model_name="qwen-2.5-72b-instruct",
                temperature=0.7,  # Higher for reasoning/thinking
                max_tokens=4000,
                api_key_env="GROQ_API_KEY"
            ),

            # Additional Groq models for variety
            "llama-3.1-8b-instant": ModelConfig(
                provider=LLMProvider.GROQ,
                model_name="llama-3.1-8b-instant",
                temperature=0.3,
                max_tokens=1000,
                api_key_env="GROQ_API_KEY"
            ),

            # OpenAI models (optional fallbacks)
            "gpt-4o-mini": ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o-mini",
                temperature=0.3,
                max_tokens=500,
                api_key_env="OPENAI_API_KEY"
            ),
            "gpt-4o": ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4o",
                temperature=0.7,
                max_tokens=4000,
                api_key_env="OPENAI_API_KEY"
            ),

            # Anthropic models (optional)
            "claude-3-5-sonnet-20241022": ModelConfig(
                provider=LLMProvider.ANTHROPIC,
                model_name="claude-3-5-sonnet-20241022",
                temperature=0.7,
                max_tokens=4000,
                api_key_env="ANTHROPIC_API_KEY"
            ),

            # Google models (for future expansion)
            "gemini-1.5-pro": ModelConfig(
                provider=LLMProvider.GOOGLE,
                model_name="gemini-1.5-pro",
                temperature=0.7,
                max_tokens=4000,
                api_key_env="GOOGLE_API_KEY"
            ),

            # DeepSeek models (optional)
            "deepseek-r1-distill-llama-70b": ModelConfig(
                provider=LLMProvider.DEEPSEEK,
                model_name="deepseek-r1-distill-llama-70b",
                temperature=0.7,
                max_tokens=4000,
                api_key_env="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com"
            )
        }

    def _setup_default_task_assignments(self):
        """Setup default task to model assignments."""
        self.task_models = {
            TaskType.SEARCH_QUERY_GENERATION: "llama-3.3-70b-versatile",
            TaskType.THINKING_REASONING: "qwen-2.5-72b-instruct",
            TaskType.REPORT_GENERATION: "qwen-2.5-72b-instruct",
            TaskType.CHAT_RESPONSE: "llama-3.3-70b-versatile"
        }

    def _setup_default_fallbacks(self):
        """Setup default fallback chains."""
        self.fallback_chains = {
            TaskType.SEARCH_QUERY_GENERATION: [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "gpt-4o-mini",
                "qwen-2.5-72b-instruct"
            ],
            TaskType.THINKING_REASONING: [
                "qwen-2.5-72b-instruct",
                "llama-3.3-70b-versatile",
                "deepseek-r1-distill-llama-70b",
                "gpt-4o",
                "claude-3-5-sonnet-20241022"
            ],
            TaskType.REPORT_GENERATION: [
                "qwen-2.5-72b-instruct",
                "llama-3.3-70b-versatile",
                "gpt-4o",
                "claude-3-5-sonnet-20241022"
            ],
            TaskType.CHAT_RESPONSE: [
                "llama-3.3-70b-versatile",
                "qwen-2.5-72b-instruct",
                "gpt-4o",
                "claude-3-5-sonnet-20241022"
            ]
        }

    def get_model_for_task(self, task_type: TaskType) -> str:
        """Get the primary model for a given task type."""
        return self.task_models.get(task_type, "llama-3.3-70b-versatile")

    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        return self.models[model_name]

    def get_fallback_chain(self, task_type: TaskType) -> List[str]:
        """Get fallback chain for a task type."""
        return self.fallback_chains.get(task_type, ["llama-3.3-70b-versatile"])

    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate that required API keys are available."""
        validation_results = {}

        for model_name, config in self.models.items():
            api_key = os.getenv(config.api_key_env)
            validation_results[model_name] = api_key is not None and api_key.strip() != ""

        return validation_results

    def get_available_models(self) -> List[str]:
        """Get list of models with valid API keys."""
        validation_results = self.validate_api_keys()
        return [model for model, is_valid in validation_results.items() if is_valid]


class ConfigurationManager:
    """Manager class for LLM configuration operations."""

    def __init__(self, config: Optional[LLMConfiguration] = None):
        """Initialize configuration manager."""
        self.config = config or LLMConfiguration()
        self._validate_environment()

    def _validate_environment(self):
        """Validate environment variables and warn about missing keys."""
        validation_results = self.config.validate_api_keys()
        missing_keys = [
            model for model, is_valid in validation_results.items()
            if not is_valid
        ]

        if missing_keys:
            print(f"Warning: Missing API keys for models: {missing_keys}")
            print("Some models may not be available for use.")

    def get_task_model_config(self, task_type: TaskType) -> ModelConfig:
        """Get model configuration for a specific task type."""
        model_name = self.config.get_model_for_task(task_type)
        return self.config.get_model_config(model_name)

    def update_task_assignment(self, task_type: TaskType, model_name: str):
        """Update task to model assignment."""
        if model_name not in self.config.models:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        self.config.task_models[task_type] = model_name

    def add_custom_model(self, name: str, config: ModelConfig):
        """Add a custom model configuration."""
        self.config.models[name] = config

    def get_runtime_config(self, task_type: TaskType) -> Dict[str, Any]:
        """Get runtime configuration for instantiating LLM clients."""
        model_config = self.get_task_model_config(task_type)

        runtime_config = {
            "provider": model_config.provider.value,
            "model_name": model_config.model_name,
            "temperature": model_config.temperature,
            "timeout": model_config.timeout,
            "max_retries": model_config.max_retries,
            "api_key": os.getenv(model_config.api_key_env),
        }

        if model_config.max_tokens:
            runtime_config["max_tokens"] = model_config.max_tokens

        if model_config.base_url:
            runtime_config["base_url"] = model_config.base_url

        runtime_config.update(model_config.custom_params)

        return runtime_config


# Global configuration instance
_global_config = None


def get_config() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    global _global_config
    if _global_config is None:
        _global_config = ConfigurationManager()
    return _global_config


def set_config(config: LLMConfiguration):
    """Set the global configuration."""
    global _global_config
    _global_config = ConfigurationManager(config)


# Environment validation functions
def check_required_env_vars() -> Dict[str, str]:
    """Check for required environment variables and return status."""
    required_vars = {
        "GROQ_API_KEY": "Required for Groq models (backward compatibility)",
        "OPENAI_API_KEY": "Required for OpenAI models (GPT-4, GPT-4o-mini)",
        "DEEPSEEK_API_KEY": "Required for DeepSeek R1 reasoning models",
        "ANTHROPIC_API_KEY": "Optional for Claude models"
    }

    status = {}
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value and value.strip():
            status[var] = "✅ Available"
        else:
            status[var] = f"❌ Missing - {description}"

    return status


def print_configuration_status():
    """Print current configuration status."""
    print("LLM Configuration Status")
    print("=" * 50)

    config_manager = get_config()

    print("\nAPI Key Status:")
    env_status = check_required_env_vars()
    for var, status in env_status.items():
        print(f"  {var}: {status}")

    print("\nAvailable Models:")
    available_models = config_manager.config.get_available_models()
    for model in available_models:
        config = config_manager.config.get_model_config(model)
        print(f"  ✅ {model} ({config.provider.value})")

    print("\nTask Assignments:")
    for task_type in TaskType:
        model_name = config_manager.config.get_model_for_task(task_type)
        print(f"  {task_type.value}: {model_name}")

    print("\nFallback Chains:")
    for task_type in TaskType:
        chain = config_manager.config.get_fallback_chain(task_type)
        print(f"  {task_type.value}: {' → '.join(chain)}")


if __name__ == "__main__":
    print_configuration_status()