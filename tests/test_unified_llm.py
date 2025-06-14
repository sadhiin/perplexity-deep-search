"""
Test script for the Unified LLM Provider

This script tests the unified LLM provider with different configurations
to ensure it can work with various LLM providers.
"""

import os
import asyncio
from langchain_core.messages import HumanMessage, SystemMessage
from config import ConfigurationManager, TaskType
from models.unified_llm_provider import UnifiedLLMProvider, LLMResponse


def test_provider_availability():
    """Test which providers are available in the current environment."""
    print("🔍 Checking provider availability...")

    config_manager = ConfigurationManager()
    available_providers = []

    for task_type in TaskType:
        configs = config_manager.get_models_for_task(task_type)
        for config in configs:
            if UnifiedLLMProvider.is_provider_available(config.provider, config.api_key_env):
                available_providers.append(f"{config.provider.value}: {config.model_name}")
                print(f"✅ {config.provider.value} ({config.model_name}) - Available")
            else:
                print(f"❌ {config.provider.value} ({config.model_name}) - Not available")

    return available_providers


def test_search_query_generation():
    """Test search query generation using the configured search LLM."""
    print("\n🔍 Testing Search Query Generation...")

    try:
        config_manager = ConfigurationManager()
        search_config = config_manager.get_model_for_task(TaskType.SEARCH_QUERY)

        if not UnifiedLLMProvider.is_provider_available(search_config.provider, search_config.api_key_env):
            print(f"❌ Search provider {search_config.provider.value} not available")
            return

        messages = [
            SystemMessage(content="You are a search query generator. Generate concise, effective search queries."),
            HumanMessage(content="I want to learn about machine learning algorithms for natural language processing")
        ]

        response = UnifiedLLMProvider.invoke(search_config, messages, max_tokens=100)

        print(f"✅ Search Query Response from {response.provider} ({response.model}):")
        print(f"Content: {response.content}")
        print(f"Usage: {response.usage}")

    except Exception as e:
        print(f"❌ Search query test failed: {e}")


def test_thinking_generation():
    """Test thinking/reasoning using the configured thinking LLM."""
    print("\n🧠 Testing Thinking Generation...")

    try:
        config_manager = ConfigurationManager()
        thinking_config = config_manager.get_model_for_task(TaskType.THINKING)

        if not UnifiedLLMProvider.is_provider_available(thinking_config.provider, thinking_config.api_key_env):
            print(f"❌ Thinking provider {thinking_config.provider.value} not available")
            return

        messages = [
            SystemMessage(content="You are a deep thinking AI assistant. Provide thoughtful, analytical responses."),
            HumanMessage(content="Explain the key differences between supervised and unsupervised learning")
        ]

        response = UnifiedLLMProvider.invoke(thinking_config, messages, max_tokens=200)

        print(f"✅ Thinking Response from {response.provider} ({response.model}):")
        print(f"Content: {response.content[:200]}...")
        print(f"Usage: {response.usage}")

    except Exception as e:
        print(f"❌ Thinking test failed: {e}")


def test_streaming():
    """Test streaming responses."""
    print("\n🌊 Testing Streaming...")

    try:
        config_manager = ConfigurationManager()
        config = config_manager.get_model_for_task(TaskType.SEARCH_QUERY)

        if not UnifiedLLMProvider.is_provider_available(config.provider, config.api_key_env):
            print(f"❌ Provider {config.provider.value} not available for streaming")
            return

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Count from 1 to 5")
        ]

        print(f"✅ Streaming from {config.provider.value} ({config.model_name}):")

        for chunk in UnifiedLLMProvider.stream(config, messages, max_tokens=50):
            print(chunk, end="", flush=True)

        print("\n✅ Streaming completed successfully")

    except Exception as e:
        print(f"❌ Streaming test failed: {e}")


def main():
    """Run all tests."""
    print("🚀 Testing Unified LLM Provider\n")

    # Test provider availability
    available_providers = test_provider_availability()

    if not available_providers:
        print("\n❌ No providers available. Please check your API keys and environment setup.")
        return

    print(f"\n✅ Found {len(available_providers)} available provider configurations")

    # Test core functionality
    test_search_query_generation()
    test_thinking_generation()
    test_streaming()

    print("\n🎉 All tests completed!")


if __name__ == "__main__":
    main()