#!/usr/bin/env python3
"""
Test script for the new LLM configuration system.

This script validates the multi-model configuration setup and demonstrates
the capabilities of the new dual LLM architecture.
"""

import os
import sys
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from backend.config import (
    print_configuration_status,
    get_config,
    TaskType,
    LLMProvider,
    ModelConfig,
    LLMConfiguration,
    RateLimitConfig,
    ConfigurationManager
)
from models.model_manager import ModelManager
from models.search_query_llm import SearchQueryLLM
from models.thinking_llm import ThinkingLLM
from utils import call_llm, call_search_query_llm, call_thinking_llm


def test_configuration_system():
    """Test the configuration system setup."""
    print("🔧 Testing Configuration System")
    print("=" * 60)

    # Print current configuration status
    print_configuration_status()

    # Test configuration access
    config_manager = get_config()
    print(f"\nConfiguration loaded successfully: {config_manager is not None}")

    # Test task assignments
    print("\n📋 Task Model Assignments:")
    for task_type in TaskType:
        model_name = config_manager.config.get_model_for_task(task_type)
        print(f"  {task_type.value}: {model_name}")

    print("\n" + "=" * 60)


def test_model_manager():
    """Test the model manager functionality."""
    print("🤖 Testing Model Manager")
    print("=" * 60)

    try:
        # Initialize model manager
        model_manager = ModelManager()
        print("✅ Model manager initialized successfully")

        # Test available models
        available_models = model_manager.list_available_models()
        print(f"📊 Available models: {len(available_models)}")
        for model in available_models:
            print(f"  - {model}")

        # Test model info
        if available_models:
            model_info = model_manager.get_model_info(available_models[0])
            print(f"\n📋 Sample model info ({available_models[0]}):")
            for key, value in model_info.items():
                print(f"  {key}: {value}")

        # Test task-specific model retrieval
        print(f"\n🎯 Task-specific models:")
        for task_type in TaskType:
            try:
                model = model_manager.get_model_for_task(task_type)
                print(f"  ✅ {task_type.value}: {type(model).__name__}")
            except Exception as e:
                print(f"  ❌ {task_type.value}: {e}")

    except Exception as e:
        print(f"❌ Model manager test failed: {e}")

    print("\n" + "=" * 60)


def test_rate_limit_and_cost_controls():
    """Test rate limiting and cost-aware selection logic."""
    print("⚖️ Testing Rate Limiting & Cost Optimization")
    print("=" * 60)

    try:
        from types import SimpleNamespace
        import time

        os.environ.setdefault("OPENAI_API_KEY", "test-openai")
        os.environ.setdefault("GROQ_API_KEY", "test-groq")

        shared_models = {
            "expensive-model": ModelConfig(
                provider=LLMProvider.OPENAI,
                model_name="expensive-model",
                api_key_env="OPENAI_API_KEY",
                cost_per_1k_input=10.0,
                cost_per_1k_output=10.0,
            ),
            "cheap-model": ModelConfig(
                provider=LLMProvider.GROQ,
                model_name="cheap-model",
                api_key_env="GROQ_API_KEY",
                cost_per_1k_input=0.5,
                cost_per_1k_output=0.5,
            ),
        }

        fallback_chain = {task: ["cheap-model", "expensive-model"] for task in TaskType}
        task_assignments = {task: "expensive-model" for task in TaskType}

        custom_config = LLMConfiguration(
            task_models=task_assignments,
            models=shared_models,
            fallback_chains=fallback_chain,
            rate_limits={
                LLMProvider.OPENAI: RateLimitConfig(requests_per_minute=5),
                LLMProvider.GROQ: RateLimitConfig(requests_per_minute=1),
            },
            cost_optimization=True,
            enable_fallbacks=True,
        )

        config_manager = ConfigurationManager(custom_config)
        model_manager = ModelManager(config_manager)

        class DummyModel:
            def __init__(self, name: str):
                self.name = name

            def invoke(self, *_, **__):
                return SimpleNamespace(content=f"dummy response from {self.name}")

        def fake_create(self, model_name: str):
            return DummyModel(model_name)

        model_manager._create_model_instance = fake_create.__get__(model_manager, ModelManager)

        preferred = model_manager._select_cost_optimized_model(
            TaskType.SEARCH_QUERY_GENERATION,
            "expensive-model",
        )
        print(f"   Preferred model with cost optimization: {preferred}")

        limiter = model_manager._rate_limiters.get(LLMProvider.GROQ)
        if limiter:
            with limiter._lock:
                limiter._request_times.clear()
                now = time.time()
                for _ in range(limiter.requests_per_minute or 0):
                    limiter._request_times.append(now)

        rerouted = model_manager._select_cost_optimized_model(
            TaskType.SEARCH_QUERY_GENERATION,
            "expensive-model",
        )
        print(f"   Preferred model when cheap option throttled: {rerouted}")

        # Clean up limiter state for other tests
        if limiter:
            with limiter._lock:
                limiter._request_times.clear()

        return True

    except Exception as e:
        print(f"❌ Rate limit & cost optimization test failed: {e}")
        return False




def test_search_query_llm():
    """Test the search query LLM functionality."""
    print("🔍 Testing Search Query LLM")
    print("=" * 60)

    try:
        # Initialize search query LLM
        search_llm = SearchQueryLLM()
        print("✅ Search Query LLM initialized successfully")

        # Test query generation
        test_query = "What are the latest developments in renewable energy?"
        print(f"\n📝 Generating queries for: '{test_query}'")

        # Test using the wrapper function
        queries = call_search_query_llm(test_query, max_queries=3)
        print(f"✅ Generated {len(queries)} search queries:")
        for i, query in enumerate(queries, 1):
            print(f"  {i}. {query}")

        # Test query validation
        if queries:
            validation = search_llm.validate_query(queries[0])
            print(f"\n🔍 Query validation for '{queries[0]}':")
            print(f"  Valid: {validation['is_valid']}")
            print(f"  Score: {validation['score']:.2f}")
            if validation['issues']:
                print(f"  Issues: {validation['issues']}")

    except Exception as e:
        print(f"❌ Search Query LLM test failed: {e}")

    print("\n" + "=" * 60)


def test_thinking_llm():
    """Test the thinking LLM functionality."""
    print("🧠 Testing Thinking LLM")
    print("=" * 60)

    try:
        # Initialize thinking LLM
        thinking_llm = ThinkingLLM()
        print("✅ Thinking LLM initialized successfully")

        # Test step-by-step reasoning
        test_problem = "How can we reduce carbon emissions in urban transportation?"
        print(f"\n🤔 Reasoning about: '{test_problem}'")

        # Test using the wrapper function
        reasoning = call_thinking_llm(test_problem, task="reasoning")
        print("✅ Step-by-step reasoning completed:")
        print(f"  Preview: {reasoning[:200]}...")

        # Test chat response
        chat_response = call_thinking_llm(
            "What are the key factors to consider when implementing renewable energy?",
            task="chat"
        )
        print(f"\n💬 Chat response preview: {chat_response[:150]}...")

    except Exception as e:
        print(f"❌ Thinking LLM test failed: {e}")

    print("\n" + "=" * 60)


def test_backward_compatibility():
    """Test backward compatibility with existing code."""
    print("🔄 Testing Backward Compatibility")
    print("=" * 60)

    try:
        # Test original call_llm function
        prompt = "What is artificial intelligence?"
        response = call_llm(prompt)
        print("✅ Original call_llm function works")
        print(f"  Response preview: {response[:100]}...")

        # Test with task type
        response_with_task = call_llm(prompt, task_type=TaskType.CHAT_RESPONSE)
        print("✅ call_llm with task_type works")
        print(f"  Response preview: {response_with_task[:100]}...")

    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")

    print("\n" + "=" * 60)


def test_fallback_system():
    """Test the fallback system."""
    print("🛡️ Testing Fallback System")
    print("=" * 60)

    try:
        model_manager = ModelManager()

        # Test with a non-existent model
        print("Testing fallback for non-existent model...")
        try:
            # This should trigger fallback
            model = model_manager.get_model("non-existent-model")
            print("✅ Fallback system activated successfully")
        except Exception as e:
            print(f"ℹ️  Expected fallback behavior: {e}")

        # Test fallback chains
        print("\n📋 Fallback chains configured:")
        for task_type in TaskType:
            chain = model_manager.config.get_fallback_chain(task_type)
            print(f"  {task_type.value}: {' → '.join(chain)}")

    except Exception as e:
        print(f"❌ Fallback system test failed: {e}")

    print("\n" + "=" * 60)


def main():
    """Run all tests."""
    print("🚀 LLM Configuration System Test Suite")
    print("=" * 60)

    # Run tests
    test_configuration_system()
    test_model_manager()
    test_rate_limit_and_cost_controls()
    test_search_query_llm()
    test_thinking_llm()
    test_backward_compatibility()
    test_fallback_system()

    print("✨ Test suite completed!")
    print("\n📋 Next Steps:")
    print("  1. Set up additional API keys in your .env file")
    print("  2. Configure task assignments as needed")
    print("  3. Integrate with existing workflow in workflow.py")
    print("  4. Test with real research queries")


if __name__ == "__main__":
    main()
