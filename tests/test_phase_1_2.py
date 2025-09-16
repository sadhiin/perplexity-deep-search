#!/usr/bin/env python3
"""
Test script for Phase 1.2: Search Query Generation Model
Tests the enhanced workflow with SearchQueryLLM integration
"""

import asyncio
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_search_query_llm():
    """Test the SearchQueryLLM directly"""
    print("=" * 60)
    print("TESTING SEARCH QUERY LLM (Phase 1.2)")
    print("=" * 60)

    try:
        from models.search_query_llm import SearchQueryLLM
        from models.model_manager import ModelManager

        # Initialize components
        model_manager = ModelManager()
        search_llm = SearchQueryLLM(model_manager)

        # Test query generation
        test_query = "impact of artificial intelligence on job market in 2024"
        current_date = datetime.today().strftime("%d %B %Y")
        context = f"Current date: {current_date}. Generate diverse queries to research this topic thoroughly."

        print(f"\n1. Testing Initial Query Generation")
        print(f"   Research Question: {test_query}")
        print(f"   Context: {context}")

        initial_queries = search_llm.generate_initial_queries(
            user_query=test_query,
            max_queries=3,
            context=context
        )

        print(f"\n   Generated {len(initial_queries)} initial queries:")
        for i, query in enumerate(initial_queries, 1):
            print(f"   {i}. {query}")

            # Test query validation
            validation = search_llm.validate_query(query)
            print(f"      Validation Score: {validation['score']:.2f}")
            if validation['issues']:
                print(f"      Issues: {', '.join(validation['issues'])}")

        # Test query refinement
        print(f"\n2. Testing Query Refinement")
        mock_search_results = """
        Title: AI Job Market 2024 Report
        Content: Recent studies show that AI is creating new jobs while automating others. The net effect varies by industry...

        Title: Skills Gap in AI Era
        Content: Workers need to develop new skills to remain relevant in the AI-driven economy...
        """

        refined_queries = search_llm.refine_queries(
            original_query=test_query,
            previous_queries=initial_queries,
            search_results_summary=mock_search_results,
            max_queries=3
        )

        print(f"   Generated {len(refined_queries)} refined queries:")
        for i, query in enumerate(refined_queries, 1):
            print(f"   {i}. {query}")

            # Test query validation for refined queries
            validation = search_llm.validate_query(query)
            print(f"      Validation Score: {validation['score']:.2f}")

        # Test model info
        print(f"\n3. Model Information:")
        model_info = search_llm.get_model_info()
        print(f"   Model: {model_info.get('model_name', 'N/A')}")
        print(f"   Provider: {model_info.get('provider', 'N/A')}")

        return True

    except Exception as e:
        print(f"   ❌ SearchQueryLLM test failed: {e}")
        return False


def test_enhanced_workflow():
    """Test the enhanced workflow with SearchQueryLLM integration"""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED WORKFLOW (Phase 1.2)")
    print("=" * 60)

    try:
        from workflow import query_planner, should_refine_query, final_report_generator, DeepResearchState

        # Initialize test state
        test_state = {
            "user_query": "latest developments in quantum computing 2024",
            "search_queries": [],
            "search_results": [],
            "individual_page_summaries": [],
            "report_markdown": "",
            "query_generation_count": 0
        }

        print(f"\n1. Testing Enhanced Query Planner")
        print(f"   User Query: {test_state['user_query']}")

        # Test query planner
        query_result = query_planner(test_state)

        print(f"   Generated {len(query_result['search_queries'])} search queries:")
        for i, query in enumerate(query_result['search_queries'], 1):
            print(f"   {i}. {query}")

        print(f"   Found {len(query_result['search_results'])} search results")

        # Update state for refinement test
        test_state.update(query_result)

        print(f"\n2. Testing Enhanced Query Refinement")
        # Test refinement (this might go to final report if no refinement needed)
        refine_result = should_refine_query(test_state)

        print(f"   Refinement Command: {refine_result}")

        if hasattr(refine_result, 'update') and refine_result.update:
            print(f"   Generated {len(refine_result.update.get('search_queries', []))} refined queries:")
            for i, query in enumerate(refine_result.update.get('search_queries', []), 1):
                print(f"   {i}. {query}")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced workflow test failed: {e}")
        import traceback
        print(f"   Error details: {traceback.format_exc()}")
        return False


def test_utils_integration():
    """Test the utils integration with new LLM functions"""
    print("\n" + "=" * 60)
    print("TESTING UTILS INTEGRATION (Phase 1.2)")
    print("=" * 60)

    try:
        from utils import call_search_query_llm, call_thinking_llm, get_search_query_llm
        from config import TaskType

        print(f"\n1. Testing call_search_query_llm function")
        test_query = "climate change impact on agriculture"

        search_queries = call_search_query_llm(
            user_query=test_query,
            max_queries=3,
            context="Focus on recent research and data"
        )

        print(f"   Generated {len(search_queries)} queries:")
        for i, query in enumerate(search_queries, 1):
            print(f"   {i}. {query}")

        print(f"\n2. Testing SearchQueryLLM instance creation")
        search_llm = get_search_query_llm()
        model_info = search_llm.get_model_info()
        print(f"   Model: {model_info.get('model_name', 'N/A')}")
        print(f"   Provider: {model_info.get('provider', 'N/A')}")

        return True

    except Exception as e:
        print(f"   ❌ Utils integration test failed: {e}")
        return False


def test_prompt_enhancements():
    """Test the enhanced prompts"""
    print("\n" + "=" * 60)
    print("TESTING ENHANCED PROMPTS (Phase 1.2)")
    print("=" * 60)

    try:
        from prompt import (
            enhanced_search_queries_prompt,
            enhanced_refine_queries_prompt,
            enhanced_final_report_prompt
        )

        print(f"\n1. Enhanced Search Queries Prompt Available: ✓")
        print(f"   Length: {len(enhanced_search_queries_prompt)} characters")

        print(f"\n2. Enhanced Refine Queries Prompt Available: ✓")
        print(f"   Length: {len(enhanced_refine_queries_prompt)} characters")

        print(f"\n3. Enhanced Final Report Prompt Available: ✓")
        print(f"   Length: {len(enhanced_final_report_prompt)} characters")

        # Test prompt formatting
        print(f"\n4. Testing Prompt Formatting")
        test_formatted = enhanced_search_queries_prompt.format(
            user_query="test query",
            context="test context",
            max_queries=3
        )
        print(f"   Formatted prompt length: {len(test_formatted)} characters")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced prompts test failed: {e}")
        return False


def main():
    """Run all Phase 1.2 tests"""
    print("🚀 Starting Phase 1.2 Tests: Search Query Generation Model")
    print("=" * 80)

    tests = [
        ("SearchQueryLLM Direct Testing", test_search_query_llm),
        ("Enhanced Workflow Testing", test_enhanced_workflow),
        ("Utils Integration Testing", test_utils_integration),
        ("Enhanced Prompts Testing", test_prompt_enhancements),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 1.2 TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} - {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 Phase 1.2 implementation is complete and working!")
        print("\n✅ PHASE 1.2 COMPLETED:")
        print("   • SearchQueryLLM specialized model implemented")
        print("   • Enhanced workflow with query validation")
        print("   • Intelligent query refinement system")
        print("   • Optimized prompts for better query generation")
        print("   • Full integration with existing system")
        print("\n📋 Ready for Phase 2: Thread Chat Memory System")
    else:
        print("❌ Some tests failed. Please review the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)