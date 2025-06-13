# TODO Plan: Perplexity Deep Research Enhancement

## Project Overview
Current foundational version of the perplexity-like tool for LLM deep research and chatting needs enhancement with:
1. **Dual LLM Architecture**: Separate models for search query generation and thinking/reporting
2. **Thread Chat Memory**: Persistent conversation history and context management
3. **Enhanced User Experience**: Better chat interface and context-aware responses

---

## Phase 1: Dual LLM Architecture Implementation

### 1.1 Model Configuration Setup ✅ COMPLETED
- [x] **Create LLM configuration system**
  - ✅ Add configuration for multiple LLM providers (OpenAI, Groq, DeepSeek, etc.)
  - ✅ Create `config.py` for model settings and provider management
  - ✅ Add environment variables for different API keys
  - ✅ Support model fallbacks and load balancing

- [x] **Update dependencies**
  - ✅ Add OpenAI SDK for GPT models
  - ✅ Add DeepSeek API integration
  - ✅ Add anthropic for Claude models (optional)
  - ✅ Update `pyproject.toml` with new dependencies

**Implementation Summary:**
- Created comprehensive `config.py` with support for multiple LLM providers
- Implemented `ModelManager` class with health checking and fallback systems
- Built specialized `SearchQueryLLM` and `ThinkingLLM` classes
- Updated `utils.py` with backward compatibility while adding new capabilities
- Added comprehensive configuration validation and testing
- Created structured model directory with proper imports
- Updated dependencies in `pyproject.toml` with latest LangChain/LangGraph versions

### 1.2 Search Query Generation Model
- [ ] **Create dedicated search query generator**
  - Implement `SearchQueryLLM` class in new `models/` directory
  - Optimize for fast, efficient query generation (lighter models like GPT-4o-mini, Llama-3.1-8B)
  - Fine-tune prompts specifically for search query optimization
  - Add query validation and filtering logic

- [ ] **Update workflow for search query model**
  - Modify `query_planner()` function in `workflow.py`
  - Implement model-specific prompting strategies
  - Add query quality scoring and refinement

### 1.3 Thinking/Reasoning Model Integration
- [ ] **Implement thinking model system**
  - Create `ThinkingLLM` class for report generation and chat
  - Integrate DeepSeek R1 or similar reasoning models
  - Add support for chain-of-thought reasoning
  - Implement step-by-step thinking display in UI

- [ ] **Update report generation**
  - Modify `final_report_generator()` function
  - Add reasoning traces and thought processes
  - Implement multi-step analysis workflow
  - Add confidence scoring for claims and findings

### 1.4 Model Management System
- [ ] **Create model manager**
  - Implement `ModelManager` class for handling multiple models
  - Add model selection logic based on task type
  - Implement rate limiting and cost optimization
  - Add model health monitoring and fallbacks

---

## Phase 2: Thread Chat Memory System

### 2.1 Database and Storage Setup
- [ ] **Choose and setup database**
  - Implement SQLite for local development
  - Add support for PostgreSQL for production
  - Create database schemas for conversations, messages, and metadata
  - Add migration system for schema updates

- [ ] **Create data models**
  - Design `Conversation`, `Message`, `SearchSession` models
  - Implement user session management
  - Add conversation metadata (title, timestamps, tags)
  - Create indexes for efficient querying

### 2.2 Memory Management Implementation
- [ ] **Implement conversation storage**
  - Create `ConversationManager` class
  - Add functions for saving/loading chat history
  - Implement conversation summarization for long chats
  - Add conversation search and filtering capabilities

- [ ] **Context management system**
  - Implement sliding window context management
  - Add conversation summarization for token limit management
  - Create context relevance scoring
  - Implement smart context pruning strategies

### 2.3 Chat Interface Enhancement
- [ ] **Upgrade Streamlit UI for chat**
  - Replace single query input with chat interface
  - Add chat history display with collapsible messages
  - Implement conversation list sidebar
  - Add conversation management (new, delete, rename)

- [ ] **Chat message handling**
  - Implement message types (user, assistant, system, research)
  - Add support for rich message content (markdown, links, images)
  - Create message editing and regeneration features
  - Add message reactions and bookmarking

### 2.4 Context-Aware Research
- [ ] **Enhance workflow for chat context**
  - Modify workflow to consider conversation history
  - Implement follow-up question handling
  - Add reference to previous research sessions
  - Create context-aware search query generation

---

## Phase 3: Advanced Features and Optimizations

### 3.1 Enhanced Search and Analysis
- [ ] **Improve search capabilities**
  - Add multiple search engine support (Google, Bing, Academic)
  - Implement search result caching and deduplication
  - Add domain-specific search (news, academic, social media)
  - Create search result quality scoring

- [ ] **Advanced analysis features**
  - Add fact-checking and source verification
  - Implement multi-perspective analysis
  - Add temporal analysis for trending topics
  - Create comparative analysis features

### 3.2 User Experience Improvements
- [ ] **UI/UX enhancements**
  - Add dark/light theme toggle
  - Implement responsive design for mobile
  - Add keyboard shortcuts for common actions
  - Create customizable dashboard layout

- [ ] **Export and sharing features**
  - Add conversation export (PDF, markdown, JSON)
  - Implement conversation sharing via links
  - Add research report templates
  - Create citation management system

### 3.3 Performance and Scalability
- [ ] **Performance optimizations**
  - Implement async processing for search and LLM calls
  - Add caching layers (Redis) for frequent queries
  - Optimize database queries and add connection pooling
  - Implement request batching for LLM calls

- [ ] **Monitoring and analytics**
  - Add usage tracking and analytics
  - Implement error logging and monitoring
  - Create performance metrics dashboard
  - Add cost tracking for API usage

---

## Phase 4: Testing and Documentation

### 4.1 Testing Framework
- [ ] **Unit testing**
  - Create tests for all utility functions
  - Test LLM integration and model management
  - Add database operation tests
  - Test conversation management functions

- [ ] **Integration testing**
  - Test complete research workflows
  - Test chat conversation flows
  - Add UI automation tests (Selenium/Playwright)
  - Test API integrations and error handling

### 4.2 Documentation
- [ ] **Technical documentation**
  - Update README with new features
  - Create API documentation
  - Add architecture diagrams
  - Document configuration options

- [ ] **User documentation**
  - Create user guide for chat features
  - Add research workflow tutorials
  - Document advanced features and tips
  - Create troubleshooting guide

---

## Implementation Priority Order

### High Priority (Phase 1 & 2)
1. Model configuration system and dual LLM setup
2. Basic chat memory implementation
3. Enhanced UI for chat interface
4. Context-aware research workflow

### Medium Priority (Phase 3)
1. Advanced search features
2. Performance optimizations
3. Export and sharing capabilities
4. Enhanced analysis features

### Low Priority (Phase 4)
1. Comprehensive testing suite
2. Complete documentation
3. Monitoring and analytics
4. Advanced UI features

---

## Technical Architecture Changes

### New File Structure
```
perplexity-deep-search/
├── models/
│   ├── __init__.py
│   ├── search_query_llm.py
│   ├── thinking_llm.py
│   └── model_manager.py
├── database/
│   ├── __init__.py
│   ├── models.py
│   ├── migrations/
│   └── connection.py
├── memory/
│   ├── __init__.py
│   ├── conversation_manager.py
│   └── context_manager.py
├── ui/
│   ├── __init__.py
│   ├── chat_interface.py
│   └── components.py
├── config.py
├── main.py (enhanced)
├── workflow.py (updated)
├── utils.py (updated)
└── prompt.py (updated)
```

### Key Dependencies to Add
- `openai` for GPT models
- `deepseek-api` for DeepSeek R1
- `sqlalchemy` for database ORM
- `alembic` for database migrations
- `redis` for caching (optional)
- `streamlit-chat` for better chat UI

---

## Notes for Implementation
- Maintain backward compatibility with current single-query workflow
- Implement gradual rollout of features
- Add feature flags for testing new capabilities
- Consider rate limiting and cost management from the start
- Plan for multi-user support in the future
- Keep the system modular for easy maintenance and updates

---

## Success Metrics
- Dual LLM system reduces search query generation time by 50%
- Thinking model improves report quality and reasoning depth
- Chat memory enables coherent multi-turn conversations
- User engagement increases with persistent conversation history
- Overall research efficiency improves by 30-40%