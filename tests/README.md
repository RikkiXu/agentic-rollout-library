# Tests Directory

This directory contains essential tests for the Agentic Rollout Library's core functionality.

## Core Tests

### 1. **test_general_agent.py**
- **Purpose**: Tests the GeneralAgent class with mock LLM
- **Features**: ReAct framework, JSON action format, error handling, trajectory management
- **Usage**: `python test_general_agent.py`

### 2. **test_r2e_general_agent.py**
- **Purpose**: Tests GeneralAgent with R2E tools for K8S execution
- **Features**: 
  - Custom tool descriptions
  - XML action parser
  - Dynamic prompt generation
  - K8S pod execution
- **Usage**: `python test_r2e_general_agent.py`
- **Required Environment Variables**:
  - `LLM_API_KEY`
  - `LLM_BASE_URL`
  - `LLM_MODEL_NAME`

### 3. **test_llm_client.py**
- **Purpose**: Tests the LLM client functionality
- **Features**: API connection, retry mechanism, error handling
- **Usage**: `python test_llm_client.py`

### 4. **test_tool_schemas.py**
- **Purpose**: Tests tool schema definitions and validation
- **Features**: OpenAI function schemas, parameter validation
- **Usage**: `python test_tool_schemas.py`

### 5. **test_factory_pattern.py**
- **Purpose**: Tests factory pattern for creating tools and agents
- **Features**: Dynamic tool/agent creation, registration system
- **Usage**: `python test_factory_pattern.py`

### 6. **test_prompt_builder_example.py**
- **Purpose**: Demonstrates PromptBuilder utility usage
- **Features**: Dynamic prompt generation, variable substitution, tool formatting
- **Usage**: `python test_prompt_builder_example.py`

## Running Tests

1. Set environment variables:
   ```bash
   export LLM_API_KEY='your-api-key'
   export LLM_BASE_URL='your-base-url'
   export LLM_MODEL_NAME='gpt-4'
   ```

2. Run individual tests:
   ```bash
   python test_name.py
   ```

3. Run all tests:
   ```bash
   python -m pytest tests/
   ```

## Test Configuration

See `.env.example` in the root directory for configuration options.