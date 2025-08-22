# Agentic Rollout Library Examples

This directory contains clean, focused examples demonstrating the capabilities of the Agentic Rollout Library. The library works both independently and in integration with VERL (Versatile Environment for Reinforcement Learning).

## 🚀 Quick Start

### Prerequisites

**For Standalone Usage:**
```bash
pip install pydantic pyyaml asyncio
```

**For VERL Integration:**
```bash
# Ensure VERL is installed and in your Python path
export PYTHONPATH="/path/to/verl:$PYTHONPATH"
```

## 📁 Example Structure

### Core Examples

#### 1. **Basic Usage** (`basic_usage.py`)
Fundamental concepts and patterns:
- Creating agentic rollouts with different configurations
- Using ReAct agents for multi-step reasoning
- Custom agent implementations
- Batch processing and tool integration
- Trajectory management and analysis

**Run:** `python basic_usage.py`

#### 2. **Advanced Integration** (`advanced_integration.py`)
Production-ready patterns:
- Real LLM integration (OpenAI/Anthropic compatible)
- Complex multi-tool workflows
- Agent composition and chaining
- Error handling and performance monitoring
- Production deployment patterns

**Run:** `python advanced_integration.py`

#### 3. **Core Tools Demo** (`core_tools_demo.py`)
Demonstrates the new unified tool system:
- Individual tool usage (Calculator, Bash, File Editor, Search)
- Tool registry management
- Tool parameter validation
- Error handling and logging

**Run:** `python core_tools_demo.py`

#### 4. **Tool Integration Demo** (`tool_integration_demo.py`)
Complete integration patterns:
- Tools with ReAct agents
- Global tool registry usage
- Mock LLM integration
- Real-world workflow examples

**Run:** `python tool_integration_demo.py`

### Configuration Templates (`config/`)
Ready-to-use configuration files:
- `agentic_rollout_config.yaml`: General rollout settings
- `react_math_config.yaml`: Math-focused ReAct agent configuration
- `coding_agent_config.yaml`: Software engineering task configuration

### Testing Examples

#### **Basic Testing** (`test_agentic_rollout.py`)
- Unit test patterns for rollout functionality
- Mock LLM testing scenarios
- Agent behavior validation

#### **Coding Agent Testing** (`test_coding_agent.py`)
- Integration with new core tools
- End-to-end workflow testing
- Tool validation patterns

#### **Validation Suite** (`validation_tests.py`)
- Comprehensive system validation
- Performance benchmarking
- Cross-configuration testing

### Documentation (`docs/`)
- `AgenticRollout_Technical_Design_Document.md`: Detailed technical documentation

## 🛠️ Integration Patterns

### Standalone Usage
```python
from workers import (
    AgenticRollout, AgenticRolloutConfig,
    CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool,
    get_global_tool_registry, register_tool
)

# Register tools
register_tool(CalculatorTool, {"debug": True})
register_tool(BashExecutorTool, {"timeout": 30})

# Create rollout
config = AgenticRolloutConfig(agent_type="react", max_steps=10)
rollout = AgenticRollout(config=config, llm_generate_func=your_llm_func)

# Run trajectory
trajectory = await rollout.agent.run_trajectory(
    prompt={"content": "Calculate the factorial of 5"},
    llm_generate_func=your_llm_func,
    request_id="example"
)
```

### VERL Integration
```python
# When VERL is available, tools automatically inherit VERL compatibility
from verl.workers.rollout.agentic_rollout import AgenticRollout
from workers import CalculatorTool  # Still use our unified tools

# Use as a VERL rollout worker
rollout = AgenticRollout(config, llm_generate_func, tokenizer)
output_data = await rollout.generate_sequences(input_prompts)
```

## 🔧 Key Features Demonstrated

### Unified Tool Framework
- **Core Tools**: Calculator, Bash Executor, File Editor, Search Tool
- **VERL Compatibility**: Automatic detection and integration
- **Security Features**: Safe execution with configurable restrictions
- **Tool Registry**: Centralized management and discovery

### Agent System
- **ReAct Agents**: Built-in reasoning and acting capabilities
- **Custom Agents**: Easy extension framework
- **Tool Integration**: Seamless tool usage within agents
- **Trajectory Management**: Complete execution tracking

### LLM Integration
- **Multiple Providers**: OpenAI, Anthropic, custom endpoints
- **Mock Clients**: Testing without external dependencies
- **Async Support**: Full async/await compatibility
- **Error Handling**: Robust error recovery

## 🚀 Running Examples

All examples are self-contained and can be run directly:

```bash
cd examples

# Basic functionality
python basic_usage.py

# Advanced patterns  
python advanced_integration.py

# Tool system demos
python core_tools_demo.py
python tool_integration_demo.py

# Testing
python test_agentic_rollout.py
python test_coding_agent.py
python validation_tests.py
```

## 📊 What's New

This cleaned-up examples directory focuses on:

✅ **Unified Tool System**: New core tools with VERL compatibility  
✅ **Simplified Structure**: Removed redundant and outdated examples  
✅ **Updated Imports**: Compatible with both standalone and VERL usage  
✅ **Better Documentation**: Clear, focused examples with comprehensive comments  
✅ **Production Patterns**: Real-world usage scenarios and best practices  

## 🤝 Contributing Examples

When adding new examples:
1. Follow the established import patterns (`from workers import ...`)
2. Include comprehensive docstrings and comments
3. Add proper error handling and logging
4. Provide both standalone and VERL compatibility
5. Include configuration examples where relevant
6. Test with both mock and real LLM integrations

## 📝 Migration Notes

If you're migrating from older examples:
- Update imports to use `from workers import ...`
- Replace old tool implementations with new core tools
- Use the global tool registry for tool management
- Update agent configurations to use new schema format

For detailed API documentation, see the main project README and the `/workers/tools/README.md` file.