# Agentic Rollout Library

> English Version | [中文版](README.md)

A highly customizable agent reasoning framework with tool integration, custom system prompts, and flexible action parsing. This library provides all the core components needed to build production-ready AI agents.

## 🌟 Core Features

### 🎯 Highly Customizable
- **Tool Customization**: Each tool can have custom descriptions supporting different prompt formats
- **System Prompt Customization**: Full control over system prompt generation with dynamic variable injection
- **Action Parser Customization**: Support for custom action parsers (JSON, XML, etc.)
- **Agent Behavior Customization**: Configurable termination conditions, max rounds, debug mode, etc.

### 🤖 General Agent Framework (GeneralAgent)
- **ReAct Framework**: Built-in Think-Act-Observe loop
- **Flexible Tool System**: Dynamic tool registration and management
- **Trajectory Management**: Complete execution trajectory tracking and saving
- **Termination Tool Support**: Configure which tools trigger agent termination
- **Debug Mode**: Detailed LLM input/output logging

### 🛠️ Powerful Tool System
- **Unified Tool Interface**: All tools inherit from `AgenticBaseTool`
- **OpenAI Schema Support**: Automatic OpenAI function calling format generation
- **Execution Modes**: Support for local execution and K8s Pod execution
- **R2E Tool Suite**: Tools designed specifically for code repository editing
  - `R2EBashExecutor`: Safe bash command execution
  - `R2EFileEditor`: Advanced file editing (view/create/str_replace/insert/undo)
  - `R2ESearch`: Code search tool
  - `R2ESubmit`: Task completion submission

### 🏗️ Prompt Building System (PromptBuilder)
```python
# Create dynamic prompts with PromptBuilder
builder = PromptBuilder()
prompt = (builder
    .add_variable("task", "fix bug #123")
    .add_tools(tools, formatter=custom_formatter)
    .add_context({"repo": "pandas", "version": "2.0"})
    .add_section("Instructions", "Please analyze the code carefully...")
    .build())
```

### 🏭 Factory Pattern System
- **Tool Factory**: Dynamically create tool instances by name
- **Agent Factory**: Unified agent creation interface
- **Auto Registration**: Automatic component registration using decorators

## 📁 Project Structure

```
agentic-rollout-library/
├── workers/
│   ├── agents/
│   │   └── general_agent.py        # General ReAct agent
│   ├── core/
│   │   ├── base_agent.py          # Agent base class
│   │   ├── base_tool.py           # Tool base class
│   │   ├── tool_factory.py        # Tool factory
│   │   ├── agent_factory.py       # Agent factory
│   │   └── trajectory.py          # Trajectory management
│   ├── tools/
│   │   ├── bash_executor_tool.py  # Bash execution tool
│   │   ├── file_editor_tool.py    # File editor tool
│   │   ├── search_tool.py         # Search tool
│   │   └── r2e_tools/            # R2E tool suite
│   └── utils/
│       ├── llm_client.py          # LLM client
│       └── prompt_builder.py      # Prompt builder
└── tests/
    └── test_r2e_general_agent.py  # Complete example
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd agentic-rollout-library

# Install dependencies
pip install -e .
```

### Environment Setup

Create `.env` file or set environment variables:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="your-base-url"
export LLM_MODEL_NAME="gpt-4"
```

### Basic Usage Example

```python
from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client

# 1. Create tools
tools = {
    "bash": create_tool("BashExecutor"),
    "editor": create_tool("FileEditor"),
    "search": create_tool("Search"),
    "finish": create_tool("Finish")
}

# 2. Create agent
agent = GeneralAgent(
    max_rounds=10,
    termination_tool_names=["finish"]
)
agent.set_tools(tools)

# 3. Create LLM client
llm_client = create_llm_client(
    api_key="your-key",
    base_url="your-url",
    model="gpt-4"
)

# 4. Run task
result = await agent.run_trajectory(
    prompt="Create a hello.py file in the current directory",
    llm_generate_func=llm_client.generate,
    request_id="task-001"
)
```

### Advanced Customization Examples

#### 1. Custom Tool Descriptions

```python
class CustomDescriptionWrapper:
    def __init__(self, tool, description):
        self.tool = tool
        self.custom_description = description
    
    def get_description(self):
        return self.custom_description
    
    def __getattr__(self, name):
        return getattr(self.tool, name)

# Wrap tool with custom description
wrapped_tool = CustomDescriptionWrapper(
    original_tool,
    "My custom tool description..."
)
```

#### 2. Dynamic System Prompts

```python
def generate_custom_prompt(tools, **kwargs):
    task = kwargs.get('task_description', 'default task')
    return f"""
    You are a professional {kwargs.get('role', 'assistant')}.
    
    Task: {task}
    
    Available tools:
    {tools['editor'].get_description()}
    {tools['bash'].get_description()}
    
    {kwargs.get('additional_instructions', '')}
    """

# Use dynamic prompt
agent.system_prompt = generate_custom_prompt(
    tools,
    role="Python developer",
    task_description="Fix bugs in the code",
    additional_instructions="Please follow PEP8 standards"
)
```

#### 3. Custom Action Parser

```python
def parse_xml_action(output: str):
    """Parse XML format actions"""
    import re
    match = re.search(r'<function=(\w+)>(.*?)</function>', output, re.DOTALL)
    if match:
        tool_name = match.group(1)
        # Parse parameters...
        return {"tool_name": tool_name, "tool_args": {...}}
    return None

# Use custom parser
agent = GeneralAgent(
    action_parser=parse_xml_action
)
```

## 🔧 K8s Execution Mode

Support for executing tools in Kubernetes Pods:

```python
k8s_config = {
    "execution_mode": "k8s",
    "pod_name": "my-dev-pod",
    "namespace": "default"
}

# Create K8s-executed tools
bash_tool = create_tool("BashExecutor", k8s_config)
file_tool = create_tool("FileEditor", k8s_config)
```

## 📚 Core Concepts

### Tools
- Inherit from `AgenticBaseTool`
- Implement `execute_tool` method
- Provide `get_openai_tool_schema` for tool description
- Support `get_description` for custom descriptions

### Agents
- Inherit from `BaseAgent`
- Manage tool collections
- Handle LLM interactions
- Maintain execution trajectory

### Trajectory
- Record all thoughts, actions, and observations
- Support serialization and deserialization
- Used for debugging and analysis

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.