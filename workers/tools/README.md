# Agentic Rollout Library - Core Tools

The core tools module provides a unified framework for tool development and integration that works with or without VERL. This system allows for easy tool development, registration, and usage across different agents.

## Architecture Overview

### Tool Framework Components

1. **Base Tool Classes** (`base_tool.py`)
   - `BaseAgenticTool`: Abstract base class for all tools
   - `SimpleAgenticTool`: Simplified base class for basic tools
   - `AgenticBaseTool`: VERL-compatible base class (auto-detects VERL)

2. **Tool Registry** (`tool_registry.py`)
   - `ToolRegistry`: Centralized tool management
   - Global registry for tool registration and discovery
   - Instance lifecycle management

3. **Tool Schemas** (`tool_schemas.py`)
   - OpenAI function schema compatibility
   - Standardized tool result formats
   - Parameter validation utilities

### VERL Compatibility

The tool system automatically detects VERL availability:

- **With VERL**: Inherits from VERL's `BaseTool` and provides full compatibility
- **Without VERL**: Uses standalone implementation with identical interface
- **Seamless Integration**: Same API regardless of VERL presence

## Core Tools

### 1. Calculator Tool (`calculator_tool.py`)

Advanced mathematical computation tool supporting:
- Expression evaluation with safe AST parsing
- Basic arithmetic operations
- Scientific functions (trigonometry, logarithms, etc.)
- Statistical operations on lists
- Calculation history tracking

**Usage Example:**
```python
from workers.tools import CalculatorTool

calc = CalculatorTool()
instance_id = await calc.create_instance()

# Expression evaluation
result = await calc.execute_tool(instance_id, {
    "expression": "sqrt(16) + factorial(4)"
})

# Specific operations
result = await calc.execute_tool(instance_id, {
    "operation": "power",
    "base": 2,
    "exponent": 8
})
```

### 2. Bash Executor Tool (`bash_executor_tool.py`)

Safe bash command execution with:
- Security filtering for dangerous commands
- Timeout protection
- Output capture and streaming
- Working directory support
- Execution history

**Usage Example:**
```python
from workers.tools import BashExecutorTool

bash = BashExecutorTool({"timeout": 30})
instance_id = await bash.create_instance()

result = await bash.execute_tool(instance_id, {
    "command": "ls -la",
    "working_directory": "/tmp",
    "timeout": 10
})
```

### 3. File Editor Tool (`file_editor_tool.py`)

Comprehensive file operations supporting:
- File and directory viewing
- File creation and editing
- String replacement with uniqueness checking
- Text insertion at specific lines
- Edit history and undo functionality
- Syntax linting for Python files

**Usage Example:**
```python
from workers.tools import FileEditorTool

editor = FileEditorTool()
instance_id = await editor.create_instance()

# Create file
result = await editor.execute_tool(instance_id, {
    "command": "create",
    "path": "/tmp/example.py",
    "file_text": "print('Hello World')"
})

# View file
result = await editor.execute_tool(instance_id, {
    "command": "view",
    "path": "/tmp/example.py"
})

# String replacement
result = await editor.execute_tool(instance_id, {
    "command": "str_replace",
    "path": "/tmp/example.py",
    "old_str": "Hello World",
    "new_str": "Hello Tools!"
})
```

### 4. Search Tool (`search_tool.py`)

Powerful search capabilities including:
- Text search in files with regex support
- File name pattern matching
- Directory structure search
- Context line display
- Configurable result limits
- File type filtering

**Usage Example:**
```python
from workers.tools import SearchTool

search = SearchTool()
instance_id = await search.create_instance()

# Search for text in files
result = await search.execute_tool(instance_id, {
    "command": "search_text",
    "pattern": "import",
    "path": "/path/to/project",
    "file_extensions": [".py"],
    "context_lines": 2
})

# Search for files by name
result = await search.execute_tool(instance_id, {
    "command": "search_files", 
    "pattern": "test.*\\.py",
    "path": "/path/to/project",
    "regex": True
})
```

## Tool Registry System

### Global Registry

```python
from workers import get_global_tool_registry, register_tool

# Register tools globally
register_tool(CalculatorTool, {"debug": True}, "calc")
register_tool(BashExecutorTool, {"timeout": 15}, "bash")

# Get global registry
registry = get_global_tool_registry()

# Create and use tool instances
calc_instance = await registry.create_tool_instance("calc")
result = await registry.execute_tool("calc", calc_instance, {
    "expression": "2 + 2"
})
```

### Local Registry

```python
from workers.core import ToolRegistry

# Create local registry
registry = ToolRegistry()

# Register tools
registry.register_tool(CalculatorTool, {"precision": 10})

# Use tools
tool_names = registry.get_tool_names()
schemas = registry.get_tool_schemas()
```

## Creating Custom Tools

### Method 1: Extend SimpleAgenticTool

```python
from workers.core.base_tool import SimpleAgenticTool

class WeatherTool(SimpleAgenticTool):
    def __init__(self, config=None):
        super().__init__(
            name="weather",
            description="Get weather information for a location",
            parameters={
                "location": {
                    "type": "string",
                    "description": "City name or coordinates"
                },
                "units": {
                    "type": "string", 
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            required=["location"],
            config=config
        )
    
    async def simple_execute(self, parameters, **kwargs):
        location = parameters["location"]
        units = parameters.get("units", "celsius")
        
        # Your weather API implementation here
        weather_data = await self.fetch_weather(location, units)
        return weather_data
```

### Method 2: Extend AgenticBaseTool

```python
from workers.core.base_tool import AgenticBaseTool
from workers.core.tool_schemas import create_openai_tool_schema, ToolResult

class DatabaseTool(AgenticBaseTool):
    def __init__(self, config=None):
        super().__init__(config)
        self.db_connections = {}
    
    def get_openai_tool_schema(self):
        return create_openai_tool_schema(
            name="database",
            description="Execute database queries",
            parameters={
                "query": {"type": "string", "description": "SQL query"},
                "database": {"type": "string", "description": "Database name"}
            },
            required=["query", "database"]
        )
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        try:
            query = parameters["query"]
            db_name = parameters["database"]
            
            # Your database implementation here
            results = await self.execute_query(db_name, query)
            
            return ToolResult(
                success=True,
                result=results,
                metrics={"rows_affected": len(results)}
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))
```

## Configuration

### Tool Configuration Options

```python
config = {
    # Security settings
    "blocked_commands": ["rm", "sudo"],  # For BashExecutorTool
    "allow_dangerous": False,
    
    # Performance settings
    "timeout": 30,
    "max_file_size": 1024 * 1024,  # 1MB
    "max_results": 100,
    
    # Feature flags
    "debug": True,
    "enable_linting": True,
    
    # File restrictions
    "allowed_extensions": [".py", ".txt", ".md"],
    "exclude_dirs": [".git", "__pycache__"]
}

tool = CalculatorTool(config)
```

## Integration with Agents

Tools integrate seamlessly with the agent system:

```python
from workers import ReactAgent, get_global_tool_registry

# Register tools
register_tool(CalculatorTool)
register_tool(FileEditorTool)

# Create agent with tools
agent = ReactAgent(max_steps=10)
registry = get_global_tool_registry()

# Agent can now use registered tools
tools = {name: registry for name in registry.get_tool_names()}
agent.set_tools(tools)
```

## Best Practices

1. **Security**: Always validate inputs and use appropriate security measures
2. **Error Handling**: Provide detailed error messages and graceful failure handling
3. **Resource Management**: Properly clean up resources in `_cleanup_instance`
4. **Documentation**: Include comprehensive docstrings and parameter descriptions
5. **Testing**: Test both standalone and VERL integration scenarios
6. **Configuration**: Make tools configurable for different use cases

## Example Applications

- **Code Analysis**: Use SearchTool + FileEditorTool for code refactoring
- **Data Processing**: Combine CalculatorTool + BashExecutorTool for computations
- **System Administration**: Use BashExecutorTool + FileEditorTool for system management
- **Development Workflows**: All tools together for comprehensive development assistance

The tool system provides a solid foundation for building powerful agentic applications that can work across different environments and use cases.