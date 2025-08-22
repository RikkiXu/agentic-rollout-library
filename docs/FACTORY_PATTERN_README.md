# 工厂模式 - 基于类名和配置的工具与智能体创建

agentic-rollout-library 实现了完整的工厂模式，支持基于类名和配置参数创建工具和智能体实例。

## 核心特性

- 🏭 **基于类名创建**: 通过字符串类名创建实例，无需手动导入
- ⚙️ **配置参数支持**: 支持将配置参数传递给构造函数
- 🚀 **自动模块加载**: 自动加载和缓存类定义
- 📦 **批量创建**: 支持批量创建多个实例
- 🔍 **信息查询**: 内置工具和智能体信息查询功能

## ToolFactory - 工具工厂

### 基本使用

```python
from workers.core import create_tool, create_tools

# 创建单个工具
calculator = create_tool("Calculator", {
    "debug": True,
    "precision": 10
})

# 批量创建工具
tools = create_tools({
    "Calculator": {"debug": False, "precision": 6},
    "Search": {"max_results": 100},
    "FileEditor": {"encoding": "utf-8", "backup": True},
    "Finish": {}
})
```

### 支持的工具类型

| 工具名称 | 类名 | 主要配置参数 |
|----------|------|-------------|
| Calculator | CalculatorTool | debug, precision |
| Search | SearchTool | max_results, max_file_size, search_extensions |
| FileEditor | FileEditorTool | encoding, backup, max_file_size |
| BashExecutor | BashExecutorTool | timeout, shell, working_dir |
| Finish | FinishTool | (通常无需配置) |
| K8sBashExecutor | K8sBashExecutorTool | pod_name, namespace, timeout |
| K8sFileEditor | K8sFileEditorTool | pod_name, namespace, encoding |
| K8sSearch | K8sSearchTool | pod_name, namespace, max_results |

### 工具配置示例

```python
# 数学计算场景
math_tools = create_tools({
    "Calculator": {
        "debug": True,
        "precision": 10
    },
    "Finish": {}
})

# 文件处理场景
file_tools = create_tools({
    "FileEditor": {
        "encoding": "utf-8",
        "backup": True,
        "max_file_size": 10485760  # 10MB
    },
    "Search": {
        "max_results": 100,
        "search_extensions": [".py", ".txt", ".md"],
        "exclude_dirs": [".git", "__pycache__"]
    },
    "Finish": {}
})

# 系统管理场景
system_tools = create_tools({
    "BashExecutor": {
        "timeout": 30,
        "shell": "/bin/bash",
        "working_dir": "/tmp"
    },
    "FileEditor": {
        "encoding": "utf-8"
    },
    "Finish": {}
})
```

## AgentFactory - 智能体工厂

### 基本使用

```python
from workers.core import create_agent, create_agents

# 创建单个智能体
agent = create_agent("General", {
    "max_rounds": 5,
    "system_prompt": "你是一个数学助手。",
    "termination_tool_names": ["finish"]
})

# 批量创建智能体
agents = create_agents({
    "General": {
        "max_rounds": 5,
        "system_prompt": "你是一个编程助手。"
    },
    "React": {
        "max_steps": 10,
        "temperature": 0.7
    }
})
```

### 支持的智能体类型

| 智能体名称 | 类名 | 主要配置参数 |
|-----------|------|-------------|
| General | GeneralAgent | max_rounds, system_prompt, termination_tool_names |
| React | ReactAgent | max_steps, temperature, require_thought_before_action |
| Tool | ToolAgent | max_steps, tool_selection_strategy |
| Coding | CodingAgent | max_steps, programming_language, code_style |

### 智能体配置示例

```python
# 不同角色的智能体配置
role_agents = create_agents({
    # 数学导师
    "MathTutor": {
        "max_rounds": 5,
        "system_prompt": """你是一个耐心的数学导师，专门帮助学生理解数学概念。
        使用ReAct框架：
        1. Thought: 分析学生的问题
        2. Action: 使用适当的工具
        3. 重复直到问题解决""",
        "termination_tool_names": ["finish"]
    },
    
    # 代码审查员
    "CodeReviewer": {
        "max_rounds": 8,
        "system_prompt": """你是一个经验丰富的代码审查员。
        专注于：
        - 代码质量和可维护性
        - 安全性检查
        - 性能优化建议
        - 最佳实践""",
        "termination_tool_names": ["finish"]
    },
    
    # 问题解决者
    "ProblemSolver": {
        "max_rounds": 10,
        "system_prompt": """你是一个系统性的问题解决者。
        方法：
        1. 问题分析和分解
        2. 解决方案设计
        3. 实施和验证
        4. 总结和优化""",
        "termination_tool_names": ["finish"]  
    }
})
```

## 完整工作流示例

```python
import asyncio
from workers.core import create_tools, create_agent

async def complete_workflow():
    # 1. 创建工具
    tools = create_tools({
        "Calculator": {"debug": False},
        "Search": {"max_results": 20},
        "FileEditor": {"encoding": "utf-8"},
        "Finish": {}
    })
    
    # 2. 创建智能体
    agent = create_agent("General", {
        "max_rounds": 5,
        "system_prompt": """你是一个全能助手。可以：
        - 进行数学计算
        - 搜索和处理文件
        - 编辑文档
        使用ReAct框架系统性地解决问题。""",
        "termination_tool_names": ["finish"]
    })
    
    # 3. 配置智能体工具
    agent.set_tools(tools)
    
    # 4. 执行任务
    trajectory = await agent.run_trajectory(
        prompt="分析项目中的Python文件数量并计算平均文件大小",
        llm_generate_func=your_llm_function,
        request_id="workflow_001"
    )
    
    return trajectory
```

## 高级功能

### 自定义工具注册

```python
from workers.core import get_global_tool_factory
from workers.core.base_tool import BaseAgenticTool

# 自定义工具类
class MyCustomTool(BaseAgenticTool):
    def __init__(self, config):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30)
    
    def get_openai_tool_schema(self):
        # 定义工具schema
        pass
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        # 实现工具逻辑
        pass

# 注册自定义工具
factory = get_global_tool_factory()
factory.register_tool_class("MyCustom", MyCustomTool)

# 使用自定义工具
my_tool = create_tool("MyCustom", {
    "api_key": "your-api-key",
    "timeout": 60
})
```

### 自定义智能体注册

```python
from workers.core import get_global_agent_factory
from workers.core.base_agent import BaseAgent

# 自定义智能体类
class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = kwargs.get("custom_param", "default")
    
    async def run_trajectory(self, prompt, llm_generate_func, request_id, **kwargs):
        # 实现智能体逻辑
        pass

# 注册自定义智能体
factory = get_global_agent_factory()
factory.register_agent_class("MyCustom", MyCustomAgent)

# 使用自定义智能体
my_agent = create_agent("MyCustom", {
    "custom_param": "custom_value",
    "max_steps": 15
})
```

### 信息查询功能

```python
from workers.core import get_global_tool_factory, get_global_agent_factory

# 查询可用工具
tool_factory = get_global_tool_factory()
available_tools = tool_factory.list_available_tools()
print("Available tools:", list(available_tools.keys()))

# 获取工具详细信息
calculator_info = tool_factory.get_tool_info("Calculator")
print("Calculator schema:", calculator_info["schema"])

# 查询可用智能体
agent_factory = get_global_agent_factory()
available_agents = agent_factory.list_available_agents()
print("Available agents:", list(available_agents.keys()))

# 获取智能体详细信息
general_info = agent_factory.get_agent_info("General")
print("General agent class:", general_info["class"])
```

## 配置最佳实践

### 场景化配置

```python
# 定义不同场景的配置模板
SCENARIO_CONFIGS = {
    "data_analysis": {
        "tools": {
            "Calculator": {"precision": 10, "debug": False},
            "Search": {"max_results": 200, "search_extensions": [".csv", ".json", ".py"]},
            "FileEditor": {"encoding": "utf-8"},
            "Finish": {}
        },
        "agent": {
            "max_rounds": 8,
            "system_prompt": "你是一个数据分析专家...",
            "termination_tool_names": ["finish"]
        }
    },
    
    "code_review": {
        "tools": {
            "Search": {"search_extensions": [".py", ".js", ".java", ".cpp"]},
            "FileEditor": {"encoding": "utf-8", "backup": True},
            "BashExecutor": {"timeout": 60},
            "Finish": {}
        },
        "agent": {
            "max_rounds": 10,
            "system_prompt": "你是一个代码审查专家...",
            "termination_tool_names": ["finish"]
        }
    }
}

# 使用场景配置
def create_scenario(scenario_name):
    config = SCENARIO_CONFIGS[scenario_name]
    
    tools = create_tools(config["tools"])
    agent = create_agent("General", config["agent"])
    agent.set_tools(tools)
    
    return agent, tools
```

### 环境变量配置

```python
import os

# 从环境变量读取配置
def create_from_env():
    tools = create_tools({
        "Calculator": {
            "debug": os.getenv("CALCULATOR_DEBUG", "false").lower() == "true",
            "precision": int(os.getenv("CALCULATOR_PRECISION", "6"))
        },
        "Search": {
            "max_results": int(os.getenv("SEARCH_MAX_RESULTS", "100")),
            "max_file_size": int(os.getenv("SEARCH_MAX_FILE_SIZE", "1048576"))
        },
        "Finish": {}
    })
    
    agent = create_agent("General", {
        "max_rounds": int(os.getenv("AGENT_MAX_ROUNDS", "5")),
        "system_prompt": os.getenv("AGENT_SYSTEM_PROMPT", "你是一个helpful助手。"),
        "termination_tool_names": ["finish"]
    })
    
    return agent, tools
```

## 错误处理

```python
from workers.core import create_tool, create_agent

# 工具创建错误处理
try:
    tool = create_tool("Calculator", {"precision": 10})
except ValueError as e:
    print(f"工具创建失败: {e}")
    # 使用默认配置
    tool = create_tool("Calculator", {})

# 智能体创建错误处理
try:
    agent = create_agent("General", {"max_rounds": 5})
except ValueError as e:
    print(f"智能体创建失败: {e}")
    # 使用备选智能体
    agent = create_agent("React", {"max_steps": 10})
```

## 性能优化

- ✅ **类缓存**: 已加载的类会被缓存，避免重复导入
- ✅ **延迟加载**: 只有在实际使用时才加载类定义
- ✅ **批量创建**: 使用`create_tools`和`create_agents`批量创建更高效
- ✅ **配置复用**: 相同配置的实例可以复用

## 总结

工厂模式提供了一个灵活、可配置的方式来创建工具和智能体实例：

1. **简化使用**: 无需手动导入，通过字符串名称即可创建
2. **配置驱动**: 通过配置参数控制实例行为
3. **扩展性好**: 易于注册和使用自定义工具/智能体
4. **统一接口**: 提供一致的创建和管理接口
5. **性能优化**: 内置缓存和优化机制

这使得整个系统更加模块化和易于维护，特别适合需要动态配置和大规模部署的场景。