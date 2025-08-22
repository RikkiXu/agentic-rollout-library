# Agentic Rollout Library 示例文档

这个目录包含了清晰、聚焦的示例，展示了Agentic Rollout Library的功能特性。该库既可以独立使用，也可以与VERL (Versatile Environment for Reinforcement Learning) 集成使用。

## 🚀 快速开始

### 前置要求

**独立使用：**
```bash
pip install pydantic pyyaml asyncio
```

**VERL集成：**
```bash
# 确保VERL已安装并在Python路径中
export PYTHONPATH="/path/to/verl:$PYTHONPATH"
```

## 📁 示例结构

### 核心示例

#### 1. **基础使用** (`basic_usage.py`)
基础概念和模式：
- 创建不同配置的智能体rollout
- 使用ReAct智能体进行多步推理
- 自定义智能体实现
- 批处理和工具集成
- 轨迹管理和分析

**运行：** `python basic_usage.py`

#### 2. **高级集成** (`advanced_integration.py`)
生产就绪的模式：
- 真实LLM集成（兼容OpenAI/Anthropic）
- 复杂的多工具工作流
- 智能体组合和链式调用
- 错误处理和性能监控
- 生产部署模式

**运行：** `python advanced_integration.py`

#### 3. **核心工具演示** (`core_tools_demo.py`)
展示新的统一工具系统：
- 单独工具使用（计算器、Bash、文件编辑器、搜索）
- 工具注册表管理
- 工具参数验证
- 错误处理和日志记录

**运行：** `python core_tools_demo.py`

#### 4. **工具集成演示** (`tool_integration_demo.py`)
完整的集成模式：
- 工具与ReAct智能体结合使用
- 全局工具注册表使用
- 模拟LLM集成
- 真实世界工作流示例

**运行：** `python tool_integration_demo.py`

### 配置模板 (`config/`)
即用型配置文件：
- `agentic_rollout_config.yaml`：通用rollout设置
- `react_math_config.yaml`：数学专用ReAct智能体配置
- `coding_agent_config.yaml`：软件工程任务配置

### 测试示例

#### **基础测试** (`test_agentic_rollout.py`)
- rollout功能的单元测试模式
- 模拟LLM测试场景
- 智能体行为验证

#### **编码智能体测试** (`test_coding_agent.py`)
- 与新核心工具的集成
- 端到端工作流测试
- 工具验证模式

#### **验证套件** (`validation_tests.py`)
- 全面的系统验证
- 性能基准测试
- 跨配置测试

### 文档 (`docs/`)
- `AgenticRollout_Technical_Design_Document.md`：详细技术文档

## 🛠️ 集成模式

### 独立使用
```python
from workers import (
    AgenticRollout, AgenticRolloutConfig,
    CalculatorTool, BashExecutorTool, FileEditorTool, SearchTool,
    get_global_tool_registry, register_tool
)

# 注册工具
register_tool(CalculatorTool, {"debug": True})
register_tool(BashExecutorTool, {"timeout": 30})

# 创建rollout
config = AgenticRolloutConfig(agent_type="react", max_steps=10)
rollout = AgenticRollout(config=config, llm_generate_func=your_llm_func)

# 运行轨迹
trajectory = await rollout.agent.run_trajectory(
    prompt={"content": "计算5的阶乘"},
    llm_generate_func=your_llm_func,
    request_id="example"
)
```

### VERL集成
```python
# 当VERL可用时，工具自动继承VERL兼容性
from verl.workers.rollout.agentic_rollout import AgenticRollout
from workers import CalculatorTool  # 仍然使用我们的统一工具

# 作为VERL rollout worker使用
rollout = AgenticRollout(config, llm_generate_func, tokenizer)
output_data = await rollout.generate_sequences(input_prompts)
```

## 🔧 演示的关键特性

### 统一工具框架
- **核心工具**：计算器、Bash执行器、文件编辑器、搜索工具
- **VERL兼容性**：自动检测和集成
- **安全特性**：带有可配置限制的安全执行
- **工具注册表**：集中化管理和发现

### 智能体系统
- **ReAct智能体**：内置推理和行动能力
- **自定义智能体**：易于扩展的框架
- **工具集成**：智能体内的无缝工具使用
- **轨迹管理**：完整的执行跟踪

### LLM集成
- **多个提供商**：OpenAI、Anthropic、自定义端点
- **模拟客户端**：无外部依赖的测试
- **异步支持**：完整的async/await兼容性
- **错误处理**：健壮的错误恢复

## 🚀 运行示例

所有示例都是自包含的，可以直接运行：

```bash
cd examples

# 基础功能
python basic_usage.py

# 高级模式
python advanced_integration.py

# 工具系统演示
python core_tools_demo.py
python tool_integration_demo.py

# 测试
python test_agentic_rollout.py
python test_coding_agent.py
python validation_tests.py
```

## 📊 新功能亮点

这个清理后的示例目录专注于：

✅ **统一工具系统**：带有VERL兼容性的新核心工具  
✅ **简化结构**：移除了冗余和过时的示例  
✅ **更新的导入**：兼容独立和VERL使用  
✅ **更好的文档**：清晰、聚焦的示例和全面的注释  
✅ **生产模式**：真实世界的使用场景和最佳实践  

## 🔍 详细功能介绍

### 核心工具详解

#### 计算器工具 (`CalculatorTool`)
- **表达式求值**：支持复杂数学表达式，如 `sqrt(16) + factorial(4)`
- **科学函数**：三角函数、对数、指数运算
- **统计操作**：平均值、标准差、最值计算
- **安全执行**：使用AST解析确保安全的数学计算

```python
# 使用示例
calc = CalculatorTool()
result = await calc.execute_tool(instance_id, {
    "expression": "sin(pi/2) + log(e)"  # 结果：2.0
})
```

#### Bash执行器工具 (`BashExecutorTool`)
- **安全命令执行**：内置危险命令过滤
- **超时保护**：可配置的执行超时
- **输出捕获**：完整的stdout/stderr捕获
- **工作目录支持**：可指定命令执行目录

```python
# 使用示例
bash = BashExecutorTool({"timeout": 30})
result = await bash.execute_tool(instance_id, {
    "command": "python -c 'print(\"Hello World\")'",
    "working_directory": "/tmp"
})
```

#### 文件编辑器工具 (`FileEditorTool`)
- **文件操作**：创建、查看、编辑文件
- **字符串替换**：精确的字符串替换功能
- **撤销功能**：支持编辑历史和撤销
- **语法检查**：Python文件的语法验证

```python
# 使用示例
editor = FileEditorTool()
result = await editor.execute_tool(instance_id, {
    "command": "create",
    "path": "/tmp/test.py",
    "file_text": "print('Hello from Agentic Tools!')"
})
```

#### 搜索工具 (`SearchTool`)
- **文本搜索**：在文件中搜索文本模式，支持正则表达式
- **文件名搜索**：按文件名模式查找文件
- **目录结构搜索**：搜索目录结构和路径
- **上下文显示**：显示匹配行的上下文

```python
# 使用示例
search = SearchTool()
result = await search.execute_tool(instance_id, {
    "command": "search_text",
    "pattern": "import",
    "path": "/path/to/project",
    "file_extensions": [".py"]
})
```

### 智能体集成模式

#### ReAct智能体工作流
1. **观察**：接收任务或环境反馈
2. **思考**：分析当前情况并制定计划
3. **行动**：执行工具调用或给出最终答案
4. **观察**：获取行动结果
5. **循环**：重复直到问题解决

```python
# ReAct智能体示例
agent = ReactAgent(max_steps=8)
trajectory = await agent.run_trajectory(
    prompt={"content": "创建一个Python脚本来计算斐波那契数列"},
    llm_generate_func=llm_func,
    request_id="fibonacci_task"
)
```

#### 工具注册和管理
```python
from workers import get_global_tool_registry, register_tool

# 注册工具到全局注册表
register_tool(CalculatorTool, {"precision": 10}, "calc")
register_tool(FileEditorTool, {"max_file_size": 1024*1024}, "editor")

# 获取注册表并使用
registry = get_global_tool_registry()
calc_instance = await registry.create_tool_instance("calc")
result = await registry.execute_tool("calc", calc_instance, {
    "expression": "2**10"
})
```

## 🎯 应用场景

### 1. 数学和科学计算
```python
# 科学计算流水线
trajectory = await agent.run_trajectory(
    prompt={"content": "计算正态分布的95%置信区间"},
    llm_generate_func=llm_func
)
```

### 2. 代码生成和测试
```python
# 代码生成工作流
trajectory = await agent.run_trajectory(
    prompt={"content": "创建一个排序算法并测试其正确性"},
    llm_generate_func=llm_func
)
```

### 3. 文件处理和分析
```python
# 文件分析流水线
trajectory = await agent.run_trajectory(
    prompt={"content": "分析项目中的Python文件并生成导入依赖报告"},
    llm_generate_func=llm_func
)
```

### 4. 系统管理任务
```python
# 系统管理自动化
trajectory = await agent.run_trajectory(
    prompt={"content": "检查系统磁盘使用情况并清理临时文件"},
    llm_generate_func=llm_func
)
```

## ⚙️ 配置选项

### AgenticRolloutConfig 配置参数
```python
config = AgenticRolloutConfig(
    # 智能体设置
    agent_type="react",              # 智能体类型：react, tool, custom
    max_steps=10,                    # 最大轨迹步数
    max_tokens_per_step=512,         # 每步最大token数
    temperature=0.7,                 # LLM采样温度
    
    # 工具配置
    tools_config={                   # 工具特定设置
        "calculator": {"precision": 10},
        "bash": {"timeout": 30}
    },
    
    # 性能设置
    concurrent_requests=2,           # 并发请求数
    batch_size=1,                    # 批处理大小
    
    # 输出设置
    include_trajectory_in_output=True,  # 输出中包含轨迹
    save_trajectories=True,             # 保存轨迹到磁盘
    trajectory_save_path="./trajectories"  # 轨迹保存路径
)
```

### 工具配置示例
```python
# 计算器工具配置
calc_config = {
    "debug": True,                   # 调试模式
    "precision": 10                  # 计算精度
}

# Bash执行器配置
bash_config = {
    "timeout": 30,                   # 超时时间（秒）
    "blocked_commands": ["rm", "sudo"],  # 禁用命令
    "allow_dangerous": False         # 是否允许危险操作
}

# 文件编辑器配置
editor_config = {
    "max_file_size": 1024*1024,     # 最大文件大小
    "allowed_extensions": [".py", ".txt"],  # 允许的文件扩展名
    "enable_linting": True           # 启用语法检查
}

# 搜索工具配置
search_config = {
    "max_results": 100,             # 最大搜索结果数
    "search_extensions": [".py", ".js"],  # 搜索文件类型
    "exclude_dirs": [".git", "__pycache__"]  # 排除目录
}
```

## 🧪 测试和调试

### 启用调试日志
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或者只启用特定组件的调试
logging.getLogger("workers.tools").setLevel(logging.DEBUG)
```

### 轨迹分析
```python
# 分析轨迹执行情况
trajectory = await agent.run_trajectory(...)

print(f"总步数: {len(trajectory.steps)}")
print(f"工具调用次数: {len(trajectory.get_tool_calls())}")
print(f"最终奖励: {trajectory.get_total_reward()}")

# 保存轨迹用于分析
with open("trajectory.json", "w") as f:
    json.dump(trajectory.to_dict(), f, indent=2)
```

### 性能监控
```python
import time

start_time = time.time()
trajectory = await agent.run_trajectory(...)
execution_time = time.time() - start_time

print(f"执行时间: {execution_time:.2f}秒")
print(f"平均每步时间: {execution_time/len(trajectory.steps):.2f}秒")
```

## 🤝 贡献示例

添加新示例时请遵循以下准则：

1. **导入模式**：使用既定的导入模式 (`from workers import ...`)
2. **文档字符串**：包含全面的文档字符串和注释
3. **错误处理**：添加适当的错误处理和日志记录
4. **兼容性**：提供独立和VERL集成兼容性
5. **配置示例**：在相关处包含配置示例
6. **测试**：使用模拟和真实LLM集成进行测试

### 示例模板
```python
#!/usr/bin/env python3
"""
新示例的描述
展示特定功能或使用模式
"""

import asyncio
import logging
import sys
from pathlib import Path

# 标准导入模式
sys.path.append(str(Path(__file__).parent.parent))
from workers import AgenticRollout, AgenticRolloutConfig

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_function():
    """演示函数的详细描述"""
    try:
        # 实现示例逻辑
        pass
    except Exception as e:
        logger.error(f"演示失败: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(demo_function())
```

## 📝 迁移指南

如果你正在从旧版本示例迁移：

1. **更新导入**：使用 `from workers import ...`
2. **替换工具实现**：使用新的核心工具替代旧的工具实现
3. **使用工具注册表**：通过全局工具注册表管理工具
4. **更新配置格式**：使用新的配置架构格式
5. **测试兼容性**：确保同时支持独立和VERL使用

## 🔗 相关文档

- 主项目README：`../README.md`
- 工具系统文档：`../workers/tools/README.md`
- 技术设计文档：`./docs/AgenticRollout_Technical_Design_Document.md`
- API参考：查看各模块的docstring文档

## 💡 最佳实践

1. **工具选择**：根据任务特性选择合适的工具组合
2. **错误处理**：实现健壮的错误处理和恢复机制
3. **性能优化**：合理设置并发参数和超时时间
4. **安全考虑**：特别是使用Bash执行器时要注意安全设置
5. **日志记录**：使用适当的日志级别进行调试和监控
6. **配置管理**：将配置外部化以便于部署和调试

这个示例集合为你提供了使用Agentic Rollout Library的完整指南，无论是学习、开发还是生产部署都能找到合适的参考模式！