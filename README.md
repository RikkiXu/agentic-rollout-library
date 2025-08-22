# Agentic Rollout Library

> [English Version](README_EN.md) | 中文版

一个高度可定制的智能体推理框架，支持工具集成、自定义系统提示和灵活的动作解析。该库提供了构建生产级AI智能体所需的所有核心组件。

## 🌟 核心特性

### 🎯 高度可定制化
- **工具定制**：每个工具都可以自定义描述，支持不同的提示格式
- **系统提示定制**：完全控制系统提示的生成，支持动态变量注入
- **动作解析定制**：支持自定义动作解析器（JSON、XML等格式）
- **智能体行为定制**：可配置的终止条件、最大轮数、调试模式等

### 🤖 通用智能体框架 (GeneralAgent)
- **ReAct框架**：内置思考-行动-观察循环
- **灵活的工具系统**：动态注册和管理工具
- **轨迹管理**：完整的执行轨迹跟踪和保存
- **终止工具支持**：可配置哪些工具触发智能体终止
- **调试模式**：详细的LLM输入/输出日志

### 🛠️ 强大的工具系统
- **统一工具接口**：所有工具继承自 `AgenticBaseTool`
- **OpenAI Schema支持**：自动生成OpenAI函数调用格式
- **执行模式**：支持本地执行和K8s Pod执行
- **R2E工具集**：专为代码仓库编辑设计的工具集
  - `R2EBashExecutor`：安全的bash命令执行
  - `R2EFileEditor`：高级文件编辑（view/create/str_replace/insert/undo）
  - `R2ESearch`：代码搜索工具
  - `R2ESubmit`：任务完成提交

### 🏗️ 提示构建系统 (PromptBuilder)
```python
# 使用 PromptBuilder 创建动态提示
builder = PromptBuilder()
prompt = (builder
    .add_variable("task", "修复bug #123")
    .add_tools(tools, formatter=custom_formatter)
    .add_context({"repo": "pandas", "version": "2.0"})
    .add_section("Instructions", "请仔细分析代码...")
    .build())
```

### 🏭 工厂模式系统
- **工具工厂**：基于名称动态创建工具实例
- **智能体工厂**：统一的智能体创建接口
- **自动注册**：使用装饰器自动注册新组件

## 📁 项目结构

```
agentic-rollout-library/
├── workers/
│   ├── agents/
│   │   └── general_agent.py        # 通用ReAct智能体
│   ├── core/
│   │   ├── base_agent.py          # 智能体基类
│   │   ├── base_tool.py           # 工具基类
│   │   ├── tool_factory.py        # 工具工厂
│   │   ├── agent_factory.py       # 智能体工厂
│   │   └── trajectory.py          # 轨迹管理
│   ├── tools/
│   │   ├── bash_executor_tool.py  # Bash执行工具
│   │   ├── file_editor_tool.py    # 文件编辑工具
│   │   ├── search_tool.py         # 搜索工具
│   │   └── r2e_tools/            # R2E工具集
│   └── utils/
│       ├── llm_client.py          # LLM客户端
│       └── prompt_builder.py      # 提示构建器
└── tests/
    └── test_r2e_general_agent.py  # 完整示例
```

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd agentic-rollout-library

# 安装依赖
pip install -e .
```

### 环境配置

创建 `.env` 文件或设置环境变量：

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="your-base-url"
export LLM_MODEL_NAME="gpt-4"
```

### 基础使用示例

```python
from workers.agents.general_agent import GeneralAgent
from workers.core import create_tool
from workers.utils import create_llm_client

# 1. 创建工具
tools = {
    "bash": create_tool("BashExecutor"),
    "editor": create_tool("FileEditor"),
    "search": create_tool("Search"),
    "finish": create_tool("Finish")
}

# 2. 创建智能体
agent = GeneralAgent(
    max_rounds=10,
    termination_tool_names=["finish"]
)
agent.set_tools(tools)

# 3. 创建LLM客户端
llm_client = create_llm_client(
    api_key="your-key",
    base_url="your-url",
    model="gpt-4"
)

# 4. 运行任务
result = await agent.run_trajectory(
    prompt="在当前目录创建一个 hello.py 文件",
    llm_generate_func=llm_client.generate,
    request_id="task-001"
)
```

### 高级定制示例

#### 1. 自定义工具描述

```python
class CustomDescriptionWrapper:
    def __init__(self, tool, description):
        self.tool = tool
        self.custom_description = description
    
    def get_description(self):
        return self.custom_description
    
    def __getattr__(self, name):
        return getattr(self.tool, name)

# 包装工具with自定义描述
wrapped_tool = CustomDescriptionWrapper(
    original_tool,
    "我的自定义工具描述..."
)
```

#### 2. 动态系统提示

```python
def generate_custom_prompt(tools, **kwargs):
    task = kwargs.get('task_description', 'default task')
    return f"""
    你是一个专业的{kwargs.get('role', '助手')}。
    
    任务：{task}
    
    可用工具：
    {tools['editor'].get_description()}
    {tools['bash'].get_description()}
    
    {kwargs.get('additional_instructions', '')}
    """

# 使用动态提示
agent.system_prompt = generate_custom_prompt(
    tools,
    role="Python开发者",
    task_description="修复代码中的bug",
    additional_instructions="请遵循PEP8规范"
)
```

#### 3. 自定义动作解析器

```python
def parse_xml_action(output: str):
    """解析XML格式的动作"""
    import re
    match = re.search(r'<function=(\w+)>(.*?)</function>', output, re.DOTALL)
    if match:
        tool_name = match.group(1)
        # 解析参数...
        return {"tool_name": tool_name, "tool_args": {...}}
    return None

# 使用自定义解析器
agent = GeneralAgent(
    action_parser=parse_xml_action
)
```

## 🔧 K8s 执行模式

支持在Kubernetes Pod中执行工具：

```python
k8s_config = {
    "execution_mode": "k8s",
    "pod_name": "my-dev-pod",
    "namespace": "default"
}

# 创建K8s执行的工具
bash_tool = create_tool("BashExecutor", k8s_config)
file_tool = create_tool("FileEditor", k8s_config)
```

## 📚 核心概念

### 工具 (Tools)
- 继承自 `AgenticBaseTool`
- 实现 `execute_tool` 方法
- 提供 `get_openai_tool_schema` 返回工具描述
- 支持 `get_description` 自定义描述

### 智能体 (Agents)
- 继承自 `BaseAgent`
- 管理工具集合
- 处理LLM交互
- 维护执行轨迹

### 轨迹 (Trajectory)
- 记录所有思考、动作和观察
- 支持序列化和反序列化
- 用于调试和分析

## 🤝 贡献

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 Apache 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。