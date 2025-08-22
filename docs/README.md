# 文档目录

本目录包含 agentic-rollout-library 的详细文档。

## 文档索引

### 核心功能文档

- **[GeneralAgent 使用指南](GENERAL_AGENT_README.md)** - 通用ReAct智能体的完整使用文档
- **[工厂模式指南](FACTORY_PATTERN_README.md)** - 基于类名和配置的工具与智能体创建文档

### 快速开始

1. **新用户**: 建议先阅读 [GeneralAgent 使用指南](GENERAL_AGENT_README.md) 了解基本概念
2. **高级用户**: 查看 [工厂模式指南](FACTORY_PATTERN_README.md) 了解灵活的创建方式
3. **开发者**: 参考 `tests/` 目录下的示例代码

## 文档结构

```
docs/
├── README.md                    # 本文件 - 文档索引
├── GENERAL_AGENT_README.md      # GeneralAgent详细文档
└── FACTORY_PATTERN_README.md    # 工厂模式详细文档
```

## 主要特性概览

### GeneralAgent
- 基于ReAct框架的通用智能体
- 支持可配置的系统提示
- 灵活的工具配置和终止条件
- 完整的轨迹记录和导出

### 工厂模式
- 基于类名创建工具和智能体
- 支持配置参数传递
- 自动模块加载和缓存
- 批量创建和信息查询

## 使用流程

### 基本使用流程
```
1. 创建工具 → 2. 创建智能体 → 3. 配置工具 → 4. 执行任务 → 5. 处理结果
```

### 工厂模式流程
```
1. 定义配置 → 2. 批量创建 → 3. 智能体配置 → 4. 执行任务 → 5. 结果分析
```

## 代码示例

### 快速开始示例

```python
from workers.core import create_tool, create_agent

# 创建工具
tools = {
    "calculator": create_tool("Calculator", {"debug": False}),
    "finish": create_tool("Finish")
}

# 创建智能体
agent = create_agent("General", {
    "max_rounds": 5,
    "termination_tool_names": ["finish"]
})

# 配置和使用
agent.set_tools(tools)
trajectory = await agent.run_trajectory(
    prompt="Calculate 2+3*4",
    llm_generate_func=your_llm_function,
    request_id="example_001"
)
```

## 相关资源

- **测试代码**: `tests/` 目录包含完整的测试和示例
- **源码**: `workers/` 目录包含核心实现
- **配置示例**: 文档中包含各种场景的配置示例

## 贡献指南

### 文档更新
1. 修改相应的Markdown文件
2. 确保示例代码可以正常运行
3. 更新本索引文件（如需要）

### 新增文档
1. 在docs目录创建新的Markdown文件
2. 遵循现有文档的格式和风格
3. 在本README中添加索引链接
4. 在相关测试中添加对应示例

## 版本历史

- **v0.1.0**: 初始版本，包含GeneralAgent和工厂模式基础功能
- 后续版本将在此处记录主要变更

## 支持与反馈

如有问题或建议，请：
1. 查看相关文档是否有解答
2. 检查tests目录的示例代码
3. 提交Issue或PR

---

*本文档持续更新中，欢迎贡献改进建议*