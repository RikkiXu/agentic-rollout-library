# GeneralAgent - 通用ReAct智能体

GeneralAgent是一个基于ReAct框架的通用智能体，已集成到workers/agents/中，支持多轮工具调用和配置化使用。

## 主要特性

- ✅ **ReAct推理框架**: 思考-行动-观察的循环推理模式
- ✅ **可配置系统提示**: 自定义智能体的行为和角色
- ✅ **灵活工具配置**: 支持注册和使用多种工具
- ✅ **可调节推理轮数**: 控制最大推理步数
- ✅ **自定义终止条件**: 配置终止工具名称
- ✅ **完整轨迹记录**: 记录每一步的推理和行动
- ✅ **轨迹导出功能**: 支持JSON和TXT格式导出

## 快速开始

### 基本使用

```python
import asyncio
from workers.agents.general_agent import GeneralAgent, dump_trajectory
from workers.tools import CalculatorTool, FinishTool

async def your_llm_function(messages, **kwargs):
    """你的LLM生成函数"""
    # 调用你的LLM API
    pass

async def main():
    # 创建GeneralAgent
    agent = GeneralAgent(
        max_rounds=5,              # 最大推理轮数
        termination_tool_names=["finish"]  # 终止工具名
    )
    
    # 设置工具 (在实际使用中，这通常由AgenticRollout系统完成)
    tools = {
        "calculator": CalculatorTool(),
        "finish": FinishTool()
    }
    agent.set_tools(tools)
    
    # 运行推理轨迹
    trajectory = await agent.run_trajectory(
        prompt="Calculate 2 + 3 * 4",
        llm_generate_func=your_llm_function,
        request_id="task_001"
    )
    
    # 保存轨迹
    dump_trajectory(trajectory, "trajectory.json", "json")
    dump_trajectory(trajectory, "trajectory.txt", "txt")

asyncio.run(main())
```

### 自定义系统提示

```python
custom_prompt = """你是一个数学教师助手。使用ReAct框架来解决数学问题。

遵循以下模式:
1. Thought: 分析问题，制定解决策略
2. Action: 使用工具进行计算
3. 重复直到得到答案

可用工具:
- calculator: 进行数学计算
- finish: 提供最终答案"""

agent = GeneralAgent(
    system_prompt=custom_prompt,
    max_rounds=3,
    termination_tool_names=["finish"]
)

# 设置工具
tools = {
    "calculator": CalculatorTool(),
    "finish": FinishTool()
}
agent.set_tools(tools)
```

## 配置参数

### GeneralAgent初始化参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `system_prompt` | str | None | 自定义系统提示，如果为None则使用默认提示 |
| `max_rounds` | int | 10 | 最大推理轮数 |
| `termination_tool_names` | List[str] | ["finish"] | 终止工具名称列表 |

**注意**: 工具通过`agent.set_tools(tools)`方法设置，在实际使用中通常由AgenticRollout系统自动管理。

### 其他配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_tokens_per_step` | int | 512 | 每步最大token数 |
| `temperature` | float | 0.7 | 生成温度 |
| `require_thought_before_action` | bool | True | 是否要求行动前思考 |
| `max_consecutive_thoughts` | int | 3 | 最大连续思考次数 |

## 支持的工具

### 内置工具

- **calculator**: 数学计算工具，支持表达式求值和各种数学运算
- **search**: 文本搜索工具，支持文件和目录搜索
- **file_editor**: 文件编辑工具，支持文件读写和编辑
- **bash_executor**: Bash命令执行工具
- **finish**: 任务完成工具（自动包含）

### K8s工具（需要kodo依赖）

- **k8s_bash_executor**: K8s环境下的Bash执行工具
- **k8s_file_editor**: K8s环境下的文件编辑工具
- **k8s_search**: K8s环境下的搜索工具

## 轨迹格式

### JSON格式
```json
{
  "request_id": "task_001",
  "is_completed": true,
  "final_reward": 0.0,
  "total_tokens": 68,
  "steps": [
    {
      "step_type": "observation",
      "content": "Task: Calculate 2 + 3 * 4",
      "metadata": {...},
      "reward_score": null
    },
    {
      "step_type": "thought",
      "content": "I need to solve this math problem step by step.",
      "metadata": {...},
      "reward_score": null
    },
    {
      "step_type": "action",
      "content": "calculator(expression=2+3*4)",
      "metadata": {...},
      "tool_name": "calculator",
      "tool_args": {"expression": "2+3*4"}
    }
  ]
}
```

### TXT格式
```
Trajectory: task_001
Completed: True
Final Reward: 0.0
Total Steps: 7
==================================================

Step 1: OBSERVATION
Content: Task: Calculate 2 + 3 * 4
------------------------------

Step 2: THOUGHT
Content: I need to solve this math problem step by step.
------------------------------

Step 3: ACTION
Content: calculator(expression=2+3*4)
Tool: calculator
Args: {'expression': '2+3*4'}
------------------------------
```

## 使用场景

### 数学问题求解
```python
agent = GeneralAgent(tool_names=["calculator"])
trajectory = await agent.run_trajectory(
    prompt="解方程 2x + 3 = 11，求x的值",
    llm_generate_func=llm_func,
    request_id="math_001"
)
```

### 代码分析
```python
agent = GeneralAgent(tool_names=["search", "file_editor"])
trajectory = await agent.run_trajectory(
    prompt="在项目中找到所有使用了某个函数的地方",
    llm_generate_func=llm_func,
    request_id="code_001"
)
```

### 复杂任务推理
```python
agent = GeneralAgent(
    tool_names=["calculator", "search", "file_editor"],
    max_rounds=10
)
trajectory = await agent.run_trajectory(
    prompt="分析代码性能并提出优化建议",
    llm_generate_func=llm_func,
    request_id="complex_001"
)
```

## ReAct推理模式

GeneralAgent遵循ReAct（Reasoning + Acting）框架：

1. **Thought**: 智能体分析当前情况，思考下一步应该做什么
2. **Action**: 智能体调用工具执行具体动作
3. **Observation**: 智能体接收工具执行结果
4. **重复**: 循环以上过程直到任务完成

### 示例推理过程

```
观察: Task: Calculate 2 + 3 * 4

思考: I need to solve this math problem step by step.

行动: calculator(expression=2+3*4)

观察: {'result': 14, 'formatted_result': '14'}

思考: The result is 14. Now I should provide the final answer.

行动: finish(answer=The result of 2+3*4 is 14)

完成: 任务成功完成
```

## 高级特性

### 自定义工具

可以通过工具注册系统添加自定义工具：

```python
from workers.core.base_tool import BaseAgenticTool
from workers.core.tool_registry import get_global_tool_registry

class MyCustomTool(BaseAgenticTool):
    def get_openai_tool_schema(self):
        # 定义工具schema
        pass
    
    async def execute_tool(self, instance_id, parameters, **kwargs):
        # 实现工具逻辑
        pass

# 注册工具
registry = get_global_tool_registry()
registry.register_tool(MyCustomTool)
```

### 轨迹分析

```python
import json

# 加载轨迹
with open('trajectory.json', 'r') as f:
    data = json.load(f)

# 分析步骤类型分布
step_types = [step['step_type'] for step in data['steps']]
print(f"步骤类型分布: {dict(Counter(step_types))}")

# 分析工具使用情况
tool_usage = [step.get('tool_name') for step in data['steps'] if step.get('tool_name')]
print(f"工具使用情况: {dict(Counter(tool_usage))}")
```

## 故障排除

### 常见问题

1. **工具未找到**
   - 确保工具名称在`tool_names`中
   - 检查工具是否正确注册

2. **推理循环过多**
   - 调整`max_rounds`参数
   - 检查终止条件是否合理

3. **系统提示无效**
   - 确保提示格式正确
   - 检查LLM是否遵循ReAct格式

### 调试技巧

1. 启用详细日志：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. 检查轨迹步骤：
```python
for i, step in enumerate(trajectory.steps):
    print(f"Step {i+1}: {step.step_type} - {step.content[:100]}")
```

## 许可证

MIT License - 详见LICENSE文件

## 贡献

欢迎提交Issue和Pull Request来改进GeneralAgent！