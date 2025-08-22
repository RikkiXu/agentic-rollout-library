# OpenHands Tools

这个目录包含了从 OpenHands 的 codeact_agent 工具转换而来的工具实现，支持本地执行和 K8s 环境。

## 工具列表

### 1. OpenHandsBashTool
- **功能**: 执行 bash 命令
- **特性**: 
  - 支持本地和 K8s 执行模式
  - 持久化会话支持（OpenHands 风格）
  - 软超时机制（默认 10 秒）
  - 支持与运行中进程的交互

### 2. OpenHandsStrReplaceEditorTool  
- **功能**: 文件查看、创建和编辑
- **支持命令**:
  - `view`: 查看文件或目录
  - `create`: 创建新文件
  - `str_replace`: 字符串替换
  - `insert`: 插入文本
  - `undo_edit`: 撤销编辑
- **特性**:
  - 支持本地和 K8s 文件操作
  - 精确的字符串匹配和替换
  - 编辑历史追踪

### 3. OpenHandsFinishTool
- **功能**: 任务完成信号
- **用途**: 标志任务完成或无法继续进行

## 配置选项

所有工具都支持以下配置:

```python
config = {
    "execution_mode": "local",  # 或 "k8s" 
    "pod_name": "your-pod-name",  # K8s 模式必需
    "namespace": "default",
    "kubeconfig_path": None,
    "working_dir": "/workspace",  # OpenHands 默认工作目录
    "use_custom_description": True  # 使用 OpenHands 风格的描述
}
```

## 使用示例

参见 `/tests/test_openhands.py` 获取完整的使用示例。

### 基本用法

```python
from workers.tools.openhands_tools import (
    OpenHandsBashTool,
    OpenHandsStrReplaceEditorTool,
    OpenHandsFinishTool
)

# 创建配置
config = {
    "execution_mode": "local",
    "working_dir": "/workspace"
}

# 创建工具
bash_tool = OpenHandsBashTool(config)
editor_tool = OpenHandsStrReplaceEditorTool(config)
finish_tool = OpenHandsFinishTool(config)

# 使用工具
result = await bash_tool.execute_tool("test", {"command": "echo 'Hello OpenHands!'"})
```

### 与 GeneralAgent 集成

```python
from workers.agents.general_agent import GeneralAgent

# 创建工具映射
tools = {
    "openhands_bash": OpenHandsBashTool(config),
    "openhands_str_replace_editor": OpenHandsStrReplaceEditorTool(config),
    "openhands_finish": OpenHandsFinishTool(config)
}

# 创建 agent
agent = GeneralAgent(
    llm_client=llm_client,
    tools=tools,
    action_parser=parse_xml_action_openhands,
    system_prompt=generate_openhands_system_prompt(list(tools.keys()))
)

# 运行任务
result = await agent.arun("Create a Python script that prints hello world")
```

## K8s 部署

工具支持在 K8s 环境中运行，只需设置相应的配置参数：

```python
k8s_config = {
    "execution_mode": "k8s",
    "pod_name": "openhands-worker-pod",
    "namespace": "openhands",
    "kubeconfig_path": "/path/to/kubeconfig",
    "working_dir": "/workspace"
}
```

确保已安装 `kodo` 库用于 K8s 集成：
```bash
pip install git+https://github.com/baidubce/kodo.git
```

## 注意事项

1. **文件路径**: 始终使用绝对路径
2. **字符串替换**: `old_str` 必须在文件中唯一匹配
3. **K8s 模式**: 需要正确配置 pod_name 和权限
4. **emoji 保留**: 日志语句中的 emoji 会被保留 [[memory:6314335]]
