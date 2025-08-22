#!/usr/bin/env python3
"""
Core tools for the agentic rollout library.
"""

from .calculator_tool import CalculatorTool
from .bash_executor_tool import BashExecutorTool
from .file_editor_tool import FileEditorTool
from .search_tool import SearchTool
from .finish_tool import FinishTool

# K8s tools (require kodo dependency)
try:
    from .k8s_bash_executor_tool import K8sBashExecutorTool
    from .k8s_file_editor_tool import K8sFileEditorTool
    from .k8s_search_tool import K8sSearchTool
    K8S_TOOLS_AVAILABLE = True
except ImportError:
    K8sBashExecutorTool = None
    K8sFileEditorTool = None
    K8sSearchTool = None
    K8S_TOOLS_AVAILABLE = False

# R2E tools (support both local and K8s execution)
try:
    from .r2e_tools import (
        R2EBashExecutorTool,
        R2ESearchTool,
        R2EFileEditorTool,
        R2EStrReplaceEditorTool,
        R2ESubmitTool
    )
    R2E_TOOLS_AVAILABLE = True
except ImportError as e:
    R2EBashExecutorTool = None
    R2ESearchTool = None
    R2EFileEditorTool = None
    R2EStrReplaceEditorTool = None
    R2ESubmitTool = None
    R2E_TOOLS_AVAILABLE = False

# OpenHands tools (support both local and K8s execution)
try:
    from .openhands_tools import (
        OpenHandsBashTool,
        OpenHandsStrReplaceEditorTool,
        OpenHandsFinishTool
    )
    OPENHANDS_TOOLS_AVAILABLE = True
except ImportError as e:
    OpenHandsBashTool = None
    OpenHandsStrReplaceEditorTool = None
    OpenHandsFinishTool = None
    OPENHANDS_TOOLS_AVAILABLE = False

# Import R2E configurations separately to avoid circular imports
try:
    from .r2e_configs import (
        CUSTOM_TOOL_DESCRIPTIONS,
        parse_xml_action_custom,
        CustomDescriptionWrapper,
        generate_custom_system_prompt
    )
    R2E_CONFIGS_AVAILABLE = True
except ImportError:
    CUSTOM_TOOL_DESCRIPTIONS = None
    parse_xml_action_custom = None
    CustomDescriptionWrapper = None
    generate_custom_system_prompt = None
    R2E_CONFIGS_AVAILABLE = False

__all__ = [
    'CalculatorTool',
    'BashExecutorTool', 
    'FileEditorTool',
    'SearchTool',
    'FinishTool'
]

# Add K8s tools to __all__ if available
if K8S_TOOLS_AVAILABLE:
    __all__.extend([
        'K8sBashExecutorTool',
        'K8sFileEditorTool', 
        'K8sSearchTool'
    ])

# Add R2E tools to __all__ if available
if R2E_TOOLS_AVAILABLE:
    __all__.extend([
        'R2EBashExecutorTool',
        'R2ESearchTool',
        'R2EFileEditorTool',
        'R2EStrReplaceEditorTool',
        'R2ESubmitTool'
    ])

# Add R2E configs to __all__ if available
if R2E_CONFIGS_AVAILABLE:
    __all__.extend([
        'CUSTOM_TOOL_DESCRIPTIONS',
        'parse_xml_action_custom',
        'CustomDescriptionWrapper',
        'generate_custom_system_prompt'
    ])

# Add OpenHands tools to __all__ if available
if OPENHANDS_TOOLS_AVAILABLE:
    __all__.extend([
        'OpenHandsBashTool',
        'OpenHandsStrReplaceEditorTool',
        'OpenHandsFinishTool'
    ])