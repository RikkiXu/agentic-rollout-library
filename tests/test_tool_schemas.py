#!/usr/bin/env python3
"""
Test tool schemas to verify they're properly exposed
"""

import sys
import os
import json

# Add the parent directory to the path so we can import workers module
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.core import create_tool

def test_tool_schemas():
    """Test that tools properly expose their schemas"""
    print("🔧 Testing Tool Schemas")
    print("=" * 80)
    
    # K8S configuration
    k8s_config = {
        "execution_mode": "k8s",
        "pod_name": "swebench-xarray-pod",
        "namespace": "default",
        "timeout": 30
    }
    
    # Create tools
    tools = {
        "bash_executor": create_tool("BashExecutor", k8s_config.copy()),
        "file_editor": create_tool("FileEditor", k8s_config.copy()),
        "search": create_tool("Search", k8s_config.copy()),
        "finish": create_tool("Finish")
    }
    
    # Test each tool's schema
    for tool_name, tool in tools.items():
        print(f"\n📋 Tool: {tool_name}")
        print("-" * 40)
        
        if hasattr(tool, 'get_openai_tool_schema'):
            schema = tool.get_openai_tool_schema()
            # Convert Pydantic model to dict
            schema_dict = schema.model_dump() if hasattr(schema, 'model_dump') else schema.dict()
            print(json.dumps(schema_dict, indent=2))
            
            # Verify schema structure
            if 'function' in schema_dict:
                func = schema_dict['function']
                print(f"\n✅ Schema has 'function' field")
                print(f"   Name: {func.get('name')}")
                print(f"   Description: {func.get('description', '')[:100]}...")
                
                if 'parameters' in func:
                    params = func['parameters']
                    if 'properties' in params:
                        print(f"   Parameters:")
                        for param_name, param_def in params['properties'].items():
                            print(f"     - {param_name}: {param_def.get('type')} - {param_def.get('description', '')[:50]}...")
                    
                    if 'required' in params:
                        print(f"   Required: {params['required']}")
            else:
                print("❌ Schema missing 'function' field")
        else:
            print("❌ Tool doesn't have get_openai_tool_schema method")


if __name__ == "__main__":
    test_tool_schemas()