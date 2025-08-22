#!/usr/bin/env python3
"""
Utility module for building dynamic system prompts.
Provides a flexible PromptBuilder class for creating context-aware prompts.
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import json


class PromptBuilder:
    """Builder class for creating dynamic system prompts with variable substitution."""
    
    def __init__(self, template: Optional[str] = None):
        """
        Initialize PromptBuilder.
        
        Args:
            template: Base template string with {variable} placeholders
        """
        self.template = template
        self.sections = []
        self.variables = {}
        self.tool_formatter = self._default_tool_formatter
    
    def set_template(self, template: str) -> "PromptBuilder":
        """Set the base template."""
        self.template = template
        return self
    
    def add_variable(self, name: str, value: Any) -> "PromptBuilder":
        """Add a variable for substitution."""
        self.variables[name] = value
        return self
    
    def add_variables(self, **kwargs) -> "PromptBuilder":
        """Add multiple variables at once."""
        self.variables.update(kwargs)
        return self
    
    def add_section(self, title: str, content: str, condition: bool = True) -> "PromptBuilder":
        """Add a conditional section to the prompt."""
        if condition:
            self.sections.append(f"\n{title}:\n{content}")
        return self
    
    def add_tools(self, tools: Dict[str, Any], formatter: Optional[Callable] = None) -> "PromptBuilder":
        """
        Add tool descriptions to the prompt.
        
        Args:
            tools: Dictionary of tool instances
            formatter: Optional custom formatter function
        """
        if formatter:
            self.tool_formatter = formatter
        
        tool_section = self._format_tools(tools)
        self.add_section("Available Tools", tool_section)
        return self
    
    def _default_tool_formatter(self, tool_name: str, tool: Any) -> str:
        """Default tool formatter."""
        if hasattr(tool, 'get_description'):
            return f"## {tool_name}\n{tool.get_description()}"
        else:
            return f"## {tool_name}\nNo description available"
    
    def _format_tools(self, tools: Dict[str, Any]) -> str:
        """Format tools section."""
        tool_descriptions = []
        for name, tool in tools.items():
            tool_descriptions.append(self.tool_formatter(name, tool))
        return "\n\n".join(tool_descriptions)
    
    def add_examples(self, examples: List[Dict[str, str]]) -> "PromptBuilder":
        """Add examples section."""
        if not examples:
            return self
        
        example_text = []
        for i, example in enumerate(examples, 1):
            example_text.append(f"Example {i}:")
            if "input" in example:
                example_text.append(f"Input: {example['input']}")
            if "output" in example:
                example_text.append(f"Output: {example['output']}")
            if "explanation" in example:
                example_text.append(f"Explanation: {example['explanation']}")
            example_text.append("")
        
        self.add_section("Examples", "\n".join(example_text))
        return self
    
    def add_context(self, context: Dict[str, Any]) -> "PromptBuilder":
        """Add context information."""
        context_lines = []
        for key, value in context.items():
            context_lines.append(f"- {key}: {value}")
        
        self.add_section("Context", "\n".join(context_lines))
        return self
    
    def add_timestamp(self) -> "PromptBuilder":
        """Add current timestamp."""
        self.add_variable("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        return self
    
    def build(self) -> str:
        """Build the final prompt."""
        if self.template:
            # Use template with variable substitution
            prompt = self.template.format(**self.variables)
            
            # Append sections if any
            if self.sections:
                prompt += "\n" + "\n".join(self.sections)
        else:
            # Build from sections only
            if not self.sections:
                raise ValueError("No template or sections provided")
            prompt = "\n".join(self.sections).strip()
        
        return prompt


class PromptLibrary:
    """Collection of pre-built prompt templates."""
    
    @staticmethod
    def react_agent_prompt(tools: Dict[str, Any], **kwargs) -> str:
        """Standard ReAct agent prompt."""
        builder = PromptBuilder()
        
        task_description = kwargs.get('task_description', 'complete the given task')
        json_format = kwargs.get('json_format', True)
        
        builder.add_section(
            "Role",
            f"You are an AI assistant that uses the ReAct framework to {task_description}."
        )
        
        builder.add_tools(tools)
        
        if json_format:
            builder.add_section(
                "Format",
                """Your response must follow this format:

Thought: [Your reasoning about what to do next]

Action:
{
  "name": "tool_name",
  "parameters": {
    "param1": "value1"
  }
}"""
            )
        
        builder.add_section(
            "Instructions",
            """1. Analyze the task and plan your approach
2. Use tools to gather information and take actions
3. Review results and adjust your approach
4. Continue until the task is complete"""
        )
        
        return builder.build()
    
    @staticmethod
    def code_assistant_prompt(tools: Dict[str, Any], **kwargs) -> str:
        """Code assistant prompt for development tasks."""
        builder = PromptBuilder()
        
        language = kwargs.get('language', 'Python')
        project_type = kwargs.get('project_type', 'general')
        working_dir = kwargs.get('working_dir', '/workspace')
        
        builder.add_context({
            "Language": language,
            "Project Type": project_type,
            "Working Directory": working_dir,
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        builder.add_section(
            "Role",
            f"You are an expert {language} developer working on a {project_type} project."
        )
        
        builder.add_tools(tools)
        
        builder.add_section(
            "Best Practices",
            f"""- Write clean, maintainable {language} code
- Follow established coding conventions
- Include appropriate error handling
- Add comments for complex logic
- Consider performance implications"""
        )
        
        return builder.build()
    
    @staticmethod
    def swe_bench_prompt(tools: Dict[str, Any], issue: str, repo: str, **kwargs) -> str:
        """SWE-bench style prompt for issue resolution."""
        builder = PromptBuilder(
            template="""You are an expert software engineer working on the {repo} repository.

Issue to resolve:
{issue}

Your task is to:
1. Understand the issue
2. Locate relevant code
3. Implement a fix
4. Verify the solution

Current working directory: {working_dir}"""
        )
        
        builder.add_variables(
            repo=repo,
            issue=issue,
            working_dir=kwargs.get('working_dir', '/testbed')
        )
        
        builder.add_tools(tools)
        
        test_cmd = kwargs.get('test_command')
        if test_cmd:
            builder.add_section(
                "Testing",
                f"Run tests using: {test_cmd}"
            )
        
        return builder.build()


# Convenience functions
def build_prompt(template: str, tools: Dict[str, Any], **variables) -> str:
    """Quick function to build a prompt from template."""
    builder = PromptBuilder(template)
    builder.add_variables(**variables)
    builder.add_tools(tools)
    return builder.build()


def build_react_prompt(tools: Dict[str, Any], **kwargs) -> str:
    """Build a ReAct agent prompt."""
    return PromptLibrary.react_agent_prompt(tools, **kwargs)


def build_code_prompt(tools: Dict[str, Any], **kwargs) -> str:
    """Build a code assistant prompt."""
    return PromptLibrary.code_assistant_prompt(tools, **kwargs)


def build_swe_prompt(tools: Dict[str, Any], issue: str, repo: str, **kwargs) -> str:
    """Build a SWE-bench prompt."""
    return PromptLibrary.swe_bench_prompt(tools, issue, repo, **kwargs)