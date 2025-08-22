#!/usr/bin/env python3
"""
Test GeneralAgent with R2E tools for SWE-bench-verified processing using subprocess for true concurrency.
This version uses subprocess to run each instance in a separate process, avoiding thread blocking issues.
"""

import asyncio
import sys
import os
import json
import time
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import tempfile
import pickle
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # python-dotenv not installed, skip

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")


def process_single_instance(instance_data_file: str, output_file: str, model_name: str = None, log_file: str = None, base_url: str = None) -> None:
    """
    Process a single instance in a subprocess.
    This function will be called by subprocess.run().
    
    Args:
        instance_data_file: Path to pickled instance data
        output_file: Path to save the patch result
        model_name: Model name to use (with index if load balancing)
        log_file: Path to save the subprocess logs
        base_url: LLM base URL to use (overrides environment variable)
    """
    import sys
    import os
    import json
    import asyncio
    import logging
    import pickle
    import signal
    import atexit
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from workers.agents.general_agent import GeneralAgent, dump_trajectory
    from workers.core import create_tool
    from workers.utils.llm_helper import call_llm
    from workers.core.trajectory import TrajectoryStep, StepType
    from workers.core.safe_profiler import SafeProfiler
    from workers.core.profiler_visualizer import ProfilerVisualizer
    
    # Import kodo for pod management
    try:
        from kodo import ContainerRunner
    except ImportError:
        print("ERROR: kodo package not found. Please install it with: pip install kodo")
        sys.exit(1)
    
    # Import R2E configurations
    from workers.tools.r2e_configs import (
        CUSTOM_TOOL_DESCRIPTIONS,
        parse_xml_action_custom,
        CustomDescriptionWrapper,
        generate_custom_system_prompt
    )
    
    # Configure logging
    handlers = []
    log_format = '%(asctime)s - [%(process)d] - %(levelname)s - %(name)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger to capture all logs
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    
    # Ensure R2E tools loggers propagate to root logger
    # Allow debug level to be controlled via environment variable
    r2e_log_level = os.environ.get('R2E_LOG_LEVEL', 'DEBUG')
    if r2e_log_level.upper() == 'DEBUG':
        logging.getLogger('workers.tools.r2e_tools').setLevel(logging.DEBUG)
    else:
        logging.getLogger('workers.tools.r2e_tools').setLevel(logging.INFO)
    
    # Also ensure kodo and other libraries log properly
    logging.getLogger('kodo').setLevel(logging.INFO)
    logging.getLogger('workers').setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")
        # Log active logger configuration
        logger.info(f"Logger configuration:")
        logger.info(f"  Root logger level: {logging.getLogger().level}")
        logger.info(f"  R2E tools logger level: {logging.getLogger('workers.tools.r2e_tools').level}")
        logger.info(f"  Workers logger level: {logging.getLogger('workers').level}")
        logger.info(f"  R2E_LOG_LEVEL env var: {r2e_log_level}")
        # Test that R2E tools logging is working
        test_logger = logging.getLogger('workers.tools.r2e_tools.test')
        test_logger.info("R2E tools logging test - this should appear in the log file")
    
    # Load instance data
    with open(instance_data_file, 'rb') as f:
        data = pickle.load(f)
    
    instance = data['instance']
    namespace = data['namespace']
    kubeconfig_path = data['kubeconfig_path']
    output_dir = data['output_dir']
    enable_profiling = data['enable_profiling']
    max_tokens = data.get('max_tokens', 16000)  # Get max_tokens from data
    provided_base_url = data.get('base_url', None)  # Get base_url from data
    
    instance_id = instance.get("instance_id", "unknown")
    issue = instance.get("problem_statement", "")
    image = instance.get("image", "")
    base_commit = instance.get("base_commit", None)  # Get base_commit from instance data
    
    # Use provided model name or default
    actual_model_name = model_name or MODEL_NAME
    # Use provided base_url or environment variable or default
    actual_base_url = base_url or provided_base_url or BASE_URL
    
    logger.info(f"Process {os.getpid()}: Processing instance {instance_id} with model {actual_model_name} and {actual_base_url}")
    
    # Global variables for cleanup
    global_kodo_runner = None
    global_pod = None
    global_pod_name = None
    
    def cleanup_pod():
        """Cleanup function to ensure pod is stopped."""
        global global_kodo_runner, global_pod, global_pod_name
        if global_pod and global_kodo_runner:
            try:
                logger.warning(f"Cleaning up pod {global_pod_name} due to process termination")
                global_kodo_runner.stop_container(global_pod)
                logger.info(f"Pod {global_pod_name} stopped during cleanup")
            except Exception as e:
                logger.error(f"Error stopping pod during cleanup: {e}")
    
    # Register cleanup handlers
    atexit.register(cleanup_pod)
    
    def signal_handler(signum, frame):
        """Handle termination signals."""
        logger.warning(f"Received signal {signum}, cleaning up...")
        cleanup_pod()
        sys.exit(1)
    
    # Register signal handlers for common termination signals
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    async def run_instance():
        """Async function to process the instance."""
        global global_kodo_runner, global_pod, global_pod_name
        
        # Initialize kodo
        kodo_runner = ContainerRunner(
            backend="kubernetes",
            namespace=namespace,
            kubeconfig_path=kubeconfig_path
        )
        global_kodo_runner = kodo_runner
        
        # Create pod name with unique identifier to avoid conflicts
        try:
            import uuid
            unique_suffix = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID
        except ImportError:
            # Fallback to timestamp + process ID if uuid not available
            import time
            unique_suffix = f"{int(time.time()) % 1000000:06d}-{os.getpid() % 100:02d}"
        
        # Keep pod name reasonable length (max 63 chars for K8s)
        instance_part = instance_id.lower().replace('/', '-').replace('_', '-')[:35]
        pod_name = f"swe-{instance_part}-{unique_suffix}"
        global_pod_name = pod_name
        pod = None
        
        try:
            # Start pod with retry logic
            logger.info(f"Starting pod: {pod_name} (unique suffix: {unique_suffix})")
            logger.info(f"Using image: {image}")
            
            max_retries = 3
            retry_delay = 5
            pod = None
            
            for attempt in range(max_retries):
                try:
                    pod = global_pod = kodo_runner.start_container(
                        image,
                        name=pod_name,
                        environment={
                            "PYTHONPATH": "/testbed",
                            "SWE_INSTANCE_ID": instance_id,
                            # UTF-8 encoding settings for LLM-generated code
                            # This prevents UnicodeEncodeError when LLM generates Unicode characters
                            # like checkmarks (✓), crosses (✗), or other special symbols
                            "PYTHONIOENCODING": "utf-8",
                            "LANG": "C.UTF-8",
                            "LC_ALL": "C.UTF-8",
                            # Proxy settings
                            "http_proxy": "http://agent.baidu.com:8891",
                            "https_proxy": "http://agent.baidu.com:8891",
                            "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
                            "PIP_TRUSTED_HOST": "pip.baidu.com"
                        }
                    )

                    logger.info(f"Pod {pod_name} started successfully on attempt {attempt + 1}")
                    break
                except Exception as pod_error:
                    error_msg = str(pod_error)
                    logger.error(f"Attempt {attempt + 1}/{max_retries} failed to start pod {pod_name}: {error_msg}")
                    
                    # Check for specific errors
                    if "Response ended prematurely" in error_msg or "ProtocolError" in error_msg:
                        logger.warning("Network/Protocol error detected, will retry...")
                    elif "already exists" in error_msg:
                        # This should be rare now with unique suffixes
                        logger.warning(f"Pod {pod_name} already exists (unexpected with UUID), attempting cleanup...")
                        try:
                            kodo_runner.stop_container(pod_name)
                            await asyncio.sleep(2)
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        logger.error(f"Failed to start pod after {max_retries} attempts")
                        logger.error(f"Image: {image}")
                        logger.error(f"Namespace: {namespace}")
                        raise
            
            # Wait for pod to be ready
            logger.info(f"Waiting for pod {pod_name} to be ready...")
            await asyncio.sleep(3)
            logger.info(f"Pod {pod_name} should be ready now")
            
            # Setup environment
            kodo_runner.execute_command(pod, f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            
            # Create agent with R2E tools
            # R2E assumes working directory is /testbed
            working_dir = "/testbed"
            k8s_config = {
                "execution_mode": "k8s",
                "pod_name": pod_name,
                "namespace": namespace,
                "kubeconfig_path": kubeconfig_path,
                "working_dir": working_dir  # Important: R2E tools need to know the working directory
            }
            
            # Create R2E tools with working directory
            base_tools = {
                "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
                "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
                "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
                "r2e_submit": create_tool("R2ESubmit", {})
            }
            
            # Create a wrapper class to log tool executions
            class LoggingToolWrapper:
                def __init__(self, tool, tool_name):
                    self.tool = tool
                    self.tool_name = tool_name
                    self.execution_count = 0
                
                async def execute_tool(self, instance_id, tool_args):
                    self.execution_count += 1
                    import time
                    start_time = time.time()
                    
                    # Log tool invocation
                    logger.info(f"\n{'='*60}")
                    logger.info(f"TOOL EXECUTION #{self.execution_count}: {self.tool_name}")
                    logger.info(f"{'='*60}")
                    logger.info(f"Instance: {instance_id}")
                    logger.info(f"Tool: {self.tool_name}")
                    logger.info(f"Arguments:")
                    for key, value in tool_args.items():
                        if isinstance(value, str) and len(value) > 1000:
                            logger.info(f"  {key}: [String with {len(value)} characters]")
                            logger.info(f"    Preview: {value[:200]}...")
                        else:
                            logger.info(f"  {key}: {value}")
                    
                    try:
                        # Execute the tool
                        result = await self.tool.execute_tool(instance_id, tool_args)
                        execution_time = time.time() - start_time
                        
                        # Log successful execution
                        logger.info(f"\nExecution Result:")
                        logger.info(f"  Status: SUCCESS")
                        logger.info(f"  Execution Time: {execution_time:.2f} seconds")
                        
                        if hasattr(result, 'result'):
                            result_str = str(result.result)
                            if len(result_str) > 2000:
                                logger.info(f"  Output: [Result with {len(result_str)} characters]")
                                logger.info(f"  Output Preview (first 500 chars):")
                                logger.info(f"    {result_str[:500]}...")
                                logger.info(f"  Output Preview (last 500 chars):")
                                logger.info(f"    ...{result_str[-500:]}")
                            else:
                                logger.info(f"  Output: {result_str}")
                        
                        logger.info(f"{'='*60}\n")
                        return result
                        
                    except Exception as e:
                        execution_time = time.time() - start_time
                        
                        # Log error
                        logger.error(f"\nExecution Result:")
                        logger.error(f"  Status: FAILED")
                        logger.error(f"  Execution Time: {execution_time:.2f} seconds")
                        logger.error(f"  Error Type: {type(e).__name__}")
                        logger.error(f"  Error Message: {str(e)}")
                        
                        import traceback
                        logger.error(f"  Traceback:")
                        for line in traceback.format_exc().split('\n'):
                            logger.error(f"    {line}")
                        
                        logger.error(f"{'='*60}\n")
                        raise
                
                def __getattr__(self, name):
                    return getattr(self.tool, name)
            
            # Wrap tools with logging and custom descriptions
            tools = {}
            for tool_name, tool in base_tools.items():
                # First wrap with logging
                logged_tool = LoggingToolWrapper(tool, tool_name)
                
                # Then wrap with custom description if available
                if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
                    tools[tool_name] = CustomDescriptionWrapper(logged_tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name])
                else:
                    tools[tool_name] = logged_tool
            
            # Generate custom system prompt
            custom_system_prompt = generate_custom_system_prompt(
                tools,
                task_description="analyze and fix the reported issue in the repository",
                working_directory="/testbed",
                additional_instructions="\n- Focus on the specific issue described\n- Make minimal changes to fix the issue\n- Ensure your changes don't break existing functionality"
            )
            
            # Create profiler if enabled
            profiler = None
            if enable_profiling:
                profiler = SafeProfiler(enabled=True)
                logger.info(f"Profiling enabled for instance {instance_id}")
            
            # Create agent
            agent = GeneralAgent(
                max_rounds=100,
                debug=True,
                termination_tool_names=["r2e_submit"],
                action_parser=parse_xml_action_custom,
                system_prompt=custom_system_prompt,
                profiler=profiler
            )
            agent.set_tools(tools)
            
            # Run agent
            logger.info("Running agent to solve the issue...")
            logger.info(f"LLM Configuration: model={actual_model_name}, base_url={actual_base_url}, max_tokens={max_tokens}")
            
            # 设置环境变量以供 llm_helper 使用
            os.environ['LLM_API_KEY'] = API_KEY
            os.environ['LLM_BASE_URL'] = actual_base_url
            os.environ['LLM_MODEL_NAME'] = actual_model_name
            
            # 保存代理设置（重要：这些值在函数创建时就固定了）
            saved_http_proxy = os.environ.get('http_proxy')
            saved_https_proxy = os.environ.get('https_proxy')
            if saved_http_proxy or saved_https_proxy:
                logger.info(f"Proxy configured: http={saved_http_proxy}, https={saved_https_proxy}")
            
            # 创建一个包装函数，将同步的 call_llm 转换为异步接口
            llm_call_count = 0
            
            async def llm_generate_func(messages, **kwargs):
                nonlocal llm_call_count
                llm_call_count += 1
                
                # 重要：对于内网 LLM 服务，需要清除代理设置
                # 因为内网服务不需要通过代理访问
                if 'http_proxy' in os.environ:
                    del os.environ['http_proxy']
                if 'https_proxy' in os.environ:
                    del os.environ['https_proxy']
                
                # 使用我们的默认 max_tokens
                if 'max_tokens' not in kwargs:
                    kwargs['max_tokens'] = max_tokens
                
                # Log LLM call
                logger.info(f"\n{'~'*60}")
                logger.info(f"LLM CALL #{llm_call_count}")
                logger.info(f"{'~'*60}")
                logger.info(f"Model: {actual_model_name}")
                logger.info(f"Max Tokens: {kwargs.get('max_tokens', max_tokens)}")
                logger.info(f"Temperature: {kwargs.get('temperature', 0.7)}")
                logger.info(f"Messages: {len(messages)} messages")
                
                # Log all messages for better debugging
                logger.info(f"\nConversation History:")
                for idx, msg in enumerate(messages):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    logger.info(f"\n  Message #{idx+1} - Role: {role}")
                    logger.info(f"  Content Length: {len(content)} characters")
                    
                    if role == 'system':
                        # System message - log first 500 chars
                        if len(content) > 500:
                            logger.info(f"  Content Preview: {content[:500]}...")
                        else:
                            logger.info(f"  Content: {content}")
                    elif role == 'user':
                        # User message - log first 1000 chars
                        if len(content) > 1000:
                            logger.info(f"  Content Preview: {content[:1000]}...")
                        else:
                            logger.info(f"  Content: {content}")
                    elif role == 'assistant':
                        # Assistant message - log first 2000 chars (more detail for debugging)
                        if len(content) > 2000:
                            logger.info(f"  Content Preview: {content[:2000]}...")
                            # Also check if there are tool calls
                            if '<function=' in content:
                                import re
                                tools_called = re.findall(r'<function=([^>]+)>', content)
                                if tools_called:
                                    logger.info(f"  Tools Called: {', '.join(tools_called)}")
                        else:
                            logger.info(f"  Content: {content}")
                
                import time
                start_time = time.time()
                
                try:
                    # 调试：检查调用前的代理状态
                    current_http_proxy = os.environ.get('http_proxy', 'NOT SET')
                    current_https_proxy = os.environ.get('https_proxy', 'NOT SET')
                    logger.debug(f"Before call_llm - http_proxy: {current_http_proxy}, https_proxy: {current_https_proxy}")
                    
                    # 重要：使用 asyncio.to_thread 在线程中运行同步函数
                    # 这样不会阻塞事件循环，避免与其他异步操作冲突
                    import asyncio
                    response = await asyncio.to_thread(
                        call_llm,
                        messages=messages,
                        model=actual_model_name,
                        temperature=kwargs.get('temperature', 0.7),
                        top_p=kwargs.get('top_p', 0.95),
                        max_tokens=kwargs.get('max_tokens', max_tokens),
                        timeout=60
                    )
                    execution_time = time.time() - start_time
                    
                    # Log response with full content
                    logger.info(f"\nLLM Response:")
                    logger.info(f"  Status: SUCCESS")
                    logger.info(f"  Response Time: {execution_time:.2f} seconds")
                    
                    if isinstance(response, str):
                        logger.info(f"  Response Length: {len(response)} characters")
                        logger.info(f"  Response Content:")
                        
                        # Log full response content with proper formatting
                        response_lines = response.split('\n')
                        for i, line in enumerate(response_lines):
                            if i < 100:  # Log first 100 lines in full
                                logger.info(f"    {line}")
                            elif i == 100:
                                logger.info(f"    ... ({len(response_lines) - 100} more lines omitted)")
                                break
                        
                        # Also log a summary of what the LLM is trying to do
                        if '<function=' in response:
                            # Extract function calls
                            import re
                            function_calls = re.findall(r'<function=([^>]+)>', response)
                            if function_calls:
                                logger.info(f"  LLM Actions: Calling tools: {', '.join(function_calls)}")
                        elif 'thought>' in response.lower() or 'reasoning>' in response.lower():
                            logger.info(f"  LLM Actions: Thinking/Reasoning")
                        elif 'answer>' in response.lower() or 'response>' in response.lower():
                            logger.info(f"  LLM Actions: Providing answer")
                    
                    logger.info(f"{'~'*60}\n")
                    return response
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    logger.error(f"\nLLM Response:")
                    logger.error(f"  Status: FAILED")
                    logger.error(f"  Response Time: {execution_time:.2f} seconds")
                    logger.error(f"  Error: {e}")
                    logger.error(f"{'~'*60}\n")
                    raise
            
            # 使用我们的包装函数
            
            prompt = f"""
Consider the following github issue:
<github_issue>            
{issue}
</github_issue>

Can you help me implement the necessary changes to the repository to fix the <github_issue>?
I've already taken care of all changes to any of the test files described in the <github_issue>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the /testbed directory to ensure the <github_issue> is satisfied.

IMPORTANT TIP:
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to explore the repo to familiarize yourself with its structure.
2. Create a script ('reproduce_issue.py') to reproduce the error and execute it to confirm the error
3. Edit the sourcecode of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
6. When viewing large files, use specific line-ranges, usually within 50 to 100 lines) as required
7. NOTE: The repository is at '/testbed' and the current working directory is already '/testbed', so DO NOT include 'testbed/' or 'testbed.' in relative paths in bash commands or reproduction python files.
8. Use pip source http://pip.baidu.com/pypi/simple if you need to install new package with python
"""
            
            # Track timing for statistics
            import time as timing
            
            try:
                logger.info(f"\n{'#'*80}")
                logger.info(f"# STARTING TRAJECTORY EXECUTION")
                logger.info(f"# Instance: {instance_id}")
                logger.info(f"# Model: {actual_model_name}")
                logger.info(f"# Max Tokens: {max_tokens}")
                logger.info(f"# Request ID: swe_{instance_id}")
                logger.info(f"{'#'*80}\n")
                
                # Track trajectory steps
                step_count = 0
                
                # Create a custom callback to log trajectory steps
                original_run = agent.run_trajectory
                async def logged_run_trajectory(*args, **kwargs):
                    nonlocal step_count
                    # Run the original trajectory
                    result = await original_run(*args, **kwargs)
                    
                    # Log trajectory summary
                    logger.info(f"\n{'#'*80}")
                    logger.info(f"# TRAJECTORY COMPLETED")
                    logger.info(f"# Total Steps: {len(result.steps)}")
                    logger.info(f"# Completed: {result.is_completed}")
                    logger.info(f"{'#'*80}\n")
                    
                    return result
                
                agent.run_trajectory = logged_run_trajectory
                
                result = await agent.run_trajectory(
                    prompt=prompt,
                    llm_generate_func=llm_generate_func,
                    request_id=f"swe_{instance_id}"
                )
            except Exception as traj_error:
                logger.error(f"\n{'!'*80}")
                logger.error(f"! TRAJECTORY EXECUTION FAILED")
                logger.error(f"! Error Type: {type(traj_error).__name__}")
                logger.error(f"! Error Message: {traj_error}")
                logger.error(f"{'!'*80}\n")
                
                # Check for specific error types
                error_msg = str(traj_error).lower()
                if 'response ended prematurely' in error_msg:
                    logger.error("This appears to be a network/API connection issue.")
                    logger.error("Possible causes:")
                    logger.error("  1. API endpoint timeout or connection reset")
                    logger.error("  2. Request size too large")
                    logger.error("  3. Network instability")
                    logger.error(f"  4. Model endpoint issue (model: {actual_model_name})")
                elif 'timeout' in error_msg:
                    logger.error("Request timed out. Consider increasing timeout or using a faster model.")
                elif 'rate limit' in error_msg:
                    logger.error("Rate limit exceeded. Consider reducing concurrency or adding delays.")
                raise
            
            logger.info(f"Agent completed: {result.is_completed}")
            logger.info(f"Total trajectory steps: {len(result.steps)}")
            
            # Check if agent reached max rounds without calling termination tool
            reached_max_rounds = len(result.steps) >= 100 * 2  # Rough estimate: 50 rounds * 2 steps per round
            called_submit = any(
                step.tool_name == "r2e_submit" 
                for step in result.steps 
                if hasattr(step, 'tool_name')
            )
            
            if not called_submit and reached_max_rounds:
                logger.warning(f"âš ï¸ Agent reached max rounds (100) without calling r2e_submit")
                logger.warning(f"âš ï¸ Patch will still be generated from current changes")
            elif called_submit:
                logger.info(f"âœ“ Agent called r2e_submit - normal termination")
            else:
                logger.info(f"Agent stopped after {len(result.steps)} steps")
            
            # Log trajectory steps summary
            logger.info(f"\n{'='*80}")
            logger.info(f"TRAJECTORY SUMMARY")
            logger.info(f"{'='*80}")
            logger.info(f"Instance: {instance_id}")
            logger.info(f"Completed: {result.is_completed}")
            logger.info(f"Total Steps: {len(result.steps)}")
            
            # Count step types
            step_types_count = {}
            for step in result.steps:
                step_type = step.step_type.value if hasattr(step.step_type, 'value') else str(step.step_type)
                step_types_count[step_type] = step_types_count.get(step_type, 0) + 1
            
            logger.info(f"Step Types:")
            for step_type, count in step_types_count.items():
                logger.info(f"  {step_type}: {count}")
            
            # Collect statistics from the trajectory
            stats = {
                'rounds': 0,
                'trajectory_length': len(result.steps),
                'tool_calls': [],
                'llm_calls': [],
                'is_completed': result.is_completed,
                'step_types': step_types_count,
                'called_submit': called_submit,
                'reached_max_rounds': reached_max_rounds,
                'termination_reason': 'submit_called' if called_submit else ('max_rounds' if reached_max_rounds else 'other')
            }
            
            # Count rounds (each action is a round)
            for step in result.steps:
                if step.step_type.value == 'action':
                    stats['rounds'] += 1
            
            # Extract tool call statistics with timing
            tool_start_times = {}
            for i, step in enumerate(result.steps):
                # Track tool execution times
                if step.step_type.value == 'action' and hasattr(step, 'tool_name') and step.tool_name:
                    tool_start_times[i] = timing.time()
                elif step.step_type.value == 'action_result' and hasattr(step, 'tool_name') and step.tool_name:
                    # Find corresponding action step
                    for j in range(i-1, -1, -1):
                        if j in tool_start_times:
                            execution_time = timing.time() - tool_start_times[j]
                            stats['tool_calls'].append({
                                'tool': step.tool_name,
                                'time': execution_time
                            })
                            del tool_start_times[j]
                            break
            
            # Count LLM calls (each thought/action generation is an LLM call)
            llm_steps = [s for s in result.steps if s.step_type.value in ['thought', 'action']]
            for step in llm_steps:
                # Estimate generation time based on content length (rough approximation)
                # In real scenario, this would be tracked in the agent
                estimated_time = len(step.content) / 1000.0  # Rough estimate: 1 second per 1000 chars
                stats['llm_calls'].append({'time': estimated_time})
            
            # Save trajectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_dir = os.path.join(output_dir, "trajectories")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Use original instance_id format for file naming (just replace / with __ for filesystem compatibility)
            file_safe_instance_id = instance_id.replace('/', '__')
            trajectory_file = os.path.join(trajectory_dir, f"{file_safe_instance_id}_{timestamp}.jsonl")
            dump_trajectory(result, trajectory_file, format="jsonl")
            logger.info(f"Saved trajectory to: {trajectory_file}")
            
            # Save profiler data if enabled
            profiler_file = None
            timeline_file = None
            if profiler and profiler.enabled:
                try:
                    profiler_dir = os.path.join(output_dir, "profiles")
                    os.makedirs(profiler_dir, exist_ok=True)
                    
                    # Save profiler data
                    profiler_file = os.path.join(profiler_dir, f"{file_safe_instance_id}_{timestamp}_profile.json")
                    profiler.export_events(profiler_file)
                    logger.info(f"Saved profiler data to: {profiler_file}")
                    
                    # Generate timeline visualization
                    timeline_file = os.path.join(profiler_dir, f"{file_safe_instance_id}_{timestamp}_timeline.html")
                    visualizer = ProfilerVisualizer({
                        "summary": profiler.get_summary(),
                        "events": [event.to_dict() for event in profiler.events]
                    })
                    visualizer.generate_html_timeline(timeline_file, title=f"Timeline: {instance_id}")
                    logger.info(f"Generated timeline visualization: {timeline_file}")
                except Exception as e:
                    logger.error(f"Failed to save profiler data: {e}")
            
            # Get patch
            logger.info(f"\n{'='*80}")
            logger.info(f"PATCH GENERATION")
            logger.info(f"{'='*80}")
            logger.info("Adding all changes to git...")
            kodo_runner.execute_command(pod, "cd /testbed && git add -A")
            
            # Generate patch based on base_commit if provided, otherwise use --cached
            if base_commit:
                logger.info(f"Generating patch against base_commit: {base_commit}")
                output, exit_code = kodo_runner.execute_command(
                    pod, 
                    f"cd /testbed && git diff {base_commit}"
                )
            else:
                logger.info("Generating patch against staged changes")
                output, exit_code = kodo_runner.execute_command(pod, "cd /testbed && git diff --cached")
            
            if int(exit_code) == 0:
                patch = output.strip()
                logger.info(f"Patch generated successfully")
                logger.info(f"Patch size: {len(patch)} characters")
                
                # Log patch preview
                if patch:
                    lines = patch.split('\n')
                    logger.info(f"Patch preview (first 10 lines):")
                    for line in lines[:10]:
                        logger.info(f"  {line}")
                    if len(lines) > 10:
                        logger.info(f"  ... ({len(lines) - 10} more lines)")
                
                # Save patch
                patches_dir = os.path.join(output_dir, "patches")
                os.makedirs(patches_dir, exist_ok=True)
                
                patch_filepath = os.path.join(patches_dir, f"{file_safe_instance_id}_{timestamp}.patch")
                with open(patch_filepath, 'w', encoding='utf-8') as f:
                    f.write(patch)
                logger.info(f"Saved patch to: {patch_filepath}")
                
                # Save result for parent process
                result_data = {
                    'success': True,
                    'instance_id': instance_id,
                    'patch': patch,
                    'patch_file': patch_filepath,
                    'trajectory_file': trajectory_file,
                    'stats': stats if 'stats' in locals() else {}
                }
                
                # Add profiler files if they exist
                if profiler_file:
                    result_data['profiler_file'] = profiler_file
                if timeline_file:
                    result_data['timeline_file'] = timeline_file
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f)
                
                return patch
            else:
                logger.warning(f"No patch generated for {instance_id}")
                result_data = {
                    'success': False,
                    'instance_id': instance_id,
                    'error': 'No patch generated',
                    'trajectory_file': trajectory_file,
                    'stats': stats if 'stats' in locals() else {}
                }
                
                # Add profiler files even for failed runs
                if profiler_file:
                    result_data['profiler_file'] = profiler_file
                if timeline_file:
                    result_data['timeline_file'] = timeline_file
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f)
                return None
                
        except Exception as e:
            # Get detailed error information
            import traceback
            import sys
            
            # Get the full traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            full_traceback = ''.join(tb_lines)
            
            # Log detailed error information
            logger.error(f"Error processing instance {instance_id}: {type(e).__name__}: {e}")
            logger.error(f"Error type: {exc_type}")
            logger.error(f"Error details: {exc_value}")
            logger.error(f"Full traceback:\n{full_traceback}")
            
            # Also print to console for immediate visibility
            print(f"\n{'='*80}")
            print(f"ERROR in instance {instance_id}")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Full traceback:")
            print(full_traceback)
            print(f"{'='*80}\n")
            
            # Prepare detailed error data
            error_details = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': full_traceback,
                'last_checkpoint': 'unknown'
            }
            
            # Try to identify where the error occurred
            if 'pod started successfully' in full_traceback:
                error_details['last_checkpoint'] = 'pod_startup'
            elif 'Running agent' in full_traceback:
                error_details['last_checkpoint'] = 'agent_execution'
            elif 'Generating patch' in full_traceback:
                error_details['last_checkpoint'] = 'patch_generation'
            
            result_data = {
                'success': False,
                'instance_id': instance_id,
                'error': str(e),
                'error_details': error_details,
                'stats': stats if 'stats' in locals() else {}
            }
            
            # Try to include any files that were created before the error
            if 'trajectory_file' in locals() and trajectory_file:
                result_data['trajectory_file'] = trajectory_file
            if 'profiler_file' in locals() and profiler_file:
                result_data['profiler_file'] = profiler_file
            if 'timeline_file' in locals() and timeline_file:
                result_data['timeline_file'] = timeline_file
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f)
            return None
            
        finally:
            # Clean up pod
            if pod and kodo_runner:
                try:
                    logger.info(f"Stopping pod: {pod_name}")
                    kodo_runner.stop_container(pod)
                    logger.info(f"Pod {pod_name} stopped")
                except Exception as e:
                    logger.error(f"Error stopping pod: {e}")
    
    # Run the async function
    asyncio.run(run_instance())


class SubprocessSWEBenchRunner:
    """Runner for SWE-bench instances using subprocess for true concurrency."""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: Optional[str] = None, 
                 output_dir: str = "./swe_patches", max_concurrent: int = 1, 
                 enable_profiling: bool = False, model_name: str = None,
                 model_index_range: Optional[Tuple[int, int]] = None,
                 timeout_seconds: int = 600,  # Default 10 minutes timeout
                 max_tokens: int = 16000,  # Maximum tokens for model output
                 base_url: str = None,  # LLM base URL
                 api_key: str = "",
                 local_mode: bool = False):  # Local mode for debugging
        """Initialize the runner.
        
        Args:
            timeout_seconds: Maximum time in seconds for each instance (default: 600 = 10 minutes)
            max_tokens: Maximum tokens for model output (default: 16000)
            base_url: LLM base URL (optional, overrides environment variable)
            local_mode: If True, run instances locally instead of in subprocess
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.enable_profiling = enable_profiling
        self.model_name = model_name or MODEL_NAME
        self.model_index_range = model_index_range
        self.timeout_seconds = timeout_seconds
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.api_key = api_key
        self.local_mode = local_mode
        self.patches = {}
    
    def _has_existing_trajectory(self, instance_id: str) -> bool:
        """Check if a trajectory file already exists for the given instance.
        
        Args:
            instance_id: The instance ID to check
            
        Returns:
            True if a trajectory file exists, False otherwise
        """
        trajectory_dir = os.path.join(self.output_dir, "trajectories")
        if not os.path.exists(trajectory_dir):
            return False
        
        # Replace / with __ for filesystem compatibility (same as in process_single_instance)
        file_safe_instance_id = instance_id.replace('/', '__')
        
        # Check if any trajectory file exists for this instance
        # Pattern: {instance_id}_{timestamp}.jsonl
        import glob
        pattern = os.path.join(trajectory_dir, f"{file_safe_instance_id}_*.jsonl")
        existing_files = glob.glob(pattern)
        
        if existing_files:
            # Return the most recent trajectory file for logging
            existing_files.sort()
            logger.info(f"Found existing trajectory for {instance_id}: {os.path.basename(existing_files[-1])}")
            return True
        
        return False
        
    async def process_instances(self, instances: List[Dict[str, Any]]):
        """Process multiple instances concurrently using subprocess or locally."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logs directory for subprocess logs
        if not self.local_mode:
            logs_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Subprocess logs will be saved to: {logs_dir}")
        
        temp_dir = tempfile.mkdtemp(prefix="swe_bench_")
        
        # Filter out instances with existing trajectories
        instances_to_process = []
        skipped_instances = []
        
        for instance in instances:
            instance_id = instance.get('instance_id', 'unknown')
            if self._has_existing_trajectory(instance_id):
                skipped_instances.append(instance_id)
                logger.info(f"Skipping {instance_id} - trajectory already exists")
            else:
                instances_to_process.append(instance)
        
        # Log summary of skipped instances
        if skipped_instances:
            logger.info(f"\n{'='*60}")
            logger.info(f"SKIPPED INSTANCES WITH EXISTING TRAJECTORIES")
            logger.info(f"{'='*60}")
            logger.info(f"Total skipped: {len(skipped_instances)}")
            logger.info(f"Instances to process: {len(instances_to_process)}")
            for instance_id in skipped_instances[:10]:  # Show first 10
                logger.info(f"  - {instance_id}")
            if len(skipped_instances) > 10:
                logger.info(f"  ... and {len(skipped_instances) - 10} more")
            logger.info(f"{'='*60}\n")
        
        # Update instances to only include those without existing trajectories
        instances = instances_to_process
        
        if not instances:
            logger.info("All instances have existing trajectories. Nothing to process.")
            return
        
        logger.info(f"Processing {len(instances)} instances with max concurrency: {self.max_concurrent}")
        if self.model_index_range:
            start_idx, end_idx = self.model_index_range
            model_count = end_idx - start_idx + 1
            logger.info(f"Using {model_count} models for load balancing: {self.model_name}-{start_idx} to {self.model_name}-{end_idx}")
        
        # Create subprocess tasks
        tasks = []
        for i, instance in enumerate(instances):
            # Determine model name for this task
            if self.model_index_range:
                start_idx, end_idx = self.model_index_range
                model_range = end_idx - start_idx + 1
                model_index = (i % model_range) + start_idx
                actual_model_name = f"{self.model_name}-{model_index}"
            else:
                actual_model_name = self.model_name
            
            # Prepare data for subprocess
            instance_data = {
                'instance': instance,
                'namespace': self.namespace,
                'kubeconfig_path': self.kubeconfig_path,
                'output_dir': self.output_dir,
                'enable_profiling': self.enable_profiling,
                'max_tokens': self.max_tokens,  # Add max_tokens to instance data
                'base_url': self.base_url,  # Add base_url to instance data
                'api_key': self.api_key
            }
            
            # Save instance data to temp file
            instance_data_file = os.path.join(temp_dir, f"instance_{i}.pkl")
            with open(instance_data_file, 'wb') as f:
                pickle.dump(instance_data, f)
            
            # Output file for results
            output_file = os.path.join(temp_dir, f"result_{i}.json")
            
            # Create async task for subprocess or local execution
            if self.local_mode:
                # Local mode: run directly without subprocess
                task = self._run_local_instance(i, len(instances), instance_data_file, output_file, actual_model_name, self.base_url)
            else:
                # Subprocess mode: run in separate process with logging
                instance_id = instance.get('instance_id', f'instance_{i}')
                # Use original instance_id format for log file naming (just replace / with __ for filesystem compatibility)
                file_safe_instance_id = instance_id.replace('/', '__')
                log_file = os.path.join(self.output_dir, "logs", f"{file_safe_instance_id}.log")
                task = self._run_subprocess(i, len(instances), instance_data_file, output_file, actual_model_name, log_file, self.base_url)
            tasks.append(task)
        
        # Run tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def run_with_semaphore(task_func):
            async with semaphore:
                return await task_func
        
        # Execute all tasks
        start_time = time.time()
        results = await asyncio.gather(*[run_with_semaphore(task) for task in tasks], return_exceptions=True)
        elapsed_time = time.time() - start_time
        
        # Process results and collect statistics
        successful = 0
        failed = 0
        timeout_count = 0
        failed_instances = []  # Track failed instance IDs
        timeout_instances = []  # Track timeout instance IDs
        successful_instances = []  # Track successful instance IDs
        profiler_files = []
        timeline_files = []
        
        # Global statistics
        global_stats = {
            'total_tool_calls': 0,
            'total_tool_time': 0.0,
            'tool_calls_by_type': {},
            'total_llm_calls': 0,
            'total_llm_time': 0.0,
            'rounds': {'min': float('inf'), 'max': 0, 'total': 0, 'count': 0},
            'trajectory_lengths': {'min': float('inf'), 'max': 0, 'total': 0, 'count': 0},
            'throughput': {'start_time': start_time, 'completed': 0},
            'termination_stats': {
                'submit_called': 0,
                'max_rounds_reached': 0,
                'other': 0
            }
        }
        
        # Process each result and update statistics
        for i, result in enumerate(results):
            instance_id = instances[i].get('instance_id', f'instance_{i}')
            
            # Update progress
            completed = successful + failed
            progress = (completed / len(instances)) * 100 if len(instances) > 0 else 0
            throughput = completed / elapsed_time if elapsed_time > 0 else 0
            
            if completed % 5 == 0 or completed == len(instances):  # Log every 5 completions
                logger.info(f"Progress: {completed}/{len(instances)} ({progress:.1f}%) | "
                          f"Throughput: {throughput:.2f} rollouts/sec")
            
            if isinstance(result, Exception):
                logger.error(f"Task {i} ({instance_id}) failed with exception: {result}")
                failed += 1
                failed_instances.append(instance_id)
            elif result:
                # Extract statistics if available
                if 'stats' in result and result['stats']:
                    stats = result['stats']
                    
                    # Update tool statistics
                    if 'tool_calls' in stats:
                        for tool_call in stats['tool_calls']:
                            global_stats['total_tool_calls'] += 1
                            global_stats['total_tool_time'] += tool_call.get('time', 0)
                            tool_name = tool_call.get('tool', 'unknown')
                            if tool_name not in global_stats['tool_calls_by_type']:
                                global_stats['tool_calls_by_type'][tool_name] = {'count': 0, 'time': 0}
                            global_stats['tool_calls_by_type'][tool_name]['count'] += 1
                            global_stats['tool_calls_by_type'][tool_name]['time'] += tool_call.get('time', 0)
                    
                    # Update LLM statistics
                    if 'llm_calls' in stats:
                        global_stats['total_llm_calls'] += len(stats['llm_calls'])
                        global_stats['total_llm_time'] += sum(call.get('time', 0) for call in stats['llm_calls'])
                    
                    # Update rounds statistics
                    if 'rounds' in stats:
                        rounds = stats['rounds']
                        global_stats['rounds']['min'] = min(global_stats['rounds']['min'], rounds)
                        global_stats['rounds']['max'] = max(global_stats['rounds']['max'], rounds)
                        global_stats['rounds']['total'] += rounds
                        global_stats['rounds']['count'] += 1
                    
                    # Update trajectory length statistics
                    if 'trajectory_length' in stats:
                        length = stats['trajectory_length']
                        global_stats['trajectory_lengths']['min'] = min(global_stats['trajectory_lengths']['min'], length)
                        global_stats['trajectory_lengths']['max'] = max(global_stats['trajectory_lengths']['max'], length)
                        global_stats['trajectory_lengths']['total'] += length
                        global_stats['trajectory_lengths']['count'] += 1
                    
                    # Update termination statistics
                    if 'termination_reason' in stats:
                        reason = stats['termination_reason']
                        if reason in global_stats['termination_stats']:
                            global_stats['termination_stats'][reason] += 1
                
                if result.get('success'):
                    self.patches[result['instance_id']] = result.get('patch', '')
                    successful += 1
                    successful_instances.append(result['instance_id'])
                    global_stats['throughput']['completed'] += 1
                    
                    # Collect profiler files
                    if result.get('profiler_file'):
                        profiler_files.append(result['profiler_file'])
                    if result.get('timeline_file'):
                        timeline_files.append(result['timeline_file'])
                else:
                    failed += 1
                    if result.get('timeout'):
                        timeout_count += 1
                        timeout_instances.append(result.get('instance_id', instance_id))
                    else:
                        failed_instances.append(result.get('instance_id', instance_id))
            else:
                failed += 1
                failed_instances.append(instance_id)
        
        # Calculate final statistics
        avg_tool_time = global_stats['total_tool_time'] / global_stats['total_tool_calls'] if global_stats['total_tool_calls'] > 0 else 0
        avg_llm_time = global_stats['total_llm_time'] / global_stats['total_llm_calls'] if global_stats['total_llm_calls'] > 0 else 0
        avg_rounds = global_stats['rounds']['total'] / global_stats['rounds']['count'] if global_stats['rounds']['count'] > 0 else 0
        avg_trajectory_length = global_stats['trajectory_lengths']['total'] / global_stats['trajectory_lengths']['count'] if global_stats['trajectory_lengths']['count'] > 0 else 0
        overall_throughput = len(instances) / elapsed_time if elapsed_time > 0 else 0
        
        # Print comprehensive summary
        logger.info(f"\n{'='*80}")
        logger.info(f"PROCESSING SUMMARY")
        logger.info(f"{'='*80}")
        
        # Basic metrics
        logger.info(f"Basic Metrics:")
        logger.info(f"  Total instances: {len(instances)}")
        if skipped_instances:
            logger.info(f"  Skipped (existing): {len(skipped_instances)}")
        logger.info(f"  Successful: {successful} ({successful/len(instances)*100:.1f}%)")
        logger.info(f"  Failed: {failed} ({failed/len(instances)*100:.1f}%)")
        logger.info(f"  Timeout: {timeout_count} ({timeout_count/len(instances)*100:.1f}%)")
        logger.info(f"  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"  Average time per instance: {elapsed_time/len(instances):.2f} seconds")
        
        # Throughput metrics
        logger.info(f"Throughput:")
        logger.info(f"  Overall: {overall_throughput:.3f} rollouts/sec")
        logger.info(f"  Completed rollouts: {global_stats['throughput']['completed']}")
        if self.max_concurrent > 1:
            logger.info(f"  Concurrency: {self.max_concurrent}x parallel")
        
        # Tool usage statistics
        logger.info(f"Tool Usage:")
        logger.info(f"  Total tool calls: {global_stats['total_tool_calls']}")
        if global_stats['total_tool_calls'] > 0:
            logger.info(f"  Average tool execution time: {avg_tool_time:.3f} seconds")
            logger.info(f"  Tool calls by type:")
            for tool_name, tool_stats in global_stats['tool_calls_by_type'].items():
                avg_time = tool_stats['time'] / tool_stats['count'] if tool_stats['count'] > 0 else 0
                logger.info(f"    - {tool_name}: {tool_stats['count']} calls, avg {avg_time:.3f}s")
        
        # LLM usage statistics
        logger.info(f"LLM Usage:")
        logger.info(f"  Total LLM calls: {global_stats['total_llm_calls']}")
        if global_stats['total_llm_calls'] > 0:
            logger.info(f"  Average LLM call time: {avg_llm_time:.3f} seconds")
            logger.info(f"  Average LLM calls per instance: {global_stats['total_llm_calls']/len(instances):.1f}")
        
        # Trajectory statistics
        logger.info(f"Trajectory Statistics:")
        if global_stats['rounds']['count'] > 0:
            logger.info(f"  Rounds: min={global_stats['rounds']['min']}, "
                      f"max={global_stats['rounds']['max']}, avg={avg_rounds:.1f}")
        if global_stats['trajectory_lengths']['count'] > 0:
            logger.info(f"  Trajectory length: min={global_stats['trajectory_lengths']['min']}, "
                      f"max={global_stats['trajectory_lengths']['max']}, avg={avg_trajectory_length:.1f}")
        
        # Termination reasons
        logger.info(f"Termination Reasons:")
        term_stats = global_stats['termination_stats']
        total_terminations = sum(term_stats.values())
        if total_terminations > 0:
            logger.info(f"  Normal (r2e_submit called): {term_stats['submit_called']} ({term_stats['submit_called']/total_terminations*100:.1f}%)")
            logger.info(f"  Max rounds reached: {term_stats['max_rounds_reached']} ({term_stats['max_rounds_reached']/total_terminations*100:.1f}%)")
            logger.info(f"  Other: {term_stats['other']} ({term_stats['other']/total_terminations*100:.1f}%)")
            
            if term_stats['max_rounds_reached'] > 0:
                logger.warning(f"  âš ï¸ {term_stats['max_rounds_reached']} instances reached max rounds without calling r2e_submit")
                logger.warning(f"  âš ï¸ These patches were generated from partial work")
        
        logger.info(f"{'='*80}")
        
        # Save comprehensive summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        summary = {
            "total_instances": len(instances),
            "skipped_instances": len(skipped_instances) if 'skipped_instances' in locals() else 0,
            "successful_patches": successful,
            "failed_instances": failed,
            "timeout_instances": timeout_count,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
            "local_mode": self.local_mode,
            "successful_instance_ids": successful_instances,
            "failed_instance_ids": failed_instances,
            "timeout_instance_ids": timeout_instances,
            "skipped_instance_ids": skipped_instances if 'skipped_instances' in locals() else [],
            "timestamp": datetime.now().isoformat(),
            "elapsed_time": elapsed_time,
            "max_concurrent": self.max_concurrent,
            "model_name": self.model_name,
            "model_index_range": self.model_index_range,
            "profiling_enabled": self.enable_profiling,
            "statistics": {
                "throughput": {
                    "overall_rollouts_per_sec": overall_throughput,
                    "avg_time_per_instance": elapsed_time/len(instances) if len(instances) > 0 else 0
                },
                "tool_usage": {
                    "total_calls": global_stats['total_tool_calls'],
                    "total_time": global_stats['total_tool_time'],
                    "avg_time": avg_tool_time,
                    "by_type": global_stats['tool_calls_by_type']
                },
                "llm_usage": {
                    "total_calls": global_stats['total_llm_calls'],
                    "total_time": global_stats['total_llm_time'],
                    "avg_time": avg_llm_time,
                    "avg_calls_per_instance": global_stats['total_llm_calls']/len(instances) if len(instances) > 0 else 0
                },
                "trajectory": {
                    "rounds": {
                        "min": global_stats['rounds']['min'] if global_stats['rounds']['count'] > 0 else None,
                        "max": global_stats['rounds']['max'] if global_stats['rounds']['count'] > 0 else None,
                        "avg": avg_rounds
                    },
                    "length": {
                        "min": global_stats['trajectory_lengths']['min'] if global_stats['trajectory_lengths']['count'] > 0 else None,
                        "max": global_stats['trajectory_lengths']['max'] if global_stats['trajectory_lengths']['count'] > 0 else None,
                        "avg": avg_trajectory_length
                    },
                    "termination_reasons": {
                        "submit_called": global_stats['termination_stats']['submit_called'],
                        "max_rounds_reached": global_stats['termination_stats']['max_rounds_reached'],
                        "other": global_stats['termination_stats']['other']
                    }
                }
            }
        }
        
        # Add profiling info if enabled
        if self.enable_profiling:
            summary["profiler_files"] = profiler_files
            summary["timeline_files"] = timeline_files
            summary["profiles_dir"] = os.path.join(self.output_dir, "profiles")
            logger.info(f"Generated {len(profiler_files)} profiler files and {len(timeline_files)} timeline visualizations")
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
    async def _run_subprocess(self, index: int, total: int, instance_data_file: str, 
                             output_file: str, model_name: str, log_file: str = None, base_url: str = None) -> Optional[Dict]:
        """Run a single instance in a subprocess with timeout."""
        # Load instance data to get instance_id
        with open(instance_data_file, 'rb') as f:
            data = pickle.load(f)
        instance_id = data['instance'].get('instance_id', f'instance_{index}')
        
        logger.info(f"[{index+1}/{total}] Starting subprocess for {instance_id} with model {model_name} (timeout: {self.timeout_seconds}s)")
        if log_file:
            logger.info(f"  Log file: {log_file}")
        
        # Prepare subprocess command
        cmd = [
            sys.executable,
            "-c",
            f"""
import sys
sys.path.insert(0, '{os.path.dirname(os.path.dirname(__file__))}')
from tests.test_r2e_general_agent_on_swe_subprocess import process_single_instance
process_single_instance('{instance_data_file}', '{output_file}', '{model_name}', '{log_file if log_file else ""}', '{base_url if base_url else ""}')
"""
        ]
        
        # Run subprocess
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            # Wait for completion with timeout
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout_seconds
            )
            
            if process.returncode != 0:
                logger.error(f"[{index+1}/{total}] Subprocess for {instance_id} failed with return code {process.returncode}")
                if stderr:
                    stderr_text = stderr.decode()
                    # Show more of stderr for debugging
                    logger.error(f"Stderr output (first 2000 chars):")
                    logger.error(stderr_text[:2000])
                    
                    # Try to extract specific error information
                    if 'Response ended prematurely' in stderr_text:
                        logger.error("Network/API error detected - Response ended prematurely")
                    if 'traceback' in stderr_text.lower():
                        # Find and show the actual error
                        lines = stderr_text.split('\n')
                        for i, line in enumerate(lines):
                            if 'Traceback' in line:
                                # Show the traceback and following lines
                                traceback_end = min(i + 20, len(lines))
                                logger.error("Python traceback found:")
                                for j in range(i, traceback_end):
                                    logger.error(f"  {lines[j]}")
                                break
                
                if stdout:
                    stdout_text = stdout.decode()
                    if stdout_text.strip():
                        logger.info(f"Stdout output (first 1000 chars):")
                        logger.info(stdout_text[:1000])
                
                return {
                    'success': False,
                    'instance_id': instance_id,
                    'error': f'Process failed with return code {process.returncode}',
                    'stderr': stderr.decode()[:2000] if stderr else None
                }
            
        except asyncio.TimeoutError:
            logger.error(f"[{index+1}/{total}] TIMEOUT: Instance {instance_id} exceeded {self.timeout_seconds}s limit")
            
            # Kill the subprocess
            try:
                process.kill()
                await process.wait()  # Wait for process to actually terminate
            except Exception as e:
                logger.error(f"Error killing subprocess for {instance_id}: {e}")
            
            # Return timeout result
            return {
                'success': False,
                'instance_id': instance_id,
                'error': f'Timeout after {self.timeout_seconds} seconds',
                'timeout': True
            }
        
        # Read result
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                result = json.load(f)
            logger.info(f"[{index+1}/{total}] Subprocess completed for {instance_id}: {result.get('success', False)}")
            return result
        else:
            logger.error(f"[{index+1}/{total}] No output file generated for {instance_id}")
            return {
                'success': False,
                'instance_id': instance_id,
                'error': 'No output file generated'
            }
    
    async def _run_local_instance(self, index: int, total: int, instance_data_file: str,
                                 output_file: str, model_name: str, base_url: str = None) -> Optional[Dict]:
        """Run a single instance locally (without subprocess) for debugging."""
        # Load instance data to get instance_id
        with open(instance_data_file, 'rb') as f:
            data = pickle.load(f)
        instance_id = data['instance'].get('instance_id', f'instance_{index}')
        
        logger.info(f"[{index+1}/{total}] Starting local execution for {instance_id} with model {model_name}")
        
        try:
            # Run the instance processing directly
            process_single_instance(instance_data_file, output_file, model_name, None, base_url)
            
            # Read result
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result = json.load(f)
                logger.info(f"[{index+1}/{total}] Local execution completed for {instance_id}: {result.get('success', False)}")
                return result
            else:
                logger.error(f"[{index+1}/{total}] No output file generated for {instance_id}")
                return {
                    'success': False,
                    'instance_id': instance_id,
                    'error': 'No output file generated'
                }
        except Exception as e:
            logger.error(f"[{index+1}/{total}] Local execution failed for {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'instance_id': instance_id,
                'error': str(e)
            }


async def main():
    """Main function to run SWE-bench processing with subprocess."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run R2E agent on SWE-bench instances using subprocess")
    parser.add_argument("jsonl_file", help="Path to JSONL file with SWE-bench instances")
    parser.add_argument("--output-dir", default="./swe_patches", help="Directory to save patches")
    parser.add_argument("--namespace", default=os.getenv("K8S_NAMESPACE", "default"),
                       help="Kubernetes namespace")
    parser.add_argument("--kubeconfig", default=os.getenv("KUBECONFIG", None),
                       help="Path to kubeconfig file")
    parser.add_argument("--max-concurrent", type=int, default=1,
                       help="Maximum number of concurrent subprocesses")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable performance profiling")
    parser.add_argument("--model-name", type=str, default=None,
                       help="Base model name for load balancing")
    parser.add_argument("--model-index-range", type=str, default=None,
                       help="Model index range for load balancing, format: 'start,end'")
    parser.add_argument("--timeout", type=int, default=600,
                       help="Timeout in seconds for each instance (default: 600 = 10 minutes)")
    parser.add_argument("--max-tokens", type=int, default=16000,
                       help="Maximum tokens for model output (default: 16000)")
    parser.add_argument("--base-url", type=str, default=None,
                       help="LLM base URL (overrides environment variable)")
    parser.add_argument("--api-key", type=str, default=None,
                       help="LLM API key (overrides environment variable)")
    parser.add_argument("--local-mode", action="store_true",
                       help="Run in local mode without subprocess for debugging")
    
    args = parser.parse_args()
    
    # Parse model index range
    model_index_range = None
    if args.model_index_range:
        try:
            parts = args.model_index_range.split(',')
            if len(parts) == 2:
                model_index_range = (int(parts[0]), int(parts[1]))
                logger.info(f"Model index range: {model_index_range[0]} to {model_index_range[1]}")
        except ValueError as e:
            logger.error(f"Invalid model index range: {e}")
            sys.exit(1)
    
    # Load instances
    instances = []
    with open(args.jsonl_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                instances.append(json.loads(line))
    
    logger.info(f"Loaded {len(instances)} instances from {args.jsonl_file}")
    
    # Create runner
    runner = SubprocessSWEBenchRunner(
        namespace=args.namespace,
        kubeconfig_path=args.kubeconfig,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        enable_profiling=args.enable_profiling,
        model_name=args.model_name,
        model_index_range=model_index_range,
        timeout_seconds=args.timeout,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        api_key=args.api_key,
        local_mode=args.local_mode
    )
    
    # Process instances
    await runner.process_instances(instances)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python test_r2e_general_agent_on_swe_subprocess.py <jsonl_file> [options]")
        print("\nExample:")
        print("  python test_r2e_general_agent_on_swe_subprocess.py swe_bench.jsonl --max-concurrent 20")
        print("\nWith model load balancing:")
        print("  python test_r2e_general_agent_on_swe_subprocess.py swe_bench.jsonl \\")
        print("    --max-concurrent 20 --model-name swe-8676-0807 --model-index-range 0,19")
        print("\nWith custom base URL:")
        print("  python test_r2e_general_agent_on_swe_subprocess.py swe_bench.jsonl \\")
        print("    --base-url http://custom-llm-endpoint:8080/v1 --max-concurrent 10")
        sys.exit(1)
    
    asyncio.run(main())
