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


def process_single_instance(instance_data_file: str, output_file: str, model_name: str = None, log_file: str = None) -> None:
    """
    Process a single instance in a subprocess.
    This function will be called by subprocess.run().
    
    Args:
        instance_data_file: Path to pickled instance data
        output_file: Path to save the patch result
        model_name: Model name to use (with index if load balancing)
        log_file: Path to save the subprocess logs
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
    from workers.utils import create_llm_client
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
    log_format = '%(asctime)s - [%(process)d] - %(levelname)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logger = logging.getLogger(__name__)
    
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    # Load instance data
    with open(instance_data_file, 'rb') as f:
        data = pickle.load(f)
    
    instance = data['instance']
    namespace = data['namespace']
    kubeconfig_path = data['kubeconfig_path']
    output_dir = data['output_dir']
    enable_profiling = data['enable_profiling']
    max_tokens = data.get('max_tokens', 16000)  # Get max_tokens from data
    
    instance_id = instance.get("instance_id", "unknown")
    issue = instance.get("problem_statement", "")
    image = instance.get("image", "")
    base_commit = instance.get("base_commit", None)  # Get base_commit from instance data
    
    # Use provided model name or default
    actual_model_name = model_name or MODEL_NAME
    
    logger.info(f"Process {os.getpid()}: Processing instance {instance_id} with model {actual_model_name}")
    
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
        
        # Create pod name
        pod_name = f"swe-{instance_id.lower().replace('/', '-').replace('_', '-')[:40]}"
        global_pod_name = pod_name
        pod = None
        
        try:
            # Start pod
            logger.info(f"Starting pod: {pod_name}")
            pod = global_pod = kodo_runner.start_container(
                image,
                name=pod_name,
                environment={
                    "PYTHONPATH": "/testbed",
                    "SWE_INSTANCE_ID": instance_id,
                    "http_proxy": "http://agent.baidu.com:8891",
                    "https_proxy": "http://agent.baidu.com:8891",
                    "PIP_INDEX_URL": "http://pip.baidu.com/pypi/simple",
                    "PIP_TRUSTED_HOST": "pip.baidu.com"
                }
            )
            
            logger.info(f"Pod {pod_name} started successfully")
            
            # Wait for pod to be ready
            await asyncio.sleep(3)
            
            # Setup environment
            kodo_runner.execute_command(pod, f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            
            # Create agent with R2E tools
            k8s_config = {
                "execution_mode": "k8s",
                "pod_name": pod_name,
                "namespace": namespace,
                "kubeconfig_path": kubeconfig_path
            }
            
            # Create R2E tools
            base_tools = {
                "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config.copy()),
                "r2e_file_editor": create_tool("R2EFileEditor", k8s_config.copy()),
                "r2e_search": create_tool("R2ESearch", k8s_config.copy()),
                "r2e_submit": create_tool("R2ESubmit", {})
            }
            
            # Wrap tools with custom descriptions
            tools = {}
            for tool_name, tool in base_tools.items():
                if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
                    tools[tool_name] = CustomDescriptionWrapper(tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name])
                else:
                    tools[tool_name] = tool
            
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
                max_rounds=50,
                debug=True,
                termination_tool_names=["r2e_submit"],
                action_parser=parse_xml_action_custom,
                system_prompt=custom_system_prompt,
                profiler=profiler
            )
            agent.set_tools(tools)
            
            # Run agent
            logger.info("Running agent to solve the issue...")
            llm_client = create_llm_client(
                api_key=API_KEY,
                base_url=BASE_URL,
                model=actual_model_name,
                debug=True
            )
            
            # Override the generate method to use custom max_tokens
            original_generate = llm_client.generate
            async def generate_with_custom_tokens(messages, max_tokens_param=None, **kwargs):
                return await original_generate(messages, max_tokens=max_tokens, **kwargs)
            llm_client.generate = generate_with_custom_tokens
            
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
            
            result = await agent.run_trajectory(
                prompt=prompt,
                llm_generate_func=llm_client.generate,
                request_id=f"swe_{instance_id}"
            )
            
            logger.info(f"Agent completed: {result.is_completed}")
            
            # Collect statistics from the trajectory
            stats = {
                'rounds': 0,
                'trajectory_length': len(result.steps),
                'tool_calls': [],
                'llm_calls': [],
                'is_completed': result.is_completed
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
            
            safe_instance_id = instance_id.replace('/', '_').replace('__', '-')
            trajectory_file = os.path.join(trajectory_dir, f"{safe_instance_id}_{timestamp}.jsonl")
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
                    profiler_file = os.path.join(profiler_dir, f"{safe_instance_id}_{timestamp}_profile.json")
                    profiler.export_events(profiler_file)
                    logger.info(f"Saved profiler data to: {profiler_file}")
                    
                    # Generate timeline visualization
                    timeline_file = os.path.join(profiler_dir, f"{safe_instance_id}_{timestamp}_timeline.html")
                    visualizer = ProfilerVisualizer({
                        "summary": profiler.get_summary(),
                        "events": [event.to_dict() for event in profiler.events]
                    })
                    visualizer.generate_html_timeline(timeline_file, title=f"Timeline: {instance_id}")
                    logger.info(f"Generated timeline visualization: {timeline_file}")
                except Exception as e:
                    logger.error(f"Failed to save profiler data: {e}")
            
            # Get patch
            logger.info("Generating patch...")
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
                logger.info(f"Generated patch ({len(patch)} chars)")
                
                # Save patch
                patches_dir = os.path.join(output_dir, "patches")
                os.makedirs(patches_dir, exist_ok=True)
                
                patch_filepath = os.path.join(patches_dir, f"{safe_instance_id}_{timestamp}.patch")
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
            logger.error(f"Error processing instance {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            
            result_data = {
                'success': False,
                'instance_id': instance_id,
                'error': str(e),
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
                 local_mode: bool = False):  # Local mode for debugging
        """Initialize the runner.
        
        Args:
            timeout_seconds: Maximum time in seconds for each instance (default: 600 = 10 minutes)
            max_tokens: Maximum tokens for model output (default: 16000)
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
        self.local_mode = local_mode
        self.patches = {}
        
    async def process_instances(self, instances: List[Dict[str, Any]]):
        """Process multiple instances concurrently using subprocess or locally."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create logs directory for subprocess logs
        if not self.local_mode:
            logs_dir = os.path.join(self.output_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
            logger.info(f"Subprocess logs will be saved to: {logs_dir}")
        
        temp_dir = tempfile.mkdtemp(prefix="swe_bench_")
        
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
                'max_tokens': self.max_tokens  # Add max_tokens to instance data
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
                task = self._run_local_instance(i, len(instances), instance_data_file, output_file, actual_model_name)
            else:
                # Subprocess mode: run in separate process with logging
                instance_id = instance.get('instance_id', f'instance_{i}')
                safe_instance_id = instance_id.replace('/', '_').replace('__', '-')
                log_file = os.path.join(self.output_dir, "logs", f"{safe_instance_id}.log")
                task = self._run_subprocess(i, len(instances), instance_data_file, output_file, actual_model_name, log_file)
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
            'throughput': {'start_time': start_time, 'completed': 0}
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
        logger.info(f"\n📊 Basic Metrics:")
        logger.info(f"  Total instances: {len(instances)}")
        logger.info(f"  Successful: {successful} ({successful/len(instances)*100:.1f}%)")
        logger.info(f"  Failed: {failed} ({failed/len(instances)*100:.1f}%)")
        logger.info(f"  Timeout: {timeout_count} ({timeout_count/len(instances)*100:.1f}%)")
        logger.info(f"  Total time: {elapsed_time:.2f} seconds")
        logger.info(f"  Average time per instance: {elapsed_time/len(instances):.2f} seconds")
        
        # Throughput metrics
        logger.info(f"\n⚡ Throughput:")
        logger.info(f"  Overall: {overall_throughput:.3f} rollouts/sec")
        logger.info(f"  Completed rollouts: {global_stats['throughput']['completed']}")
        if self.max_concurrent > 1:
            logger.info(f"  Concurrency: {self.max_concurrent}x parallel")
        
        # Tool usage statistics
        logger.info(f"\n🔧 Tool Usage:")
        logger.info(f"  Total tool calls: {global_stats['total_tool_calls']}")
        if global_stats['total_tool_calls'] > 0:
            logger.info(f"  Average tool execution time: {avg_tool_time:.3f} seconds")
            logger.info(f"  Tool calls by type:")
            for tool_name, tool_stats in global_stats['tool_calls_by_type'].items():
                avg_time = tool_stats['time'] / tool_stats['count'] if tool_stats['count'] > 0 else 0
                logger.info(f"    - {tool_name}: {tool_stats['count']} calls, avg {avg_time:.3f}s")
        
        # LLM usage statistics
        logger.info(f"\n🤖 LLM Usage:")
        logger.info(f"  Total LLM calls: {global_stats['total_llm_calls']}")
        if global_stats['total_llm_calls'] > 0:
            logger.info(f"  Average LLM call time: {avg_llm_time:.3f} seconds")
            logger.info(f"  Average LLM calls per instance: {global_stats['total_llm_calls']/len(instances):.1f}")
        
        # Trajectory statistics
        logger.info(f"\n📈 Trajectory Statistics:")
        if global_stats['rounds']['count'] > 0:
            logger.info(f"  Rounds: min={global_stats['rounds']['min']}, "
                      f"max={global_stats['rounds']['max']}, avg={avg_rounds:.1f}")
        if global_stats['trajectory_lengths']['count'] > 0:
            logger.info(f"  Trajectory length: min={global_stats['trajectory_lengths']['min']}, "
                      f"max={global_stats['trajectory_lengths']['max']}, avg={avg_trajectory_length:.1f}")
        
        logger.info(f"{'='*80}")
        
        # Save comprehensive summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        summary = {
            "total_instances": len(instances),
            "successful_patches": successful,
            "failed_instances": failed,
            "timeout_instances": timeout_count,
            "timeout_seconds": self.timeout_seconds,
            "max_tokens": self.max_tokens,
            "local_mode": self.local_mode,
            "successful_instance_ids": successful_instances,
            "failed_instance_ids": failed_instances,
            "timeout_instance_ids": timeout_instances,
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
                             output_file: str, model_name: str, log_file: str = None) -> Optional[Dict]:
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
process_single_instance('{instance_data_file}', '{output_file}', '{model_name}', '{log_file if log_file else ""}')
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
                    logger.error(f"Stderr: {stderr.decode()[:500]}")  # Limit stderr output
                return {
                    'success': False,
                    'instance_id': instance_id,
                    'error': f'Process failed with return code {process.returncode}'
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
                                 output_file: str, model_name: str) -> Optional[Dict]:
        """Run a single instance locally (without subprocess) for debugging."""
        # Load instance data to get instance_id
        with open(instance_data_file, 'rb') as f:
            data = pickle.load(f)
        instance_id = data['instance'].get('instance_id', f'instance_{index}')
        
        logger.info(f"[{index+1}/{total}] Starting local execution for {instance_id} with model {model_name}")
        
        try:
            # Run the instance processing directly
            process_single_instance(instance_data_file, output_file, model_name, None)
            
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
        sys.exit(1)
    
    asyncio.run(main())