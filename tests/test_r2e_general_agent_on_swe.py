#!/usr/bin/env python3
"""
Test GeneralAgent with R2E tools for SWE-bench-verified processing.
This script reads JSONL files, creates pods using kodo, runs agents, and collects patches.
"""

import asyncio
import sys
import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent, dump_trajectory
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.core.trajectory import TrajectoryStep, StepType
from workers.core.profiler import RolloutProfiler, EventType
from workers.core.safe_profiler import SafeProfiler
from workers.core.profiler_visualizer import ProfilerVisualizer
import logging
import re

# Import kodo for pod management
try:
    from kodo import ContainerRunner
except ImportError:
    print("ERROR: kodo package not found. Please install it with: pip install kodo")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "your-base-url-here")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "claude-3-sonnet")

# Import R2E configurations
from workers.tools.r2e_configs import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    CustomDescriptionWrapper,
    generate_custom_system_prompt
)


def get_event_type_safe(event_type_name: str) -> Optional['EventType']:
    """Safely get an EventType, falling back to CUSTOM if not available.
    
    Args:
        event_type_name: Name of the EventType attribute (e.g., 'INSTANCE_PROCESSING')
        
    Returns:
        The EventType if available, otherwise EventType.CUSTOM or None if that's also not available
    """
    try:
        return getattr(EventType, event_type_name)
    except AttributeError:
        logger.warning(f"EventType.{event_type_name} not available, trying CUSTOM")
        try:
            return EventType.CUSTOM
        except AttributeError:
            logger.error(f"EventType.CUSTOM also not available, profiling may not work correctly")
            # Return the first available EventType
            for attr in dir(EventType):
                if not attr.startswith('_'):
                    try:
                        return getattr(EventType, attr)
                    except:
                        continue
            return None


class SWEBenchRunner:
    """Runner for SWE-bench-verified instances."""
    
    def __init__(self, namespace: str = "default", kubeconfig_path: Optional[str] = None, output_dir: str = "./swe_patches", max_concurrent: int = 1, enable_profiling: bool = False):
        """Initialize the SWE-bench runner.
        
        Args:
            namespace: Kubernetes namespace to use
            kubeconfig_path: Path to kubeconfig file (optional)
            output_dir: Directory to save outputs
            max_concurrent: Maximum number of concurrent instances to process
            enable_profiling: Enable performance profiling for each instance
        """
        self.namespace = namespace
        self.kubeconfig_path = kubeconfig_path
        self.kodo_runner = None
        self.patches = {}  # Store patches by instance_id
        self.output_dir = output_dir
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.enable_profiling = enable_profiling
        
    def _init_kodo(self):
        """Initialize kodo ContainerRunner."""
        if not self.kodo_runner:
            self.kodo_runner = ContainerRunner(
                backend="kubernetes",
                namespace=self.namespace,
                kubeconfig_path=self.kubeconfig_path
            )
            logger.info(f"Initialized kodo runner for namespace: {self.namespace}")
    
    async def process_instance(self, instance: Dict[str, Any]) -> Optional[str]:
        """Process a single SWE-bench instance.
        
        Args:
            instance: Dictionary with instance_id, issue, image fields
            
        Returns:
            The patch string if successful, None otherwise
        """
        async with self.semaphore:  # Limit concurrent executions
            return await self._process_instance_impl(instance)
    
    async def _process_instance_impl(self, instance: Dict[str, Any]) -> Optional[str]:
        """Internal implementation of process_instance."""
        instance_id = instance.get("instance_id", "unknown")
        issue = instance.get("problem_statement", "")
        image = instance.get("image", "")
        base_commit = instance.get("base_commit", "")
        
        if not all([instance_id, issue, image]):
            logger.error(f"Missing required fields for instance {instance_id}")
            return None
            
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing instance: {instance_id}")
        logger.info(f"Image: {image}")
        logger.info(f"Issue: {issue[:100]}..." if len(issue) > 100 else f"Issue: {issue}")
        logger.info(f"{'='*80}")
        
        # Create overall instance profiler if enabled
        instance_profiler = None
        instance_event_id = None
        if self.enable_profiling:
            try:
                # Use SafeProfiler to avoid potential deadlocks with external libraries
                instance_profiler = SafeProfiler(enabled=True)
                event_type = get_event_type_safe('INSTANCE_PROCESSING')
                if event_type:
                    instance_event_id = instance_profiler.start_event(
                        f"instance_{instance_id}",
                        event_type,
                        {"instance_id": instance_id, "image": image}
                    )
                    logger.debug(f"Started profiling instance with event_id: {instance_event_id}")
                else:
                    logger.warning("Could not get any EventType, profiling disabled for this instance")
                    instance_profiler = None
            except Exception as e:
                logger.error(f"Failed to create profiler for {instance_id}: {e}")
                import traceback
                logger.error(f"Profiler creation traceback:\n{traceback.format_exc()}")
                # Continue without profiling
                instance_profiler = None
        
        # Initialize kodo if needed
        self._init_kodo()
        
        # Create pod name from instance_id
        pod_name = f"swe-{instance_id.lower().replace('/', '-').replace('_', '-')[:40]}"
        pod = None
        
        try:
            # 1. Start pod using kodo
            logger.info(f"Starting pod: {pod_name}")
            
            # Profile pod creation if profiler is available
            pod_creation_event_id = None
            if instance_profiler and instance_profiler.enabled:
                try:
                    event_type = get_event_type_safe('POD_CREATION')
                    if event_type:
                        pod_creation_event_id = instance_profiler.start_event(
                            f"pod_creation_{instance_id}",
                            event_type,
                            {"instance_id": instance_id, "pod_name": pod_name, "image": image}
                        )
                        logger.debug(f"Started profiling pod creation with event_id: {pod_creation_event_id}")
                except Exception as e:
                    logger.warning(f"Failed to start pod creation profiling: {e}")
                    # Continue without profiling this event
            
            # End pod creation profiling BEFORE calling kodo_runner to avoid potential deadlock
            if instance_profiler and instance_profiler.enabled and pod_creation_event_id:
                try:
                    instance_profiler.end_event(pod_creation_event_id)
                    logger.debug(f"Ended profiling pod creation BEFORE kodo call with event_id: {pod_creation_event_id}")
                    pod_creation_event_id = None  # Clear to avoid double-ending
                except Exception as e:
                    logger.warning(f"Failed to end pod creation profiling: {e}")
            
            logger.info(f"Calling kodo_runner.start_container for pod: {pod_name}")
            pod = self.kodo_runner.start_container(
                image=image,
                name=pod_name,
                node_selector={"nvme": "ok"},
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
            await asyncio.sleep(5)
            output, exit_code = self.kodo_runner.execute_command(pod, f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            # 2. Initialize git repo to track changes
            #logger.info("Initializing git repository...")
            #output, exit_code = self.kodo_runner.execute_command(
            #    pod, 
            #    "cd /testbed && git init && git config user.email 'agent@swe.bench' && git config user.name 'SWE Agent' && git add -A && git commit -m 'Initial commit' || true"
            #)
            #if exit_code != 0:
            #    logger.warning(f"Git init had issues: {output}")
            
            # 3. Create agent with R2E tools configured for this pod
            k8s_config = {
                "execution_mode": "k8s",
                "pod_name": pod_name,
                "namespace": self.namespace,
                "kubeconfig_path": self.kubeconfig_path
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
            
            # Use the instance profiler for the agent
            profiler = instance_profiler
            
            # Create agent
            agent = GeneralAgent(
                max_rounds=30,  # More rounds for complex issues
                debug=False,
                termination_tool_names=["r2e_submit"],
                action_parser=parse_xml_action_custom,
                system_prompt=custom_system_prompt,
                profiler=profiler  # Pass profiler if enabled
            )
            agent.set_tools(tools)
            
            # 4. Run agent with the issue
            logger.info("Running agent to solve the issue...")
            llm_client = create_llm_client(
                api_key=API_KEY,
                base_url=BASE_URL,
                model=MODEL_NAME,
                debug=False
            )
            
            # Prepare the prompt with issue details
            prompt = f"""
Please analyze and fix the following issue in the repository at /testbed:

{issue}

First explore the repository structure, understand the codebase, locate the relevant files, and then make the necessary changes to fix the issue. When you're done, call the finish function to submit your solution.
"""
            
            result = await agent.run_trajectory(
                prompt=prompt,
                llm_generate_func=llm_client.generate,
                request_id=f"swe_{instance_id}"
            )
            
            logger.info(f"[DEBUG] run_trajectory completed for {instance_id}")
            logger.info(f"Agent completed: {result.is_completed}")
            logger.info(f"Total steps: {len(result.steps)}")
            
            # Save trajectory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trajectory_dir = os.path.join(self.output_dir, "trajectories")
            os.makedirs(trajectory_dir, exist_ok=True)
            
            # Use instance_id with timestamp for filename
            safe_instance_id = instance_id.replace('/', '_').replace('__', '-')
            trajectory_file = os.path.join(trajectory_dir, f"{safe_instance_id}_{timestamp}.jsonl")
            
            logger.info(f"[DEBUG] Starting dump_trajectory for {instance_id}")
            try:
                dump_trajectory(result, trajectory_file, format="jsonl")
                logger.info(f"[DEBUG] dump_trajectory completed for {instance_id}")
                logger.info(f"Saved trajectory to: {trajectory_file}")
            except Exception as e:
                logger.error(f"[DEBUG] dump_trajectory failed for {instance_id}: {e}")
                import traceback
                traceback.print_exc()
            
            # Save profiler data if enabled
            if instance_profiler and instance_profiler.enabled:
                profiler_dir = os.path.join(self.output_dir, "profiles")
                os.makedirs(profiler_dir, exist_ok=True)
                
                # Save profiler data
                profiler_file = os.path.join(profiler_dir, f"{safe_instance_id}_{timestamp}_profile.json")
                instance_profiler.export_events(profiler_file)
                logger.info(f"Saved profiler data to: {profiler_file}")
                
                # Generate timeline visualization
                timeline_file = os.path.join(profiler_dir, f"{safe_instance_id}_{timestamp}_timeline.html")
                visualizer = ProfilerVisualizer({
                    "summary": instance_profiler.get_summary(),
                    "events": [event.to_dict() for event in instance_profiler.events]
                })
                visualizer.generate_html_timeline(timeline_file, title=f"Timeline: {instance_id}")
                logger.info(f"Generated timeline visualization: {timeline_file}")
            
            # 5. Get the patch using git diff
            logger.info(f"[DEBUG] Starting patch generation for {instance_id}")
            
            # First add all changes
            logger.info(f"[DEBUG] Running git add -A for {instance_id}")
            try:
                output, exit_code = self.kodo_runner.execute_command(
                    pod,
                    "cd /testbed && git add -A"
                )
                logger.info(f"[DEBUG] git add completed with exit code: {exit_code}")
            except Exception as e:
                logger.error(f"[DEBUG] git add failed: {e}")
                return None
            
            # Get the diff
            logger.info(f"[DEBUG] Running git diff {base_commit} for {instance_id}")
            try:
                output, exit_code = self.kodo_runner.execute_command(
                    pod,
                    f"git add -A && git diff {base_commit} > /testbed/{base_commit}_diff.patch && cat /testbed/{base_commit}_diff.patch"
                )
                logger.info(f"[DEBUG] git diff completed with exit code: {exit_code}, output length: {len(output) if output else 0}")
            except Exception as e:
                logger.error(f"[DEBUG] git diff failed: {e}")
                return None

            logger.info("begin to generate patch file")
            logger.info(exit_code)
            logger.info(output.strip())
            logger.info("ok")
            if int(exit_code) == 0:
                patch = output.strip()
                logger.info(f"Generated patch ({len(patch)} chars)")
                logger.debug(f"Patch preview: {patch[:500]}..." if len(patch) > 500 else f"Patch: {patch}")
                
                # Save patch immediately to results directory
                patches_dir = os.path.join(self.output_dir, "patches")
                os.makedirs(patches_dir, exist_ok=True)
                
                # Create patch filename with timestamp
                patch_filename = f"{safe_instance_id}_{timestamp}.patch"
                patch_filepath = os.path.join(patches_dir, patch_filename)
                
                # Write patch to file
                with open(patch_filepath, 'w', encoding='utf-8') as f:
                    f.write(patch)
                logger.info(f"Saved patch to: {patch_filepath}")
                
                # Also save a latest version without timestamp for easy access
                latest_patch_filepath = os.path.join(patches_dir, f"{safe_instance_id}_latest.patch")
                with open(latest_patch_filepath, 'w', encoding='utf-8') as f:
                    f.write(patch)
                logger.info(f"Saved latest patch to: {latest_patch_filepath}")
                
                return patch
            else:
                logger.warning(f"No patch generated or git diff failed: {output}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing instance {instance_id}: {e}")
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Clean up pod
            if pod and self.kodo_runner:
                try:
                    logger.info(f"Stopping container/pod: {pod_name}")
                    
                    # Profile pod deletion if profiler is available
                    pod_deletion_event_id = None
                    if instance_profiler and instance_profiler.enabled:
                        try:
                            event_type = get_event_type_safe('POD_DELETION')
                            if event_type:
                                pod_deletion_event_id = instance_profiler.start_event(
                                    f"pod_deletion_{instance_id}",
                                    event_type,
                                    {"instance_id": instance_id, "pod_name": pod_name}
                                )
                                logger.debug(f"Started profiling pod deletion with event_id: {pod_deletion_event_id}")
                        except Exception as e:
                            logger.warning(f"Failed to start pod deletion profiling: {e}")
                    
                    # Stop the container to end this rollout
                    self.kodo_runner.stop_container(pod)
                    
                    # End pod deletion profiling
                    if instance_profiler and instance_profiler.enabled and pod_deletion_event_id:
                        try:
                            instance_profiler.end_event(pod_deletion_event_id)
                            logger.debug(f"Ended profiling pod deletion with event_id: {pod_deletion_event_id}")
                        except Exception as e:
                            logger.warning(f"Failed to end pod deletion profiling: {e}")
                    
                    logger.info(f"Container/pod {pod_name} stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping container/pod: {e}")
            
            # End overall instance processing profiling
            if instance_profiler and instance_profiler.enabled and instance_event_id:
                instance_profiler.end_event(instance_event_id)
    
    async def process_jsonl_file(self, jsonl_path: str):
        """Process all instances in a JSONL file.
        
        Args:
            jsonl_path: Path to the JSONL file
        """
        if not os.path.exists(jsonl_path):
            logger.error(f"JSONL file not found: {jsonl_path}")
            return
            
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Read instances
        instances = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        instances.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line: {e}")
        
        logger.info(f"Loaded {len(instances)} instances from {jsonl_path}")
        logger.info(f"Processing with max concurrency: {self.max_concurrent}")
        
        # Create tasks for all instances
        tasks = []
        for i, instance in enumerate(instances):
            task = asyncio.create_task(self._process_with_logging(instance, i+1, len(instances)))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, (instance, result) in enumerate(zip(instances, results)):
            if isinstance(result, Exception):
                logger.error(f"Exception processing instance {i+1}: {result}")
            elif result:
                instance_id = instance.get("instance_id", f"unknown_{i}")
                self.patches[instance_id] = result
                # Patch is already saved in _process_instance_impl
        
        # Save summary
        summary_file = os.path.join(self.output_dir, "summary.json")
        summary = {
            "total_instances": len(instances),
            "successful_patches": len(self.patches),
            "instance_ids": list(self.patches.keys()),
            "timestamp": datetime.now().isoformat(),
            "trajectory_dir": os.path.join(self.output_dir, "trajectories"),
            "patches_dir": os.path.join(self.output_dir, "patches"),
            "max_concurrent": self.max_concurrent,
            "profiling_enabled": self.enable_profiling
        }
        
        if self.enable_profiling:
            summary["profiles_dir"] = os.path.join(self.output_dir, "profiles")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to: {summary_file}")
    
    async def _process_with_logging(self, instance: Dict[str, Any], index: int, total: int) -> Optional[str]:
        """Process an instance with logging."""
        instance_id = instance.get("instance_id", f"unknown_{index}")
        logger.info(f"\n[{index}/{total}] Starting processing of instance: {instance_id}")
        
        try:
            patch = await self.process_instance(instance)
            if patch:
                logger.info(f"[{index}/{total}] Successfully processed instance: {instance_id}")
            else:
                logger.warning(f"[{index}/{total}] No patch generated for instance: {instance_id}")
            return patch
        except Exception as e:
            logger.error(f"[{index}/{total}] Error processing instance {instance_id}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        
        # Cleanup kodo runner
        if self.kodo_runner:
            try:
                self.kodo_runner.cleanup()
            except Exception as e:
                logger.error(f"Error during final cleanup: {e}")
    
    def get_patches(self) -> Dict[str, str]:
        """Get all collected patches."""
        return self.patches.copy()


async def main():
    """Main function to run SWE-bench processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run R2E agent on SWE-bench-verified instances")
    parser.add_argument("jsonl_file", help="Path to JSONL file with SWE-bench instances")
    parser.add_argument("--output-dir", default="./swe_patches", help="Directory to save patches")
    parser.add_argument("--namespace", default=os.getenv("K8S_NAMESPACE", "default"), 
                       help="Kubernetes namespace (default: from K8S_NAMESPACE env or 'default')")
    parser.add_argument("--kubeconfig", default=os.getenv("KUBECONFIG", None),
                       help="Path to kubeconfig file (default: from KUBECONFIG env)")
    parser.add_argument("--max-concurrent", type=int, default=1,
                       help="Maximum number of concurrent instances to process (default: 1)")
    parser.add_argument("--enable-profiling", action="store_true",
                       help="Enable performance profiling for each instance")
    
    args = parser.parse_args()
    
    logger.info("ðŸš€ Starting SWE-bench R2E Agent Runner")
    logger.info(f"JSONL file: {args.jsonl_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Namespace: {args.namespace}")
    logger.info(f"Max concurrent: {args.max_concurrent}")
    logger.info(f"Profiling enabled: {args.enable_profiling}")
    
    # Create runner with output directory and concurrency
    runner = SWEBenchRunner(
        namespace=args.namespace,
        kubeconfig_path=args.kubeconfig,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        enable_profiling=args.enable_profiling
    )
    
    # Process JSONL file
    await runner.process_jsonl_file(args.jsonl_file)
    
    # Print summary
    patches = runner.get_patches()
    logger.info(f"\n{'='*80}")
    logger.info(f"âœ… Processing complete!")
    logger.info(f"Total patches generated: {len(patches)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # If no arguments provided, show usage example
    if len(sys.argv) == 1:
        print("Usage: python test_r2e_general_agent_on_swe.py <jsonl_file> [options]")
        print("\nExample:")
        print("  python test_r2e_general_agent_on_swe.py swe_bench_verified.jsonl --output-dir ./patches")
        print("\nConcurrent processing example:")
        print("  python test_r2e_general_agent_on_swe.py swe_bench_verified.jsonl --max-concurrent 10")
        print("\nFor testing with a sample JSONL:")
        print('  echo \'{"instance_id": "test-001", "issue": "Fix import error", "image": "ubuntu:20.04"}\' > test.jsonl')
        print("  python test_r2e_general_agent_on_swe.py test.jsonl")
        sys.exit(1)
    
    asyncio.run(main())