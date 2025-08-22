#!/usr/bin/env python3
"""
Test GeneralAgent with R2E tools for K8S execution.
This configures a GeneralAgent instance with R2E tools that execute in K8S.
Supports reading data from JSON files and processing multiple instances.
"""

import asyncio
import sys
import os
import json
import yaml
from datetime import datetime
import argparse

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # python-dotenv not installed, skip

# Import kodo for pod management
try:
    from kodo import ContainerRunner
    KODO_AVAILABLE = True
except ImportError:
    ContainerRunner = None
    KODO_AVAILABLE = False

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from workers.agents.general_agent import GeneralAgent, dump_trajectory, save_trajectory_as_messages
from workers.core import create_tool
from workers.utils import create_llm_client
from workers.core.trajectory import TrajectoryStep, StepType
from workers.tools.r2e_configs import (
    CUSTOM_TOOL_DESCRIPTIONS,
    parse_xml_action_custom,
    CustomDescriptionWrapper
)
import logging
import re
from typing import Dict, Any, List, Optional, Union

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# LLM configuration
API_KEY = os.getenv("LLM_API_KEY", "your-api-key-here")
BASE_URL = os.getenv("LLM_BASE_URL", "xxx")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "xxx")
print(f"API_KEY: {API_KEY}")
print(f"BASE_URL: {BASE_URL}")
print(f"MODEL_NAME: {MODEL_NAME}")


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        raise


def load_instances_from_json(json_file_path: str) -> List[Dict[str, Any]]:
    """Load instances from JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing instances
        
    Returns:
        List of instance dictionaries
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            instances = json.load(f)
        
        logger.info(f"Loaded {len(instances)} instances from {json_file_path}")
        return instances
    except Exception as e:
        logger.error(f"Failed to load instances from {json_file_path}: {e}")
        raise


def generate_k8s_safe_name(instance_id: str, max_length: int = 30) -> str:
    """Generate a Kubernetes-safe name from instance_id.
    
    Args:
        instance_id: Original instance ID
        max_length: Maximum length for the cleaned name (before suffix)
        
    Returns:
        Kubernetes RFC 1123 compliant name
    """
    # Remove special characters and convert to lowercase
    cleaned_name = re.sub(r'[^a-z0-9-]', '-', instance_id.lower())
    # Replace multiple dashes with single dash
    cleaned_name = re.sub(r'-+', '-', cleaned_name)
    # Remove leading/trailing dashes
    cleaned_name = cleaned_name.strip('-')
    
    # Ensure it starts with a letter (K8s requirement)
    if cleaned_name and not cleaned_name[0].isalpha():
        cleaned_name = f"pod-{cleaned_name}"
    
    # Limit length and ensure it ends with alphanumeric
    if len(cleaned_name) > max_length:
        cleaned_name = cleaned_name[:max_length].rstrip('-')
    
    return cleaned_name


def create_instance_output_dir(base_output_dir: str, instance_id: str) -> str:
    """Create output directory for a specific instance.
    
    Args:
        base_output_dir: Base output directory
        instance_id: Instance ID to create directory for
        
    Returns:
        Path to the instance-specific output directory
    """
    # Clean instance_id for use as directory name (more permissive than K8s)
    safe_instance_id = re.sub(r'[^\w\-_.]', '_', instance_id)
    # Limit length to avoid filesystem issues
    if len(safe_instance_id) > 100:
        safe_instance_id = safe_instance_id[:100]
    instance_dir = os.path.join(base_output_dir, safe_instance_id)
    os.makedirs(instance_dir, exist_ok=True)
    return instance_dir


def create_trajectory_subdir(instance_output_dir: str, status: str = "success") -> str:
    """Create a subdirectory for trajectory files with status and numbering.
    
    Args:
        instance_output_dir: Base instance output directory
        status: Status identifier ("success" or "failure")
        
    Returns:
        Path to the trajectory subdirectory
    """
    # List existing subdirectories to find the next available number
    existing_dirs = []
    for item in os.listdir(instance_output_dir):
        item_path = os.path.join(instance_output_dir, item)
        if os.path.isdir(item_path) and item.endswith(f"-{status}"):
            # Extract number from directory name like "0-success" or "1-failure"
            match = re.search(rf"(\d+)-{status}", item)
            if match:
                existing_dirs.append(int(match.group(1)))
    
    # Find the next available number
    next_number = 0
    if existing_dirs:
        next_number = max(existing_dirs) + 1
    
    # Create subdirectory
    subdir_name = f"{next_number}-{status}"
    trajectory_subdir = os.path.join(instance_output_dir, subdir_name)
    os.makedirs(trajectory_subdir, exist_ok=True)
    
    return trajectory_subdir


def generate_trajectory_filename(base_name: str, extension: str) -> str:
    """Generate a simple trajectory filename.
    
    Args:
        base_name: Base name for the file (e.g., "trajectory")
        extension: File extension (e.g., "json", "jsonl")
        
    Returns:
        Generated filename
    """
    return f"{base_name}.{extension}"


async def apply_patch(patch_content: str, k8s_config: Dict[str, Any]) -> bool:
    """Apply a git patch to the codebase.
    
    Args:
        patch_content: The patch content as a string
        k8s_config: K8S configuration for tool execution
        
    Returns:
        True if patch was applied successfully, False otherwise
    """
    try:
        # Create bash executor tool for applying patch
        bash_tool = create_tool("R2EBashExecutor", k8s_config.copy())
        
        # Save patch to a temporary file
        save_patch_command = f"""cat > /tmp/patch.diff << 'EOF'
{patch_content}
EOF"""
        
        result = await bash_tool.execute_tool(
            instance_id="apply_patch_save",
            parameters={"command": save_patch_command}
        )
        
        if not result.success:
            logger.error(f"Failed to save patch file: {result.error}")
            return False
        
        # Apply the patch using git apply
        apply_command = "cd /testbed && git apply /tmp/patch.diff"
        result = await bash_tool.execute_tool(
            instance_id="apply_patch_git",
            parameters={"command": apply_command}
        )
        
        if result.success:
            exit_code = result.result.get("return_code", -1)
            if exit_code == 0:
                logger.info("Patch applied successfully with git apply")
                return True
            else:
                logger.warning(f"git apply failed with exit code {exit_code}, trying patch command...")
        else:
            logger.warning(f"git apply failed: {result.error}, trying patch command...")
        
        # If git apply fails, try using patch command
        patch_command = "cd /testbed && patch -p1 < /tmp/patch.diff"
        result = await bash_tool.execute_tool(
            instance_id="apply_patch_patch",
            parameters={"command": patch_command}
        )
        
        if result.success:
            exit_code = result.result.get("return_code", -1)
            if exit_code == 0:
                logger.info("Patch applied successfully with patch command")
                return True
            else:
                logger.error(f"patch command failed with exit code {exit_code}")
                # Log the output for debugging
                output = result.result.get("stdout", "")
                error = result.result.get("stderr", "")
                logger.error(f"patch stdout: {output}")
                logger.error(f"patch stderr: {error}")
        else:
            logger.error(f"patch command failed: {result.error}")
        
        return False
        
    except Exception as e:
        logger.error(f"Exception during patch application: {e}")
        return False


async def process_single_instance(
    instance: Dict[str, Any], 
    base_output_dir: str,
    k8s_config: Dict[str, Any]
) -> bool:
    """Process a single instance and save trajectory.
    
    Args:
        instance: Instance data dictionary
        base_output_dir: Base output directory
        k8s_config: K8S configuration
        
    Returns:
        True if processing was successful, False otherwise
    """
    instance_id = instance.get("instance_id", "unknown")
    problem_statement = instance.get("problem_statement", "")
    image_name = instance.get("image_name", "")
    repo = instance.get("repo", "")
    
    # Generate unique pod name from instance_id
    import uuid
    pod_suffix = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    
    # Generate Kubernetes-safe name
    cleaned_instance_id = generate_k8s_safe_name(instance_id, max_length=30)
    pod_name = f"{cleaned_instance_id}-{pod_suffix}"
    
    logger.info(f"Generated pod name: {pod_name}")
    logger.info(f"Using image: {image_name}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing instance: {instance_id}")
    logger.info(f"Repo: {repo}")
    logger.info(f"Image: {image_name}")
    logger.info(f"Problem: {problem_statement[:100]}..." if len(problem_statement) > 100 else f"Problem: {problem_statement}")
    logger.info(f"{'='*80}")
    
    # Create instance-specific output directory
    instance_output_dir = create_instance_output_dir(base_output_dir, instance_id)
    logger.info(f"Instance output directory: {instance_output_dir}")
    
    # Initialize kodo runner and create pod
    pod = None
    kodo_runner = None
    
    if not KODO_AVAILABLE:
        logger.error("❌ Kodo library not available. Cannot create pod.")
        return False
    
    # Initialize variables for cleanup
    pod = None
    kodo_runner = None
    
    try:
        # Create kodo runner
        kodo_runner = ContainerRunner(
            backend="kubernetes",
            namespace=k8s_config.get("namespace", "default"),
            kubeconfig_path=k8s_config.get("kubeconfig_path")
        )
        
        # Create pod with the specified image
        logger.info(f"Creating pod {pod_name} with image {image_name}...")
        pod = kodo_runner.start_container(
            image=image_name,
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
        
        logger.info(f"✅ Pod {pod_name} created successfully")
        
        # Wait for pod to be ready
        await asyncio.sleep(5)
        
        # Setup environment
        kodo_runner.execute_command(pod, f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
        
        # Update k8s_config with the actual pod name
        k8s_config_with_pod = k8s_config.copy()
        k8s_config_with_pod["pod_name"] = pod_name
        
        # Create R2E tools with K8S configuration
        logger.info("Creating R2E tools for K8S execution...")
        
        # Create base tools
        base_tools = {
            "r2e_bash_executor": create_tool("R2EBashExecutor", k8s_config_with_pod.copy()),
            "r2e_file_editor": create_tool("R2EFileEditor", k8s_config_with_pod.copy()),
            "r2e_search": create_tool("R2ESearch", k8s_config_with_pod.copy()),
            "r2e_submit": create_tool("R2ESubmit", {})
        }
        
        # Wrap tools with custom descriptions
        tools = {}
        for tool_name, tool in base_tools.items():
            if tool_name in CUSTOM_TOOL_DESCRIPTIONS:
                tools[tool_name] = CustomDescriptionWrapper(tool, CUSTOM_TOOL_DESCRIPTIONS[tool_name])
            else:
                tools[tool_name] = tool
        
        logger.info(f"Created {len(tools)} R2E tools")
        
        # Create GeneralAgent with R2E tools
        logger.info("Creating GeneralAgent with R2E tools...")
        
        # Load configuration from YAML file
        config_path = "/mnt/cfs_bj_mt/xuruijie/collect_traj/agentic-rollout-library/config/r2egym/edit_fn_calling.yaml"
        config = load_config_from_yaml(config_path)
        
        # Use system prompt and instance prompt from YAML config
        system_prompt = config.get("system_prompt", "")
        instance_prompt_template = config.get("instance_prompt", "")
        
        # Generate instance prompt by replacing placeholder
        instance_prompt = instance_prompt_template.replace("{problem_statement}", problem_statement)
        
        # Create agent instance with termination tool and custom XML parser
        agent = GeneralAgent(
            max_rounds=30,  # More rounds for complex issues
            debug=False,  # Disable debug output for cleaner logs
            termination_tool_names=["r2e_submit"],  # Mark r2e_submit as termination tool
            action_parser=parse_xml_action_custom,  # Use custom XML action parser
            system_prompt=system_prompt  # Use system prompt from YAML config
        )
        agent.set_tools(tools)
        
        logger.info("GeneralAgent created successfully")
        
        # Apply patch before running agent (if available)
        patch_content = instance.get("patch", "")
        if patch_content:
            logger.info(f"🩹 Applying patch before running agent...")
            patch_applied = await apply_patch(patch_content, k8s_config_with_pod)
            if patch_applied:
                logger.info(f"✅ Patch applied successfully before agent execution")
            else:
                logger.error(f"❌ Failed to apply patch before agent execution")
        else:
            logger.info(f"ℹ️  No patch found in instance data")
        
        # Use instance prompt from YAML config
        task_prompt = instance_prompt
        
        logger.info("Executing agent...")
        
        # Create LLM client
        llm_client = create_llm_client(
            api_key=API_KEY,
            base_url=BASE_URL,
            model=MODEL_NAME,
            debug=False
        )
        
        result = await agent.run_trajectory(
            prompt=task_prompt,
            llm_generate_func=llm_client.generate,
            request_id=f"instance_{instance_id}"
        )
        
        logger.info(f"✅ Instance {instance_id} completed")
        logger.info(f"Total steps: {len(result.steps)}")
        logger.info(f"Completed: {result.is_completed}")
        
        # Validate code changes if validation data is available
        validation_result = await validate_code_changes(instance, instance_output_dir, k8s_config_with_pod, pod, kodo_runner)
        
        # Determine status based on validation results
        # If no validation tests, consider it a success
        # If validation tests exist, check if all tests passed
        status = "success"  # default status
        
        if validation_result and not validation_result.get("summary", {}).get("error", False):
            # Check if all tests passed
            all_tests_passed = True
            
            # Check FAIL_TO_PASS tests (these should pass after the fix)
            if "fail_to_pass" in validation_result.get("summary", {}):
                fail_summary = validation_result["summary"]["fail_to_pass"]
                if fail_summary["total"] > 0 and fail_summary["passed"] < fail_summary["total"]:
                    all_tests_passed = False
            
            # Check PASS_TO_PASS tests (these should continue to pass)
            if "pass_to_pass" in validation_result.get("summary", {}):
                pass_summary = validation_result["summary"]["pass_to_pass"]
                if pass_summary["total"] > 0 and pass_summary["passed"] < pass_summary["total"]:
                    all_tests_passed = False
            
            # Set status based on test results
            status = "success" if all_tests_passed else "failure"
            
            logger.info(f"🔍 Test validation result: {'✅ ALL TESTS PASSED' if all_tests_passed else '❌ SOME TESTS FAILED'}")
            logger.info(f"📝 Will save trajectory with status: {status}")
        
        # Create trajectory subdirectory with status and numbering
        trajectory_subdir = create_trajectory_subdir(instance_output_dir, status)
        logger.info(f"📁 Created trajectory subdirectory: {os.path.basename(trajectory_subdir)}")
        
        # Generate simple filename (only JSONL format)
        trajectory_filename = generate_trajectory_filename("trajectory", "jsonl")
        
        # Save JSONL format
        trajectory_file = os.path.join(trajectory_subdir, trajectory_filename)
        dump_trajectory(result, trajectory_file, format="jsonl")
        logger.info(f"📝 Saved trajectory to: {trajectory_file}")
        
        # Save instance metadata
        metadata_file = os.path.join(trajectory_subdir, "metadata.json")
        metadata = {
            "instance_id": instance_id,
            "repo": repo,
            "image_name": image_name,
            "problem_statement": problem_statement,
            "processing_time": datetime.now().isoformat(),
            "total_steps": len(result.steps),
            "completed": result.is_completed,
            "trajectory_status": status,
            "trajectory_files": [
                trajectory_filename
            ]
        }
        
        # Add validation results to metadata if available
        if validation_result:
            metadata["validation"] = validation_result
            metadata["all_tests_passed"] = (status == "success")
            
            # Also save validation results as a separate file
            validation_file = os.path.join(trajectory_subdir, "validation_results.json")
            with open(validation_file, 'w') as f:
                json.dump(validation_result, f, indent=2)
            logger.info(f"📝 Saved validation results to: {validation_file}")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"📝 Saved metadata to: {metadata_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Instance {instance_id} failed: {e}")
        
        # Save error information
        error_file = os.path.join(instance_output_dir, "error_log.json")
        error_info = {
            "instance_id": instance_id,
            "error_time": datetime.now().isoformat(),
            "error_message": str(e),
            "repo": repo,
            "image_name": image_name
        }
        with open(error_file, 'w') as f:
            json.dump(error_info, f, indent=2)
        logger.info(f"📝 Saved error log to: {error_file}")
        
        # Save failure trajectory files if we have any partial results
        try:
            if 'result' in locals() and hasattr(result, 'steps') and len(result.steps) > 0:
                # Create failure trajectory subdirectory
                failure_subdir = create_trajectory_subdir(instance_output_dir, "failure")
                logger.info(f"📁 Created failure trajectory subdirectory: {os.path.basename(failure_subdir)}")
                
                # Generate simple filename (only JSONL format)
                failure_trajectory_filename = generate_trajectory_filename("trajectory", "jsonl")
                
                # Save partial trajectory as failure
                failure_trajectory_file = os.path.join(failure_subdir, failure_trajectory_filename)
                dump_trajectory(result, failure_trajectory_file, format="jsonl")
                logger.info(f"📝 Saved failure trajectory to: {failure_trajectory_file}")
                
                # Save failure metadata
                failure_metadata_file = os.path.join(failure_subdir, "metadata.json")
                failure_metadata = {
                    "instance_id": instance_id,
                    "repo": repo,
                    "image_name": image_name,
                    "failure_time": datetime.now().isoformat(),
                    "failure_reason": "program_exception",
                    "error_message": str(e),
                    "total_steps": len(result.steps),
                    "completed": result.is_completed,
                    "trajectory_files": [
                        failure_trajectory_filename
                    ]
                }
                with open(failure_metadata_file, 'w') as f:
                    json.dump(failure_metadata, f, indent=2)
                logger.info(f"📝 Saved failure metadata to: {failure_metadata_file}")
                
                # If we have validation results, save them too
                if 'validation_result' in locals() and validation_result:
                    failure_validation_file = os.path.join(failure_subdir, "validation_results.json")
                    with open(failure_validation_file, 'w') as f:
                        json.dump(validation_result, f, indent=2)
                    logger.info(f"📝 Saved failure validation results to: {failure_validation_file}")
                
                # Update error info with trajectory files
                error_info["trajectory_files"] = [
                    failure_trajectory_filename
                ]
                error_info["failure_reason"] = "program_exception"
                error_info["trajectory_subdir"] = os.path.basename(failure_subdir)
                with open(error_file, 'w') as f:
                    json.dump(error_info, f, indent=2)
        except Exception as save_error:
            logger.error(f"Failed to save failure trajectory: {save_error}")
        
        return False
        
    finally:
        # Clean up pod (aligned with test_r2e_general_agent_on_swe.py approach)
        if pod and kodo_runner:
            try:
                logger.info(f"Stopping container/pod: {pod_name}")
                kodo_runner.stop_container(pod)
                logger.info(f"Container/pod {pod_name} stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping container/pod: {e}")


async def validate_code_changes(
    instance: Dict[str, Any], 
    instance_output_dir: str,
    k8s_config: Dict[str, Any],
    pod=None,
    kodo_runner=None
) -> Optional[Dict[str, Any]]:
    """Validate code changes by running tests specified in the instance.
    
    Args:
        instance: Instance data dictionary
        instance_output_dir: Output directory for this instance
        k8s_config: K8S configuration
        pod: K8S pod object for executing commands
        kodo_runner: Kodo runner for pod management
        
    Returns:
        Validation results dictionary or None if no validation data
    """
    instance_id = instance.get("instance_id", "unknown")
    fail_to_pass_tests = instance.get("FAIL_TO_PASS", [])
    pass_to_pass_tests = instance.get("PASS_TO_PASS", [])
    
    if not fail_to_pass_tests and not pass_to_pass_tests:
        logger.info(f"No validation tests specified for instance {instance_id}")
        return None
    
    logger.info(f"🔍 Validating code changes for instance {instance_id}")
    logger.info(f"   FAIL_TO_PASS tests: {len(fail_to_pass_tests)}")
    logger.info(f"   PASS_TO_PASS tests: {len(pass_to_pass_tests)}")
    
    validation_results = {
        "validation_time": datetime.now().isoformat(),
        "fail_to_pass_results": {},
        "pass_to_pass_results": {},
        "summary": {}
    }
    
    try:
        # Create bash executor tool for running tests
        bash_tool = create_tool("R2EBashExecutor", k8s_config.copy())
        
        # Test FAIL_TO_PASS tests (should pass after fix)
        if fail_to_pass_tests:
            logger.info(f"Running FAIL_TO_PASS tests...")
            fail_to_pass_results = {}
            
            for test_name in fail_to_pass_tests:
                try:
                    # Run the specific test
                    test_command = f"cd /testbed && python -m pytest {test_name} -v"
                    logger.info(f"Running test: {test_name}")
                    
                    result = await bash_tool.execute_tool(
                        instance_id=f"validation_{instance_id}",
                        parameters={"command": test_command}
                    )
                    
                    if result.success:
                        test_output = result.result.get("stdout", "")
                        exit_code = result.result.get("return_code", -1)
                        
                        # Check if test passed (exit code 0)
                        test_passed = exit_code == 0
                        fail_to_pass_results[test_name] = {
                            "passed": test_passed,
                            "exit_code": exit_code,
                            "output": test_output[:1000] + "..." if len(test_output) > 1000 else test_output
                        }
                        
                        logger.info(f"   {test_name}: {'✅ PASSED' if test_passed else '❌ FAILED'}")
                    else:
                        fail_to_pass_results[test_name] = {
                            "passed": False,
                            "exit_code": -1,
                            "output": f"Tool execution failed: {result.error}"
                        }
                        logger.error(f"   {test_name}: ❌ TOOL_ERROR")
                        
                except Exception as e:
                    fail_to_pass_results[test_name] = {
                        "passed": False,
                        "exit_code": -1,
                        "output": f"Exception: {str(e)}"
                    }
                    logger.error(f"   {test_name}: ❌ EXCEPTION - {e}")
            
            validation_results["fail_to_pass_results"] = fail_to_pass_results
        
        # Test PASS_TO_PASS tests (should continue to pass)
        if pass_to_pass_tests:
            logger.info(f"Running PASS_TO_PASS tests...")
            pass_to_pass_results = {}
            
            for test_name in pass_to_pass_tests:
                try:
                    # Run the specific test
                    test_command = f"cd /testbed && python -m pytest {test_name} -v"
                    logger.info(f"Running test: {test_name}")
                    
                    result = await bash_tool.execute_tool(
                        instance_id=f"validation_{instance_id}",
                        parameters={"command": test_command}
                    )
                    
                    if result.success:
                        test_output = result.result.get("stdout", "")
                        exit_code = result.result.get("return_code", -1)
                        
                        # Check if test passed (exit code 0)
                        test_passed = exit_code == 0
                        pass_to_pass_results[test_name] = {
                            "passed": test_passed,
                            "exit_code": exit_code,
                            "output": test_output[:1000] + "..." if len(test_output) > 1000 else test_output
                        }
                        
                        logger.info(f"   {test_name}: {'✅ PASSED' if test_passed else '❌ FAILED'}")
                    else:
                        pass_to_pass_results[test_name] = {
                            "passed": False,
                            "exit_code": -1,
                            "output": f"Tool execution failed: {result.error}"
                        }
                        logger.error(f"   {test_name}: ❌ TOOL_ERROR")
                        
                except Exception as e:
                    pass_to_pass_results[test_name] = {
                        "passed": False,
                        "exit_code": -1,
                        "output": f"Exception: {str(e)}"
                    }
                    logger.error(f"   {test_name}: ❌ EXCEPTION - {e}")
            
            validation_results["pass_to_pass_results"] = pass_to_pass_results
        
        # Calculate summary statistics
        summary = {}
        
        if fail_to_pass_tests:
            fail_to_pass_passed = sum(1 for r in validation_results["fail_to_pass_results"].values() if r["passed"])
            fail_to_pass_total = len(validation_results["fail_to_pass_results"])
            summary["fail_to_pass"] = {
                "passed": fail_to_pass_passed,
                "total": fail_to_pass_total,
                "success_rate": fail_to_pass_passed / fail_to_pass_total * 100 if fail_to_pass_total > 0 else 0
            }
        
        if pass_to_pass_tests:
            pass_to_pass_passed = sum(1 for r in validation_results["pass_to_pass_results"].values() if r["passed"])
            pass_to_pass_total = len(validation_results["pass_to_pass_results"])
            summary["pass_to_pass"] = {
                "passed": pass_to_pass_passed,
                "total": pass_to_pass_total,
                "success_rate": pass_to_pass_passed / pass_to_pass_total * 100 if pass_to_pass_total > 0 else 0
            }
        
        validation_results["summary"] = summary
        
        # Save validation results to file (this will be called from process_single_instance)
        # The validation results will be included in the metadata.json file
        logger.info(f"📝 Validation results will be saved in metadata.json")
        
        # Log summary
        logger.info(f"🔍 Validation Summary for {instance_id}:")
        if fail_to_pass_tests:
            logger.info(f"   FAIL_TO_PASS: {summary['fail_to_pass']['passed']}/{summary['fail_to_pass']['total']} ({summary['fail_to_pass']['success_rate']:.1f}%)")
        if pass_to_pass_tests:
            logger.info(f"   PASS_TO_PASS: {summary['pass_to_pass']['passed']}/{summary['pass_to_pass']['total']} ({summary['pass_to_pass']['success_rate']:.1f}%)")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"❌ Validation failed for instance {instance_id}: {e}")
        error_result = {
            "validation_time": datetime.now().isoformat(),
            "error": str(e),
            "summary": {"error": True}
        }
        
        # Save error to file
        validation_file = os.path.join(instance_output_dir, "validation_error.json")
        with open(validation_file, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        return error_result


async def process_instances_from_json(
    json_file_path: str, 
    output_dir: str = "./trajectories",
    start_index: int = 0,
    max_instances: Optional[int] = None
):
    """Process instances from JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing instances
        output_dir: Base directory to save trajectory files
        start_index: Starting index for processing (for resuming)
        max_instances: Maximum number of instances to process (None for all)
    """
    print("\n" + "="*80)
    print("🚀 Processing Instances from JSON File")
    print("="*80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n📁 Base output directory: {output_dir}")
    
    # Load instances from JSON file
    instances = load_instances_from_json(json_file_path)
    
    # Apply start_index and max_instances filters
    if start_index > 0:
        instances = instances[start_index:]
        print(f"Starting from index {start_index}")
    
    if max_instances is not None:
        instances = instances[:max_instances]
        print(f"Processing maximum {max_instances} instances")
    
    print(f"📊 Total instances to process: {len(instances)}")
    
    # K8S configuration (pod_name will be set dynamically for each instance)
    k8s_config = {
        "execution_mode": "k8s",
        "namespace": "xxx",
        "kubeconfig_path": "xxx"
    }
    
    print(f"🔧 K8S Configuration:")
    print(f"   Pod: Dynamic (based on instance_id)")
    print(f"   Namespace: {k8s_config['namespace']}")
    print(f"   Kubeconfig: {k8s_config.get('kubeconfig_path') or 'default'}")
    
    print(f"🤖 LLM Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Base URL: {BASE_URL}")
    
    # Process each instance
    successful_count = 0
    failed_count = 0
    
    for i, instance in enumerate(instances, start=start_index + 1):
        print(f"\n{'='*60}")
        print(f"Processing instance {i}/{len(instances) + start_index}")
        print(f"{'='*60}")
        
        try:
            success = await process_single_instance(instance, output_dir, k8s_config)
            if success:
                successful_count += 1
            else:
                failed_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing instance {i}: {e}")
            failed_count += 1
        
        # Add a small delay between instances to avoid overwhelming the system
        await asyncio.sleep(2)
    
    # Collect actual validation statistics from saved files
    print("\n🔍 Collecting validation statistics from saved files...")
    validation_stats = collect_validation_statistics(output_dir)
    
    # Create summary
    print("\n" + "="*80)
    print("🎉 Processing completed!")
    print("="*80)
    
    summary = {
        "processing_time": datetime.now().isoformat(),
        "json_file": json_file_path,
        "output_directory": output_dir,
        "total_instances": len(instances),
        "successful_count": successful_count,
        "failed_count": failed_count,
        "start_index": start_index,
        "max_instances": max_instances,
        "validation_statistics": validation_stats,
        "configuration": {
            "model": MODEL_NAME,
            "k8s_pod": "Dynamic (based on instance_id)",
            "k8s_namespace": k8s_config["namespace"]
        }
    }
    
    summary_file = os.path.join(output_dir, "processing_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary:")
    print(f"   Total instances: {len(instances)}")
    print(f"   Successful: {successful_count}")
    print(f"   Failed: {failed_count}")
    print(f"   Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%" if (successful_count+failed_count) > 0 else "   Success rate: N/A")
    
    if validation_stats["total_instances_with_validation"] > 0:
        print(f"🔍 Validation Statistics:")
        print(f"   Instances with validation: {validation_stats['total_instances_with_validation']}")
        print(f"   All tests passed (success): {validation_stats['validation_success_count']}")
        print(f"   Some tests failed (failure): {validation_stats['validation_failure_count']}")
        if validation_stats['validation_success_count'] + validation_stats['validation_failure_count'] > 0:
            success_rate = validation_stats['validation_success_count'] / (validation_stats['validation_success_count'] + validation_stats['validation_failure_count']) * 100
            print(f"   Test success rate: {success_rate:.1f}%")
    
    print(f"📝 Summary saved to: {summary_file}")
    print(f"📁 All trajectories saved to: {output_dir}")


def collect_validation_statistics(output_dir: str) -> Dict[str, Any]:
    """Collect validation statistics from saved trajectory files.
    
    Args:
        output_dir: Base output directory containing instance folders
        
    Returns:
        Dictionary with validation statistics
    """
    validation_stats = {
        "total_instances_with_validation": 0,
        "validation_success_count": 0,
        "validation_failure_count": 0
    }
    
    try:
        # Iterate through all instance directories
        for item in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item)
            if not os.path.isdir(item_path):
                continue
                
            # Look for trajectory subdirectories (e.g., "0-success", "1-failure")
            has_validation = False
            instance_success_count = 0
            instance_failure_count = 0
            
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if not os.path.isdir(subitem_path):
                    continue
                    
                # Check if this is a trajectory subdirectory
                if re.match(r'\d+-(success|failure)', subitem):
                    has_validation = True
                    
                    # Check if there's a metadata.json file
                    metadata_file = os.path.join(subitem_path, "metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Check the trajectory_status
                            status = metadata.get("trajectory_status", "unknown")
                            if status == "success":
                                instance_success_count += 1
                            elif status == "failure":
                                instance_failure_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
            
            if has_validation:
                validation_stats["total_instances_with_validation"] += 1
                validation_stats["validation_success_count"] += instance_success_count
                validation_stats["validation_failure_count"] += instance_failure_count
                
    except Exception as e:
        logger.error(f"Error collecting validation statistics: {e}")
    
    return validation_stats


async def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(description="Test R2E GeneralAgent with trajectory saving")
    parser.add_argument("--mode", choices=["test", "process"], default="test",
                       help="Mode: 'test' for original test tasks, 'process' for processing JSON instances")
    parser.add_argument("--json-file", type=str,
                       help="Path to JSON file containing instances (required for 'process' mode)")
    parser.add_argument("--output-dir", default="./trajectories", 
                       help="Directory to save trajectory files (default: ./trajectories)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Starting index for processing instances (default: 0)")
    parser.add_argument("--max-instances", type=int,
                       help="Maximum number of instances to process (default: all)")
    
    args = parser.parse_args()
    
    print("🧪 R2E GeneralAgent Test Suite")
    print(f"📁 Output directory: {args.output_dir}")
    
    if args.mode == "process":
        if not args.json_file:
            print("❌ Error: --json-file is required for 'process' mode")
            sys.exit(1)
        
        print(f"Running in process mode")
        print(f"JSON file: {args.json_file}")
        print(f"Start index: {args.start_index}")
        if args.max_instances:
            print(f"Max instances: {args.max_instances}")
        
        await process_instances_from_json(
            json_file_path=args.json_file,
            output_dir=args.output_dir,
            start_index=args.start_index,
            max_instances=args.max_instances
        )
    
    print("\n" + "="*80)
    print("✅ All operations completed!")
    print(f"📁 Check {args.output_dir} for saved trajectories")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
