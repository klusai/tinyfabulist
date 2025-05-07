import os
import glob
import yaml
from collections import defaultdict
import subprocess
import time
import datetime
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable, Any

from tinyfabulist.logger import *

# Constants
CONFIG_DIR = "conf"

logger = setup_logging()

def load_jsonl_entries(file_path: str) -> List[Dict]:
    """
    Load entries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the entries
    """
    entries = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON line in {file_path}: {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise
    
    return entries

def process_file_or_directory(input_path: str, process_func: Callable, file_pattern: str = None, output_dir: str = None) -> None:
    """
    Process either a single file or all matching files in a directory.
    
    Args:
        input_path: Path to a file or directory
        process_func: Function to process each file
        file_pattern: Optional pattern to match files (e.g., "*.jsonl")
        output_dir: Optional output directory for results
    """
    if os.path.isfile(input_path):
        # Process single file
        process_func(input_path, output_dir)
    elif os.path.isdir(input_path):
        # Process all matching files in directory
        if file_pattern:
            files = glob.glob(os.path.join(input_path, file_pattern))
        else:
            files = glob.glob(os.path.join(input_path, "*"))
        
        for file_path in files:
            if os.path.isfile(file_path):
                process_func(file_path, output_dir)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

class EvaluationUtils:
    """Utility class for handling fable evaluations."""
    
    def __init__(self, language: str = "en"):
        self.language = language
        self.settings = load_settings()
        self.evaluator_config = self.settings.get("evaluator", {})
        self.lock = threading.Lock()
    
    def get_prompts(self) -> tuple:
        """Get the system prompt and evaluation prompt template."""
        prompts = self.evaluator_config.get("prompts", {})
        return prompts.get("system"), prompts.get("evaluation")
    
    def render_template(self, template: str, context: Dict) -> str:
        """Render a template with the given context."""
        try:
            return template.format(**context)
        except KeyError as e:
            logger.error(f"Missing key in template context: {e}")
            return template
    
    def retry_operation(self, operation: Callable, max_retries: int = 3, **kwargs) -> Dict:
        """Retry an operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return operation(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    return {"error": str(e)}
                time.sleep(2 ** attempt)
        return {"error": "Max retries exceeded"}
    
    def call_evaluation_api(self, system_prompt: str, user_prompt: str) -> Dict:
        """Call the evaluation API with the given prompts."""
        # This is a placeholder - the actual implementation would use the OpenAI API
        # or whatever API is configured for evaluation
        return {"error": "API call not implemented"}
    
    def process_entries(self, entries: List[Dict], process_func: Callable, 
                       max_workers: int = 25, **kwargs) -> List[Dict]:
        """Process entries in parallel using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, entry in enumerate(entries):
                future = executor.submit(process_func, entry, i, **kwargs)
                futures.append(future)
            
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results
    
    def create_output_path(self, input_path: str, output_dir: Optional[str] = None) -> str:
        """Create an output path for evaluation results."""
        if output_dir is None:
            output_dir = os.path.join("data", "evaluations")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Create timestamp
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        
        # Create output filename
        output_filename = f"eval_{base_name}_{timestamp}.jsonl"
        
        return os.path.join(output_dir, output_filename)
    
    def save_results(self, results: List[Dict], output_path: str) -> None:
        """Save evaluation results to a JSONL file."""
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

def deep_update(source, update):
    """
    Recursively update nested dictionaries
    """
    for key, value in update.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            source[key] = deep_update(source[key], value)
        else:
            source[key] = value
    return source

def load_settings() -> dict:
    """
    Loads settings from all YAML files in the conf directory.
    Returns a dictionary containing all configuration settings merged together.
    Raises ConfigError if the directory is not found or files have invalid YAML format.
    """
    settings = {}
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_DIR)
    
    if not os.path.exists(config_path):
        logger.error(f"Configuration directory '{config_path}' not found")
        raise ConfigError(f"Configuration directory '{config_path}' not found")
    
    config_files = glob.glob(os.path.join(config_path, "*.yaml"))
    
    if not config_files:
        logger.error(f"No YAML configuration files found in '{config_path}'")
        raise ConfigError(f"No YAML configuration files found in '{config_path}'")
    
    for config_file in sorted(config_files):
        try:
            with open(config_file, "r") as file:
                file_settings = yaml.safe_load(file)
                if file_settings:
                    settings = deep_update(settings, file_settings)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise ConfigError(f"Invalid YAML format in {os.path.basename(config_file)}: {e}")
    
    logger.info(f"Successfully loaded settings from {len(config_files)} configuration files")
    return settings

def load_specific_config(config_name):
    """
    Loads a specific configuration file from the conf directory.
    Args:
        config_name: Name of the configuration file (without .yaml extension)
    Returns:
        Dictionary containing the configuration settings
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG_DIR)
    config_file = os.path.join(config_path, f"{config_name}.yaml")
    
    try:
        with open(config_file, "r") as file:
            settings = yaml.safe_load(file)
            logger.info(f"Loaded specific configuration from {config_name}.yaml")
            return settings
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_name}.yaml' not found")
        raise ConfigError(f"Configuration file '{config_name}.yaml' not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {config_file}: {e}")
        raise ConfigError(f"Invalid YAML format in {config_name}.yaml: {e}")

def get_major_version() -> str:
    """
    Read the major version from the version.yaml config file.
    Returns the major version as a string, defaulting to "0.1" if not found.
    """
    try:
        version_config = load_specific_config("version")
        return version_config.get("version", {}).get("major", "0.1")
    except ConfigError as e:
        logger.warning(f"Could not load version config: {e}. Using default major version 0.1")
        return "0.1"

def get_version_info():
    """
    Calculate the current version of the package in the format:
    major_version.nr_of_commits_on_main.date (e.g., 0.1.42.230512)
    
    Returns:
        dict: A dictionary containing version info with the following keys:
            - 'version': The full version string
            - 'major_version': The manually set major version
            - 'commit_count': The number of commits on main
            - 'date': The date in YYMMDD format
            - 'last_commit_hash': The hash of the last commit
            - 'last_commit_msg': The message of the last commit
    """
    # Get major version from config
    major_version = get_major_version()
    
    try:
        # Get number of commits on main
        commit_count = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        # Get the date in YYMMDD format
        date_str = datetime.datetime.now().strftime("%y%m%d")
        
        # Get the last commit hash and message
        last_commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        last_commit_msg = subprocess.check_output(
            ["git", "log", "-1", "--pretty=%B"],
            stderr=subprocess.STDOUT
        ).decode('utf-8').strip()
        
        # Build the version string
        version = f"{major_version}.{commit_count}.{date_str}"
        
        return {
            'version': version,
            'major_version': major_version,
            'commit_count': commit_count,
            'date': date_str,
            'last_commit_hash': last_commit_hash,
            'last_commit_msg': last_commit_msg
        }
    except subprocess.CalledProcessError:
        # If git commands fail (e.g., not a git repo), return fallback version
        date_str = datetime.datetime.now().strftime("%y%m%d")
        return {
            'version': f"{major_version}.0.{date_str}",
            'major_version': major_version,
            'commit_count': "0",
            'date': date_str,
            'last_commit_hash': "unknown",
            'last_commit_msg': "unknown"
        }

def update_changelog(version_info=None):
    """
    Update the CHANGELOG.md file with the latest version information.
    
    Args:
        version_info (dict, optional): Version info dictionary. If None, it will be calculated.
        
    Returns:
        str: The path to the changelog file
    """
    if version_info is None:
        version_info = get_version_info()
    
    changelog_path = Path("CHANGELOG.md")
    
    # Create changelog file if it doesn't exist
    if not changelog_path.exists():
        with open(changelog_path, "w") as f:
            f.write("# Changelog\n\n")
            f.write("All notable changes to this project will be documented in this file.\n\n")
    
    # Read existing changelog
    with open(changelog_path, "r") as f:
        content = f.read()
    
    # Check if this version is already in the changelog
    version_header = f"## [{version_info['version']}]"
    if version_header in content:
        return str(changelog_path)
    
    # Get the timestamp in ISO format
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Build the new entry
    new_entry = f"{version_header} - {timestamp}\n\n"
    
    # Add commit information
    if version_info['last_commit_hash'] != "unknown":
        new_entry += f"- Commit: {version_info['last_commit_hash']} - {version_info['last_commit_msg']}\n\n"
    
    # Find the position to insert the new entry (after the header and description)
    lines = content.split("\n")
    insert_pos = 0
    
    for i, line in enumerate(lines):
        if line.startswith("## ["):
            insert_pos = i
            break
        elif i > 5:  # If we've checked several lines and not found a version header
            insert_pos = i
    
    # Insert the new entry
    if insert_pos > 0:
        updated_content = "\n".join(lines[:insert_pos]) + "\n" + new_entry + "\n".join(lines[insert_pos:])
    else:
        updated_content = content + "\n" + new_entry
    
    # Write updated changelog
    with open(changelog_path, "w") as f:
        f.write(updated_content)
    
    return str(changelog_path)