import os
import glob
import yaml
from collections import defaultdict
import subprocess
import time
import datetime
from pathlib import Path
from pybars import Compiler
import json
import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from tinyfabulist.logger import *

# Constants
CONFIG_DIR = "tinyfabulist/conf"

logger = setup_logging()

class ConfigError(Exception):
    """Exception raised for configuration errors."""
    pass

class EvaluationUtils:
    def __init__(self, language="en"):
        self.language = language
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.compiler = Compiler()
    def get_prompts(self):
        """Get the system prompt and evaluation prompt template for the current language."""
        config = load_specific_config("evaluator")
        if self.language == "en":
            return config["evaluator"]["prompt"]["system"], config["evaluator"]["prompt"]["evaluation"]
        else:
            return config["evaluator"]["prompt"]["system_ro"], config["evaluator"]["prompt"]["evaluation_ro"]
    
    def render_template(self, template_str: str, context: dict[str, any]) -> str:
        """
        Render a template with the provided context.
        
        Args:
            template_str: The template string to render
            context: Dictionary of values to use in the template
            
        Returns:
            The rendered template string
        """
        template = self.compiler.compile(template_str)
        return template(context)
    
    def call_evaluation_api(self, system_prompt, user_prompt):
        """Call the evaluation API with the given prompts."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # or whatever model is configured
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Error calling evaluation API: {e}")
            return {"error": str(e)}
    
    def retry_operation(self, operation, max_retries=3, **kwargs):
        """Retry an operation with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return operation(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Operation failed after {max_retries} attempts: {e}")
                    return {"error": str(e)}
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def process_entries(self, entries, process_func, max_workers=25, **kwargs):
        """Process entries in parallel using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, entry, i, **kwargs) 
                      for i, entry in enumerate(entries)]
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
        return results
    
    def create_output_path(self, input_path, output_dir=None):
        """Create an output path for evaluation results."""
        # Always use data/evaluations as the output directory
        output_dir = os.path.join("data", "evaluations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get the base name from the input path
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
        return os.path.join(output_dir, f"Evaluation_{timestamp}.jsonl")
    
    def save_results(self, results, output_path):
        """Save evaluation results to a JSONL file."""
        with open(output_path, "w", encoding="utf-8") as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write("\n")
    
    @staticmethod
    def get_readability(text):
        """Calculate Flesch Reading Ease score for a text."""
        return textstat.flesch_reading_ease(text)
    
    @staticmethod
    def distinct_n(text, n):
        """Calculate distinct-n score for a text."""
        tokens = nltk.word_tokenize(text.lower())
        ngrams_list = list(ngrams(tokens, n))
        if not ngrams_list:
            return 0
        return len(set(ngrams_list)) / len(ngrams_list)
    
    @staticmethod
    def compute_self_bleu(texts):
        """Compute self-BLEU scores for a list of texts."""
        if len(texts) < 2:
            return None, []
        
        bleu_scores = []
        for i, text in enumerate(texts):
            references = texts[:i] + texts[i+1:]
            if not references:
                continue
            
            hypothesis = nltk.word_tokenize(text.lower())
            references = [nltk.word_tokenize(ref.lower()) for ref in references]
            
            score = sentence_bleu(references, hypothesis, smoothing_function=SmoothingFunction().method1)
            bleu_scores.append(score)
        
        return sum(bleu_scores) / len(bleu_scores) if bleu_scores else None, bleu_scores

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
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), CONFIG_DIR)
    
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
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), CONFIG_DIR)
    config_file = os.path.join(config_path, f"{config_name}.yaml")

    try:
        with open(config_file, "r") as file:
            settings = yaml.safe_load(file)
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

def load_jsonl_entries(file_path):
    """
    Load entries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the entries
    """
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
    except Exception as e:
        logger.error(f"Error loading JSONL file {file_path}: {e}")
        raise

def process_file_or_directory(input_path, process_func, file_pattern=None, output_dir=None):
    """
    Process either a single file or all matching files in a directory.
    
    Args:
        input_path: Path to a file or directory
        process_func: Function to process each file
        file_pattern: Optional pattern to match files (e.g., "tf_fables")
        output_dir: Optional output directory for results
    """
    if os.path.isfile(input_path):
        # Process single file
        if file_pattern is None or file_pattern in os.path.basename(input_path):
            process_func(input_path, output_dir)
    elif os.path.isdir(input_path):
        # Process all matching files in directory
        for root, _, files in os.walk(input_path):
            for file in files:
                if file_pattern is None or file_pattern in file:
                    file_path = os.path.join(root, file)
                    process_func(file_path, output_dir)
    else:
        raise ConfigError(f"Input path does not exist: {input_path}")