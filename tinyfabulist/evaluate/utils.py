import json
import os
import sys
import time
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple

from decouple import config
from dotenv import load_dotenv
from openai import OpenAI
from pybars import Compiler

from tinyfabulist.logger import setup_logging
from tinyfabulist.utils import load_settings

logger = setup_logging()


class EvaluationUtils:
    """
    Utility class that provides common functionality for fable evaluation
    across different languages.
    """
    
    def __init__(self, language: str = "en"):
        """
        Initialize the evaluation utilities with language-specific settings.
        
        Args:
            language: Two-letter language code ('en' or 'ro')
        """
        self.language = language
        self.settings = load_settings()
        self.evaluator_config = self.settings.get("evaluator", {})
        self.model = self.evaluator_config.get("model", "gpt-4o")
        
        # Get language-specific prompt keys
        self.system_prompt_key = f"system{'_' + language if language != 'en' else ''}"
        self.evaluation_prompt_key = f"evaluation{'_' + language if language != 'en' else ''}"
        
        # Initialize template compiler
        self.compiler = Compiler()
    
    def get_prompts(self) -> Tuple[str, str]:
        """
        Get language-specific system and evaluation prompts.
        
        Returns:
            Tuple containing (system_prompt, evaluation_prompt_template)
        """
        prompts = self.evaluator_config.get("prompt", {})
        system_prompt = prompts.get(self.system_prompt_key, "")
        evaluation_prompt_template = prompts.get(self.evaluation_prompt_key, "")
        return system_prompt, evaluation_prompt_template
    
    def render_template(self, template_str: str, context: Dict[str, Any]) -> str:
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
    
    def call_evaluation_api(self, system_prompt: str, user_prompt: str) -> Dict:
        """
        Call the evaluation API with the provided prompts.
        
        Args:
            system_prompt: The system prompt to send to the API
            user_prompt: The user prompt to send to the API
            
        Returns:
            Dictionary containing the parsed JSON response or an error
        """
        load_dotenv()
        client = OpenAI(api_key=config("OPENAI_API_KEY"))
        
        try:
            chat_completion = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                reasoning_effort="high",
            )
            evaluation_text = chat_completion.choices[0].message.content.strip()
            
            try:
                evaluation_json = json.loads(evaluation_text)
                return evaluation_json
            except json.JSONDecodeError as json_err:
                error_msg = f"JSON decoding error: {str(json_err)}. Raw response: {evaluation_text[:200]}"
                logger.error(error_msg)
                self.save_debug_response(evaluation_text, error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = f"API error: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}
    
    def save_debug_response(self, text: str, error: str) -> None:
        """
        Save a problematic API response for debugging.
        
        Args:
            text: The API response text
            error: The error message
        """
        debug_dir = os.path.join("tinyfabulist", "data", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create a unique filename
        filename = f"api_response_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
        filepath = os.path.join(debug_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(f"ERROR: {error}\n\n")
            f.write("API RESPONSE:\n")
            f.write(text)
        
        logger.info(f"Saved problematic API response to {filepath}")
    
    def process_entries(self, 
                       entries: List[Dict], 
                       process_func: Callable, 
                       max_workers: int = 25, 
                       **kwargs) -> List[Dict]:
        """
        Process multiple entries in parallel using ThreadPoolExecutor.
        
        Args:
            entries: List of entries to process
            process_func: Function to apply to each entry
            max_workers: Maximum number of worker threads
            **kwargs: Additional arguments to pass to process_func
            
        Returns:
            List of processed entries
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i, entry in enumerate(entries):
                if entry:
                    futures.append(executor.submit(process_func, entry, i, **kwargs))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict], output_path: str) -> None:
        """
        Save results to a JSONL file.
        
        Args:
            results: List of results to save
            output_path: Path to the output file
        """
        with open(output_path, "w", encoding="utf-8") as outfile:
            for result in results:
                outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        logger.info(f"Results saved to {output_path}")
    
    def create_output_path(self, input_path: str, base_dir: str = None) -> str:
        """
        Create an output path for evaluation results.
        
        Args:
            input_path: Path to the input file
            base_dir: Base directory for output (defaults to language-specific dir)
            
        Returns:
            Path to the output file
        """
        if base_dir is None:
            base_dir = os.path.join("tinyfabulist", "data", f"evaluations{'_' + self.language if self.language != 'en' else ''}")
        
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if os.path.isfile(input_path):
            basename = os.path.basename(input_path)
            return os.path.join(base_dir, f"evaluations_eval_e_{self.model}_{timestamp}_{basename}")
        else:
            # This shouldn't happen in normal operations but provides a fallback
            return os.path.join(base_dir, f"evaluations_eval_e_{self.model}_{timestamp}.jsonl")
    
    def get_original_prompt(self) -> str:
        """
        Get the original prompt template from the generator configuration.
        
        Returns:
            The full prompt template used for generating fables
        """
        generator_config = self.settings.get("generator", {})
        prompt_config = generator_config.get("prompt", {})
        
        system_prompt = prompt_config.get("system", "")
        fable_prompt = prompt_config.get("fable", "")
        
        combined_prompt = f"System Prompt:\n{system_prompt}\n\nFable Prompt:\n{fable_prompt}"
        return combined_prompt
    
    def retry_operation(self, 
                        operation: Callable, 
                        max_attempts: int = 5, 
                        delay: int = 1, 
                        **kwargs) -> Any:
        """
        Retry an operation multiple times before giving up.
        
        Args:
            operation: Function to retry
            max_attempts: Maximum number of retry attempts
            delay: Delay between retries in seconds
            **kwargs: Arguments to pass to the operation
            
        Returns:
            Result of the operation or error information
        """
        for attempt in range(1, max_attempts + 1):
            try:
                result = operation(**kwargs)
                if isinstance(result, dict) and "error" not in result:
                    return result
                else:
                    logger.error(f"Error on attempt {attempt}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Exception on attempt {attempt}: {e}")
            
            if attempt < max_attempts:
                time.sleep(delay)
        
        error_msg = f"Operation failed after {max_attempts} attempts"
        logger.error(error_msg)
        return {"error": error_msg}


# Helper functions for common evaluation patterns

def load_jsonl_entries(file_path: str) -> List[Dict]:
    """Load entries from a JSONL file."""
    entries = []
    try:
        with open(file_path, "r", encoding="utf-8") as infile:
            for line in infile:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        entries.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON line in {file_path}")
    except Exception as e:
        logger.error(f"Error loading entries from {file_path}: {e}")
    
    return entries


def process_file_or_directory(input_path: str, 
                            process_file_func: Callable, 
                            file_pattern: str = None,
                            **kwargs) -> None:
    """
    Process a file or all files in a directory.
    
    Args:
        input_path: Path to file or directory
        process_file_func: Function to process each file
        file_pattern: Pattern to match files in directory (e.g., "tf_fables")
        **kwargs: Additional arguments to pass to process_file_func
    """
    if os.path.isfile(input_path):
        # Process a single file
        process_file_func(input_path, **kwargs)
    
    elif os.path.isdir(input_path):
        # Process files in directory
        for root, _, files in os.walk(input_path):
            for file in files:
                if file_pattern is None or (file_pattern in file and file.endswith(".jsonl")):
                    file_path = os.path.join(root, file)
                    process_file_func(file_path, **kwargs)
    
    else:
        logger.error(f"Path does not exist: {input_path}")
        sys.exit(1)