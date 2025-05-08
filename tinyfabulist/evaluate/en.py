import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict

from tinyfabulist.evaluate.utils import EvaluationUtils, load_jsonl_entries, process_file_or_directory
from tinyfabulist.logger import setup_logging

logger = setup_logging()
utils = EvaluationUtils(language="en")


def evaluate_fable_threaded(entry: Dict, index: int, **kwargs) -> Dict:
    """
    Thread-safe wrapper for evaluating a single fable.
    
    Args:
        entry: The entry containing the fable to evaluate
        index: Index of the entry in the batch
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing the evaluation results
    """
    try:
        # Get the lock from kwargs
        lock = kwargs.get("lock")
        
        # Get prompts
        system_prompt, evaluation_prompt_template = utils.get_prompts()
        
        # Prepare context for template
        context = {
            "fable": entry.get("fable", ""),
            "moral": entry.get("moral", ""),
            "age_group": entry.get("age_group", ""),
            "language": entry.get("language", "en"),
            "prompt": entry.get("prompt", "")  # Add the original prompt
        }
        
        # Render the evaluation prompt
        user_prompt = utils.render_template(evaluation_prompt_template, context)
        
        # Call the API with retries
        evaluation = utils.retry_operation(
            utils.call_evaluation_api,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
        
        if "error" in evaluation:
            logger.error(f"Error evaluating fable {index}: {evaluation['error']}")
            return None
        
        # Keep all original fields and add evaluation results
        result = entry.copy()  # This preserves all original fields
        result["evaluation"] = evaluation
        result["input_file"] = kwargs.get("input_file", "")  # Add input file path
        result["pipeline_stage"] = "evaluation"  # Add pipeline stage
        
        # Ensure these fields exist (they should be in the original entry, but just in case)
        if "hash" not in result:
            result["hash"] = entry.get("hash", "")
        if "llm_input_tokens" not in result:
            result["llm_input_tokens"] = entry.get("llm_input_tokens", 0)
        if "llm_output_tokens" not in result:
            result["llm_output_tokens"] = entry.get("llm_output_tokens", 0)
        if "llm_inference_time" not in result:
            result["llm_inference_time"] = entry.get("llm_inference_time", 0)
        if "host_provider" not in result:
            result["host_provider"] = entry.get("host_provider", "unknown")
        if "host_gpu" not in result:
            result["host_gpu"] = entry.get("host_gpu", "unknown")
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing fable {index}: {e}")
        return None


def evaluate_file(file_path: str, output_dir: str = None) -> None:
    """
    Reads the JSONL file at file_path, evaluates each fable (if present),
    and saves the results to an output file.
    """
    logger.info(f"Evaluating file: {file_path}")
    entries = load_jsonl_entries(file_path)
    
    # Filter entries to only include those with fables
    fables_to_evaluate = [entry for entry in entries if "fable" in entry]
    logger.info(f"Found {len(fables_to_evaluate)} fables to evaluate")
    
    # Process entries in parallel
    results = utils.process_entries(
        fables_to_evaluate, 
        evaluate_fable_threaded, 
        max_workers=25,
        input_file=file_path  # Pass the input file path
    )
    
    # Create output path using the standardized function
    output_path = utils.create_output_path(file_path, output_dir)
    
    # Save results
    utils.save_results(results, output_path)
    logger.info(f"Evaluations saved to {output_path}")


def run_evaluate(args) -> None:
    """
    Process the input path, evaluating fables in files/directories.
    """
    # Process files or directories
    process_file_or_directory(
        args.input, 
        evaluate_file, 
        file_pattern="tf_fables"
    )


def add_evaluate_subparser(subparsers) -> None:
    """Add the evaluate subparser to the main parser"""
    eval_parser = subparsers.add_parser(
        "evaluate",
        help='Evaluate generated fables from a JSONL file or a directory containing files starting with "tf_fables"',
    )
    eval_parser.add_argument(
        "--input",
        type=str,
        help="Path to a JSONL file or a directory to evaluate fables from",
        required=True,
    )
    eval_parser.set_defaults(func=run_evaluate)
