import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from tinyfabulist.evaluate.utils import EvaluationUtils, load_jsonl_entries, process_file_or_directory
from tinyfabulist.logger import setup_logging

logger = setup_logging()
utils = EvaluationUtils(language="en")


def evaluate_fable_threaded(fable_data: dict, idx: int = 0) -> dict:
    """
    Worker function to evaluate a single fable.
    Returns the original data with evaluation results added.
    """
    llm_name = fable_data.get("llm_name", "unknown")
    hash_value = fable_data.get("hash", "")
    fable_text = fable_data.get("fable", "")
    prompt = fable_data.get("prompt", "")
    llm_input_tokens = fable_data.get("llm_input_tokens", 0)
    llm_output_tokens = fable_data.get("llm_output_tokens", 0)
    llm_inference_time = fable_data.get("llm_inference_time", 0)
    host_provider = fable_data.get("host_provider", "unknown")
    host_gpu = fable_data.get("host_gpu", "unknown")
    
    # Define the evaluation operation
    def evaluate_operation():
        system_prompt, evaluation_template = utils.get_prompts()
        evaluation_prompt = utils.render_template(
            evaluation_template, 
            {"fable": fable_text, "prompt": prompt}
        )
        return utils.call_evaluation_api(system_prompt, evaluation_prompt)
    
    # Perform the evaluation with retries
    evaluation = utils.retry_operation(evaluate_operation)
    
    return {
        "llm_name": llm_name,
        "evaluation": evaluation,
        "hash": hash_value,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "llm_inference_time": llm_inference_time,
        "host_provider": host_provider,
        "host_gpu": host_gpu,
    }


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
        max_workers=25
    )
    
    # Create output path
    if output_dir is None:
        output_path = utils.create_output_path(file_path)
    else:
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        output_path = os.path.join(
            output_dir, f"{base_name}_jsonl_eval_e{utils.model}_dt{timestamp}.jsonl"
        )
    
    # Save results
    utils.save_results(results, output_path)
    logger.info(f"Evaluations for {file_path} written to {output_path}")


def run_evaluate(args) -> None:
    """
    Process the input path, evaluating fables in files/directories.
    """
    output_dir = os.path.join("data", "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process files or directories
    process_file_or_directory(
        args.input, 
        evaluate_file, 
        file_pattern="tf_fables", 
        output_dir=output_dir
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
