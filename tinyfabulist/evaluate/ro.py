import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from tinyfabulist.evaluate.utils import EvaluationUtils, load_jsonl_entries, process_file_or_directory
from tinyfabulist.logger import setup_logging

logger = setup_logging()
utils = EvaluationUtils(language="ro")


def evaluate_fable_save(fable_data: dict, idx: int = 0, lock: threading.Lock = None) -> dict:
    """
    Evaluates a translated fable and returns the evaluation result
    along with the original data.
    """
    if not fable_data:
        return None
    
    fable_text = fable_data.get("fable", "")
    translated_fable_text = fable_data.get("translated_fable", "")
    
    # Skip entries without both original and translated fables
    if not fable_text or not translated_fable_text:
        logger.warning(f"Skipping entry {idx+1}: Missing fable or translation")
        return None
    
    # Get prompts with thread safety if lock provided
    context = {"original_fable": fable_text, "translated_fable": translated_fable_text}
    if lock:
        with lock:
            system_prompt, evaluation_template = utils.get_prompts()
            evaluation_prompt = utils.render_template(evaluation_template, context)
    else:
        system_prompt, evaluation_template = utils.get_prompts() 
        evaluation_prompt = utils.render_template(evaluation_template, context)
    
    # Call the evaluation API
    evaluation_json = utils.call_evaluation_api(system_prompt, evaluation_prompt)
    
    # Merge the original fable data with the evaluation results
    result = dict(fable_data)
    result.update({
        "evaluation": evaluation_json,
        "evaluation_timestamp": datetime.now().isoformat(),
    })
    
    return result


def evaluate_file(input_path: str, output_path: str = None) -> None:
    """
    Reads entries from an input file, evaluates them, and saves results.
    """
    logger.info(f"Evaluating file: {input_path}")
    entries = load_jsonl_entries(input_path)
    
    # Create a lock for thread-safe template compilation
    output_lock = threading.Lock()
    
    # Process entries in parallel
    results = utils.process_entries(
        entries,
        evaluate_fable_save, 
        max_workers=25,
        lock=output_lock
    )
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Create output path if not provided
    if output_path is None:
        output_path = utils.create_output_path(input_path)
    
    # Save results
    utils.save_results(results, output_path)
    logger.info(f"Evaluations have been completed and saved in {output_path}")


def run_evaluate(args) -> None:
    """Main entry point for Romanian evaluation"""
    logger.info("Running EN-RO Translation Evaluation")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join("data", "evaluations_ro")
    os.makedirs(output_base_dir, exist_ok=True)
    
    process_file_or_directory(
        args.input,
        evaluate_file,
        file_pattern=".jsonl"  # Process all JSONL files
    )


def add_evaluate_ro_subparser(subparsers) -> None:
    """Add the Romanian evaluate subparser to the main parser"""
    eval_parser = subparsers.add_parser(
        "evaluate_ro",
        help="Evaluate translations of fables from Romanian to English",
    )
    eval_parser.add_argument(
        "--input",
        type=str,
        help="Path to a JSONL file or directory containing translated fables",
        required=True,
    )
    eval_parser.set_defaults(func=run_evaluate)
