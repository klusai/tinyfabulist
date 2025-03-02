import json
import sys
import os
from decouple import config
from openai import OpenAI
import yaml
from logger import *
from utils import load_settings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import uuid  # Add this import for generating unique filenames

logger = setup_logging()


def save_debug_response(text, error):
    """
    Saves a problematic API response to a file for debugging.

    Args:
        text (str): The API response text
        error (str): The error message
    """
    debug_dir = os.path.join("data", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Create a unique filename
    filename = f"api_response_{uuid.uuid4().hex[:8]}_{datetime.now().strftime('%y%m%d-%H%M%S')}.txt"
    filepath = os.path.join(debug_dir, filename)

    with open(filepath, "w") as f:
        f.write(f"ERROR: {error}\n\n")
        f.write("API RESPONSE:\n")
        f.write(text)

    logger.info(f"Saved problematic API response to {filepath}")


def evaluate_fable(fable: str, original_prompt: str = None) -> dict:
    """
    Evaluates a fable based on four specific criteria with detailed scoring guidelines:

    1. Grammar & Style (1-10): Assesses writing quality, language usage, and structural clarity
    2. Creativity & Originality (1-10): Evaluates uniqueness and innovation while maintaining fable format
    3. Moral Clarity (1-10): Measures how effectively the moral lesson is conveyed
    4. Adherence to Prompt (1-10): Determines how well all required elements from the prompt are incorporated

    Each criterion is scored on a 1-10 scale with specific definitions for low (1-3),
    medium (4-6), and high (7-10) scores.

    Args:
        fable (str): The fable text to evaluate
        original_prompt (str, optional): The original prompt used to generate the fable. Defaults to None.

    Returns:
        dict: JSON evaluation with scores and explanations for each criterion
    """
    try:
        # Load settings from config file
        settings = load_settings()
        evaluator_config = settings.get("evaluator", {})
        model = evaluator_config.get("model", "gpt-4o")
        max_tokens = evaluator_config.get("max_tokens", 350)
        temperature = evaluator_config.get("temperature", 0.0)

        # Get prompts from config
        prompts = evaluator_config.get("prompt", {})
        system_prompt = prompts.get(
            "system",
            None,
        )
        evaluation_prompt_template = prompts.get("evaluation", "")

        # If original_prompt is not provided, use a default message
        if original_prompt is None:
            original_prompt = "Original prompt not available."

        # Format the evaluation prompt with the fable and original prompt
        evaluation_prompt = evaluation_prompt_template.format(
            fable=fable, original_prompt=original_prompt
        )

        # Add an explicit reminder to return valid JSON
        evaluation_prompt += "\n\nRemember to respond with only a valid JSON object, with no additional text before or after the JSON."

        client = OpenAI(api_key=config("OPENAI_API_KEY"))
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        evaluation_text = chat_completion.choices[0].message.content.strip()

        try:
            logger.info(
                f"Parsing response: {evaluation_text[:100]}..."
            )  # Log first 100 chars for debugging
            evaluation_json = json.loads(evaluation_text)
            return evaluation_json
        except json.JSONDecodeError as json_err:
            # Improved logging for JSON parsing errors
            error_context = f"JSON error: {str(json_err)}, Response starts with: {evaluation_text[:200]}"
            logger.error(error_context)
            save_debug_response(evaluation_text, error_context)
            return {"error": error_context}

    except Exception as e:
        logger.error(f"OpenAI API error during evaluation: {str(e)}")
        save_debug_response(f"Error evaluating fable: {str(e)}", str(e))
        return {"error": f"Error evaluating fable: {str(e)}"}


def evaluate_fable_threaded(fable_data: dict) -> dict:
    """
    Worker function to evaluate a single fable.
    Returns a dictionary with model, evaluation result, and the hash.
    """
    try:
        model = fable_data["model"]
        hash_value = fable_data["hash"]
        fable_text = fable_data["fable"]

        # Extract original prompt if available
        original_prompt = fable_data.get("original_prompt", None)

        # Pass the original prompt to the evaluation function
        evaluation = evaluate_fable(fable_text, original_prompt)

        return {"model": model, "evaluation": evaluation, "hash": hash_value}
    except Exception as e:
        logger.error(f"Error in evaluating fable: {e}")
        return {"error": f"Error evaluating fable: {e}"}


def get_original_prompt() -> str:
    """
    Gets the original prompt template from the generator configuration.

    Returns:
        str: The full prompt template used for generating fables
    """
    settings = load_settings()
    generator_config = settings.get("generator", {})
    prompt_config = generator_config.get("prompt", {})

    system_prompt = prompt_config.get("system", "")
    fable_prompt = prompt_config.get("fable", "")

    combined_prompt = (
        f"System Prompt:\n{system_prompt}\n\nFable Prompt:\n{fable_prompt}"
    )
    return combined_prompt


def evaluate_file(file_path: str) -> list:
    """
    Reads the JSONL file at file_path, evaluates each fable (if present)
    using a ThreadPoolExecutor, and returns a list of evaluation results.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()
    fables_to_evaluate = [json.loads(line) for line in lines]

    # Get the original prompt from the configuration
    original_prompt = get_original_prompt()

    # Add the original prompt to each fable data if not already present
    for fable_data in fables_to_evaluate:
        if "fable" in fable_data and "original_prompt" not in fable_data:
            fable_data["original_prompt"] = original_prompt

    results = []
    with ThreadPoolExecutor(max_workers=600) as executor:
        futures = [
            executor.submit(evaluate_fable_threaded, fable_data)
            for fable_data in fables_to_evaluate
            if "fable" in fable_data
        ]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    return results


def run_evaluate(args) -> None:
    """
    Depending on whether the provided --jsonl argument is a file or a directory:
      - If it is a file, evaluates that file.
      - If it is a directory, recursively finds all files that start with "tf_fables"
        and evaluates each one.
    The evaluation results for each input file are stored in the folder
    data/evaluations with a filename format:
      {initial_file_name}_jsonl_eval_e{evaluator_name}_dt{yymmdd-hhmmss}.jsonl
    """
    # Get the evaluator model name from config
    settings = load_settings()
    evaluator_config = settings.get("evaluator", {})
    evaluator_name = evaluator_config.get("model", "gpt-4o")

    output_dir = os.path.join("data", "evaluations")
    os.makedirs(output_dir, exist_ok=True)

    input_path = args.jsonl

    if os.path.isfile(input_path):
        # Process a single file
        results = evaluate_file(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        output_file = os.path.join(
            output_dir, f"{base_name}_jsonl_eval_e{evaluator_name}_dt{timestamp}.jsonl"
        )
        with open(output_file, "w") as out_file:
            for res in results:
                out_file.write(json.dumps(res) + "\n")
        print(f"Evaluations written to {output_file}")

    elif os.path.isdir(input_path):
        # Process all files starting with "tf_fables" recursively
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.startswith("tf_fables"):
                    file_path = os.path.join(root, file)
                    results = evaluate_file(file_path)
                    base_name = os.path.splitext(file)[0]
                    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
                    output_file = os.path.join(
                        output_dir,
                        f"{base_name}_jsonl_eval_e{evaluator_name}_dt{timestamp}.jsonl",
                    )
                    with open(output_file, "w") as out_file:
                        for res in results:
                            out_file.write(json.dumps(res) + "\n")
                    print(f"Evaluations for {file_path} written to {output_file}")
    else:
        logger.error("Provided path is neither a file nor a directory.")
        sys.exit(1)


def add_evaluate_subparser(subparsers) -> None:
    eval_parser = subparsers.add_parser(
        "evaluate",
        help='Evaluate generated fables from a JSONL file or a directory containing files starting with "tf_fables"',
    )
    eval_parser.add_argument(
        "--jsonl",
        type=str,
        help="Path to a JSONL file or a directory to evaluate fables from",
        required=True,
    )
    eval_parser.set_defaults(func=run_evaluate)
