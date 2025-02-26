import json
import sys
import os
from decouple import config
from openai import OpenAI
import yaml
from logger import *
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Constants
CONFIG_FILE = "tinyfabulist.yaml"
EVALUATOR = "gpt-4o"

logger = setup_logging()

def load_settings() -> dict:
    try:
        with open(CONFIG_FILE, "r") as file:
            settings = yaml.safe_load(file)
            logger.info("Settings loaded successfully")
            return settings
    except FileNotFoundError:
        logger.error(f"Settings file '{CONFIG_FILE}' not found")
        raise ConfigError(f"Settings file '{CONFIG_FILE}' not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ConfigError(f"Invalid YAML format: {e}")

def evaluate_fable(fable: str) -> dict:
    """
    Evaluates a fable on three criteria: grammar, creativity, and consistency.
    The model is instructed to respond with a JSON object with keys "grammar",
    "creativity", and "consistency", along with an explanation.
    The entire response must be under 300 tokens.
    """
    try:
        client = OpenAI(api_key=config('OPENAI_API_KEY'))
        evaluation_prompt = f"""
            Be very strict in your evaluations, but just and concise.
            Please evaluate the following fable based on the criteria below:
            1. **Grammar:** Is the fable grammatically correct and well-written?
            2. **Creativity:** Does it exhibit originality while following a classic fable format?
            3. **Moral Clarity:** Is the moral lesson clearly conveyed and thought-provoking?

            Ensure your entire response is under 300 tokens
            Return your evaluation as a JSON object with the following format. Each criterion should receive a grade between 1 and 10:
            {{
                "type": "Fable Evaluation",
                "grammar": <grade between 1 and 10>,
                "creativity": <grade between 1 and 10>,
                "consistency": <grade between 1 and 10>,
                "explanation": [
                    "Brief explanation for the grammar score.",
                    "Brief explanation for the creativity score.",
                    "Brief explanation for the consistency score."
                ]
            }}

            Fable:
            {fable}
        """
        chat_completion = client.chat.completions.create(
            model=EVALUATOR,
            messages=[
                {"role": "system", "content": "You are a fable critic providing detailed evaluations. Keep your entire response under 300 tokens."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=350,
            temperature=0.0
        )
        evaluation_text = chat_completion.choices[0].message.content.strip()
        # Optionally, remove any extraneous header/footer lines if present.
        evaluation_text = "\n".join(evaluation_text.split("\n")[1:-1])
        evaluation_json = json.loads(evaluation_text)
        return evaluation_json
    except Exception as e:
        logger.error(f"OpenAI API error during evaluation: {e}")
        return {"error": f"Error evaluating fable: {e}"}

def evaluate_fable_threaded(fable_data: dict) -> dict:
    """
    Worker function to evaluate a single fable.
    Returns a dictionary with model, evaluation result, and the hash.
    """
    try:
        model = fable_data['model']
        hash_value = fable_data['hash']
        fable_text = fable_data['fable']
        evaluation = evaluate_fable(fable_text)
        return {
            "model": model,
            "evaluation": evaluation,
            "hash": hash_value
        }
    except Exception as e:
        logger.error(f"Error in evaluating fable: {e}")
        return {"error": f"Error evaluating fable: {e}"}

def evaluate_file(file_path: str) -> list:
    """
    Reads the JSONL file at file_path, evaluates each fable (if present)
    using a ThreadPoolExecutor, and returns a list of evaluation results.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    fables_to_evaluate = [json.loads(line) for line in lines]
    results = []
    with ThreadPoolExecutor(max_workers=600) as executor:
        futures = [
            executor.submit(evaluate_fable_threaded, fable_data)
            for fable_data in fables_to_evaluate if 'fable' in fable_data
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
    evaluator_name = EVALUATOR
    output_dir = os.path.join("data", "evaluations")
    os.makedirs(output_dir, exist_ok=True)
    
    input_path = args.jsonl

    if os.path.isfile(input_path):
        # Process a single file
        results = evaluate_file(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_jsonl_eval_e{evaluator_name}_dt{timestamp}.jsonl")
        with open(output_file, 'w') as out_file:
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
                    output_file = os.path.join(output_dir, f"{base_name}_jsonl_eval_e{evaluator_name}_dt{timestamp}.jsonl")
                    with open(output_file, 'w') as out_file:
                        for res in results:
                            out_file.write(json.dumps(res) + "\n")
                    print(f"Evaluations for {file_path} written to {output_file}")
    else:
        logger.error("Provided path is neither a file nor a directory.")
        sys.exit(1)

def add_evaluate_subparser(subparsers) -> None:
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate generated fables from a JSONL file or a directory containing files starting with "tf_fables"'
    )
    eval_parser.add_argument(
        '--jsonl',
        type=str,
        help='Path to a JSONL file or a directory to evaluate fables from',
        required=True
    )
    eval_parser.set_defaults(func=run_evaluate)
