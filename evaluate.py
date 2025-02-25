import json
import sys
from decouple import config
from openai import OpenAI
import yaml
from logger import *
from concurrent.futures import ThreadPoolExecutor

# Constants
CONFIG_FILE = 'tinyfabulist.yaml'

logger = setup_logging()

def load_settings() -> dict:
    try:
        with open(CONFIG_FILE, 'r') as file:
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
            Please evaluate the following fable and return a JSON object with the following format. Ensure your entire response is under 300 tokens:

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
            model="gpt-4o",
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

def run_evaluate(args) -> None:
    # Read the input JSONL file containing the fables to evaluate.
    with open(args.jsonl, 'r') as file:
        lines = file.readlines()
    fables_to_evaluate = [json.loads(line) for line in lines]

    results = []
    # Use a thread pool to evaluate fables concurrently.
    with ThreadPoolExecutor(max_workers=40) as executor:
        # Submit tasks only for entries that contain a 'fable' key.
        futures = [
            executor.submit(evaluate_fable_threaded, fable_data)
            for fable_data in fables_to_evaluate if 'fable' in fable_data
        ]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)

    # Output each result as a JSON line.
    for res in results:
        json.dump(res, sys.stdout)
        sys.stdout.write('\n')

def add_evaluate_subparser(subparsers) -> None:
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated fables')
    eval_parser.add_argument('--jsonl', type=str, help='Evaluate fables from a JSONL file', required=True)
    eval_parser.set_defaults(func=run_evaluate)
