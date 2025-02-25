import json
import sys
from decouple import config
from openai import OpenAI
import yaml
from logger import *

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
    "creativity", and "consistency", each having a grade from 1 to 10.
    """
    try:
        client = OpenAI(api_key=config('OPENAI_API_KEY'))
        evaluation_prompt = f"""
            Please evaluate the following fable and return a JSON object with the following format:

            {{
                "type": "Fable Evaluation",
                "explanation": [
                    "Explanation for the grammar score.",
                    "Explanation for the creativity score.",
                    "Explanation for the consistency score."
                ]
                "grammar": <grade between 1 and 10>,
                "creativity": <grade between 1 and 10>,
                "consistency": <grade between 1 and 10>,
            }}

            Fable:
            {fable}"""
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fable critic providing detailed evaluations. Ensure that your entire response is under 300 tokens"},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=350,
            temperature=0.0
        )
        evaluation_text = chat_completion.choices[0].message.content.strip()

        #remove first and last line of chatgpt answer
        evaluation_text = "\n".join(evaluation_text.split("\n")[1:-1])

        evaluation_json = json.loads(evaluation_text)
        return evaluation_json
    except Exception as e:
        logger.error(f"OpenAI API error during evaluation: {e}")
        return {"error": f"Error evaluating fable: {e}"}

def run_evaluate(args) -> None:
    results = []
    with open(args.jsonl, 'r') as file:
        lines = file.readlines()

    fables_to_evaluate = [json.loads(line) for line in lines]
    for fable_data in fables_to_evaluate:
        if 'fable' in fable_data:
            model = fable_data['model']
            hash_value = fable_data['hash']

            fable_text = fable_data['fable']
            evaluation = evaluate_fable(fable_text)

            result = {
                "model": model,
                "evaluation": evaluation,
                "hash": hash_value
            }
            print(result)
            results.append(result)
        else:
            logger.warning(f"Skipping entry due to missing 'fable' key: {fable_data}")
    
    for res in results:
        json.dump(res, sys.stdout)
        sys.stdout.write('\n')

def add_evaluate_subparser(subparsers) -> None:
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated fables')
    eval_parser.add_argument('--jsonl', type=str, help='Evaluate fables from a JSONL file', required=True)
    eval_parser.set_defaults(func=run_evaluate)
