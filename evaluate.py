import json
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

def evaluate_fable(fable: str) -> str:
    try:
        client = OpenAI(api_key=config('OPENAI_TOKEN'))
        evaluation_prompt = f"""
        Please evaluate the following fable based on its creativity, coherence, 
        and moral lesson. Provide a grade from 1 to 10 (inclusive), where 1 is very poor and 10 is excellent.
        Respond with the grade only.

        Fable:
        {fable}
        """
        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a fable critic providing grades."},
                {"role": "user", "content": evaluation_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        evaluation_text = chat_completion.choices[0].message.content.strip()
        return evaluation_text
    except Exception as e:
        logger.error(f"OpenAI API error during evaluation: {e}")
        return f"Error evaluating fable: {e}"

def run_evaluate(args) -> None:
    with open(args.jsonl, 'r') as file:
        lines = file.readlines()
    fables_to_evaluate = [json.loads(line) for line in lines]
    for fable_data in fables_to_evaluate:
        if 'fable' in fable_data:
            model = fable_data['model']
            fable_text = fable_data['fable']
            evaluation = evaluate_fable(fable_text)
            logger.info(f"model: {model} | evaluation: {evaluation}/10\n{'-'*80}")
        else:
            logger.warning(f"Skipping entry due to missing 'fable' key: {fable_data}")

def add_evaluate_subparser(subparsers) -> None:
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate generated fables')
    eval_parser.add_argument('--jsonl', type=str, help='Evaluate fables from a JSONL file', required=True)
    eval_parser.set_defaults(func=run_evaluate)
