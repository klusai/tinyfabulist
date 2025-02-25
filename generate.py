import os
import time
import yaml
import json
import sys
import csv
import threading
import hashlib  # For computing SHA-256 hash
from pybars import Compiler
from random import sample
from decouple import config
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

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

def generate_prompts(config: dict, count: int = 10, randomize: bool = False):
    features = config['generator']['features']
    prompts = []
    used_combinations = set()
    compiler = Compiler()
    system_template = compiler.compile(config['generator']['prompt']['system'])
    generator_template = compiler.compile(config['generator']['prompt']['fable'])
    system_prompt = system_template({})

    while len(prompts) < count:
        if randomize:
            combination = (
                sample(features['characters'], 1)[0],
                sample(features['traits'], 1)[0],
                sample(features['settings'], 1)[0],
                sample(features['conflicts'], 1)[0],
                sample(features['resolutions'], 1)[0],
                sample(features['morals'], 1)[0]
            )
            if combination not in used_combinations:
                used_combinations.add(combination)
                char, trait, setting, conflict, resolution, moral = combination
                context = {
                    'character': char,
                    'trait': trait,
                    'setting': setting,
                    'conflict': conflict,
                    'resolution': resolution,
                    'moral': moral
                }
                prompts.append(generator_template(context))
        else:
            idx = len(prompts)
            context = {
                'character': features['characters'][idx % len(features['characters'])],
                'trait': features['traits'][idx % len(features['traits'])],
                'setting': features['settings'][idx % len(features['settings'])],
                'conflict': features['conflicts'][idx % len(features['conflicts'])],
                'resolution': features['resolutions'][idx % len(features['resolutions'])],
                'moral': features['morals'][idx % len(features['morals'])]
            }
            prompts.append(generator_template(context))

    return system_prompt, prompts

def read_prompts(filename: str):
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    prompt_list = json.loads(line)
                    for prompt in prompt_list:
                        yield prompt
    except FileNotFoundError:
        logger.error(f"Prompt file '{filename}' not found")
        raise ConfigError(f"Prompt file '{filename}' not found")
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSONL file: {e}")
        raise ConfigError(f"Invalid JSONL format: {e}")

def generate_fable(system_prompt: str, fable_prompt: str, base_url: str) -> str:
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=config('HF_ACCESS_TOKEN')
        )
        chat_completion = client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fable_prompt}
            ],
            max_tokens=1000,
            temperature=0.7,
            stream=True
        )
        fable_text = ""
        for message in chat_completion:
            if message.choices[0].delta.content is not None:
                fable_text += message.choices[0].delta.content
        return fable_text
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"Error generating fable: {e}"

def compute_hash(model: str, prompt: str) -> str:
    """
    Computes a SHA-256 hash from the model and prompt.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update((model + prompt).encode('utf-8'))
    return hash_obj.hexdigest()

def load_existing_hashes(output_file: str, output_format: str) -> set:
    """
    Loads existing hashes from the output file based on the chosen format.
    Returns a set of hash strings.
    """
    hashes = set()
    if not os.path.exists(output_file):
        return hashes

    try:
        with open(output_file, 'r') as f:
            if output_format == 'jsonl':
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if 'hash' in record:
                            hashes.add(record['hash'])
                    except Exception as e:
                        logger.error(f"Error parsing JSON line: {e}")
            elif output_format == 'csv':
                reader = csv.DictReader(f)
                for row in reader:
                    if 'hash' in row:
                        hashes.add(row['hash'])
            # For text format, we cannot reliably extract hashes.
    except Exception as e:
        logger.error(f"Error reading output file {output_file}: {e}")
    return hashes

def generate_fable_threaded(model_name: str, model_config: dict, prompt: str,
                             system_prompt: str, output_format: str,
                             lock: threading.Lock, existing_hashes: set) -> None:
    try:
        fable = generate_fable(system_prompt, prompt, model_config['base_url'])
        # Compute the hash based on the model and prompt
        hash_val = compute_hash(model_config['name'], prompt)
        with lock:
            if hash_val in existing_hashes:
                logger.info(f"Skipping duplicate fable for hash: {hash_val}")
                return
            # Add the new hash to the set so subsequent tasks see it.
            existing_hashes.add(hash_val)
        result = {
            'model': model_config['name'],
            'prompt': prompt,
            'fable': fable,
            'hash': hash_val
        }
        with lock:
            if output_format == 'csv':
                writer = csv.DictWriter(sys.stdout, fieldnames=['model', 'prompt', 'fable', 'hash'])
                writer.writerow(result)
                sys.stdout.flush()
            elif output_format == 'jsonl':
                json.dump(result, sys.stdout)
                sys.stdout.write('\n')
                sys.stdout.flush()
            else:  # text
                print(f"\nModel: {result['model']}")
                print(f"\nPrompt:\n{result['prompt']}")
                print(f"\nFable:\n{result['fable']}")
                print(f"\nHash: {result['hash']}")
                print("-" * 80)
        logger.info(f"Generated fable for prompt: {prompt[:50]}... using model {model_name}")
    except Exception as e:
        logger.error(f"Error generating fable in thread: {e}")

def write_output(system_prompt: str, fable_templates: list, output_format: str) -> None:
    if output_format == 'jsonl':
        for template in fable_templates:
            json.dump([
                {
                    'prompt_type': 'system_prompt',
                    'content': system_prompt
                },
                {
                    'prompt_type': 'generator_prompt',
                    'content': template
                }
            ], sys.stdout)
            sys.stdout.write('\n')
    else:
        print("System prompt:", system_prompt)
        print("\nFable templates:")
        for i, template in enumerate(fable_templates, 1):
            print(f"\n{i}. {template}")

def run_generate(args) -> None:
    start_time = time.time()
    settings = load_settings()
    if args.generate_prompts:
        system_prompt, fable_templates = generate_prompts(
            settings, count=args.count, randomize=args.randomize
        )
        write_output(system_prompt, fable_templates, args.output)
    else:
        available_models = settings.get('llms', {}).get('hf-models', {})
        if not available_models:
            raise ConfigError("No models found in configuration")
        models_to_use = args.models if args.models else list(available_models.keys())
        invalid_models = [m for m in models_to_use if m not in available_models]
        if invalid_models:
            raise ConfigError(f"Invalid models: {', '.join(invalid_models)}")
        prompts = list(read_prompts(args.generate_fables))
        system_prompt = next((p['content'] for p in prompts if p['prompt_type'] == 'system_prompt'), None)
        fable_prompts = [p['content'] for p in prompts if p['prompt_type'] == 'generator_prompt']
        if not system_prompt:
            raise ConfigError("No system prompt found in prompt file.")

        # Load existing hashes from the output file
        existing_hashes = load_existing_hashes(args.output_file, args.output)
        logger.info(f"Found {len(existing_hashes)} existing hashes in {args.output_file}")

        output_lock = threading.Lock()
        # For CSV, write the header once before starting the threads if file is empty
        if args.output == 'csv' and not os.path.exists(args.output_file):
            with output_lock:
                writer = csv.DictWriter(sys.stdout, fieldnames=['model', 'prompt', 'fable', 'hash'])
                writer.writeheader()
                sys.stdout.flush()

        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = []
            for model_name in models_to_use:
                model_config = available_models[model_name]
                logger.info(f"Generating fables using model: {model_config['name']}")
                for prompt in fable_prompts:
                    futures.append(
                        executor.submit(
                            generate_fable_threaded,
                            model_name, model_config, prompt, system_prompt,
                            args.output, output_lock, existing_hashes
                        )
                    )
        elapsed_time = time.time() - start_time
        logger.info(f"Fable generation completed in {elapsed_time:.2f} seconds")

def add_generate_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser('generate', help='Generate fable prompts or fables')
    generate_parser.add_argument('--generate-prompts', action='store_true', help='Generate fable prompts')
    generate_parser.add_argument('--generate-fables', type=str, help='Generate fables from a JSONL prompt file')
    generate_parser.add_argument('--randomize', action='store_true', help='Randomize feature selection')
    generate_parser.add_argument('--output', choices=['text', 'jsonl', 'csv'], default='text', help='Output format (default: text)')
    generate_parser.add_argument('--output-file', type=str, default='results.jsonl', help='Output file')
    generate_parser.add_argument('--count', type=int, default=100, help='Number of prompts to generate (default: 100)')
    generate_parser.add_argument('--models', nargs='+', help='Specify models to use')
    generate_parser.set_defaults(func=run_generate)
