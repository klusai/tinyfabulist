import time
import yaml
import json
import sys
import csv
import threading
from pybars import Compiler
from random import sample
from decouple import config
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def generate_fable_threaded(model_name: str, model_config: dict, prompt: str, system_prompt: str, all_fables: list, lock: threading.Lock) -> None:
    try:
        fable = generate_fable(system_prompt, prompt, model_config['base_url'])
        with lock:
            all_fables.append({
                'model': model_config['name'],
                'prompt': prompt,
                'fable': fable
            })
        logger.info(f"Generated fable for prompt: {prompt[:50]}... using model {model_name}")
    except Exception as e:
        logger.error(f"Error generating fable in thread: {e}")

def write_fables(fables: list, output_format: str = 'text') -> None:
    fields = ['model', 'prompt', 'fable']
    if output_format == 'csv':
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fables)
    elif output_format == 'jsonl':
        for fable in fables:
            output = {field: fable[field] for field in fields}
            json.dump(output, sys.stdout)
            sys.stdout.write('\n')
    else:
        for fable in fables:
            print(f"\nModel: {fable['model']}")
            print(f"\nPrompt:\n{fable['prompt']}")
            print(f"\nFable:\n{fable['fable']}\n")
            print("-" * 80)

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
        all_fables = []
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = []
            for model_name in models_to_use:
                model_config = available_models[model_name]
                logger.info(f"Generating fables using model: {model_config['name']}")
                for prompt in fable_prompts:
                    futures.append(
                        executor.submit(
                            generate_fable_threaded,
                            model_name, model_config, prompt, system_prompt, all_fables, lock
                        )
                    )
            for future in as_completed(futures):
                pass
        write_fables(all_fables, args.output)
        elapsed_time = time.time() - start_time
        logger.info(f"Fable generation completed in {elapsed_time:.2f} seconds")

def add_generate_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser('generate', help='Generate fable prompts or fables')
    generate_parser.add_argument('--generate-prompts', action='store_true', help='Generate fable prompts')
    generate_parser.add_argument('--generate-fables', type=str, help='Generate fables from a JSONL prompt file')
    generate_parser.add_argument('--randomize', action='store_true', help='Randomize feature selection')
    generate_parser.add_argument('--output', choices=['text', 'jsonl', 'csv'], default='text', help='Output format (default: text)')
    generate_parser.add_argument('--count', type=int, default=100, help='Number of prompts to generate (default: 100)')
    generate_parser.add_argument('--models', nargs='+', help='Specify models to use')
    generate_parser.set_defaults(func=run_generate)
