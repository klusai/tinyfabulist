import csv
import hashlib  # For computing SHA-256 hash
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import count
from random import sample

import yaml
from decouple import config
from openai import OpenAI
from pybars import Compiler
from transformers import AutoTokenizer

from tinyfabulist.logger import *
from tinyfabulist.utils import load_settings as load_settings_utils

# Constants
CONFIG_FILE = "conf/generator.yaml"
PROMPTS_FOLDER = "tinyfabulist/data/prompts/"
FABLES_FOLDER = "tinyfabulist/data/fables/"

logger = setup_logging()


def load_settings() -> dict:
    try:
        logger.info("Settings loaded successfully")
        return load_settings_utils()
    except FileNotFoundError:
        logger.error(f"Settings file '{CONFIG_FILE}' not found")
        raise ConfigError(f"Settings file '{CONFIG_FILE}' not found")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise ConfigError(f"Invalid YAML format: {e}")


def generate_prompts(config: dict, count: int = 10, randomize: bool = False):
    features = config["generator"]["features"]
    prompts = []
    used_combinations = set()
    compiler = Compiler()
    system_template = compiler.compile(config["generator"]["prompt"]["system"])
    generator_template = compiler.compile(config["generator"]["prompt"]["fable"])
    system_prompt = system_template({})

    while len(prompts) < count:
        if randomize:
            combination = (
                sample(features["characters"], 1)[0],
                sample(features["traits"], 1)[0],
                sample(features["settings"], 1)[0],
                sample(features["conflicts"], 1)[0],
                sample(features["resolutions"], 1)[0],
                sample(features["morals"], 1)[0],
            )
            if combination not in used_combinations:
                used_combinations.add(combination)
                char, trait, setting, conflict, resolution, moral = combination
                context = {
                    "character": char,
                    "trait": trait,
                    "setting": setting,
                    "conflict": conflict,
                    "resolution": resolution,
                    "moral": moral,
                }
                prompts.append(generator_template(context))
        else:
            idx = len(prompts)
            context = {
                "character": features["characters"][idx % len(features["characters"])],
                "trait": features["traits"][idx % len(features["traits"])],
                "setting": features["settings"][idx % len(features["settings"])],
                "conflict": features["conflicts"][idx % len(features["conflicts"])],
                "resolution": features["resolutions"][
                    idx % len(features["resolutions"])
                ],
                "moral": features["morals"][idx % len(features["morals"])],
            }
            prompts.append(generator_template(context))

    return system_prompt, prompts


def read_prompts(filename: str):
    try:
        with open(filename, "r") as f:
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
        client = OpenAI(base_url=base_url, api_key=config("HF_ACCESS_TOKEN"))
        chat_completion = client.chat.completions.create(
            model="tgi",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": fable_prompt},
            ],
            max_tokens=1000,
            temperature=0.7,
            stream=True,
        )
        fable_text = ""
        for message in chat_completion:
            if message.choices[0].delta.content is not None:
                fable_text += message.choices[0].delta.content
        return fable_text
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise Exception(f"Error generating fable: {e}")


def compute_hash(model: str, prompt: str) -> str:
    """
    Computes a SHA-256 hash from the model and prompt.
    """
    hash_obj = hashlib.sha256()
    hash_obj.update((model + prompt).encode("utf-8"))
    return hash_obj.hexdigest()


def load_existing_hashes(input_file: str, output_format: str) -> set:
    """
    Loads existing hashes from the output file based on the chosen format.
    Returns a set of hash strings.
    """
    hashes = set()
    if not os.path.exists(input_file):
        return hashes

    try:
        with open(input_file, "r") as f:
            if output_format == "jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if "hash" in record:
                            hashes.add(record["hash"])
                    except Exception as e:
                        logger.error(f"Error parsing JSON line: {e}")
            elif output_format == "csv":
                reader = csv.DictReader(f)
                for row in reader:
                    if "hash" in row:
                        hashes.add(row["hash"])
            # For text format, we cannot reliably extract hashes.
    except Exception as e:
        logger.error(f"Error reading output file {input_file}: {e}")
    return hashes


def generate_fable_threaded(
    model_name: str,
    model_config: dict,
    prompt: str,
    system_prompt: str,
    output_format: str,
    lock: threading.Lock,
    existing_hashes: set,
    output_files: dict,
    counter,
    metadata: dict,
    max_retries: int = 8,
    retry_delay: float = 15.0,
) -> None:
    # Measure the inference start time.
    start_inference_time = time.time()
    attempt = 0

    # Compute the hash based on the model and prompt.
    hash_val = compute_hash(model_config["name"], prompt)
    with lock:
        if hash_val in existing_hashes:
            logger.info(f"Skipping duplicate fable for hash: {hash_val}")
            return

    # Retry loop for generating the fable.
    fable = None
    while attempt < max_retries:
        try:
            fable = generate_fable(system_prompt, prompt, model_config["base_url"])
            break  # Successful generation, exit the loop.
        except Exception as e:
            attempt += 1
            logger.error(
                f"Error generating fable (attempt {attempt}/{max_retries}) for fable with hash: {hash_val}: {e}"
            )
            if attempt < max_retries:
                retry_delay += 5
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Max retries reached. Failed to generate fable for fable with hash: {hash_val}"
                )
                return  # Give up after maximum retries.

    # Calculate inference time.
    inference_time = time.time() - start_inference_time

    try:
        with lock:
            # Add the new hash so subsequent tasks skip duplicates.
            existing_hashes.add(hash_val)

        llm_name = model_config.get("name", "unknown")
        llm_input_tokens = None
        llm_output_tokens = None

        if llm_name != "unknown":
            try:
                tokenizer = AutoTokenizer.from_pretrained(llm_name)
                llm_input_tokens = len(tokenizer.encode(prompt))
                llm_output_tokens = len(tokenizer.encode(fable))
            except Exception as e:
                logger.error(f"Error computing LLM token counts: {e}")

        # Build the result dictionary with additional metadata.
        result = {
            "language": "en",
            "prompt": prompt,
            "hash": hash_val,
            "fable": fable,
            "llm_name": llm_name,
            "llm_input_tokens": llm_input_tokens,
            "llm_output_tokens": llm_output_tokens,
            "llm_inference_time": inference_time,
            "host_provider": metadata.get("host_provider"),
            "host_dc_provider": metadata.get("host_dc_provider"),
            "host_dc_location": metadata.get("host_dc_location"),
            "host_gpu": model_config.get("host_gpu"),
            "host_gpu_vram": model_config.get("host_gpu_vram"),
            "host_cost_per_hour": model_config.get("host_cost_per_hour"),
            "generation_datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_version": metadata.get("pipeline_version"),
        }
        with lock:
            # Write the result immediately to the file corresponding to this model.
            f = output_files[model_name]
            if output_format == "csv":
                writer = csv.DictWriter(f, fieldnames=list(result.keys()))
                writer.writerow(result)
            elif output_format == "jsonl":
                json.dump(result, f)
                f.write("\n")
            else:
                f.write(f"Language: {result['language']}\n")
                f.write(f"Model: {result['model']}\n")
                f.write(f"Prompt:\n{result['prompt']}\n")
                f.write(f"Fable:\n{result['fable']}\n")
                f.write(f"Hash: {result['hash']}\n")
                f.write("-" * 80 + "\n")
            f.flush()
        with lock:
            current_count = next(counter)
        logger.info(
            f"Generated fable #{current_count} with hash: {hash_val} using model {model_name}"
        )
        return result
    except Exception as e:
        logger.error(
            f"Error processing result in thread for fable with hash: {hash_val}: {e}"
        )


def write_generated_prompts(system_prompt: str, fable_templates: list) -> None:
    """
    Writes the system prompt and generated fable prompts to a JSONL file in the PROMPTS_FOLDER.
    The file is named as: tf_prompts_c{n}_dt{yymmdd-hhmmss}.jsonl,
    where n is the number of fable templates.
    """
    os.makedirs(PROMPTS_FOLDER, exist_ok=True)

    n = len(fable_templates)
    timestamp = time.strftime("%y%m%d-%H%M%S")
    file_name = f"tf_prompts_c{n}_dt{timestamp}.jsonl"
    full_path = os.path.join(PROMPTS_FOLDER, file_name)

    with open(full_path, "w") as f:
        json.dump([{"prompt_type": "system_prompt", "content": system_prompt}], f)
        f.write("\n")
        for template in fable_templates:
            json.dump([{"prompt_type": "generator_prompt", "content": template}], f)
            f.write("\n")

    logger.info(f"Generated prompts written to {full_path}")


def write_output(system_prompt: str, fable_templates: list, output_format: str) -> None:
    # For fable generation output (if writing to stdout)
    if output_format == "jsonl":
        for template in fable_templates:
            json.dump(
                [
                    {"prompt_type": "system_prompt", "content": system_prompt},
                    {"prompt_type": "generator_prompt", "content": template},
                ],
                sys.stdout,
            )
            sys.stdout.write("\n")
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
        write_generated_prompts(system_prompt, fable_templates)
    else:
        available_models = settings.get("llms", {}).get("hf-models", {})
        if not available_models:
            raise ConfigError("No models found in configuration")
        models_to_use = args.models if args.models else list(available_models.keys())
        invalid_models = [m for m in models_to_use if m not in available_models]
        if invalid_models:
            raise ConfigError(f"Invalid models: {', '.join(invalid_models)}")
        prompts = list(read_prompts(args.generate_fables))
        system_prompt = next(
            (p["content"] for p in prompts if p["prompt_type"] == "system_prompt"), None
        )
        fable_prompts = [
            p["content"] for p in prompts if p["prompt_type"] == "generator_prompt"
        ]
        if not system_prompt:
            raise ConfigError("No system prompt found in prompt file.")

        existing_hashes = load_existing_hashes(args.input_file, args.output)
        logger.info(
            f"Found {len(existing_hashes)} existing hashes in {args.input_file}"
        )

        output_lock = threading.Lock()

        # Open output files for each model so results can be written gradually.
        output_files = {}
        for model_name in models_to_use:
            model_folder = os.path.join(FABLES_FOLDER, model_name)
            os.makedirs(model_folder, exist_ok=True)
            timestamp = time.strftime("%y%m%d-%H%M%S")
            if args.output == "csv":
                file_name = f"tf_fables_{model_name}_dt{timestamp}.csv"
            elif args.output == "jsonl":
                file_name = f"tf_fables_{model_name}_dt{timestamp}.jsonl"
            else:
                file_name = f"tf_fables_{model_name}_dt{timestamp}.txt"
            full_path = os.path.join(model_folder, file_name)
            f = open(full_path, "w", encoding="utf-8")
            if args.output == "csv":
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "language",
                        "model",
                        "prompt",
                        "fable",
                        "hash",
                        "llm_name",
                        "llm_input_tokens",
                        "llm_output_tokens",
                        "llm_inference_time",
                        "llm_inference_cost_usd",
                        "host_provider",
                        "host_dc_provider",
                        "host_dc_location",
                        "host_gpu",
                        "host_gpu_vram",
                        "host_cost_per_hour",
                        "generation_datetime",
                        "pipeline_version",
                    ],
                )
                writer.writeheader()
                f.flush()
            output_files[model_name] = f

        # Create a shared counter for fable generation.
        counter = count(1)

        # Extract metadata from settings. You can store these details in a dedicated key in your YAML.
        metadata = settings.get("metadata", {})

        futures = []
        with ThreadPoolExecutor(max_workers=100) as executor:
            for model_name in models_to_use:
                model_config = available_models[model_name]
                logger.info(f"Generating fables using model: {model_config['name']}")
                for prompt in fable_prompts:
                    futures.append(
                        executor.submit(
                            generate_fable_threaded,
                            model_name,
                            model_config,
                            prompt,
                            system_prompt,
                            args.output,
                            output_lock,
                            existing_hashes,
                            output_files,
                            counter,
                            metadata,
                        )
                    )

        # Wait for all threads to finish.
        for future in futures:
            future.result()

        # Close all output files.
        for f in output_files.values():
            f.close()

        elapsed_time = time.time() - start_time
        logger.info(f"Fable generation completed in {elapsed_time:.2f} seconds")


def add_generate_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        "generate", help="Generate fable prompts or fables"
    )
    generate_parser.add_argument(
        "--generate-prompts", action="store_true", help="Generate fable prompts"
    )
    generate_parser.add_argument(
        "--generate-fables", type=str, help="Generate fables from a JSONL prompt file"
    )
    generate_parser.add_argument(
        "--randomize", action="store_true", help="Randomize feature selection"
    )
    generate_parser.add_argument(
        "--output",
        choices=["text", "jsonl", "csv"],
        default="text",
        help="Output format (default: text)",
    )
    generate_parser.add_argument(
        "--input-file", type=str, default="results.jsonl", help="Input file"
    )
    generate_parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Number of prompts to generate (default: 100)",
    )
    generate_parser.add_argument("--models", nargs="+", help="Specify models to use")
    generate_parser.set_defaults(func=run_generate)
