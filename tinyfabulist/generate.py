import csv
import hashlib  # For computing SHA-256 hash
import json
import os
import sys
import time
import asyncio
import random
import logging
from itertools import count
from random import sample
from tqdm import tqdm

import yaml
from decouple import config
from openai import AsyncOpenAI
from pybars import Compiler
from transformers import AutoTokenizer

from tinyfabulist.logger import *
from tinyfabulist.utils import load_settings as load_settings_utils
from tinyfabulist.utils import get_version_info, update_changelog

# Global tokenizer cache to avoid reloading tokenizers repeatedly
TOKENIZER_CACHE = {}
# Global OpenAI client cache
CLIENT_CACHE = {}

# Configure OpenAI client logging to suppress HTTP request logs
logging.getLogger("openai._client").setLevel(logging.WARNING)
logging.getLogger("openai.http_client").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

PROMPTS_FOLDER = "data/prompts/"
FABLES_FOLDER = "data/fables/"

# Constants for concurrency and batching
BATCH_SIZE = 1000                   # Maximum number of prompts to process in a batch
MAX_CONCURRENCY = 600  # Maximum number of concurrent requests

# Constants for request management
MAX_RETRIES = 8
INITIAL_RETRY_DELAY = 1.0

logger = setup_logging()

def get_tokenizer(llm_name):
    if llm_name not in TOKENIZER_CACHE:
        try:
            tokenizer = AutoTokenizer.from_pretrained(llm_name)
        except Exception as e:
            logger.error(f"Error loading tokenizer for {llm_name}: {e}")
            tokenizer = None
        TOKENIZER_CACHE[llm_name] = tokenizer
    return TOKENIZER_CACHE[llm_name]

def get_client(base_url, api_key):
    """Get or create an AsyncOpenAI client for the given base_url"""
    cache_key = base_url
    if cache_key not in CLIENT_CACHE:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        CLIENT_CACHE[cache_key] = client
    return CLIENT_CACHE[cache_key]

def load_settings() -> dict:
    try:
        logger.info("Settings loaded successfully")
        return load_settings_utils()
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


async def generate_fable_async(system_prompt: str, fable_prompt: str, base_url: str, api_key: str) -> str:
    """Async version of fable generation using OpenAI client"""
    client = get_client(base_url, api_key)
    
    attempt = 0
    backoff_time = INITIAL_RETRY_DELAY
    
    while attempt < MAX_RETRIES:
        try:
            chat_completion = await client.chat.completions.create(
                model="tgi",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fable_prompt},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            logger.error(f"Error during API call (attempt {attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                # Exponential backoff with jitter
                backoff_time = min(30, backoff_time * 1.5) + (random.random() * 2)
                logger.info(f"Retrying in {backoff_time:.2f} seconds...")
                await asyncio.sleep(backoff_time)
            else:
                logger.error(f"Max retries reached. Failed to generate fable.")
                raise Exception(f"Failed after {MAX_RETRIES} retries: {e}")
    
    raise Exception("Failed to generate fable after multiple attempts")


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
                        if "prompt_hash" in record:
                            hashes.add(record["prompt_hash"])
                    except Exception as e:
                        logger.error(f"Error parsing JSON line: {e}")
            elif output_format == "csv":
                reader = csv.DictReader(f)
                for row in reader:
                    if "prompt_hash" in row:
                        hashes.add(row["prompt_hash"])
            # For text format, we cannot reliably extract hashes.
    except Exception as e:
        logger.error(f"Error reading output file {input_file}: {e}")
    return hashes


async def process_single_generation(
    model_name: str,
    model_config: dict,
    prompt: str,
    system_prompt: str,
    output_format: str,
    existing_hashes: set,
    output_files: dict,
    counter,
    metadata: dict,
    semaphore,
    api_key: str,
) -> None:
    """Process a single fable generation with rate limiting via semaphore"""
    async with semaphore:
        start_inference_time = time.time()
        
        # Compute hash to avoid duplicates
        hash_val = compute_hash(model_config["name"], prompt)
        if hash_val in existing_hashes:
            logger.info(f"Skipping duplicate fable for prompt hash: {hash_val}")
            return None
        
        try:
            # Generate the fable using the OpenAI client
            fable = await generate_fable_async(
                system_prompt, 
                prompt, 
                model_config["base_url"], 
                api_key
            )
            
            # Calculate inference time
            inference_time = time.time() - start_inference_time
            
            # Add the result to existing hashes
            existing_hashes.add(hash_val)
            
            # Calculate token counts if possible (use cached tokenizer)
            llm_name = model_config.get("name", "unknown")
            llm_input_tokens = None
            llm_output_tokens = None
            
            if llm_name != "unknown":
                try:
                    tokenizer = get_tokenizer(llm_name)
                    if tokenizer:
                        llm_input_tokens = len(tokenizer.encode(prompt))
                        llm_output_tokens = len(tokenizer.encode(fable))
                    else:
                        llm_input_tokens = 0
                        llm_output_tokens = 0
                except Exception as e:
                    logger.error(f"Error computing LLM token counts: {e}")
            
            # Build result dictionary
            result = {
                "language": "en",
                "prompt": prompt,
                "prompt_hash": hash_val,
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
                "currency": model_config.get("currency"),
                "generation_datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": metadata.get("pipeline_version"),
            }
            
            # Get current count for logging
            current_count = next(counter)
            # logger.info(
            #     f"Generated fable #{current_count} with hash: {hash_val} using model {model_name}"
            # )
            
            return result
        except Exception as e:
            logger.error(f"Error generating fable with hash: {hash_val}: {e}")
            return None


async def write_result_to_file(result, model_name, output_files, output_format):
    """Write a single result to the appropriate output file"""
    if result is None:
        return
    
    f = output_files[model_name]
    if output_format == "csv":
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writerow(result)
    elif output_format == "jsonl":
        json.dump(result, f)
        f.write("\n")
    else:
        f.write(f"Language: {result['language']}\n")
        f.write(f"Model: {result['llm_name']}\n")
        f.write(f"Prompt:\n{result['prompt']}\n")
        f.write(f"Fable:\n{result['fable']}\n")
        f.write(f"Prompt Hash: {result['prompt_hash']}\n")
        f.write("-" * 80 + "\n")
    f.flush()


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


async def async_generate_fables(
    models_to_use,
    available_models,
    system_prompt,
    fable_prompts,
    args,
    existing_hashes,
    output_files,
    metadata,
):
    """Main async function to coordinate fable generation"""
    # Create a counter for logging
    counter = count(1)
    
    # Get the API key from environment
    api_key = config("HF_ACCESS_TOKEN")
    
    # Create a semaphore to limit concurrent connections
    # Use a reasonable number based on available system resources
    concurrency_limit = getattr(args, "max_concurrency", MAX_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    # First determine how many tasks we'll create per model
    total_tasks = len(models_to_use) * len(fable_prompts)
    logger.info(f"Setting up {total_tasks} generation tasks across {len(models_to_use)} models")
    
    # Track timing statistics
    batch_times = []
    show_progress = getattr(args, "show_progress", False)
    
    # Create tasks for all models in parallel but control concurrency with semaphore
    for model_idx, model_name in enumerate(models_to_use):
        model_config = available_models[model_name]
        logger.info(f"Queuing tasks for model: {model_config['name']} ({model_idx+1}/{len(models_to_use)})")
        
        # Process prompts in efficient batches
        batch_size = min(BATCH_SIZE, len(fable_prompts))
        total_batches = (len(fable_prompts) + batch_size - 1) // batch_size
        
        # Create progress bar if enabled
        batch_iterator = range(0, len(fable_prompts), batch_size)
        if show_progress:
            batch_iterator = tqdm(
                batch_iterator, 
                total=total_batches,
                desc=f"Model {model_name}",
                unit="batch"
            )
        
        for batch_idx, i in enumerate(batch_iterator):
            batch_start_time = time.time()
            batch_prompts = fable_prompts[i:i+batch_size]
            batch_tasks = []
            
            for prompt in batch_prompts:
                task = process_single_generation(
                    model_name,
                    model_config,
                    prompt,
                    system_prompt,
                    args.output,
                    existing_hashes,
                    output_files,
                    counter,
                    metadata,
                    semaphore,
                    api_key,
                )
                batch_tasks.append(task)
            
            # Log batch processing start
            batch_id = f"{model_name}-{batch_idx+1}/{total_batches}"
            logger.info(f"Processing batch {batch_id} with {len(batch_tasks)} prompts")
            
            # Start the batch processing
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process and write results from this batch immediately
            successful_results = 0
            write_tasks = []
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with error: {result}")
                    continue
                    
                if result is not None:
                    successful_results += 1
                    # Write result to file
                    write_task = write_result_to_file(
                        result, model_name, output_files, args.output
                    )
                    write_tasks.append(write_task)
            
            # Wait for all write operations in this batch to complete
            if write_tasks:
                await asyncio.gather(*write_tasks)
            
            # Calculate and log batch timing statistics
            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            batch_times.append(batch_duration)
            
            # Calculate average time per item and estimate remaining time
            avg_time_per_batch = sum(batch_times) / len(batch_times)
            remaining_batches = total_batches * len(models_to_use) - (model_idx * total_batches + batch_idx + 1)
            est_remaining_time = avg_time_per_batch * remaining_batches
            
            # Log batch completion with timing info
            logger.info(
                f"Completed batch {batch_id}: {successful_results}/{len(batch_tasks)} successful "
                f"in {batch_duration:.2f}s (avg: {avg_time_per_batch:.2f}s/batch, "
                f"est. remaining: {est_remaining_time/60:.1f}m)"
            )
            
            # Update progress bar if enabled
            if show_progress and hasattr(batch_iterator, "set_postfix"):
                batch_iterator.set_postfix({
                    "success": f"{successful_results}/{len(batch_tasks)}",
                    "time": f"{batch_duration:.1f}s",
                    "remaining": f"{est_remaining_time/60:.1f}m"
                })
            
            # Small delay between batches to allow recovery
            await asyncio.sleep(0.1)


async def run_generate_async(args):
    """Async entry point for the generate command"""
    start_time = time.time()
    
    # Get version info and update changelog
    version_info = get_version_info()
    changelog_path = update_changelog(version_info)
    logger.info(f"Current version: {version_info['version']}")
    logger.info(f"Changelog updated at {changelog_path}")
    
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

        # NEW: Check for worker-specific environment variables to split work across machines.
        worker_id = int(os.environ.get("WORKER_ID", 0))
        total_workers = int(os.environ.get("TOTAL_WORKERS", 1))
        original_count = len(fable_prompts)
        fable_prompts = [p for idx, p in enumerate(fable_prompts) if idx % total_workers == worker_id]
        logger.info(f"Worker {worker_id}/{total_workers}: Processing {len(fable_prompts)} prompts out of {original_count}")

        if total_workers > 1:
            original_count = len(fable_prompts)
            # Each worker processes only the prompts where index % total_workers == worker_id
            fable_prompts = [p for idx, p in enumerate(fable_prompts) if idx % total_workers == worker_id]
            logger.info(
                f"Worker {worker_id}/{total_workers}: Processing {len(fable_prompts)} prompts out of {original_count}"
            )

        existing_hashes = load_existing_hashes(args.input_file, args.output)
        logger.info(
            f"Found {len(existing_hashes)} existing hashes in {args.input_file}"
        )

        # Open output files for each model
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
                        "prompt_hash",
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

        # Extract metadata from settings and add version info
        metadata = settings.get("metadata", {})
        metadata.update({
            "pipeline_version": version_info["version"]
        })

        # Run the async generation
        await async_generate_fables(
            models_to_use,
            available_models,
            system_prompt,
            fable_prompts,
            args,
            existing_hashes,
            output_files,
            metadata,
        )

        # Close all output files
        for f in output_files.values():
            f.close()

        elapsed_time = time.time() - start_time
        logger.info(f"Fable generation completed in {elapsed_time:.2f} seconds")


def run_generate(args):
    """Synchronous wrapper for the async function"""
    # Create and run the event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generate_async(args))


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
    generate_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=MAX_CONCURRENCY,
        help="Maximum number of concurrent requests (default: 500)",
    )
    generate_parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars for batch processing",
    )
    generate_parser.set_defaults(func=run_generate)
