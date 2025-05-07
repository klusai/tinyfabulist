import os
import json
import time
import asyncio
import random
from itertools import count
from pathlib import Path
from typing import Dict, List, Set
from openai import OpenAI
from dotenv import load_dotenv
from tinyfabulist.logger import setup_logging
from tinyfabulist.evaluate.utils import load_settings, get_version_info, update_changelog
from tinyfabulist.translate.utils import read_api_key, load_translator_config

logger = setup_logging()

# Constants
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_CONCURRENCY = 500
BATCH_SIZE = 10

async def translate_fable_async(system_prompt: str, translation_prompt: str, base_url: str, api_key: str) -> str:
    """Async version of fable translation using OpenAI client"""
    client = OpenAI(base_url=base_url, api_key=api_key)
    
    attempt = 0
    backoff_time = INITIAL_RETRY_DELAY
    
    while attempt < MAX_RETRIES:
        try:
            chat_completion = await client.chat.completions.create(
                model="tgi",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": translation_prompt},
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
                logger.error(f"Max retries reached. Failed to translate fable.")
                raise Exception(f"Failed after {MAX_RETRIES} retries: {e}")
    
    raise Exception("Failed to translate fable after multiple attempts")

async def process_single_translation(
    model_name: str,
    model_config: dict,
    fable: str,
    system_prompt: str,
    output_format: str,
    existing_hashes: set,
    output_files: dict,
    counter,
    metadata: dict,
    semaphore,
    api_key: str,
    source_lang: str,
    target_lang: str,
) -> None:
    """Process a single fable translation with rate limiting via semaphore"""
    async with semaphore:
        start_inference_time = time.time()
        
        # Compute hash to avoid duplicates
        hash_val = hash(f"{model_name}_{fable}")
        if hash_val in existing_hashes:
            logger.info(f"Skipping duplicate translation for fable hash: {hash_val}")
            return None
        
        try:
            # Create translation prompt
            translation_prompt = f"Translate the following fable from {source_lang} to {target_lang}:\n\n{fable}"
            
            # Generate the translation using the OpenAI client
            translation = await translate_fable_async(
                system_prompt, 
                translation_prompt, 
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
                        llm_input_tokens = len(tokenizer.encode(translation_prompt))
                        llm_output_tokens = len(tokenizer.encode(translation))
                    else:
                        llm_input_tokens = 0
                        llm_output_tokens = 0
                except Exception as e:
                    logger.error(f"Error computing LLM token counts: {e}")
            
            # Build result dictionary
            result = {
                "source_language": source_lang,
                "target_language": target_lang,
                "original_fable": fable,
                "translated_fable": translation,
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
                "translation_datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pipeline_version": metadata.get("pipeline_version"),
            }
            
            # Get current count for logging
            current_count = next(counter)
            
            return result
        except Exception as e:
            logger.error(f"Error translating fable with hash: {hash_val}: {e}")
            return None

async def write_result_to_file(result, model_name, output_files, output_format):
    """Write a single result to the appropriate output file"""
    if result is None:
        return
    
    result["pipeline_stage"] = "translation"
    
    f = output_files[model_name]
    if output_format == "csv":
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writerow(result)
    elif output_format == "jsonl":
        json.dump(result, f)
        f.write("\n")
    else:
        f.write(f"Source Language: {result['source_language']}\n")
        f.write(f"Target Language: {result['target_language']}\n")
        f.write(f"Model: {result['llm_name']}\n")
        f.write(f"Original Fable:\n{result['original_fable']}\n")
        f.write(f"Translated Fable:\n{result['translated_fable']}\n")
        f.write("-" * 80 + "\n")
    f.flush()

async def async_translate_fables(
    models_to_use,
    available_models,
    system_prompt,
    fables,
    args,
    existing_hashes,
    output_files,
    metadata,
):
    """Main async function to coordinate fable translation"""
    # Create a counter for logging
    counter = count(1)
    
    # Get the API key from environment
    api_key = read_api_key("HF_ACCESS_TOKEN")
    
    # Create a semaphore to limit concurrent connections
    concurrency_limit = getattr(args, "max_concurrency", MAX_CONCURRENCY)
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    # First determine how many tasks we'll create per model
    total_tasks = len(models_to_use) * len(fables)
    logger.info(f"Setting up {total_tasks} translation tasks across {len(models_to_use)} models")
    
    # Track timing statistics
    batch_times = []
    show_progress = getattr(args, "show_progress", False)
    
    # Create tasks for all models in parallel but control concurrency with semaphore
    for model_idx, model_name in enumerate(models_to_use):
        model_config = available_models[model_name]
        logger.info(f"Queuing tasks for model: {model_config['name']} ({model_idx+1}/{len(models_to_use)})")
        
        # Process fables in efficient batches
        batch_size = min(BATCH_SIZE, len(fables))
        total_batches = (len(fables) + batch_size - 1) // batch_size
        
        # Create progress bar if enabled
        batch_iterator = range(0, len(fables), batch_size)
        if show_progress:
            batch_iterator = tqdm(
                batch_iterator, 
                total=total_batches,
                desc=f"Model {model_name}",
                unit="batch"
            )
        
        for batch_idx, i in enumerate(batch_iterator):
            batch_start_time = time.time()
            batch_fables = fables[i:i+batch_size]
            batch_tasks = []
            
            for fable in batch_fables:
                task = process_single_translation(
                    model_name,
                    model_config,
                    fable,
                    system_prompt,
                    args.output,
                    existing_hashes,
                    output_files,
                    counter,
                    metadata,
                    semaphore,
                    api_key,
                    args.source_lang,
                    args.target_lang,
                )
                batch_tasks.append(task)
            
            # Log batch processing start
            batch_id = f"{model_name}-{batch_idx+1}/{total_batches}"
            logger.info(f"Processing batch {batch_id} with {len(batch_tasks)} fables")
            
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

async def run_translate_async(args):
    """Async entry point for the translate command"""
    start_time = time.time()
    
    # Get version info and update changelog
    version_info = get_version_info()
    changelog_path = update_changelog(version_info)
    logger.info(f"Current version: {version_info['version']}")
    logger.info(f"Changelog updated at {changelog_path}")
    
    settings = load_settings()
    
    # Load translator configuration
    available_models = settings.get("llms", {}).get("hf-models", {})
    if not available_models:
        raise ConfigError("No models found in configuration")
        
    models_to_use = args.models if args.models else list(available_models.keys())
    invalid_models = [m for m in models_to_use if m not in available_models]
    if invalid_models:
        raise ConfigError(f"Invalid models: {', '.join(invalid_models)}")
    
    # Load fables from input file
    fables = []
    with open(args.input, 'r') as f:
        for line in f:
            data = json.loads(line)
            if isinstance(data, list):
                data = data[0]  # Handle list format
            if "fable" in data:
                fables.append(data["fable"])
    
    if not fables:
        raise ConfigError("No fables found in input file")
    
    # Load translation prompts
    translator_config = load_settings().get("translator", {})
    system_prompt = translator_config.get("prompt", {}).get("system", "")
    if not system_prompt:
        raise ConfigError("No system prompt found in translator configuration")
    
    # Create output files for each model
    output_files = {}
    for model_name in models_to_use:
        model_folder = os.path.join("data", "translations", model_name)
        os.makedirs(model_folder, exist_ok=True)
        timestamp = time.strftime("%y%m%d-%H%M%S")
        if args.output == "csv":
            file_name = f"tf_translations_{model_name}_dt{timestamp}.csv"
        elif args.output == "jsonl":
            file_name = f"tf_translations_{model_name}_dt{timestamp}.jsonl"
        else:
            file_name = f"tf_translations_{model_name}_dt{timestamp}.txt"
        full_path = os.path.join(model_folder, file_name)
        f = open(full_path, "w", encoding="utf-8")
        if args.output == "csv":
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_language",
                    "target_language",
                    "original_fable",
                    "translated_fable",
                    "llm_name",
                    "llm_input_tokens",
                    "llm_output_tokens",
                    "llm_inference_time",
                    "host_provider",
                    "host_dc_provider",
                    "host_dc_location",
                    "host_gpu",
                    "host_gpu_vram",
                    "host_cost_per_hour",
                    "translation_datetime",
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
    
    # Initialize set for tracking existing hashes
    existing_hashes = set()
    
    # Run the async translation
    await async_translate_fables(
        models_to_use,
        available_models,
        system_prompt,
        fables,
        args,
        existing_hashes,
        output_files,
        metadata,
    )
    
    # Close all output files
    for f in output_files.values():
        f.close()
    
    elapsed_time = time.time() - start_time
    logger.info(f"Fable translation completed in {elapsed_time:.2f} seconds")

def run_translate(args):
    """Entry point for the translate command"""
    asyncio.run(run_translate_async(args)) 