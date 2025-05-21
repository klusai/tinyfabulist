import csv
import datetime
import glob
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
FABLES_FOLDER = "tinyfabulist/data/fables/"
TRANSLATIONS_FOLDER = "data/translations/"

# Constants for concurrency and batching
BATCH_SIZE = 50                  # Maximum number of prompts to process in a batch
MAX_CONCURRENCY = 50 # Maximum number of concurrent requests

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

async def generate_translation_async(system_prompt: str, fable_text: str, base_url: str, api_key: str) -> str:
    """Async translation call using the same LLM client"""
    client = get_client(base_url, api_key)
    attempt = 0
    backoff = INITIAL_RETRY_DELAY
    while attempt < MAX_RETRIES:
        try:
            chat_completion = await client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": fable_text},
                ],
                max_tokens=1000,
                temperature=0.7,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            attempt += 1
            logger.error(f"Error during translation API call (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                backoff = min(30, backoff * 1.5) + (random.random() * 2)
                await asyncio.sleep(backoff)
            else:
                raise

def load_translation_prompt_config(path: str) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        tpl = cfg.get('translator', {}).get('prompt', {}).get('translation')
        system = cfg.get('translator', {}).get('prompt', {}).get('system')
        if not tpl or not system:
            raise KeyError
        return {'system': system, 'template': tpl}
    except Exception as e:
        logger.error(f"Failed to load translation prompt config: {e}")
        raise ConfigError("Invalid file")


def format_translation_prompt(template: str, **kwargs) -> str:
    # Simple placeholder replacement
    result = template
    for key, val in kwargs.items():
        result = result.replace(f"{{{{{key}}}}}", val)
    return result


async def process_single_translation(
    entry: dict,
    system_prompt: str,
    template: str,
    base_url: str,
    api_key: str,
    output_file,
    semaphore,
):
    async with semaphore:
        fable = entry.get('fable')
        if not fable:
            return
        content = format_translation_prompt(
            template,
            target_language=entry.get('target_lang', 'ro'),
            fable_text=fable
        )
        translated = await generate_translation_async(system_prompt, content, base_url, api_key)
        # Keep original fable and store translation in a new field
        entry['fable'] = fable
        entry['translated_fable'] = translated
        entry['pipeline_stage'] = 'translation'
        json.dump(entry, output_file, ensure_ascii=False)
        output_file.write("\n")
        await asyncio.sleep(3)  # 3-second delay between requests


async def async_generate_translations(
    fable_files: list,
    args,
    translations_folder: str=TRANSLATIONS_FOLDER, 
    suggested_improvements: str = "",
    max_fables: int = 10_000_000
):
    # Load translation prompt
    cfg = load_translation_prompt_config('tinyfabulist/conf/translator_prompt.yaml')
    system_prompt = cfg['system']
    template = cfg['template']
    api_key = config('HF_ACCESS_TOKEN')

    if suggested_improvements:
        system_prompt += f"\n\nHere are some suggested improvements for the translation:\n{suggested_improvements}"
        template += f"\n\nHere are some suggested improvements for the translation:\n{suggested_improvements}"

    with open('tinyfabulist/conf/translator.yaml', 'r', encoding='utf-8') as f:
        translator_cfg = yaml.safe_load(f)

    translator_llm = translator_cfg.get('translator_ro', {}).get('model')
    translator_endpoint = translator_cfg.get('translator_ro', {}).get('endpoint')
    logger.info(f"Translator endpoint: {translator_endpoint}, Translator LLM: {translator_llm}")

    os.makedirs(translations_folder, exist_ok=True)
    timestamp = time.strftime("%y%m%d-%H%M%S")
    out_path = os.path.join(translations_folder, f"translations_{args.source_lang}-{args.target_lang}_dt{timestamp}.jsonl")
    
    # Count total number of entries to process
    total_entries = 0
    for ffile in fable_files:
        with open(ffile, 'r', encoding='utf-8') as inf:
            for line in inf:
                if line.strip():
                    total_entries += 1
    
    logger.info(f"Processing {total_entries} entries for translation")
    
    with open(out_path, 'w', encoding='utf-8') as outfile:
        semaphore = asyncio.Semaphore(getattr(args, 'max_concurrency', MAX_CONCURRENCY))
        tasks = []
        
        processed_count = 0
        for ffile in fable_files:
            with open(ffile, 'r', encoding='utf-8') as inf:
                for line in inf:
                    if not line.strip(): continue
                    try:
                        entry = json.loads(line)
                        entry['source_lang'] = args.source_lang
                        entry['target_lang'] = args.target_lang
                        
                        # Skip if exceeding max_fables limit
                        if max_fables and processed_count >= max_fables:
                            break
                            
                        processed_count += 1
                        task = process_single_translation(
                            entry,
                            system_prompt,
                            template,
                            translator_endpoint,
                            api_key,
                            outfile,
                            semaphore
                        )
                        tasks.append(task)
                    except json.JSONDecodeError:
                        logger.error(f"Error parsing JSON line in {ffile}")
        
        # Show progress while translating
        if tasks:
            logger.info(f"Starting translation of {len(tasks)} tasks")
            for i, task_completed in enumerate(asyncio.as_completed(tasks)):
                try:
                    await task_completed
                    if (i+1) % 5 == 0 or i+1 == len(tasks):
                        logger.info(f"Completed {i+1}/{len(tasks)} translations")
                except Exception as e:
                    logger.error(f"Error in translation task: {e}")
                    
    logger.info(f"Translations saved to {out_path}")
    return out_path

def process_evaluation_explanations(explanation_data):
    """
    Process evaluation explanations from different formats into usable improvement guidance
    
    Handles:
    - List of explanation strings
    - Single explanation string
    - JSON-formatted explanation with specific criteria
    
    Returns a structured set of improvement points
    """
    improvement_points = []
    
    # Handle if it's already a list of strings
    if isinstance(explanation_data, list):
        for item in explanation_data:
            if isinstance(item, str):
                # Clean up the explanation text
                cleaned = item.strip().replace('"', '').replace('[', '').replace(']', '')
                if cleaned:
                    improvement_points.append(cleaned)
    
    # Handle single string that might be JSON-formatted
    elif isinstance(explanation_data, str):
        # Check if it's a JSON string and try to parse it
        if explanation_data.startswith('[') and explanation_data.endswith(']'):
            try:
                parsed = json.loads(explanation_data)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, str):
                            improvement_points.append(item.strip())
            except json.JSONDecodeError:
                # If it's not valid JSON, treat as plain text
                improvement_points.append(explanation_data.strip())
        else:
            # Plain text explanation
            improvement_points.append(explanation_data.strip())
    
    # Return all unique improvement points
    return list(set(improvement_points))

async def evaluate_translation(translated_entry, api_key, endpoint):
    """Evaluate a translated fable and return evaluation results"""
    client = get_client(endpoint, api_key)
    
    original = translated_entry.get('fable', '')
    translation = translated_entry.get('translated_fable', '')
    
    if not original or not translation:
        return None
    
    system_prompt = """You are a professional literary translator evaluator. You will be given an original text in English and a translation in Romanian.
Your job is to evaluate the translation on the following criteria:
1. Accuracy (1-10): How well the translation preserves the original's meaning
2. Fluency (1-10): How natural and grammatically correct the translation is
3. Style preservation (1-10): How well the translation maintains the original's style and tone
4. Moral clarity (1-10): How clearly the moral of the fable is conveyed

For each criterion, provide a brief explanation of your score.
Also provide 2-3 specific suggestions for improvement."""

    user_prompt = f"""Original text (English):
{original}

Translation (Romanian):
{translation}

Please evaluate this translation according to the criteria."""
    
    try:
        response = await client.chat.completions.create(
            model="meta-llama/Llama-3.1-70B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=800,
            temperature=0.2,
        )
        evaluation_text = response.choices[0].message.content
        
        # Extract scores and explanations from the evaluation
        evaluation_result = {
            "full_evaluation": evaluation_text,
            "explanation": []
        }
        
        # Extract explanations and suggestions
        for line in evaluation_text.split("\n"):
            if "score is" in line.lower() or "because" in line.lower():
                evaluation_result["explanation"].append(line.strip())
            elif "suggestion" in line.lower() or "improve" in line.lower():
                if "suggested_improvements" not in evaluation_result:
                    evaluation_result["suggested_improvements"] = line.strip()
                else:
                    evaluation_result["suggested_improvements"] += "\n" + line.strip()
        
        return evaluation_result
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return None

async def evaluation_step(translation_file, api_key, endpoint):
    """Process a translation file and add evaluations"""
    evaluated_entries = []
    logger.info(f"Evaluating translations in {translation_file}")
    
    try:
        with open(translation_file, 'r', encoding='utf-8') as f:
            entries = [json.loads(line) for line in f if line.strip()]
        
        total = len(entries)
        logger.info(f"Found {total} entries to evaluate")
        
        # Use semaphore to limit concurrent evaluations
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent evaluations
        
        async def evaluate_with_semaphore(entry):
            async with semaphore:
                evaluation = await evaluate_translation(entry, api_key, endpoint)
                if evaluation:
                    entry['evaluation'] = evaluation
                return entry
        
        # Create tasks for all entries
        tasks = [evaluate_with_semaphore(entry) for entry in entries]
        
        # Process evaluations and show progress
        for i, task_completed in enumerate(asyncio.as_completed(tasks)):
            try:
                entry = await task_completed
                evaluated_entries.append(entry)
                if (i+1) % 5 == 0 or (i+1) == total:
                    logger.info(f"Evaluated {i+1}/{total} translations")
            except Exception as e:
                logger.error(f"Error in evaluation task: {e}")
        
        # Save evaluated entries to new file
        eval_file = translation_file.replace('.jsonl', '_evaluated.jsonl')
        with open(eval_file, 'w', encoding='utf-8') as f:
            for entry in evaluated_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(evaluated_entries)} evaluated entries to {eval_file}")
        return eval_file
    
    except Exception as e:
        logger.error(f"Error during evaluation step: {e}")
        return None

async def translation_routine(fable_files: list, args, iterations: int = 3):
    from datetime import datetime
    translation_folder = f"data/translations/{datetime.now().strftime('%y%m%d-%H%M%S')}/"
    os.makedirs(translation_folder, exist_ok=True)
    suggested_improvements = ""
    explanations = []
    
    # Get API key for evaluations
    api_key = config('HF_ACCESS_TOKEN')
    # Load translator config for evaluation endpoint
    with open('tinyfabulist/conf/translator.yaml', 'r', encoding='utf-8') as f:
        translator_cfg = yaml.safe_load(f)
    evaluation_endpoint = translator_cfg.get('translator_ro', {}).get('endpoint')
    
    logger.info(f"Starting translation routine with {iterations} iterations")
    
    try:
        last_translation_file = None
        for iter in range(iterations):
            logger.info(f"Starting iteration {iter+1}/{iterations}")
            
            if iter > 0 and last_translation_file:
                logger.info(f"Suggested improvements: {suggested_improvements}, ")
                # First evaluate the previous translations
                logger.info(f"Evaluating translations from previous iteration")
                evaluated_file = await evaluation_step(last_translation_file, api_key, evaluation_endpoint)
                
                if not evaluated_file:
                    logger.warning("No evaluations created, continuing with original translations")
                    evaluated_file = last_translation_file
                
                # Now collect improvements from evaluations
                if os.path.exists(evaluated_file):
                    with open(evaluated_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            try:
                                entry = json.loads(line)
                                
                                # Extract suggested improvements
                                suggested_improvements_entry = entry.get('evaluation', {}).get('suggested_improvements', '')
                                if suggested_improvements_entry:
                                    suggested_improvements += f"{suggested_improvements_entry}\n\n"
                                
                                # Extract explanation details
                                explanation_data = entry.get('evaluation', {}).get('explanation')
                                if explanation_data:
                                    processed_explanations = process_evaluation_explanations(explanation_data)
                                    explanations.extend(processed_explanations)
                                    
                            except json.JSONDecodeError:
                                logger.error(f"Error parsing JSON in {evaluated_file}")
                
                # Create improvement context from explanations
                improvement_context = ""
                if explanations:
                    improvement_context = "Based on previous translation evaluations, please improve on these aspects:\n\n"
                    for i, explanation in enumerate(explanations):
                        improvement_context += f"{i+1}. {explanation}\n"
                
                # Combine with suggested improvements
                if suggested_improvements:
                    improvement_context += f"\n\nSpecific improvement suggestions:\n{suggested_improvements}"
                
                logger.info(f"Created improvement context with {len(explanations)} points")
                
                # Run translation with different parameters based on iteration
                if iter == iterations - 1:
                    logger.info(f"Final iteration - using all collected feedback")
                    last_translation_file = await async_generate_translations(
                        fable_files, 
                        args, 
                        translation_folder, 
                        improvement_context
                    )
                else:
                    logger.info(f"Iteration {iter+1} - processing with feedback")
                    last_translation_file = await async_generate_translations(
                        fable_files, 
                        args, 
                        translation_folder, 
                        improvement_context,
                        max_fables=5
                    )
            else:
                # First iteration - process initial fable files
                logger.info("First iteration - processing input files")
                last_translation_file = await async_generate_translations(
                    fable_files, 
                    args, 
                    translation_folder, 
                    max_fables=5
                )
            
            # Log completion of current iteration
            logger.info(f"Completed iteration {iter+1}/{iterations}")
            
            # Allow some time between iterations
            if iter < iterations - 1:
                await asyncio.sleep(1)
        
        # Final evaluation for the last batch
        if last_translation_file:
            logger.info("Running final evaluation on translations")
            final_evaluated_file = await evaluation_step(last_translation_file, api_key, evaluation_endpoint)
            if final_evaluated_file:
                logger.info(f"Final evaluated translations saved to {final_evaluated_file}")
    
    except Exception as e:
        logger.error(f"Error in translation routine: {e}")
        
    logger.info(f"Translation routine completed - results in {translation_folder}")

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
    
    result["pipeline_stage"] = "fable"
    
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

    # Versioning and changelog
    version_info = get_version_info()
    changelog_path = update_changelog(version_info)
    logger.info(f"Current version: {version_info['version']}")
    logger.info(f"Changelog updated at {changelog_path}")

    settings = load_settings()

    # 1) Generate prompts
    if getattr(args, 'generate_prompts', False):
        system_prompt, fable_templates = generate_prompts(
            settings, count=args.count, randomize=args.randomize
        )
        write_generated_prompts(system_prompt, fable_templates)
        return

    # 2) Generate translations
    if getattr(args, 'generate_translations', False):
        # Recursively collect every JSONL under data/fables/
        from pathlib import Path
        base = Path(FABLES_FOLDER)
        fable_files = ['/home/andrei/Documents/Work/tinyfabulist/data/fables/llama-3-1-8b-instruct-l4t/tf_fables_llama-3-1-8b-instruct-l4t_dt250412-221506.jsonl']
        if not fable_files:
            raise ConfigError(f"No fable files found to translate under {FABLES_FOLDER!r}")

        logger.info(
            f"Translating {len(fable_files)} fable files from {args.source_lang} to {args.target_lang}"
        )
        await translation_routine(fable_files, args)
        elapsed = time.time() - start_time
        logger.info(f"Translation completed in {elapsed:.2f} seconds")
        return

    # 3) Generate fables
    if getattr(args, 'generate_fables', None):
        available_models = settings.get("llms", {}).get("hf-models", {})
        if not available_models:
            raise ConfigError("No models found in configuration")

        models_to_use = args.models if args.models else list(available_models.keys())
        invalid = [m for m in models_to_use if m not in available_models]
        if invalid:
            raise ConfigError(f"Invalid models: {', '.join(invalid)}")

        prompts = list(read_prompts(args.generate_fables))
        system_prompt = next(
            (p["content"] for p in prompts if p["prompt_type"] == "system_prompt"), None
        )
        fable_prompts = [p["content"] for p in prompts if p["prompt_type"] == "generator_prompt"]
        if not system_prompt:
            raise ConfigError("No system prompt found in prompt file.")

        # Distributed worker splitting
        worker_id = int(os.environ.get("WORKER_ID", 0))
        total_workers = int(os.environ.get("TOTAL_WORKERS", 1))
        orig = len(fable_prompts)
        fable_prompts = [p for idx, p in enumerate(fable_prompts) if idx % total_workers == worker_id]
        logger.info(f"Worker {worker_id}/{total_workers}: Processing {len(fable_prompts)}/{orig} prompts")

        existing_hashes = load_existing_hashes(args.input_file, args.output)
        logger.info(f"Found {len(existing_hashes)} existing hashes in {args.input_file}")

        # Prepare output files per model
        output_files = {}
        for model_name in models_to_use:
            folder = os.path.join(FABLES_FOLDER, model_name)
            os.makedirs(folder, exist_ok=True)
            ts = time.strftime("%y%m%d-%H%M%S")
            if args.output == "csv":
                fname = f"tf_fables_{model_name}_dt{ts}.csv"
            elif args.output == "jsonl":
                fname = f"tf_fables_{model_name}_dt{ts}.jsonl"
            else:
                fname = f"tf_fables_{model_name}_dt{ts}.txt"
            path = os.path.join(folder, fname)
            f = open(path, "w", encoding="utf-8")
            if args.output == "csv":
                writer = csv.DictWriter(f, fieldnames=[
                    "language","model","prompt","fable","prompt_hash",
                    "llm_name","llm_input_tokens","llm_output_tokens",
                    "llm_inference_time","llm_inference_cost_usd",
                    "host_provider","host_dc_provider","host_dc_location",
                    "host_gpu","host_gpu_vram","host_cost_per_hour",
                    "generation_datetime","pipeline_version"
                ])
                writer.writeheader()
                f.flush()
            output_files[model_name] = f

        metadata = settings.get("metadata", {})
        metadata.update({"pipeline_version": version_info["version"]})

        # Run fable generation
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

        # Close files and log completion
        for f in output_files.values():
            f.close()
        elapsed = time.time() - start_time
        logger.info(f"Fable generation completed in {elapsed:.2f} seconds")
    else:
        raise ConfigError("Please specify --generate-prompts, --generate-fables, or --generate-translations")


def run_generate(args):
    """Synchronous wrapper for the async function"""
    # Create and run the event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_generate_async(args))


def add_generate_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        "generate", help="Generate fable prompts, fables, or translations"
    )
    generate_parser.add_argument(
        "--generate-prompts", action="store_true", help="Generate fable prompts"
    )
    generate_parser.add_argument(
        "--generate-fables", type=str, help="Generate fables from a JSONL prompt file"
    )
    generate_parser.add_argument(
        "--generate-translations", action="store_true", help="Generate fable translations"
    )
    generate_parser.add_argument(
        "--source-lang", type=str, default="en", help="Source language code"
    )
    generate_parser.add_argument(
        "--target-lang", type=str, default="ro", help="Target language code"
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
        help="Maximum number of concurrent requests (default: 10)",
    )
    generate_parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show progress bars for batch processing",
    )
    generate_parser.set_defaults(func=run_generate)