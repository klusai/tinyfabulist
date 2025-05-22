import argparse
import asyncio
import json
import os
import random
import time
import orjson
from openai import AsyncOpenAI
import yaml
from decouple import config
import aiofiles
import gc
from functools import wraps
from collections import defaultdict
import uvloop  # Add uvloop for faster event loop
import httpx   # For HTTP/2 customization

from tinyfabulist.logger import setup_logging

# Use uvloop for faster event loop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

CLIENT_CACHE = {}
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
TRANSLATIONS_FOLDER = 'data/translations'
MAX_CONCURRENCY = 120
BATCH_SIZE = 20  # Process 8 fables per API call for optimal throughput
FABLES_FILES = ['/home/andrei/Documents/Work/tinyfabulist/data/fables/llama-3-1-8b-instruct-a10gt/tf_fables_llama-3-1-8b-instruct-a10gt_dt250412-221516.jsonl']
ARGS = argparse.Namespace(source_lang='English', target_lang='Romanian')

logger = setup_logging()

# Track active tasks for proper cleanup
ACTIVE_TASKS = set()

# Setup profiling metrics
PROFILING_ENABLED = True
PROFILING_STATS = defaultdict(list)
PROFILING_FILE = os.path.join('logs', 'profiling_stats.jsonl')

# File writer queue and entry processing queue
WRITE_QUEUE = asyncio.Queue()
ENTRY_QUEUE = asyncio.Queue()  # Queue for worker pool
WRITE_BATCH_SIZE = 40 

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Profiling decorator to measure performance
import inspect

def profile(name):
    def decorator(func):
        # If this is an async generator function…
        if inspect.isasyncgenfunction(func):
            @wraps(func)
            async def gen_wrapper(*args, **kwargs):
                agen = func(*args, **kwargs)
                count = 0
                async for item in agen:
                    start = time.perf_counter()
                    yield item
                    elapsed = time.perf_counter() - start
                    PROFILING_STATS[name].append(elapsed)
                    count += 1
                    if count % 20 == 0:
                        avg = sum(PROFILING_STATS[name]) / len(PROFILING_STATS[name])
                        logger.info(f"PROFILE - {name}: avg {avg:.4f}s over {len(PROFILING_STATS[name])} calls")
            return gen_wrapper

        # Otherwise it's a "normal" async function
        @wraps(func)
        async def coro_wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            PROFILING_STATS[name].append(elapsed)
            if len(PROFILING_STATS[name]) % 20 == 0:
                avg = sum(PROFILING_STATS[name]) / len(PROFILING_STATS[name])
                logger.info(f"PROFILE - {name}: avg {avg:.4f}s over {len(PROFILING_STATS[name])} calls")
            return result
        return coro_wrapper

    return decorator

#### HELPER FUNCTIONS ####

def yaml_safe_load(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    
def get_prompts():
    prompts_cfg = yaml_safe_load('tinyfabulist/conf/translator_prompt.yaml')
    prompts_cfg = prompts_cfg['translator']['prompt']

    system_prompt = prompts_cfg['system']
    template = prompts_cfg['translation']

    return system_prompt, template

def get_translator():
    translator_cfg = yaml_safe_load('tinyfabulist/conf/translator.yaml')

    translator_llm = translator_cfg.get('translator_ro', {}).get('model')
    translator_endpoint = translator_cfg.get('translator_ro', {}).get('endpoint')

    return translator_llm, translator_endpoint


# module-level, not inside any function
PROMPTS_SYSTEM, PROMPTS_TEMPLATE = get_prompts()
TRANSLATOR_CFG = get_translator()
API_KEY = config('HF_ACCESS_TOKEN')

# Configure HTTP/2 connection limits for optimal pooling
HTTP_MAX_CONNECTIONS = 100
HTTP_MAX_KEEPALIVE_CONNECTIONS = 20
HTTP_KEEPALIVE_EXPIRY = 60  # seconds

class DependecyContainer:
    def __init__(self):
        self.client = None
        self.prompts = PROMPTS_SYSTEM, PROMPTS_TEMPLATE
        self.translator = TRANSLATOR_CFG
        self.api_key = API_KEY
        self.max_tokens = 1000
        self.temperature = 0.7

# Create once at module level
DEPENDENCY_CONTAINER = DependecyContainer()

def get_client():
    """Get or create an AsyncOpenAI client for any endpoint"""
    model_name, endpoint = DEPENDENCY_CONTAINER.translator
    api_key = DEPENDENCY_CONTAINER.api_key
        
    # Log the endpoint being used
    logger.info(f"Using endpoint: {endpoint}")
    
    cache_key = endpoint
    if cache_key not in CLIENT_CACHE:
        try:
            logger.info(f"Creating AsyncOpenAI client for {endpoint}")
            client = AsyncOpenAI(
                base_url=endpoint, 
                api_key=api_key,
                timeout=60.0,
                max_retries=3
            )
            CLIENT_CACHE[cache_key] = client
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            raise
    
    return CLIENT_CACHE[cache_key]

async def execute_llm_call(system_prompt: str, user_prompt: str, model: str):
    """Call the LLM API with appropriate format for the endpoint"""
    max_tokens = DEPENDENCY_CONTAINER.max_tokens
    temperature = DEPENDENCY_CONTAINER.temperature
    
    client = get_client()
    _, endpoint = DEPENDENCY_CONTAINER.translator
    
    # Log request details for debugging
    logger.debug(f"Sending request to endpoint: {endpoint}")
    logger.debug(f"System prompt: {system_prompt[:50]}...")
    logger.debug(f"User prompt: {user_prompt[:50]}...")
    
    try:
        # Try standard format first (OpenAI-compatible)
        try:
            logger.debug("Attempting OpenAI-compatible chat completion format")
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.debug("OpenAI format succeeded")
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            logger.warning(f"OpenAI format failed: {e}")
            
            # If 404 or other indication endpoint doesn't support OpenAI format
            if '404' in error_msg or 'not found' in error_msg:
                # Try Hugging Face text generation API format
                logger.debug("Attempting Hugging Face text generation format")
                
                # Prepare combined prompt
                combined_prompt = f"{system_prompt}\n\n{user_prompt}"
                
                # Make a direct POST request to the endpoint
                base_url = client.base_url
                headers = {
                    "Authorization": f"Bearer {client.api_key}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    payload = {
                        "inputs": combined_prompt,
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "temperature": temperature,
                            "return_full_text": False
                        }
                    }
                    
                    logger.debug(f"Sending direct request to: {base_url}")
                    response = await http_client.post(
                        str(base_url), 
                        json=payload,
                        headers=headers
                    )
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if isinstance(result, list) and result:
                        # Standard HF format
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        # Alternative format
                        return result.get('generated_text', '') or result.get('text', '')
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return "Error: Unexpected response format"
            else:
                # Not a 404 error, re-raise
                raise
    except Exception as e:
        logger.error(f"Error during LLM call: {e}")
        return None

def format_translation_prompt(template: str, **kwargs) -> str:
    # Simple placeholder replacement
    result = template
    for key, val in kwargs.items():
        result = result.replace(f"{{{{{key}}}}}", val)
    return result

#### TRANSLATION FUNCTIONS ####

@profile("translation")
async def generate_translation_async(system_prompt: str, fable_text: str) -> str:
    """Async translation call using direct HTTP requests"""
    attempt = 0
    backoff = INITIAL_RETRY_DELAY
    model = DEPENDENCY_CONTAINER.model
    
    while attempt < MAX_RETRIES:
        try:
            # Call the LLM API directly
            response = await execute_llm_call(system_prompt, fable_text, model)
            if response:
                return response
            raise Exception("Empty response from LLM")
        except Exception as e:
            attempt += 1
            logger.error(f"Error during translation API call (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                backoff = min(30, backoff * 1.5) + (random.random() * 2)
                await asyncio.sleep(backoff)
            else:
                raise

# Function to read JSONL entries asynchronously
async def read_entries(fable_files, max_fables=10_000_000, source_lang='English', target_lang='Romanian'):
    """Async generator that yields entries one at a time from fable files"""
    processed_count = 0
    
    for ffile in fable_files:
        async with aiofiles.open(ffile, 'r', encoding='utf-8') as inf:
            async for line in inf:
                if not line.strip(): 
                    continue
                    
                try:
                    entry = json.loads(line)
                    entry['source_lang'] = source_lang
                    entry['target_lang'] = target_lang
                    
                    # Skip if exceeding max_fables limit
                    if max_fables and processed_count >= max_fables:
                        return
                        
                    processed_count += 1
                    yield entry
                except json.JSONDecodeError:
                    logger.error(f"Error parsing JSON line in {ffile}")

# File writer task
async def file_writer_task(output_file):
    """Background task to handle file writing without blocking the main process"""
    buffer = []
    last_flush_time = time.time()
    
    while True:
        try:
            # Get an entry from the queue with a timeout
            try:
                entry_bytes = await asyncio.wait_for(WRITE_QUEUE.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # If the buffer has items and it's been a while since last flush, write them
                current_time = time.time()
                if buffer and (current_time - last_flush_time > 5 or len(buffer) > WRITE_BATCH_SIZE/2):
                    await output_file.write(b''.join(buffer))
                    last_flush_time = current_time
                    buffer.clear()
                continue
                
            # Add to buffer
            buffer.append(entry_bytes)
            
            # Mark task as done in the queue
            WRITE_QUEUE.task_done()
            
            # Only flush if buffer is large enough or it's been a while
            if len(buffer) >= WRITE_BATCH_SIZE or (WRITE_QUEUE.empty() and buffer):
                await output_file.write(b''.join(buffer))
                await output_file.flush()  # Only flush at batch boundaries
                last_flush_time = time.time()
                buffer.clear()
                
        except asyncio.CancelledError:
            # Final flush of buffer when cancelled
            if buffer:
                await output_file.write(b''.join(buffer))
                await output_file.flush()
            raise
        except Exception as e:
            logger.error(f"Error in file writer task: {e}")
            # Even on error, try to flush what we have
            if buffer:
                try:
                    await output_file.write(b''.join(buffer))
                    await output_file.flush()
                except Exception:
                    pass  # If this fails too, we've done our best

# Function to wait for all active tasks
async def wait_for_all_tasks():
    if not ACTIVE_TASKS:
        return
    pending = list(ACTIVE_TASKS)
    logger.info(f"Waiting for {len(pending)} pending tasks to complete...")
    while pending:
        done, pending = await asyncio.wait(pending, timeout=5, return_when=asyncio.FIRST_COMPLETED)
        logger.info(f"Progress: {len(ACTIVE_TASKS) - len(pending)}/{len(ACTIVE_TASKS)} tasks completed")
        
# Track task completion and remove from active set
def task_done_callback(task):
    try:
        # Get result to propagate any exceptions
        task.result()
    except Exception as e:
        logger.error(f"Task failed with error: {e}")
    finally:
        # Remove from active tasks
        ACTIVE_TASKS.discard(task)

async def process_single_translation(
    entry: dict,
    system_prompt: str,
    template: str,
    semaphore,
):
    try:
        original_fable = entry.get('fable')
        if not original_fable:
            return

        # 1) Initial translation
        prompt_content = format_translation_prompt(
            template,
            target_language=entry.get('target_lang', 'ro'),
            fable_text=original_fable
        )
        translated = await generate_translation_async(system_prompt, prompt_content)

        # Use a timestamp once instead of calling time.strftime for each entry
        generation_time = int(time.time())
        
        # Only keep essential fields to reduce memory usage
        output_entry = {
            'fable': original_fable,
            'translated_fable': translated,
            'pipeline_stage': 'translation',
            'source_lang': entry.get('source_lang'),
            'target_lang': entry.get('target_lang'),
            'llm_name': DEPENDENCY_CONTAINER.model,
            'generation_timestamp': generation_time,
        }

        # 3) Put onto write queue instead of writing directly
        # Use orjson for faster serialization and write bytes directly
        entry_bytes = orjson.dumps(output_entry) + b'\n'
        await WRITE_QUEUE.put(entry_bytes)

    finally:
        # Always release the semaphore
        semaphore.release()

# Worker function for the worker pool - simplified to process individual entries
async def translation_worker(worker_id, semaphore, system_prompt, template):
    """Worker that processes entries from the queue"""
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get an entry from the queue
            entry = await ENTRY_QUEUE.get()
            
            # Acquire semaphore for processing
            await semaphore.acquire()
            
            # Process the single entry
            await process_single_translation(entry, system_prompt, template, semaphore)
            
            # Mark task as done in the queue
            ENTRY_QUEUE.task_done()
                
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            # Make sure to release semaphore on error
            try:
                semaphore.release()
            except:
                pass

#### PARALLEL TRANSLATION ####

async def async_generate_translations(
    fable_files: list = FABLES_FILES,
    args: dict = ARGS,
    translations_folder: str=TRANSLATIONS_FOLDER, 
    suggested_improvements: str = "",
    max_fables: int = 10_000_000
):
    logger.info("Translation started - using optimized streaming approach with worker pool")
    logger.info(f"Translator endpoint: {DEPENDENCY_CONTAINER.translator}")
    logger.info(f"Max concurrency: {MAX_CONCURRENCY}")
    
    # Load translation prompt
    system_prompt, template = DEPENDENCY_CONTAINER.prompts

    if suggested_improvements:
        system_prompt += f"\n\nHere are some suggested improvements for the translation:\n{suggested_improvements}"
        template += f"\n\nHere are some suggested improvements for the translation:\n{suggested_improvements}"

    os.makedirs(translations_folder, exist_ok=True)
    timestamp = time.strftime("%y%m%d-%H%M%S")
    out_path = os.path.join(translations_folder, f"translations_{args.source_lang}-{args.target_lang}_dt{timestamp}.jsonl")
    
    # Create bounded semaphore with a slight buffer to prevent stalling
    semaphore = asyncio.BoundedSemaphore(MAX_CONCURRENCY + 10)  # Add buffer to prevent stalling
    
    # Use a prefetch queue with larger size to ensure workers always have entries
    entry_prefetch_task = None
    entry_prefetch_running = True
    
    async def prefetch_entries(fable_files, max_fables, source_lang, target_lang):
        """Prefetches entries into queue without blocking on queue size"""
        nonlocal entry_prefetch_running
        processed_count = 0
        try:
            entries = read_entries(fable_files, max_fables, source_lang, target_lang)
            async for entry in entries:
                if not entry_prefetch_running:
                    break
                await ENTRY_QUEUE.put(entry)
                processed_count += 1
                
                # Log progress but don't block on it
                if processed_count % 100 == 0:
                    logger.info(f"Prefetched {processed_count} entries for processing")
            
            logger.info(f"Entry prefetching completed, queued {processed_count} entries")
            return processed_count
        except Exception as e:
            logger.error(f"Error during entry prefetching: {e}")
            return processed_count
    
    processed_count = 0
    
    # Process entries through streaming approach with worker pool
    async with aiofiles.open(out_path, 'wb') as outfile:  # Open in binary mode for direct bytes writing
        # Start the background file writer task
        writer_task = asyncio.create_task(file_writer_task(outfile))
        
        # Start worker pool - create MAX_CONCURRENCY workers
        workers = []
        for i in range(MAX_CONCURRENCY):  # Create one worker per concurrency slot
            worker = asyncio.create_task(
                translation_worker(i, semaphore, system_prompt, template)
            )
            workers.append(worker)
            ACTIVE_TASKS.add(worker)
        
        start_time = time.perf_counter()
        
        try:
            # Start prefetching entries in the background
            entry_prefetch_task = asyncio.create_task(
                prefetch_entries(fable_files, max_fables, args.source_lang, args.target_lang)
            )
            
            # Wait for all entries to be processed
            while True:
                # Check if prefetcher is still running
                if entry_prefetch_task.done():
                    processed_count = entry_prefetch_task.result()
                    if ENTRY_QUEUE.empty():
                        break
                
                # Periodically check and log progress
                queue_size = ENTRY_QUEUE.qsize()
                if processed_count > 0:
                    elapsed = time.perf_counter() - start_time
                    rate = processed_count / elapsed if elapsed > 0 else 0
                
                # Periodic garbage collection
                if processed_count % 500 == 0:
                    gc.collect()
                    
                # Give other tasks a chance to run
                await asyncio.sleep(2.0)  # Check progress every 2 seconds
            
            logger.info(f"All {processed_count} entries queued, waiting for processing to complete...")
            await ENTRY_QUEUE.join()
            
            # Cancel all workers
            for worker in workers:
                worker.cancel()
                
            # Wait for cancellation to complete
            try:
                await asyncio.gather(*workers, return_exceptions=True)
            except asyncio.CancelledError:
                pass
            
            # Wait for write queue to be empty
            await WRITE_QUEUE.join()
            
        finally:
            # Stop prefetching
            entry_prefetch_running = False
            if entry_prefetch_task and not entry_prefetch_task.done():
                entry_prefetch_task.cancel()
                try:
                    await entry_prefetch_task
                except asyncio.CancelledError:
                    pass
            
            # Cancel all tasks
            for worker in workers:
                if not worker.done():
                    worker.cancel()
                    
            # Cancel the writer task
            writer_task.cancel()
            try:
                await writer_task
            except asyncio.CancelledError:
                pass
    
    # Log performance info
    if PROFILING_ENABLED:
        logger.info("=== PERFORMANCE REPORT ===")
        for name, times in PROFILING_STATS.items():
            if times:
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                logger.info(f"{name}: avg {avg_time:.4f}s, total {total_time:.2f}s, calls {len(times)}")
        
        # Save profiling stats to file for later analysis
        save_profiling_stats()
    
    logger.info(f"Translations completed and saved to {out_path}")
    return out_path

# Function to save profiling stats to file
def save_profiling_stats():
    """Save the collected profiling stats to a file for later analysis"""
    if not PROFILING_ENABLED or not PROFILING_STATS:
        return
        
    os.makedirs(os.path.dirname(PROFILING_FILE), exist_ok=True)
    
    # Prepare summary data
    stats_summary = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'stats': {}
    }
    
    for name, times in PROFILING_STATS.items():
        if not times:
            continue
            
        avg_time = sum(times) / len(times)
        total_time = sum(times)
        max_time = max(times)
        min_time = min(times)
        
        stats_summary['stats'][name] = {
            'avg_time': avg_time,
            'total_time': total_time,
            'calls': len(times),
            'max_time': max_time,
            'min_time': min_time,
            'p95_time': sorted(times)[int(len(times) * 0.95)] if len(times) > 20 else None
        }
    
    # Write to file - append to existing file
    try:
        with open(PROFILING_FILE, 'a', encoding='utf-8') as f:
            f.write(orjson.dumps(stats_summary).decode('utf-8') + '\n')
        logger.info(f"Profiling stats saved to {PROFILING_FILE}")
    except Exception as e:
        logger.error(f"Failed to save profiling stats: {e}")

async def validate_endpoint_config():
    """Validate the endpoint URL and API key configuration"""
    _, endpoint = DEPENDENCY_CONTAINER.translator
    api_key = DEPENDENCY_CONTAINER.api_key
    model = DEPENDENCY_CONTAINER.model
    
    # Validate endpoint URL
    if not endpoint or not endpoint.startswith(('http://', 'https://')):
        logger.error(f"Invalid endpoint URL: {endpoint}. Must start with http:// or https://")
        return False
        
    # Simple validation of API key
    if not api_key or len(api_key) < 8:
        logger.error(f"API key appears invalid or too short")
        return False
        
    # Check if model is specified
    if not model:
        logger.error(f"No model specified")
        return False
        
    logger.info(f"Configuration validation passed. Endpoint: {endpoint}, Model: {model}")
    return True

async def test_connection():
    """Test connection to the endpoint with fallback mechanisms"""
    try:
        # Get endpoint info
        _, endpoint = DEPENDENCY_CONTAINER.translator
        api_key = DEPENDENCY_CONTAINER.api_key
        model = DEPENDENCY_CONTAINER.model
        
        # Log diagnostic info
        logger.info(f"Testing connection to endpoint: {endpoint}")
        logger.info(f"Using model: {model}")
        logger.info(f"API key length: {len(api_key)} chars, starts with: {api_key[:4]}...")
        
        # Simple test prompt
        system_prompt = "You are a helpful assistant."
        user_prompt = "Say hello."
        
        # Try a simple call
        response = await execute_llm_call(system_prompt, user_prompt, model)
        
        if response:
            logger.info(f"Connection test successful. Response: {response[:100]}...")
            return True
        else:
            logger.error("Connection test failed: Empty response")
            return False
            
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Print detailed error info
        logger.error(f"Error connecting to endpoint: {error_type}: {error_msg}")
        return False

async def main():
    # Validate configuration first
    if not await validate_endpoint_config():
        logger.error("Failed configuration validation. Check your .env file and translator.yaml")
        return
        
    # Test connection
    connection_ok = await test_connection()
    if not connection_ok:
        logger.error("Failed to connect to the model endpoint. Please check your configuration.")
        return
    
    # Run the translation process
    try:
        await async_generate_translations()
    except Exception as e:
        logger.error(f"Translation process failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Run the main async function
    asyncio.run(main())
