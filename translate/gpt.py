import os
import json
import argparse
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from tiny_fabulist.logger import setup_logging
from translate.subparser import add_translate_subparser

logger = setup_logging()

def translate_text(text: str, api_key: str, model: str, source_lang: str = "en", target_lang: str = "ro",
                   max_retries: int = 3, backoff_factor: float = 5.0) -> str:
    """
    Translate text using the ChatGPT API with a retry mechanism.
    
    Parameters:
        text: Text to translate.
        api_key: OpenAI API key for authentication.
        model: ChatGPT model to use (e.g., 'gpt-3.5-turbo').
        source_lang: Source language (e.g., 'en').
        target_lang: Target language (e.g., 'ro').
        max_retries: Number of retry attempts before giving up.
        backoff_factor: Seconds to sleep between retries.
        
    Returns:
        Translated text (or the original text if translation fails).
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    # Construct the prompt for translation
    prompt = f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}."
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt.strip()}
        ],
        "temperature": 0,
        "max_tokens": 1000
    }
    
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Extract the translated text from the response
                if "choices" in result and len(result["choices"]) > 0:
                    translated_text = result["choices"][0]["message"]["content"].strip()
                    return translated_text
                else:
                    return str(result)
            else:
                logger.error(
                    f"ChatGPT API returned status code {response.status_code}: {response.text}. "
                    f"Attempt {attempt}/{max_retries}."
                )
                time.sleep(backoff_factor)
        except Exception as e:
            logger.error(
                f"Translation error on attempt {attempt}/{max_retries}: {e}. "
                f"Sleeping for {backoff_factor} seconds."
            )
            time.sleep(backoff_factor)
    logger.error("Max retries exceeded. Returning original text.")
    return text

def translate_record(record: Dict[str, Any],
                     fields: List[str],
                     api_key: str,
                     model: str,
                     source_lang: str,
                     target_lang: str) -> Dict[str, Any]:
    """
    Translate specified fields in a record using the ChatGPT API.
    
    Parameters:
        record: A dictionary representing a JSONL record.
        fields: List of fields to translate.
        api_key: OpenAI API key for authentication.
        model: ChatGPT model to use (e.g., 'gpt-3.5-turbo').
        source_lang: Source language.
        target_lang: Target language.
        
    Returns:
        The record with the specified fields translated and a 'language' field updated.
    """
    for field in fields:
        if field in record and record[field]:
            record[field] = translate_text(record[field], api_key, model, source_lang, target_lang)
            time.sleep(0.1)
    record['language'] = target_lang.split('_')[0]  # e.g., 'ro'
    return record

def save_progress(records: List[Dict[str, Any]], output_file: str, is_first_batch: bool) -> None:
    """
    Save a batch of translated records to the output file.
    
    Parameters:
        records: List of translated records.
        output_file: Path to the output file.
        is_first_batch: If True, the file will be overwritten; otherwise, records are appended.
    """
    mode = 'w' if is_first_batch else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def translate_jsonl(input_file: str,
                    output_file: str,
                    api_key: str,
                    model: str,
                    source_lang: str,
                    target_lang: str,
                    batch_size: int = 100,
                    fields_to_translate: Optional[List[str]] = None,
                    max_workers: int = 10) -> None:
    """
    Translate JSONL file content using the ChatGPT API.
    
    Parameters:
        input_file: Path to the input JSONL file.
        output_file: Path to the output JSONL file.
        api_key: OpenAI API key for authentication.
        model: ChatGPT model to use (e.g., 'gpt-3.5-turbo').
        source_lang: Source language.
        target_lang: Target language.
        batch_size: Number of records to process before saving progress.
        fields_to_translate: List of JSON fields to translate.
        max_workers: Maximum number of threads to use.
    """
    if fields_to_translate is None:
        fields_to_translate = ['prompt', 'fable']
    
    # Count total lines for progress tracking
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    translated_records = []
    processed_count = 0
    
    # Use a ThreadPoolExecutor to process records concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with open(input_file, 'r', encoding='utf-8') as f, tqdm(total=total_lines, desc="Translating using ChatGPT") as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    record = json.loads(line)
                    future = executor.submit(
                        translate_record,
                        record,
                        fields_to_translate,
                        api_key,
                        model,
                        source_lang,
                        target_lang
                    )
                    futures.append(future)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decoding error: {e} - Line: {line}")
                    pbar.update(1)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    translated_records.append(result)
                    processed_count += 1
                    pbar.update(1)
                except Exception as e:
                    logger.exception(f"Unexpected error processing record: {e}")
                
                # Save progress after processing each batch
                if processed_count % batch_size == 0 and translated_records:
                    save_progress(translated_records, output_file, processed_count == batch_size)
                    translated_records.clear()
    
    # Save any remaining records
    if translated_records:
        save_progress(translated_records, output_file, False)
    
    logger.info(f"Translation complete. Processed {processed_count} records.")

def translate_fables(args):
    """
    Main function to translate fables from a JSONL file using the ChatGPT API.
    """
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.critical("OPENAI_API_KEY must be set in the .env file")
        raise ValueError("OPENAI_API_KEY must be set in the .env file")
    
    model = 'o3-mini-2025-01-31' #'gpt-4o'

    source_lang = args.source_lang
    target_lang = args.target_lang
    
    output_file = args.output
    if not output_file:
        base_name = os.path.basename(args.input)
        name_parts = os.path.splitext(base_name)
        output_dir = os.path.join('data', 'translations')
        timestamp = time.strftime("%y%m%d-%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{name_parts[0]}_translation_{target_lang}_{model}_{timestamp}.jsonl"
        )
    
    logger.info(f"Translating {args.input} from {source_lang} to {target_lang} using model: {model}")
    logger.info(f"Output will be saved to {output_file}")
    
    translate_jsonl(
        input_file=args.input,
        output_file=output_file,
        api_key=api_key,
        model=model,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=args.batch_size,
        fields_to_translate=args.fields.split(',') if args.fields else ['fable', 'prompt'],
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate JSONL content using the ChatGPT API')
    subparsers = parser.add_subparsers()
    add_translate_subparser(subparsers, translate_fables, 'en', 'ro')
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
