import os
import json
import argparse
import time
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import deepl  # Using the official DeepL library

from tiny_fabulist.logger import setup_logging
from tiny_fabulist.translate.subparser import add_translate_subparser

logger = setup_logging()

def translate_text(text: str, target_lang: str, auth_key: str,
                   max_retries: int = 3, backoff_factor: float = 5.0) -> str:
    """
    Translate text using the DeepL library with a retry mechanism.
    
    Parameters:
        text: Text to translate.
        target_lang: Target language code (e.g., 'RO' for Romanian).
        auth_key: DeepL API authentication key.
        max_retries: Number of retry attempts before giving up.
        backoff_factor: Seconds to sleep between retries.
        
    Returns:
        Translated text (or the original text if translation fails).
    """
    for attempt in range(1, max_retries + 1):
        try:
            translator = deepl.Translator(auth_key)
            result = translator.translate_text(text, target_lang=target_lang)
            return result.text
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
                     target_lang: str,
                     auth_key: str) -> Dict[str, Any]:
    """
    Translate specified fields in a record.
    
    Parameters:
        record: A dictionary representing a JSONL record.
        fields: List of fields to translate.
        target_lang: Target language code.
        auth_key: DeepL API authentication key.
        
    Returns:
        The record with the specified fields translated and a 'language' field updated.
    """
    for field in fields:
        if field in record and record[field]:
            record[field] = translate_text(record[field], target_lang, auth_key)
            # Small delay to avoid hitting API rate limits too quickly
            time.sleep(0.1)
    record['language'] = target_lang.lower()
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
                    target_lang: str,
                    auth_key: str,
                    batch_size: int = 100,
                    fields_to_translate: Optional[List[str]] = None,
                    max_workers: int = 10) -> None:
    """
    Translate records in a JSONL file concurrently.
    
    Parameters:
        input_file: Path to the input JSONL file.
        output_file: Path to the output JSONL file.
        target_lang: Target language code.
        auth_key: DeepL API authentication key.
        batch_size: Number of records to process before saving progress.
        fields_to_translate: List of JSON fields to translate.
        max_workers: Maximum number of threads to use.
    """
    if fields_to_translate is None:
        fields_to_translate = ['prompt', 'fable']
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Count total lines for progress tracking
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    translated_records = []
    processed_count = 0
    
    # Use a ThreadPoolExecutor to process records concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with open(input_file, 'r', encoding='utf-8') as f, tqdm(total=total_lines, desc="Translating") as pbar:
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
                        target_lang,
                        auth_key
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
    Main function to translate fables from a JSONL file.
    """
    load_dotenv()
    
    auth_key = os.getenv('DEEPL_AUTH_KEY')
    print(auth_key)
    if not auth_key:
        logger.critical("DEEPL_AUTH_KEY must be set in the .env file")
        raise ValueError("DEEPL_AUTH_KEY must be set in the .env file")
    
    output_file = args.output
    if not output_file:
        base_name = os.path.basename(args.input)
        name_parts = os.path.splitext(base_name)
        output_dir = os.path.join('data', 'translations')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir,
            f"{name_parts[0]}_translation_{args.target_lang.lower()}{name_parts[1]}"
        )
    
    logger.info(f"Translating {args.input} to {args.target_lang}")
    logger.info(f"Output will be saved to {output_file}")
    
    translate_jsonl(
        input_file=args.input,
        output_file=output_file,
        target_lang=args.target_lang,
        auth_key=auth_key,
        batch_size=args.batch_size,
        fields_to_translate=args.fields.split(',') if args.fields else ['fable', 'prompt'],
        max_workers=args.max_workers
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate JSONL content using DeepL API')
    subparsers = parser.add_subparsers()
    add_translate_subparser(subparsers,translate_fables,'EN', 'RO')
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
