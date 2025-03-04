import os
import json
import argparse
import time
from typing import Dict, Any, List, Optional
import requests
from tqdm import tqdm
from dotenv import load_dotenv

from logger import setup_logging

logger = setup_logging()

def translate_text(text: str, target_lang: str, auth_key: str) -> str:
    """
    Translate text using DeepL API.
    
    Parameters:
        text: Text to translate
        target_lang: Target language code (e.g., 'RO' for Romanian)
        auth_key: DeepL API authentication key
        
    Returns:
        Translated text
    """
    url = "https://api-free.deepl.com/v2/translate"
    
    params = {
        "auth_key": auth_key,
        "text": text,
        "target_lang": target_lang
    }
    
    try:
        response = requests.post(url, data=params)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        return result["translations"][0]["text"]
    except requests.exceptions.RequestException as e:
        logger.error(f"Translation request error: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing translation response: {e}")

    return text  # Return original text on error

def translate_jsonl(input_file: str, output_file: str, target_lang: str, 
                   auth_key: str, batch_size: int = 100, 
                   fields_to_translate: Optional[List[str]] = None) -> None:
    """
    Translate content in a JSONL file to the target language.
    
    Parameters:
        input_file: Path to input JSONL file
        output_file: Path to output translated JSONL file
        target_lang: Target language code (e.g., 'RO' for Romanian)
        auth_key: DeepL API authentication key
        batch_size: Number of records to process before saving progress
        fields_to_translate: List of JSON fields to translate (default: ['fable'])
    """
    if fields_to_translate is None:
        fields_to_translate = ['prompt','fable']
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Count total lines for progress bar
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    # Process the file
    translated_records = []
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f, tqdm(total=total_lines, desc="Translating") as pbar:
        for line in f:
            try:
                record = json.loads(line.strip())
                
                # Translate specified fields
                for field in fields_to_translate:
                    if field in record and record[field]:
                        record[field] = translate_text(record[field], target_lang, auth_key)
                        # Add a small delay to avoid hitting API rate limits
                        time.sleep(0.1)
                
                # Update language field
                record['language'] = target_lang.lower()
                
                
                translated_records.append(record)
                processed_count += 1
                pbar.update(1)
                
                # Save progress in batches
                if processed_count % batch_size == 0:
                    save_progress(translated_records, output_file, processed_count == batch_size)
                    translated_records.clear()
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding error: {e} - Line: {line.strip()}")
            except Exception as e:
                logger.exception(f"Unexpected error processing line: {e}")
    
    # Save any remaining records
    if translated_records:
        save_progress(translated_records, output_file, False)
    
    logger.info(f"Translation complete. Processed {processed_count} records.")

def save_progress(records: List[Dict[str, Any]], output_file: str, is_first_batch: bool) -> None:
    """
    Save translated records to the output file.
    
    Parameters:
        records: List of translated records
        output_file: Path to output file
        is_first_batch: Whether this is the first batch (to overwrite or append)
    """
    mode = 'w' if is_first_batch else 'a'
    with open(output_file, mode, encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def translate_fables(args):
    """
    Main function to translate fables from a JSONL file.
    
    Parameters:
        args: Command line arguments
    """
    load_dotenv()

    auth_key = os.getenv('DEEPL_AUTH_KEY')
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
        fields_to_translate=args.fields.split(',') if args.fields else ['fable','prompt']
    )

def add_translate_subparser(subparsers) -> None:
    """
    Add the translate subparser to the main parser.
    
    Parameters:
        subparsers: Subparsers object from the main parser
    """
    translate_parser = subparsers.add_parser(
        'translate', 
        help='Translate content in a JSONL file to a specified language using DeepL API'
    )
    
    translate_parser.add_argument(
        '--input', 
        required=True,
        help='Path to input JSONL file'
    )
    
    translate_parser.add_argument(
        '--output', 
        help='Path to output translated JSONL file (default: input_filename_targetlang.jsonl)'
    )
    
    translate_parser.add_argument(
        '--target-lang', 
        required=True,
        help='Target language code (e.g., RO for Romanian, EN for English)'
    )
    
    translate_parser.add_argument(
        '--batch-size', 
        type=int,
        default=100,
        help='Number of records to process before saving progress (default: 100)'
    )
    
    translate_parser.add_argument(
        '--fields', 
        help='Comma-separated list of fields to translate (default: fable)'
    )
    
    translate_parser.set_defaults(func=translate_fables)

if __name__ == "__main__":
    # For testing the script directly
    parser = argparse.ArgumentParser(description='Translate JSONL content using DeepL API')
    add_translate_subparser(parser.add_subparsers())
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help() 