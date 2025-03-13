import os
import json
import argparse
import time
from openai import OpenAI
import yaml
from typing import Dict, Any, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

from tiny_fabulist.logger import setup_logging
from tiny_fabulist.translate.subparser import add_translate_subparser

logger = setup_logging()



def load_translator_config(config_file: str, translator_key: str) -> Dict[str, str]:
    """
    Load translator configuration from YAML file.
    
    Parameters:
        config_file: Path to the YAML configuration file.
        translator_key: Key in the YAML file identifying the translator configuration.
        
    Returns:
        Dictionary containing model and endpoint information.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if translator_key not in config:
            logger.error(f"Translator key '{translator_key}' not found in config file")
            raise ValueError(f"Translator key '{translator_key}' not found in config file")
        
        return config[translator_key]
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise

def translate_text(text: str, api_key: str, endpoint: str, model_type: str = "chat", 
                   source_lang: str = "EN", target_lang: str = "RO",
                   max_retries: int = 3, backoff_factor: float = 5.0) -> str:
    """
    Translate text using either chat-based LLMs or direct translation models.
    
    Parameters:
        text: Text to translate.
        api_key: API key for authentication.
        endpoint: The API endpoint for the translation service.
        model_type: Type of model - "chat" or "translation"
        source_lang: Source language code.
        target_lang: Target language code.
        max_retries: Number of retry attempts before giving up.
        backoff_factor: Seconds to sleep between retries.
        
    Returns:
        Translated text (or the original text if translation fails).
    """
    for attempt in range(1, max_retries + 1):
        try:
            if model_type == "translation":
                # Direct translation model approach (e.g., MADLAD-400)
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                # Format specific to translation models
                payload = {
                    "inputs": text,
                    "parameters": {
                        "src_lang": source_lang,
                        "tgt_lang": target_lang,
                    }
                }
                
                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors
                
                # Extract translated text from response
                translation = response.json()
                
                # Handle different response formats
                if isinstance(translation, list) and len(translation) > 0:
                    return translation[0]["translation_text"]
                elif isinstance(translation, dict) and "translation_text" in translation:
                    return translation["translation_text"]
                else:
                    logger.warning(f"Unexpected response format: {translation}")
                    return text
            
            else:
                # Chat-based translation (LLM approach)
                client = OpenAI(
                    base_url=endpoint,
                    api_key=api_key
                )

                system_prompt = "Ești un asistent de traducere. Tradu textul următor din limba engleză în limba română."
                fable_prompt = f"Te rog tradu: '{text}'"

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
                
                fable_translation = ""
                for message in chat_completion:
                    if message.choices[0].delta.content is not None:
                        fable_translation += message.choices[0].delta.content
                return fable_translation
        except Exception as e:
            logger.error(
                f"Translation error on attempt {attempt}/{max_retries}: {e}. "
                f"Sleeping for {backoff_factor} seconds."
            )
            if attempt < max_retries:
                time.sleep(backoff_factor)
            else:
                logger.error("Max retries exceeded. Returning original text.")
                return text

def translate_record(record: Dict[str, Any],
                     fields: List[str],
                     api_key: str,
                     endpoint: str,
                     source_lang: str,
                     target_lang: str,
                     model_name:str = "") -> Dict[str, Any]:
    """
    
    Parameters:
        record: A dictionary representing a JSONL record.
        fields: List of fields to translate.
        api_key: Hugging Face API key for authentication.
        endpoint: The API endpoint for the translation service.
        source_lang: Source language code.
        target_lang: Target language code.
        
    Returns:
        The record with the specified fields translated and a 'language' field updated.
    """
    for field in fields:
        if field in record and record[field]:
            record[field] = translate_text(record[field], api_key, endpoint, source_lang=source_lang, target_lang=target_lang)
            time.sleep(0.1)
    record['language'] = 'ro'  # Hardcoded to Romanian

    if 'llm_name' in record:
        record['llm_name'] += f"_{model_name}"

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
                    endpoint: str,
                    source_lang: str,
                    target_lang: str,
                    batch_size: int = 100,
                    fields_to_translate: Optional[List[str]] = None,
                    max_workers: int = 30,
                    model_name: str = "") -> None:
    """
    
    Parameters:
        input_file: Path to the input JSONL file.
        output_file: Path to the output JSONL file.
        api_key: Hugging Face API key for authentication.
        endpoint: The API endpoint for the translation service.
        source_lang: Source language code.
        target_lang: Target language code.
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
        with open(input_file, 'r', encoding='utf-8') as f, tqdm(total=total_lines, desc="Translating to Romanian") as pbar:
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
                        endpoint,
                        source_lang,
                        target_lang,
                        model_name
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
    
    api_key = os.getenv('HF_ACCESS_TOKEN')
    if not api_key:
        logger.critical("HF_ACCESS_TOKEN must be set in the .env file")
        raise ValueError("HF_ACCESS_TOKEN must be set in the .env file")
    
    # Load the translator configuration
    config = load_translator_config(args.config, args.translator_key)
    endpoint = config.get('endpoint')
    hf_model = config.get('model','')
    
    if not endpoint:
        logger.critical(f"Endpoint not found in config for translator key: {args.translator_key}")
        raise ValueError(f"Endpoint not found in config for translator key: {args.translator_key}")
    
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
            f"{name_parts[0]}_translation_ro_{hf_model}_{timestamp}.jsonl"
        )
    
    logger.info(f"Translating {args.input} from {source_lang} to {target_lang} using endpoint: {endpoint}")
    logger.info(f"Output will be saved to {output_file}")
    
    translate_jsonl(
        input_file=args.input,
        output_file=output_file,
        api_key=api_key,
        endpoint=endpoint,
        source_lang=source_lang,
        target_lang=target_lang,
        batch_size=args.batch_size,
        fields_to_translate=args.fields.split(',') if args.fields else ['fable', 'prompt'],
        max_workers=args.max_workers,
        model_name=hf_model
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate JSONL content to Romanian using Open Source models')
    subparsers = parser.add_subparsers()
    add_translate_subparser(subparsers, translate_fables)
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()