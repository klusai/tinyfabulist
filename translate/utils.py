import json
import os
import time
from typing import List,Dict,Any
import yaml

from tiny_fabulist.logger import setup_logging

logger = setup_logging()


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

def translate_record(translate_text,
                     record: Dict[str, Any],
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

def read_api_key(key):
    api_key = os.getenv(f'{key}')
    if not api_key:
        logger.critical(f"{key} must be set in the .env file")
        raise ValueError(f"{key} must be set in the .env file")
    
    return api_key
    
def build_output_path(args, model):
    base_name = os.path.basename(args.input)
    name_parts = os.path.splitext(base_name)
    output_dir = os.path.join('data', 'translations')
    timestamp = time.strftime("%y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{name_parts[0]}_translation_ro_{model}_{timestamp}.jsonl"
    )


    logger.info(f"Translating {args.input} to {args.target_lang}")
    logger.info(f"Output will be saved to {output_file}")

    return output_file
