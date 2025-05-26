import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from datetime import datetime

import yaml
from tqdm import tqdm

from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.subparser import add_translate_subparser

logger = setup_logging()


def save_progress(
    records: List[Dict[str, Any]], output_file: str, is_first_batch: bool, model: str="o3-mini-2025-01-31"
) -> None:
    """
    Save a batch of translated records to the output file.

    Parameters:
        records: List of translated records.
        output_file: Path to the output file.
        is_first_batch: If True, the file will be overwritten; otherwise, records are appended.
    """
    mode = "w" if is_first_batch else "a"
    with open(output_file, mode, encoding="utf-8") as f:
        for record in records:
            record["pipeline_stage"] = "translation"
            record["translation_model"] = model
            record["translation_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def translate_record(
    translate_text, record: Dict[str, Any], fields: List[str], model_name: str, **kwargs
) -> Dict[str, Any]:
    """
    Returns:
        The record with the specified fields translated and a 'language' field updated.
    """
    for field in fields:
        if field in record and record[field]:
            field_name = "translated_" + field
            record[field_name] = translate_text(record[field], **kwargs)
            time.sleep(0.1)
    record["language"] = "ro"  # Hardcoded to Romanian

    if "llm_name" in record:
        record["llm_name"] += f"_{model_name}"

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
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        if translator_key not in config:
            logger.error(f"Translator key '{translator_key}' not found in config file")
            raise ValueError(
                f"Translator key '{translator_key}' not found in config file"
            )

        return config[translator_key]
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise


def read_api_key(key):
    api_key = os.getenv(f"{key}")
    if not api_key:
        logger.critical(f"{key} must be set in the .env file")
        raise ValueError(f"{key} must be set in the .env file")

    return api_key


def build_output_path(args, model):
    base_name = os.path.basename(args.input)
    output_dir = os.path.join("data", "translations")
    timestamp = time.strftime("%y%m%d-%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"translations_{timestamp}.jsonl"
    )

    logger.info(f"Translating {args.input} to {args.target_lang}")
    logger.info(f"Output will be saved to {output_file}")

    return output_file


def translate_jsonl(
    input_file: str,
    output_file: str,
    executor_submit,
    batch_size: int = 100,
    fields_to_translate: Optional[List[str]] = None,
    max_workers: int = 60,
) -> None:
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
        fields_to_translate = ["prompt", "fable"]

    # Count total lines for progress tracking
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    translated_records = []
    processed_count = 0

    # Use a ThreadPoolExecutor to process records concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with open(input_file, "r", encoding="utf-8") as f, tqdm(
            total=total_lines, desc="Translating"
        ) as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    record = json.loads(line)
                    future = executor_submit(record, executor)
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
                    save_progress(
                        translated_records, output_file, processed_count == batch_size
                    )
                    translated_records.clear()

    # Save any remaining records
    if translated_records:
        save_progress(translated_records, output_file, False)

    logger.info(f"Translation complete. Processed {processed_count} records.")


def translate_main(
    translate_fables,
    source_lang,
    target_lang,
    description="Translate JSONL (EN --> RO)",
):
    parser = argparse.ArgumentParser(description=description)

    subparsers = parser.add_subparsers(
        title="translate", dest="translate"
    )
    add_translate_subparser(subparsers, translate_fables, source_lang, target_lang)
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def translate_jsonl(
    translate_text,
    input_file: str,
    output_file: str,
    batch_size: int = 100,
    fields_to_translate: Optional[List[str]] = None,
    max_workers: int = 60,
    model_name: str = "",
    **kwargs,
) -> None:
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
        fields_to_translate = ["prompt", "fable"]

    # Count total lines for progress tracking
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    translated_records = []
    processed_count = 0

    # Use a ThreadPoolExecutor to process records concurrently
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        with open(input_file, "r", encoding="utf-8") as f, tqdm(
            total=total_lines, desc="Translating"
        ) as pbar:
            for line in f:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    record = json.loads(line)
                    future = executor.submit(
                        translate_record,
                        translate_text,
                        record,
                        fields_to_translate,
                        model_name,
                        **kwargs,
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
                    save_progress(
                        translated_records, output_file, processed_count == batch_size, model_name
                    )
                    translated_records.clear()

    # Save any remaining records
    if translated_records:
        save_progress(translated_records, output_file, False, model_name)

    logger.info(f"Translation complete. Processed {processed_count} records.")
