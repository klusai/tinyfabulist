import time

import requests
from dotenv import load_dotenv

from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.utils import load_translator_config
from tinyfabulist.translate.utils import (
    build_output_path,
    read_api_key,
    translate_jsonl,
    translate_main,
)

logger = setup_logging()


def translate_text(
    text: str,
    api_key: str,
    endpoint: str,
    model_type: str = "mbart",
    source_lang: str = "en_XX",
    target_lang: str = "ro_RO",
    max_retries: int = 3,
    backoff_factor: float = 5.0,
) -> str:
    """
    Translate text using mBART or other translation models.

    Parameters:
        text: Text to translate.
        api_key: API key for authentication.
        endpoint: The API endpoint for the translation service.
        model_type: Type of model - "mbart", "translation", or "chat"
        source_lang: Source language code.
        target_lang: Target language code.
        max_retries: Number of retry attempts before giving up.
        backoff_factor: Seconds to sleep between retries.

    Returns:
        Translated text (or the original text if translation fails).
    """
    for attempt in range(1, max_retries + 1):
        try:
            # mBART-specific approach
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            # Format specific to mBART models
            payload = {
                "inputs": text,
                "parameters": {
                    "src_lang": source_lang,
                    "tgt_lang": target_lang,
                    "max_length": 1024,
                },
            }

            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors

            # Extract translated text from response
            translation = response.json()

            # Handle mBART response format
            if isinstance(translation, list) and len(translation) > 0:
                if "translation_text" in translation[0]:
                    return translation[0]["translation_text"]
                elif "generated_text" in translation[0]:
                    return translation[0]["generated_text"]
                else:
                    logger.warning(f"Unexpected mBART response format: {translation}")
                    return text
            elif isinstance(translation, dict):
                if "translation_text" in translation:
                    return translation["translation_text"]
                elif "generated_text" in translation:
                    return translation["generated_text"]
                else:
                    logger.warning(f"Unexpected mBART response format: {translation}")
                    return text
            else:
                logger.warning(f"Unexpected response format: {translation}")
                return text
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


def translate_fables(args):
    """
    Main function to translate fables from a JSONL file.
    """
    load_dotenv()

    api_key = read_api_key("HF_ACCESS_TOKEN")

    # Load the translator configuration
    config = load_translator_config(args.config, args.translator_key)
    endpoint = config.get("endpoint")
    hf_model = config.get("model")
    model_type = config.get("model_type", "mbart")  # Default to mbart

    if not endpoint:
        logger.critical(
            f"Endpoint not found in config for translator key: {args.translator_key}"
        )
        raise ValueError(
            f"Endpoint not found in config for translator key: {args.translator_key}"
        )

    source_lang = args.source_lang
    target_lang = args.target_lang

    output_file = args.output
    if not output_file:
        output_file = build_output_path(args, hf_model.split("/")[-1])

    translate_jsonl(
        translate_text=translate_text,
        input_file=args.input,
        output_file=output_file,
        batch_size=args.batch_size,
        fields_to_translate=(
            args.fields.split(",") if args.fields else ["fable", "prompt"]
        ),
        max_workers=args.max_workers,
        model_name=hf_model,
        **{
            "api_key": api_key,
            "endpoint": endpoint,
            "model_type": model_type,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )


if __name__ == "__main__":
    translate_main(
        translate_fables,
        "en_XX",
        "ro_RO",
        description="Translate JSONL content using mBART or other models",
    )
