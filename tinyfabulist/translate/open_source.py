import time

import requests
from dotenv import load_dotenv
from openai import OpenAI

from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.utils import load_translator_config
from tinyfabulist.translate.subparser import add_translate_subparser
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
    model_type: str = "chat",
    source_lang: str = "EN",
    target_lang: str = "RO",
    max_retries: int = 3,
    backoff_factor: float = 5.0,
) -> str:
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
                    "Content-Type": "application/json",
                }

                # Format specific to translation models
                payload = {
                    "inputs": text,
                    "parameters": {
                        "src_lang": source_lang,
                        "tgt_lang": target_lang,
                    },
                }

                response = requests.post(endpoint, headers=headers, json=payload)
                response.raise_for_status()  # Raise exception for HTTP errors

                # Extract translated text from response
                translation = response.json()

                # Handle different response formats
                if isinstance(translation, list) and len(translation) > 0:
                    return translation[0]["translation_text"]
                elif (
                    isinstance(translation, dict) and "translation_text" in translation
                ):
                    return translation["translation_text"]
                else:
                    logger.warning(f"Unexpected response format: {translation}")
                    return text

            else:
                # Chat-based translation (LLM approach)
                client = OpenAI(base_url=endpoint, api_key=api_key)

                system_prompt = "Ești un asistent de traducere. Tradu textul următor din limba engleză în limba română."
                fable_prompt = f"Te rog tradu: '{text}'"

                chat_completion = client.chat.completions.create(
                    model="tgi",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": fable_prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                    stream=True,
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


def translate_fables(args):
    """
    Main function to translate fables from a JSONL file.
    """
    load_dotenv()

    api_key = read_api_key("HF_ACCESS_TOKEN")

    # Load the translator configuration
    config = load_translator_config(args.config, args.translator_key)
    endpoint = config.get("endpoint")
    hf_model = config.get("model", "")

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
        output_file = build_output_path(args, hf_model)

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
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )


def open_source_translate_subparser(subparsers):
    return add_translate_subparser(subparsers, translate_fables, "EN", "RO")


if __name__ == "__main__":
    translate_main(
        translate_fables,
        "EN",
        "RO",
        description="Translate JSONL content to Romanian using Open Source models",
    )
