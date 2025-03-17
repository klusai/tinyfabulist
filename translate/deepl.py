import time

import deepl  # Using the official DeepL library
from dotenv import load_dotenv

from tiny_fabulist.logger import setup_logging
from translate.subparser import add_translate_subparser
from translate.utils import (
    build_output_path,
    read_api_key,
    translate_jsonl,
    translate_main,
)

logger = setup_logging()


def translate_text(
    text: str,
    target_lang: str,
    auth_key: str,
    max_retries: int = 3,
    backoff_factor: float = 5.0,
) -> str:
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


def translate_fables(args):
    """
    Main function to translate fables from a JSONL file.
    """
    load_dotenv()

    auth_key = read_api_key("DEEPL_AUTH_KEY")

    output_file = args.output

    if not output_file:
        output_file = build_output_path(args, "deepl")

    translate_jsonl(
        translate_text=translate_text,
        input_file=args.input,
        output_file=output_file,
        batch_size=args.batch_size,
        fields_to_translate=(
            args.fields.split(",") if args.fields else ["fable", "prompt"]
        ),
        max_workers=args.max_workers,
        model_name="deepl",
        **{"auth_key": auth_key, "target_lang": args.target_lang},
    )


def deepl_subparser(subparsers):
    return add_translate_subparser(subparsers, translate_fables, "EN", "RO")


if __name__ == "__main__":
    translate_main(
        translate_fables,
        "EN",
        "RO",
        description="Translate JSONL content using DeepL API",
    )
