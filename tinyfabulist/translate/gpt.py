import time

import requests
from dotenv import load_dotenv
import yaml

from tinyfabulist.logger import setup_logging
from tinyfabulist.translate.subparser import add_translate_subparser
from tinyfabulist.translate.utils import (
    build_output_path,
    read_api_key,
    translate_jsonl,
    translate_main,
)
from openai import OpenAI
logger = setup_logging()
MODEL = "google/gemini-flash-1.5-8b"

api_key = read_api_key("OPENROUTER_KEY")
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=api_key,
)

def translate_text(
    text: str,
    model: str,
    source_lang: str = "en",
    target_lang: str = "ro",
    max_retries: int = 3,
    backoff_factor: float = 5.0,
) -> str:
    """
    Translate text using the ChatGPT API with a retry mechanism.

    Parameters:
        text: Text to translate.
        model: ChatGPT model to use (e.g., 'gpt-3.5-turbo').
        source_lang: Source language (e.g., 'en').
        target_lang: Target language (e.g., 'ro').
        max_retries: Number of retry attempts before giving up.
        backoff_factor: Seconds to sleep between retries.

    Returns:
        Translated text (or the original text if translation fails).
    """
    # Construct the prompt for translation
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
            model=model,
            messages=[
                {
                "role": "user",
                "content": prompt.strip()
                }
            ]
            )
            
            # Extract the translated text directly from the completion object
            if response.choices and len(response.choices) > 0:
                translated_text = response.choices[0].message.content.strip()
                return translated_text
            else:
                logger.error(f"No choices returned in response. Attempt {attempt}/{max_retries}.")
                time.sleep(backoff_factor)
                
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
    Main function to translate fables from a JSONL file using the ChatGPT API.
    """
    load_dotenv()

    model = MODEL  

    source_lang = args.source_lang
    target_lang = args.target_lang
    output_file = args.output

    if not output_file:
        output_file = build_output_path(args, model)

    translate_jsonl(
        translate_text=translate_text,
        input_file=args.input,
        output_file=output_file,
        batch_size=args.batch_size,
        fields_to_translate=(
            args.fields.split(",") if args.fields else ["fable", "prompt"]
        ),
        max_workers=args.max_workers,
        model_name=model,
        **{
            "model": model,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )


def gpt_subparser(args):
    return translate_fables(args)


if __name__ == "__main__":
    translate_main(
        translate_fables,
        "en",
        "ro",
        description="Translate JSONL content using the ChatGPT API",
    )
