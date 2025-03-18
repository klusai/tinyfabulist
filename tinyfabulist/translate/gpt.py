import time

import requests
from dotenv import load_dotenv

from tinyfabulist.logger import setup_logging
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
    api_key: str,
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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    # Construct the prompt for translation
    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}:\n\n{text}."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": prompt.strip()},
        ],
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


def translate_fables(args):
    """
    Main function to translate fables from a JSONL file using the ChatGPT API.
    """
    load_dotenv()

    api_key = read_api_key("OPENAI_API_KEY")

    model = "gpt-4o"  #'gpt-4o'

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
            "api_key": api_key,
            "model": model,
            "source_lang": source_lang,
            "target_lang": target_lang,
        },
    )


def gpt_subparser(subparsers):
    return add_translate_subparser(subparsers, translate_fables, "en", "ro")


if __name__ == "__main__":
    translate_main(
        translate_fables,
        "en",
        "ro",
        description="Translate JSONL content using the ChatGPT API",
    )
