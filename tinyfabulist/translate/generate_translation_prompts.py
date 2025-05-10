#!/usr/bin/env python3
"""
Script: generate_translation_prompts.py
Location: tinyfabulist/translations/

This script automates:
  1. Generating translation prompt JSONL files from fables in data/fables/.
  2. Ensuring every JSONL in data/ has a "pipeline_stage": "translation" field.

Usage:
  python generate_translation_prompts.py
"""
import json
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

class TranslationPromptGenerator:
    """
    Encapsulates logic for generating translation prompts from fables and
    ensuring pipeline_stage metadata across all JSONL data files.
    """
    def __init__(
        self,
        fables_dir: Path,
        prompts_dir: Path,
        data_dir: Path,
        pipeline_stage: str = 'translation'
    ):
        self.fables_dir = fables_dir
        self.prompts_dir = prompts_dir
        self.data_dir = data_dir
        self.pipeline_stage = pipeline_stage

        # Create the translation prompts directory if it doesn't exist
        self.prompts_dir.mkdir(parents=True, exist_ok=True)

    def generate_translation_prompts(self):
        """
        Traverse each .jsonl in fables_dir, extract 'fable', and write a new JSONL
        with translation prompts to prompts_dir.
        """
        for fable_file in self.fables_dir.rglob('*.jsonl'):
            logging.info(f"Processing fable file: {fable_file}")
            out_file = self.prompts_dir / fable_file.name

            try:
                with fable_file.open('r', encoding='utf-8') as infile, \
                     out_file.open('w', encoding='utf-8') as outfile:
                    for lineno, line in enumerate(infile, start=1):
                        text = line.strip()
                        if not text:
                            continue

                        # Parse the JSON fable entry
                        try:
                            entry = json.loads(text)
                        except json.JSONDecodeError as e:
                            logging.warning(
                                f"Invalid JSON in {fable_file}:{lineno}: {e}"
                            )
                            continue

                        fable = entry.get('fable')
                        if not fable:
                            logging.warning(
                                f"Missing 'fable' field in {fable_file}:{lineno}. Skipping."
                            )
                            continue

                        # Build the translation prompt object
                        prompt_obj = {
                            'prompt_type': 'translation_prompt',
                            'content': f"translate the following fable: {fable}",
                            'pipeline_stage': self.pipeline_stage
                        }
                        # Write as one JSON line
                        outfile.write(json.dumps(prompt_obj, ensure_ascii=False) + '\n')

                logging.info(f"Wrote prompts to {out_file}")

            except Exception as e:
                logging.error(f"Error processing {fable_file}: {e}")

    def ensure_pipeline_stage_field(self):
        """
        Walk through all .jsonl under data_dir and ensure each JSON record
        has pipeline_stage set to the configured value.
        Overwrites files in place when updates are needed.
        """
        for data_file in self.data_dir.rglob('*.jsonl'):
            logging.info(f"Ensuring pipeline_stage in {data_file}")
            try:
                lines = data_file.read_text(encoding='utf-8').splitlines()
            except Exception as e:
                logging.error(f"Cannot read {data_file}: {e}")
                continue

            updated = False
            new_lines = []
            for lineno, text in enumerate(lines, start=1):
                if not text.strip():
                    new_lines.append(text)
                    continue

                try:
                    record = json.loads(text)
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"Invalid JSON in {data_file}:{lineno}: {e}"
                    )
                    new_lines.append(text)
                    continue

                # Add/update pipeline_stage
                if record.get('pipeline_stage') != self.pipeline_stage:
                    record['pipeline_stage'] = self.pipeline_stage
                    updated = True

                new_lines.append(json.dumps(record, ensure_ascii=False))

            if updated:
                try:
                    data_file.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
                    logging.info(f"Updated pipeline_stage in {data_file}")
                except Exception as e:
                    logging.error(f"Cannot write {data_file}: {e}")

    def run(self):
        """Run both generation and metadata enforcement steps."""
        self.generate_translation_prompts()
        self.ensure_pipeline_stage_field()


def main():
    # Assuming this script lives in tinyfabulist/translations/
    base_dir = Path(__file__).parent.parent  # project root
    generator = TranslationPromptGenerator(
        fables_dir=base_dir / 'data' / 'fables',
        prompts_dir=base_dir / 'data' / 'translation_prompts',
        data_dir=base_dir / 'data'
    )
    generator.run()


if __name__ == '__main__':
    main()
