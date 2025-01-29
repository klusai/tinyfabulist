import logging
import os
import yaml
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
import csv
import json
from itertools import product
import random
from core import generate_fable

def main():

    with open("fables/config.yml", "r") as file:
        config = yaml.safe_load(file)

    characters = config["characters"]
    traits = config["traits"]
    settings = config["settings"]
    conflicts = config["conflicts"]
    resolutions = config["resolutions"]
    morals = config["morals"]

    # Generate all combinations
    fable_combinations = list(product(characters, traits, settings, conflicts, resolutions, morals))
    random.shuffle(fable_combinations)
    selected_combos = fable_combinations[:100]
    
    # We'll store the dictionaries returned by generate_fable in this list
    meta_rows = []

    for (character, trait, setting, conflict, resolution, moral) in selected_combos:
        row = generate_fable(character, trait, setting, conflict, resolution, moral)
        meta_rows.append(row)

    # 3.3) Write everything to fables_with_meta.csv
    fieldnames = [
        "fable_config",
        "fable_prompt",
        "fable_text_en",
        "llm_name",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_inference_time",
        "llm_inference_cost_usd",
        "host_provider",
        "host_dc_provider",
        "host_dc_location",
        "host_gpu",
        "host_gpu_vram",
        "host_cost_per_hour",
        "generation_datetime",
        "pipeline_version"
    ]

    output_file = "fables_with_meta2.csv"
    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in meta_rows:
            # Convert the "fable_config" dict to a JSON string
            row["fable_config"] = json.dumps(row["fable_config"])
            writer.writerow(row)

    print(f"Fables with metadata have been saved to {output_file}")

if __name__ == "__main__":
    main()