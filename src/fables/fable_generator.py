import yaml
import csv
import json
from itertools import product
import random
from core import generate_fable

class FableGenerator:
    def __init__(self, config_path="fables/config.yml", output_file="artifacts/fables_with_meta2.csv", num_fables=100):
        self.config_path = config_path
        self.output_file = output_file
        self.num_fables = num_fables
        self.config = self._load_config()

    def _load_config(self):
        """Loads the configuration from the YAML file."""
        with open(self.config_path, "r") as file:
            return yaml.safe_load(file)

    def generate_fable_combinations(self):
        """Generates and shuffles fable combinations based on the config."""
        characters = self.config["characters"]
        traits = self.config["traits"]
        settings = self.config["settings"]
        conflicts = self.config["conflicts"]
        resolutions = self.config["resolutions"]
        morals = self.config["morals"]

        fable_combinations = list(product(characters, traits, settings, conflicts, resolutions, morals))
        random.shuffle(fable_combinations)
        return fable_combinations[:self.num_fables]

    def create_fables_with_meta(self, selected_combos):
        """Generates fables with metadata for the given combinations."""
        meta_rows = []
        for (character, trait, setting, conflict, resolution, moral) in selected_combos:
            row = generate_fable(character, trait, setting, conflict, resolution, moral)
            meta_rows.append(row)
        return meta_rows

    def write_fables_to_csv(self, meta_rows):
        """Writes the generated fables with metadata to a CSV file."""
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

        with open(self.output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in meta_rows:
                row["fable_config"] = json.dumps(row["fable_config"])
                writer.writerow(row)

        print(f"Fables with metadata have been saved to {self.output_file}")

    def run(self):
        """Runs the fable generation process."""
        selected_combos = self.generate_fable_combinations()
        meta_rows = self.create_fables_with_meta(selected_combos)
        self.write_fables_to_csv(meta_rows)

def main():
    generator = FableGenerator()
    generator.run()

if __name__ == "__main__":
    main()