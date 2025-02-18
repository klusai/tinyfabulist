import yaml
import csv
import json
from itertools import product
import random

from src.utils.ai.generator import GenerativeAICore
from src.utils.config.environment import EnvConfig

class FableGenerator:        
    def __init__(self, config_path="src/fables/config.yml", output_file="src/artifacts/fables_with_meta.csv", num_fables=1):
        self.__config_path = config_path
        self.__output_file = output_file
        self.__num_fables = num_fables

        # Load Yaml Config
        self.config = self._load_config()
        self.__system_prompt = self.config["system_prompt"]
        self.__fable_prompt = self.config["fable_prompt"]

        # Load .env Config
        env_config = EnvConfig()
        self.__hf_endpoint_url = env_config.hf_endpoint_url
        self.__hf_token = env_config.hf_token
    
    def _load_config(self):
        """Loads the configuration from the YAML file."""
        with open(self.__config_path, "r") as file:
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
        return fable_combinations[:self.__num_fables]
    
    def generate_fable(self, character, trait, setting, conflict, resolution, moral):
        ai_generator = GenerativeAICore(
            system_prompt=self.__system_prompt,
            fable_prompt=self.__fable_prompt,
            endpoint_url=self.__hf_endpoint_url,
            api_key=self.__hf_token,
            model="Llama-3.1-8B-Instruct"
        )

        return ai_generator.generate_fable(
            character=character,
            trait=trait,
            setting=setting,
            conflict=conflict,
            resolution=resolution,
            moral=moral
        )

    def create_fables_with_meta(self, selected_combos):
        """Generates fables with metadata for the given combinations."""
        meta_rows = []
        for (character, trait, setting, conflict, resolution, moral) in selected_combos:
            row = self.generate_fable(character, trait, setting, conflict, resolution, moral)
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

        with open(self.__output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in meta_rows:
                row["fable_config"] = json.dumps(row["fable_config"])
                writer.writerow(row)

        print(f"Fables with metadata have been saved to {self.__output_file}")

    def run(self):
        """Runs the fable generation process."""
        selected_combos = self.generate_fable_combinations()
        meta_rows = self.create_fables_with_meta(selected_combos)
        self.write_fables_to_csv(meta_rows)