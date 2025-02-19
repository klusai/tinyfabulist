import yaml
import csv
import json
from itertools import product
import random

from src.utils.ai.generator import GenerativeAICore
from src.utils.config.environment import EnvConfig
from src.utils.data_manager import DataManager

class FableGenerator:        
    def __init__(self, model="Llama-3.1-8B-Instruct", config_path="src/generation/config.yml", output_file="src/artifacts/fables_with_meta.csv", num_fables=100):
        self.__model = model
        self.__config_path = config_path
        self.__output_file = output_file
        self.__num_fables = num_fables

        # Load Yaml Config
        self.config = self._load_config()
        self.__system_prompt = self.config["system_prompt"]
        self.__fable_prompt = self.config["fable_prompt"]

        # Load HuggingFace Config
        env_config = EnvConfig()

        ai_models = DataManager.read_yaml("ai_models.yaml")['AiModels']
        self.__hf_endpoint_url = next((model[self.__model] for model in ai_models if self.__model in model), None)
        if not self.__hf_endpoint_url:
            raise ValueError(f"Model '{self.__model}' not found in ai_models.yaml")
        
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
            model=self.__model
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
            max_attempts = 3  # Maximum number of attempts to generate a fable
            attempt = 0

            while attempt < max_attempts:
                try:
                    row = self.generate_fable(character, trait, setting, conflict, resolution, moral)
                    meta_rows.append(row)
                    break  # If successful, break out of the retry loop
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    attempt += 1

            if attempt == max_attempts:
                print(f"Failed to generate fable after {max_attempts} attempts for combo: {(character, trait, setting, conflict, resolution, moral)}")
        return meta_rows

    def run(self):
        """Runs the fable generation process."""
        selected_combos = self.generate_fable_combinations()
        meta_rows = self.create_fables_with_meta(selected_combos)
        DataManager.write_fables_to_csv(meta_rows=meta_rows, output_file=self.__output_file)