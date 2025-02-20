import json
import os
import re
import yaml
import csv

class DataManager:
    def __init__(self, csv_path=None, yaml_path=None):
        """
        Initialize the DataManager with paths to the CSV and YAML files.

        Args:
            csv_path (str): Path to the CSV file containing fable data.
            yaml_path (str): Path to the YAML configuration file.
        """
        self.csv_path = csv_path
        self.yaml_path = yaml_path

    def read_from_csv(file_path: str):
        """
        Reads a CSV file and returns its content as a list of dictionaries.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            list[dict]: List of dictionaries representing each row in the CSV file.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file '{file_path}' does not exist.")

        try:
            with open(file_path, "r", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                return list(reader)
        except Exception as e:
            print(f"Error reading CSV file '{file_path}': {e}")
            return []

    def load_fables_from_csv(self):
        """
        Reads fables from the CSV file and extracts necessary fields.

        Returns:
            list[dict]: A list of dictionaries with fable details.
        """
        rows = self.read_from_csv(self.csv_path)

        fables = []
        for row in rows:
            try:
                fable_config = yaml.safe_load(row["fable_config"])
                fables.append({
                    "character": fable_config["character"],
                    "trait": fable_config["trait"],
                    "setting": fable_config["setting"],
                    "conflict": fable_config["conflict"],
                    "resolution": fable_config["resolution"],
                    "moral": fable_config["moral"],
                    "generated_fab": row["fable_text_en"]
                })
            except yaml.YAMLError as e:
                print(f"Error parsing YAML in row {row}: {e}")

        return fables

    def update_yaml(self, new_data: dict):
        """
        Updates the YAML file with new data.

        Args:
            new_data (dict): The new data to update in the YAML file.
        """
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML file '{self.yaml_path}' does not exist.")

        with open(self.yaml_path, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file) or {}

        config.update(new_data)

        with open(self.yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"Updated {self.yaml_path} with new data.")

    def update_yaml_with_fables(self, fables):
        """
        Updates the YAML configuration file by replacing its 'data' section with fables.

        Args:
            fables (list[dict]): A list of fable dictionaries.
        """
        self.update_yaml({"data": fables})
        print(f"Updated {self.yaml_path} with {len(fables)} fables.")

    def load_fables_from_yaml(self, num_fables):
        """
        Extracts the 'generated_fab' field from the first `num_fables` entries in the YAML configuration.

        Args:
            num_fables (int): Number of fables to load.

        Returns:
            list[str]: A list of generated fable texts.
        """
        config = self.read_yaml(self.yaml_path)
        return [entry["generated_fab"] for entry in config.get("data", [])[:num_fables]] if config else []

    @staticmethod
    def extract_data_from_json(response_text: str):
        """
        Extracts a JSON-like structure and any additional comments from a response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            tuple (dict or None, str): Parsed JSON data (or None if not found) and extracted comments.
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_text = json_match.group(0).strip()
                parsed_json = json.loads(json_text)
                
                # Extract additional comments (anything after the JSON structure)
                end_of_json = response_text.rindex('}') + 1
                comments = response_text[end_of_json:].strip()
                
                return parsed_json, comments if comments else "No additional comments provided."
            else:
                return None, "No valid JSON found in response."
        except json.JSONDecodeError as e:
            print(f"Error extracting JSON: {e}\nResponse text:\n{response_text}")
            return None, "No additional comments provided."

    @staticmethod
    def extract_json_from_response(response_text: str):
        """
        Extracts the first JSON-like structure from the response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            dict or None: Parsed JSON data if extraction is successful, else None.
        """
        parsed_json, _ = DataManager.extract_data_from_json(response_text)
        return parsed_json

    @staticmethod
    def extract_additional_comments(response_text: str) -> str:
        """
        Extracts additional comments from the evaluator's response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            str: The extracted comments or a default message.
        """
        _, comments = DataManager.extract_data_from_json(response_text)
        return comments

    @staticmethod
    def save_to_json(data, file_path: str, append: bool = False):
        """
        Saves or appends data to a JSON file.

        Args:
            data (dict or list): The data to save.
            file_path (str): Path to the output JSON file.
            append (bool): Whether to append data to an existing JSON file.
        """
        try:
            if append and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    try:
                        existing_data = json.load(file)
                        if not isinstance(existing_data, list):
                            existing_data = [existing_data]
                    except json.JSONDecodeError:
                        existing_data = []
                data = existing_data + ([data] if isinstance(data, dict) else data)

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=4, ensure_ascii=False)

            print(f"Data successfully saved to {file_path}")
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")

    def save_diversity_scores(self, fables, evaluation_data, output_file: str):
        """Saves diversity evaluation results to a JSON file."""
        self.save_to_json({"fables": fables, "diversity_evaluation": evaluation_data}, output_file)

    def append_to_json_file(self, file_path: str, entry: dict):
        """Appends a dictionary entry to a JSON file."""
        self.save_to_json(entry, file_path, append=True)

    @staticmethod
    def write_to_csv(data: list[dict], output_file: str, fieldnames: list[str]):
        """
        Writes a list of dictionaries to a CSV file.

        Args:
            data (list[dict]): List of dictionaries to write.
            output_file (str): Path to the CSV output file.
            fieldnames (list[str]): List of column names for the CSV file.
        """
        try:
            with open(output_file, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            print(f"CSV data successfully saved to {output_file}")
        except Exception as e:
            print(f"Error writing to CSV file '{output_file}': {e}")

    @staticmethod
    def write_fables_to_csv(meta_rows, output_file):
        """
        Writes the generated fables with metadata to a CSV file.

        Args:
            meta_rows (list[dict]): List of fables with metadata.
            output_file (str): Path to the CSV output file.
        """
        fieldnames = [
            "fable_config", "fable_prompt", "fable_text_en", "llm_name",
            "llm_input_tokens", "llm_output_tokens", "llm_inference_time",
            "llm_inference_cost_usd", "host_provider", "host_dc_provider",
            "host_dc_location", "host_gpu", "host_gpu_vram", "host_cost_per_hour",
            "generation_datetime", "pipeline_version"
        ]

        # Convert `fable_config` from dict to JSON string format for CSV storage
        for row in meta_rows:
            if isinstance(row.get("fable_config"), dict):
                row["fable_config"] = json.dumps(row["fable_config"])

        DataManager.write_to_csv(meta_rows, output_file, fieldnames)

    @staticmethod
    def read_json_file(filename: str) -> dict | None:
        """Reads a JSON file and returns its content as a dictionary."""
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading JSON file '{filename}': {e}")
            return None

    @staticmethod
    def read_yaml(file_path: str):
        """Reads the YAML file containing AI model configurations."""
        try:
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error reading YAML file '{file_path}': {e}")
            return None
