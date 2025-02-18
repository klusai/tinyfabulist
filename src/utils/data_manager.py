import json
import os
import re
import yaml
import csv

class DataManager:
    def __init__(self, csv_path=None, yaml_path=None):
        """
        Initialize the FableManager with paths to the CSV and YAML files.

        Args:
            csv_path (str): Path to the CSV file containing fable data.
            yaml_path (str): Path to the YAML configuration file.
        """
        self.csv_path = csv_path
        self.yaml_path = yaml_path

    def load_fables_from_csv(self):
        """
        Reads fables from the CSV file and extracts necessary fields.

        Returns:
            list[dict]: A list of dictionaries with fable details.
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file '{self.csv_path}' does not exist.")

        fables = []
        with open(self.csv_path, "r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Parse the YAML string in the "fable_config" field into a dictionary.
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
        return fables

    def update_yaml_with_fables(self, fables):
        """
        Updates the YAML configuration file by replacing its 'data' section with the provided fables.

        Args:
            fables (list[dict]): A list of fable dictionaries.
        """
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"YAML file '{self.yaml_path}' does not exist.")

        # Load the existing YAML configuration.
        with open(self.yaml_path, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file)

        # Update the "data" section with the new fables.
        config["data"] = fables

        # Write the updated configuration back to the YAML file.
        with open(self.yaml_path, "w", encoding="utf-8") as yaml_file:
            yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

        print(f"Updated {self.yaml_path} with {len(fables)} fables.")

    def load_fables_from_yaml(self, num_fables):
        """
        Extracts the 'generated_fab' field from the first `num_fables` entries in the YAML configuration.

        Args:
            num_fables (int): Number of fables to load.

        Returns:
            list[str]: A list of generated fable texts.
        """
        with open(self.yaml_path, "r", encoding="utf-8") as yaml_file:
            config = yaml.safe_load(yaml_file)

        fables = [entry["generated_fab"] for entry in config.get("data", [])[:num_fables]]
        return fables

    @staticmethod
    def extract_json_from_response(response_text: str):
        """
        Extracts the first JSON-like structure from the response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            dict or None: Parsed JSON data if extraction is successful, else None.
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_text = json_match.group(0).strip()
                return json.loads(json_text)
            else:
                print("No valid JSON found in response.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error extracting JSON: {e}\nResponse text:\n{response_text}")
            return None

    @staticmethod
    def extract_additional_comments(response_text: str) -> str:
        """
        Extracts additional comments from the evaluator's response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            str: The extracted comments or a default message.
        """
        try:
            end_of_json = response_text.rindex('}') + 1
            comments = response_text[end_of_json:].strip()
            return comments if comments else "No additional comments provided."
        except ValueError:
            return "No additional comments provided."

    @staticmethod
    def save_diversity_scores(fables, evaluation_data, output_file: str):
        """
        Saves diversity evaluation results to a JSON file.

        Args:
            fables (list): List of fable texts.
            evaluation_data (dict): The diversity evaluation data.
            output_file (str): Path to the output file.
        """
        print("Saving diversity scores to JSON...")
        diversity_results = {
            "fables": fables,
            "diversity_evaluation": evaluation_data
        }
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(diversity_results, json_file, indent=4, ensure_ascii=False)
        print(f"Diversity evaluation saved to {output_file}")

    @staticmethod
    def append_to_json_file(file_path: str, entry: dict) -> None:
        """
        Appends a dictionary entry to a JSON file.
        If the file doesn't exist or is invalid, a new list is created.

        Args:
            file_path (str): Path to the JSON file.
            entry (dict): Dictionary entry to append.
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []

            data.append(entry)

            with open(file_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Entry successfully appended to {file_path}")
        except Exception as e:
            print(f"Failed to append entry to {file_path}: {e}")
