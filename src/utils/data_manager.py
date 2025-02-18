import os
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

