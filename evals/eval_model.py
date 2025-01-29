import os
import yaml
import csv
import shlex


def load_fables_from_csv(csv_path):
    """
    Reads fables from fables_final.csv and extracts necessary fields for the data section.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file '{csv_path}' does not exist.")

    fables = []
    with open(csv_path, "r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            fable_config = yaml.safe_load(row["fable_config"])  # Parse the JSON string into a dictionary

            # Append to fables list with necessary fields
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


def update_yaml_with_fables(yaml_path, fables):
    """
    Updates the evals_config.yml file with the fables in the data section.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML file '{yaml_path}' does not exist.")

    # Load the existing YAML configuration
    with open(yaml_path, "r", encoding="utf-8") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Update the "data" section with the fables
    config["data"] = fables

    # Write back the updated YAML configuration
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Updated {yaml_path} with {len(fables)} fables.")


def main():
    # File paths
    csv_path = "fables_with_meta.csv"
    yaml_path = "/Users/Andreea/KlusAI/TinyFabulist/tiny-fabulist-1/evals/evals_config.yml"

    # Step 1: Load fables from CSV
    fables = load_fables_from_csv(csv_path)

    # Step 2: Update YAML file with fables
    update_yaml_with_fables(yaml_path, fables)
    yaml_path = "/Users/Andreea/KlusAI/TinyFabulist/tiny-fabulist-1/evals/evals_config.yml"
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    # Extract data section
    data = config.get("data", [])

    # Loop through each set of structured input and generated fable
    for i, pair in enumerate(data):
        character = pair["character"]
        trait = pair["trait"]
        setting = pair["setting"]
        conflict = pair["conflict"]
        resolution = pair["resolution"]
        moral = pair["moral"]
        generated_fab = pair["generated_fab"]

        output_file = f"evaluation_results.json"

        # Run evals_cli.py with the required arguments

        os.system(
            f'python3.11 evals/evals_cli.py '
            f'--yaml_path {shlex.quote(yaml_path)} '
            f'--character {shlex.quote(character)} '
            f'--trait {shlex.quote(trait)} '
            f'--setting {shlex.quote(setting)} '
            f'--conflict {shlex.quote(conflict)} '
            f'--resolution {shlex.quote(resolution)} '
            f'--moral {shlex.quote(moral)} '
            f'--generated_fab {shlex.quote(generated_fab)} '
            f'--output {shlex.quote(output_file)}'
    )




if __name__ == "__main__":
    main()
# Load YAML file
