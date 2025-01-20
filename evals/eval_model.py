import os
import yaml

# Load YAML file
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
        f'--yaml_path "{yaml_path}" '
        f'--character "{character}" '
        f'--trait "{trait}" '
        f'--setting "{setting}" '
        f'--conflict "{conflict}" '
        f'--resolution "{resolution}" '
        f'--moral "{moral}" '
        f'--generated_fab "{generated_fab}" '
        f'--output {output_file}'
    )
