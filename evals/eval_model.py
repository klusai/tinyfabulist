import os

# List of prompts and model responses
data = [
    {
        "character": "Fox",
        "trait": "Clever",
        "setting": "Forest",
        "conflict": "Helping someone in need",
        "resolution": "Reward",
        "moral": "Kindness is rewarded",
        "generated_fab": "Once there was a clever fox in a forest. One day, it was helping a trapped bird escape from a hunter's net. After much effort, the fox used its cleverness to gnaw through the net and free the bird. In gratitude, the bird led the fox to a hidden cache of food. Moral: Kindness is rewarded."
    }
]

# Loop through each set of structured input and generated fable
for i, pair in enumerate(data):
    character = pair["character"]
    trait = pair["trait"]
    setting = pair["setting"]
    conflict = pair["conflict"]
    resolution = pair["resolution"]
    moral = pair["moral"]
    generated_fab = pair["generated_fab"]

    output_file = f"evaluation_results_{i+1}.json"

    # Run evals_cli.py for each pair
    os.system(
        f'python evals_cli.py '
        f'--character "{character}" '
        f'--trait "{trait}" '
        f'--setting "{setting}" '
        f'--conflict "{conflict}" '
        f'--resolution "{resolution}" '
        f'--moral "{moral}" '
        f'--generated_fab "{generated_fab}" '
        f'--output {output_file}'
    )
