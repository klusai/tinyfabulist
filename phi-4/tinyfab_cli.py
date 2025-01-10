import csv
from core import generate_fable
from itertools import product

def save_fables_to_csv(filename, fables):
    """
    Saves generated fables to a CSV file.

    Args:
        filename (str): The name of the CSV file.
        fables (list of dict): A list of fables, where each fable is a dictionary.
    """
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Character", "Trait", "Setting", "Conflict", "Resolution", "Moral", "Fable"])
        writer.writeheader()
        writer.writerows(fables)

def main():
    # Define the input parameters for fable generation
    characters = ['Rabbit', 'Fox', 'Squirrel']
    traits = ['Brave', 'Greedy']
    settings = ['Forest', 'River']
    conflicts = ['Competing for food', 'Helping someone in need']
    resolutions = ['Reward', 'Punishment']
    morals = ['Kindness is rewarded', 'Hard work pays off']

    # Generate all combinations of fables
    fable_combinations = list(product(characters, traits, settings, conflicts, resolutions, morals))
    fables = []

    for fable in fable_combinations:
        character, trait, setting, conflict, resolution, moral = fable
        generated_fable = generate_fable(character, trait, setting, conflict, resolution, moral)
        
        # Append the fable to the list as a dictionary
        fables.append({
            "Character": character,
            "Trait": trait,
            "Setting": setting,
            "Conflict": conflict,
            "Resolution": resolution,
            "Moral": moral,
            "Fable": generated_fable
        })

    # Save the generated fables to a CSV file
    save_fables_to_csv("fables.csv", fables)
    print("Fables have been saved to fables.csv")

if __name__ == "__main__":
    main()
