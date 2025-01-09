import csv
from itertools import product

# Define the lists of elements for the fables
characters = ['Rabbit', 'Fox', 'Squirrel']
traits = ['Brave', 'Greedy']
settings = ['Forest', 'River']
conflicts = ['Competing for food', 'Helping someone in need']
resolutions = ['Reward', 'Punishment']
morals = ['Kindness is rewarded', 'Hard work pays off']

# Generate all combinations of fables
fables = list(product(characters, traits, settings, conflicts, resolutions, morals))

# Prepare the output CSV file
output_file = "generate_fables/fables.csv"

# Write the fables to the CSV file
with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    
    # Write the header row
    writer.writerow(["Character", "Trait", "Setting", "Conflict", "Resolution", "Moral", "Fable"])
    
    # Write each fable
    for fable in fables:
        character, trait, setting, conflict, resolution, moral = fable
        fable_text = (f"Once there was a {trait.lower()} {character.lower()} in a {setting.lower()}. "
                      f"One day, it was {conflict.lower()}. After much effort, {resolution.lower()}.\n"
                      f"*Moral:* {moral}.")
        
        writer.writerow([character, trait, setting, conflict, resolution, moral, fable_text])

print(f"Fables have been generated and saved to '{output_file}'.")
