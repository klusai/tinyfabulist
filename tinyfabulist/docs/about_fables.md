# About Fables

## What is a fable?

A fable is a short story, often featuring animals, plants, or inanimate objects that are personified (given human characteristics), and it conveys a moral lesson or ethical principle. Fables are a form of storytelling that is simple, yet profound, making them suitable for teaching values or lessons, especially to children.

Characteristics of Fables:
1. Brevity: Fables are concise and to the point, making them easy to remember and retell.
2. Moral Lesson: They often end with a clear takeaway or moral, which might be explicitly stated (e.g., “The moral of the story is…”).
3. Personification: Non-human characters, such as animals or objects, are given human traits to make the story engaging and relatable.
4. Universal Themes: Fables often deal with themes like honesty, kindness, cleverness, or the consequences of greed, laziness, or dishonesty.
5. Timeless Appeal: They transcend cultures and eras, making them relevant across generations.

## Examples of famous fables
1. The Tortoise and the Hare (Aesop)
  - A slow and steady tortoise wins a race against an overconfident hare.
  - Moral: Slow and steady wins the race.
2. The Fox and the Grapes (Aesop):
  - A fox fails to reach some grapes and dismisses them as sour.
  - Moral: It’s easy to despise what you cannot have (origin of “sour grapes”).
3. The Ant and the Grasshopper (Aesop):
  - An ant works hard during summer, while a grasshopper idles. The ant is prepared for winter, while the grasshopper suffers.
  - Moral: Prepare for the future instead of only living for the moment.

### Fables in Romanian culture
In Romanian culture, “fabule” are similar and often reflect local traditions or wisdom, such as those by Grigore Alexandrescu, a Romanian fabulist. His works, like “Boul și Vițelul” (“The Ox and the Calf”), carry moral lessons rooted in Romanian society.

## Merits for synthetic dataset generation
Fables make a good inspiration for a NLP-focused dataset, as they are inherently compact, use simple language, and serve to convey reasoning or moral understanding, which aligns with the goals of training small language models.

## Synthentically generated fables

### English (EN)

#### 1. The Ant and the Butterfly

Once there was a busy ant named Ana. She worked all day collecting crumbs for winter. One day, she saw a beautiful butterfly dancing in the sun. “Come dance with me, Ana,” said the butterfly. “I can’t. I must prepare for the cold,” said Ana. The butterfly laughed and flew away. When winter came, the butterfly had no food. She knocked on Ana’s door. “Please, Ana, can you share your crumbs?” Ana gave the butterfly some crumbs. “Next time, work when you can,” she said. The butterfly nodded, feeling thankful and wiser.

Moral: Prepare today for what you’ll need tomorrow.

#### 2. The Lazy Fox and the Clever Crow

Once a fox saw a crow with a big piece of cheese in her beak. “Dear Crow,” said the fox, “your feathers are so shiny. Surely, you have the sweetest voice!” The crow smiled and opened her beak to sing, but the cheese fell to the ground. The fox grabbed the cheese and said, “Thank you for the meal!” The crow felt foolish and said, “Flattery is no friend.”

Moral: Beware of those who only praise you to get what they want.

### Romanian (RO)

#### 1. Furnica și Fluturele

Într-o zi, o furnică pe nume Ana aduna firimituri pentru iarnă. Fluturele, frumos și colorat, dansa în razele soarelui. “Vino să dansezi cu mine, Ana,” spuse fluturele. “Nu pot. Trebuie să mă pregătesc pentru frig,” răspunse furnica. Fluturele râse și zbură mai departe. Când a venit iarna, fluturele nu avea mâncare. A bătut la ușa furnicii. “Te rog, Ana, îmi poți da niște firimituri?” Furnica i-a dat câteva. “Data viitoare, muncește la timp,” spuse ea. Fluturele a înțeles și i-a mulțumit.

Morală: Pregătește-te azi pentru ce vei avea nevoie mâine.

#### 2. Vulpea leneșă și Corbul isteț

O vulpe vicleană văzu un corb cu o bucată mare de brânză în cioc. “Dragă Corbule,” spuse vulpea, “penele tale sunt atât de strălucitoare! Cu siguranță ai și o voce minunată.” Corbul zâmbi și deschise ciocul să cânte, dar brânza căzu pe jos. Vulpea o luă repede și spuse: “Mulțumesc pentru masă!” Corbul oftă și spuse: “Lingusirea nu este prieten adevărat.”

Morală: Ferește-te de cei care te laudă doar pentru a obține ce vor.

## Considerations for synthetic data generation

To reliably generate relatively large datasets of, say, over 2.5 million unique fables, an algorithm must identify and combine key patterns and features that make fables distinct. In the following sections we provide a breakdown of the patterns, features, and a strategy to ensure enough uniqueness.

### Key patterns in fables
1. Characters and Roles:
  - Fables often feature personified animals, nature elements, or objects with human traits.
  - Examples: Fox as cunning, Rabbit as quick, Ant as hardworking.
2. Setting:
  - Simple, relatable environments like forests, ponds, villages, or homes.
  - Example: “In a quiet forest,” or “By a sparkling river.”
3. Conflict or Challenge:
  - A central conflict or problem that drives the plot.
  - Example: A choice between selfishness and kindness, or a challenge to solve.
4. Resolution and Moral:
  - A resolution that ties back to the moral lesson.
  - Example: The lazy character suffers, while the diligent one thrives.

### Features of combinatorial uniquness
Each fable can be viewed as a combination of multiple feature categories. By varying these features systematically, you can create a vast number of unique stories.
1. Character Types:
  - Categories: Animals, humans, objects, or mythical creatures.
  - Example: Rabbit, Crow, Fox, Cat, Rock, Cloud.
2. Character Traits:
  - Categories: Positive (brave, clever) or negative (lazy, greedy).
  - Example: Brave squirrel, greedy lion.
3. Relationships:
  - Categories: Friend, rival, family, stranger.
  - Example: Rabbit and Fox are friends.
4. Setting:
  - Categories: Forest, village, sea, mountain, meadow, desert.
  - Example: A wise turtle in the desert.
5. Conflict Type:
  - Categories: Resource scarcity, betrayal, misunderstanding, helping others.
  - Example: Two animals compete for food.
6. Resolution Type:
  - Categories: Punishment, reward, collaboration, self-realization.
  - Example: The cunning fox is outsmarted.
7. Moral Themes:
  - Categories: Hard work, kindness, honesty, sharing, humility.
  - Example: “Kindness is always rewarded.”
8. Story Features:
  - Categories: Dialogue, plot twist, humor, unexpected endings.
  - Example: Dialogue between the squirrel and the tree.

### Algorithm for generating unique fables

#### Step 1: Define a Template (or in the case of LLM-based generations, a Prompt, or Prompt Stub)

Create templates that define the structure of a fable:

```text
[Character1] and [Character2] were in [Setting]. One day, [Conflict occurred]. In the end, [Resolution occurred].  
*Moral:* [Moral Lesson].
```

#### Step 2: Feature combinations

Create lists for each feature category:
  - Characters = [Rabbit, Fox, Squirrel, Crow, Cat, Turtle]
  - Traits = [Brave, Greedy, Kind, Lazy, Clever]
  - Settings = [Forest, River, Village, Meadow, Mountain]
  - Conflicts = [Competing for food, Losing a friend, Helping someone in need]
  - Resolutions = [Collaboration, Punishment, Reward]
  - Morals = [Kindness is rewarded, Hard work pays off, Honesty is the best policy]

#### Step 3: Generate variations

Combine features programatically:

```python
from itertools import product
characters = ['Rabbit', 'Fox', 'Squirrel']
traits = ['Brave', 'Greedy']
settings = ['Forest', 'River']
conflicts = ['Competing for food', 'Helping someone in need']
resolutions = ['Reward', 'Punishment']
morals = ['Kindness is rewarded', 'Hard work pays off']

fables = list(product(characters, traits, settings, conflicts, resolutions, morals))

# Example output:
# ('Rabbit', 'Brave', 'Forest', 'Competing for food', 'Reward', 'Kindness is rewarded')
```

#### Step 4: Fill template

Programmatically fill the template with the selected features:

```python
for fable in fables:
    character, trait, setting, conflict, resolution, moral = fable
    print(f"""
    Once there was a {trait.lower()} {character.lower()} in a {setting.lower()}. 
    One day, it was {conflict.lower()}. After much effort, {resolution.lower()}.
    *Moral:* {moral}.
    """)
```

#### Step 5: Ensure uniqueness
- Randomization: Add random adjectives, synonyms, or details for variability.
- Structural Variations: Use multiple templates to introduce diverse story formats.
- Feature Expansion: Increase the number of options in each feature category (e.g., 50+ characters, 20+ morals).

#### Tips for ensuring scale (over 2.5 million rows)
1. Multiply the options for each feature:
  - Characters (50) × Traits (10) × Settings (20) × Conflicts (10) × Resolutions (5) × Morals (10) = 500,000 unique combinations.
  - Add additional layers of variability (e.g., random dialogues, adjectives, or plot twists).
2. Introduce procedural generation rules:
  - For example: Alternate between two-character and three-character fables.
  - Vary moral placement (start, end, or implied in dialogue).
3. Use synonym replacement:
  - Replace words like “kind” with “generous” or “selfish” with “greedy” programmatically.

By structuring the dataset this way, an algorithm can easily generate millions of unique, well-formed fables.

