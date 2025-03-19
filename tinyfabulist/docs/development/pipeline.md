# **Pipeline V1: AI-Generated Fable Creation**

## **1. Introduction**
This document describes the first version (**V1**) of the pipeline used to generate structured fables using an AI model. The pipeline automates the process of producing fables by defining structured inputs, passing them through a **Hugging Face OpenAI-compatible endpoint**, and ensuring the output follows a strict narrative format.

The goal is to generate fables that are:
- **Creative and engaging**
- **Consistent in structure**
- **Aligned with predefined moral lessons**
- **Diverse in vocabulary, themes, and settings**

This document also includes a tutorial on generating fables from scratch, explaining each stage of the pipeline and its role in ensuring a high-quality dataset.

---

## **2. Pipeline Overview**
The pipeline consists of the following components, each playing a crucial role in data generation:

1. **Configuration Setup** – Stores predefined storytelling elements to structure each fable.
2. **Data Generation Logic** – Dynamically constructs structured inputs for AI generation.
3. **AI Model Invocation** – Sends structured prompts to an AI model for fable creation.
4. **Metadata Storage** – Saves generated fables along with inference statistics for further analysis.

The core idea is to **define structured prompts**, feed them into an AI model, and ensure outputs follow a strict storytelling format while maintaining variation and creativity.

---

## **3. Structured Input Definition**
Each fable is generated based on a predefined structure composed of key storytelling elements:

```yaml
character: Fox
trait: Clever
setting: Forest
conflict: Helping someone in need
resolution: Reward
moral: Kindness is rewarded
```

### **Role of Structured Inputs**
- **Ensures consistency** – All fables follow the same format, making them easier to compare and evaluate.
- **Encourages diversity** – Different combinations of characters, settings, and morals produce unique fables.
- **Controls storytelling elements** – Ensures that generated narratives align with intended moral lessons.

The AI model uses these structured inputs to generate fables in a defined format:
```text
Once there was a clever fox in a forest. 
One day, it was helping a trapped bird escape from a hunter’s net. 
After much effort, the fox gnawed through the net and freed the bird. 
Grateful for its rescue, the bird led the fox to a hidden cache of berries. 
Moral: Kindness is rewarded.
```
This approach guarantees both **uniformity** and **engaging storytelling**.

---

## **4. Implementing the Pipeline**

### **4.1. Configuration Setup**
A YAML configuration file stores the possible storytelling components:
```yaml
characters:
  - Rabbit
  - Fox
  - Squirrel

traits:
  - Brave
  - Greedy
  - Wise

settings:
  - Forest
  - River

conflicts:
  - Competing for food
  - Helping someone in need

resolutions:
  - Reward
  - Punishment

morals:
  - Kindness is rewarded
  - Hard work pays off
```
### **Role of Configuration Setup**
- **Defines storytelling elements** – Serves as a structured reference for generating fables.
- **Ensures variation** – Provides multiple options for character attributes, settings, and resolutions.
- **Standardizes input structure** – Ensures AI receives structured prompts for coherent output.

### **4.2. Generating Input Prompts**
A structured input is created from the configuration values:
```python
import random
from itertools import product

# Load predefined options from config
characters = ["Rabbit", "Fox", "Squirrel"]
traits = ["Brave", "Greedy", "Wise"]
settings = ["Forest", "River"]
conflicts = ["Competing for food", "Helping someone in need"]
resolutions = ["Reward", "Punishment"]
morals = ["Kindness is rewarded", "Hard work pays off"]

# Generate all possible combinations
combinations = list(product(characters, traits, settings, conflicts, resolutions, morals))
random.shuffle(combinations)

# Select a random fable structure
character, trait, setting, conflict, resolution, moral = random.choice(combinations)
```
### **Role of Data Generation Logic**
- **Automates input generation** – Dynamically selects different combinations for unique fables.
- **Eliminates redundancy** – Ensures broad coverage of different character traits and moral lessons.
- **Prepares inputs for AI processing** – Formats inputs for seamless AI model integration.

---

## **5. AI Model Integration**

### **5.1. Constructing the Model Prompt**
The structured input is converted into an AI-ready prompt:
```python
prompt_text = (
    "You are an AI that generates structured fables.\n\n"
    "Once there was a {trait} {character} in a {setting}. "
    "One day, it was {conflict}. "
    "After much effort, {resolution}. "
    "Moral: {moral}."
).format(
    character=character, trait=trait, setting=setting,
    conflict=conflict, resolution=resolution, moral=moral
)
```
### **Role of AI Model Invocation**
- **Guides AI response** – Ensures AI-generated output follows a strict template.
- **Maintains creativity within constraints** – Allows engaging narratives while keeping structure intact.
- **Facilitates automation** – Enables batch generation of multiple fables at once.

### **5.2. Sending the Prompt to the AI Model**
```python
from openai import OpenAI
import os

# Load environment variables for API
ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize the client
client = OpenAI(base_url=ENDPOINT_URL, api_key=HF_TOKEN)

# Generate fable
response = client.completions.create(
    model="Llama-3.1-8B-Instruct",
    prompt=prompt_text,
    max_tokens=300,
    stream=True
)

# Collect generated output
fable_text = "".join(chunk.choices[0].text for chunk in response)
```
### **Role of AI Model Processing**
- **Interprets structured input** – Uses the predefined structure to generate meaningful fables.
- **Applies AI creativity** – Introduces unique variations while maintaining coherence.
- **Ensures scalability** – Enables generation of large datasets efficiently.

---

## **6. Storing Generated Fables**
```python
import json
from datetime import datetime

data_entry = {
    "fable_config": {
        "character": character,
        "trait": trait,
        "setting": setting,
        "conflict": conflict,
        "resolution": resolution,
        "moral": moral
    },
    "fable_text": fable_text,
    "timestamp": datetime.utcnow().isoformat()
}

# Save to JSON file
with open("generated_fables.json", "a") as file:
    file.write(json.dumps(data_entry) + "\n")
```
### **Role of Metadata Storage**
- **Preserves generated data** – Enables further analysis and evaluation.
- **Facilitates future improvements** – Provides insights into AI model behavior and areas of enhancement.
- **Supports dataset curation** – Helps filter out redundant or low-quality fables.

---

## **7. Conclusion**
This pipeline provides a **structured, automated** approach to fable generation. By defining controlled inputs and enforcing a strict structure, it ensures **coherent, engaging, and meaningful** stories while leveraging AI to introduce creativity. Future improvements may include **fine-tuning the model, expanding themes, and increasing lexical diversity**.

