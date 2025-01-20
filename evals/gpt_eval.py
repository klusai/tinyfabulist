from openai import OpenAI
import os
from dotenv import load_dotenv
import yaml

# Load .env file to fetch API key
load_dotenv()

# Fetch the API key from the environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set it in a .env file.")

# Initialize the OpenAI client after the key is loaded
client = OpenAI(api_key=OPENAI_API_KEY)

class GPTEvaluator:
    def __init__(self, yaml_path):
        # Load YAML file
        with open(yaml_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def evaluate(self, character, trait, setting, conflict, resolution, moral, generated_fab):
        """
        Use GPT-4 to evaluate a fable based on structured input and its generated fable.
        """
        structured_input = (
            f"Character: {character}\n"
            f"Trait: {trait}\n"
            f"Setting: {setting}\n"
            f"Conflict: {conflict}\n"
            f"Resolution: {resolution}\n"
            f"Moral: {moral}"
        )

        # Load the evaluation prompt from YAML and format it
        evaluation_prompt = self.config['prompts']['evaluation_prompt'].format(
            character=character,
            trait=trait,
            setting=setting,
            conflict=conflict,
            resolution=resolution,
            moral=moral,
            generated_fab=generated_fab
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative writer and evaluator specializing in generating and assessing fables based on structured input."},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error while generating and evaluating fable: {e}")
            return None