from openai import OpenAI
import yaml
from src.utils.config.environment import EnvConfig


class GptEvaluator:
    def __init__(self, yaml_path, gpt_model="gpt-4o", system_prompt=None):
        """
        Initializes the GptEvaluator with configurations from a YAML file and a specified GPT model.
        """

        # Load YAML configuration
        try:
            with open(yaml_path, 'r') as file:
                self.__config = yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading YAML file: {e}")

        # Initialize the OpenAI client after the key is loaded
        environment = EnvConfig()

        self.__client = OpenAI(api_key=environment.openai_api_key)
        self.__gpt_model = gpt_model

        # Load System Prompt
        self.__system_prompt = "You are a creative writer and evaluator specializing in generating and assessing fables based on structured input."

        if system_prompt:
            self.__system_prompt = system_prompt

    def evaluate(self, character, trait, setting, conflict, resolution, moral, generated_fab):
        """
        Use GPT-4 to evaluate a fable based on structured input and its generated fable.
        """

        # Load the evaluation prompt from YAML and format it
        evaluation_prompt = self.__config['prompts']['evaluation_prompt'].format(
            character=character,
            trait=trait,
            setting=setting,
            conflict=conflict,
            resolution=resolution,
            moral=moral,
            generated_fab=generated_fab
        )

        try:
            response = self.__client.chat.completions.create(
                model=self.__gpt_model,
                messages=[
                    {"role": "system", "content": self.__system_prompt},
                    {"role": "user", "content": evaluation_prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error while generating and evaluating fable: {e}")
            return None

    def evaluate_diversity(self, fables):
        """
        Use GPT-4 to evaluate the diversity of a list of fables.
        """
        # Load the diversity evaluation prompt from YAML
        diversity_prompt = self.__config['prompts']['diversity_eval_prompt']

        # Prepare the prompt with all generated fables
        prompt = f"{diversity_prompt}\n\n"
        
        for fable in fables:
            prompt += f"Fable: {fable}\n\n"

        try:
            response = self.__client.chat.completions.create(
                model=self.__gpt_model,
                messages=[
                    {"role": "system", "content": self.__system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error while evaluating diversity of fables: {e}")
            return None