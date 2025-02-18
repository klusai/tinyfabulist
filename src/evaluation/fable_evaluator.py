import json
import os
import re
from src.utils.ai.evaluator import GptEvaluator

class FableEvaluator:
    """
    A service class to run evaluations on model-generated fables using a GPT-based evaluator.
    It supports both single fable evaluations and diversity evaluations.
    """
    def __init__(self, yaml_path: str):
        """
        Initialize the EvaluationService with the YAML configuration file.

        Args:
            yaml_path (str): Path to the YAML configuration file.
        """
        self.yaml_path = yaml_path
        self.evaluator = GptEvaluator(yaml_path=yaml_path)

    @staticmethod
    def extract_json_from_response(response_text: str):
        """
        Extracts the first JSON-like structure from the response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            dict or None: Parsed JSON data if extraction is successful, else None.
        """
        try:
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_text = json_match.group(0).strip()
                return json.loads(json_text)
            else:
                print("No valid JSON found in response.")
                return None
        except json.JSONDecodeError as e:
            print(f"Error extracting JSON: {e}\nResponse text:\n{response_text}")
            return None

    @staticmethod
    def extract_additional_comments(response_text: str) -> str:
        """
        Extracts additional comments from the evaluator's response text.

        Args:
            response_text (str): The evaluator's response text.

        Returns:
            str: The extracted comments or a default message.
        """
        try:
            end_of_json = response_text.rindex('}') + 1
            comments = response_text[end_of_json:].strip()
            return comments if comments else "No additional comments provided."
        except ValueError:
            return "No additional comments provided."

    @staticmethod
    def save_diversity_scores(fables, evaluation_data, output_file: str):
        """
        Saves diversity evaluation results to a JSON file.

        Args:
            fables (list): List of fable texts.
            evaluation_data (dict): The diversity evaluation data.
            output_file (str): Path to the output file.
        """
        print("Saving diversity scores to JSON...")
        diversity_results = {
            "fables": fables,
            "diversity_evaluation": evaluation_data
        }
        with open(output_file, "w", encoding="utf-8") as json_file:
            json.dump(diversity_results, json_file, indent=4, ensure_ascii=False)
        print(f"Diversity evaluation saved to {output_file}")

    def evaluate_fable(self, character: str, trait: str, setting: str, conflict: str,
                       resolution: str, moral: str, generated_fab: str,
                       output: str = "evaluation_results.json"):
        """
        Evaluates a single fable and appends the result to a JSON file.

        Args:
            character (str): The character in the fable.
            trait (str): The trait of the character.
            setting (str): The fable's setting.
            conflict (str): The conflict in the fable.
            resolution (str): The resolution of the conflict.
            moral (str): The moral of the fable.
            generated_fab (str): The generated fable text.
            output (str, optional): Output file for the evaluation results.
        """
        print("Evaluating the model's fable...")
        evaluation_result = self.evaluator.evaluate(
            character=character,
            trait=trait,
            setting=setting,
            conflict=conflict,
            resolution=resolution,
            moral=moral,
            generated_fab=generated_fab
        )

        if evaluation_result:
            evaluation_data = self.extract_json_from_response(evaluation_result)
            additional_comments = self.extract_additional_comments(evaluation_result)

            if evaluation_data:
                entry = {
                    "character": character,
                    "trait": trait,
                    "setting": setting,
                    "conflict": conflict,
                    "resolution": resolution,
                    "moral": moral,
                    "generated_fable": generated_fab,
                    "grammar": evaluation_data.get("Grammar", "n/a"),
                    "creativity": evaluation_data.get("Creativity", "n/a"),
                    "consistency": evaluation_data.get("Consistency", "n/a"),
                    "age_group": evaluation_data.get("Age group", "n/a"),
                    "comments": additional_comments
                }

                # Load existing evaluation results if the file exists.
                if os.path.exists(output):
                    try:
                        with open(output, "r") as f:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = [data]
                    except json.JSONDecodeError:
                        data = []
                else:
                    data = []

                data.append(entry)
                with open(output, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Results saved to {output}")
            else:
                print("Failed to extract JSON from the evaluation result.")
        else:
            print("No evaluation result received.")

    def evaluate_diversity(self, fables, diversity_output: str = "diversity_evaluation.csv"):
        """
        Runs a diversity evaluation on a list of fables and saves the result.

        Args:
            fables (list): List of fable texts.
            diversity_output (str, optional): Output file for the diversity evaluation.
        """
        print("Running diversity evaluation...")
        evaluation_result = self.evaluator.evaluate_diversity(fables)
        print("Diversity Evaluation Response:", evaluation_result)
        if evaluation_result:
            evaluation_data = self.extract_json_from_response(evaluation_result)
            if evaluation_data:
                self.save_diversity_scores(fables, evaluation_data, diversity_output)
        else:
            print("No diversity evaluation result received.")

    def run_evaluation(self, evaluation_type: str, **kwargs):
        """
        Executes an evaluation based on the provided type.

        Args:
            evaluation_type (str): Either "evaluation_prompt" for a single fable evaluation
                                   or "diversity_eval_prompt" for a diversity evaluation.
            kwargs: Additional parameters required for the chosen evaluation.
        """
        if evaluation_type == "evaluation_prompt":
            self.evaluate_fable(
                character=kwargs.get("character"),
                trait=kwargs.get("trait"),
                setting=kwargs.get("setting"),
                conflict=kwargs.get("conflict"),
                resolution=kwargs.get("resolution"),
                moral=kwargs.get("moral"),
                generated_fab=kwargs.get("generated_fab"),
                output=kwargs.get("output", "evaluation_results.json")
            )
        elif evaluation_type == "diversity_eval_prompt":
            self.evaluate_diversity(
                fables=kwargs.get("fables"),
                diversity_output=kwargs.get("diversity_output", "diversity_evaluation.csv")
            )
        else:
            raise ValueError("Invalid evaluation type. Use 'evaluation_prompt' or 'diversity_eval_prompt'.")
