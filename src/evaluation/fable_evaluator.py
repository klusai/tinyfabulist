from src.utils.ai.evaluator import GptEvaluator
from src.utils.data_manager import DataManager

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
        self.evaluator = GptEvaluator(yaml_path=yaml_path)

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

        if not evaluation_result:
            print("No evaluation result received.")
            return

        evaluation_data = DataManager.extract_json_from_response(evaluation_result)
        additional_comments = DataManager.extract_additional_comments(evaluation_result)

        if not evaluation_data:
            print("Failed to extract JSON from the evaluation result.")
            return

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

        DataManager.append_to_json_file(output, entry)

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
            evaluation_data = DataManager.extract_json_from_response(evaluation_result)
            if evaluation_data:
                DataManager.save_diversity_scores(fables, evaluation_data, diversity_output)
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
