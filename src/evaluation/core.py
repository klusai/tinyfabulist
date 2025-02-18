import yaml
from src.evaluation.fable_evaluator import FableEvaluator
from src.utils.data_manager import DataManager

def execute_evaluations(csv_path: str, yaml_path: str, num_fables: int = 20,
                                      evaluation_output: str = "evaluation_results.json",
                                      diversity_output: str = "diversity_evaluation.json") -> None:
    """
    Execute a pipeline of fable evaluations based on provided CSV and YAML configuration files.

    Args:
        csv_path (str): Path to the CSV file containing fables with metadata.
        yaml_path (str): Path to the YAML configuration file.
        num_fables (int): Number of fables to evaluate for diversity.
        evaluation_output (str): Filename for individual evaluation results.
        diversity_output (str): Filename for diversity evaluation results.
    """
    # Initialize the data manager and update the YAML config with CSV data.
    data_manager = DataManager(csv_path, yaml_path)
    fables = data_manager.load_fables_from_csv()
    data_manager.update_yaml_with_fables(fables)

    # Read the updated YAML configuration.
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    data = config.get("data", [])

    # Create an instance of the FableEvaluator.
    eval_service = FableEvaluator(yaml_path=yaml_path)

    # Evaluate each fable individually.
    for pair in data:
        eval_service.run_evaluation(
            evaluation_type="evaluation_prompt",
            character=pair["character"],
            trait=pair["trait"],
            setting=pair["setting"],
            conflict=pair["conflict"],
            resolution=pair["resolution"],
            moral=pair["moral"],
            generated_fab=pair["generated_fab"],
            output=evaluation_output
        )

    # For diversity evaluation, load a subset of fables.
    fables_subset = data_manager.load_fables_from_yaml(num_fables)
    eval_service.run_evaluation(
        evaluation_type="diversity_eval_prompt",
        fables=fables_subset,
        diversity_output=diversity_output
    )

if __name__ == "__main__":
    # Example usage with specified file paths and number of fables.
    csv_path = "src/artifacts/fables_with_meta.csv"
    yaml_path = "src/evaluation/config.yml"
    execute_evaluations(csv_path, yaml_path)
