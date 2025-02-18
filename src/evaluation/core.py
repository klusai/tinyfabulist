import yaml
from src.evaluation.fable_evaluator import FableEvaluator
from src.utils.data_manager import DataManager

def run():
    csv_path = "/home/ap/Documents/Work/Research/tiny-fabulist/src/artifacts/fables_with_meta.csv"
    yaml_path = "/home/ap/Documents/Work/Research/tiny-fabulist/src/evaluation/config.yml"
    num_fables = 5  # Change this number to control how many fables are evaluated for diversity

    # Initialize the data manager and update the YAML config with CSV data.
    data_manager = DataManager(csv_path, yaml_path)
    fables = data_manager.load_fables_from_csv()
    data_manager.update_yaml_with_fables(fables)

    # Read the updated YAML configuration.
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    data = config.get("data", [])

    # Create an instance of the EvaluationService.
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
            output="evaluation_results.json"
        )

    # For diversity evaluation, load a subset of fables.
    fables_subset = data_manager.load_fables_from_yaml(num_fables)
    eval_service.run_evaluation(
        evaluation_type="diversity_eval_prompt",
        fables=fables_subset,
        diversity_output="diversity_evaluation.json"
    )

if __name__ == "__main__":
    run()
