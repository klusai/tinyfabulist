import argparse
import json
import os
import csv
from gpt_eval import GPTEvaluator
import re

def extract_json_from_response(response_text):
    try:
        # Use regex to find the first JSON-like structure
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


def save_diversity_scores(fables, evaluation_data, output_file):
    """
    Saves the diversity evaluation results to a JSON file.
    """
    print("Saving diversity scores to JSON...")

    diversity_results = {
        "fables": fables,
        "diversity_evaluation": evaluation_data
    }

    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(diversity_results, json_file, indent=4, ensure_ascii=False)
    
    print(f"Diversity evaluation saved to {output_file}")



def extract_additional_comments(response_text):
    """
    Extracts the additional comments section from the evaluator response.
    """
    try:
        end_of_json = response_text.rindex('}') + 1
        comments = response_text[end_of_json:].strip()
        return comments if comments else "No additional comments provided."
    except ValueError:
        return "No additional comments provided."

def main():
    parser = argparse.ArgumentParser(description="Evaluate model generated fables using GPT-4.")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to the YAML file.")
    parser.add_argument("--evaluation_type", type=str, required=True, choices=["evaluation_prompt", "diversity_eval_prompt"], help="Type of evaluation to run.")
    parser.add_argument("--character", type=str, help="The character of the fable.")
    parser.add_argument("--trait", type=str, help="The trait of the character.")
    parser.add_argument("--setting", type=str, help="The setting of the fable.")
    parser.add_argument("--conflict", type=str, help="The conflict in the fable.")
    parser.add_argument("--resolution", type=str, help="The resolution of the fable.")
    parser.add_argument("--moral", type=str, help="The moral of the fable.")
    parser.add_argument("--generated_fab", type=str, help="The model's generated fable.")
    parser.add_argument("--fables", type=str, nargs='*', help="List of fables for diversity evaluation.")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file to save the evaluation results.")
    parser.add_argument("--diversity_output", type=str, default="diversity_evaluation.csv", help="Output file for diversity scores.")
    
    args = parser.parse_args()
    evaluator = GPTEvaluator(yaml_path=args.yaml_path)

    if args.evaluation_type == "evaluation_prompt":
        print("Evaluating the model's fable...")
        evaluation_result = evaluator.evaluate(
            character=args.character,
            trait=args.trait,
            setting=args.setting,
            conflict=args.conflict,
            resolution=args.resolution,
            moral=args.moral,
            generated_fab=args.generated_fab
        )

        if evaluation_result:
            evaluation_data = extract_json_from_response(evaluation_result)
            additional_comments = extract_additional_comments(evaluation_result)
            
            if evaluation_data:
                entry = {
                    "character": args.character,
                    "trait": args.trait,
                    "setting": args.setting,
                    "conflict": args.conflict,
                    "resolution": args.resolution,
                    "moral": args.moral,
                    "generated_fable": args.generated_fab,
                    "grammar": evaluation_data.get("Grammar", "n/a"),
                    "creativity": evaluation_data.get("Creativity", "n/a"),
                    "consistency": evaluation_data.get("Consistency", "n/a"),
                    "age_group": evaluation_data.get("Age group", "n/a"),
                    "comments": additional_comments
                }

                if os.path.exists(args.output):
                    with open(args.output, "r") as f:
                        try:
                            data = json.load(f)
                            if not isinstance(data, list):
                                data = [data]
                        except json.JSONDecodeError:
                            data = []
                else:
                    data = []

                data.append(entry)
                with open(args.output, "w") as f:
                    json.dump(data, f, indent=4)
                print(f"Results saved to {args.output}")
            else:
                print("Failed to extract JSON from the evaluation result.")

    elif args.evaluation_type == "diversity_eval_prompt":
        print("Running diversity evaluation...")
        evaluation_result = evaluator.evaluate_diversity(args.fables)
        print("Diversity Evaluation Response:", evaluation_result)
        if evaluation_result:
            evaluation_data = extract_json_from_response(evaluation_result)
            if evaluation_data:
                scores = [evaluation_data["diversity_score"]] if "diversity_score" in evaluation_data else []
                save_diversity_scores(args.fables, evaluation_data, args.diversity_output)

if __name__ == "__main__":
    main()
