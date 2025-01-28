import argparse
import json
import os
from gpt_eval import GPTEvaluator

def extract_json_from_response(response_text):
    """
    Extracts the JSON portion from a mixed plain-text response and returns a dictionary.
    """
    try:
        # Find the JSON portion between the first '{' and the last '}'
        start_idx = response_text.index('{')
        end_idx = response_text.rindex('}') + 1
        json_text = response_text[start_idx:end_idx]

        # Parse the JSON portion
        return json.loads(json_text)  # Convert to dictionary
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error extracting JSON: {e}")
        return None

def extract_additional_comments(response_text):
    """
    Extracts the additional comments section from the evaluator response.
    """
    # Extract anything after the JSON block as comments
    try:
        end_of_json = response_text.rindex('}') + 1
        comments = response_text[end_of_json:].strip()
        return comments if comments else "No additional comments provided."
    except ValueError:
        return "No additional comments provided."

def main():
    parser = argparse.ArgumentParser(description="Evaluate model generated fables using GPT-4.")

    parser.add_argument(
        "--yaml_path",
        type=str,
        required=False,
        help="Path to the YAML file."
    )

    parser.add_argument(
        "--character",
        type=str,
        required=True,
        help="The character of the fable."
    )

    parser.add_argument(
        "--trait",
        type=str,
        required=True,
        help="The trait of the character."
    )

    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        help="The setting of the fable."
    )

    parser.add_argument(
        "--conflict",
        type=str,
        required=True,
        help="The conflict in the fable."
    )

    parser.add_argument(
        "--resolution",
        type=str,
        required=True,
        help="The resolution of the fable."
    )

    parser.add_argument(
        "--moral",
        type=str,
        required=True,
        help="The moral of the fable."
    )

    parser.add_argument(
        "--generated_fab",
        type=str,
        required=True,
        help="The model's generated fable."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="evaluation_results.json",
        help="Output file to save the evaluation results (default: evaluation_results.json)."
    )

    args = parser.parse_args()

    yaml_path = args.yaml_path
    # Initialize the evaluator
    evaluator = GPTEvaluator(yaml_path=yaml_path)

    # Evaluate the fable
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
        print("Evaluation completed. Results:")
        print(evaluation_result)

        # Extract JSON and comments
        evaluation_data = extract_json_from_response(evaluation_result)
        additional_comments = extract_additional_comments(evaluation_result)

        if evaluation_data:
            # Prepare the updated JSON structure
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

            # Check if output file exists, and load existing data
            if os.path.exists(args.output):
                with open(args.output, "r") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]  # Convert to list if it's a single dictionary
                    except json.JSONDecodeError:
                        data = []  # If the file is empty or corrupted, start fresh
            else:
                data = []

            # Append the new entry
            data.append(entry)

            # Write back the updated data
            with open(args.output, "w") as f:
                json.dump(data, f, indent=4)

            print(f"Results saved to {args.output}")
        else:
            print("Failed to extract JSON from the evaluation result.")

if __name__ == "__main__":
    main()
