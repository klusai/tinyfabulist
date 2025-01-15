import argparse
import json
from gpt_eval import GPTEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate model generated fables using GPT-4.")

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

    evaluator = GPTEvaluator()

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

        with open(args.output, "w") as f:
            json.dump({
                "character": args.character,
                "trait": args.trait,
                "setting": args.setting,
                "conflict": args.conflict,
                "resolution": args.resolution,
                "moral": args.moral,
                "generated_fable": args.generated_fab,
                "evaluation": evaluation_result
            }, f, indent=4)

        print(f"Results saved to {args.output}")
    else:
        print("Evaluation failed. Please check the error logs.")

if __name__ == "__main__":
    main()