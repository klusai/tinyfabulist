import argparse
import json
from gpt_eval import GPTEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate model completions using GPT-4.")

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="The beginning of the story (prompt) to evaluate."
    )

    parser.add_argument(
        "--completion",
        type=str,
        required=True,
        help="The model's completion of the story."
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

    print("Evaluating the model's completion...")
    evaluation_result = evaluator.evaluate(args.prompt, args.completion)

    if evaluation_result:
        print("Evaluation completed. Results:")
        print(evaluation_result)

        with open(args.output, "w") as f:
            json.dump({"prompt": args.prompt, "completion": args.completion, "evaluation": evaluation_result}, f, indent=4)

        print(f"Results saved to {args.output}")
    else:
        print("Evaluation failed. Please check the error logs.")

if __name__ == "__main__":
    main()
