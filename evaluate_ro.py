import json
import sys
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from decouple import config
from openai import OpenAI
from dotenv import load_dotenv
from pybars import Compiler

from logger import setup_logging
from utils import load_settings

logger = setup_logging()


def evaluate_fable_save(fable_data: dict, lock:threading.Lock) -> dict:
    """
    Takes a JSON object representing a generated fable (which may contain additional stats),
    evaluates the fable using the OpenAI API based on predefined criteria, and returns the evaluation result
    along with the original data and a timestamp.
    
    The evaluation is based on the following criteria:
      1. Grammar and Style (1-10)
      2. Creativity and Originality (1-10)
      3. Moral Clarity (1-10)
      4. Adherence to Instructions (1-10)
      
    The API is instructed to respond in a valid JSON format containing these scores and brief explanations.
    """

    if not fable_data:
        return

    load_dotenv()
    # Load settings and get evaluator configuration
    settings = load_settings()
    evaluator_config = settings.get("evaluator", {})
    model = evaluator_config.get("model", "gpt-4o")

    # Get prompts from config
    prompts = evaluator_config.get("prompt", {})
    system_prompt = prompts.get("system_ro", None)
    evaluation_prompt_template = prompts.get("evaluation_ro", "")

    fable_text = fable_data.get("fable", "")
    prompt = fable_data.get("prompt", "")
    
    with lock:
        compiler = Compiler()
        template = compiler.compile(evaluation_prompt_template)
        # Render the template with the provided fable and prompt
        evaluation_prompt = template({"fable": fable_text, "prompt": prompt})


    client = OpenAI(api_key=config("OPENAI_API_KEY"))
    try:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": evaluation_prompt},
            ],
            response_format={"type": "json_object"},
            reasoning_effort="medium"
        )
        evaluation_text = chat_completion.choices[0].message.content.strip()
        try:
            evaluation_json = json.loads(evaluation_text)
        except json.JSONDecodeError as json_err:
            evaluation_json = {"error": f"JSON decoding error: {str(json_err)}. Raw response: {evaluation_text[:200]}"}
    except Exception as e:
        evaluation_json = {"error": f"OpenAI API error: {str(e)}"}
    
    # Merge the original fable data (including any additional stats) with the evaluation results.
    result = dict(fable_data)
    result.update({
        "evaluation": evaluation_json,
        "evaluation_timestamp": datetime.now().isoformat()
    })
    return result


def evaluate_file(input_path: str, output_path: str) -> None:
    """
    Reads each line from the input JSONL file (including all stats), evaluates the fable using evaluate_fable_save,
    and writes the combined evaluation results to the output JSONL file.
    """
    results = []
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    output_lock = threading.Lock()

    # Use ThreadPoolExecutor for parallel evaluations.
    with ThreadPoolExecutor(max_workers=50) as executor:
        futures = []
        for line in lines:
            if line.strip():
                try:
                    fable_data = json.loads(line)
                    futures.append(executor.submit(evaluate_fable_save, fable_data,output_lock))
                except json.JSONDecodeError:
                    print("Skipping invalid JSON line.")
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    # Write the combined evaluation results to the output JSONL file.
    with open(output_path, "w", encoding="utf-8") as outfile:
        for res in results:
            outfile.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(f"Evaluations have been completed and saved in {output_path}")


if __name__ == "__main__":
    import argparse

    settings = load_settings()
    evaluator_config = settings.get("evaluator", {})
    model = evaluator_config.get("model", "gpt-4o")

    parser = argparse.ArgumentParser(
        description="Evaluate the fables from the specified file or directory (including all existing stats) and automatically save the results with a timestamp in data/evaluations_ro."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input JSONL file or directory containing JSONL files"
    )
    args = parser.parse_args()
    
    input_path = args.input
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = os.path.join("data", "evaluations_ro")
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    if os.path.isfile(input_path):
        # Processing a single file
        basename = os.path.basename(input_path)
        output_file = os.path.join(output_base_dir, f"evaluations_{timestamp}_{basename}")
        evaluate_file(input_path, output_file)
    elif os.path.isdir(input_path):
        # Recursively processing all JSONL files in the directory
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('.jsonl'):
                    input_file = os.path.join(root, file)
                    # Preserve the relative directory structure
                    rel_path = os.path.relpath(root, input_path)
                    output_dir = os.path.join(output_base_dir, rel_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_file = os.path.join(output_dir, f"{file}_jsonl_eval_e{model}_dt{timestamp}.jsonl")
                    evaluate_file(input_file, output_file)
    else:
        print("The provided input path is not a valid file or directory.")
        sys.exit(1)
