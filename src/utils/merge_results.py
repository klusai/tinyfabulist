#!/usr/bin/env python3

import csv
import json
import ast

def load_fables_csv(csv_path):
    """
    Loads the CSV of fables with meta info.
    Returns a list of dictionaries (rows from CSV).
    """
    entries = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(row)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
    return entries

def parse_fable_config(fable_config_str):
    """
    Parses the fable_config string (either Python dict or JSON format).
    """
    try:
        return ast.literal_eval(fable_config_str)
    except (SyntaxError, ValueError):
        try:
            return json.loads(fable_config_str)
        except json.JSONDecodeError:
            return None

def make_unique_key(character, trait, setting, conflict, resolution, moral):
    """
    Create a unique key for matching generation and evaluation
    by the structured input fields.
    """
    return f"{character}|{trait}|{setting}|{conflict}|{resolution}|{moral}"

def load_evaluations(json_path):
    """
    Loads the JSON array of evaluations from GPT-4 (list of dicts).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File {json_path} is not valid JSON.")
        return []

def parse_score(score_str):
    """
    Parses a score string to extract the numeric value or handles 'N/A'.
    """
    try:
        if "n/a" in score_str.lower():
            return None
        return float(score_str.split("/")[0].strip())
    except:
        return None

def parse_evaluation(evaluation):
    """
    Parses evaluation data to extract avg_score and comments for eval_llm_prompt.
    """
    scores = []
    comments_text = evaluation.get("comments", "")

    # Parse numeric scores for grammar, creativity, and consistency
    for score_str in ["grammar", "creativity", "consistency"]:
        score = parse_score(evaluation.get(score_str, ""))
        if score is not None:
            scores.append(score)

    # Compute the average score
    avg_score = round(sum(scores) / len(scores), 2) if scores else None

    return {
        "eval_avg_score": avg_score,
        "eval_llm_prompt": comments_text.strip() if comments_text else ""
    }

def merge_data(generated_entries, eval_entries):
    """
    Merges the generated data (from CSV) with the eval data (from JSON).
    Matches on the unique key derived from (character, trait, setting, conflict, resolution, moral).
    """
    eval_dict = {}

    # Build a dictionary from the evaluation JSON
    for e in eval_entries:
        character = e.get("character", "").strip()
        trait = e.get("trait", "").strip()
        setting = e.get("setting", "").strip()
        conflict = e.get("conflict", "").strip()
        resolution = e.get("resolution", "").strip()
        moral = e.get("moral", "").strip()

        unique_key = make_unique_key(character, trait, setting, conflict, resolution, moral)
        eval_data = parse_evaluation(e)

        # Store evaluation data using the unique key
        eval_dict[unique_key] = eval_data

    merged_rows = []
    for row in generated_entries:
        # Parse the fable_config to extract input details
        parsed_config = parse_fable_config(row.get("fable_config", ""))
        if not parsed_config:
            merged_rows.append(row)
            continue

        character = parsed_config.get("character", "").strip()
        trait = parsed_config.get("trait", "").strip()
        setting = parsed_config.get("setting", "").strip()
        conflict = parsed_config.get("conflict", "").strip()
        resolution = parsed_config.get("resolution", "").strip()
        moral = parsed_config.get("moral", "").strip()

        key = make_unique_key(character, trait, setting, conflict, resolution, moral)

        # Add evaluation data if there's a match
        if key in eval_dict:
            row["eval_avg_score"] = eval_dict[key]["eval_avg_score"]
            row["eval_llm_name"] = "gpt-4"  # Assuming the LLM name
            row["eval_llm_prompt"] = eval_dict[key]["eval_llm_prompt"]

        merged_rows.append(row)

    return merged_rows

def write_final_csv(merged_rows, output_path="fables_final.csv"):
    """
    Writes the final merged data to a CSV file.
    """
    fieldnames = [
        "fable_config",
        "fable_prompt",
        "fable_text_en",
        "eval_avg_score",
        "eval_llm_name",
        "eval_llm_prompt",
        "llm_name",
        "llm_input_tokens",
        "llm_output_tokens",
        "llm_inference_time",
        "llm_inference_cost_usd",
        "host_provider",
        "host_dc_provider",
        "host_dc_location",
        "host_gpu",
        "host_gpu_vram",
        "host_cost_per_hour",
        "generation_datetime",
        "pipeline_version"
    ]

    try:
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in merged_rows:
                writer.writerow(row)
    except Exception as e:
        print(f"Error writing CSV: {e}")

def main():
    """
    Main function to load, process, and merge data.
    """
    generated_csv = "fables_with_meta.csv"
    evaluation_json = "evaluation_results.json"

    generated_entries = load_fables_csv(generated_csv)
    eval_entries = load_evaluations(evaluation_json)

    merged_rows = merge_data(generated_entries, eval_entries)
    write_final_csv(merged_rows, "fables_final.csv")

    print("Final merged CSV with all columns saved to fables_final.csv")

if __name__ == "__main__":
    main()
