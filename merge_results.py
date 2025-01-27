#!/usr/bin/env python3

import csv
import json

def load_fables_csv(csv_path):
    """
    Loads the CSV of fables with meta info.
    Returns a list of dictionaries (rows from CSV).
    """
    entries = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(row)
    return entries

def parse_fable_config(fable_config_str):
    """
    If you stored fable_config as a string of a Python dict, use ast.literal_eval.
    Or if you stored as JSON, do json.loads.
    """
    try:
        import ast
        parsed = ast.literal_eval(fable_config_str)
        return parsed
    except:
        # Fallback: maybe it's already a JSON string
        try:
            return json.loads(fable_config_str)
        except:
            return None

def make_unique_key(character, trait, setting, conflict, resolution, moral):
    """
    Create a unique key for matching generation and evaluation
    by the structured input fields.
    """
    return f"{character}|{trait}|{setting}|{conflict}|{resolution}|{moral}"

def load_evaluations(json_path):
    """
    Loads the JSON array of evaluations from GPT-4 (the list of dicts).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def parse_evaluation_text(evaluation_text):
    """
    Parses a multiline string like:
      Grammar: 10/10
      Creativity: 9/10
      Consistency: 10/10
      Age group: F: 13-16

      General Assessment: ...
    Extracts numeric scores for grammar, creativity, consistency,
    computes an average, and returns it as "score".
    We also return the raw text in "comments" or any leftover lines.

    Return example:
      {
        "score": 9.67,           # average
        "comments": "...",       # the raw text or leftover lines
        "prompt": ""             # if you want to store something else
      }
    """
    lines = evaluation_text.strip().split("\n")
    grammar_score = None
    creativity_score = None
    consistency_score = None

    # We'll accumulate leftover text lines into comments
    comments = []
    
    def parse_score(s: str):
        # Expects something like "10/10". We'll grab the number before '/'
        try:
            num_part = s.split("/")[0].strip()
            return float(num_part)
        except:
            return None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("grammar:"):
            # e.g. "Grammar: 10/10"
            after_colon = line.split(":", 1)[1].strip()
            grammar_score = parse_score(after_colon)
        elif line.lower().startswith("creativity:"):
            after_colon = line.split(":", 1)[1].strip()
            creativity_score = parse_score(after_colon)
        elif line.lower().startswith("consistency:"):
            after_colon = line.split(":", 1)[1].strip()
            consistency_score = parse_score(after_colon)
        else:
            # All other lines go into comments (Age group, general text, etc.)
            comments.append(line)

    # Compute average if we have any valid scores
    valid_scores = []
    for s in [grammar_score, creativity_score, consistency_score]:
        if s is not None:
            valid_scores.append(s)
    avg_score = None
    if valid_scores:
        avg_score = round(sum(valid_scores) / len(valid_scores), 2)

    return {
        "score": avg_score if avg_score is not None else "",
        "comments": "\n".join(comments),
        "prompt": ""  # If you want to store the actual evaluation prompt used, put it here
    }

def merge_data(generated_entries, eval_entries):
    """
    Merges the generated data (from CSV) with the eval data (from JSON).
    We match on the combination of (character, trait, setting, conflict, resolution, moral).
    Then we fill in the 'eval_*' columns (eval_avg_score, eval_llm_name, eval_llm_prompt).
    """
    # Build a lookup dict from that unique key -> eval results
    eval_dict = {}

    for e in eval_entries:
        character = e["character"]
        trait = e["trait"]
        setting = e["setting"]
        conflict = e["conflict"]
        resolution = e["resolution"]
        moral = e["moral"]
        evaluation_text = e.get("evaluation", "")

        # Parse that multiline text to get numeric scores and leftover text
        eval_data = parse_evaluation_text(evaluation_text)

        unique_key = make_unique_key(character, trait, setting, conflict, resolution, moral)
        eval_dict[unique_key] = {
            "eval_avg_score": eval_data["score"],
            "eval_llm_name": "gpt-4",  # Hard-code or derive from your actual code
            "eval_llm_prompt": eval_data["comments"]  # or store the entire text in 'comments'
        }

    # Now update each row from your generated CSV
    merged_rows = []
    for row in generated_entries:
        parsed_config = parse_fable_config(row.get("fable_config", ""))
        if not parsed_config:
            # Can't match if there's no config
            merged_rows.append(row)
            continue

        character = parsed_config.get("character")
        trait = parsed_config.get("trait")
        setting = parsed_config.get("setting")
        conflict = parsed_config.get("conflict")
        resolution = parsed_config.get("resolution")
        moral = parsed_config.get("moral")

        key = make_unique_key(character, trait, setting, conflict, resolution, moral)

        if key in eval_dict:
            row["eval_avg_score"] = eval_dict[key]["eval_avg_score"]
            row["eval_llm_name"] = eval_dict[key]["eval_llm_name"]
            row["eval_llm_prompt"] = eval_dict[key]["eval_llm_prompt"]

        merged_rows.append(row)

    return merged_rows

def write_final_csv(merged_rows, output_path="fables_final.csv"):
    """
    Writes the final merged data to CSV. We assume
    merged_rows already contain the columns needed.
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

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow(row)

def main():
    """
    Run the merge process:
      - Load your fables_with_meta.csv (the generation step with all metadata).
      - Load your evaluation_results.json (the multi-line text evaluations).
      - Merge them on (character, trait, setting, conflict, resolution, moral).
      - Write final CSV with all columns (including eval_avg_score).
    """
    generated_csv = "fables_with_meta.csv"      # adapt path if needed
    evaluation_json = "evaluation_results.json" # adapt path if needed

    generated_entries = load_fables_csv(generated_csv)
    eval_entries = load_evaluations(evaluation_json)

    merged_rows = merge_data(generated_entries, eval_entries)
    write_final_csv(merged_rows, "fables_final.csv")

    print("Final merged CSV with all columns saved to fables_final.csv")

if __name__ == "__main__":
    main()
