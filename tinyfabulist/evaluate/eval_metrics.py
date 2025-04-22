#!/usr/bin/env python3
"""
A simple evaluation script for fables that computes:
  - Distinct-n (diversity) scores,
  - Flesch Reading Ease,
  - Self-BLEU (across all fables in the file).

Additionally, it computes an overall distinct-n score and an overall
Flesch Reading Ease score for the entire file (averaging the per-fable scores).

Usage:
    python evaluate_fables_metrics.py --input path/to/fables.jsonl --output_dir path/to/output
"""

import json
import os
from datetime import datetime
import argparse

import nltk
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import textstat

from tinyfabulist.evaluate.utils import EvaluationUtils, load_jsonl_entries, process_file_or_directory
# Ensure necessary resources are downloaded (for tokenization).
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def evaluate_fable(entry):
    """
    Evaluate a single fable entry by computing its metrics.
    Expected that the entry contains a "fable" field.
    Adds the following keys:
       - "flesch_reading_ease"
       - "distinct_1" (distinct unigram ratio)
       - "distinct_2" (distinct bigram ratio)
    
    :param entry: Dictionary representing a fable entry.
    :return: Updated dictionary with added metrics.
    """
    fable_text = entry.get("fable", "")
    entry["flesch_reading_ease"] = EvaluationUtils.get_readability(fable_text)
    entry["distinct_1"] = EvaluationUtils.distinct_n(fable_text, 1)
    entry["distinct_2"] = EvaluationUtils.distinct_n(fable_text, 2)
    return entry


def evaluate_file(file_path, output_dir=None):
    """
    Read the JSONL file from file_path, evaluate each fable,
    compute an overall self-BLEU across all fables, and save the results.
    
    The output will be a JSON file containing:
      - "evaluated_entries": list of evaluated fable entries
      - "self_bleu": overall average self-BLEU score
      - "self_bleu_scores": list of individual self-BLEU scores
      - "overall_distinct_1": average distinct-1 score of all evaluated fables
      - "overall_flesch_reading_ease": average Flesch Reading Ease score of all evaluated fables
    
    :param file_path: Path to the input JSONL file.
    :param output_dir: Directory to save results; defaults to same directory as file_path.
    """
    print(f"Evaluating file: {file_path}")
    entries = load_jsonl_entries(file_path)
    
    evaluated_entries = []
    fable_texts = []
    distinct_1_scores = []
    reading_ease_scores = []
    
    for entry in entries:
        if "fable" in entry:
            evaluated_entry = evaluate_fable(entry)
            evaluated_entries.append(evaluated_entry)
            fable_texts.append(evaluated_entry["fable"])
            distinct_1_scores.append(evaluated_entry["distinct_1"])
            reading_ease_scores.append(evaluated_entry["flesch_reading_ease"])
        else:
            evaluated_entries.append(entry)
    
    # Compute self-BLEU if there are at least two fables.
    if len(fable_texts) > 1:
        avg_bleu, bleu_scores = EvaluationUtils.compute_self_bleu(fable_texts)
    else:
        avg_bleu = None
        bleu_scores = []
    
    # Compute overall distinct_1 and overall Flesch Reading Ease scores (averages)
    if distinct_1_scores:
        overall_distinct_1 = sum(distinct_1_scores) / len(distinct_1_scores)
    else:
        overall_distinct_1 = None
    
    if reading_ease_scores:
        overall_flesch_reading_ease = sum(reading_ease_scores) / len(reading_ease_scores)
    else:
        overall_flesch_reading_ease = None
    
    result = {
        "evaluated_entries": evaluated_entries,
        "self_bleu": avg_bleu,
        "self_bleu_scores": bleu_scores,
        "distinct_1": overall_distinct_1,
        "flesch_reading_ease": overall_flesch_reading_ease
    }
    
    if output_dir is None:
        output_dir = os.path.dirname(file_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_file = os.path.join(output_dir, f"{base_name}_evaluated_{timestamp}.json")
    
    with open(output_file, "w", encoding="utf-8") as out_f:
        json.dump(result, out_f, indent=2)
    print(f"Evaluation results written to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fables using distinct n-gram, Flesch Reading Ease, and self-BLEU metrics."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to the JSONL file with fables")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    args = parser.parse_args()
    evaluate_file(args.input, args.output_dir)


if __name__ == "__main__":
    main()
