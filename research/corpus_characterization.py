#!/usr/bin/env python3
"""Corpus characterization for the TinyFabulist 3M fable dataset.

Computes vocabulary stats, length distributions, readability, lexical diversity,
and near-duplicate detection on a stratified sample from HuggingFace.

Outputs:
  research/corpus_stats.json   — machine-readable summary
  research/corpus_report.txt   — human-readable report for paper writing
"""

import json
import os
import sys
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Optional heavy imports guarded for clarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import ngrams
import textstat

SAMPLE_SIZE = 10_000
SEED = 42
HF_DATASET = "klusai/ds-tf1-en-3m"
OUTPUT_DIR = Path(__file__).parent


def ensure_nltk_data():
    for resource in ["punkt", "punkt_tab", "averaged_perceptron_tagger"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def load_sample(n=SAMPLE_SIZE):
    """Load a random sample from the HF dataset."""
    from datasets import load_dataset

    print(f"Loading dataset {HF_DATASET} (streaming)...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    reservoir = []
    rng = random.Random(SEED)
    for i, row in enumerate(ds):
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
        if (i + 1) % 100_000 == 0:
            print(f"  Scanned {i+1:,} rows...")
    print(f"  Total rows scanned: {i+1:,}")
    print(f"  Sample size: {len(reservoir)}")
    return reservoir


def length_stats(fables):
    """Word count and sentence count distributions."""
    word_counts = []
    sent_counts = []
    char_counts = []
    for f in fables:
        text = f["fable"]
        words = word_tokenize(text)
        sents = sent_tokenize(text)
        word_counts.append(len(words))
        sent_counts.append(len(sents))
        char_counts.append(len(text))

    return {
        "word_count": _describe(word_counts),
        "sentence_count": _describe(sent_counts),
        "character_count": _describe(char_counts),
    }


def _describe(arr):
    a = np.array(arr, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
        "n": len(arr),
    }


def vocabulary_stats(fables):
    """Vocabulary size, type-token ratio, hapax legomena."""
    all_tokens = []
    for f in fables:
        tokens = word_tokenize(f["fable"].lower())
        all_tokens.extend(tokens)

    freq = Counter(all_tokens)
    total_tokens = len(all_tokens)
    vocab_size = len(freq)
    hapax = sum(1 for c in freq.values() if c == 1)
    ttr = vocab_size / total_tokens if total_tokens else 0

    top_50 = freq.most_common(50)
    top_50_no_punct = [(w, c) for w, c in freq.most_common(200)
                       if re.match(r'^[a-z]', w)][:50]

    return {
        "total_tokens": total_tokens,
        "vocabulary_size": vocab_size,
        "type_token_ratio": round(ttr, 6),
        "hapax_legomena": hapax,
        "hapax_ratio": round(hapax / vocab_size, 4) if vocab_size else 0,
        "top_50_words": top_50_no_punct,
    }


def lexical_diversity(fables, sample_n=500):
    """Distinct-n and MATTR (Moving Average TTR) on a sub-sample."""
    rng = random.Random(SEED + 1)
    sub = rng.sample(fables, min(sample_n, len(fables)))

    distinct_1_vals = []
    distinct_2_vals = []
    distinct_3_vals = []

    for f in sub:
        tokens = word_tokenize(f["fable"].lower())
        if len(tokens) < 3:
            continue
        for n_val, arr in [(1, distinct_1_vals), (2, distinct_2_vals), (3, distinct_3_vals)]:
            ng = list(ngrams(tokens, n_val))
            if ng:
                arr.append(len(set(ng)) / len(ng))

    return {
        "distinct_1": _describe(distinct_1_vals),
        "distinct_2": _describe(distinct_2_vals),
        "distinct_3": _describe(distinct_3_vals),
    }


def readability_stats(fables):
    """Flesch Reading Ease and Flesch-Kincaid Grade Level."""
    fre_scores = []
    fkgl_scores = []

    for f in fables:
        text = f["fable"]
        if len(text.split()) < 10:
            continue
        fre = textstat.flesch_reading_ease(text)
        fkgl = textstat.flesch_kincaid_grade(text)
        fre_scores.append(fre)
        fkgl_scores.append(fkgl)

    return {
        "flesch_reading_ease": _describe(fre_scores),
        "flesch_kincaid_grade": _describe(fkgl_scores),
    }


def near_duplicate_detection(fables, shingle_size=5, threshold=0.8, check_n=2000):
    """Jaccard-based near-duplicate detection on a sub-sample."""
    rng = random.Random(SEED + 2)
    sub = rng.sample(fables, min(check_n, len(fables)))

    def shingle_set(text):
        tokens = text.lower().split()
        return set(ngrams(tokens, shingle_size)) if len(tokens) >= shingle_size else set()

    shingle_sets = [shingle_set(f["fable"]) for f in sub]

    pair_count = 0
    near_dup_count = 0
    jaccard_scores = []

    for i in range(len(shingle_sets)):
        for j in range(i + 1, min(i + 50, len(shingle_sets))):
            if not shingle_sets[i] or not shingle_sets[j]:
                continue
            pair_count += 1
            intersection = len(shingle_sets[i] & shingle_sets[j])
            union = len(shingle_sets[i] | shingle_sets[j])
            jaccard = intersection / union if union else 0
            jaccard_scores.append(jaccard)
            if jaccard >= threshold:
                near_dup_count += 1

    return {
        "pairs_checked": pair_count,
        "near_duplicates": near_dup_count,
        "near_duplicate_rate": round(near_dup_count / pair_count, 6) if pair_count else 0,
        "jaccard_distribution": _describe(jaccard_scores) if jaccard_scores else {},
        "threshold": threshold,
        "shingle_size": shingle_size,
    }


def model_distribution(fables):
    """Distribution of fables across generator models."""
    model_counts = Counter()
    for f in fables:
        model = f.get("llm_name", "unknown").split("/")[-1]
        model_counts[model] += 1
    return dict(model_counts.most_common())


def age_group_distribution(fables):
    """Distribution across prompted age groups (extracted from prompt text)."""
    age_pattern = re.compile(r"children ages? (\d+)-(\d+)", re.IGNORECASE)
    age_counts = Counter()
    for f in fables:
        prompt = f.get("prompt", "")
        match = age_pattern.search(prompt)
        if match:
            age_counts[f"{match.group(1)}-{match.group(2)}"] += 1
        else:
            age_counts["unknown"] += 1
    return dict(age_counts.most_common())


def generate_report(results):
    """Generate a human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CORPUS CHARACTERIZATION: TinyFabulist 3M Fable Dataset")
    lines.append("=" * 60)
    lines.append(f"Sample size: {results['sample_size']:,} fables (reservoir-sampled)")
    lines.append(f"Total dataset rows scanned: {results.get('total_scanned', 'N/A')}")
    lines.append("")

    ls = results["length_stats"]
    lines.append("LENGTH STATISTICS")
    lines.append("-" * 40)
    wc = ls["word_count"]
    lines.append(f"  Word count:  {wc['mean']:.0f} ± {wc['std']:.0f} "
                 f"(median {wc['median']:.0f}, range {wc['min']}–{wc['max']})")
    sc = ls["sentence_count"]
    lines.append(f"  Sentences:   {sc['mean']:.1f} ± {sc['std']:.1f} "
                 f"(median {sc['median']:.0f})")
    lines.append("")

    vs = results["vocabulary_stats"]
    lines.append("VOCABULARY")
    lines.append("-" * 40)
    lines.append(f"  Total tokens:     {vs['total_tokens']:,}")
    lines.append(f"  Vocabulary size:  {vs['vocabulary_size']:,}")
    lines.append(f"  Type-Token Ratio: {vs['type_token_ratio']:.4f}")
    lines.append(f"  Hapax legomena:   {vs['hapax_legomena']:,} "
                 f"({vs['hapax_ratio']:.1%} of vocab)")
    lines.append(f"  Top 20 words: {', '.join(w for w, _ in vs['top_50_words'][:20])}")
    lines.append("")

    ld = results["lexical_diversity"]
    lines.append("LEXICAL DIVERSITY (per-fable)")
    lines.append("-" * 40)
    for key in ["distinct_1", "distinct_2", "distinct_3"]:
        d = ld[key]
        lines.append(f"  {key}: {d['mean']:.3f} ± {d['std']:.3f}")
    lines.append("")

    rs = results["readability"]
    lines.append("READABILITY")
    lines.append("-" * 40)
    fre = rs["flesch_reading_ease"]
    fkgl = rs["flesch_kincaid_grade"]
    lines.append(f"  Flesch Reading Ease:   {fre['mean']:.1f} ± {fre['std']:.1f}")
    lines.append(f"  Flesch-Kincaid Grade:  {fkgl['mean']:.1f} ± {fkgl['std']:.1f}")
    lines.append("")

    nd = results["near_duplicates"]
    lines.append("NEAR-DUPLICATE DETECTION")
    lines.append("-" * 40)
    lines.append(f"  Pairs checked:      {nd['pairs_checked']:,}")
    lines.append(f"  Near-duplicates:    {nd['near_duplicates']} "
                 f"({nd['near_duplicate_rate']:.4%})")
    if nd.get("jaccard_distribution"):
        jd = nd["jaccard_distribution"]
        lines.append(f"  Mean Jaccard:       {jd['mean']:.4f}")
        lines.append(f"  Max Jaccard:        {jd['max']:.4f}" if 'max' in jd else "")
    lines.append("")

    lines.append("MODEL DISTRIBUTION (in sample)")
    lines.append("-" * 40)
    for model, count in results["model_distribution"].items():
        pct = count / results["sample_size"] * 100
        lines.append(f"  {model:40s} {count:5d} ({pct:.1f}%)")
    lines.append("")

    lines.append("AGE GROUP DISTRIBUTION (in sample)")
    lines.append("-" * 40)
    for age, count in results["age_distribution"].items():
        pct = count / results["sample_size"] * 100
        lines.append(f"  {age:15s} {count:5d} ({pct:.1f}%)")

    return "\n".join(lines)


def main():
    ensure_nltk_data()

    fables = load_sample(SAMPLE_SIZE)

    print("\nComputing length statistics...")
    lstats = length_stats(fables)

    print("Computing vocabulary statistics...")
    vstats = vocabulary_stats(fables)

    print("Computing lexical diversity...")
    ldiv = lexical_diversity(fables)

    print("Computing readability...")
    rstats = readability_stats(fables)

    print("Computing near-duplicate detection...")
    ndups = near_duplicate_detection(fables)

    print("Computing distributions...")
    mdist = model_distribution(fables)
    adist = age_group_distribution(fables)

    results = {
        "sample_size": len(fables),
        "length_stats": lstats,
        "vocabulary_stats": vstats,
        "lexical_diversity": ldiv,
        "readability": rstats,
        "near_duplicates": ndups,
        "model_distribution": mdist,
        "age_distribution": adist,
    }

    json_path = OUTPUT_DIR / "corpus_stats.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON stats to {json_path}")

    report = generate_report(results)
    report_path = OUTPUT_DIR / "corpus_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
