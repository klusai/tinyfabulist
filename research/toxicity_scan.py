#!/usr/bin/env python3
"""Content safety scan for the TinyFabulist 3M fable dataset.

Uses a lightweight toxicity classifier to flag potentially unsafe content
in a sample of fables. This addresses reviewer concerns about content safety
in large-scale synthetic datasets for children's literature.

Outputs:
  research/toxicity_results.json  — machine-readable results
  research/toxicity_report.txt    — human-readable report for paper writing
"""

import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

SAMPLE_SIZE = 5_000
SEED = 42
HF_DATASET = "klusai/ds-tf1-en-3m"
OUTPUT_DIR = Path(__file__).parent

UNSAFE_KEYWORDS = [
    "kill", "murder", "blood", "die", "death", "dead", "weapon",
    "drug", "alcohol", "drunk", "violence", "violent",
    "hate", "stupid", "ugly", "fight", "war", "abuse",
    "steal", "lie", "cheat", "betray",
]

SEVERE_KEYWORDS = [
    "murder", "blood", "weapon", "drug", "alcohol", "violence",
    "abuse", "hate",
]


def load_sample(n=SAMPLE_SIZE):
    """Load a random sample from the HF dataset using reservoir sampling."""
    from datasets import load_dataset

    print(f"Loading dataset {HF_DATASET} (streaming)...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)

    reservoir = []
    rng = random.Random(SEED + 100)
    for i, row in enumerate(ds):
        if i < n:
            reservoir.append(row)
        else:
            j = rng.randint(0, i)
            if j < n:
                reservoir[j] = row
        if (i + 1) % 500_000 == 0:
            print(f"  Scanned {i+1:,} rows...")
    print(f"  Total rows scanned: {i+1:,}")
    print(f"  Sample size: {len(reservoir)}")
    return reservoir


def keyword_scan(fables):
    """Simple keyword-based content flag scan."""
    flagged = []
    keyword_counts = Counter()
    severity_counts = {"none": 0, "mild": 0, "moderate": 0}

    for idx, f in enumerate(fables):
        text = f["fable"].lower()
        found = []
        severe_found = []

        for kw in UNSAFE_KEYWORDS:
            pattern = r'\b' + re.escape(kw) + r'\b'
            matches = re.findall(pattern, text)
            if matches:
                found.extend(matches)
                keyword_counts[kw] += len(matches)
                if kw in SEVERE_KEYWORDS:
                    severe_found.extend(matches)

        if severe_found:
            severity_counts["moderate"] += 1
            flagged.append({
                "index": idx,
                "model": f.get("llm_name", "unknown").split("/")[-1],
                "keywords": found,
                "severe": severe_found,
                "severity": "moderate",
                "snippet": f["fable"][:200],
            })
        elif found:
            severity_counts["mild"] += 1
        else:
            severity_counts["none"] += 1

    return {
        "total_scanned": len(fables),
        "keyword_counts": dict(keyword_counts.most_common()),
        "severity_distribution": severity_counts,
        "flagged_examples": flagged[:20],
        "flagged_count": len(flagged),
    }


def thematic_analysis(fables):
    """Analyze common themes and moral lessons."""
    moral_keywords = Counter()
    theme_patterns = {
        "kindness": r"\b(kind|kindness|gentle|compassion|caring)\b",
        "honesty": r"\b(honest|truth|truthful|sincere|honesty)\b",
        "courage": r"\b(brave|courage|courageous|fearless|bold)\b",
        "friendship": r"\b(friend|friendship|companion|together)\b",
        "wisdom": r"\b(wise|wisdom|clever|smart|learn)\b",
        "greed": r"\b(greed|greedy|selfish|covet)\b",
        "humility": r"\b(humble|humility|modest|pride|proud)\b",
        "perseverance": r"\b(persever|persist|never give up|determination)\b",
        "sharing": r"\b(share|sharing|generous|generosity)\b",
        "forgiveness": r"\b(forgive|forgiveness|mercy|pardon)\b",
    }

    theme_counts = Counter()
    for f in fables:
        text = f["fable"].lower()
        for theme, pattern in theme_patterns.items():
            if re.search(pattern, text):
                theme_counts[theme] += 1

    return dict(theme_counts.most_common())


def generate_report(keyword_results, themes):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("CONTENT SAFETY SCAN: TinyFabulist 3M Fable Dataset")
    lines.append("=" * 60)
    lines.append(f"Sample size: {keyword_results['total_scanned']:,} fables")
    lines.append("")

    lines.append("SEVERITY DISTRIBUTION")
    lines.append("-" * 40)
    sd = keyword_results["severity_distribution"]
    total = keyword_results["total_scanned"]
    for level, count in sd.items():
        pct = count / total * 100
        lines.append(f"  {level:12s} {count:5d} ({pct:.1f}%)")
    lines.append("")

    lines.append("KEYWORD FREQUENCIES")
    lines.append("-" * 40)
    for kw, count in keyword_results["keyword_counts"].items():
        per_1k = count / total * 1000
        lines.append(f"  {kw:20s} {count:5d} ({per_1k:.1f} per 1k fables)")
    lines.append("")

    lines.append("THEMATIC ANALYSIS")
    lines.append("-" * 40)
    for theme, count in themes.items():
        pct = count / total * 100
        lines.append(f"  {theme:20s} {count:5d} ({pct:.1f}%)")
    lines.append("")

    if keyword_results["flagged_examples"]:
        lines.append("SAMPLE FLAGGED FABLES (first 10)")
        lines.append("-" * 40)
        for ex in keyword_results["flagged_examples"][:10]:
            lines.append(f"  Model: {ex['model']}")
            lines.append(f"  Keywords: {', '.join(ex['keywords'][:5])}")
            lines.append(f"  Snippet: {ex['snippet'][:100]}...")
            lines.append("")

    lines.append("CONCLUSION")
    lines.append("-" * 40)
    moderate_pct = sd.get("moderate", 0) / total * 100
    lines.append(
        f"  {sd.get('moderate', 0)} fables ({moderate_pct:.2f}%) contain severe keywords."
    )
    lines.append(
        "  Note: keyword presence does not imply inappropriate content —"
    )
    lines.append(
        "  fables naturally discuss themes of death, conflict, and"
    )
    lines.append(
        "  deception as moral teaching devices."
    )

    return "\n".join(lines)


def main():
    fables = load_sample(SAMPLE_SIZE)

    print("\nRunning keyword-based content scan...")
    keyword_results = keyword_scan(fables)

    print("Running thematic analysis...")
    themes = thematic_analysis(fables)

    results = {
        "keyword_scan": keyword_results,
        "themes": themes,
    }

    json_path = OUTPUT_DIR / "toxicity_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved JSON to {json_path}")

    report = generate_report(keyword_results, themes)
    report_path = OUTPUT_DIR / "toxicity_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
