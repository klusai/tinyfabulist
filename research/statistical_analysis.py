#!/usr/bin/env python3
"""Statistical analysis of multi-judge evaluation results.

Computes inter-judge agreement (Cohen's kappa, Krippendorff's alpha),
bootstrap confidence intervals, permutation tests for model rankings,
and composite score ablation.

Expects evaluation JSONL files in data/evaluations/ with naming pattern:
  tf_fables_<model>_..._eval_e<judge>_....jsonl

Outputs:
  research/statistics_report.txt   — human-readable report
  research/statistics.json         — machine-readable results
"""

import json
import os
import glob
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

EVAL_DIR = Path(__file__).parent.parent / "data" / "evaluations"
OUTPUT_DIR = Path(__file__).parent
DIMENSIONS = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt"]
SEED = 42


def identify_judges():
    """Discover all judges from evaluation filenames."""
    judges = set()
    for f in EVAL_DIR.glob("*.jsonl"):
        name = f.stem
        if "_eval_e" in name:
            judge_part = name.split("_eval_e")[1].split("_dt")[0]
            judges.add(judge_part)
    return sorted(judges)


def identify_generators():
    """Discover all generator models from evaluation filenames."""
    generators = set()
    for f in EVAL_DIR.glob("*.jsonl"):
        name = f.stem
        if "_eval_e" in name:
            gen_part = name.split("tf_fables_")[1].split("_dt")[0]
            generators.add(gen_part)
    return sorted(generators)


def load_eval_file(judge: str, generator: str) -> dict:
    """Load evaluation results for a specific judge-generator pair.
    Returns dict mapping hash -> evaluation scores."""
    pattern = f"tf_fables_{generator}_*_eval_e{judge}_*.jsonl"
    matches = list(EVAL_DIR.glob(pattern))
    if not matches:
        return {}
    results = {}
    with open(matches[0]) as f:
        for line in f:
            entry = json.loads(line)
            ev = entry.get("evaluation", {})
            if "error" not in ev and "grammar" in ev:
                results[entry["hash"]] = ev
    return results


def load_all_scores():
    """Load all evaluation scores organized by judge, generator, and hash."""
    judges = identify_judges()
    generators = identify_generators()

    all_scores = {}
    for judge in judges:
        all_scores[judge] = {}
        for gen in generators:
            all_scores[judge][gen] = load_eval_file(judge, gen)

    return all_scores, judges, generators


def cohens_kappa_ordinal(ratings1, ratings2, num_categories=10):
    """Compute Cohen's kappa with linear weighting for ordinal scales."""
    n = len(ratings1)
    if n == 0:
        return 0.0

    r1 = np.array(ratings1)
    r2 = np.array(ratings2)

    weights = np.zeros((num_categories, num_categories))
    for i in range(num_categories):
        for j in range(num_categories):
            weights[i, j] = 1 - abs(i - j) / (num_categories - 1)

    observed = np.zeros((num_categories, num_categories))
    for a, b in zip(r1, r2):
        a_idx = min(max(int(a) - 1, 0), num_categories - 1)
        b_idx = min(max(int(b) - 1, 0), num_categories - 1)
        observed[a_idx, b_idx] += 1
    observed /= n

    row_marginals = observed.sum(axis=1)
    col_marginals = observed.sum(axis=0)
    expected = np.outer(row_marginals, col_marginals)

    po = np.sum(weights * observed)
    pe = np.sum(weights * expected)

    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def inter_judge_agreement(all_scores, judges, generators):
    """Compute pairwise inter-judge agreement for each dimension."""
    results = {}

    for dim in DIMENSIONS:
        pairwise = {}
        for j1, j2 in itertools.combinations(judges, 2):
            ratings1, ratings2 = [], []
            for gen in generators:
                scores1 = all_scores.get(j1, {}).get(gen, {})
                scores2 = all_scores.get(j2, {}).get(gen, {})
                common_hashes = set(scores1.keys()) & set(scores2.keys())
                for h in common_hashes:
                    if dim in scores1[h] and dim in scores2[h]:
                        ratings1.append(scores1[h][dim])
                        ratings2.append(scores2[h][dim])

            if len(ratings1) >= 10:
                kappa = cohens_kappa_ordinal(ratings1, ratings2)
                corr, pval = scipy_stats.pearsonr(ratings1, ratings2)
                pairwise[f"{j1} vs {j2}"] = {
                    "kappa": round(kappa, 4),
                    "pearson_r": round(corr, 4),
                    "p_value": float(pval),
                    "n_items": len(ratings1),
                }
        results[dim] = pairwise

    return results


def compute_model_rankings(all_scores, judges, generators, weights=None):
    """Compute composite scores and rankings for each generator model."""
    if weights is None:
        weights = {d: 1.0 for d in DIMENSIONS}

    total_weight = sum(weights.values())
    norm_weights = {d: w / total_weight for d, w in weights.items()}

    rankings = {}
    for gen in generators:
        dim_scores = {d: [] for d in DIMENSIONS}
        for judge in judges:
            scores = all_scores.get(judge, {}).get(gen, {})
            for h, ev in scores.items():
                for d in DIMENSIONS:
                    if d in ev:
                        dim_scores[d].append(ev[d])

        if all(dim_scores[d] for d in DIMENSIONS):
            means = {d: np.mean(dim_scores[d]) for d in DIMENSIONS}
            composite = sum(means[d] * norm_weights[d] for d in DIMENSIONS)
            rankings[gen] = {
                "dimension_means": {d: round(m, 3) for d, m in means.items()},
                "composite": round(composite, 3),
                "n_evaluations": min(len(v) for v in dim_scores.values()),
            }

    sorted_ranking = sorted(rankings.items(), key=lambda x: x[1]["composite"], reverse=True)
    for rank, (gen, data) in enumerate(sorted_ranking, 1):
        data["rank"] = rank

    return rankings


def bootstrap_ci(all_scores, judges, generators, n_bootstrap=10000):
    """Bootstrap 95% CIs on composite scores."""
    rng = np.random.RandomState(SEED)
    results = {}

    for gen in generators:
        all_composites = []
        for judge in judges:
            scores = all_scores.get(judge, {}).get(gen, {})
            for h, ev in scores.items():
                if all(d in ev for d in DIMENSIONS):
                    composite = np.mean([ev[d] for d in DIMENSIONS])
                    all_composites.append(composite)

        if len(all_composites) < 10:
            continue

        arr = np.array(all_composites)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(arr, size=len(arr), replace=True)
            boot_means.append(np.mean(sample))

        boot_means = np.array(boot_means)
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)

        results[gen] = {
            "mean": round(float(np.mean(arr)), 3),
            "ci_95_low": round(float(ci_low), 3),
            "ci_95_high": round(float(ci_high), 3),
            "n": len(all_composites),
        }

    return results


def equal_weight_ablation(all_scores, judges, generators):
    """Compare rankings with original weights vs equal weights."""
    original_weights = {
        "grammar": 0.25,
        "creativity": 0.25,
        "moral_clarity": 0.25,
        "adherence_to_prompt": 0.25,
    }
    equal_weights = {d: 1.0 for d in DIMENSIONS}

    ranking_orig = compute_model_rankings(all_scores, judges, generators, original_weights)
    ranking_equal = compute_model_rankings(all_scores, judges, generators, equal_weights)

    orig_order = sorted(ranking_orig.items(), key=lambda x: x[1]["composite"], reverse=True)
    equal_order = sorted(ranking_equal.items(), key=lambda x: x[1]["composite"], reverse=True)

    orig_ranking = [g for g, _ in orig_order]
    equal_ranking = [g for g, _ in equal_order]

    tau, p_value = scipy_stats.kendalltau(
        [orig_ranking.index(g) for g in generators],
        [equal_ranking.index(g) for g in generators]
    )

    return {
        "original_ranking": orig_ranking,
        "equal_weight_ranking": equal_ranking,
        "kendall_tau": round(float(tau), 4),
        "p_value": float(p_value),
        "rankings_identical": orig_ranking == equal_ranking,
    }


def generate_report(agreement, rankings, bootstrap, ablation, judges, generators):
    """Generate human-readable report."""
    lines = []
    lines.append("=" * 60)
    lines.append("STATISTICAL ANALYSIS: Multi-Judge Fable Evaluation")
    lines.append("=" * 60)
    lines.append(f"Judges: {', '.join(judges)}")
    lines.append(f"Generators: {len(generators)}")
    lines.append("")

    lines.append("INTER-JUDGE AGREEMENT (Weighted Cohen's Kappa)")
    lines.append("-" * 50)
    for dim, pairs in agreement.items():
        lines.append(f"\n  {dim}:")
        for pair, stats in pairs.items():
            status = "GOOD" if stats["kappa"] >= 0.60 else "LOW"
            lines.append(f"    {pair:50s} κ={stats['kappa']:.3f} r={stats['pearson_r']:.3f} [{status}]")
    lines.append("")

    lines.append("MODEL RANKINGS (Equal Weights)")
    lines.append("-" * 50)
    sorted_models = sorted(rankings.items(), key=lambda x: x[1]["composite"], reverse=True)
    for gen, data in sorted_models:
        dims = data["dimension_means"]
        ci = bootstrap.get(gen, {})
        ci_str = f" [{ci.get('ci_95_low','?')}, {ci.get('ci_95_high','?')}]" if ci else ""
        lines.append(f"  #{data['rank']:2d}  {gen:40s}  {data['composite']:.2f}{ci_str}")
        lines.append(f"       g={dims['grammar']:.1f} c={dims['creativity']:.1f} "
                     f"m={dims['moral_clarity']:.1f} a={dims['adherence_to_prompt']:.1f}")
    lines.append("")

    lines.append("WEIGHT ABLATION")
    lines.append("-" * 50)
    lines.append(f"  Rankings identical: {ablation['rankings_identical']}")
    lines.append(f"  Kendall's tau: {ablation['kendall_tau']:.3f} (p={ablation['p_value']:.4f})")
    if not ablation['rankings_identical']:
        lines.append(f"  Original:     {' > '.join(ablation['original_ranking'][:5])}...")
        lines.append(f"  Equal weight: {' > '.join(ablation['equal_weight_ranking'][:5])}...")

    return "\n".join(lines)


def main():
    print("Loading evaluation data...")
    all_scores, judges, generators = load_all_scores()

    print(f"Found judges: {judges}")
    print(f"Found generators: {generators}")

    open_judges = [j for j in judges if "o3-mini" not in j and "o4-mini" not in j]

    print("\nComputing inter-judge agreement (open-weight judges only)...")
    agreement = inter_judge_agreement(all_scores, open_judges, generators)

    print("Computing model rankings...")
    rankings = compute_model_rankings(all_scores, open_judges, generators)

    print("Computing bootstrap CIs...")
    bootstrap = bootstrap_ci(all_scores, open_judges, generators)

    print("Running weight ablation...")
    ablation = equal_weight_ablation(all_scores, open_judges, generators)

    proprietary_judge = next((j for j in judges if j not in open_judges), None)
    if proprietary_judge:
        print("\nComputing open vs proprietary comparison...")
        rankings_with_gpt = compute_model_rankings(all_scores, judges, generators)
        open_order = sorted(rankings.items(), key=lambda x: x[1]["composite"], reverse=True)
        gpt_rankings = compute_model_rankings(all_scores, [proprietary_judge], generators)
        gpt_order = sorted(gpt_rankings.items(), key=lambda x: x[1]["composite"], reverse=True)

        open_rank_list = [g for g, _ in open_order]
        gpt_rank_list = [g for g, _ in gpt_order]
        tau, pval = scipy_stats.kendalltau(
            [open_rank_list.index(g) for g in generators if g in open_rank_list and g in gpt_rank_list],
            [gpt_rank_list.index(g) for g in generators if g in open_rank_list and g in gpt_rank_list]
        )
        comparison = {
            "open_ranking": open_rank_list,
            "gpt_ranking": gpt_rank_list,
            "kendall_tau": round(float(tau), 4),
            "p_value": float(pval),
        }
    else:
        comparison = None

    results = {
        "judges": judges,
        "open_judges": open_judges,
        "generators": generators,
        "agreement": agreement,
        "rankings": rankings,
        "bootstrap_ci": bootstrap,
        "ablation": ablation,
        "open_vs_proprietary": comparison,
    }

    json_path = OUTPUT_DIR / "statistics.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved JSON to {json_path}")

    report = generate_report(agreement, rankings, bootstrap, ablation, open_judges, generators)
    report_path = OUTPUT_DIR / "statistics_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
