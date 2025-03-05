import os
import time
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def plot_model_averages(args):
    """
    Reads one or more JSONL files from a given file or directory, then:
      - For files with 'eval_e' in their filename: aggregates evaluation scores 
        (grammar, creativity, moral_clarity, adherence_to_prompt) as well as performance 
        metrics (input tokens, output tokens, inference time). It computes per-model averages, 
        generates a Markdown table, and plots a grouped bar chart.
      - For files whose names start with 'tf_fables': generates a SINGLE consolidated report (Markdown table)
        of how many fables each model has across all files along with performance metrics.
    The Markdown files and plot image are saved to the folder data/stats/.
    
    Parameters:
        args: An object with attribute 'jsonl' specifying the path to a JSONL file 
              or a folder to scan recursively.
    """
    # Separate files into evaluation files and tf_fable files.
    eval_files = []
    tf_files = []
    input_path = args.jsonl

    if os.path.isfile(input_path):
        base_name = os.path.basename(input_path)
        if "eval_e" in base_name:
            eval_files.append(input_path)
        elif base_name.startswith("tf_fables"):
            tf_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_path = os.path.join(root, file)
                    base_name = os.path.basename(file_path)
                    if "eval_e" in base_name:
                        eval_files.append(file_path)
                    elif base_name.startswith("tf_fables"):
                        tf_files.append(file_path)
    else:
        print("Provided path is neither a file nor a directory.")
        return

    stats_folder = "data/stats/"
    os.makedirs(stats_folder, exist_ok=True)
    timestamp = time.strftime("%y%m%d-%H%M%S")

    # --- Process tf_fable Files first (so performance metrics are available) ---
    tf_model_stats = None
    if tf_files:
        tf_model_stats = defaultdict(lambda: {
            "count": 0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_inference_time": 0,
            "host_providers": defaultdict(int),
            "host_gpus": defaultdict(int),
            "host_gpu_vram": defaultdict(int),
            "host_dc_providers": defaultdict(int),
            "host_dc_locations": defaultdict(int),
            "host_cost_per_hour": 0.0
        })
        for file_path in tf_files:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    model = data.get("llm_name", "unknown").split("/")[-1]
                    stats = tf_model_stats[model]
                    stats["count"] += 1
                    stats["avg_input_tokens"] += data.get("llm_input_tokens", 0)
                    stats["avg_output_tokens"] += data.get("llm_output_tokens", 0)
                    stats["avg_inference_time"] += data.get("llm_inference_time", 0)
                    stats["host_providers"][data.get("host_provider", "unknown")] += 1
                    stats["host_gpus"][data.get("host_gpu", "unknown")] += 1
                    stats["host_dc_providers"][data.get("host_dc_provider", "unknown")] += 1
                    stats["host_dc_locations"][data.get("host_dc_location", "unknown")] += 1
                    stats["host_cost_per_hour"] += data.get("host_cost_per_hour", 0.0)
                    stats["host_gpu_vram"][data.get("host_gpu_vram", 0)] += 1

        # Calculate averages for tf_model_stats.
        for model, stats in tf_model_stats.items():
            count = stats["count"]
            if count > 0:
                stats["avg_input_tokens"] /= count
                stats["avg_output_tokens"] /= count
                stats["avg_inference_time"] /= count
                stats["host_cost_per_hour"] /= count

        # Generate consolidated Markdown table for tf_fable files.
        md_table_tf = "# Consolidated Fable Statistics\n\n"
        md_table_tf += "## Summary (All Files)\n"
        md_table_tf += (
            "| Model | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) | "
            "Primary Host Provider | Primary GPU | GPU VRAM (GB) | DC Provider | DC Location | "
            "Avg Cost per Hour ($) |\n"
        )
        md_table_tf += (
            "|-------|-------|-----------------|------------------|---------------------|"
            "-------------------|-------------|---------------|-------------|-------------|------------------------|\n"
        )
        for model, stats in sorted(tf_model_stats.items()):
            primary_provider = max(stats["host_providers"].items(), key=lambda x: x[1])[0] if stats["host_providers"] else "unknown"
            primary_gpu = max(stats["host_gpus"].items(), key=lambda x: x[1])[0] if stats["host_gpus"] else "unknown"
            primary_gpu_vram = max(stats["host_gpu_vram"].items(), key=lambda x: x[1])[0] if stats["host_gpu_vram"] else "unknown"
            primary_dc_provider = max(stats["host_dc_providers"].items(), key=lambda x: x[1])[0] if stats["host_dc_providers"] else "unknown"
            primary_dc_location = max(stats["host_dc_locations"].items(), key=lambda x: x[1])[0] if stats["host_dc_locations"] else "unknown"
            avg_cost_per_hour = stats["host_cost_per_hour"]
            md_table_tf += f"| {model} | {stats['count']} | {stats['avg_input_tokens']:.1f} | {stats['avg_output_tokens']:.1f} | {stats['avg_inference_time']:.2f} | {primary_provider} | {primary_gpu} | {primary_gpu_vram} | {primary_dc_provider} | {primary_dc_location} | {avg_cost_per_hour:.2f} |\n"
        
        md_tf_filename = os.path.join(stats_folder, f"tf_stats_fables_{timestamp}.md")
        with open(md_tf_filename, "w") as f:
            f.write(md_table_tf)
        print(f"Consolidated tf_fable report saved to {md_tf_filename}")

    # --- Process Evaluation Files ---
    if eval_files:
        # Now also aggregate performance metrics (tokens and inference time)
        score_totals = defaultdict(lambda: {
            "grammar": 0,
            "creativity": 0,
            "moral_clarity": 0,
            "adherence_to_prompt": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "count": 0
        })
        for file_path in eval_files:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    model = data.get("llm_name", "unknown").split("/")[-1]
                    if "evaluation" in data:
                        evaluation = data["evaluation"]
                        score_totals[model]["grammar"] += evaluation.get("grammar", 0)
                        score_totals[model]["creativity"] += evaluation.get("creativity", 0)
                        score_totals[model]["moral_clarity"] += evaluation.get("moral_clarity", 0)
                        score_totals[model]["adherence_to_prompt"] += evaluation.get("adherence_to_prompt", 0)
                    # Aggregate performance fields if present.
                    score_totals[model]["input_tokens"] += data.get("llm_input_tokens", 0)
                    score_totals[model]["output_tokens"] += data.get("llm_output_tokens", 0)
                    score_totals[model]["inference_time"] += data.get("llm_inference_time", 0)
                    score_totals[model]["count"] += 1

        # Compute per-model averages.
        averages = {}
        for model, scores in score_totals.items():
            count = scores["count"]
            if count > 0:
                averages[model] = {
                    "grammar": scores["grammar"] / count,
                    "creativity": scores["creativity"] / count,
                    "moral_clarity": scores["moral_clarity"] / count,
                    "adherence_to_prompt": scores["adherence_to_prompt"] / count,
                    "average_score_(mean)": (scores["grammar"] + scores["creativity"] + scores["moral_clarity"] + scores["adherence_to_prompt"]) / (count * 4),
                    "input_tokens": scores["input_tokens"] / count,
                    "output_tokens": scores["output_tokens"] / count,
                    "inference_time": scores["inference_time"] / count,
                    "count": count
                }

        # Build Markdown table for evaluation averages including performance metrics.
        md_table_eval = (
            "| Model | Grammar | Creativity | Moral Clarity | Adherence to Prompt | Average Score (Mean) | Count | Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) |\n"
            "|-------|---------|------------|---------------|---------------------|-----------------|-------|-----------------|------------------|------------------------|\n"
        )
        for model, metrics in averages.items():
            md_table_eval += (
                f"| {model} | {metrics['grammar']:.2f} | {metrics['creativity']:.2f} | {metrics['moral_clarity']:.2f} "
                f"| {metrics['adherence_to_prompt']:.2f} | {metrics['average_score_(mean)']:.2f} | {metrics['count']} | "
                f"{metrics['input_tokens']:.1f} | {metrics['output_tokens']:.1f} | {metrics['inference_time']:.2f} |\n"
            )
        
        md_eval_filename = os.path.join(stats_folder, f"tf_stats_eval_table_{timestamp}.md")
        with open(md_eval_filename, "w") as f:
            f.write("## Evaluation Averages\n")
            f.write(md_table_eval)
        print(f"Evaluation Markdown table saved to {md_eval_filename}")

        # --- Plotting ---
        metrics_list = ["grammar", "creativity", "moral_clarity", "adherence_to_prompt", "average_score_(mean)"]
        models_list = list(averages.keys())

        # Create subplots: always one for evaluation scores.
        # Use tf_model_stats for performance metrics if available; otherwise, use evaluation performance data.
        if tf_model_stats:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Plot evaluation scores on ax1.
        x = np.arange(len(models_list))
        width = 0.17
        colors = ['#1DB954', '#191414', '#535353', '#B3B3B3', '#E60012']

        for i, metric in enumerate(metrics_list):
            values = [averages[model][metric] for model in models_list]
            offset = (i - 2) * width
            bars = ax1.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), color=colors[i % len(colors)])
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=9, fontweight='bold')
        ax1.set_title("Average Evaluation Scores by Model", fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_list, fontsize=12)
        ax1.set_ylim(0, 10)
        ax1.legend(fontsize=12)
        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        # Plot performance metrics on ax2.
        # Prefer tf_model_stats if available; otherwise, use evaluation aggregates.
        if tf_model_stats:
            performance_metrics = ["avg_input_tokens", "avg_output_tokens", "avg_inference_time"]
            source = tf_model_stats
        else:
            performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
            source = averages

        width_perf = 0.25
        colors_perf = ['#4CAF50', '#2196F3', '#FFC107']
        for i, metric in enumerate(performance_metrics):
            # For each model, if the metric is missing, default to 0.
            values = [source[model][metric] if model in source else 0 for model in models_list]
            offset = (i - 1) * width_perf
            bars = ax2.bar(x + offset, values, width_perf, label=metric.replace('avg_', '').replace('_', ' ').title(), color=colors_perf[i % len(colors_perf)])
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3),
                             textcoords="offset points",
                             ha='center', va='bottom',
                             fontsize=9, fontweight='bold')
        ax2.set_title("Performance Metrics by Model", fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models_list, fontsize=12)
        ax2.legend(fontsize=12)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plot_filename = os.path.join(stats_folder, f"tf_stats_eval_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Evaluation plot saved to {plot_filename}")
        plt.show()
    else:
        print("No evaluation files (with 'eval_e' in filename) found.")

def add_stats_subparser(subparsers) -> None:
    generate_parser = subparsers.add_parser(
        'stats', 
        help='Compute aggregated statistics from evaluation JSONL files (file or directory).'
    )
    generate_parser.add_argument(
        '--jsonl', 
        default="evaluate.jsonl", 
        help='Path to a JSONL file or a folder containing JSONL files'
    )
    generate_parser.set_defaults(func=plot_model_averages)
