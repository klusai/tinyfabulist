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
        (grammar, creativity, consistency), computes per-model averages, generates
        a Markdown table, and plots a grouped bar chart.
      - For files whose names start with 'tf_fables': generates a SINGLE consolidated report (Markdown table)
        of how many fables each model has across all files.
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

    # --- Process Evaluation Files (those with 'eval_e' in their filename) ---
    if eval_files:
        score_totals = defaultdict(lambda: {"grammar": 0, "creativity": 0, "consistency": 0, "count": 0})
        for file_path in eval_files:
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    model = data.get("model", "unknown")
                    if "evaluation" in data:
                        evaluation = data["evaluation"]
                        score_totals[model]["grammar"] += evaluation.get("grammar", 0)
                        score_totals[model]["creativity"] += evaluation.get("creativity", 0)
                        score_totals[model]["consistency"] += evaluation.get("consistency", 0)
                        score_totals[model]["count"] += 1
                    else:
                        score_totals[model]["count"] += 1

        # Compute averages per model.
        averages = {}
        for model, scores in score_totals.items():
            count = scores["count"]
            if count > 0:
                averages[model] = {
                    "grammar": scores["grammar"] / count,
                    "creativity": scores["creativity"] / count,
                    "consistency": scores["consistency"] / count,
                    "count": count
                }

        # Build Markdown table for evaluation averages.
        md_table_eval = "| Model | Grammar | Creativity | Consistency | Count |\n"
        md_table_eval += "|-------|---------|------------|-------------|-------|\n"
        for model, metrics in averages.items():
            md_table_eval += f"| {model} | {metrics['grammar']:.2f} | {metrics['creativity']:.2f} | {metrics['consistency']:.2f} | {metrics['count']} |\n"
        
        md_eval_filename = os.path.join(stats_folder, f"tf_stats_eval_table_{timestamp}.md")
        with open(md_eval_filename, "w") as f:
            f.write("## Evaluation Averages\n")
            f.write(md_table_eval)
        print(f"Evaluation Markdown table saved to {md_eval_filename}")

        # Plot grouped bar chart for evaluation scores.
        metrics_list = ["grammar", "creativity", "consistency"]
        models_list = list(averages.keys())
        x = np.arange(len(models_list))
        width = 0.2

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = ['#1DB954', '#191414', '#535353']

        for i, metric in enumerate(metrics_list):
            values = [averages[model][metric] for model in models_list]
            offset = (i - 1) * width
            bars = ax.bar(x + offset, values, width, label=metric.capitalize(), color=colors[i])
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom',
                            fontsize=10, fontweight='bold')
        ax.set_ylabel("Average Score", fontsize=12)
        ax.set_title("Average Evaluation Scores by Model", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models_list, fontsize=12)
        ax.set_ylim(0, 10)
        ax.legend(fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plot_filename = os.path.join(stats_folder, f"tf_stats_eval_plot_{timestamp}.png")
        plt.savefig(plot_filename)
        print(f"Evaluation plot saved to {plot_filename}")
        plt.show()
    else:
        print("No evaluation files (with 'eval_e' in filename) found.")

    # --- Process tf_fable Files (those whose name starts with 'tf_fables') ---
    if tf_files:
        # Create a single consolidated report for all tf_fable files
        model_counts = defaultdict(int)
        file_counts = defaultdict(int)  # Track counts per file
        
        for file_path in tf_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            file_model_counts = defaultdict(int)
            
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    model = data.get("model", "unknown")
                    model_counts[model] += 1
                    file_model_counts[model] += 1
            
            # Store counts for this file
            file_counts[base_name] = file_model_counts
        
        # Generate a single consolidated markdown table
        md_table_tf = "# Consolidated Fable Counts\n\n"
        
        # First, create a summary table with totals across all files
        md_table_tf += "## Summary (All Files)\n"
        md_table_tf += "| Model | Total Fable Count |\n"
        md_table_tf += "|-------|----------------|\n"
        for model, count in sorted(model_counts.items()):
            md_table_tf += f"| {model} | {count} |\n"
        
        # Then, create breakdowns by file
        md_table_tf += "\n## Breakdown by File\n"
        for file_name, file_model_counts in file_counts.items():
            md_table_tf += f"\n### {file_name}\n"
            md_table_tf += "| Model | Fable Count |\n"
            md_table_tf += "|-------|-------------|\n"
            for model, count in sorted(file_model_counts.items()):
                md_table_tf += f"| {model} | {count} |\n"
        
        # Write the consolidated report
        md_tf_filename = os.path.join(stats_folder, f"tf_stats_fables_{timestamp}.md")
        with open(md_tf_filename, "w") as f:
            f.write(md_table_tf)
        print(f"Consolidated tf_fable report saved to {md_tf_filename}")
    else:
        print("No tf_fable files found.")

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