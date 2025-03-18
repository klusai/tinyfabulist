import argparse
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- FILE PROCESSING FUNCTIONS ---


def identify_files(input_path):
    """
    Identify evaluation and fable files from the provided path.

    Parameters:
        input_path: Path to a file or directory

    Returns:
        tuple: (eval_files, tf_files) - Lists of evaluation and tf_fables files
    """
    eval_files = []
    tf_files = []

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

    return eval_files, tf_files


# --- FABLES DATA PROCESSING ---


def parse_fables_data(tf_files):
    """
    Parse data from tf_fables files.

    Parameters:
        tf_files: List of tf_fables files

    Returns:
        dict: Statistics for each model
    """
    tf_model_stats = defaultdict(
        lambda: {
            "count": 0,
            "avg_input_tokens": 0,
            "avg_output_tokens": 0,
            "avg_inference_time": 0,
            "host_providers": defaultdict(int),
            "host_gpus": defaultdict(int),
            "host_gpu_vram": defaultdict(int),
            "host_dc_providers": defaultdict(int),
            "host_dc_locations": defaultdict(int),
            "host_cost_per_hour": 0.0,
        }
    )

    for file_path in tf_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                model = data.get("llm_name", "unknown").split("/")[-1]
                stats = tf_model_stats[model]

                # Count and performance metrics
                stats["count"] += 1
                stats["avg_input_tokens"] += data.get("llm_input_tokens", 0)
                stats["avg_output_tokens"] += data.get("llm_output_tokens", 0)
                stats["avg_inference_time"] += data.get("llm_inference_time", 0)

                # Host information
                stats["host_providers"][data.get("host_provider", "unknown")] += 1
                stats["host_gpus"][data.get("host_gpu", "unknown")] += 1
                stats["host_dc_providers"][data.get("host_dc_provider", "unknown")] += 1
                stats["host_dc_locations"][data.get("host_dc_location", "unknown")] += 1
                stats["host_cost_per_hour"] += data.get("host_cost_per_hour", 0.0)
                stats["host_gpu_vram"][data.get("host_gpu_vram", 0)] += 1

    # Calculate averages
    for model, stats in tf_model_stats.items():
        count = stats["count"]
        if count > 0:
            stats["avg_input_tokens"] /= count
            stats["avg_output_tokens"] /= count
            stats["avg_inference_time"] /= count
            stats["host_cost_per_hour"] /= count

    return tf_model_stats


def create_fables_markdown_table(tf_model_stats):
    """
    Create a markdown table from tf_fables data.

    Parameters:
        tf_model_stats: Dictionary of model statistics

    Returns:
        str: Formatted markdown table
    """
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
        # Find most common values for each host attribute
        primary_provider = get_max_key(stats["host_providers"])
        primary_gpu = get_max_key(stats["host_gpus"])
        primary_gpu_vram = get_max_key(stats["host_gpu_vram"])
        primary_dc_provider = get_max_key(stats["host_dc_providers"])
        primary_dc_location = get_max_key(stats["host_dc_locations"])
        avg_cost_per_hour = stats["host_cost_per_hour"]

        # Add row to table
        md_table_tf += (
            f"| {model} | {stats['count']} | {stats['avg_input_tokens']:.1f} | "
            f"{stats['avg_output_tokens']:.1f} | {stats['avg_inference_time']:.2f} | "
            f"{primary_provider} | {primary_gpu} | {primary_gpu_vram} | "
            f"{primary_dc_provider} | {primary_dc_location} | {avg_cost_per_hour:.2f} |\n"
        )

    return md_table_tf


def get_max_key(counter_dict):
    """Helper function to get the most common item from a counter dictionary (For Categorical/NonNumerical Data)"""
    return (
        max(counter_dict.items(), key=lambda x: x[1])[0] if counter_dict else "unknown"
    )


def process_tf_files(tf_files, stats_folder, timestamp, output_mode):
    """
    Process tf_fables files to generate a consolidated report.

    Parameters:
        tf_files: List of files whose names start with 'tf_fables'
        stats_folder: Folder where to save output files
        timestamp: Timestamp string for filenames
        output_mode: Controls where output is displayed

    Returns:
        dict: Processed model statistics
    """
    should_output_files = output_mode in ["files", "both"]
    should_output_terminal = output_mode in ["terminal", "both"]

    # Parse data from files
    tf_model_stats = parse_fables_data(tf_files)

    # Create markdown table
    md_table_tf = create_fables_markdown_table(tf_model_stats)

    # Output according to mode
    if should_output_files:
        md_tf_filename = os.path.join(stats_folder, f"tf_stats_fables_{timestamp}.md")
        with open(md_tf_filename, "w") as f:
            f.write(md_table_tf)
        print(f"Consolidated tf_fable report saved to {md_tf_filename}")

    if should_output_terminal:
        print("\n" + md_table_tf + "\n")

    return tf_model_stats


# --- EVALUATION DATA PROCESSING ---


def parse_evaluation_data(eval_files):
    """
    Parse data from evaluation files.

    Parameters:
        eval_files: List of evaluation files

    Returns:
        dict: Aggregated scores and performance metrics for each model
    """
    score_totals = defaultdict(
        lambda: {
            "grammar": 0,
            "creativity": 0,
            "moral_clarity": 0,
            "adherence_to_prompt": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "count": 0,
        }
    )

    for file_path in eval_files:
        with open(file_path, "r") as f:
            for line in f:
                data = json.loads(line)
                model = data.get("llm_name", "unknown").split("/")[-1]

                # Process evaluation scores
                if "evaluation" in data:
                    evaluation = data["evaluation"]
                    score_totals[model]["grammar"] += evaluation.get("grammar", 0)
                    score_totals[model]["creativity"] += evaluation.get("creativity", 0)
                    score_totals[model]["moral_clarity"] += evaluation.get(
                        "moral_clarity", 0
                    )
                    score_totals[model]["adherence_to_prompt"] += evaluation.get(
                        "adherence_to_prompt", 0
                    )

                # Process performance metrics
                score_totals[model]["input_tokens"] += data.get("llm_input_tokens", 0)
                score_totals[model]["output_tokens"] += data.get("llm_output_tokens", 0)
                score_totals[model]["inference_time"] += data.get(
                    "llm_inference_time", 0
                )
                score_totals[model]["count"] += 1

    return score_totals

def parse_translation_evaluation_data(translation_eval_files):
    """
    Parse data from translation evaluation files.

    Parameters:
        translation_eval_files: List of translation evaluation files

    Returns:
        dict: Aggregated scores and performance metrics for each model's translation evaluations
    """
    translation_score_totals = defaultdict(
        lambda: {
            "translation_accuracy": 0,
            "fluency": 0,
            "style_preservation": 0,
            "moral_clarity": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_time": 0,
            "count": 0,
        }
    )

    for file_path in translation_eval_files:
        with open(file_path, "r") as f:
            for line in f:
                print(line)
                data = json.loads(line)
                model = data.get("llm_name", "unknown").split("/")[-1]

                # Process translation evaluation scores
                if "evaluation" in data:
                    evaluation = data["evaluation"]
                    
                    # Check if this is a translation evaluation (contains translation_accuracy)
                    if "translation_accuracy" in evaluation:
                        translation_score_totals[model]["translation_accuracy"] += evaluation.get("translation_accuracy", 0)
                        translation_score_totals[model]["fluency"] += evaluation.get("fluency", 0)
                        translation_score_totals[model]["style_preservation"] += evaluation.get("style_preservation", 0)
                        translation_score_totals[model]["moral_clarity"] += evaluation.get("moral_clarity", 0)
                        
                        # Process performance metrics
                        translation_score_totals[model]["input_tokens"] += data.get("llm_input_tokens", 0)
                        translation_score_totals[model]["output_tokens"] += data.get("llm_output_tokens", 0)
                        translation_score_totals[model]["inference_time"] += data.get("llm_inference_time", 0)
                        translation_score_totals[model]["count"] += 1
    print(translation_score_totals)
    return translation_score_totals

def compute_translation_averages(translation_score_totals):
    """
    Compute average scores from aggregated translation evaluation totals.

    Parameters:
        translation_score_totals: Dictionary of aggregated translation scores and counts

    Returns:
        dict: Average metrics for each model's translation performance
    """
    translation_averages = {}
    for model, scores in translation_score_totals.items():
        count = scores["count"]
        if count > 0:
            print(scores)
            evaluation = scores["evaluation"]
            # Calculate translation evaluation averages
            translation_accuracy = evaluation["translation_accuracy"] / count
            fluency = evaluation["fluency"] / count
            style_preservation = evaluation["style_preservation"] / count
            moral_clarity = evaluation["moral_clarity"] / count

            # Store in dictionary
            translation_averages[model] = {
                "translation_accuracy": translation_accuracy,
                "fluency": fluency,
                "style_preservation": style_preservation,
                "moral_clarity": moral_clarity,
                "average_score_(mean)": (
                    translation_accuracy + fluency + style_preservation + moral_clarity
                ) / 4,
                "input_tokens": scores["input_tokens"] / count,
                "output_tokens": scores["output_tokens"] / count,
                "inference_time": scores["inference_time"] / count,
                "count": count,
            }

    return translation_averages

def compute_averages(score_totals):
    """
    Compute average scores from aggregated totals.

    Parameters:
        score_totals: Dictionary of aggregated scores and counts

    Returns:
        dict: Average metrics for each model
    """
    averages = {}
    for model, scores in score_totals.items():
        count = scores["count"]
        if count > 0:
            # Calculate evaluation averages
            grammar = scores["grammar"] / count
            creativity = scores["creativity"] / count
            moral_clarity = scores["moral_clarity"] / count
            adherence = scores["adherence_to_prompt"] / count

            # Store in dictionary
            averages[model] = {
                "grammar": grammar,
                "creativity": creativity,
                "moral_clarity": moral_clarity,
                "adherence_to_prompt": adherence,
                "average_score_(mean)": (
                    grammar + creativity + moral_clarity + adherence
                )
                / 4,
                "input_tokens": scores["input_tokens"] / count,
                "output_tokens": scores["output_tokens"] / count,
                "inference_time": scores["inference_time"] / count,
                "count": count,
            }

    return averages


def create_evaluation_markdown_table(averages):
    """
    Create a markdown table from evaluation averages.

    Parameters:
        averages: Dictionary of model averages

    Returns:
        str: Formatted markdown table
    """
    md_table_eval = "## Evaluation Averages\n\n"
    md_table_eval += (
        "| Model | Grammar | Creativity | Moral Clarity | Adherence to Prompt | Average Score (Mean) | Count | "
        "Avg Input Tokens | Avg Output Tokens | Avg Inference Time (s) |\n"
        "|-------|---------|------------|---------------|---------------------|-----------------|-------|"
        "-----------------|------------------|------------------------|\n"
    )

    for model, metrics in averages.items():
        md_table_eval += (
            f"| {model} | {metrics['grammar']:.2f} | {metrics['creativity']:.2f} | "
            f"{metrics['moral_clarity']:.2f} | {metrics['adherence_to_prompt']:.2f} | "
            f"{metrics['average_score_(mean)']:.2f} | {metrics['count']} | "
            f"{metrics['input_tokens']:.1f} | {metrics['output_tokens']:.1f} | "
            f"{metrics['inference_time']:.2f} |\n"
        )

    return md_table_eval


# --- VISUALIZATION FUNCTIONS ---


def generate_plotly_visualizations(
    averages, tf_model_stats, stats_folder, timestamp, output_mode
):
    """
    Generate interactive Plotly visualizations for model evaluation and performance.

    Parameters:
        averages: Dictionary with computed averages for each model's evaluation scores
        tf_model_stats: Dictionary containing model performance statistics
        stats_folder: Folder where to save output files
        timestamp: Timestamp string for filenames
        output_mode: Controls where output is displayed ('terminal', 'files', or 'both')
    """
    should_output_files = output_mode in ["files", "both"]
    should_output_terminal = output_mode in ["terminal", "both"]

    # Define metrics and model lists
    metrics_list = [
        "grammar",
        "creativity",
        "moral_clarity",
        "adherence_to_prompt",
        "average_score_(mean)",
    ]
    models_list = list(averages.keys())

    # Create visualization
    fig = create_plotly_figure(averages, tf_model_stats, metrics_list, models_list)

    # Save and display plots according to output mode
    if should_output_files:
        # Save as static image
        plot_filename = os.path.join(
            stats_folder, f"tf_stats_eval_plot_{timestamp}.png"
        )
        fig.write_image(plot_filename)

        # Also save as interactive HTML
        html_filename = os.path.join(
            stats_folder, f"tf_stats_eval_plot_{timestamp}.html"
        )
        fig.write_html(html_filename)

        print(f"Evaluation plots saved to {plot_filename} and {html_filename}")

    # Show plot in browser if terminal output is enabled
    if should_output_terminal:
        fig.show()


def create_plotly_figure(averages, tf_model_stats, metrics_list, models_list):
    """
    Create a Plotly figure with evaluation and performance data.

    Parameters:
        averages: Dictionary with computed averages for each model's evaluation scores
        tf_model_stats: Dictionary containing model performance statistics
        metrics_list: List of evaluation metrics to display
        models_list: List of models to display

    Returns:
        go.Figure: Plotly figure object
    """
    # Create a subplot with 2 rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Average Evaluation Scores by Model",
            "Performance Metrics by Model",
        ),
        vertical_spacing=0.2,
        specs=[[{"type": "bar"}], [{"type": "bar"}]],
    )

    # Define color schemes
    eval_colors = {
        "grammar": "#1DB954",  # Spotify green
        "creativity": "#191414",  # Dark gray/black
        "moral_clarity": "#535353",  # Medium gray
        "adherence_to_prompt": "#B3B3B3",  # Light gray
        "average_score_(mean)": "#E60012",  # Red
    }

    perf_colors = {
        "input_tokens": "#4CAF50",  # Green
        "output_tokens": "#2196F3",  # Blue
        "inference_time": "#FFC107",  # Amber
    }

    # Add evaluation score bars
    for i, metric in enumerate(metrics_list):
        values = [averages[model][metric] for model in models_list]

        fig.add_trace(
            go.Bar(
                x=models_list,
                y=values,
                name=metric.replace("_", " ").title(),
                text=[f"{val:.2f}" for val in values],
                textposition="auto",
                marker_color=eval_colors.get(metric, f"hsl({50 + i * 70}, 70%, 50%)"),
                hovertemplate=f'%{{x}}<br>{metric.replace("_", " ").title()}: %{{y:.2f}}<extra></extra>',
            ),
            row=1,
            col=1,
        )

    # Add performance metric bars
    # Prefer tf_model_stats if available; otherwise, use evaluation aggregates
    if tf_model_stats:
        performance_metrics = [
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_inference_time",
        ]
        source = tf_model_stats
    else:
        performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
        source = averages

    for i, metric in enumerate(performance_metrics):
        values = [
            source[model][metric] if model in source else 0 for model in models_list
        ]

        display_name = metric.replace("avg_", "").replace("_", " ").title()
        color_key = metric.replace("avg_", "")

        fig.add_trace(
            go.Bar(
                x=models_list,
                y=values,
                name=display_name,
                text=[f"{val:.1f}" for val in values],
                textposition="auto",
                marker_color=perf_colors.get(
                    color_key, f"hsl({50 + i * 70}, 70%, 50%)"
                ),
                hovertemplate=f"%{{x}}<br>{display_name}: %{{y:.1f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Set up layout and styling
    configure_plotly_layout(fig)

    return fig


def configure_plotly_layout(fig):
    """
    Configure the layout and styling of a Plotly figure.

    Parameters:
        fig: Plotly figure to configure
    """
    # Update overall layout
    fig.update_layout(
        title_text="Model Evaluation and Performance Analytics",
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        template="plotly_white",
        height=900,
        width=1000,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
    )

    # Configure axes
    fig.update_yaxes(title_text="Score", range=[0, 10], row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_xaxes(title_text="Models", row=1, col=1)
    fig.update_xaxes(title_text="Models", row=2, col=1)


def generate_matplotlib_visualizations(
    averages, tf_model_stats, stats_folder, timestamp, output_mode
):
    """
    Generate Matplotlib visualizations for model evaluation and performance.

    Parameters:
        averages: Dictionary with computed averages for each model's evaluation scores
        tf_model_stats: Dictionary containing model performance statistics
        stats_folder: Folder where to save output files
        timestamp: Timestamp string for filenames
        output_mode: Controls where output is displayed ('terminal', 'files', or 'both')
    """
    should_output_files = output_mode in ["files", "both"]
    should_output_terminal = output_mode in ["terminal", "both"]

    # Define metrics and models
    metrics_list = [
        "grammar",
        "creativity",
        "moral_clarity",
        "adherence_to_prompt",
        "average_score_(mean)",
    ]
    models_list = list(averages.keys())

    # Create the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot evaluation scores on first subplot
    plot_evaluation_scores(ax1, averages, models_list, metrics_list)

    # Plot performance metrics on second subplot
    plot_performance_metrics(ax2, averages, tf_model_stats, models_list)

    plt.tight_layout()

    # Save and display according to output mode
    if should_output_files:
        plot_filename = os.path.join(
            stats_folder, f"tf_stats_eval_plot_{timestamp}.png"
        )
        plt.savefig(plot_filename)
        print(f"Evaluation plot saved to {plot_filename}")

    # Show plot in terminal mode
    if should_output_terminal:
        plt.show()


def plot_evaluation_scores(ax, averages, models_list, metrics_list):
    """
    Plot evaluation scores on the provided matplotlib axis.

    Parameters:
        ax: Matplotlib axis to plot on
        averages: Dictionary with evaluation score averages
        models_list: List of models to display
        metrics_list: List of metrics to display
    """
    x = np.arange(len(models_list))
    width = 0.17
    colors = ["#1DB954", "#191414", "#535353", "#B3B3B3", "#E60012"]

    for i, metric in enumerate(metrics_list):
        values = [averages[model][metric] for model in models_list]
        offset = (i - 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=metric.replace("_", " ").title(),
            color=colors[i % len(colors)],
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Configure axis
    ax.set_title("Average Evaluation Scores by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=12)
    ax.set_ylim(0, 10)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def plot_performance_metrics(ax, averages, tf_model_stats, models_list):
    """
    Plot performance metrics on the provided matplotlib axis.

    Parameters:
        ax: Matplotlib axis to plot on
        averages: Dictionary with evaluation score averages
        tf_model_stats: Dictionary containing model performance statistics
        models_list: List of models to display
    """
    # Determine data source and metrics
    if tf_model_stats:
        performance_metrics = [
            "avg_input_tokens",
            "avg_output_tokens",
            "avg_inference_time",
        ]
        source = tf_model_stats
    else:
        performance_metrics = ["input_tokens", "output_tokens", "inference_time"]
        source = averages

    x = np.arange(len(models_list))
    width_perf = 0.15
    colors_perf = ["#4CAF50", "#2196F3", "#FFC107"]

    for i, metric in enumerate(performance_metrics):
        # For each model, if the metric is missing, default to 0
        values = [
            source[model][metric] if model in source else 0 for model in models_list
        ]
        offset = (i - 1) * width_perf
        bars = ax.bar(
            x + offset,
            values,
            width_perf,
            label=metric.replace("avg_", "").replace("_", " ").title(),
            color=colors_perf[i % len(colors_perf)],
        )

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Configure axis
    ax.set_title("Performance Metrics by Model", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def process_eval_files(
    eval_files, tf_model_stats, stats_folder, timestamp, output_mode, plot_mode="plotly"
):
    """
    Process evaluation files to generate tables and plots of model performance.

    Parameters:
        eval_files: List of files with 'eval_e' in their filename
        tf_model_stats: Dictionary containing model statistics (from process_tf_files)
        stats_folder: Folder where to save output files
        timestamp: Timestamp string for filenames
        output_mode: Controls where output is displayed ('terminal', 'files', or 'both')
        plot_mode: Plotting library to use ('plotly' or 'matplotlib')
    """
    should_output_files = output_mode in ["files", "both"]
    should_output_terminal = output_mode in ["terminal", "both"]

    # Parse data and compute averages
    score_totals = parse_translation_evaluation_data(eval_files)
    print(score_totals)
    averages = compute_translation_averages(score_totals) #compute_averages(score_totals)

    # Generate evaluation Markdown table
    md_table_eval = create_evaluation_markdown_table(averages)

    # Output table according to specified mode
    if should_output_files:
        md_eval_filename = os.path.join(
            stats_folder, f"tf_stats_eval_table_{timestamp}.md"
        )
        with open(md_eval_filename, "w") as f:
            f.write(md_table_eval)
        print(f"Evaluation Markdown table saved to {md_eval_filename}")

    if should_output_terminal:
        print("\n" + md_table_eval + "\n")

    # Generate visualizations according to selected plotting library
    if plot_mode.lower() == "plotly":
        generate_plotly_visualizations(
            averages, tf_model_stats, stats_folder, timestamp, output_mode
        )
    else:  # Use matplotlib as default or fallback
        generate_matplotlib_visualizations(
            averages, tf_model_stats, stats_folder, timestamp, output_mode
        )


# --- MAIN FUNCTION AND COMMAND LINE INTERFACE ---


def plot_model_averages(args):
    """
    Main function to process evaluation and tf_fables files.

    Parameters:
        args: Arguments containing input path, output mode, and plot mode
    """
    # Identify relevant files
    eval_files, tf_files = identify_files(args.input)

    # Set up output directory
    stats_folder = "tinyfabulist/data/stats/"
    if args.output_mode in ["files", "both"]:
        os.makedirs(stats_folder, exist_ok=True)
    timestamp = time.strftime("%y%m%d-%H%M%S")

    # Process files
    tf_model_stats = None
    if tf_files:
        tf_model_stats = process_tf_files(
            tf_files, stats_folder, timestamp, args.output_mode
        )

    if eval_files:
        process_eval_files(
            eval_files,
            tf_model_stats,
            stats_folder,
            timestamp,
            args.output_mode,
            args.plot_mode,
        )
    else:
        print("No evaluation files (with 'eval_e' in filename) found.")


def add_stats_subparser(subparsers) -> None:
    """
    Add statistics subparser to the main argument parser.

    Parameters:
        subparsers: Subparser collection to add to
    """
    generate_parser = subparsers.add_parser(
        "stats",
        help="Compute aggregated statistics from evaluation JSONL files (file or directory).",
    )
    generate_parser.add_argument(
        "--input",
        default="evaluate.jsonl",
        help="Path to a JSONL file or a folder containing JSONL files",
    )
    generate_parser.add_argument(
        "--output-mode",
        choices=["terminal", "files", "both"],
        default="both",
        help="Control where to output results: terminal only, files only, or both (default: both)",
    )
    generate_parser.add_argument(
        "--plot-mode",
        choices=["plotly", "matplotlib"],
        default="plotly",
        help="Plotting library to use for visualizations (default: plotly)",
    )
    generate_parser.set_defaults(func=plot_model_averages)


# --- SCRIPT ENTRY POINT ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics from evaluation files"
    )
    subparsers = parser.add_subparsers(title="Commands", dest="command")
    add_stats_subparser(subparsers)

    args = parser.parse_args()
    args.func(args)
